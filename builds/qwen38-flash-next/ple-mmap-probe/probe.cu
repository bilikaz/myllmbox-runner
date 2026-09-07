// hmm-probe: can a GB10 (ATS/HMM) CUDA kernel gather rows straight out of an mmapped safetensors shard?
// Measures: cold (page cache evicted → NVMe faults), warm (page cache), anonymous host memory, device-resident copy.
// Build: nvcc -O2 -arch=sm_121a -o probe probe.cu     Run: ./probe <file> [rows] [row_bytes] [region_gib]
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("  !! %s -> %s\n", #x, cudaGetErrorString(e)); return -1; } } while (0)

// one block per gathered row; threads stride the row in 16-byte chunks; xor-fold so every byte is read
__global__ void gather(const uint8_t* __restrict__ base, const int64_t* __restrict__ idx, int row_bytes,
                       uint64_t* __restrict__ out) {
  const uint4* row = reinterpret_cast<const uint4*>(base + idx[blockIdx.x] * (int64_t)row_bytes);
  uint64_t acc = 0;
  for (int i = threadIdx.x; i < row_bytes / 16; i += blockDim.x) {
    uint4 v = row[i];
    acc ^= ((uint64_t)v.y << 32 | v.x) + ((uint64_t)v.w << 32 | v.z);
  }
  __shared__ uint64_t sh[256];
  sh[threadIdx.x] = acc; __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] ^= sh[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

static uint64_t cpu_gather(const uint8_t* base, const std::vector<int64_t>& idx, int row_bytes) {
  uint64_t x = 0;
  for (int64_t r : idx) {
    const uint64_t* p = reinterpret_cast<const uint64_t*>(base + r * (int64_t)row_bytes);
    uint64_t acc = 0;
    for (int i = 0; i < row_bytes / 16; i++) acc ^= p[2 * i] + p[2 * i + 1];
    x ^= acc;
  }
  return x;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// run the gather from `base` (any pointer the GPU may dereference); returns ms, fills xor
static int run(const char* label, const uint8_t* base, const int64_t* d_idx, int n, int row_bytes,
               uint64_t* d_out, uint64_t want, double* ms_out) {
  auto t0 = std::chrono::steady_clock::now();
  gather<<<n, 256>>>(base, d_idx, row_bytes, d_out);
  cudaError_t e = cudaDeviceSynchronize();
  double ms = ms_since(t0);
  if (e != cudaSuccess) { printf("  %-34s FAILED: %s\n", label, cudaGetErrorString(e)); return -1; }
  std::vector<uint64_t> h(n);
  CK(cudaMemcpy(h.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  uint64_t x = 0; for (auto v : h) x ^= v;
  double mb = (double)n * row_bytes / 1e6;
  printf("  %-34s %9.2f ms  %8.2f MB  %8.1f MB/s  %s\n", label, ms, mb, mb / (ms / 1e3), x == want ? "ok" : "MISMATCH");
  if (ms_out) *ms_out = ms;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: probe <file> [rows=4096] [row_bytes=2048] [region_gib=3]\n"); return 2; }
  const char* path = argv[1];
  int n = argc > 2 ? atoi(argv[2]) : 4096;
  int row_bytes = argc > 3 ? atoi(argv[3]) : 2048;
  double region_gib = argc > 4 ? atof(argv[4]) : 3.0;

  int fd = open(path, O_RDONLY); if (fd < 0) { perror("open"); return 1; }
  struct stat st; fstat(fd, &st);
  size_t region = (size_t)(region_gib * (1ull << 30)); if (region > (size_t)st.st_size) region = st.st_size;
  region -= region % 4096;
  int64_t nrows = region / row_bytes;
  printf("file %s  size %.2f GiB  region %.2f GiB  rows %lld x %d B  gather %d rows\n", path,
         st.st_size / 1073741824.0, region / 1073741824.0, (long long)nrows, row_bytes, n);

  int dev = 0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  int pageable = 0, hmm = 0;
  cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev);
  cudaDeviceGetAttribute(&hmm, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev);
  printf("gpu %s  sm_%d%d  pageableMemoryAccess=%d  usesHostPageTables=%d\n", prop.name, prop.major, prop.minor, pageable, hmm);

  std::mt19937_64 rng(123123123);
  std::vector<int64_t> idx(n); for (auto& r : idx) r = rng() % nrows;
  int64_t* d_idx; uint64_t* d_out;
  CK(cudaMalloc(&d_idx, n * sizeof(int64_t))); CK(cudaMalloc(&d_out, n * sizeof(uint64_t)));
  CK(cudaMemcpy(d_idx, idx.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));

  // ---- 1. file-backed mmap, page cache evicted first → GPU faults pages in from NVMe (the "first request" path)
  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  uint8_t* fmap = (uint8_t*)mmap(nullptr, region, PROT_READ, MAP_SHARED, fd, 0);
  if (fmap == MAP_FAILED) { perror("mmap"); return 1; }
  // reference checksum on the CPU (this also warms the touched rows — so compute it on a private copy of the rows)
  uint64_t want; {
    std::vector<uint8_t> rows((size_t)n * row_bytes);
    for (int i = 0; i < n; i++) { pread(fd, rows.data() + (size_t)i * row_bytes, row_bytes, idx[i] * (int64_t)row_bytes); }
    std::vector<int64_t> lin(n); for (int i = 0; i < n; i++) lin[i] = i;
    want = cpu_gather(rows.data(), lin, row_bytes);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // evict what pread pulled in
  }
  printf("\n[1] file mmap, cache evicted (cold, NVMe faults)\n");
  double cold = 0;
  if (run("gpu gather @ mmap (cold)", fmap, d_idx, n, row_bytes, d_out, want, &cold)) return 1;
  printf("[2] same rows again (page cache warm)\n");
  double warm = 0;
  if (run("gpu gather @ mmap (warm)", fmap, d_idx, n, row_bytes, d_out, want, &warm)) return 1;
  if (run("gpu gather @ mmap (warm, 2nd)", fmap, d_idx, n, row_bytes, d_out, want, nullptr)) return 1;

  // ---- 3. anonymous host memory holding the same region (what a plain malloc'd table would be)
  printf("[3] anonymous host memory (malloc + memcpy of the region)\n");
  uint8_t* anon = (uint8_t*)malloc(region);
  { auto t0 = std::chrono::steady_clock::now(); size_t off = 0;   // pread caps at 2 GiB per call
    while (off < region) { ssize_t got = pread(fd, anon + off, region - off, off); if (got <= 0) break; off += got; }
    printf("  (filled %.2f GiB in %.0f ms — this also warms the whole file's page cache)\n", region / 1073741824.0, ms_since(t0)); }
  if (run("gpu gather @ malloc (first touch)", anon, d_idx, n, row_bytes, d_out, want, nullptr)) return 1;
  if (run("gpu gather @ malloc (2nd)", anon, d_idx, n, row_bytes, d_out, want, nullptr)) return 1;

  // ---- 4. device-resident copy (today's patch-03 layout)
  printf("[4] cudaMalloc'd device copy (resident table, the baseline)\n");
  uint8_t* dres; CK(cudaMalloc(&dres, region));
  CK(cudaMemcpy(dres, anon, region, cudaMemcpyHostToDevice));
  if (run("gpu gather @ device", dres, d_idx, n, row_bytes, d_out, want, nullptr)) return 1;
  if (run("gpu gather @ device (2nd)", dres, d_idx, n, row_bytes, d_out, want, nullptr)) return 1;

  // ---- 5. fully warm file mapping, fresh random rows (the steady state of a hot table living in page cache)
  printf("[5] mmap, whole region warm, new random rows\n");
  for (auto& r : idx) r = rng() % nrows;
  CK(cudaMemcpy(d_idx, idx.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
  uint64_t want2 = cpu_gather(anon, idx, row_bytes);
  if (run("gpu gather @ mmap (all warm)", fmap, d_idx, n, row_bytes, d_out, want2, nullptr)) return 1;
  if (run("gpu gather @ device (same rows)", dres, d_idx, n, row_bytes, d_out, want2, nullptr)) return 1;

  // ---- 5b. per-page fault cost: touch ONE new page per launch, page-cache warm but PTE absent (madvise DONTNEED drops PTEs)
  printf("[5b] one-row launches on a warm-cache, PTE-less mapping (GPU page fault per launch)\n");
  { madvise(fmap, region, MADV_DONTNEED);            // drop this mapping's PTEs; page cache stays (file-backed, clean)
    std::vector<int64_t> one(1); int64_t* d_one; CK(cudaMalloc(&d_one, sizeof(int64_t)));
    double tot = 0; int reps = 64;
    for (int i = 0; i < reps; i++) { one[0] = rng() % nrows; CK(cudaMemcpy(d_one, one.data(), 8, cudaMemcpyHostToDevice));
      auto t0 = std::chrono::steady_clock::now(); gather<<<1, 256>>>(fmap, d_one, row_bytes, d_out); CK(cudaDeviceSynchronize()); tot += ms_since(t0); }
    printf("  %-34s %9.1f us per launch (fault + read)\n", "1 row, PTE absent", tot * 1e3 / reps);
    tot = 0; for (int i = 0; i < reps; i++) { one[0] = rng() % nrows; CK(cudaMemcpy(d_one, one.data(), 8, cudaMemcpyHostToDevice));
      volatile uint8_t sink = fmap[one[0] * (int64_t)row_bytes]; (void)sink;       // CPU touch → PTE present
      auto t0 = std::chrono::steady_clock::now(); gather<<<1, 256>>>(fmap, d_one, row_bytes, d_out); CK(cudaDeviceSynchronize()); tot += ms_since(t0); }
    printf("  %-34s %9.1f us per launch (no fault)\n", "1 row, CPU pre-touched", tot * 1e3 / reps);
    tot = 0; for (int i = 0; i < reps; i++) { one[0] = rng() % nrows; CK(cudaMemcpy(d_one, one.data(), 8, cudaMemcpyHostToDevice));
      auto t0 = std::chrono::steady_clock::now(); gather<<<1, 256>>>(dres, d_one, row_bytes, d_out); CK(cudaDeviceSynchronize()); tot += ms_since(t0); }
    printf("  %-34s %9.1f us per launch\n", "1 row, device copy", tot * 1e3 / reps);
    // CPU pre-touch cost itself, per page, warm cache
    madvise(fmap, region, MADV_DONTNEED);
    auto t0 = std::chrono::steady_clock::now(); uint64_t sink = 0;
    for (int i = 0; i < 4096; i++) sink += fmap[(rng() % nrows) * (int64_t)row_bytes];
    printf("  %-34s %9.1f us per page (sink %llu)\n", "CPU touch, warm cache, PTE absent", ms_since(t0) * 1e3 / 4096, (unsigned long long)(sink & 1));
  }
  // ---- 5c. 4096 random rows, warm cache: PTE-less (GPU faults) vs CPU pre-touched
  printf("[5c] 4096 random rows, page cache warm\n");
  { madvise(fmap, region, MADV_DONTNEED);
    for (auto& r : idx) r = rng() % nrows; CK(cudaMemcpy(d_idx, idx.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    uint64_t w = cpu_gather(anon, idx, row_bytes);
    if (run("gpu gather, PTE absent (GPU faults)", fmap, d_idx, n, row_bytes, d_out, w, nullptr)) return 1;
    madvise(fmap, region, MADV_DONTNEED);
    auto t0 = std::chrono::steady_clock::now(); uint64_t sink = 0;
    for (auto r : idx) sink += fmap[r * (int64_t)row_bytes];
    printf("  %-34s %9.2f ms (sink %llu)\n", "CPU pre-touch of the 4096 rows", ms_since(t0), (unsigned long long)(sink & 1));
    if (run("gpu gather after CPU pre-touch", fmap, d_idx, n, row_bytes, d_out, w, nullptr)) return 1;
    if (run("gpu gather, 2nd", fmap, d_idx, n, row_bytes, d_out, w, nullptr)) return 1;
  }
  // ---- 6. streaming read of the whole region from the mapping vs device: raw bandwidth
  printf("[6] streaming: every row of the region once (bandwidth)\n");
  { std::vector<int64_t> all(nrows); for (int64_t i = 0; i < nrows; i++) all[i] = i;
    int64_t* d_all; uint64_t* d_o2; CK(cudaMalloc(&d_all, nrows * sizeof(int64_t))); CK(cudaMalloc(&d_o2, nrows * sizeof(uint64_t)));
    CK(cudaMemcpy(d_all, all.data(), nrows * sizeof(int64_t), cudaMemcpyHostToDevice));
    uint64_t wall = cpu_gather(anon, all, row_bytes);
    if (run("stream @ mmap (warm)", fmap, d_all, (int)nrows, row_bytes, d_o2, wall, nullptr)) return 1;
    if (run("stream @ malloc", anon, d_all, (int)nrows, row_bytes, d_o2, wall, nullptr)) return 1;
    if (run("stream @ device", dres, d_all, (int)nrows, row_bytes, d_o2, wall, nullptr)) return 1;
    madvise(fmap, region, MADV_DONTNEED); posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (run("stream @ mmap (COLD, whole region)", fmap, d_all, (int)nrows, row_bytes, d_o2, wall, nullptr)) return 1;
  }
  printf("\ncold/warm per-row: %.1f us vs %.1f us over %d rows\n", cold * 1e3 / n, warm * 1e3 / n, n);
  return 0;
}
