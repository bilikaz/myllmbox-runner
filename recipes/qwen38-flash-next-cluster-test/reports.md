# qwen38-flash-next-cluster-test — runs (NVFP4 PLE table on the GPUs)

Reference to beat: ../qwen38-flash-next-cluster boot 2 (K=4, int3 table in a CPU worker per box):
engine steps/s c=1 16.6 · c=8 7.5 · c=16 5.9 · c=32 3.4 · c=64 2.35; pasture/thinking-off tok/s 68 / 254 / 396 / 464 / 626.
The A/B question: how many ms per step does the CPU detour cost (expect 2–5 ms, i.e. 4–9 % at c=1, ~1 % at c=64),
and does NVFP4 measure closer to the bf16 table than int3 on bench/logprob.py.

## Checkpoint build (make-ple-nvfp4.py)
| date | src int3 | src bf16 | dst | global scale | shards | wall | notes |
|---|---|---|---|---|---|---|---|

## Boots
| date | boot | note | Model loading took (GiB) | PLE log line | KV available | steps/s c=1 | Δ vs reference |
|---|---|---|---|---|---|---|---|

## logprob (bench/logprob.py, pooled pasture+fish)
| lane | table | avg logprob | ppl | tokens |
|---|---|---|---|---|
| qwen38-flash-next-off | bf16 | | | |
| qwen38-flash-next-cluster | int3 g160 | | | |
| qwen38-flash-next-cluster-test | NVFP4 | | | |
