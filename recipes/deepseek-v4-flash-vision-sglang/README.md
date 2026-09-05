# deepseek-v4-flash-vision-sglang — SGLang b12x lane (A/B partner)

The SGLang team's own DGX Spark preview for DeepSeek-V4-Flash-Vision-Exp
(`lmsysorg/sglang:dev-v4f-2dgx-v2`, branch `b12x-vision`), wrapped so our runner drives it: same
weights, same two boxes, same proxy/tunnel front as `../deepseek-v4-flash-vision`. Their cell's env
and flags are carried verbatim (see yaml comments); our runner adds the multi-node plumbing.

```
./run.sh deepseek-v4-flash-vision-sglang
```

Why it matters: b12x (the FP4 MoE kernel library that only existed in vLLM forks until now) is a
first-party `--moe-runner-backend` here, with DSpark and native `image_url` input. Whether it beats
the vLLM lane's FlashInfer DSV4 + k=5 adaptive path on GB10 is the A/B question — measure, don't
assume. Known carry-overs from our SGLang notes: the runner's `SGLANG_HOST_IP` pin is mandatory on
this LAN; DSV4's KV pool is fp8-only in SGLang (robot's admission probe).

Reference image only (pinned tag, digest in the yaml). Not our build.
