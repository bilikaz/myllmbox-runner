# TODO — the porting queue

Recipes being brought over from our previous serving stack. Use the **add-recipe skill**
(`.claude/skills/add-recipe/SKILL.md`) — its "porting an old-structure recipe" section maps the old
fields; the old plumbing (sidecar proxies, LAN ports) is replaced by this runner's proxy + tunnel.

- [ ] **`Qwen/Qwen-Image-Edit-2511`** — image editing (OpenAI Images `/v1/images/edits`).
      Generic-server recipe; NVFP4 via the quantizer + the 4-step Lightning LoRA. Known model
      quirk to carry into the recipe/README: never request an edit at the input's exact size when
      a dimension is exactly 1024 (internal condition-grid collision) — nudge any dimension.
- [ ] **`Lightricks/LTX-2.3`** — video + synchronized audio, one pass (`/v1/videos`), text→video
      and keyframe image→video. fp8-cast (~half the memory). Text encoder Gemma-3-12B: the google
      repo is gated — use the ungated Lightricks mirror.
- [ ] **`MiniMaxAI/MiniMax-H3`** — video; after LTX.
- [ ] Open a tracking issue per port (the myllmbox.com site links "coming" models to them).
- [ ] First real-world validation of the **add-recipe** and **add-dashboard** skills is these
      ports — fix the skill where it misleads, in the same change.
