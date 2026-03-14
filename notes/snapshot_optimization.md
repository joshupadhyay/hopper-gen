# SDXL Inference Optimization — Memory Snapshots + Compile

Based on [Modal's FLUX 3x faster blog post](https://modal.com/blog/flux-3x-faster).

## Architecture: `@app.cls` with Memory Snapshots

Converted from `@app.function` (stateless, reloads model every call) to `@app.cls` with lifecycle hooks:

- `@modal.enter(snap=True)` — loads base SDXL pipeline, applies optimizations. Runs once, gets snapshotted (CPU + GPU memory serialized by Modal).
- `@modal.enter(snap=False)` — runs after every snapshot restore. Currently a no-op.
- `@modal.method()` — per-request: loads LoRA, generates, unloads LoRA.

LoRA stays out of the snapshot so we can swap adapters without re-snapshotting the base model.

## Snapshots vs Volumes

- **Volume** = persistent disk you manage. Read/write files, shared across containers. Where our training data and model weights live.
- **Snapshot** = frozen container memory dump (RAM + GPU). Modal creates and restores it automatically. You don't interact with it — it just makes cold starts faster by skipping initialization. Invalidated when code or image definition changes.

## Expected Speedups

### Cold start: ~30-60s → ~5-10s

Without snapshot, every container spin-up does:
- Import torch, diffusers, transformers (~5-10s)
- Load SDXL weights from disk → CPU → GPU (~20-30s)
- torch.compile JIT compilation (minutes on first run)

With snapshot, all of that is already done — the container restores with the model on GPU, already compiled. The entire initialization becomes a single memory restore.

### Per-image inference: no improvement

Current: ~19s per image (50 steps at 2.67 it/s on A10G).
Expected with torch.compile: ~12-13s per image — **not achieved**.

- `torch.compile` — incompatible with SDXL (see "What Didn't Work")
- `fuse_qkv_projections(vae=False)` — active, but effect is within noise (~0.01 it/s difference)

The snapshot doesn't speed up diffusion itself — it eliminates the wait before generation starts.

### Warm container reuse: `scaledown_window`

Set to 5 minutes. If you generate multiple images within 5 min, subsequent calls hit an already-warm container — no cold start at all, no snapshot restore needed.

## What We Didn't Implement

- **First-block caching (TEACache)** — 2x speedup in the blog post, but FLUX-specific. No SDXL equivalent available yet.
- **Channels-last memory layout** — `torch.compile` with `max-autotune` handles this automatically.
- **TORCH_COMPILE_CACHE env var** — unnecessary since the GPU snapshot already captures the compiled state.

## What Didn't Work

- **`torch.compile` on full pipeline** — crashed with `FakeTensor Device Propagation` error (cpu vs cuda mismatch in `aten.cat`). SDXL has mixed-device ops that torch.compile can't handle at the pipeline level.
- **`torch.compile` on UNet only (`max-autotune`)** — same error. The blog post used FLUX (different architecture).
- **`torch.compile` on UNet only (`default`)** — runs without crashing, but adds ~100s overhead per call. First run: 4m33s (JIT compilation + snapshot creation). Warm runs: ~2m10s (vs ~21s without compile). The compiled graph likely recompiles guards on each of the 50 denoising steps. Both modes are unusable for SDXL — revisit with newer torch/diffusers versions.
- **`modal run` with snapshots** — snapshots only work with `modal deploy`. Need a separate runner script (`run_generate.py`) that calls the deployed app via `modal.Cls.from_name()`.

## Benchmarks

### Original App: `hopper-lora-generate` (no snapshots, `@app.function`)

| Run Type | Total Time | Model Load | Inference (50 steps) | it/s |
|----------|-----------|------------|---------------------|------|
| Cold (`modal run`, ephemeral) | **49.2s** | ~4s | ~19s | 2.63 |
| Warm | 22.18s | 0s (cached) | ~19s | 2.64 |
| Warm | 20.59s | 0s (cached) | ~19s | 2.64 |
| Warm | 22.09s | 0s (cached) | ~19s | 2.64 |

The ~25s gap between cold and warm is Modal overhead: container allocation, image pull, mount, Python imports, `from_pretrained`.

### Optimized App: `Optimized-Generate-Hopper` (GPU snapshots, `@app.cls`)

| Run | Cold Start | Execution | Total | Notes |
|-----|-----------|-----------|-------|-------|
| 1st (snapshot creation) | N/A | ~23s | ~23s | Model loaded + snapshot created |
| 2nd (warm) | 0ms | ~21.5s | ~21.5s | Container still alive |
| 3rd (warm) | 0ms | ~20-22s | ~20-22s | Within scaledown_window |
| After 7min wait (cold) | **11s** | ~22s | ~33s | Snapshot restore — no model reload |

### Head-to-Head: Same Prompt, Same LoRA (v12), seed 42

| Scenario | Original | Optimized | Speedup |
|----------|----------|-----------|---------|
| Cold start (total) | **49.2s** | **~33s** (11s restore + 22s) | **1.5x** |
| Warm (total) | ~21s | ~21s | 1x |
| Inference only | ~19s @ 2.63 it/s | ~19s @ 2.64 it/s | ~same |

### Analysis

- **Cold start: 49s → 33s (1.5x faster).** The snapshot saves ~16s by skipping `from_pretrained` and Python imports. The 11s restore replaces ~25s of container setup + model loading.
- **Warm execution identical** — both ~21s. Inference speed is the same since `torch.compile` is disabled.
- **`torch.compile` not contributing** — disabled due to SDXL compatibility issues (device propagation error). `fuse_qkv_projections` is active but effect is within noise.
- **`scaledown_window` (5 min)** is the biggest practical win for interactive use — keeps container warm for back-to-back generations at ~21s.

### v2: tf32 + compile_repeated_blocks (kept)

Added `torch.backends.cuda.matmul.allow_tf32 = True` and `pipeline.unet.compile_repeated_blocks(fullgraph=True)`.
Warm execution dropped from ~21s to ~17.7s (16% faster) with no quality impact.

Also tried reducing steps 50→35 and `fuse_lora(lora_scale=1.2)` — both degraded image quality (color/style regression). Reverted those, kept tf32 + regional compilation.

### v3: bfloat16 + torch.compile (testing)

Based on [HuggingFace fast_diffusion tutorial](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion).

**Changes:**
- **fp16 → bfloat16** — wider exponent range, fewer overflows, better torch.compile compatibility. Same model size (both 16-bit). Removed `variant="fp16"` since there's no bf16 variant on HF (loads fp32 and casts).
- **Inductor config flags** — `conv_1x1_as_mm`, `coordinate_descent_tuning`, `epilogue_fusion=False`, `coordinate_descent_check_all_directions`. These tune the torch.compile kernel selection.
- **channels_last memory format** — `unet.to(memory_format=torch.channels_last)` and same for VAE. Better GPU memory access patterns for convolutions.
- **Full torch.compile on UNet** — `torch.compile(unet, mode="max-autotune", fullgraph=True)`. Previously crashed with FakeTensor error in fp16. bfloat16 may avoid this.
- **torch.compile on VAE decode** — minor win (runs once per image), but free to add.
- **Removed compile_repeated_blocks** — replaced by full torch.compile.

**Why this might work now (when fp16 torch.compile crashed before):**
- bfloat16 handles the numerical edge cases that caused FakeTensor device propagation errors
- The inductor config flags guide kernel selection for SDXL's architecture
- channels_last aligns memory layout with how convolutions actually access data

**torch.compile modes tested:**

| | `reduce-overhead` | `max-autotune` |
|---|---|---|
| CUDA graphs | Yes | Yes |
| Kernel selection | Default "good enough" kernels | Benchmarks 7-8 candidates per op, picks fastest |
| Compile time | ~1-3 min | ~10-20 min |
| Runtime speed | ~90-95% of max-autotune | 100% (best possible) |

`max-autotune` timed out at 15 minutes (Modal's function timeout) — the per-op benchmarking on SDXL's hundreds of unique ops exceeds 900s on A10G. Switched to `reduce-overhead` which gives the same CUDA graph replay benefit without exhaustive kernel search. The ~5-10% kernel optimization gap is not worth 10x longer compile times.

**torch.compile + LoRA incompatibility (critical finding):**

`torch.compile` is fundamentally incompatible with dynamic LoRA loading. `load_lora_weights()` mutates the model weights, which invalidates the compiled CUDA graph and forces a full recompile (~3 min) on every `generate()` call. Even with a warmup pass in `@modal.enter(snap=True)`, the compiled graph only applies to the base model — loading LoRA after snapshot restore triggers recompilation.

Results:
- `max-autotune`: timed out at 15min (exhaustive per-op benchmarking too slow for A10G)
- `reduce-overhead`: compiled successfully, but 186s per warm call (recompile on every LoRA load)
- Both modes unusable with our LoRA load/unload pattern

**Final v3 config (what we kept at the time):**

Dropped torch.compile entirely. Kept bfloat16 + channels_last + tf32 + compile_repeated_blocks.

- First run (snapshot creation): 60.4s
- Warm: **18.9s** (down from 21s with fp16, 17.7s with fp16 + tf32 only)
- Quality: identical to fp16, Hopper style preserved

**Cumulative optimization journey:**

| Version | Config | Warm Time | vs Baseline |
|---|---|---|---|
| Baseline | fp16, no optimizations | ~21s | — |
| v2 | fp16 + tf32 + compile_repeated_blocks | 17.7s | 16% faster |
| v3 | bf16 + channels_last + tf32 + compile_repeated_blocks | 18.9s | 10% faster |
| v3 (expected) | torch.compile reduce-overhead | ~12-13s | blocked by LoRA |

Note: v3 is slightly slower than v2 (18.9s vs 17.7s). bfloat16 may have higher per-op overhead than fp16 on A10G despite better numerical properties. channels_last benefit may not offset this.

## Follow-up: revert the default inference path

After more cold-start testing, the best default for the deployed Hopper path is:

- `torch.float16` with `variant="fp16"` for the base SDXL weights
- `madebyollin/sdxl-vae-fp16-fix` for safe fp16 VAE decode
- TF32 enabled on A10G
- `channels_last` kept
- `compile_repeated_blocks` disabled by default

Why:

- The bf16 path removed the pretrained fp16 checkpoint advantage and appears to increase snapshot restore cost.
- `compile_repeated_blocks` improves warm-only latency, but the first inference after snapshot restore still appears to pay a compile tax once LoRA weights are loaded dynamically.
- PyTorch 2 SDPA is already enabled by default in diffusers, so there is no extra efficient-attention knob to turn here unless we add a different attention backend.

In practice, this trades a small amount of best-case warm speed for much more stable cold-start and first-request behavior, which matters more for the website.

## Final production default

The finalized default inference path is:

- Base model in `float16`
- `variant="fp16"` for direct fp16 weight loading
- `madebyollin/sdxl-vae-fp16-fix`
- TF32 enabled
- `channels_last` enabled
- Hopper LoRA loaded during `@modal.enter(snap=True)` and kept in the snapshot
- one warm-up inference run before the snapshot is captured
- Modal GPU memory snapshots enabled
- `compile_repeated_blocks` disabled by default

Benchmark workflow:

- Deploy the app
- Run `python scripts/benchmark_inference.py`
- Run `python scripts/validate_inference.py`
- Record first request, warm requests, and first request after scale-down
- Compare against the old no-snapshot baseline and prior compile experiments

**HF tutorial benchmarks (A100 80GB):**
| Optimization | Time | Cumulative Speedup |
|---|---|---|
| Baseline (fp32) | 7.36s | — |
| bfloat16 | 4.63s | 37% |
| + SDPA (default) | 3.31s | 55% |
| + torch.compile max-autotune | 2.54s | 65% |
| + fuse_qkv | 2.52s | 66% |
| + int8 quantization | 2.43s | 67% |

Note: A100 is much more powerful than our A10G. Absolute numbers will differ, but relative speedups should be similar.

## Useful References

- [HuggingFace fast_diffusion tutorial](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion) — bfloat16, torch.compile, channels_last, int8 quantization. Source for v3 optimization changes.
- [Modal's FLUX 3x faster blog post](https://modal.com/blog/flux-3x-faster) — GPU snapshots, compile, TEACache. Source for v1 snapshot architecture.
- [diffusers `train_dreambooth_lora_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py) — official LoRA save pattern
- [GitHub issue #6392](https://github.com/huggingface/diffusers/issues/6392) — `base_model.model.` prefix mismatch bug
