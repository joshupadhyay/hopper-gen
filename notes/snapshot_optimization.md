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
