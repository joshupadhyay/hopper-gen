# SDXL Inference Optimization Log

## Baseline (GPU snapshots + fuse_qkv only)
- Cold start: ~33s startup, ~21s execution
- Warm: ~0ms startup, ~21s execution
- Steps: 50, no tf32, no torch.compile

## v2: tf32 + 35 steps
- Cold start: 34.5s startup, 16s execution
- Warm: ~0ms startup, 14s execution
- Changes: `torch.backends.cuda.matmul.allow_tf32 = True`, steps 50→35
- **Warm speedup: ~33% (21s → 14s)**

## v3: compile_repeated_blocks (testing)
- Changes: `pipeline.unet.compile_repeated_blocks(fullgraph=True)`
- Expected: ~10-15s one-time compile cost, 1.2-1.5x per-step speedup
- Results: TBD
