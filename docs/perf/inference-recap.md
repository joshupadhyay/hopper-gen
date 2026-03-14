# Hopper SDXL Inference Recap

> See also: [Interactive blogpost](../blog-serving-finetuned-model.html) for a narrative version with interactive charts.

This folder captures the main timing story for Hopper SDXL inference on Modal A10G.

## Key takeaways

- The biggest cold-start win came from moving from a plain function to a snapshotted `@app.cls` flow.
- `compile_repeated_blocks` produced the best warm-only number, but it was not chosen as the default because first-request behavior after restore was less trustworthy.
- The current production-default path favors lower variance:
  - fp16 base weights
  - fp16-safe VAE
  - TF32
  - `channels_last`
  - Hopper LoRA loaded into the snapshot
  - a warm-up inference before snapshot capture
  - no compile by default

## Charts

### Cold / first-request latency

![Cold start progression](/Users/joshu/Github/hopper-gen/docs/perf/cold_start_progression.png)

### Warm latency progression

![Warm latency progression](/Users/joshu/Github/hopper-gen/docs/perf/warm_latency_progression.png)

### Cold vs warm tradeoff

![Cold vs warm tradeoff](/Users/joshu/Github/hopper-gen/docs/perf/tradeoff_scatter.png)

### Final-path warm-run consistency

![Final warm variance](/Users/joshu/Github/hopper-gen/docs/perf/final_warm_variance.png)

### Final production path summary

![Final path summary](/Users/joshu/Github/hopper-gen/docs/perf/final_path_summary.png)

## Source timings used

From notes and current benchmark:

| Path | Cold / first request | Warm |
|---|---:|---:|
| No snapshot baseline | 49.20s | ~21.0s |
| Snapshot base model | ~33.0s | ~21.0s |
| fp16 + `compile_repeated_blocks` | ~33.0s | 17.7s |
| bf16 + `channels_last` + `compile_repeated_blocks` | 99.45s | 18.9s |
| Final path: snapshotted Hopper LoRA | 31.03s after restore, 28.37s first request after deploy | 19.53s mean |

## Narrative

### Optimize cold start vs warm start

- If the only goal were lowest warm latency, `compile_repeated_blocks` would still be attractive.
- But the website cares about the first request after a scale-up or restore, not only a hot container.
- The final path gives up a small amount of best-case warm speed in exchange for better cold behavior and more predictable request latency.

### 50 steps vs 30 steps

Add your side-by-side image comparison here once ready:

- `50 steps`: expected higher finish/detail, safer quality baseline
- `30 steps`: expected lower latency, but needs visual validation for Hopper lighting/composition quality
