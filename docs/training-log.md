# Training Log: SDXL LoRA Fine-Tuning for Edward Hopper Style

## Project Overview
Fine-tuning Stable Diffusion XL with LoRA to generate paintings in Edward Hopper's style — dramatic lighting, isolation, geometric composition, muted palettes.

- **Base model:** stabilityai/stable-diffusion-xl-base-1.0
- **Dataset:** 27 Edward Hopper paintings from Wikimedia Commons, resized to 1024x1024
- **Compute:** Modal A10G GPU (~$1.10/hr)
- **Activation phrase:** "hopper style" in all captions

---

## Data Collection

### Attempt 1: WikiArt (Failed)
- Scraped WikiArt with `og:image` meta tag extraction
- **Result:** 0/35 downloaded — all requests returned 403 Forbidden
- WikiArt blocks programmatic access regardless of User-Agent

### Attempt 2: Wikimedia Commons (Partial Success)
- Switched to Wikimedia Commons API with thumbnail endpoint (`iiurlwidth`)
- First run: 8/35 downloaded, then hit 429 rate limits
- Added retry logic with exponential backoff (5s, 10s, 15s) and 2s delays between images
- **Final result:** 27/35 downloaded. 8 paintings not on Commons at all (Sun in an Empty Room, Sunlight in a Cafeteria, Conference at Night, etc.)
- 27 images is plenty for LoRA style transfer

---

## The NaN Saga

### v1-v2: Loss = NaN Every Step

First training attempts showed `nan` loss from step 1 through step 1000. The model was "training" but producing garbage.

**Debugging approach:** Added a sanity check before training that inspects every input tensor:
```
latents: has_nan=True  range=[nan, nan]      ← THE PROBLEM
prompt_embeds: has_nan=False range=[-809, 854]
pooled: has_nan=False range=[-4.75, 4.65]
UNet params with NaN: 0
```

**Root cause: SDXL's VAE overflows in fp16.** This is a known issue — the VAE encoder produces NaN when running in float16 precision. The fix was trivially simple:

```python
# Before (broken)
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float16)

# After (fixed)
vae.to(dtype=torch.float32)  # upcast before encoding
pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float32)
```

Other fixes applied along the way:
- Load UNet natively in fp32 (not upcast from fp16)
- Use `DDPMScheduler` explicitly (the training scheduler, not pipeline's inference scheduler)
- Use `get_peft_model()` for proper PEFT LoRA wrapping
- Pre-compute and cache all latents + text embeddings, then free VAE/encoder VRAM

---

## v4: Loss is Real, But Flat

### Config
| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| Target modules | to_k, to_q, to_v, to_out.0 (attention only) |
| Optimizer | AdamW |
| Learning rate | 1e-4 (cosine schedule) |
| Warmup | 50 steps |
| Grad accumulation | 4 |
| Steps | 1000 |

### Results
```
First loss value: 0.049575
Step 50/1000  | Loss: 0.0802 | LR: 2.40e-05
Step 100/1000 | Loss: 0.1106 | LR: 5.00e-05
Step 150/1000 | Loss: 0.1055 | LR: 7.40e-05
Step 200/1000 | Loss: 0.1298 | LR: 1.00e-04
Step 250/1000 | Loss: 0.0980 | LR: 1.00e-04
Step 300/1000 | Loss: 0.1348 | LR: 9.98e-05
Step 350/1000 | Loss: 0.1071 | LR: 9.96e-05
Step 400/1000 | Loss: 0.1217 | LR: 9.93e-05
Step 450/1000 | Loss: 0.1459 | LR: 9.90e-05
Step 500/1000 | Loss: 0.1117 | LR: 9.85e-05
Step 550/1000 | Loss: 0.1226 | LR: 9.79e-05
Step 600/1000 | Loss: 0.1271 | LR: 9.73e-05
Step 650/1000 | Loss: 0.1245 | LR: 9.66e-05
Step 700/1000 | Loss: 0.1382 | LR: 9.58e-05
Step 750/1000 | Loss: 0.1409 | LR: 9.50e-05
Step 800/1000 | Loss: 0.1045 | LR: 9.40e-05
Step 850/1000 | Loss: 0.1054 | LR: 9.30e-05
Step 900/1000 | Loss: 0.1351 | LR: 9.19e-05
Step 950/1000 | Loss: 0.1020 | LR: 9.07e-05
Step 1000/1000| Loss: 0.1062 | LR: 8.95e-05
```

### Analysis: The Model Learned Nothing

Loss bounced between 0.08-0.15 for the entire run with zero convergence trend.

**Key observations:**

1. **First loss (0.05) was the lowest loss in the entire run.** That's backwards — randomly-initialized LoRA weights predicted noise better than after 1000 steps of "training." The optimizer actively made things worse before settling into a plateau.

2. **LR ramped from 2.4e-5 → 1e-4 over 200 steps**, and loss went UP (0.08 → 0.13) during that ramp. The warmup was so slow that by the time LR hit target, the model was stuck.

3. **Cosine decay started at step 200** — LR dropped from 1e-4 → 8.95e-05 while loss stayed flat. The schedule was decaying an already-too-low learning rate.

4. **The oscillation pattern** (±0.04 every 50 steps) = the gradient signal from 27 images is too noisy at this LR. Each 50-step window is ~2 passes through the dataset with different noise samples — variance dominated any learning signal.

**Problems identified:**
- **LR too low (1e-4):** Too conservative for 27 images. Gradient signal per step is weak.
- **No dynamic LR:** Cosine schedule is just a decay shape — doesn't adapt to training dynamics.
- **No dropout:** 0.0 dropout on 27 images is asking for trouble (though we didn't even get far enough to overfit).
- **Targets too narrow:** Attention-only LoRA misses feed-forward layers needed for style/color.
- **Rank too low (16):** Insufficient capacity for a complex artistic style.

---

## v5: Prodigy + Dropout + Expanded Targets

### Config Changes
| Parameter | v4 | v5 | Why |
|-----------|----|----|-----|
| Optimizer | AdamW, lr=1e-4 | **Prodigy, lr=1.0** | Auto-tunes LR from gradient signal — no guessing |
| LoRA dropout | 0.0 | **0.1** | Regularization for 27-image dataset |
| LoRA rank | 16 | **32** | More capacity for complex style |
| LoRA alpha | 32 | **64** | Scaled with rank (2x ratio) |
| Target modules | attention only | **attention + feed-forward** | Capture palette/lighting, not just attention |
| Warmup | 50 steps | **None (Prodigy handles it)** | Prodigy has built-in safeguard warmup |
| Steps | 1000 | **500** | Dynamic LR converges faster |
| Log frequency | every 50 | **every 25** | More granular visibility |

**Why Prodigy?** It's a D-Adaptation variant that auto-tunes the learning rate based on gradient-to-weight ratio. You set lr=1.0 and it figures out the right scale. Eliminates the "is 1e-4 right or should it be 1e-3?" guesswork. Its `d` parameter in the logs shows what effective LR it's using — useful diagnostic on its own.

### Results
*(To be filled after v5 run completes)*

---

## Understanding Loss in Diffusion Training

The loss is MSE between the UNet's noise prediction and the actual noise added to latents. It measures "how well can the model predict what noise was added?" — not directly "does this look like Hopper?"

For SDXL LoRA fine-tuning:
- **~0.05-0.15** is a healthy operating range
- **Convergence = steady decrease**, not a specific target number
- **< 0.03** = likely overfitting (memorized training images, won't generalize)
- **Flat at 0.10** = not learning (v4's problem)

**The real evaluation is visual.** Loss tells you training is working, but can't tell you whether the model learned Hopper's lighting vs. just his color palette. That's why we generate holdout prompts (scenes Hopper never painted) with and without LoRA, same seed, for side-by-side comparison.

---

## Cost Tracking

| Run | Steps | GPU | Duration | Est. Cost | Result |
|-----|-------|-----|----------|-----------|--------|
| v1 (NaN) | 1000 | A10G | ~35 min | ~$0.65 | NaN loss (fp16 VAE) |
| v2 (NaN) | 300 | A10G | ~15 min | ~$0.28 | NaN loss (same root cause) |
| v3-debug | 10 | A10G | ~8 min | ~$0.15 | Confirmed NaN source |
| v4 | 1000 | A10G | ~40 min | ~$0.75 | Flat loss, no learning |
| v4-debug | 10 | A10G | ~8 min | ~$0.15 | Confirmed fp32 VAE fix |
| v5 | 500 | A10G | TBD | ~$0.40 | Prodigy + dropout (pending) |
| **Total** | | | | **~$2.38** | |
