# SDXL LoRA Training — Notes & Learnings

## What Worked

- **Model:** SDXL 1.0 base (`stabilityai/stable-diffusion-xl-base-1.0`)
- **Config:** AdamW 3e-5 constant LR, rank 32 / alpha 32, SNR gamma 5, caption dropout 0.05
- **Dataset:** 27 Hopper paintings with detailed captions, 3x repeats
- **Training:** 5 epochs (405 steps) on A10G, ~15-20 min, ~$0.75
- **Final loss:** 0.071 — loss plateaus around 0.08-0.10 by epoch 2, then slowly improves
- **Result:** Clear Hopper style transfer confirmed on 5 holdout prompts (pixel diff ~12.9 vs base)

## The LoRA Save/Load Bug (Critical)

The biggest time sink was getting saved LoRA weights to actually load and affect generation. We went through 4 broken iterations before finding the fix.

### Root Cause

Three components in the save pipeline each handle key prefixes differently:

1. **`get_peft_model_state_dict(peft_unet)`** — returns keys WITH `base_model.model.` prefix (because we don't use Accelerator's `unwrap_model()`)
2. **`convert_state_dict_to_diffusers()`** — converts `lora_A` → `lora.down` / `lora_B` → `lora.up`, but KEEPS the `base_model.model.` prefix
3. **`pipeline.load_lora_weights()`** — expects keys WITHOUT `base_model.model.` prefix. Routes by top-level prefix: `unet.` for UNet, `text_encoder.` for CLIP

### What Broke

| Attempt | What We Did | What Happened |
|---------|------------|---------------|
| 1 | `unet.save_pretrained()` + `weight_name="adapter_model.safetensors"` | Keys had no `unet.` prefix → `load_lora_weights` couldn't route them → silently ignored, identical images |
| 2 | Manually added `unet.` prefix, stripped `base_model.model.` | Keys were `unet.*.lora_A.weight` (wrong internal format, should be `lora.down`) → weights loaded into wrong matrix shapes → **black screen** |
| 3 | Used official diffusers pattern but didn't strip `base_model.model.` | Error: "Target modules `base_model.model.up_blocks...` not found in the base model" |
| 4 | Strip `base_model.model.` AFTER `convert_state_dict_to_diffusers` | Works. Pixel diff 12.9, visible style transfer. |

### The Fix

```python
def save_lora_diffusers(peft_unet, save_path):
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    from diffusers import StableDiffusionXLPipeline

    raw_state = get_peft_model_state_dict(peft_unet)
    converted = convert_state_dict_to_diffusers(raw_state)
    unet_lora_state = {k.replace("base_model.model.", ""): v for k, v in converted.items()}
    StableDiffusionXLPipeline.save_lora_weights(save_path, unet_lora_layers=unet_lora_state)
```

### How to Verify

Always verify after saving:
- Check key format: should be `unet.down_blocks.*.lora.down.weight` (no `base_model.model.`)
- Generate an image with and without LoRA using the same seed
- Compute pixel diff — should be >5.0 if LoRA is working

## Process Lessons

1. **Research before running.** We started training before understanding the save/load pipeline. Reading the official `train_dreambooth_lora_sdxl.py` script would have prevented 3 of the 4 broken attempts.

2. **Sanity-check scale.** Our first plan was 30 epochs / 2000+ steps — way too much for 27 images. 5 epochs (405 steps) was plenty.

3. **Smoke-test end-to-end first.** A 2-epoch smoke test with the full save → load → generate pipeline would have caught the key format bug before wasting GPU time on long runs.

4. **Look at the outputs.** We ran multiple training runs without checking if the generated images actually showed style transfer. Always visually verify.

5. **Use `--detach` with Modal.** Long training runs die if the local client disconnects. Always use `modal run --detach` for anything over a few minutes.

## Key Files

- `scripts/train.py` — Training with verified save function
- `scripts/generate.py` — Single-image inference with LoRA
- `scripts/evaluate.py` — Side-by-side holdout prompt comparison
- `scripts/debug_lora.py` — Weight inspection and load verification

## Useful References

- [diffusers `train_dreambooth_lora_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sdxl.py) — official save pattern at lines 1949-1971
- [GitHub issue #6392](https://github.com/huggingface/diffusers/issues/6392) — documents the `base_model.model.` prefix mismatch as a known bug
