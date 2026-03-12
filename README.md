# hopper-gen

Fine-tune SDXL with LoRA to generate paintings in Edward Hopper's style — dramatic lighting, isolation, geometric composition, muted palettes.

## Quick Start

```bash
uv sync
python scripts/collect_images.py        # Download Hopper paintings
python scripts/prepare_data.py          # Resize + caption
modal run scripts/train.py              # LoRA training on Modal GPU
modal run scripts/generate.py           # Generate with trained LoRA
```

## Stack

- **Base model:** stabilityai/stable-diffusion-xl-base-1.0
- **Training:** LoRA via HuggingFace diffusers + peft
- **Compute:** Modal (A10G GPU)
- **Budget:** ~$1-2 total
