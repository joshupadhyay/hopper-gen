# Training Configuration

## Approach
LoRA fine-tuning of SDXL base model using HuggingFace diffusers `train_text_to_image_lora.py` patterns.

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | stabilityai/stable-diffusion-xl-base-1.0 | Best open-source for 1024px |
| LoRA rank | 16 | Good balance for artistic style (not character/face) |
| LoRA alpha | 32 | 2x rank, standard ratio |
| Learning rate | 1e-4 | Standard for SDXL LoRA |
| Training steps | 1000-1500 | ~30-40 epochs over ~35 images |
| Batch size | 1 | VRAM constraint on A10G |
| Gradient accumulation | 4 | Effective batch size of 4 |
| Mixed precision | fp16 | Saves VRAM, standard for SDXL |
| Resolution | 1024x1024 | SDXL native |
| GPU | A10G | ~$1.10/hr on Modal, 24GB VRAM |
| Optimizer | AdamW 8-bit | Via bitsandbytes, saves VRAM |
| Scheduler | cosine | Smooth LR decay |
| Gradient checkpointing | enabled | Trades compute for VRAM |

## Cost Estimate
- A10G at ~$1.10/hr
- ~45-60 min for 1000-1500 steps
- **Total: ~$0.75-1.50**
