"""
LoRA fine-tuning of SDXL on Edward Hopper paintings via Modal.

Loads training data from Modal volume, fine-tunes with diffusers + peft,
saves LoRA adapter weights back to the volume.

Usage:
    modal run scripts/train.py --run-name v1
"""

import modal

app = modal.App("hopper-lora-train")

# Volumes
training_data = modal.Volume.from_name("hopper-training-data", create_if_missing=True)
model_cache = modal.Volume.from_name("hopper-model-cache", create_if_missing=True)

DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "Pillow",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(run_name: str = "v1", num_steps: int = 1000):
    import json
    import torch
    from pathlib import Path
    from PIL import Image as PILImage
    from torch.utils.data import Dataset, DataLoader
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    from peft import LoraConfig
    from diffusers.optimization import get_scheduler
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    data_path = Path(DATA_DIR)
    images_dir = data_path / "processed"
    captions_path = data_path / "captions.jsonl"

    # Load captions
    captions = []
    with open(captions_path) as f:
        for line in f:
            captions.append(json.loads(line))
    print(f"Training samples: {len(captions)}")

    # --- Load pipeline ---
    print("Loading SDXL pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    model_cache.commit()

    unet = pipeline.unet
    text_encoder_one = pipeline.text_encoder
    text_encoder_two = pipeline.text_encoder_2
    vae = pipeline.vae
    tokenizer_one = pipeline.tokenizer
    tokenizer_two = pipeline.tokenizer_2
    noise_scheduler = pipeline.scheduler

    # Freeze everything except LoRA
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Move to GPU
    vae.to("cuda", dtype=torch.float16)
    text_encoder_one.to("cuda", dtype=torch.float16)
    text_encoder_two.to("cuda", dtype=torch.float16)

    # Enable gradient checkpointing for memory
    unet.enable_gradient_checkpointing()

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    # Count trainable params
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Cast LoRA params to float32 for training stability
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # --- Dataset ---
    class HopperDataset(Dataset):
        def __init__(self, captions_list, images_dir):
            self.captions = captions_list
            self.images_dir = Path(images_dir)

        def __len__(self):
            return len(self.captions)

        def __getitem__(self, idx):
            entry = self.captions[idx]
            img = PILImage.open(self.images_dir / entry["file_name"]).convert("RGB")
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(1024),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            pixel_values = transform(img)
            return {"pixel_values": pixel_values, "caption": entry["text"]}

    dataset = HopperDataset(captions, images_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-2,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=num_steps,
    )

    # --- Training loop ---
    unet.to("cuda")
    unet.train()

    print(f"Starting training for {num_steps} steps...")
    global_step = 0
    running_loss = 0.0

    while global_step < num_steps:
        for batch in dataloader:
            if global_step >= num_steps:
                break

            pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
            captions_batch = batch["caption"]

            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Encode text
            with torch.no_grad():
                tokens_one = tokenizer_one(
                    captions_batch, padding="max_length",
                    max_length=tokenizer_one.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to("cuda")
                tokens_two = tokenizer_two(
                    captions_batch, padding="max_length",
                    max_length=tokenizer_two.model_max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.to("cuda")

                encoder_hidden_states = text_encoder_one(tokens_one, output_hidden_states=True)
                encoder_hidden_states_2 = text_encoder_two(tokens_two, output_hidden_states=True)

                # SDXL uses penultimate hidden states
                prompt_embeds = torch.concat([
                    encoder_hidden_states.hidden_states[-2],
                    encoder_hidden_states_2.hidden_states[-2],
                ], dim=-1)

                pooled_prompt_embeds = encoder_hidden_states_2[0]

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda"
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SDXL additional conditioning
            add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda")

            # Predict noise
            model_pred = unet(
                noisy_latents.to(torch.float16),
                timesteps,
                encoder_hidden_states=prompt_embeds.to(torch.float16),
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds.to(torch.float16),
                    "time_ids": add_time_ids,
                },
                return_dict=False,
            )[0]

            # Loss
            loss = torch.nn.functional.mse_loss(
                model_pred.float(), noise.float(), reduction="mean"
            )

            loss.backward()

            # Gradient accumulation (effective batch size = 4)
            if (global_step + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in unet.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                lr = optimizer.param_groups[0]["lr"]
                print(f"Step {global_step}/{num_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                running_loss = 0.0

    # --- Save LoRA adapter ---
    adapter_save_path = f"{DATA_DIR}/adapters/{run_name}"
    from peft import get_peft_model_state_dict
    import safetensors.torch

    Path(adapter_save_path).mkdir(parents=True, exist_ok=True)

    # Save only LoRA weights
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            lora_state_dict[name] = param.data.cpu()

    safetensors.torch.save_file(
        lora_state_dict,
        f"{adapter_save_path}/pytorch_lora_weights.safetensors"
    )

    # Save config
    unet.peft_config["default"].save_pretrained(adapter_save_path)

    training_data.commit()
    print(f"\nTraining complete! Adapter saved to {adapter_save_path}")
    print(f"Final avg loss: {running_loss / max(global_step % 50, 1):.4f}")

    return {
        "run_name": run_name,
        "steps": num_steps,
        "adapter_path": adapter_save_path,
    }


@app.local_entrypoint()
def main(run_name: str = "v1", num_steps: int = 1000):
    result = train.remote(run_name, num_steps)
    print("\n" + "=" * 40)
    for k, v in result.items():
        print(f"  {k}: {v}")
