"""
LoRA fine-tuning of SDXL on Edward Hopper paintings via Modal.

v6: Battle-tested settings from community guides.
AdamW 3e-5, constant LR, rank 32/alpha 32, SNR gamma 5, caption dropout 0.05.

Usage:
    modal run scripts/train.py --run-name v6
"""

import modal

app = modal.App("hopper-lora-train")

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
        "torchvision",
        "safetensors",
    )
)


def save_lora_diffusers(peft_unet, save_path):
    """Save PEFT LoRA weights using the same approach as diffusers' official
    train_dreambooth_lora_sdxl.py: get_peft_model_state_dict →
    convert_state_dict_to_diffusers → StableDiffusionXLPipeline.save_lora_weights.
    """
    from pathlib import Path
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers
    from diffusers import StableDiffusionXLPipeline

    Path(save_path).mkdir(parents=True, exist_ok=True)
    unet_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(peft_unet))
    StableDiffusionXLPipeline.save_lora_weights(save_path, unet_lora_layers=unet_lora_state)
    print(f"  Saved {len(unet_lora_state)} LoRA tensors to {save_path}")


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(run_name: str = "v6", num_epochs: int = 15, repeats: int = 3):
    import json
    import math
    import random
    import torch
    from pathlib import Path
    from PIL import Image as PILImage
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    import safetensors.torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

    data_path = Path(DATA_DIR)
    images_dir = data_path / "processed"
    captions_path = data_path / "captions.jsonl"

    captions = []
    with open(captions_path) as f:
        for line in f:
            captions.append(json.loads(line))

    # Repeat dataset
    captions_repeated = captions * repeats
    num_steps = num_epochs * len(captions_repeated)
    print(f"Training samples: {len(captions)} x {repeats} repeats = {len(captions_repeated)}")
    print(f"Epochs: {num_epochs}, Total steps: {num_steps}")

    # --- Load components ---
    print("Loading model components...")

    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    tokenizer_one = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")

    text_encoder_one = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch.float16
    )

    vae = AutoencoderKL.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float16
    )

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32
    )

    model_cache.commit()
    print("Models loaded")

    # Freeze everything
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to("cuda")
    text_encoder_one.to("cuda")
    text_encoder_two.to("cuda")

    # --- LoRA config: rank 32, alpha 32 (1:1 ratio) ---
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.enable_gradient_checkpointing()
    unet.to("cuda")

    # --- Dataset ---
    img_transform = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class HopperDataset(Dataset):
        def __init__(self, captions_list, img_dir):
            self.captions = captions_list
            self.img_dir = Path(img_dir)

        def __len__(self):
            return len(self.captions)

        def __getitem__(self, idx):
            entry = self.captions[idx]
            img = PILImage.open(self.img_dir / entry["file_name"]).convert("RGB")
            return {"pixel_values": img_transform(img), "caption": entry["text"]}

    dataset = HopperDataset(captions_repeated, images_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- Pre-compute latents and text embeddings ---
    print("Pre-computing latents and text embeddings...")

    # Upcast VAE to fp32 (SDXL VAE overflows in fp16)
    vae.to(dtype=torch.float32)

    # Cache unique images only (not repeats)
    latent_cache = {}
    embed_cache = {}

    for entry in captions:
        fname = entry["file_name"]
        caption = entry["text"]

        img = PILImage.open(images_dir / fname).convert("RGB")
        pixel_values = img_transform(img).unsqueeze(0).to("cuda", dtype=torch.float32)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latent_cache[fname] = latents.cpu()

            tok1 = tokenizer_one(
                [caption], padding="max_length",
                max_length=tokenizer_one.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to("cuda")

            tok2 = tokenizer_two(
                [caption], padding="max_length",
                max_length=tokenizer_two.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to("cuda")

            enc1_out = text_encoder_one(tok1, output_hidden_states=True)
            enc2_out = text_encoder_two(tok2, output_hidden_states=True)

            prompt_embeds = torch.cat([
                enc1_out.hidden_states[-2],
                enc2_out.hidden_states[-2],
            ], dim=-1).to(dtype=torch.float32)

            pooled = enc2_out.text_embeds.to(dtype=torch.float32)

            embed_cache[fname] = {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled": pooled.cpu(),
            }

    print(f"Cached {len(latent_cache)} unique images")

    # Free encoder VRAM
    del vae, text_encoder_one, text_encoder_two
    torch.cuda.empty_cache()
    print(f"VRAM after freeing encoders: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # --- Optimizer: AdamW, constant LR 3e-5, no warmup ---
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)

    # SDXL time IDs
    add_time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32, device="cuda"
    )

    # SNR gamma for Min-SNR weighting (improves convergence)
    snr_gamma = 5.0

    def compute_snr(timesteps):
        """Compute signal-to-noise ratio for Min-SNR weighting."""
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
        sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return snr

    # Caption dropout rate
    caption_dropout_rate = 0.05

    # Build training order (all repeats × epochs, shuffled per epoch)
    training_entries = captions_repeated

    # --- Training ---
    unet.train()
    optimizer.zero_grad()

    print(f"\nStarting training: {num_epochs} epochs, {num_steps} steps")
    print(f"Config: AdamW lr=3e-5 constant | rank=32 alpha=32 | SNR gamma={snr_gamma} | caption dropout={caption_dropout_rate}")
    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        random.shuffle(training_entries)

        for entry in training_entries:
            fname = entry["file_name"]

            latents = latent_cache[fname].to("cuda")
            prompt_embeds = embed_cache[fname]["prompt_embeds"].to("cuda")
            pooled = embed_cache[fname]["pooled"].to("cuda")

            # Caption dropout: randomly use empty embeddings
            if random.random() < caption_dropout_rate:
                prompt_embeds = torch.zeros_like(prompt_embeds)
                pooled = torch.zeros_like(pooled)

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda",
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled,
                    "time_ids": add_time_ids,
                },
                return_dict=False,
            )[0]

            # MSE loss with Min-SNR weighting
            loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape))))  # per-sample loss

            # Apply Min-SNR gamma weighting
            snr = compute_snr(timesteps)
            snr_weight = torch.clamp(snr, max=snr_gamma) / snr
            loss = (loss * snr_weight).mean()

            loss_val = loss.item()

            if global_step == 0:
                print(f"  First loss: {loss_val:.6f}")
                if loss_val != loss_val:
                    print("ERROR: NaN on first step!")
                    return {"error": "NaN loss", "run_name": run_name}

            loss.backward()

            # Step optimizer every step (batch_size=1, no grad accumulation)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss_val
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                vram_now = torch.cuda.memory_allocated() / 1e9
                progress = global_step / num_steps * 100
                print(f"Step {global_step}/{num_steps} ({progress:.0f}%) | Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | VRAM: {vram_now:.1f}GB")
                running_loss = 0.0

        # End of epoch summary
        print(f"--- Epoch {epoch+1}/{num_epochs} complete (step {global_step}) ---")

        # Checkpoint at epochs 5, 9, 12
        checkpoint_epochs = {5, 9, 12}
        if (epoch + 1) in checkpoint_epochs:
            ckpt_path = f"{DATA_DIR}/adapters/{run_name}/checkpoint-epoch{epoch+1}"
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            save_lora_diffusers(unet, ckpt_path)
            training_data.commit()
            print(f"  ** Checkpoint saved to {ckpt_path}")

    # --- Save final LoRA adapter ---
    adapter_save_path = f"{DATA_DIR}/adapters/{run_name}"
    Path(adapter_save_path).mkdir(parents=True, exist_ok=True)
    save_lora_diffusers(unet, adapter_save_path)

    # --- Verify saved weights load correctly ---
    print("Verifying saved weights...")
    test_state = safetensors.torch.load_file(f"{adapter_save_path}/pytorch_lora_weights.safetensors")
    sample_keys = list(test_state.keys())[:3]
    print(f"  ✓ Saved {len(test_state)} tensors. Sample keys: {sample_keys}")

    training_data.commit()
    final_loss = running_loss / max(global_step % 50, 1)
    print(f"\nTraining complete! Adapter saved to {adapter_save_path}")
    print(f"Final avg loss: {final_loss:.4f}")
    print(f"Total steps: {global_step}")

    return {
        "run_name": run_name,
        "epochs": num_epochs,
        "steps": global_step,
        "final_loss": final_loss,
        "adapter_path": adapter_save_path,
    }


@app.local_entrypoint()
def main(run_name: str = "v6", num_epochs: int = 15, repeats: int = 3):
    result = train.remote(run_name, num_epochs, repeats)
    print("\n" + "=" * 40)
    for k, v in result.items():
        print(f"  {k}: {v}")
