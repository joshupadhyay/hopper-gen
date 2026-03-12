"""
LoRA fine-tuning of SDXL on Edward Hopper paintings via Modal.

Uses diffusers + peft + accelerate for proper mixed-precision training.
Loads training data from Modal volume, saves LoRA adapter back.

Usage:
    modal run scripts/train.py --run-name v3
    modal run scripts/train.py --run-name v3 --num-steps 1500 --detach
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(run_name: str = "v3", num_steps: int = 1000):
    import json
    import torch
    from pathlib import Path
    from PIL import Image as PILImage
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    from diffusers.optimization import get_scheduler
    import safetensors.torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

    data_path = Path(DATA_DIR)
    images_dir = data_path / "processed"
    captions_path = data_path / "captions.jsonl"

    # Load captions
    captions = []
    with open(captions_path) as f:
        for line in f:
            captions.append(json.loads(line))
    print(f"Training samples: {len(captions)}")

    # --- Load components individually (not via pipeline) ---
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

    # Move frozen models to GPU
    vae.to("cuda")
    text_encoder_one.to("cuda")
    text_encoder_two.to("cuda")

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
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

    dataset = HopperDataset(captions, images_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- Pre-compute latents and text embeddings ---
    print("Pre-computing latents and text embeddings...")
    cached_data = []

    # Upcast VAE to fp32 for stable encoding (SDXL VAE is known to overflow in fp16)
    vae.to(dtype=torch.float32)

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float32)
        caption = batch["caption"]

        with torch.no_grad():
            # Encode image (fp32 to avoid NaN)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Encode text with both CLIP models
            tok1 = tokenizer_one(
                caption, padding="max_length",
                max_length=tokenizer_one.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to("cuda")

            tok2 = tokenizer_two(
                caption, padding="max_length",
                max_length=tokenizer_two.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to("cuda")

            enc1_out = text_encoder_one(tok1, output_hidden_states=True)
            enc2_out = text_encoder_two(tok2, output_hidden_states=True)

            # SDXL concatenates penultimate hidden states from both encoders
            prompt_embeds = torch.cat([
                enc1_out.hidden_states[-2],
                enc2_out.hidden_states[-2],
            ], dim=-1).to(dtype=torch.float32)

            # Pooled output from second text encoder
            pooled = enc2_out.text_embeds.to(dtype=torch.float32)

        cached_data.append({
            "latents": latents.cpu(),
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled": pooled.cpu(),
        })

    print(f"Cached {len(cached_data)} samples")

    # Free encoder VRAM
    del vae, text_encoder_one, text_encoder_two
    torch.cuda.empty_cache()
    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after freeing encoders: {vram_used:.1f} GB")

    # --- Optimizer ---
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=num_steps,
    )

    # SDXL time IDs: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    add_time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32, device="cuda"
    )

    # --- Sanity check: verify UNet + inputs before training ---
    print("\nRunning sanity check...")
    test_cached = cached_data[0]
    test_latents = test_cached["latents"].to("cuda")
    test_embeds = test_cached["prompt_embeds"].to("cuda")
    test_pooled = test_cached["pooled"].to("cuda")

    print(f"  latents: shape={test_latents.shape} dtype={test_latents.dtype} has_nan={test_latents.isnan().any()} range=[{test_latents.min():.3f}, {test_latents.max():.3f}]")
    print(f"  prompt_embeds: shape={test_embeds.shape} dtype={test_embeds.dtype} has_nan={test_embeds.isnan().any()} range=[{test_embeds.min():.3f}, {test_embeds.max():.3f}]")
    print(f"  pooled: shape={test_pooled.shape} dtype={test_pooled.dtype} has_nan={test_pooled.isnan().any()} range=[{test_pooled.min():.3f}, {test_pooled.max():.3f}]")

    # Check UNet weights for NaN
    nan_params = 0
    for name, p in unet.named_parameters():
        if p.isnan().any():
            print(f"  NaN in UNet param: {name}")
            nan_params += 1
    print(f"  UNet params with NaN: {nan_params}")

    # Test forward pass
    unet.eval()
    with torch.no_grad():
        test_noise = torch.randn_like(test_latents)
        test_ts = torch.tensor([500], device="cuda").long()
        test_noisy = noise_scheduler.add_noise(test_latents, test_noise, test_ts)
        print(f"  noisy_latents: has_nan={test_noisy.isnan().any()} range=[{test_noisy.min():.3f}, {test_noisy.max():.3f}]")

        test_pred = unet(
            test_noisy,
            test_ts,
            encoder_hidden_states=test_embeds,
            added_cond_kwargs={"text_embeds": test_pooled, "time_ids": add_time_ids},
            return_dict=False,
        )[0]
        print(f"  test_pred: has_nan={test_pred.isnan().any()} range=[{test_pred.min():.3f}, {test_pred.max():.3f}]")

    if test_pred.isnan().any():
        print("\nERROR: UNet produces NaN even in eval mode. Aborting.")
        return {"error": "UNet produces NaN in eval", "run_name": run_name}

    # --- Training ---
    unet.train()
    optimizer.zero_grad()

    print(f"\nStarting training for {num_steps} steps...")
    global_step = 0
    running_loss = 0.0
    grad_accum_steps = 4

    while global_step < num_steps:
        for cached in cached_data:
            if global_step >= num_steps:
                break

            latents = cached["latents"].to("cuda")
            prompt_embeds = cached["prompt_embeds"].to("cuda")
            pooled = cached["pooled"].to("cuda")

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda",
            ).long()

            # Add noise to latents
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

            # MSE loss against noise target
            loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")
            loss_val = loss.item()

            # Check for NaN
            if global_step == 0:
                print(f"  First loss value: {loss_val:.6f}")
                if loss_val != loss_val:  # NaN check
                    print("ERROR: NaN loss on first step!")
                    print(f"  model_pred stats: min={model_pred.min():.4f} max={model_pred.max():.4f} mean={model_pred.mean():.4f}")
                    print(f"  noise stats: min={noise.min():.4f} max={noise.max():.4f}")
                    print(f"  noisy_latents dtype: {noisy_latents.dtype}")
                    print(f"  model_pred dtype: {model_pred.dtype}")
                    return {"error": "NaN loss", "run_name": run_name}

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            running_loss += loss_val
            global_step += 1

            if global_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                lr = optimizer.param_groups[0]["lr"]
                vram_now = torch.cuda.memory_allocated() / 1e9
                print(f"Step {global_step}/{num_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | VRAM: {vram_now:.1f}GB")
                running_loss = 0.0

    # --- Save LoRA adapter ---
    adapter_save_path = f"{DATA_DIR}/adapters/{run_name}"
    Path(adapter_save_path).mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            lora_state_dict[name] = param.data.cpu()

    safetensors.torch.save_file(
        lora_state_dict,
        f"{adapter_save_path}/pytorch_lora_weights.safetensors",
    )

    # Save PEFT config
    unet.peft_config["default"].save_pretrained(adapter_save_path)

    training_data.commit()
    final_loss = running_loss / max(global_step % 50, 1)
    print(f"\nTraining complete! Adapter saved to {adapter_save_path}")
    print(f"Final avg loss: {final_loss:.4f}")

    return {
        "run_name": run_name,
        "steps": global_step,
        "final_loss": final_loss,
        "adapter_path": adapter_save_path,
    }


@app.local_entrypoint()
def main(run_name: str = "v3", num_steps: int = 1000):
    result = train.remote(run_name, num_steps)
    print("\n" + "=" * 40)
    for k, v in result.items():
        print(f"  {k}: {v}")
