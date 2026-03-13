"""
Generate images using the trained Hopper LoRA on Modal.

Supports custom dimensions for different use cases:
  --preset desktop   → 2560x1440 (16:9 ultrawide)
  --preset phone     → 1170x2532 (iPhone 14 Pro)
  --preset square    → 1024x1024 (default SDXL)
  --preset landscape → 1920x1080 (standard HD)
  --preset portrait  → 1080x1920 (vertical HD)
  --width W --height H  → custom dimensions

Usage:
    modal run scripts/generate.py --prompt "hopper style painting, ..."
    modal run scripts/generate.py --prompt "..." --preset desktop
    modal run scripts/generate.py --prompt "..." --width 1920 --height 1080
    modal run scripts/generate.py --prompt "..." --run-name v1 --no-lora
"""

import modal

app = modal.App("hopper-lora-generate")

training_data = modal.Volume.from_name("hopper-training-data", create_if_missing=True)
model_cache = modal.Volume.from_name("hopper-model-cache", create_if_missing=True)

DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

PRESETS = {
    "square": (1024, 1024),
    "desktop": (2560, 1440),
    "phone": (1170, 2532),
    "landscape": (1920, 1080),
    "portrait": (1080, 1920),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "safetensors",
        "Pillow",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(
    prompt: str,
    run_name: str = "v1",
    use_lora: bool = True,
    num_images: int = 1,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
) -> list[bytes]:
    import io
    import torch
    from diffusers import StableDiffusionXLPipeline
    from pathlib import Path

    print(f"Loading SDXL pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    model_cache.commit()

    if use_lora:
        adapter_path = f"{DATA_DIR}/adapters/{run_name}"
        print(f"Loading LoRA adapter from {adapter_path}...")
        pipeline.load_lora_weights(adapter_path)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    # Generate at SDXL-native resolution then upscale if needed
    # SDXL works best at 1024x1024 — we generate there and resize for non-square
    gen_width, gen_height = width, height

    # Clamp to multiples of 8 (required by SDXL VAE)
    gen_width = (gen_width // 8) * 8
    gen_height = (gen_height // 8) * 8

    # For very large outputs, generate at a smaller size and upscale
    max_pixels = 1024 * 1024 * 2  # ~2 megapixels max for generation
    if gen_width * gen_height > max_pixels:
        scale = (max_pixels / (gen_width * gen_height)) ** 0.5
        gen_width = int(gen_width * scale) // 8 * 8
        gen_height = int(gen_height * scale) // 8 * 8

    print(f"Generating {num_images} image(s) at {gen_width}x{gen_height}...")
    if gen_width != width or gen_height != height:
        print(f"  (will upscale to {width}x{height})")

    images_bytes = []
    for i in range(num_images):
        result = pipeline(
            prompt=prompt,
            negative_prompt="blurry, low quality, cartoon, anime, 3d render, photograph, photo",
            width=gen_width,
            height=gen_height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]

        # Upscale if needed
        if result.size != (width, height):
            from PIL import Image
            result = result.resize((width, height), Image.LANCZOS)

        buf = io.BytesIO()
        result.save(buf, format="PNG", quality=95)
        images_bytes.append(buf.getvalue())
        print(f"  Generated image {i + 1}/{num_images}")



    return images_bytes


@app.local_entrypoint()
def main(
    prompt: str = "hopper style painting, a woman reading alone in a sunlit apartment, warm morning light, muted colors",
    run_name: str = "v1",
    no_lora: bool = False,
    num_images: int = 1,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    preset: str = "square",
    width: int = 0,
    height: int = 0,
    seed: int = -1,
):
    from pathlib import Path

    # Resolve dimensions
    if width > 0 and height > 0:
        w, h = width, height
    elif preset in PRESETS:
        w, h = PRESETS[preset]
    else:
        print(f"Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}")
        return

    print(f"Prompt: {prompt}")
    print(f"Size: {w}x{h} (preset: {preset})")
    print(f"LoRA: {'yes' if not no_lora else 'no'} (run: {run_name})")

    images = generate.remote(
        prompt=prompt,
        run_name=run_name,
        use_lora=not no_lora,
        num_images=num_images,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        width=w,
        height=h,
        seed=seed,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    lora_tag = "lora" if not no_lora else "base"
    for i, img_bytes in enumerate(images):
        filename = f"hopper_{lora_tag}_{preset}_{i}.png"
        path = output_dir / filename
        path.write_bytes(img_bytes)
        print(f"Saved: {path}")
