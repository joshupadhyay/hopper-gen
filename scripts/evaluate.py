"""
Evaluate trained LoRA by generating holdout prompts with and without the adapter.

Produces side-by-side comparison images to visually assess style transfer quality.

Usage:
    modal run scripts/evaluate.py
    modal run scripts/evaluate.py --run-name v1 --preset desktop
"""

import modal

app = modal.App("hopper-lora-evaluate")

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

HOLDOUT_PROMPTS = [
    "hopper style painting, a woman reading alone in a sunlit apartment, warm morning light",
    "hopper style painting, an empty gas station at dusk, lonely road, fading light",
    "hopper style painting, office workers seen through a window at night, fluorescent glow",
    "hopper style painting, a lighthouse on a rocky coast, bright afternoon light, clear sky",
    "hopper style painting, a solitary man at a hotel room desk, harsh overhead lamp",
]

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
    timeout=1800,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate(
    run_name: str = "v1",
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
) -> list[tuple[str, bytes, bytes]]:
    """Generate holdout prompts with and without LoRA. Returns (prompt, base_img, lora_img) tuples."""
    import io
    import torch
    from diffusers import StableDiffusionXLPipeline

    print("Loading SDXL pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    model_cache.commit()

    # Clamp to multiples of 8
    gen_width = (width // 8) * 8
    gen_height = (height // 8) * 8

    # Scale down if too large for generation
    max_pixels = 1024 * 1024 * 2
    if gen_width * gen_height > max_pixels:
        scale = (max_pixels / (gen_width * gen_height)) ** 0.5
        gen_width = int(gen_width * scale) // 8 * 8
        gen_height = int(gen_height * scale) // 8 * 8

    negative_prompt = "blurry, low quality, cartoon, anime, 3d render, photograph, photo"
    results = []

    for i, prompt in enumerate(HOLDOUT_PROMPTS):
        print(f"\n[{i + 1}/{len(HOLDOUT_PROMPTS)}] {prompt[:60]}...")
        generator = torch.Generator("cuda").manual_seed(seed)

        # --- Base SDXL (no LoRA) ---
        print("  Generating base...")
        base_img = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=gen_width,
            height=gen_height,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

        if base_img.size != (width, height):
            from PIL import Image
            base_img = base_img.resize((width, height), Image.LANCZOS)

        base_buf = io.BytesIO()
        base_img.save(base_buf, format="PNG")

        # --- With LoRA ---
        adapter_path = f"{DATA_DIR}/adapters/{run_name}"
        print(f"  Loading LoRA from {adapter_path}...")
        pipeline.load_lora_weights(adapter_path)

        generator = torch.Generator("cuda").manual_seed(seed)
        print("  Generating with LoRA...")
        lora_img = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=gen_width,
            height=gen_height,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

        if lora_img.size != (width, height):
            from PIL import Image
            lora_img = lora_img.resize((width, height), Image.LANCZOS)

        lora_buf = io.BytesIO()
        lora_img.save(lora_buf, format="PNG")

        # Unload LoRA for next base comparison
        pipeline.unload_lora_weights()

        results.append((prompt, base_buf.getvalue(), lora_buf.getvalue()))

    return results


@app.local_entrypoint()
def main(
    run_name: str = "v1",
    preset: str = "square",
    width: int = 0,
    height: int = 0,
    seed: int = 42,
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

    print(f"Evaluating LoRA '{run_name}' with {len(HOLDOUT_PROMPTS)} holdout prompts")
    print(f"Output size: {w}x{h} (preset: {preset})")

    results = evaluate.remote(
        run_name=run_name,
        width=w,
        height=h,
        seed=seed,
    )

    output_dir = Path("outputs/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (prompt, base_bytes, lora_bytes) in enumerate(results):
        base_path = output_dir / f"eval_{i}_base.png"
        lora_path = output_dir / f"eval_{i}_lora.png"
        base_path.write_bytes(base_bytes)
        lora_path.write_bytes(lora_bytes)
        print(f"\n[{i + 1}] {prompt[:60]}...")
        print(f"  Base: {base_path}")
        print(f"  LoRA: {lora_path}")

    print(f"\nAll outputs in {output_dir}/")
