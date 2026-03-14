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

"""
We've added GPU snapshotting, an experimental flag. 
We define this with "enable_gpu_snapshot": True in experimental options, and subsquently with snap=True

This isn't a Modal Volume


"""

import os
import time

import modal

app = modal.App("Optimized-Generate-Hopper")

training_data = modal.Volume.from_name("hopper-training-data", create_if_missing=True)
model_cache = modal.Volume.from_name("hopper-model-cache", create_if_missing=True)

DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
ENABLE_REGIONAL_COMPILE = False
ENABLE_FUSED_QKV = True
NEGATIVE_PROMPT = "blurry, low quality, cartoon, anime, 3d render, photograph, photo"
SNAPSHOT_ADAPTER_NAME = os.environ.get("HOPPER_ADAPTER_NAME", "v1")
WARMUP_PROMPT = "hopper style empty diner interior at dawn"
WARMUP_STEPS = 4
WARMUP_SIZE = 1024

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

MINUTES = 60 # 60s

@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window= 5 * MINUTES, # scale down after 5 min of inactivity
    timeout=30 * MINUTES,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_dict(
            {"HOPPER_ADAPTER_NAME": os.environ.get("HOPPER_ADAPTER_NAME", "v1")}
        ),
    ],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    )
class SDXLGenerator:
    @modal.enter(snap=True)
    def init(self):
        import torch
        from diffusers import AutoencoderKL
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

        # TF32 for faster matmul on Ampere GPUs (A10G)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        print("Loading SDXL pipeline in fp16 with fp16-safe VAE...")
        vae = AutoencoderKL.from_pretrained(
            VAE_ID,
            torch_dtype=torch.float16,
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        model_cache.commit()

        if ENABLE_FUSED_QKV:
            self.pipeline.fuse_qkv_projections(vae=False)

        # channels_last is a cheap optimization and doesn't mutate weights the
        # way LoRA loading does, so it's safe to keep.
        self.pipeline.unet.to(memory_format=torch.channels_last)
        self.pipeline.vae.to(memory_format=torch.channels_last)

        # Regional compilation made warm calls faster, but cold restores paid
        # the cost back by recompiling after LoRA weight mutation.
        if ENABLE_REGIONAL_COMPILE:
            self.pipeline.unet.compile_repeated_blocks(fullgraph=True)

        adapter_path = f"{DATA_DIR}/adapters/{SNAPSHOT_ADAPTER_NAME}"
        print(f"Loading snapshot LoRA adapter from {adapter_path}...")
        self.pipeline.load_lora_weights(adapter_path)
        self.snapshot_adapter_name = SNAPSHOT_ADAPTER_NAME

        print(
            f"Running warm-up inference before snapshot "
            f"({WARMUP_STEPS} steps at {WARMUP_SIZE}x{WARMUP_SIZE})..."
        )
        warmup_started_at = time.perf_counter()
        self.pipeline(
            prompt=WARMUP_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WARMUP_SIZE,
            height=WARMUP_SIZE,
            guidance_scale=7.5,
            num_inference_steps=WARMUP_STEPS,
            generator=torch.Generator("cuda").manual_seed(0),
        )
        print(f"Warm-up complete in {time.perf_counter() - warmup_started_at:.2f}s")

    @modal.method()
    def generate(
        self,
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
        from PIL import Image

        request_started_at = time.perf_counter()

        if not use_lora:
            raise ValueError(
                "Base generation is disabled for the snapshotted production path. "
                "The Hopper LoRA is baked into the snapshot."
            )

        if run_name != self.snapshot_adapter_name:
            print(
                f"Requested adapter '{run_name}' differs from snapshotted adapter "
                f"'{self.snapshot_adapter_name}'. Using the snapshotted adapter."
            )

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
            inference_started_at = time.perf_counter()
            # self.pipline() returns StableDiffusionXLPipelineOutput, but not inferrable through torch.compile
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=gen_width,
                height=gen_height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
            ).images[0] 
            inference_elapsed = time.perf_counter() - inference_started_at
            iter_per_second = num_steps / inference_elapsed if inference_elapsed > 0 else 0.0

            # Upscale if needed
            if result.size != (width, height):
                result = result.resize((width, height), Image.Resampling.LANCZOS) # thanks, S.O

            buf = io.BytesIO()
            result.save(buf, format="PNG", quality=95)
            images_bytes.append(buf.getvalue())
            print(
                f"  Generated image {i + 1}/{num_images} in {inference_elapsed:.2f}s "
                f"({iter_per_second:.2f} it/s)"
            )

        print(
            "Request summary: "
            f"total={time.perf_counter() - request_started_at:.2f}s, "
            f"lora=snapshotted({self.snapshot_adapter_name}), "
            f"compile={'on' if ENABLE_REGIONAL_COMPILE else 'off'}"
        )

        return images_bytes


## Local Entrypoint defines arguments to run the function locally
# pass args like a CLI would, like `modal run --prompt "test prompt"`, etc
@app.local_entrypoint()
def main(
    prompt: str,
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
    print(f"LoRA: snapshotted adapter '{SNAPSHOT_ADAPTER_NAME}'")
    if run_name != SNAPSHOT_ADAPTER_NAME:
        print(f"Requested run_name '{run_name}' will be ignored in favor of the snapshotted adapter.")
    if no_lora:
        print("Warning: --no-lora is unsupported for the snapshotted production path.")

    images = SDXLGenerator().generate.remote(
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
