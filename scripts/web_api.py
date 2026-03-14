"""
Modal web endpoints for Hopper generation and image file delivery.

Deploy with:
    modal deploy scripts/web_api.py
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import modal

from scripts.class_generate import (
    DATA_DIR,
    ENABLE_FUSED_QKV,
    ENABLE_REGIONAL_COMPILE,
    HF_CACHE_DIR,
    MODEL_ID,
    NEGATIVE_PROMPT,
    SNAPSHOT_ADAPTER_NAME,
    VAE_ID,
    WARMUP_PROMPT,
    WARMUP_SIZE,
    WARMUP_STEPS,
    model_cache,
    image,
)

app = modal.App("hopper-web-api")

gallery_volume = modal.Volume.from_name("hopper-gallery", create_if_missing=True)
GALLERY_DIR = "/gallery"


@app.cls(
    image=image.pip_install("fastapi[standard]"),
    gpu="A10G",
    scaledown_window=5 * 60,
    timeout=15 * 60,
    volumes={
        DATA_DIR: modal.Volume.from_name("hopper-training-data", create_if_missing=True),
        HF_CACHE_DIR: model_cache,
        GALLERY_DIR: gallery_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_dict(
            {"HOPPER_ADAPTER_NAME": os.environ.get("HOPPER_ADAPTER_NAME", "v1")}
        ),
    ],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class HopperWebGenerator:
    @modal.enter(snap=True)
    def init(self):
        import torch
        from diffusers import AutoencoderKL
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
            StableDiffusionXLPipeline,
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
        self.pipeline.unet.to(memory_format=torch.channels_last)
        self.pipeline.vae.to(memory_format=torch.channels_last)

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
    def generate_and_store(
        self,
        prompt: str,
        run_name: str,
        adapter_name: str = "v1",
        guidance_scale: float = 7.5,
        num_steps: int = 50,
        width: int = 1024,
        height: int = 1024,
    ):
        import torch

        request_started_at = time.perf_counter()
        modal_path = f"generated/{run_name}.png"
        output_path = Path(GALLERY_DIR) / modal_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if adapter_name != self.snapshot_adapter_name:
            raise ValueError(
                f"Requested adapter '{adapter_name}' does not match the snapshotted "
                f"adapter '{self.snapshot_adapter_name}'."
            )

        inference_started_at = time.perf_counter()
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=torch.Generator("cuda"),
        ).images[0]
        inference_elapsed = time.perf_counter() - inference_started_at
        iter_per_second = num_steps / inference_elapsed if inference_elapsed > 0 else 0.0
        print(f"  Generated image in {inference_elapsed:.2f}s ({iter_per_second:.2f} it/s)")

        with io.BytesIO() as buffer:
            result.save(buffer, format="PNG", quality=95)
            output_path.write_bytes(buffer.getvalue())

        gallery_volume.commit()
        print(
            "Request summary: "
            f"total={time.perf_counter() - request_started_at:.2f}s, "
            f"lora=snapshotted({self.snapshot_adapter_name}), "
            f"compile={'on' if ENABLE_REGIONAL_COMPILE else 'off'}"
        )

        return {
            "modal_path": modal_path,
            "content_type": "image/png",
            "width": width,
            "height": height,
        }

@app.function(
    image=image.pip_install("fastapi[standard]"),
    volumes={GALLERY_DIR: gallery_volume},
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException, Response
    from pydantic import BaseModel

    web_app = FastAPI()

    class GenerateRequest(BaseModel):
        prompt: str
        runName: str
        adapterName: str = "v1"
        guidanceScale: float = 7.5
        numSteps: int = 50
        width: int = 1024
        height: int = 1024

    @web_app.post("/generate")
    def generate(payload: GenerateRequest):
        return HopperWebGenerator().generate_and_store.remote(
            prompt=payload.prompt,
            run_name=payload.runName,
            adapter_name=payload.adapterName,
            guidance_scale=payload.guidanceScale,
            num_steps=payload.numSteps,
            width=payload.width,
            height=payload.height,
        )

    @web_app.get("/files/{path:path}")
    def files(path: str):
        file_path = Path(GALLERY_DIR) / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Not found")

        return Response(content=file_path.read_bytes(), media_type="image/png")

    return web_app
