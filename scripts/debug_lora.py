"""Debug script: inspect saved LoRA weights and test loading them."""
import modal

app = modal.App("hopper-lora-debug")

training_data = modal.Volume.from_name("hopper-training-data", create_if_missing=True)
model_cache = modal.Volume.from_name("hopper-model-cache", create_if_missing=True)

DATA_DIR = "/data"
HF_CACHE_DIR = "/root/.cache/huggingface"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "diffusers", "transformers", "accelerate",
        "peft", "safetensors", "Pillow",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={DATA_DIR: training_data, HF_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def debug(run_name: str = "v10-smoke"):
    import torch
    from pathlib import Path
    from safetensors.torch import load_file
    from diffusers import StableDiffusionXLPipeline

    adapter_path = f"{DATA_DIR}/adapters/{run_name}"
    print(f"\n=== Inspecting {adapter_path} ===")

    # List files
    for f in sorted(Path(adapter_path).iterdir()):
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")

    # Load and inspect keys
    weight_file = Path(adapter_path) / "pytorch_lora_weights.safetensors"
    if not weight_file.exists():
        weight_file = Path(adapter_path) / "adapter_model.safetensors"

    state = load_file(str(weight_file))
    keys = sorted(state.keys())
    print(f"\n=== {len(keys)} keys in {weight_file.name} ===")
    print("First 10 keys:")
    for k in keys[:10]:
        print(f"  {k}  shape={state[k].shape}")
    print("Last 5 keys:")
    for k in keys[-5:]:
        print(f"  {k}  shape={state[k].shape}")

    # Check for common prefixes
    prefixes = set()
    for k in keys:
        parts = k.split(".")
        if len(parts) > 0:
            prefixes.add(parts[0])
    print(f"\nTop-level prefixes: {prefixes}")

    # Try loading into pipeline
    print("\n=== Attempting pipeline.load_lora_weights() ===")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    model_cache.commit()

    try:
        pipeline.load_lora_weights(adapter_path, weight_name=weight_file.name)
        print("✓ load_lora_weights succeeded!")

        # Verify it actually affects output
        gen1 = torch.Generator("cuda").manual_seed(42)
        img1 = pipeline("test", num_inference_steps=5, generator=gen1, width=512, height=512).images[0]

        pipeline.unload_lora_weights()
        gen2 = torch.Generator("cuda").manual_seed(42)
        img2 = pipeline("test", num_inference_steps=5, generator=gen2, width=512, height=512).images[0]

        import numpy as np
        arr1, arr2 = np.array(img1), np.array(img2)
        diff = np.abs(arr1.astype(float) - arr2.astype(float)).mean()
        print(f"Mean pixel difference (lora vs base): {diff:.2f}")
        if diff < 1.0:
            print("⚠ WARNING: Images are nearly identical — LoRA may not be loading")
        else:
            print("✓ Images differ — LoRA is affecting generation!")
    except Exception as e:
        print(f"✗ load_lora_weights failed: {e}")

        # Try alternate loading: strip base_model.model. and save to temp file
        print("\n=== Trying alternate: strip base_model.model. prefix ===")
        from safetensors.torch import save_file
        stripped = {k.replace("base_model.model.", ""): v for k, v in state.items()}
        print("Stripped keys sample:")
        for k in list(stripped.keys())[:5]:
            print(f"  {k}")

        tmp_dir = "/tmp/lora_test"
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        save_file(stripped, f"{tmp_dir}/pytorch_lora_weights.safetensors")

        try:
            pipeline.load_lora_weights(tmp_dir)
            print("✓ Stripped version loaded!")

            import numpy as np
            gen1 = torch.Generator("cuda").manual_seed(42)
            img1 = pipeline("hopper style painting, woman in sunlit room", num_inference_steps=10, generator=gen1, width=512, height=512).images[0]

            pipeline.unload_lora_weights()
            gen2 = torch.Generator("cuda").manual_seed(42)
            img2 = pipeline("hopper style painting, woman in sunlit room", num_inference_steps=10, generator=gen2, width=512, height=512).images[0]

            arr1, arr2 = np.array(img1), np.array(img2)
            diff = np.abs(arr1.astype(float) - arr2.astype(float)).mean()
            print(f"Mean pixel difference (lora vs base): {diff:.2f}")
            if diff < 1.0:
                print("⚠ WARNING: Images nearly identical — LoRA not working")
            else:
                print("✓ LoRA IS WORKING! Images differ!")

            # Return images as bytes
            import io
            buf1, buf2 = io.BytesIO(), io.BytesIO()
            img1.save(buf1, format="PNG")
            img2.save(buf2, format="PNG")
            return buf1.getvalue(), buf2.getvalue()
        except Exception as e2:
            print(f"✗ Stripped version also failed: {e2}")
            return None, None


@app.local_entrypoint()
def main(run_name: str = "v10-smoke"):
    from pathlib import Path
    result = debug.remote(run_name)
    if result and result[0] and result[1]:
        out = Path("outputs/debug")
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_lora.png").write_bytes(result[0])
        (out / "test_base.png").write_bytes(result[1])
        print(f"Saved to {out}/test_lora.png and test_base.png")
