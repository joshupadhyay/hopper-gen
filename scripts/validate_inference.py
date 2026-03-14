"""Validate the finalized deployed Hopper inference path."""

from __future__ import annotations

import hashlib
from pathlib import Path

import modal

APP_NAME = "Optimized-Generate-Hopper"
CLASS_NAME = "SDXLGenerator"
RUN_NAME = "v12"
SEED = 42
PROMPT = "hopper style a quiet diner at dawn with rain on the windows"
NON_SQUARE_WIDTH = 1920
NON_SQUARE_HEIGHT = 1080


def digest_image_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_image(path: str, data: bytes) -> None:
    Path("outputs").mkdir(exist_ok=True)
    Path(path).write_bytes(data)


def main():
    generator_cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    instance = generator_cls()

    print("1. Generate with LoRA enabled")
    lora_result = instance.generate.remote(
        prompt=PROMPT,
        run_name=RUN_NAME,
        seed=SEED,
    )
    lora_image = lora_result[0]
    save_image("outputs/validate_lora.png", lora_image)
    print(f"   saved outputs/validate_lora.png sha256={digest_image_bytes(lora_image)}")

    print("2. Generate the same prompt and seed again to verify determinism")
    lora_repeat_result = instance.generate.remote(
        prompt=PROMPT,
        run_name=RUN_NAME,
        seed=SEED,
    )
    lora_repeat_image = lora_repeat_result[0]
    save_image("outputs/validate_lora_repeat.png", lora_repeat_image)
    lora_digest = digest_image_bytes(lora_image)
    lora_repeat_digest = digest_image_bytes(lora_repeat_image)
    deterministic = lora_digest == lora_repeat_digest
    print(
        f"   saved outputs/validate_lora_repeat.png sha256={lora_repeat_digest} "
        f"deterministic={'yes' if deterministic else 'no'}"
    )

    print("3. Generate a non-square image to validate clamp and upscale flow")
    wide_result = instance.generate.remote(
        prompt="hopper style drive down the west side highway in warm afternoon light",
        run_name=RUN_NAME,
        seed=SEED,
        width=NON_SQUARE_WIDTH,
        height=NON_SQUARE_HEIGHT,
    )
    wide_image = wide_result[0]
    save_image("outputs/validate_wide.png", wide_image)
    print(
        f"   saved outputs/validate_wide.png target={NON_SQUARE_WIDTH}x{NON_SQUARE_HEIGHT} "
        f"sha256={digest_image_bytes(wide_image)}"
    )

    print("\nValidation summary")
    print(f"- deterministic_same_seed={'yes' if deterministic else 'no'}")
    print("- lora_changes_output=validate separately with scripts/evaluate.py or scripts/generate.py")
    print(f"- non_square_generated=yes")


if __name__ == "__main__":
    main()
