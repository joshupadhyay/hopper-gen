"""Benchmark the deployed Hopper generator for first-request latency stability."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import modal

APP_NAME = "Optimized-Generate-Hopper"
CLASS_NAME = "SDXLGenerator"
PROMPT = (
    "hopper style nighttime street corner, neon signs, rain on pavement, "
    "solitary figure with umbrella"
)
RUN_NAME = "v12"
SEED = 42
WARM_RUNS = 3
COOLDOWN_SECONDS = 7 * 60


def timed_generate(instance, *, label: str, prompt: str, run_name: str, seed: int, save_image: bool):
    started_at = time.perf_counter()
    result = instance.generate.remote(
        prompt=prompt,
        run_name=run_name,
        seed=seed,
    )
    total = time.perf_counter() - started_at

    if save_image:
        Path("outputs").mkdir(exist_ok=True)
        Path(f"outputs/{label}.png").write_bytes(result[0])

    return {
        "label": label,
        "total_seconds": round(total, 2),
        "saved_image": save_image,
        "dynamic_lora": True,
    }


def main():
    generator_cls = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    instance = generator_cls()

    results = []

    print("Run 1: first request after deploy/snapshot rebuild")
    results.append(
        timed_generate(
            instance,
            label="benchmark_first_request",
            prompt=PROMPT,
            run_name=RUN_NAME,
            seed=SEED,
            save_image=True,
        )
    )

    print("Warm runs:")
    warm_totals = []
    for index in range(WARM_RUNS):
        result = timed_generate(
            instance,
            label=f"benchmark_warm_{index + 1}",
            prompt=PROMPT,
            run_name=RUN_NAME,
            seed=SEED,
            save_image=index == 0,
        )
        results.append(result)
        warm_totals.append(result["total_seconds"])
        print(f"  Warm {index + 1}: {result['total_seconds']}s")

    print(f"Waiting {COOLDOWN_SECONDS}s for container scale-down before cold-restore test...")
    time.sleep(COOLDOWN_SECONDS)

    print("Run after scale-down: first request after snapshot restore")
    results.append(
        timed_generate(
            instance,
            label="benchmark_after_restore",
            prompt=PROMPT,
            run_name=RUN_NAME,
            seed=SEED,
            save_image=True,
        )
    )

    print("\nSummary")
    for result in results:
        print(
            f"- {result['label']}: total={result['total_seconds']}s, "
            f"dynamic_lora={'yes' if result['dynamic_lora'] else 'no'}"
        )

    if warm_totals:
        print(
            f"- warm_mean={statistics.mean(warm_totals):.2f}s, "
            f"warm_min={min(warm_totals):.2f}s, warm_max={max(warm_totals):.2f}s"
        )


if __name__ == "__main__":
    main()
