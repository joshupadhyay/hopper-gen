"""Call the deployed Hopper generator with memory snapshots."""
import modal
from pathlib import Path

SDXLGenerator = modal.Cls.from_name("Optimized-Generate-Hopper", "SDXLGenerator")
instance = SDXLGenerator()

result = instance.generate.remote(
    prompt="hopper style sunny day at the pool",
    run_name="v12",
    seed=42,
)

Path("outputs").mkdir(exist_ok=True)
for i, img_bytes in enumerate(result):
    Path(f"outputs/snapshot_test_{i}.png").write_bytes(img_bytes)
    print(f"Saved outputs/snapshot_test_{i}.png")
