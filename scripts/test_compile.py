"""Single-image test for torch.compile benchmarking."""
import modal
from pathlib import Path
import time

SDXLGenerator = modal.Cls.from_name("Optimized-Generate-Hopper", "SDXLGenerator")
instance = SDXLGenerator()

Path("outputs").mkdir(exist_ok=True)

print("Calling generate (first call triggers JIT compile)...")
start = time.time()
result = instance.generate.remote(
    prompt="hopper style a quiet diner at dawn",
    run_name="v12",
    seed=42,
)
elapsed = time.time() - start
print(f"Total remote call: {elapsed:.1f}s")

for i, img_bytes in enumerate(result):
    Path(f"outputs/compile_test_{i}.png").write_bytes(img_bytes)
    print(f"Saved outputs/compile_test_{i}.png")
