"""Call the deployed Hopper generator with memory snapshots."""
import modal
from pathlib import Path

SDXLGenerator = modal.Cls.from_name("Optimized-Generate-Hopper", "SDXLGenerator")
instance = SDXLGenerator()

prompts = {
    "west_side_highway": "hopper style walk down west side highway with little island, warm afternoon light, hudson river, muted colors",
    "baseball_game": "hopper style baseball game, stadium lights, crowd in stands, evening atmosphere, dramatic shadows",
    "opera_house": "hopper style opera in the opera house, grand interior, warm stage lighting, audience silhouettes, red curtains",
}

Path("outputs").mkdir(exist_ok=True)
for name, prompt in prompts.items():
    print(f"Generating: {name}...")
    result = instance.generate.remote(
        prompt=prompt,
        run_name="v12",
        seed=42,
    )
    for i, img_bytes in enumerate(result):
        Path(f"outputs/{name}_{i}.png").write_bytes(img_bytes)
        print(f"Saved outputs/{name}_{i}.png")
