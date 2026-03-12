"""
Prepare training data: resize images to 1024x1024 and generate captions.

Reads downloaded paintings from data/images/, resizes them, and creates
a captions.jsonl file with Hopper-style descriptions for each image.

Usage:
    python scripts/prepare_data.py
"""

import json
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 1024

# Hand-written captions for each painting — the quality of these matters a lot
# for LoRA training. Each starts with "hopper style" activation phrase.
CAPTIONS = {
    "nighthawks": "hopper style painting, interior of a late-night diner with large glass windows, three patrons and a server, harsh fluorescent lighting against dark empty street, muted greens and warm yellows, urban isolation",
    "automat": "hopper style painting, solitary woman in a hat sitting at a cafe table, single overhead light, dark window reflecting the interior, warm browns and deep blacks, quiet contemplation",
    "morning-sun": "hopper style painting, woman sitting on a bed in bright morning sunlight streaming through a window, pink and white walls, geometric shadows, serene isolation",
    "office-at-night": "hopper style painting, man at a desk and woman standing by a filing cabinet in an office, dramatic overhead lighting and desk lamp, deep blues and warm yellows, tension and stillness",
    "gas-1940": "hopper style painting, isolated gas station at dusk with three red pumps, warm interior light against darkening sky, empty road leading into dark trees, rural solitude",
    "hotel-room": "hopper style painting, woman sitting on a hotel bed reading, harsh overhead light, suitcase on floor, muted yellows and browns, transient solitude",
    "chop-suey": "hopper style painting, two women at a restaurant table near a window, warm interior light, window sign partially visible, intimate urban scene with warm reds and yellows",
    "room-in-new-york": "hopper style painting, couple in an apartment viewed through an open window, man reading newspaper woman at piano, warm interior light, voyeuristic perspective, domestic disconnect",
    "cape-cod-evening": "hopper style painting, couple standing on a porch with a dog in tall grass, late afternoon golden light, white clapboard house, dark trees behind, quiet rural evening",
    "new-york-movie": "hopper style painting, female usherette leaning against a wall in a dimly lit movie theater, warm light on her figure, dark theater interior, urban isolation amid crowd",
    "sun-in-an-empty-room": "hopper style painting, empty room with bright sunlight streaming through a window casting geometric shadows on bare walls, no figures, pure light and geometry, meditative stillness",
    "hotel-by-a-railroad": "hopper style painting, woman at a window and man reading in a hotel room, railroad tracks visible outside, morning light, muted tones, couple in parallel solitude",
    "eleven-a-m": "hopper style painting, nude woman sitting in an armchair by a sunlit window, warm morning light, simple interior, contemplative solitude, soft warm palette",
    "western-motel": "hopper style painting, woman sitting on a bed in a motel room, large picture window showing a car and landscape, bright daylight, retro American travel",
    "second-story-sunlight": "hopper style painting, two women on an upper porch and balcony of a white house, bright afternoon sunlight, dark roof shadow, architectural geometry, suburban stillness",
    "house-by-the-railroad": "hopper style painting, Victorian mansion beside railroad tracks, harsh afternoon side-lighting, dramatic shadows, isolated architecture against empty sky, lonely grandeur",
    "night-windows": "hopper style painting, voyeuristic view through apartment windows at night, woman bending in warm interior light, three windows across dark facade, urban nighttime glimpse",
    "sunlight-in-a-cafeteria": "hopper style painting, man and woman at separate tables in a sunlit cafeteria, bright geometric light patches on floor and wall, urban daytime, quiet distance between figures",
    "people-in-the-sun": "hopper style painting, five people sitting in chairs facing the sun outside a building, bright daylight, long shadows, collective solitude, geometric composition",
    "conference-at-night": "hopper style painting, three figures in a dimly lit office at night, overhead light illuminating a desk, dark windows, tension and secrecy, film noir atmosphere",
    "summer-evening": "hopper style painting, young couple on a porch at dusk, warm light from doorway, dark surrounding landscape, intimate summer night, warm and cool contrast",
    "hotel-lobby": "hopper style painting, elderly couple and young woman in a hotel lobby, overhead lighting, columns and dark furniture, muted elegance, social observation",
    "drug-store": "hopper style painting, brightly lit drugstore at night with large windows, warm interior glow against dark street, green and red accents, nocturnal urban commerce",
    "cape-cod-morning": "hopper style painting, woman leaning out of a bay window of a white house, bright morning light, green grass, architectural geometry, anticipation and longing",
    "pennsylvania-coal-town": "hopper style painting, man raking lawn beside a row of houses, bright afternoon sun, long shadows, muted earth tones, suburban routine and quiet",
    "rooms-by-the-sea": "hopper style painting, sunlit room with open doorway showing ocean directly outside, geometric light on walls, no figures, liminal space between interior and sea",
    "compartment-c-car": "hopper style painting, woman reading alone in a train compartment, warm overhead light, dark window, plush green seat, travel solitude and concentration",
    "summertime": "hopper style painting, young woman in white dress standing at building entrance on a bright day, strong sunlight and sharp shadows, urban summer heat, classical architecture",
    "office-in-a-small-city": "hopper style painting, man sitting at desk in a small office with large windows, city buildings visible outside, bright daylight, corporate solitude, geometric framing",
    "east-wind-over-weehawken": "hopper style painting, row of houses on a hill with dramatic cloudy sky, golden afternoon light on facades, empty street, suburban landscape, architectural geometry",
    "girlie-show": "hopper style painting, burlesque performer on stage in spotlight, dark audience below, warm spotlight against deep shadows, theatrical isolation, bold figure",
    "morning-in-a-city": "hopper style painting, nude woman standing at a window in morning light, sunlight on her body and bare walls, simple bedroom interior, quiet dawn solitude",
    "ground-swell": "hopper style painting, sailboat on ocean swells with a buoy, four figures aboard, bright daylight on blue-green water, maritime serenity and subtle tension",
    "lighthouse-at-two-lights": "hopper style painting, white lighthouse and keeper's house on rocky coast, bright clear sky, strong architectural forms against landscape, coastal New England",
    "the-long-leg": "hopper style painting, sailboat on deep blue water with distant white lighthouse, bright sunny day, blue sky and sea, crisp geometric sail, maritime solitude",
}


def resize_with_padding(img: Image.Image, target_size: int) -> Image.Image:
    """Resize image to target_size x target_size with center crop."""
    # Resize so the shorter side matches target
    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))
    return img


def main():
    images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    if not images:
        print("No images found in data/images/. Run collect_images.py first.")
        return

    print(f"Found {len(images)} images. Processing...")

    captions_data = []
    for img_path in sorted(images):
        slug = img_path.stem
        caption = CAPTIONS.get(slug)
        if not caption:
            print(f"  ⚠ No caption for {slug}, skipping")
            continue

        # Resize
        img = Image.open(img_path).convert("RGB")
        img = resize_with_padding(img, TARGET_SIZE)
        out_path = PROCESSED_DIR / f"{slug}.jpg"
        img.save(out_path, quality=95)

        captions_data.append({
            "file_name": f"{slug}.jpg",
            "text": caption,
        })
        print(f"  ✓ {slug}: {img_path.stat().st_size // 1024}KB -> {out_path.stat().st_size // 1024}KB")

    # Write captions
    captions_path = DATA_DIR / "captions.jsonl"
    with open(captions_path, "w") as f:
        for entry in captions_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone: {len(captions_data)} images processed")
    print(f"Captions: {captions_path}")
    print(f"Images: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
