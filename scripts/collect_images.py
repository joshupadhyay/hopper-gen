"""
Download Edward Hopper paintings from WikiArt for LoRA training.

Uses WikiArt's public API to fetch Hopper's catalog, then downloads
the most iconic works. Falls back to a curated list if API is unavailable.

Usage:
    python scripts/collect_images.py
"""

import json
import os
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Curated list of Hopper paintings with WikiArt slugs
# These are his most representative works for style transfer
HOPPER_PAINTINGS = [
    {"slug": "nighthawks", "title": "Nighthawks", "year": 1942},
    {"slug": "automat", "title": "Automat", "year": 1927},
    {"slug": "morning-sun", "title": "Morning Sun", "year": 1952},
    {"slug": "office-at-night", "title": "Office at Night", "year": 1940},
    {"slug": "gas-1940", "title": "Gas", "year": 1940},
    {"slug": "hotel-room", "title": "Hotel Room", "year": 1931},
    {"slug": "chop-suey", "title": "Chop Suey", "year": 1929},
    {"slug": "room-in-new-york", "title": "Room in New York", "year": 1932},
    {"slug": "cape-cod-evening", "title": "Cape Cod Evening", "year": 1939},
    {"slug": "new-york-movie", "title": "New York Movie", "year": 1939},
    {"slug": "sun-in-an-empty-room", "title": "Sun in an Empty Room", "year": 1963},
    {"slug": "hotel-by-a-railroad", "title": "Hotel by a Railroad", "year": 1952},
    {"slug": "eleven-a-m", "title": "Eleven A.M.", "year": 1926},
    {"slug": "western-motel", "title": "Western Motel", "year": 1957},
    {"slug": "second-story-sunlight", "title": "Second Story Sunlight", "year": 1960},
    {"slug": "house-by-the-railroad", "title": "House by the Railroad", "year": 1925},
    {"slug": "night-windows", "title": "Night Windows", "year": 1928},
    {"slug": "sunlight-in-a-cafeteria", "title": "Sunlight in a Cafeteria", "year": 1958},
    {"slug": "people-in-the-sun", "title": "People in the Sun", "year": 1960},
    {"slug": "conference-at-night", "title": "Conference at Night", "year": 1949},
    {"slug": "summer-evening", "title": "Summer Evening", "year": 1947},
    {"slug": "hotel-lobby", "title": "Hotel Lobby", "year": 1943},
    {"slug": "drug-store", "title": "Drug Store", "year": 1927},
    {"slug": "cape-cod-morning", "title": "Cape Cod Morning", "year": 1950},
    {"slug": "pennsylvania-coal-town", "title": "Pennsylvania Coal Town", "year": 1947},
    {"slug": "rooms-by-the-sea", "title": "Rooms by the Sea", "year": 1951},
    {"slug": "compartment-c-car", "title": "Compartment C, Car 293", "year": 1938},
    {"slug": "summertime", "title": "Summertime", "year": 1943},
    {"slug": "office-in-a-small-city", "title": "Office in a Small City", "year": 1953},
    {"slug": "east-wind-over-weehawken", "title": "East Wind Over Weehawken", "year": 1934},
    {"slug": "girlie-show", "title": "Girlie Show", "year": 1941},
    {"slug": "morning-in-a-city", "title": "Morning in a City", "year": 1944},
    {"slug": "ground-swell", "title": "Ground Swell", "year": 1939},
    {"slug": "lighthouse-at-two-lights", "title": "Lighthouse at Two Lights", "year": 1929},
    {"slug": "the-long-leg", "title": "The Long Leg", "year": 1935},
]

WIKIART_BASE = "https://www.wikiart.org"


def download_painting(slug: str, title: str) -> bool:
    """Download a single painting from WikiArt."""
    filename = f"{slug}.jpg"
    filepath = IMAGES_DIR / filename

    if filepath.exists():
        print(f"  ✓ Already exists: {filename}")
        return True

    # Try WikiArt painting page to get image URL
    try:
        url = f"{WIKIART_BASE}/en/edward-hopper/{slug}"
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (educational art research)"
        })
        resp.raise_for_status()

        # Extract og:image meta tag for the full painting image
        import re
        match = re.search(r'<meta property="og:image" content="([^"]+)"', resp.text)
        if not match:
            print(f"  ✗ No image found for: {title}")
            return False

        image_url = match.group(1)

        # Download the image
        img_resp = requests.get(image_url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (educational art research)"
        })
        img_resp.raise_for_status()

        filepath.write_bytes(img_resp.content)
        print(f"  ✓ Downloaded: {title} -> {filename}")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {title} — {e}")
        return False


def main():
    print(f"Downloading {len(HOPPER_PAINTINGS)} Edward Hopper paintings...")
    print(f"Output directory: {IMAGES_DIR}\n")

    success = 0
    for painting in HOPPER_PAINTINGS:
        ok = download_painting(painting["slug"], painting["title"])
        if ok:
            success += 1
        time.sleep(1)  # Be polite to WikiArt

    print(f"\nDone: {success}/{len(HOPPER_PAINTINGS)} downloaded")

    # Save metadata for reference
    metadata_path = DATA_DIR / "paintings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(HOPPER_PAINTINGS, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
