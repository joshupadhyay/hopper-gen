"""
Download Edward Hopper paintings from Wikimedia Commons for LoRA training.

Uses the Wikimedia Commons API to fetch public domain Hopper paintings.
Hopper died in 1967 — his works are in the public domain in the US.

Usage:
    python scripts/collect_images.py
"""

import json
import time
from pathlib import Path
from urllib.parse import unquote

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "HopperGenBot/1.0 (educational ML art research; Python requests)",
})

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Mapping: slug -> Wikimedia Commons filename
# These are exact filenames on Wikimedia Commons for Hopper paintings
HOPPER_PAINTINGS = [
    {"slug": "nighthawks", "title": "Nighthawks", "year": 1942,
     "commons": "Nighthawks_by_Edward_Hopper_1942.jpg"},
    {"slug": "automat", "title": "Automat", "year": 1927,
     "commons": "Edward_Hopper_Automat_1927.jpg"},
    {"slug": "morning-sun", "title": "Morning Sun", "year": 1952,
     "commons": "Edward_Hopper_Morning_Sun.jpg"},
    {"slug": "office-at-night", "title": "Office at Night", "year": 1940,
     "commons": "Edward_Hopper_Office_at_Night.jpg"},
    {"slug": "gas-1940", "title": "Gas", "year": 1940,
     "commons": "Edward_Hopper_Gas_1940.jpg"},
    {"slug": "hotel-room", "title": "Hotel Room", "year": 1931,
     "commons": "Edward_Hopper_Hotel_Room.jpg"},
    {"slug": "chop-suey", "title": "Chop Suey", "year": 1929,
     "commons": "Edward_Hopper_Chop_Suey.jpg"},
    {"slug": "room-in-new-york", "title": "Room in New York", "year": 1932,
     "commons": "Edward_Hopper,_Room_in_New_York,_1932.jpg"},
    {"slug": "cape-cod-evening", "title": "Cape Cod Evening", "year": 1939,
     "commons": "Cape_Cod_Evening_by_Edward_Hopper_1939.jpg"},
    {"slug": "new-york-movie", "title": "New York Movie", "year": 1939,
     "commons": "New_York_Movie_1939_Edward_Hopper.jpg"},
    {"slug": "sun-in-an-empty-room", "title": "Sun in an Empty Room", "year": 1963,
     "commons": "Edward_Hopper_Sun_in_an_Empty_Room.jpg"},
    {"slug": "hotel-by-a-railroad", "title": "Hotel by a Railroad", "year": 1952,
     "commons": "Edward_Hopper_Hotel_by_a_Railroad.jpg"},
    {"slug": "western-motel", "title": "Western Motel", "year": 1957,
     "commons": "Western_Motel_Edward_Hopper_1957.jpg"},
    {"slug": "second-story-sunlight", "title": "Second Story Sunlight", "year": 1960,
     "commons": "Edward_Hopper_Second_Story_Sunlight_1960.jpg"},
    {"slug": "house-by-the-railroad", "title": "House by the Railroad", "year": 1925,
     "commons": "Edward_Hopper_House_by_the_Railroad.jpg"},
    {"slug": "night-windows", "title": "Night Windows", "year": 1928,
     "commons": "Edward_Hopper_Night_Windows_1928.jpg"},
    {"slug": "sunlight-in-a-cafeteria", "title": "Sunlight in a Cafeteria", "year": 1958,
     "commons": "Edward_Hopper_Sunlight_in_a_Cafeteria.jpg"},
    {"slug": "people-in-the-sun", "title": "People in the Sun", "year": 1960,
     "commons": "People_in_the_Sun_Edward_Hopper_1960.jpg"},
    {"slug": "conference-at-night", "title": "Conference at Night", "year": 1949,
     "commons": "Edward_Hopper_Conference_At_Night.jpg"},
    {"slug": "summer-evening", "title": "Summer Evening", "year": 1947,
     "commons": "Edward_Hopper_Summer_Evening.jpg"},
    {"slug": "hotel-lobby", "title": "Hotel Lobby", "year": 1943,
     "commons": "Edward_Hopper_Hotel_Lobby_1943.jpg"},
    {"slug": "drug-store", "title": "Drug Store", "year": 1927,
     "commons": "Drugstore_-_Edward_Hopper.jpg"},
    {"slug": "cape-cod-morning", "title": "Cape Cod Morning", "year": 1950,
     "commons": "Edward_Hopper_Cape_Cod_Morning.jpg"},
    {"slug": "pennsylvania-coal-town", "title": "Pennsylvania Coal Town", "year": 1947,
     "commons": "Edward_Hopper_Pennsylvania_Coal_Town.jpg"},
    {"slug": "rooms-by-the-sea", "title": "Rooms by the Sea", "year": 1951,
     "commons": "Edward_Hopper_Rooms_by_the_sea.jpg"},
    {"slug": "compartment-c-car", "title": "Compartment C, Car 293", "year": 1938,
     "commons": "Edward_Hopper_Compartment_C_Car.jpg"},
    {"slug": "summertime", "title": "Summertime", "year": 1943,
     "commons": "Edward_Hopper_Summertime_(1943).jpg"},
    {"slug": "office-in-a-small-city", "title": "Office in a Small City", "year": 1953,
     "commons": "Edward_Hopper_Office_in_a_Small_City.jpg"},
    {"slug": "ground-swell", "title": "Ground Swell", "year": 1939,
     "commons": "Ground_Swell_Edward_Hopper_1939.jpg"},
    {"slug": "lighthouse-at-two-lights", "title": "Lighthouse at Two Lights", "year": 1929,
     "commons": "Edward_Hopper_Lighthouse_at_Two_Lights.jpg"},
    {"slug": "the-long-leg", "title": "The Long Leg", "year": 1935,
     "commons": "Edward_Hopper_The_Long_Leg.jpg"},
    {"slug": "eleven-a-m", "title": "Eleven A.M.", "year": 1926,
     "commons": "Eleven_AM_Edward_Hopper_1926.jpg"},
    {"slug": "girlie-show", "title": "Girlie Show", "year": 1941,
     "commons": "Edward_Hopper_Girlie_Show_1941.jpg"},
    {"slug": "morning-in-a-city", "title": "Morning in a City", "year": 1944,
     "commons": "Edward_Hopper_Morning_in_a_City.jpg"},
    {"slug": "east-wind-over-weehawken", "title": "East Wind Over Weehawken", "year": 1934,
     "commons": "East_Wind_Over_Weehawken_Edward_Hopper_1934.jpg"},
]


def get_commons_image_url(filename: str, max_width: int = 1200) -> str | None:
    """Get a sized thumbnail URL from Wikimedia Commons API.

    Using iiurlwidth requests a server-side thumbnail, which is much
    less likely to trigger 429 rate limits than fetching originals.
    """
    params = {
        "action": "query",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "url|size",
        "iiurlwidth": str(max_width),
        "format": "json",
    }
    try:
        resp = SESSION.get(COMMONS_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = page.get("imageinfo", [{}])
            if imageinfo:
                info = imageinfo[0]
                # Prefer thumburl (pre-scaled), fall back to original url
                return info.get("thumburl") or info.get("url")
    except Exception as e:
        print(f"    API error: {e}")
    return None


def search_commons(title: str) -> str | None:
    """Search Wikimedia Commons for a Hopper painting by title."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"Edward Hopper {title}",
        "srnamespace": "6",  # File namespace
        "srlimit": "5",
        "format": "json",
    }
    try:
        resp = SESSION.get(COMMONS_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        for result in results:
            page_title = result.get("title", "")
            if page_title.startswith("File:") and page_title.lower().endswith((".jpg", ".jpeg", ".png")):
                return page_title.replace("File:", "")
    except Exception as e:
        print(f"    Search error: {e}")
    return None


def download_painting(painting: dict) -> bool:
    """Download a single painting, trying direct filename then search fallback."""
    slug = painting["slug"]
    title = painting["title"]
    commons_filename = painting.get("commons", "")
    filepath = IMAGES_DIR / f"{slug}.jpg"

    if filepath.exists():
        print(f"  ✓ Already exists: {slug}.jpg")
        return True

    # Try direct filename first
    url = get_commons_image_url(commons_filename)

    # Fallback: search Commons
    if not url:
        print(f"    Direct lookup failed, searching...")
        found_filename = search_commons(title)
        if found_filename:
            url = get_commons_image_url(found_filename)

    if not url:
        print(f"  ✗ Not found on Commons: {title}")
        return False

    # Download with retry on 429
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt + 1}/3)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type:
                print(f"  ✗ Not an image ({content_type}): {title}")
                return False

            filepath.write_bytes(resp.content)
            size_kb = len(resp.content) // 1024
            print(f"  ✓ Downloaded: {title} ({size_kb}KB) -> {slug}.jpg")
            return True

        except Exception as e:
            print(f"  ✗ Download failed: {title} — {e}")
            return False

    print(f"  ✗ Rate limited after 3 retries: {title}")
    return False


def main():
    print(f"Downloading {len(HOPPER_PAINTINGS)} Edward Hopper paintings from Wikimedia Commons...")
    print(f"Output directory: {IMAGES_DIR}\n")

    success = 0
    for painting in HOPPER_PAINTINGS:
        ok = download_painting(painting)
        if ok:
            success += 1
        time.sleep(2)  # Be polite to Commons API

    print(f"\nDone: {success}/{len(HOPPER_PAINTINGS)} downloaded")

    # Save metadata
    metadata_path = DATA_DIR / "paintings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(HOPPER_PAINTINGS, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
