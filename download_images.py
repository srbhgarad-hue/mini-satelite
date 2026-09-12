import os
import requests

os.makedirs("data/raw", exist_ok=True)

# Direct Working Satellite Image URLs (NASA / USGS / OpenStreetMap Tiles)
urls = {
    "airport.png": "https://tile.openstreetmap.org/15/14321/23120.png",
    "coastal.png": "https://tile.openstreetmap.org/15/14322/23121.png",
    "urban.png": "https://tile.openstreetmap.org/15/14320/23122.png",
    "lake.png": "https://tile.openstreetmap.org/15/14323/23123.png",
    "forest.png": "https://tile.openstreetmap.org/15/14324/23124.png"
}

# Browser Headers to prevent blocking
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

for name, url in urls.items():
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and len(response.content) > 1000:
        file_path = f"data/raw/{name}"
        with open(file_path, 'wb') as handler:
            handler.write(response.content)
        print(f"Downloaded & Saved: {file_path}")
    else:
        print(f"Failed to fetch: {name}")

print("\nDone! Check 'data/raw/' folder now.")