import os
import requests

os.makedirs("data/raw", exist_ok=True)

# High-resolution sample satellite/aerial imagery tile endpoints
urls = {
    "airport.png": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/16/28224/46782",
    "harbor.png": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/16/28230/46790",
    "stadium.png": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/16/28228/46785",
    "urban.png": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/16/28225/46784",
    "forest.png": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/16/28210/46770"
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("[INFO] Fetching sample satellite patches for local archive...")
for name, url in urls.items():
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and len(response.content) > 1000:
            file_path = os.path.join("data", "raw", name)
            with open(file_path, 'wb') as handler:
                handler.write(response.content)
            print(f"[OK] Downloaded & Saved: {file_path}")
        else:
            print(f"[WARN] Failed to fetch: {name} (Status Code: {response.status_code})")
    except Exception as e:
        print(f"[ERROR] Error downloading {name}: {e}")

print("\n[DONE] Sample dataset ready in 'data/raw/' folder.")