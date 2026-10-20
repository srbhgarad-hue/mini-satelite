"""
city_catalog.py
===============
Chhatrapati Sambhajinagar (Aurangabad) Whole-City Geographic Catalog.
Defines major urban expansion hubs, industrial sectors, and geocoding utilities.
"""

# Pre-configured key urban growth sectors across Chhatrapati Sambhajinagar
CITY_HOTSPOTS = {
    "CIDCO Urban Expansion (N-1 to N-12)": {
        "name": "CIDCO Urban Township",
        "description": "Dense residential, commercial complexes, and modern townships developed northeast of city center.",
        "lat": 19.8856,
        "lon": 75.3689,
        "zoom": 16,
        "category": "Residential & Commercial",
        "key_event": "Rapid apartment block and commercial plaza construction (2016-2026)"
    },
    "Beed Bypass Highway Corridor": {
        "name": "Beed Bypass Southern Corridor",
        "description": "Fastest-growing southern transit corridor with new commercial towers, auto showrooms, and residential layouts.",
        "lat": 19.8452,
        "lon": 75.3385,
        "zoom": 16,
        "category": "High-Density Commercial & Highway",
        "key_event": "Major roadside commercial towers and residential complexes replacing open ground"
    },
    "Waluj MIDC Industrial Sector": {
        "name": "Waluj MIDC Expansion",
        "description": "Major manufacturing hub on the western fringe housing auto manufacturing and large warehouse clusters.",
        "lat": 19.8320,
        "lon": 75.2410,
        "zoom": 16,
        "category": "Heavy Industrial & Logistics",
        "key_event": "Large industrial sheds, manufacturing plants, and logistics yards constructed"
    },
    "Chikalthana / Airport Road Corridor": {
        "name": "Chikalthana Commercial Belt",
        "description": "Strategic eastern corridor surrounding the Chhatrapati Sambhajinagar Airport with hotels and tech offices.",
        "lat": 19.8660,
        "lon": 75.3980,
        "zoom": 16,
        "category": "Airport & Commercial",
        "key_event": "Commercial developments and institutional campus expansions"
    },
    "Shendra MIDC / DMIC Industrial Corridor": {
        "name": "Shendra DMIC Smart City",
        "description": "Flagship industrial node of the Delhi-Mumbai Industrial Corridor on the far eastern bypass.",
        "lat": 19.8685,
        "lon": 75.4920,
        "zoom": 16,
        "category": "Smart Industrial & Infrastructure",
        "key_event": "Massive conversion of agricultural land into mega-industrial plots and road grids"
    },
    "Kranti Chowk (Dense Core City Center)": {
        "name": "Kranti Chowk Urban Core",
        "description": "Historic and high-density central junction of Chhatrapati Sambhajinagar.",
        "lat": 19.8735,
        "lon": 75.3265,
        "zoom": 17,
        "category": "Core Urban Redevelopment",
        "key_event": "Flyover construction and commercial vertical redevelopment"
    },
    "Railway Station & Station Road Zone": {
        "name": "Railway Station Transit Hub",
        "description": "South-central transport node with hospitality, transit warehousing, and urban renewal.",
        "lat": 19.8630,
        "lon": 75.3120,
        "zoom": 16,
        "category": "Transit Hub & Commercial",
        "key_event": "Station modernization and hotel/commercial construction"
    }
}

# Municipal City Boundary Bounding Box for Chhatrapati Sambhajinagar
# [min_lon, min_lat, max_lon, max_lat]
CITY_BOUNDS_BBOX = [75.2000, 19.8000, 75.5200, 19.9200]
CITY_CENTER_LAT, CITY_CENTER_LON = 19.8762, 75.3433

def get_hotspot_info(name):
    """Retrieves hotspot metadata with fallback to city center."""
    return CITY_HOTSPOTS.get(name, {
        "name": name,
        "description": "Custom geographic point in Chhatrapati Sambhajinagar.",
        "lat": CITY_CENTER_LAT,
        "lon": CITY_CENTER_LON,
        "zoom": 16,
        "category": "Custom AOI",
        "key_event": "Urban surface transition analysis"
    })
