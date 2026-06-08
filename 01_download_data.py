"""
Stage 1: Data Download
Study area : Bakirkoy, Istanbul
Target date: July 2023 (peak-summer LST)
CRS        : EPSG:32637 (UTM Zone 37N)
"""

import os
import sys
import requests
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Directory structure ──────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR   = BASE_DIR / "outputs"

for sub in ["lst", "ndvi", "landcover", "population", "buildings"]:
    (RAW_DIR / sub).mkdir(parents=True, exist_ok=True)
(PROC_DIR).mkdir(parents=True, exist_ok=True)
(OUT_DIR / "maps").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

# ─── Study-area parameters ────────────────────────────────────────────────────
STUDY_AREA  = "Bakirkoy, Istanbul, Turkey"
BBOX        = {               # WGS84
    "west"  : 28.82,
    "south" : 40.96,
    "east"  : 28.92,
    "north" : 41.02,
}
DATE_START  = "2023-07-01"
DATE_END    = "2023-07-31"
TARGET_CRS  = "EPSG:32637"   # UTM 37N
TARGET_RES  = 100            # metres — common-grid resolution

print("=" * 60)
print("Data Download")
print(f"Study area : {STUDY_AREA}")
print(f"Date range : {DATE_START} -> {DATE_END}")
print(f"Target CRS : {TARGET_CRS}  |  Resolution: {TARGET_RES}m")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 1. OSM building footprints (osmnx — no authentication required)
# ══════════════════════════════════════════════════════════════════════════════
def download_osm_buildings():
    print("\n[1/4] Downloading OSM building footprints...")
    import osmnx as ox

    tags = {"building": True}
    gdf  = ox.features_from_place(STUDY_AREA, tags=tags)

    # Keep polygon geometries only
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # Keep relevant columns
    keep = [c for c in ["geometry", "building", "height", "building:levels"]
            if c in gdf.columns]
    gdf = gdf[keep]

    # If height is missing, estimate from levels (x3 m)
    if "building:levels" in gdf.columns and "height" not in gdf.columns:
        gdf["height"] = (
            gdf["building:levels"]
            .fillna("1")
            .apply(lambda x: float(str(x).split(";")[0]) if str(x).replace(".", "").isdigit() else 3.0)
            * 3.0
        )

    gdf = gdf.to_crs(TARGET_CRS)
    out  = RAW_DIR / "buildings" / "bakirkoy_buildings.gpkg"
    gdf.to_file(out, driver="GPKG")
    print(f"  OK  {len(gdf):,} buildings saved  ->  {out.relative_to(BASE_DIR)}")
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# 2. Google Earth Engine — Landsat 8 LST + NDVI
#    Requires: `earthengine-api` + `geemap`
#    On first run: ee.Authenticate() opens a browser.
# ══════════════════════════════════════════════════════════════════════════════
def download_lst_ndvi_gee():
    print("\n[2/4] Landsat 8 LST + NDVI (Google Earth Engine)...")
    import ee
    import geemap

    # ── Authentication ────────────────────────────────────────────────────────
    try:
        ee.Initialize(project="bakirkoy-uhi")   # GEE Cloud project name
    except Exception:
        print("  Starting GEE authentication (a browser will open)...")
        ee.Authenticate()
        ee.Initialize()

    aoi = ee.Geometry.BBox(
        BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]
    )

    # ── Landsat 8 collection ────────────────────────────────────────────────
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .sort("CLOUD_COVER")
    )

    n = col.size().getInfo()
    print(f"  July 2023 image count (cloud<20%): {n}")
    if n == 0:
        print("  ! No suitable image found — raising cloud threshold to 40%...")
        col = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(aoi)
            .filterDate(DATE_START, DATE_END)
            .sort("CLOUD_COVER")
        )

    img = col.first()

    # ── LST (band ST_B10 -> Kelvin -> Celsius) ──────────────────────────────
    lst = (
        img.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST_C")
    )

    # ── NDVI (SR_B5=NIR, SR_B4=Red) ─────────────────────────────────────────
    scale_factor = lambda b: img.select(b).multiply(0.0000275).add(-0.2)
    nir  = scale_factor("SR_B5")
    red  = scale_factor("SR_B4")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    result = lst.addBands(ndvi).clip(aoi)

    # ── Direct local download (geemap) ──────────────────────────────────────
    out_lst  = str(RAW_DIR / "lst"  / "bakirkoy_LST_july2023.tif")
    out_ndvi = str(RAW_DIR / "ndvi" / "bakirkoy_NDVI_july2023.tif")

    print("  Downloading LST...")
    geemap.ee_export_image(
        lst, filename=out_lst, scale=30,
        region=aoi, crs=TARGET_CRS, file_per_band=False
    )
    print(f"  OK  LST  ->  {Path(out_lst).relative_to(BASE_DIR)}")

    print("  Downloading NDVI...")
    geemap.ee_export_image(
        ndvi, filename=out_ndvi, scale=30,
        region=aoi, crs=TARGET_CRS, file_per_band=False
    )
    print(f"  OK  NDVI ->  {Path(out_ndvi).relative_to(BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ESA WorldCover 2021 — land cover (10 m, via GEE)
# ══════════════════════════════════════════════════════════════════════════════
def download_worldcover_gee():
    print("\n[3/4] ESA WorldCover 2021 (GEE)...")
    import ee
    import geemap

    aoi = ee.Geometry.BBox(
        BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]
    )

    wc  = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
    out = str(RAW_DIR / "landcover" / "bakirkoy_worldcover2021.tif")

    geemap.ee_export_image(
        wc, filename=out, scale=10,
        region=aoi, crs=TARGET_CRS, file_per_band=False
    )
    print(f"  OK  WorldCover  ->  {Path(out).relative_to(BASE_DIR)}")

    # Class descriptions (reference)
    classes = {
        10: "Tree cover", 20: "Shrubland", 30: "Grassland",
        40: "Cropland",   50: "Built-up",  60: "Bare/sparse veg.",
        70: "Snow/ice",   80: "Water",     90: "Herbaceous wetland",
        95: "Mangroves",  100: "Moss/lichen"
    }
    print("  WorldCover classes:", classes)


# ══════════════════════════════════════════════════════════════════════════════
# 4. WorldPop population density — Turkey 2020, 100 m
#    Source: data.worldpop.org (licence: CC BY 4.0)
# ══════════════════════════════════════════════════════════════════════════════
def download_worldpop():
    print("\n[4/4] Downloading WorldPop population density...")

    url     = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/TUR/tur_ppp_2020_UNadj.tif"
    out_raw = RAW_DIR / "population" / "turkey_pop_2020_100m.tif"
    out_clp = RAW_DIR / "population" / "bakirkoy_pop_2020_100m.tif"

    # ── Download the raw Turkey file ────────────────────────────────────────
    if not out_raw.exists():
        print(f"  Downloading Turkey population tif (~200 MB)...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(out_raw, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {pct:5.1f}%  ({done/1e6:.1f}/{total/1e6:.0f} MB)", end="", flush=True)
        print(f"\n  OK  Raw data saved  ->  {out_raw.relative_to(BASE_DIR)}")
    else:
        print(f"  OK  Raw data already present: {out_raw.name}")

    # ── Clip to the Bakirkoy bbox ───────────────────────────────────────────
    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import box
    import geopandas as gpd

    clip_geom = gpd.GeoDataFrame(
        geometry=[box(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])],
        crs="EPSG:4326"
    )

    with rasterio.open(out_raw) as src:
        shapes = [geom.__geo_interface__ for geom in clip_geom.geometry]
        out_arr, out_tf = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"height": out_arr.shape[1], "width": out_arr.shape[2],
                         "transform": out_tf})
        with rasterio.open(out_clp, "w", **out_meta) as dst:
            dst.write(out_arr)

    print(f"  OK  Clipped to Bakirkoy  ->  {out_clp.relative_to(BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    steps = [
        ("OSM Buildings",    download_osm_buildings),
        ("Landsat LST+NDVI", download_lst_ndvi_gee),
        ("ESA WorldCover",   download_worldcover_gee),
        ("WorldPop",         download_worldpop),
    ]

    failed = []
    for name, fn in steps:
        try:
            fn()
        except ImportError as e:
            print(f"\n  [ERROR] Missing library: {e}")
            print(f"  -> pip install {str(e).split()[-1]}")
            failed.append(name)
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"Done — {len(steps)-len(failed)}/{len(steps)} steps succeeded")
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All data downloaded successfully!")
    print("Next step: 02_data_pipeline.py")
    print("=" * 60)
