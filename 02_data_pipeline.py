"""
Stage 2: Multi-Source Spatial Data Integration Pipeline
Overview
─────────────────────────────────────────────────────────────────────────────
a) Reprojects + resamples layers of differing resolution/projection onto a
   common grid (EPSG:32637, 100 m).
b) Produces basic statistics + a spatial distribution map for each layer.
c) Computes Moran's I for the target variable (LST) and interprets it.

Output: data/processed/analysis_grid.gpkg  (input for the  model)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import box

import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 110

# ─── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW      = BASE_DIR / "data" / "raw"
PROC     = BASE_DIR / "data" / "processed"
FIG      = BASE_DIR / "outputs" / "figures"
PROC.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:32637"
RES        = 100          # common grid resolution (metres)

LST_PATH   = RAW / "lst"        / "bakirkoy_LST_july2023.tif"
NDVI_PATH  = RAW / "ndvi"       / "bakirkoy_NDVI_july2023.tif"
WC_PATH    = RAW / "landcover"  / "bakirkoy_worldcover2021.tif"
POP_PATH   = RAW / "population" / "bakirkoy_pop_2020_100m.tif"
BLD_PATH   = RAW / "buildings"  / "bakirkoy_buildings.gpkg"

print("=" * 64)
print("Stage 2: Data Integration Pipeline")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMMON TARGET GRID — uses the LST raster as reference (already EPSG:32637)
# ══════════════════════════════════════════════════════════════════════════════
def build_target_grid(ref_path, res=RES):
    with rasterio.open(ref_path) as src:
        b   = src.bounds
        crs = src.crs
    width  = int(np.ceil((b.right - b.left)   / res))
    height = int(np.ceil((b.top   - b.bottom) / res))
    transform = from_origin(b.left, b.top, res, res)
    print(f"\n[1] Common grid: {width} x {height} cells @ {res}m  |  CRS: {crs}")
    print(f"    Extent: {width*res/1000:.1f} km x {height*res/1000:.1f} km")
    return crs, transform, width, height


def warp_raster(src_path, dst_crs, dst_tf, dst_w, dst_h, resampling, band=1):
    """Reproject + resample one raster band onto the target grid."""
    dst = np.full((dst_h, dst_w), np.nan, dtype="float32")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_tf,        dst_crs=dst_crs,
            src_nodata=src.nodata,       dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def warp_array(arr, src_tf, src_crs, dst_crs, dst_tf, dst_w, dst_h, resampling):
    """Reproject an in-memory array onto the target grid (for built-up fraction)."""
    dst = np.full((dst_h, dst_w), np.nan, dtype="float32")
    reproject(
        source=arr, destination=dst,
        src_transform=src_tf, src_crs=src_crs,
        dst_transform=dst_tf, dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst


CRS, TF, W, H = build_target_grid(LST_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 2. WARP ALL LAYERS ONTO THE COMMON GRID
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Resampling layers onto the common grid...")

# 2.1 LST (Landsat 30m -> 100m, bilinear)
LST = warp_raster(LST_PATH, CRS, TF, W, H, Resampling.bilinear)
print(f"    OK  LST  (land surface temperature, C)")

# 2.2 NDVI (Landsat 30m -> 100m, bilinear)
NDVI = warp_raster(NDVI_PATH, CRS, TF, W, H, Resampling.bilinear)
print(f"    OK  NDVI (vegetation index)")

# 2.3 WorldCover (10m, categorical) -> built-up (class 50) fraction, average
with rasterio.open(WC_PATH) as src:
    wc       = src.read(1)
    builtup  = (wc == 50).astype("float32")     # 1 = built-up
    wc_tf    = src.transform
    wc_crs   = src.crs
BUILTUP = warp_array(builtup, wc_tf, wc_crs, CRS, TF, W, H, Resampling.average)
print(f"    OK  Built-up fraction (WorldCover built-up share)")

# 2.4 Population (WorldPop 100m, WGS84 -> UTM, bilinear)
POP = warp_raster(POP_PATH, CRS, TF, W, H, Resampling.bilinear)
POP = np.where(POP < 0, np.nan, POP)
print(f"    OK  Population density (people/cell)")

# 2.5 Building footprints (vector) -> rasterize at 20m, block-average to 100m
print(f"    Rasterizing buildings...")
bld = gpd.read_file(BLD_PATH).to_crs(CRS)

fine_res = 20
fy, fx   = H * (RES // fine_res), W * (RES // fine_res)
fine_tf  = from_origin(TF.c, TF.f, fine_res, fine_res)
fine     = rasterize(
    ((g, 1) for g in bld.geometry if g is not None and not g.is_empty),
    out_shape=(fy, fx), transform=fine_tf, fill=0, dtype="uint8",
)
k = RES // fine_res
BLD_COV = fine.reshape(H, k, W, k).mean(axis=(1, 3)).astype("float32")  # 0-1 coverage
print(f"    OK  Building coverage ({len(bld):,} buildings)")

# 2.6 Mean building height (if available) -> per-cell centroid aggregation
BLD_H = np.zeros((H, W), dtype="float32")
if "height" in bld.columns and bld["height"].notna().sum() > 0:
    try:
        cents = bld.copy()
        cents["geometry"] = bld.geometry.centroid
        col = (np.floor((cents.geometry.x - TF.c) / RES)).astype(int)
        row = (np.floor((TF.f - cents.geometry.y) / RES)).astype(int)
        h_vals = pd.to_numeric(bld["height"], errors="coerce")
        tmp = pd.DataFrame({"r": row, "c": col, "h": h_vals.values}).dropna()
        tmp = tmp[(tmp.r >= 0) & (tmp.r < H) & (tmp.c >= 0) & (tmp.c < W)]
        agg = tmp.groupby(["r", "c"])["h"].mean()
        for (r, c), v in agg.items():
            BLD_H[r, c] = v
        print(f"    OK  Mean building height (in {(BLD_H > 0).sum()} cells)")
    except Exception as e:
        print(f"    !  Building height skipped: {e}")
else:
    print(f"    !  No building height in OSM -> feature will not be used")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANALYSIS TABLE — flatten valid cells into long format
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Building analysis table...")

# Cells where both LST and NDVI are valid (land / data-covered area)
valid = np.isfinite(LST) & np.isfinite(NDVI)

rows, cols = np.where(valid)
# Cell-centre coordinates (EPSG:32637)
xs = TF.c + (cols + 0.5) * RES
ys = TF.f - (rows + 0.5) * RES

df = pd.DataFrame({
    "row": rows, "col": cols, "x": xs, "y": ys,
    "LST"        : LST[valid],
    "NDVI"       : NDVI[valid],
    "built_up"   : np.nan_to_num(BUILTUP[valid], nan=0.0),
    "bld_cov"    : BLD_COV[valid],
    "pop"        : np.nan_to_num(POP[valid], nan=0.0),
    "bld_height" : BLD_H[valid],
})

# UHI intensity (for interpretation/mapping): LST - rural reference
# (rural reference = mean LST of the greenest 25% of cells)
veg_thr = df["NDVI"].quantile(0.75)
rural_ref = df.loc[df["NDVI"] >= veg_thr, "LST"].mean()
df["UHI"] = df["LST"] - rural_ref
print(f"    Rural reference LST (mean of greenest 25%): {rural_ref:.2f} C")
print(f"    Valid cell count: {len(df):,}")

# Drop bld_height from the feature set if it is zero everywhere
FEATURES = ["NDVI", "built_up", "bld_cov", "pop"]
if (df["bld_height"] > 0).sum() > 20:
    FEATURES.append("bld_height")

# Cell polygons (for choropleth + Moran's I)
def cell_polygon(r, c):
    x0 = TF.c + c * RES
    y1 = TF.f - r * RES
    return box(x0, y1 - RES, x0 + RES, y1)

geoms = [cell_polygon(r, c) for r, c in zip(df["row"], df["col"])]
gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=CRS)

out_grid = PROC / "analysis_grid.gpkg"
gdf.to_file(out_grid, driver="GPKG")
print(f"    OK  Analysis grid saved -> {out_grid.relative_to(BASE_DIR)}")
print(f"    Model features (X): {FEATURES}")
print(f"    Target variable (y): LST")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BASIC STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Basic statistics:")
stat_cols = ["LST", "NDVI", "built_up", "bld_cov", "pop", "UHI"]
stats = gdf[stat_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
stats = stats.round(3)
print(stats.to_string())
stats.to_csv(PROC / "layer_statistics.csv")
print(f"    OK  -> {(PROC / 'layer_statistics.csv').relative_to(BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SPATIAL DISTRIBUTION MAPS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Drawing distribution maps and histograms...")

panels = [
    ("LST",      "LST (C)",                 "inferno"),
    ("NDVI",     "NDVI",                    "RdYlGn"),
    ("built_up", "Built-up Fraction",       "OrRd"),
    ("bld_cov",  "Building Coverage",       "OrRd"),
    ("pop",      "Population (people/cell)","viridis"),
    ("UHI",      "UHI Intensity (C)",       "coolwarm"),
]

# 5.1 Spatial distribution maps (choropleth)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, (col, title, cmap) in zip(axes.ravel(), panels):
    gdf.plot(column=col, cmap=cmap, ax=ax, legend=True,
             legend_kwds={"shrink": 0.6})
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Bakirkoy — Spatial Distribution Maps (100 m grid)",
             fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "01_layer_maps.png", bbox_inches="tight")
plt.close(fig)
print(f"    OK  -> {(FIG / '01_layer_maps.png').relative_to(BASE_DIR)}")

# 5.2 Histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (col, title, _) in zip(axes.ravel(), panels):
    ax.hist(gdf[col].dropna(), bins=40, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Frequency")
fig.suptitle("Layer Histograms", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "01_layer_histograms.png", bbox_inches="tight")
plt.close(fig)
print(f"    OK  -> {(FIG / '01_layer_histograms.png').relative_to(BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MORAN'S I — spatial autocorrelation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Computing Moran's I (target variable: LST)...")
from libpysal.weights import KNN
from esda.moran import Moran

pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gdf["x"], gdf["y"]), crs=CRS)
w = KNN.from_dataframe(pts, k=8)
w.transform = "r"

y = gdf["LST"].values
moran = Moran(y, w, permutations=999)

print("    ---------------------------------------------")
print(f"    Moran's I        : {moran.I:.4f}")
print(f"    Expected I (H0)  : {moran.EI:.4f}")
print(f"    z-score          : {moran.z_sim:.3f}")
print(f"    p-value (sim)    : {moran.p_sim:.4f}")
print("    ---------------------------------------------")

interp = (
    "STRONG POSITIVE spatial autocorrelation. Neighbouring cells have similar LST "
    "values (hot areas cluster). Because the independence-of-errors assumption is "
    "violated, a standard RF reports an over-optimistic (misleading) score without "
    "spatial CV; therefore spatial k-fold CV is used in the pipeline."
    if (moran.I > 0.3 and moran.p_sim < 0.05) else
    "Significant spatial structure; spatial CV is recommended in model design."
)
print("    Interpretation:", interp)

with open(PROC / "moran_result.txt", "w", encoding="utf-8") as f:
    f.write(f"Moran's I (LST) = {moran.I:.4f}\n")
    f.write(f"E[I] = {moran.EI:.4f}\nz = {moran.z_sim:.3f}\np = {moran.p_sim:.4f}\n")
    f.write(f"Weights: KNN k=8, row-standardized\n\nInterpretation:\n{interp}\n")

# Moran scatterplot
try:
    from splot.esda import moran_scatterplot
    fig, ax = moran_scatterplot(moran, aspect_equal=True)
    ax.set_xlabel("LST (standardized)")
    ax.set_ylabel("Spatial lag — neighbour LST (standardized)")
    ax.set_title(f"Moran Scatterplot  |  I = {moran.I:.3f}, p = {moran.p_sim:.3f}",
                 fontweight="bold")
    fig.savefig(FIG / "02_moran_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    OK  -> {(FIG / '02_moran_scatter.png').relative_to(BASE_DIR)}")
except Exception as e:
    print(f"    !  Moran scatterplot skipped: {e}")


print("\n" + "=" * 64)
print("Stage 2 complete!")
print(f"  - Analysis grid : data/processed/analysis_grid.gpkg ({len(gdf):,} cells)")
print(f"  - Statistics    : data/processed/layer_statistics.csv")
print(f"  - Moran's I     : data/processed/moran_result.txt")
print(f"  - Figures       : outputs/figures/01_*, 02_*")
print("Next step: 03_model.py ( — GWRF / RF + spatial CV + SHAP)")
print("=" * 64)
