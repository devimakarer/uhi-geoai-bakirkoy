"""
Stage 5: Interactive Web Map
─────────────────────────────────────────────────────────────────────────────
Interactive Folium map. At least 3 layers (required):
  1. UHI intensity layer (choropleth)
  2. Model prediction errors (residual map)
  3. At least one feature-importance layer derived from SHAP values

Output: outputs/maps/uhi_interactive_map.html
"""

import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import branca.colormap as bcm

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
PROC = BASE_DIR / "data" / "processed"
MAPS = BASE_DIR / "outputs" / "maps"
MAPS.mkdir(parents=True, exist_ok=True)

print("=" * 64)
print("Stage 5: Interactive Web Map")
print("=" * 64)

# ─── Data (reproject to WGS84 for Folium) ─────────────────────────────────────
gdf = gpd.read_file(PROC / "model_predictions.gpkg").to_crs(4326)
print(f"\nCell count: {len(gdf):,}")

# Choose the most important feature for the SHAP layer (highest mean |SHAP|)
shap_cols = [c for c in gdf.columns if c.startswith("shap_") and c != "shap_driver"]
top_shap = max(shap_cols, key=lambda c: gdf[c].abs().mean())
top_feat = top_shap.replace("shap_", "")
print(f"Most influential feature (SHAP): {top_feat}")

# Round numeric fields for tooltips
for c in ["LST", "UHI", "pred", "residual", "NDVI", top_shap]:
    if c in gdf.columns:
        gdf[c] = gdf[c].round(2)

center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron",
               control_scale=True)


# ─── Helper: add a choropleth layer ───────────────────────────────────────────
def add_choropleth(col, name, palette, caption, show=False, diverging=False):
    vals = gdf[col].dropna()
    if diverging:
        v = max(abs(vals.quantile(0.02)), abs(vals.quantile(0.98)))
        vmin, vmax = -v, v
    else:
        vmin, vmax = vals.quantile(0.02), vals.quantile(0.98)

    cmap = bcm.LinearColormap(palette, vmin=vmin, vmax=vmax, caption=caption)

    # Tooltip: fixed fields + this layer's variable (if not already listed)
    tip_pairs = [("LST", "LST (C)"), ("UHI", "UHI (C)"),
                 ("pred", "Prediction (C)"), ("residual", "Residual (C)"),
                 ("NDVI", "NDVI")]
    if col not in [p[0] for p in tip_pairs]:
        tip_pairs.append((col, caption))
    fields  = [f for f, _ in tip_pairs if f in gdf.columns]
    aliases = [a for f, a in tip_pairs if f in gdf.columns]

    fg = folium.FeatureGroup(name=name, show=show)
    folium.GeoJson(
        gdf,
        style_function=lambda feat, cmap=cmap, col=col: {
            "fillColor": cmap(feat["properties"][col])
                         if feat["properties"][col] is not None else "#00000000",
            "color": "transparent", "weight": 0, "fillOpacity": 0.75,
        },
        tooltip=GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
    ).add_to(fg)
    fg.add_to(m)
    cmap.add_to(m)
    print(f"  OK  Layer: {name}")


# ── 1) UHI intensity (required) ───────────────────────────────────────────────
add_choropleth(
    "UHI", "1 - UHI Intensity (C)",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
    "UHI Intensity (C)", show=True, diverging=True,
)

# ── 2) Model residuals (required) ─────────────────────────────────────────────
add_choropleth(
    "residual", "2 - Model Residuals (Actual - Predicted, C)",
    ["#762a83", "#af8dc3", "#f7f7f7", "#7fbf7b", "#1b7837"],
    "Residual (C)", show=False, diverging=True,
)

# ── 3) SHAP feature-importance layer (required) ───────────────────────────────
add_choropleth(
    top_shap, f"3 - SHAP Contribution — {top_feat} (C)",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
    f"{top_feat} -> LST contribution (C)", show=False, diverging=True,
)

# ── Layer control + title ─────────────────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)

title_html = """
<div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
            background: white; padding: 8px 14px; border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,.3); font-family: sans-serif;">
  <b>Bakirkoy — Urban Heat Island (UHI) GeoAI Analysis</b><br>
  <span style="font-size: 12px; color:#555;">
  Layers: UHI intensity &middot; Model residuals &middot; SHAP feature contribution
  </span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

out = MAPS / "uhi_interactive_map.html"
m.save(str(out))
print(f"\nOK  Interactive map saved -> {out.relative_to(BASE_DIR)}")

# ── Static PNG versions (result maps for documentation) ──────────────────────
import matplotlib.pyplot as plt
FIG = BASE_DIR / "outputs" / "figures"

static_panels = [
    ("UHI",      "UHI Intensity (C)",                 "coolwarm"),
    ("residual", "Model Residuals (Actual - Pred, C)", "PRGn"),
    (top_shap,   f"SHAP Contribution: {top_feat} (C)", "coolwarm"),
]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (col, title, cmap) in zip(axes, static_panels):
    vmax = max(abs(gdf[col].quantile(0.02)), abs(gdf[col].quantile(0.98)))
    gdf.plot(column=col, cmap=cmap, ax=ax, legend=True,
             vmin=-vmax, vmax=vmax, legend_kwds={"shrink": 0.6})
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Bakirkoy UHI — Result Maps", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "04_result_maps.png", bbox_inches="tight", dpi=110)
plt.close(fig)
print(f"OK  Static result maps -> outputs/figures/04_result_maps.png")

print("\n" + "=" * 64)
print("Stage 5 complete!  ALL the pipeline CODE DONE.")
print(f"  - Interactive map: outputs/maps/uhi_interactive_map.html")
print("    (open in a browser, toggle layers from the top-right control)")
print("=" * 64)
