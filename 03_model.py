"""
Stage 3: Model
─────────────────────────────────────────────────────────────────────────────
Target: LST regression as a UHI-intensity proxy (Random Forest baseline).

a) Spatially-aware training: compares a standard RF under RANDOM k-fold CV vs
   SPATIAL (block) k-fold CV. Because Moran's I is high, random CV is optimistic
   while spatial CV reflects true generalisation.
b) Comparison with at least two metrics (RMSE, MAE, R2).
c) SHAP values to visualise which spatial variables drive UHI.

Output: data/processed/model_predictions.gpkg  (for  residual + SHAP map)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import shap

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 110

BASE_DIR = Path(__file__).parent
PROC = BASE_DIR / "data" / "processed"
FIG  = BASE_DIR / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print("=" * 64)
print("Stage 3: Model")
print("=" * 64)

# ─── Data ─────────────────────────────────────────────────────────────────────
gdf = gpd.read_file(PROC / "analysis_grid.gpkg")

candidate = ["NDVI", "built_up", "bld_cov", "pop", "bld_height"]
FEATURES  = [c for c in candidate if c in gdf.columns and gdf[c].notna().any()]
TARGET    = "LST"

X = gdf[FEATURES].fillna(0).values
y = gdf[TARGET].values
coords = gdf[["x", "y"]].values

print(f"\nSample count : {len(gdf):,}")
print(f"Features     : {FEATURES}")
print(f"Target       : {TARGET} (UHI intensity proxy)")


# ─── Helper: metrics ──────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


def rf():
    return RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=3,
        n_jobs=-1, random_state=42,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. RANDOM K-FOLD CV  (ignores spatial structure)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Random 5-fold CV...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rand = np.zeros_like(y, dtype=float)
rand_scores = []
for tr, te in kf.split(X):
    m = rf().fit(X[tr], y[tr])
    p = m.predict(X[te])
    oof_rand[te] = p
    rand_scores.append(metrics(y[te], p))
rand_mean = np.mean(rand_scores, axis=0)
print(f"    RMSE={rand_mean[0]:.3f}  MAE={rand_mean[1]:.3f}  R2={rand_mean[2]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPATIAL (BLOCK) K-FOLD CV — KMeans-based spatial blocks over coordinates
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Spatial (block) 5-fold CV...")
n_blocks = 5
blocks = KMeans(n_clusters=n_blocks, random_state=42, n_init=10).fit_predict(coords)
oof_spat = np.zeros_like(y, dtype=float)
spat_scores = []
for b in range(n_blocks):
    te = blocks == b
    tr = ~te
    m = rf().fit(X[tr], y[tr])
    p = m.predict(X[te])
    oof_spat[te] = p
    spat_scores.append(metrics(y[te], p))
spat_mean = np.mean(spat_scores, axis=0)
print(f"    RMSE={spat_mean[0]:.3f}  MAE={spat_mean[1]:.3f}  R2={spat_mean[2]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
comp = pd.DataFrame(
    {"Random CV": rand_mean, "Spatial CV": spat_mean},
    index=["RMSE (C)", "MAE (C)", "R2"],
).round(3)
print("\n[3] Comparison:")
print(comp.to_string())
comp.to_csv(PROC / "cv_comparison.csv")

gap = spat_mean[0] - rand_mean[0]
print(f"\n    -> Spatial CV RMSE is {gap:.2f} C higher than random CV.")
print(f"      Random CV is MISLEADINGLY good due to spatial leakage (neighbouring")
print(f"      cells appear in both train and test). Consistent with Moran's I=0.99:")
print(f"      spatial CV reflects true generalisation performance.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FINAL MODEL + SHAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Training final model and computing SHAP...")
final = rf().fit(X, y)

explainer = shap.TreeExplainer(final)
shap_vals = explainer.shap_values(X)        # (n_samples, n_features)

# SHAP summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_vals, X, feature_names=FEATURES, show=False)
plt.title("SHAP — Feature Impact on LST", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "03_shap_summary.png", bbox_inches="tight")
plt.close()
print(f"    OK  -> outputs/figures/03_shap_summary.png")

# SHAP mean absolute importance (bar)
imp = np.abs(shap_vals).mean(axis=0)
order = np.argsort(imp)[::-1]
plt.figure(figsize=(7, 4))
plt.barh([FEATURES[i] for i in order][::-1], imp[order][::-1], color="teal")
plt.xlabel("Mean |SHAP| (C)")
plt.title("Feature Importance (SHAP)", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "03_shap_importance.png", bbox_inches="tight")
plt.close()
print(f"    OK  -> outputs/figures/03_shap_importance.png")

print("\n    Feature importance ranking (mean |SHAP|):")
for i in order:
    print(f"      {FEATURES[i]:<12} {imp[i]:.3f} C")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE PREDICTION + RESIDUAL + SHAP LAYERS  (for )
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Saving prediction / residual / SHAP layers...")
gdf["pred"]     = oof_spat                  # spatial-CV out-of-fold prediction
gdf["residual"] = y - oof_spat              # actual - predicted
for j, f in enumerate(FEATURES):
    gdf[f"shap_{f}"] = shap_vals[:, j]

# Dominant driver per cell (highest |SHAP|)
dominant = np.array(FEATURES)[np.argmax(np.abs(shap_vals), axis=1)]
gdf["shap_driver"] = dominant

out = PROC / "model_predictions.gpkg"
gdf.to_file(out, driver="GPKG")
print(f"    OK  -> {out.relative_to(BASE_DIR)}")
print(f"      Added columns: pred, residual, shap_<feature>, shap_driver")

print("\n" + "=" * 64)
print("Stage 3 complete!")
print(f"  - CV comparison        : data/processed/cv_comparison.csv")
print(f"  - Prediction+residual+SHAP: data/processed/model_predictions.gpkg")
print(f"  - SHAP figures         : outputs/figures/03_*")
print("Next step: 04_tool_calling.py and 05_web_map.py")
print("=" * 64)
