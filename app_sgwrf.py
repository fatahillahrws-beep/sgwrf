# -*- coding: utf-8 -*-
"""
SGWRF Interactive Dashboard
Adaptasi dari SGWRF2.ipynb menjadi aplikasi Streamlit.

Fitur utama:
- Upload Excel/CSV; tidak ada pembacaan dataset yang di-hard-code.
- Mapping kolom interaktif, sehingga nama kolom dapat berbeda selama maknanya sesuai.
- SGWRF: adaptive Gaussian geographic + attribute weights, LOOCV bandwidth,
  RF hyperparameter tuning, local RF, permutation importance.
- Semua output utama dari notebook asli: preprocessing, bandwidth CV,
  RF tuning, local results, local diagnostics, local Ridge equations,
  maps, baseline models, output CSV.
- Dashboard interaktif: Plotly + Folium, hover, zoom, layer control,
  pemilihan variabel, filter lokasi, download hasil.
- Boundary otomatis menyesuaikan extent data. Boundary GeoJSON/SHP/ZIP dapat
  di-upload; jika tidak tersedia aplikasi mencoba BIG sebagai fallback.

Catatan metodologis:
Model utama tetap Random Forest lokal. Persamaan Ridge yang ditampilkan adalah
DIAGNOSTIK tambahan, bukan koefisien SGWRF.
"""

import io
import re
import time
import zipfile
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st

from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Optional mapping stack
try:
    import geopandas as gpd
    import folium
    from folium.plugins import Fullscreen, MousePosition, MiniMap
    from streamlit_folium import st_folium
    MAP_OK = True
except Exception:
    MAP_OK = False

# Optional shapely validation
try:
    from shapely.validation import make_valid
except Exception:
    make_valid = None


# ================================================================
# PAGE / STYLE
# ================================================================
st.set_page_config(
    page_title="SGWRF Spatial Analytics Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background: #f5f7fb; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #102a43 0%, #1f4e79 55%, #2f80ed 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 30px rgba(16,42,67,.15);
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .35rem 0 0; opacity: .9; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e7ebf2; padding: 10px;
        border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.03);
    }
    .section-card {
        background: white; padding: 1rem 1.15rem; border-radius: 14px;
        border: 1px solid #e7ebf2; margin-bottom: .8rem;
    }
    .small-note { color:#5f6b7a; font-size:.86rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🗺️ SGWRF Spatial Analytics Dashboard</h1>
  <p>Spatially Geographically Weighted Random Forest — analisis lokal, bandwidth adaptif,
  variable importance, diagnostik, baseline comparison, dan peta interaktif.</p>
</div>
""", unsafe_allow_html=True)


# ================================================================
# DEFAULT CONFIG — mengikuti notebook asli
# ================================================================
DEFAULT_X_COLS = [
    "temperature_c", "relative_humidity", "pressure_hpa", "wind_speed_ms",
    "wind_direction_deg", "precipitation_mm", "elevation_m", "landcover",
    "road_density_km_km2", "population_density",
]
DEFAULT_LABELS = {
    "temperature_c": "Suhu",
    "relative_humidity": "Kelembapan",
    "pressure_hpa": "Tekanan",
    "wind_speed_ms": "Kecepatan angin",
    "wind_direction_deg": "Arah angin",
    "precipitation_mm": "Curah hujan",
    "elevation_m": "Elevasi",
    "landcover": "Tutupan lahan",
    "road_density_km_km2": "Kepadatan jalan",
    "population_density": "Kepadatan penduduk",
}
BIG_QUERY_URL = (
    "https://geoservices.big.go.id/gis/rest/services/"
    "DISIGT/BatasWilayah/MapServer/0/query"
)


# ================================================================
# BASIC HELPERS
# ================================================================
def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")


def infer_column(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in columns:
        lc = str(c).lower()
        if any(cand.lower() in lc or lc in cand.lower() for cand in candidates):
            return c
    return None


def read_uploaded_data(uploaded):
    if uploaded is None:
        return None
    raw = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        # Try common separators/encodings.
        try:
            return pd.read_csv(io.BytesIO(raw))
        except Exception:
            return pd.read_csv(io.BytesIO(raw), sep=";")
    return pd.read_excel(io.BytesIO(raw), sheet_name=st.session_state.get("sheet_name", 0))


def validate_and_prepare(df, id_col, name_col, lat_col, lon_col, y_col, x_cols):
    required = [id_col, name_col, lat_col, lon_col, y_col] + x_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Kolom belum lengkap: " + ", ".join(map(str, missing)))

    keep = required.copy()
    out = df[keep].copy()
    numeric_cols = [lat_col, lon_col, y_col] + x_cols
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    before = len(out)
    out = out.dropna(subset=keep).reset_index(drop=True)
    if len(out) < 8:
        raise ValueError("Minimal 8 observasi valid diperlukan untuk analisis lokal.")
    return out, before, len(out)


def haversine_km(latlon):
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def adaptive_bandwidth_matrix(distance_matrix, k):
    d = np.array(distance_matrix, copy=True)
    np.fill_diagonal(d, np.inf)
    kth = np.partition(d, kth=k - 1, axis=1)[:, k - 1]
    return np.maximum(kth, 1e-8)


def combined_weights(d_geo, d_attr, k_geo, k_attr):
    bg = adaptive_bandwidth_matrix(d_geo, k_geo)
    ba = adaptive_bandwidth_matrix(d_attr, k_attr)
    wg = np.exp(-0.5 * (d_geo / bg[:, None]) ** 2)
    wa = np.exp(-0.5 * (d_attr / ba[:, None]) ** 2)
    w = wg * wa
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-15)
    return w, wg, wa, bg, ba


def make_metrics(y, pred):
    err = y - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(y) < 1e-12, np.nan, np.abs(y))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    r2 = float(r2_score(y, pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def make_rf(params, random_state, final=False):
    return RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 500 if final else 80)),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_depth=params.get("max_depth", None),
        random_state=int(random_state),
        n_jobs=-1,
    )


# ================================================================
# SGWRF CORE
# ================================================================
def optimize_bandwidth(X, y, d_geo, d_attr, min_k, max_k, trees_cv,
                       search_mode="grid", n_random=200, seed=2026,
                       leave_target_out=True, progress=None):
    n = len(y)
    ks = list(range(min_k, min(max_k, n - 1) + 1))
    candidates = list(product(ks, ks))
    if search_mode == "random" and len(candidates) > n_random:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), n_random, replace=False)
        candidates = [candidates[i] for i in idx]

    rows = []
    best = None
    t0 = time.time()
    base_params = {"n_estimators": trees_cv, "max_features": "sqrt",
                   "min_samples_leaf": 1, "max_depth": None}

    for no, (kg, ka) in enumerate(candidates, 1):
        W, _, _, _, _ = combined_weights(d_geo, d_attr, kg, ka)
        preds = np.full(n, np.nan)
        for i in range(n):
            w = W[i].copy()
            if leave_target_out:
                w[i] = 0.0
            if np.count_nonzero(w > 1e-8) < 4:
                continue
            model = make_rf(base_params, seed + i)
            model.fit(X, y, sample_weight=w)
            preds[i] = model.predict(X[i:i+1])[0]
        if np.isnan(preds).any():
            cv = np.inf
            rmse = np.inf
        else:
            residual = y - preds
            cv = float(np.sum(residual ** 2))
            rmse = float(np.sqrt(cv / n))
        row = {"iteration": no, "k_geo": kg, "k_attr": ka,
               "CV": cv, "RMSE_CV": rmse}
        rows.append(row)
        if best is None or cv < best["CV"]:
            best = row.copy()
        if progress is not None:
            progress.progress(no / len(candidates), text=f"Bandwidth {no}/{len(candidates)} — k_geo={kg}, k_attr={ka}")

    result = pd.DataFrame(rows).sort_values("CV").reset_index(drop=True)
    return (int(best["k_geo"]), int(best["k_attr"])), result, time.time() - t0


def optimize_rf(X, y, W, trees_cv, leave_target_out=True, seed=2026, progress=None):
    grid = [
        {"n_estimators": 80, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": 120, "max_features": 0.7, "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": 120, "max_features": 1.0, "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": 120, "max_features": "sqrt", "min_samples_leaf": 2, "max_depth": None},
    ]
    rows = []
    best = None
    for j, p in enumerate(grid, 1):
        p = p.copy()
        p["n_estimators"] = min(int(p["n_estimators"]), int(trees_cv))
        preds = np.full(len(y), np.nan)
        for i in range(len(y)):
            w = W[i].copy()
            if leave_target_out:
                w[i] = 0.0
            model = make_rf(p, seed + 1000 * j + i)
            model.fit(X, y, sample_weight=w)
            preds[i] = model.predict(X[i:i+1])[0]
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        rows.append({**p, "RMSE_CV": rmse})
        if best is None or rmse < best["RMSE_CV"]:
            best = rows[-1].copy()
        if progress is not None:
            progress.progress(j / len(grid), text=f"RF tuning {j}/{len(grid)}")
    return best, pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)


def train_local_models(X, y, W, df, x_cols, labels, rf_params,
                       final_trees, perm_repeats, leave_target_out=True,
                       seed=2026, progress=None):
    n, p = X.shape
    preds = np.full(n, np.nan)
    local_r2 = np.full(n, np.nan)
    local_mae = np.full(n, np.nan)
    local_rmse = np.full(n, np.nan)
    importances = np.zeros((n, p))
    models = []

    final_params = {
        "n_estimators": final_trees,
        "max_features": rf_params.get("max_features", "sqrt"),
        "min_samples_leaf": rf_params.get("min_samples_leaf", 1),
        "max_depth": rf_params.get("max_depth", None),
    }

    for i in range(n):
        w = W[i].copy()
        mask = w > 1e-8
        if leave_target_out:
            mask[i] = False
            w[i] = 0.0
        if mask.sum() < 4:
            mask[:] = True
            if leave_target_out:
                mask[i] = False

        model = make_rf(final_params, seed + i, final=True)
        model.fit(X[mask], y[mask], sample_weight=w[mask])
        preds[i] = model.predict(X[i:i+1])[0]
        train_pred = model.predict(X[mask])
        local_rmse[i] = np.sqrt(mean_squared_error(y[mask], train_pred, sample_weight=w[mask]))
        local_mae[i] = mean_absolute_error(y[mask], train_pred, sample_weight=w[mask])
        try:
            local_r2[i] = r2_score(y[mask], train_pred, sample_weight=w[mask])
        except Exception:
            local_r2[i] = np.nan

        try:
            pi = permutation_importance(
                model, X[mask], y[mask], n_repeats=perm_repeats,
                random_state=seed + i, scoring="neg_mean_squared_error", n_jobs=-1
            )
            imp = np.maximum(pi.importances_mean, 0)
            if imp.sum() > 0:
                imp = imp / imp.sum()
            importances[i] = imp
        except Exception:
            importances[i] = model.feature_importances_
            if importances[i].sum() > 0:
                importances[i] /= importances[i].sum()
        models.append(model)
        if progress is not None:
            top = int(np.argmax(importances[i]))
            progress.progress((i + 1) / n, text=f"Model lokal {i+1}/{n} — dominan: {labels[x_cols[top]]}")

    return preds, local_r2, local_mae, local_rmse, importances, models


def local_ridge_diagnostic(X, y, W, x_cols):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    coefs = np.zeros((len(y), Xz.shape[1]))
    intercepts = np.zeros(len(y))
    pred = np.zeros(len(y))
    for i in range(len(y)):
        w = W[i].copy(); w[i] = 0
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xz, y, sample_weight=w)
        coefs[i] = ridge.coef_
        intercepts[i] = ridge.intercept_
        pred[i] = ridge.predict(Xz[i:i+1])[0]
    return intercepts, coefs, pred


def build_results(df, y, pred, local_r2, local_mae, local_rmse,
                  importances, bg, ba, name_col, id_col, lat_col, lon_col, y_col,
                  x_cols, labels):
    out = df[[id_col, name_col, lat_col, lon_col, y_col]].copy()
    out.insert(0, "point_no", np.arange(1, len(out) + 1))
    out["prediction_sgwrf"] = pred
    out["residual"] = y - pred
    out["abs_error"] = np.abs(y - pred)
    out["local_train_R2"] = local_r2
    out["local_train_RMSE"] = local_rmse
    out["local_train_MAE"] = local_mae
    out["bandwidth_geo_k"] = bg
    out["bandwidth_attr_k"] = ba
    for j, c in enumerate(x_cols):
        out[f"VI_{c}"] = importances[:, j]
    top = np.argmax(importances, axis=1)
    out["dominant_variable"] = [labels.get(x_cols[j], x_cols[j]) for j in top]
    out["dominant_variable_raw"] = [x_cols[j] for j in top]
    out["dominant_importance"] = importances[np.arange(len(y)), top]
    return out


# ================================================================
# BASELINES
# ================================================================
def run_baselines(X, y, d_geo, d_attr, best_bw, final_trees, x_cols, seed=2026):
    rows = []
    rf = RandomForestRegressor(n_estimators=final_trees, max_features="sqrt",
                               min_samples_leaf=1, random_state=seed, n_jobs=-1)
    rf.fit(X, y); p = rf.predict(X)
    rows.append({"Model": "RF_global", **make_metrics(y, p)})

    kg, ka = best_bw
    Wg, _, _, _, _ = combined_weights(d_geo, d_geo, kg, kg)
    p_gwr = np.zeros(len(y))
    for i in range(len(y)):
        w = Wg[i].copy(); w[i] = 0
        ridge = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
        p_gwr[i] = ridge.predict(X[i:i+1])[0]
    rows.append({"Model": "GWR_Ridge", **make_metrics(y, p_gwr)})

    p_gwrf = np.zeros(len(y))
    for i in range(len(y)):
        w = Wg[i].copy(); w[i] = 0
        model = RandomForestRegressor(n_estimators=final_trees, max_features="sqrt",
                                      min_samples_leaf=1, random_state=seed+i, n_jobs=-1)
        model.fit(X, y, sample_weight=w)
        p_gwrf[i] = model.predict(X[i:i+1])[0]
    rows.append({"Model": "GWRF", **make_metrics(y, p_gwrf)})

    Wsg, _, _, _, _ = combined_weights(d_geo, d_attr, kg, ka)
    p_sgwr = np.zeros(len(y))
    for i in range(len(y)):
        w = Wsg[i].copy(); w[i] = 0
        ridge = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
        p_sgwr[i] = ridge.predict(X[i:i+1])[0]
    rows.append({"Model": "SGWR_Ridge", **make_metrics(y, p_sgwr)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ================================================================
# BOUNDARY
# ================================================================
def load_boundary(uploaded_boundary, df, lat_col, lon_col):
    if not MAP_OK:
        return None, "Komponen peta belum tersedia."

    gdf = None
    source = ""
    if uploaded_boundary is not None:
        name = uploaded_boundary.name.lower()
        try:
            raw = uploaded_boundary.getvalue()
            if name.endswith(".geojson") or name.endswith(".json"):
                gdf = gpd.read_file(io.BytesIO(raw))
            elif name.endswith(".zip"):
                # GeoPandas/Fiona can read zipped shapefile in many environments.
                temp = Path("/tmp/sgwrf_boundary.zip")
                temp.write_bytes(raw)
                gdf = gpd.read_file(f"zip://{temp}")
            else:
                st.warning("Untuk boundary, gunakan GeoJSON atau ZIP Shapefile.")
            source = "Boundary yang di-upload"
        except Exception as e:
            st.warning(f"Boundary upload gagal: {e}")
            gdf = None

    if gdf is None:
        try:
            import requests
            params = {
                "where": "1=1", "outFields": "*", "returnGeometry": "true",
                "outSR": 4326, "f": "geojson", "resultRecordCount": 1000,
            }
            r = requests.get(BIG_QUERY_URL, params=params, timeout=45,
                             headers={"User-Agent": "SGWRF-Streamlit-Research"})
            r.raise_for_status()
            gj = r.json()
            if "features" in gj:
                gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
                source = "BIG RBI (fallback otomatis)"
        except Exception as e:
            return None, f"Boundary otomatis gagal: {e}"

    if gdf is None or gdf.empty:
        return None, "Boundary tidak tersedia."
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    # Dynamic extent around actual points — no hard-coded Jabodetabek extent.
    minx, maxx = float(df[lon_col].min()), float(df[lon_col].max())
    miny, maxy = float(df[lat_col].min()), float(df[lat_col].max())
    dx = max((maxx - minx) * 0.20, 0.05)
    dy = max((maxy - miny) * 0.20, 0.05)
    bounds = gdf.geometry.bounds
    mask = ((bounds.maxx >= minx-dx) & (bounds.minx <= maxx+dx) &
            (bounds.maxy >= miny-dy) & (bounds.miny <= maxy+dy))
    gdf = gdf.loc[mask].copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if make_valid is not None:
        try:
            gdf["geometry"] = gdf.geometry.apply(lambda x: make_valid(x) if not x.is_valid else x)
        except Exception:
            pass
    return (gdf if not gdf.empty else None), source


# ================================================================
# INTERACTIVE VISUALS
# ================================================================
def plot_map(df, value_col, title, lat_col, lon_col, name_col,
             boundary=None, cmap="Turbo", center=None, zoom=9,
             categorical=False, tooltip_extra=None):
    if not MAP_OK:
        st.info("Install geopandas, folium, streamlit-folium untuk mengaktifkan peta.")
        return

    d = df.copy()
    if center is None:
        center = [float(d[lat_col].mean()), float(d[lon_col].mean())]

    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB Positron", control=True).add_to(m)

    if boundary is not None and not boundary.empty:
        folium.GeoJson(
            boundary.to_json(), name="Batas administrasi",
            style_function=lambda x: {"fillColor": "#dfe7ef", "color": "#667085",
                                       "weight": 1, "fillOpacity": 0.12},
            highlight_function=lambda x: {"weight": 2, "color": "#1f4e79", "fillOpacity": 0.20},
            tooltip=folium.GeoJsonTooltip(fields=[c for c in boundary.columns
                                                  if c != "geometry"][:2], sticky=False)
            if len([c for c in boundary.columns if c != "geometry"]) else None,
        ).add_to(m)

    vals = pd.to_numeric(d[value_col], errors="coerce")
    if categorical:
        cats = d[value_col].astype(str).unique().tolist()
        palette = px.colors.qualitative.Safe
        cmap_dict = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
    else:
        vmin, vmax = float(vals.min()), float(vals.max())
        if np.isclose(vmin, vmax): vmax = vmin + 1
        import branca.colormap as bcm
        color_scale = bcm.LinearColormap(px.colors.sequential.Turbo, vmin=vmin, vmax=vmax)
        color_scale.caption = value_col
        color_scale.add_to(m)

    for idx, r in d.iterrows():
        if categorical:
            color = cmap_dict[str(r[value_col])]
        else:
            color = color_scale(float(r[value_col])) if pd.notna(r[value_col]) else "gray"
        popup_html = f"<b>{r[name_col]}</b><br>Point: {r.get('point_no', idx+1)}<br>{value_col}: {r[value_col]}"
        if tooltip_extra:
            for c in tooltip_extra:
                if c in r:
                    popup_html += f"<br>{c}: {r[c]}"
        folium.CircleMarker(
            [float(r[lat_col]), float(r[lon_col])], radius=8,
            color="#111827", weight=1, fill=True, fill_color=color, fill_opacity=.88,
            tooltip=f"{r[name_col]} — {value_col}: {r[value_col]}",
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(m)
        folium.Marker(
            [float(r[lat_col]), float(r[lon_col])],
            icon=folium.DivIcon(html=f'<div style="font-size:9px;font-weight:700;color:#111827;">{r.get("point_no", idx+1)}</div>')
        ).add_to(m)

    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MousePosition(position="bottomleft", separator=" | ", num_digits=5).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=620, returned_objects=[])


def plot_scatter_actual_pred(results, y_col):
    fig = px.scatter(results, x=y_col, y="prediction_sgwrf", hover_name=results.columns[2],
                     hover_data=[c for c in ["point_no", "residual", "abs_error", "dominant_variable"] if c in results.columns],
                     title="Aktual vs Prediksi SGWRF", template="plotly_white")
    mn = min(results[y_col].min(), results["prediction_sgwrf"].min())
    mx = max(results[y_col].max(), results["prediction_sgwrf"].max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Ideal 1:1", line=dict(dash="dash")))
    fig.update_layout(height=480, xaxis_title="Aktual", yaxis_title="Prediksi SGWRF")
    return fig


def plot_bandwidth_heatmap(bw_results):
    p = bw_results.pivot(index="k_geo", columns="k_attr", values="RMSE_CV")
    fig = px.imshow(p, text_auto=".3f", aspect="auto", color_continuous_scale="Viridis",
                    labels={"x": "Bandwidth atribut (k)", "y": "Bandwidth geografis (k)", "color": "RMSE CV"},
                    title="Heatmap Optimasi Bandwidth SGWRF")
    fig.update_layout(height=520, template="plotly_white")
    return fig


def plot_global_importance(results, x_cols, labels):
    rows = []
    for c in x_cols:
        rows.append({"variable": labels.get(c, c), "importance": results[f"VI_{c}"].mean() * 100})
    d = pd.DataFrame(rows).sort_values("importance", ascending=True)
    return px.bar(d, x="importance", y="variable", orientation="h", title="Rata-rata Kepentingan Variabel Lokal",
                  labels={"importance": "Importance (%)", "variable": "Variabel"}, template="plotly_white", height=500)


def plot_local_importance_heatmap(results, x_cols, labels):
    d = results[["point_no"] + [f"VI_{c}" for c in x_cols]].copy().set_index("point_no")
    d.columns = [labels.get(c.replace("VI_", ""), c.replace("VI_", "")) for c in d.columns]
    fig = px.imshow(d * 100, aspect="auto", color_continuous_scale="Turbo",
                    labels={"x": "Variabel", "y": "Titik", "color": "Importance (%)"},
                    title="Local Variable Importance — Semua Titik")
    fig.update_layout(height=max(500, min(1000, 18 * len(d) + 300)), template="plotly_white")
    return fig


def plot_residuals(results, name_col):
    d = results.copy()
    d["status"] = np.where(d["residual"] >= 0, "Over-prediction?", "Under-prediction?")
    fig = px.bar(d, x=name_col, y="residual", color="residual", color_continuous_scale="RdBu",
                 hover_data=["point_no", "abs_error", "dominant_variable"],
                 title="Residual SGWRF per Lokasi", template="plotly_white")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=500, xaxis_tickangle=-45)
    return fig


def make_equations(df, intercepts, coefs, x_cols, labels):
    rows = []
    for i in range(len(df)):
        terms = []
        for j, c in enumerate(x_cols):
            sign = "+" if coefs[i, j] >= 0 else "-"
            terms.append(f" {sign} {abs(coefs[i,j]):.4f}·Z({labels.get(c,c)})")
        eq = f"ŷ_{i+1} = {intercepts[i]:.4f}" + "".join(terms)
        rows.append({"point_no": i+1, "location": df.iloc[i][df.columns[1]], "equation": eq})
    return pd.DataFrame(rows)


# ================================================================
# SIDEBAR — DATA INPUT + SETTINGS
# ================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")
    uploaded = st.file_uploader("Upload data (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
    boundary_upload = st.file_uploader("Boundary opsional (.geojson / .json / .zip)", type=["geojson", "json", "zip"])

    if uploaded is None:
        st.info("Upload dataset untuk memulai. Dataset tidak dibaca langsung dari script.")
        st.stop()

    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        # Read workbook names safely.
        raw = uploaded.getvalue()
        try:
            xls = pd.ExcelFile(io.BytesIO(raw))
            sheet = st.selectbox("Sheet", xls.sheet_names)
            st.session_state.sheet_name = sheet
        except Exception:
            st.session_state.sheet_name = 0
    else:
        st.session_state.sheet_name = 0

    raw_df = read_uploaded_data(uploaded)
    st.caption(f"Dataset: **{uploaded.name}** — {raw_df.shape[0]:,} baris × {raw_df.shape[1]:,} kolom")

    cols = list(raw_df.columns)
    st.subheader("Pemetaan kolom")
    id_default = infer_column(cols, ["station_id", "id", "kode", "code"])
    name_default = infer_column(cols, ["location_name", "nama", "location", "wilayah", "kabupaten", "kota"])
    lat_default = infer_column(cols, ["latitude", "lat", "lintang"])
    lon_default = infer_column(cols, ["longitude", "lon", "long", "bujur"])
    y_default = infer_column(cols, ["pm25", "pm2.5", "pm_25", "y", "target"])

    def select_col(label, default):
        options = ["— pilih —"] + cols
        idx = options.index(default) if default in options else 0
        v = st.selectbox(label, options, index=idx)
        return None if v == "— pilih —" else v

    id_col = select_col("ID titik", id_default)
    name_col = select_col("Nama lokasi", name_default)
    lat_col = select_col("Latitude", lat_default)
    lon_col = select_col("Longitude", lon_default)
    y_col = select_col("Target/Y", y_default)

    st.markdown("**Variabel X/kovariat**")
    numeric_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(raw_df[c])]
    suggested = [c for c in DEFAULT_X_COLS if c in cols]
    if not suggested:
        suggested = [c for c in numeric_candidates if c not in {lat_col, lon_col, y_col}]
    x_cols = st.multiselect("Pilih kovariat X", cols, default=suggested)

    st.markdown("**Label variabel (opsional)**")
    labels = {}
    for c in x_cols:
        labels[c] = st.text_input(f"Label: {c}", value=DEFAULT_LABELS.get(c, str(c)), key=f"label_{c}")

    st.subheader("SGWRF")
    min_k = st.number_input("Minimum k", min_value=2, max_value=50, value=5, step=1)
    max_k = st.number_input("Maximum k", min_value=2, max_value=200, value=13, step=1)
    search_mode = st.selectbox("Bandwidth search", ["grid", "random"])
    n_random = st.number_input("Jumlah kandidat random", min_value=10, max_value=2000, value=200, step=10)
    leave_target_out = st.checkbox("LOOCV / leave target out", value=True)

    st.subheader("Random Forest")
    trees_cv = st.number_input("RF trees — CV", min_value=20, max_value=500, value=80, step=10)
    final_trees = st.number_input("RF trees — final", min_value=50, max_value=2000, value=500, step=50)
    perm_repeats = st.number_input("Permutation repeats", min_value=3, max_value=100, value=20, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=2026, step=1)
    run_baseline = st.checkbox("Jalankan baseline models", value=True)

    st.subheader("Opsi dashboard")
    auto_run = st.checkbox("Jalankan analisis setelah tombol ditekan", value=True)
    run = st.button("🚀 Jalankan SGWRF", type="primary", use_container_width=True)


# ================================================================
# DATA VALIDATION
# ================================================================
if not all([id_col, name_col, lat_col, lon_col, y_col]) or len(x_cols) < 1:
    st.warning("Lengkapi ID, nama lokasi, latitude, longitude, Y, dan minimal 1 kovariat X di sidebar.")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.stop()

if run or "sgwrf_result" in st.session_state:
    pass
else:
    st.info("Tekan **🚀 Jalankan SGWRF** untuk memulai analisis.")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.stop()


# ================================================================
# RUN PIPELINE WITH SESSION CACHE
# ================================================================
config_key = str((uploaded.name, getattr(uploaded, "size", None), id_col, name_col,
                   lat_col, lon_col, y_col, tuple(x_cols), min_k, max_k, search_mode,
                   n_random, leave_target_out, trees_cv, final_trees, perm_repeats, seed, run_baseline))

if run or st.session_state.get("config_key") != config_key:
    try:
        df, n_before, n_valid = validate_and_prepare(
            raw_df, id_col, name_col, lat_col, lon_col, y_col, x_cols
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    X = df[x_cols].to_numpy(float)
    y = df[y_col].to_numpy(float)
    coords = df[[lat_col, lon_col]].to_numpy(float)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    d_geo = haversine_km(coords)
    d_attr = cdist(Z, Z, metric="euclidean")

    st.session_state["data_summary"] = {
        "n_before": n_before, "n_valid": n_valid, "p": len(x_cols),
        "geo_max": float(d_geo.max()), "attr_max": float(d_attr.max())
    }

    progress = st.progress(0, text="Mempersiapkan analisis...")
    best_bw, bw_results, bw_time = optimize_bandwidth(
        X, y, d_geo, d_attr, int(min_k), int(max_k), int(trees_cv),
        search_mode, int(n_random), int(seed), leave_target_out, progress
    )
    W, Wg, Wa, bg, ba = combined_weights(d_geo, d_attr, *best_bw)
    best_rf, rf_results = optimize_rf(X, y, W, int(trees_cv), leave_target_out, int(seed), progress)
    pred, local_r2, local_mae, local_rmse, importances, local_models = train_local_models(
        X, y, W, df, x_cols, labels, best_rf, int(final_trees), int(perm_repeats),
        leave_target_out, int(seed), progress
    )
    progress.empty()

    overall = make_metrics(y, pred)
    results = build_results(
        df, y, pred, local_r2, local_mae, local_rmse, importances, bg, ba,
        name_col, id_col, lat_col, lon_col, y_col, x_cols, labels
    )
    intercepts, coefs, ridge_pred = local_ridge_diagnostic(X, y, W, x_cols)
    equations = make_equations(df, intercepts, coefs, x_cols, labels)

    baseline = run_baselines(X, y, d_geo, d_attr, best_bw, int(final_trees), x_cols, int(seed)) if run_baseline else None

    st.session_state["sgwrf_result"] = {
        "df": df, "results": results, "bw_results": bw_results,
        "rf_results": rf_results, "baseline": baseline, "overall": overall,
        "best_bw": best_bw, "best_rf": best_rf, "W": W, "Wg": Wg, "Wa": Wa,
        "bg": bg, "ba": ba, "importances": importances, "intercepts": intercepts,
        "coefs": coefs, "equations": equations, "d_geo": d_geo, "d_attr": d_attr,
        "X": X, "y": y, "x_cols": x_cols, "labels": labels,
        "id_col": id_col, "name_col": name_col, "lat_col": lat_col,
        "lon_col": lon_col, "y_col": y_col, "bw_time": bw_time,
        "n_before": n_before, "n_valid": n_valid,
    }
    st.session_state["config_key"] = config_key
else:
    # If config is unchanged, use existing result.
    pass

R = st.session_state.get("sgwrf_result")
if R is None:
    st.stop()

results = R["results"]

# Boundary loaded after result, so it also adapts to changed data.
boundary, boundary_source = load_boundary(boundary_upload, R["df"], R["lat_col"], R["lon_col"])


# ================================================================
# TOP KPI
# ================================================================
st.markdown("### 📌 Ringkasan Model")
cols = st.columns(7)
cols[0].metric("Observasi valid", f"{R['n_valid']:,}")
cols[1].metric("Kovariat", f"{len(R['x_cols']):,}")
cols[2].metric("k geografis", R["best_bw"][0])
cols[3].metric("k atribut", R["best_bw"][1])
cols[4].metric("RMSE", f"{R['overall']['RMSE']:.4f}")
cols[5].metric("MAE", f"{R['overall']['MAE']:.4f}")
cols[6].metric("R²", f"{R['overall']['R2']:.4f}")

st.caption(f"Jarak geografis maksimum: {R['d_geo'].max():.3f} km · Jarak atribut maksimum: {R['d_attr'].max():.3f} · Kernel: Gaussian · Bandwidth adaptive")


# ================================================================
# TABS
#
# ================================================================
tabs = st.tabs([
    "🎯 Importance", "📐 Diagnostic", "⚖️ Baseline", "📥 Download"
])

with tabs[0]:
    st.subheader("Overview SGWRF")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_scatter_actual_pred(results, R["y_col"]), use_container_width=True)
    with c2:
        st.plotly_chart(plot_global_importance(results, R["x_cols"], R["labels"]), use_container_width=True)

    st.markdown("#### Statistik preprocessing")
    prep = pd.DataFrame({
        "Item": ["Observasi awal", "Observasi valid", "Jumlah kovariat", "Jarak geografis maksimum (km)",
                  "Jarak atribut maksimum", "Bandwidth geo optimum", "Bandwidth atribut optimum",
                  "Waktu optimasi bandwidth (detik)"],
        "Nilai": [R["n_before"], R["n_valid"], len(R["x_cols"]), R["d_geo"].max(), R["d_attr"].max(),
                  R["best_bw"][0], R["best_bw"][1], R["bw_time"]]
    })
    st.dataframe(prep, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Peta Interaktif")
    map_choice = st.selectbox("Variabel yang dipetakan", [R["y_col"], "prediction_sgwrf", "residual", "abs_error",
                                                            "dominant_importance", "local_train_R2", "bandwidth_geo_k", "bandwidth_attr_k"])
    st.caption(f"Boundary: {boundary_source if boundary is not None else boundary_source}")
    plot_map(results, map_choice, map_choice, R["lat_col"], R["lon_col"], R["name_col"],
             boundary=boundary, tooltip_extra=["dominant_variable", "prediction_sgwrf", "residual"])
    st.info("Peta dapat di-zoom, digeser, diklik, menampilkan popup titik, mengganti basemap, fullscreen, dan mengaktifkan/menonaktifkan layer.")

with tabs[2]:
    st.subheader("Optimasi Bandwidth Adaptive")
    st.plotly_chart(plot_bandwidth_heatmap(R["bw_results"]), use_container_width=True)
    st.dataframe(R["bw_results"], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download hasil optimasi bandwidth CSV",
                       R["bw_results"].to_csv(index=False).encode("utf-8"),
                       "hasil_optimasi_bandwidth.csv", "text/csv")

with tabs[3]:
    st.subheader("Optimasi Hyperparameter Random Forest")
    st.dataframe(R["rf_results"], use_container_width=True, hide_index=True)
    fig = px.bar(R["rf_results"].astype({"n_estimators": int}), x="RMSE_CV", y="n_estimators",
                 color="RMSE_CV", orientation="h", hover_data=["max_features", "min_samples_leaf", "max_depth"],
                 title="RMSE CV setiap konfigurasi RF", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.json({"RF terbaik": R["best_rf"]})

with tabs[4]:
    st.subheader("Hasil Model Lokal — setiap titik")
    loc_cols = ["point_no", R["name_col"], R["y_col"], "prediction_sgwrf", "residual", "abs_error",
                "dominant_variable", "dominant_importance", "local_train_R2", "local_train_RMSE",
                "local_train_MAE", "bandwidth_geo_k", "bandwidth_attr_k"]
    st.dataframe(results[loc_cols], use_container_width=True, hide_index=True)
    st.plotly_chart(plot_residuals(results, R["name_col"]), use_container_width=True)

    selected = st.selectbox("Lihat detail titik", results["point_no"].tolist())
    row = results[results["point_no"] == selected].iloc[0]
    st.markdown(f"#### Titik {int(selected)} — {row[R['name_col']]}")
    d_imp = pd.DataFrame({"variable": [R["labels"].get(c,c) for c in R["x_cols"]],
                          "importance": [row[f"VI_{c}"] * 100 for c in R["x_cols"]]}).sort_values("importance", ascending=True)
    st.plotly_chart(px.bar(d_imp, x="importance", y="variable", orientation="h", title="Local permutation importance",
                           labels={"importance": "Importance (%)"}, template="plotly_white"), use_container_width=True)

with tabs[5]:
    st.subheader("Variable Importance")
    st.plotly_chart(plot_global_importance(results, R["x_cols"], R["labels"]), use_container_width=True)
    st.plotly_chart(plot_local_importance_heatmap(results, R["x_cols"], R["labels"]), use_container_width=True)
    imp_long = []
    for _, rr in results.iterrows():
        for c in R["x_cols"]:
            imp_long.append({"point_no": rr["point_no"], "variable": R["labels"].get(c,c),
                             "importance": rr[f"VI_{c}"] * 100})
    imp_long = pd.DataFrame(imp_long)
    variable_filter = st.multiselect("Filter variabel", [R["labels"].get(c,c) for c in R["x_cols"]],
                                     default=[R["labels"].get(c,c) for c in R["x_cols"]])
    if variable_filter:
        st.dataframe(imp_long[imp_long["variable"].isin(variable_filter)], use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Diagnostik Regresi Lokal")
    st.warning("Persamaan berikut adalah Weighted Ridge lokal sebagai diagnostik tambahan. Ini BUKAN koefisien SGWRF; model utama adalah Random Forest.")
    st.dataframe(R["equations"], use_container_width=True, hide_index=True)
    st.markdown("#### Residual dan performa lokal")
    st.dataframe(results[["point_no", R["name_col"], "local_train_R2", "local_train_RMSE", "local_train_MAE", "residual", "abs_error"]],
                 use_container_width=True, hide_index=True)

    coef_rows = []
    for i in range(len(results)):
        for j, c in enumerate(R["x_cols"]):
            coef_rows.append({"point_no": i+1, "variable": R["labels"].get(c,c), "coefficient_ridge": R["coefs"][i,j]})
    coef_df = pd.DataFrame(coef_rows)
    st.plotly_chart(px.imshow(coef_df.pivot(index="point_no", columns="variable", values="coefficient_ridge"),
                              aspect="auto", color_continuous_scale="RdBu", title="Koefisien Weighted Ridge Lokal (diagnostik)",
                              labels={"color": "Koefisien"}), use_container_width=True)

with tabs[7]:
    st.subheader("Perbandingan Model Baseline")
    if R["baseline"] is None:
        st.info("Baseline tidak dijalankan.")
    else:
        st.dataframe(R["baseline"], use_container_width=True, hide_index=True)
        fig = px.bar(R["baseline"], x="Model", y="RMSE", color="Model", title="Perbandingan RMSE Model",
                     template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(R["baseline"], x="Model", y="R2", color="Model", title="Perbandingan R² Model",
                      template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[8]:
    st.subheader("Download Semua Output")
    output_map = {
        "hasil_lokal_sgwrf.csv": results,
        "hasil_optimasi_bandwidth.csv": R["bw_results"],
        "hasil_optimasi_rf.csv": R["rf_results"],
        "persamaan_ridge_lokal_diagnostik.csv": R["equations"],
    }
    if R["baseline"] is not None:
        output_map["perbandingan_model.csv"] = R["baseline"]
    imp_df = results[["point_no"] + [f"VI_{c}" for c in R["x_cols"]]].copy()
    output_map["local_variable_importance.csv"] = imp_df

    for fname, data in output_map.items():
        st.download_button(f"⬇️ {fname}", data.to_csv(index=False).encode("utf-8"), fname, "text/csv", use_container_width=True)

    st.markdown("#### Dataset setelah validasi")
    st.download_button("⬇️ Download data valid", R["df"].to_csv(index=False).encode("utf-8"),
                       "data_valid_sgwrf.csv", "text/csv", use_container_width=True)

    st.markdown("#### Konfigurasi analisis")
    config = pd.DataFrame({
        "parameter": ["ID", "Nama", "Latitude", "Longitude", "Y", "X", "min_k", "max_k", "search", "trees_cv", "trees_final", "permutation_repeats", "seed", "LOOCV"],
        "value": [R["id_col"], R["name_col"], R["lat_col"], R["lon_col"], R["y_col"], ", ".join(R["x_cols"]), min_k, max_k, search_mode, trees_cv, final_trees, perm_repeats, seed, leave_target_out]
    })
    st.dataframe(config, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<div class='small-note'>SGWRF Dashboard — adaptasi dari SGWRF2.ipynb. Model utama: local Random Forest dengan bobot gabungan jarak geografis dan atribut terstandar menggunakan kernel Gaussian dan adaptive bandwidth.</div>", unsafe_allow_html=True)
