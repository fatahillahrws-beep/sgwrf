# ================================================================
# SGWRF INTERACTIVE DASHBOARD
# Semi-parametric Geographically Weighted Random Forest
# ----------------------------------------------------------------
# Dashboard Streamlit interaktif untuk analisis SGWRF.
# Data TIDAK di-hardcode - pengguna mengunggah file sendiri
# (xlsx / csv), memetakan kolom, mengatur parameter model, lalu
# menjalankan seluruh pipeline: optimasi bandwidth adaptif,
# optimasi hyperparameter Random Forest, model lokal per titik,
# variable importance lokal, model baseline pembanding, dan
# peta / grafik interaktif (Plotly) yang otomatis menyesuaikan
# lokasi & rentang nilai data yang diunggah.
#
# Jalankan dengan:
#   pip install -r requirements.txt
#   streamlit run sgwrf_dashboard.py
# ================================================================

import io
import json
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ================================================================
# 0. KONFIGURASI HALAMAN & GAYA TAMPILAN
# ================================================================
st.set_page_config(
    page_title="SGWRF Interactive Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root{
    --sgwrf-primary:#0f4c81;
    --sgwrf-accent:#f39c12;
    --sgwrf-bg:#f5f7fa;
}
.main .block-container{padding-top:1.4rem;padding-bottom:2.5rem;max-width:1400px;}
h1,h2,h3{color:var(--sgwrf-primary);}
div[data-testid="stMetricValue"]{color:var(--sgwrf-primary);font-weight:700;}
div[data-testid="stMetric"]{
    background:white;border:1px solid #e3e8ee;border-radius:12px;
    padding:0.7rem 0.9rem;box-shadow:0 1px 3px rgba(15,76,129,0.08);
}
section[data-testid="stSidebar"]{background:#0f4c81;}
section[data-testid="stSidebar"] *{color:#eef4fb !important;}
section[data-testid="stSidebar"] .stButton>button{
    background:var(--sgwrf-accent);color:#20242b !important;font-weight:700;
    border:none;border-radius:8px;
}
.stTabs [data-baseweb="tab-list"]{gap:4px;}
.stTabs [data-baseweb="tab"]{
    background:#eef2f7;border-radius:8px 8px 0 0;padding:8px 16px;font-weight:600;
}
.stTabs [aria-selected="true"]{background:var(--sgwrf-primary);color:white;}
.sgwrf-banner{
    background:linear-gradient(90deg,#0f4c81,#1c7ed6);
    color:white;padding:1.1rem 1.4rem;border-radius:14px;margin-bottom:1.1rem;
}
.sgwrf-banner h1{color:white;margin:0;font-size:1.55rem;}
.sgwrf-banner p{color:#dbe9fb;margin:0.2rem 0 0 0;font-size:0.92rem;}
.sgwrf-note{
    background:#fff8e6;border-left:4px solid var(--sgwrf-accent);
    padding:0.6rem 0.9rem;border-radius:6px;font-size:0.88rem;
}
.sgwrf-interpret{
    background:#eef6ff;border-left:4px solid #1c7ed6;color:#123a5c;
    padding:0.55rem 0.9rem;border-radius:6px;font-size:0.87rem;margin:0.35rem 0 1.1rem 0;
}
.sgwrf-interpret b{color:#0f4c81;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="sgwrf-banner">
    <h1>🛰️ SGWRF Interactive Dashboard</h1>
    <p>Semi-parametric Geographically Weighted Random Forest — bandwidth adaptif ganda
    (geografis × atribut), kernel Gaussian, model lokal Random Forest, dan variable
    importance lokal. Unggah data Anda sendiri — peta &amp; grafik menyesuaikan otomatis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# 1. FUNGSI INTI (diadaptasi dari script SGWRF asli)
# ================================================================

def haversine_km(latlon):
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def adaptive_bandwidth_matrix(d, k):
    d = np.array(d, copy=True)
    np.fill_diagonal(d, np.inf)
    k = int(np.clip(k, 1, d.shape[0] - 1))
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


def prepare_matrices(df, x_cols, y_col, lat_col, lon_col):
    X = df[x_cols].to_numpy(float)
    y = df[y_col].to_numpy(float)
    coords = df[[lat_col, lon_col]].to_numpy(float)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    d_geo = haversine_km(coords)
    d_attr = cdist(Z, Z, metric="euclidean")
    return X, Z, y, coords, d_geo, d_attr, scaler


def candidate_bandwidths(n, min_k, max_k):
    max_k = (n - 1) if max_k is None else min(max_k, n - 1)
    min_k = max(2, min(min_k, max_k))
    return list(range(min_k, max_k + 1))


def rf_fit_predict(X_train, y_train, X_test, sample_weight, params, random_state):
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_depth=params.get("max_depth", None),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, model.predict(X_test)


def cv_score_bandwidth(k_pair, X, y, d_geo, d_attr, params, leave_target_out, random_state):
    k_geo, k_attr = k_pair
    W, *_ = combined_weights(d_geo, d_attr, k_geo, k_attr)
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        w = W[i].copy()
        if leave_target_out:
            w[i] = 0.0
        if np.count_nonzero(w > 1e-8) < 4:
            return np.inf, preds
        _, pred = rf_fit_predict(X, y, X[i:i + 1], w, params, random_state + i)
        preds[i] = pred[0]
    residual = y - preds
    return float(np.sum(residual ** 2)), preds


def optimize_bandwidth(X, y, d_geo, d_attr, cfg, progress_cb=None):
    n = len(y)
    ks = candidate_bandwidths(n, cfg["min_k"], cfg["max_k"])
    pairs = list(product(ks, ks))
    if cfg["bw_search"] == "random" and len(pairs) > cfg["n_random_bw"]:
        rng = np.random.default_rng(cfg["seed"])
        idx = rng.choice(len(pairs), size=cfg["n_random_bw"], replace=False)
        pairs = [pairs[i] for i in idx]

    cv_params = {"n_estimators": cfg["rf_cv_trees"], "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None}

    rows, best = [], None
    t0 = time.time()
    for no, pair in enumerate(pairs, 1):
        cv, preds = cv_score_bandwidth(pair, X, y, d_geo, d_attr, cv_params, cfg["leave_target_out"], cfg["seed"])
        rmse = np.sqrt(cv / n) if np.isfinite(cv) else np.inf
        rows.append({"iteration": no, "k_geo": pair[0], "k_attr": pair[1], "CV": cv, "RMSE_CV": rmse})
        if best is None or cv < best["CV"]:
            best = rows[-1].copy()
            best["preds"] = preds.copy()
        if progress_cb:
            progress_cb(no / len(pairs), f"Bandwidth {no}/{len(pairs)} — k_geo={pair[0]}, k_attr={pair[1]}, RMSE={rmse:.4f}")
    elapsed = time.time() - t0
    res = pd.DataFrame(rows).sort_values("CV").reset_index(drop=True)
    return (int(best["k_geo"]), int(best["k_attr"])), res, elapsed


def optimize_rf(X, y, W, cfg, param_grid, progress_cb=None):
    rows, best = [], None
    total = len(param_grid)
    for j, params in enumerate(param_grid, 1):
        p = params.copy()
        p["n_estimators"] = min(p["n_estimators"], cfg["rf_cv_trees"])
        preds = np.full(len(y), np.nan)
        for i in range(len(y)):
            w = W[i].copy()
            if cfg["leave_target_out"]:
                w[i] = 0
            _, pr = rf_fit_predict(X, y, X[i:i + 1], w, p, cfg["seed"] + 1000 * j + i)
            preds[i] = pr[0]
        rmse = np.sqrt(mean_squared_error(y, preds))
        rows.append({**p, "RMSE_CV": rmse})
        if best is None or rmse < best["RMSE_CV"]:
            best = rows[-1].copy()
        if progress_cb:
            progress_cb(j / total, f"RF config {j}/{total} — RMSE_CV={rmse:.4f}")
    return best, pd.DataFrame(rows)


def train_local_models(X, y, W, df, name_col, x_cols, x_labels, rf_final, cfg, progress_cb=None):
    n = len(y)
    preds = np.full(n, np.nan)
    local_r2 = np.full(n, np.nan)
    local_mae = np.full(n, np.nan)
    local_rmse = np.full(n, np.nan)
    importances = np.zeros((n, X.shape[1]))

    final_params = {
        "n_estimators": cfg["rf_final_trees"],
        "max_features": rf_final.get("max_features", "sqrt"),
        "min_samples_leaf": rf_final.get("min_samples_leaf", 1),
        "max_depth": rf_final.get("max_depth", None),
    }

    for i in range(n):
        w = W[i].copy()
        train_mask = w > 1e-8
        if cfg["leave_target_out"]:
            train_mask[i] = False
            w[i] = 0.0
        if train_mask.sum() < 4:
            train_mask[:] = True
            if cfg["leave_target_out"]:
                train_mask[i] = False

        model = RandomForestRegressor(
            n_estimators=final_params["n_estimators"],
            max_features=final_params["max_features"],
            min_samples_leaf=final_params["min_samples_leaf"],
            max_depth=final_params["max_depth"],
            random_state=cfg["seed"] + i,
            n_jobs=-1,
        )
        model.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
        preds[i] = model.predict(X[i:i + 1])[0]

        train_pred = model.predict(X[train_mask])
        local_rmse[i] = np.sqrt(mean_squared_error(y[train_mask], train_pred, sample_weight=w[train_mask]))
        local_mae[i] = mean_absolute_error(y[train_mask], train_pred, sample_weight=w[train_mask])
        try:
            local_r2[i] = r2_score(y[train_mask], train_pred, sample_weight=w[train_mask])
        except Exception:
            local_r2[i] = np.nan

        try:
            pi = permutation_importance(
                model, X[train_mask], y[train_mask],
                n_repeats=cfg["n_perm_repeats"], random_state=cfg["seed"] + i,
                scoring="neg_mean_squared_error", n_jobs=-1,
            )
            imp = np.maximum(pi.importances_mean, 0)
            imp = imp / imp.sum() if imp.sum() > 0 else imp
            importances[i, :] = imp
        except Exception:
            importances[i, :] = model.feature_importances_

        if progress_cb:
            top_idx = int(np.argmax(importances[i]))
            progress_cb((i + 1) / n, f"Titik {i+1}/{n} — {df.loc[i, name_col]} | dominan: {x_labels[x_cols[top_idx]]}")

    return preds, local_r2, local_mae, local_rmse, importances


def metrics_dict(y, pred):
    err = y - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(y) < 1e-12, np.nan, np.abs(y))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    r2 = float(r2_score(y, pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def local_regression_diagnostic(X, y, W):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    n, p = len(y), Xz.shape[1]
    coefs = np.zeros((n, p))
    intercepts = np.zeros(n)
    for i in range(n):
        w = W[i].copy()
        w[i] = 0
        ridge = Ridge(alpha=1.0).fit(Xz, y, sample_weight=w)
        coefs[i] = ridge.coef_
        intercepts[i] = ridge.intercept_
    return intercepts, coefs


def run_baselines(X, y, d_geo, d_attr, best_bw, cfg, progress_cb=None):
    rows = []
    steps = 4
    step = 0

    rf = RandomForestRegressor(n_estimators=cfg["rf_final_trees"], max_features="sqrt",
                                min_samples_leaf=1, random_state=cfg["seed"], n_jobs=-1)
    rf.fit(X, y)
    rows.append({"Model": "RF_global", **metrics_dict(y, rf.predict(X))})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: RF Global")

    k_geo, k_attr = best_bw
    Wg, *_ = combined_weights(d_geo, d_geo, k_geo, k_geo)
    p_gwr = np.zeros(len(y))
    for i in range(len(y)):
        w = Wg[i].copy(); w[i] = 0
        p_gwr[i] = Ridge(alpha=1.0).fit(X, y, sample_weight=w).predict(X[i:i + 1])[0]
    rows.append({"Model": "GWR_Ridge", **metrics_dict(y, p_gwr)})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: GWR (Ridge terboboti geografis)")

    p_gwrf = np.zeros(len(y))
    for i in range(len(y)):
        w = Wg[i].copy(); w[i] = 0
        m = RandomForestRegressor(n_estimators=cfg["rf_final_trees"], max_features="sqrt",
                                   min_samples_leaf=1, random_state=cfg["seed"] + i, n_jobs=-1)
        m.fit(X, y, sample_weight=w)
        p_gwrf[i] = m.predict(X[i:i + 1])[0]
    rows.append({"Model": "GWRF", **metrics_dict(y, p_gwrf)})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: GWRF")

    Wsg, *_ = combined_weights(d_geo, d_attr, k_geo, k_attr)
    p_sgwr = np.zeros(len(y))
    for i in range(len(y)):
        w = Wsg[i].copy(); w[i] = 0
        p_sgwr[i] = Ridge(alpha=1.0).fit(X, y, sample_weight=w).predict(X[i:i + 1])[0]
    rows.append({"Model": "SGWR_Ridge", **metrics_dict(y, p_sgwr)})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: SGWR (Ridge terboboti ganda)")

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def build_results(df, id_col, name_col, lat_col, lon_col, y_col, x_cols, x_labels,
                   y, pred, local_r2, local_mae, local_rmse, importances, bg, ba):
    out = df[[id_col, name_col, lat_col, lon_col, y_col]].copy()
    out.insert(0, "point_no", np.arange(1, len(out) + 1))
    out["pred_sgwrf"] = pred
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
    out["dominant_variable"] = [x_labels[x_cols[j]] for j in top]
    out["dominant_variable_raw"] = [x_cols[j] for j in top]
    out["dominant_importance"] = importances[np.arange(len(y)), top]
    return out


# ================================================================
# 2. UTILITAS PETA INTERAKTIF (Plotly, otomatis menyesuaikan data)
# ================================================================

def auto_zoom(lat_min, lat_max, lon_min, lon_max):
    lat_diff = max(lat_max - lat_min, 0.002)
    lon_diff = max(lon_max - lon_min, 0.002)
    max_diff = max(lat_diff, lon_diff)
    breakpoints = [0.005, 0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80]
    zoom_levels = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    for bp, z in zip(breakpoints, zoom_levels):
        if max_diff <= bp:
            return z
    return 2


def boundary_lines_from_geojson(gj):
    lats, lons = [], []

    def add_ring(ring):
        for lon, lat in ring:
            lons.append(lon); lats.append(lat)
        lons.append(None); lats.append(None)

    feats = gj.get("features", [gj]) if isinstance(gj, dict) else []
    for f in feats:
        geom = f.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            for ring in coords:
                add_ring(ring)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    add_ring(ring)
    return lats, lons


def make_continuous_map(df, lat_col, lon_col, color_col, name_col, id_col, title,
                         colorscale="YlOrRd", boundary=None, unit=""):
    center = dict(lat=float(df[lat_col].mean()), lon=float(df[lon_col].mean()))
    zoom = auto_zoom(df[lat_col].min(), df[lat_col].max(), df[lon_col].min(), df[lon_col].max())

    fig = go.Figure()
    if boundary is not None:
        blat, blon = boundary
        fig.add_trace(go.Scattermapbox(lat=blat, lon=blon, mode="lines",
                                        line=dict(width=1.3, color="#555555"),
                                        hoverinfo="skip", showlegend=False, name="Batas wilayah"))

    fig.add_trace(go.Scattermapbox(
        lat=df[lat_col], lon=df[lon_col],
        mode="markers+text",
        text=df["point_no"].astype(str) if "point_no" in df.columns else None,
        textfont=dict(size=10, color="black"),
        marker=dict(size=17, color=df[color_col], colorscale=colorscale, showscale=True,
                    colorbar=dict(title=unit or color_col), opacity=0.92),
        customdata=np.stack([df[name_col], df[id_col], df[color_col]], axis=-1),
        hovertemplate=(f"<b>%{{customdata[0]}}</b><br>ID: %{{customdata[1]}}<br>"
                        f"{color_col}: %{{customdata[2]:.3f}}<br>lat:%{{lat:.4f}} lon:%{{lon:.4f}}<extra></extra>"),
        name=color_col,
    ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=center, zoom=zoom),
        title=title, height=620, margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def make_categorical_map(df, lat_col, lon_col, cat_col, name_col, id_col, title, boundary=None):
    center = dict(lat=float(df[lat_col].mean()), lon=float(df[lon_col].mean()))
    zoom = auto_zoom(df[lat_col].min(), df[lat_col].max(), df[lon_col].min(), df[lon_col].max())
    cats = sorted(df[cat_col].unique().tolist())
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

    fig = go.Figure()
    if boundary is not None:
        blat, blon = boundary
        fig.add_trace(go.Scattermapbox(lat=blat, lon=blon, mode="lines",
                                        line=dict(width=1.3, color="#555555"),
                                        hoverinfo="skip", showlegend=False))
    for c in cats:
        sub = df[df[cat_col] == c]
        fig.add_trace(go.Scattermapbox(
            lat=sub[lat_col], lon=sub[lon_col], mode="markers+text",
            text=sub["point_no"].astype(str) if "point_no" in sub.columns else None,
            textfont=dict(size=10, color="black"),
            marker=dict(size=17, color=color_map[c], opacity=0.92),
            customdata=np.stack([sub[name_col], sub[id_col]], axis=-1),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>ID: %{{customdata[1]}}<br>{cat_col}: {c}<extra></extra>",
            name=str(c),
        ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=center, zoom=zoom),
        title=title, height=620, margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(title=cat_col, orientation="v", x=1.01, y=1),
    )
    return fig


# ================================================================
# 3. AUTO-DETEKSI PEMETAAN KOLOM
# ================================================================

def guess_column(columns, keywords, exclude=()):
    cols_lower = {c: str(c).lower() for c in columns}
    for c, lc in cols_lower.items():
        if c in exclude:
            continue
        for kw in keywords:
            if kw in lc:
                return c
    return None


def numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def note(text):
    """Kotak interpretasi otomatis di bawah grafik — nilai dihitung dinamis
    dari data yang sedang aktif, bukan teks statis, sehingga selalu relevan
    walau jumlah titik/kolom berubah."""
    st.markdown(f'<div class="sgwrf-interpret">🔍 <b>Interpretasi:</b> {text}</div>', unsafe_allow_html=True)


def data_fingerprint(uploaded_name, n_rows, id_col, name_col, lat_col, lon_col, y_col, x_cols):
    """Sidik jari konfigurasi data + pemetaan kolom saat ini. Dipakai untuk
    mendeteksi jika pengguna mengunggah data baru / menambah titik / mengubah
    pemetaan kolom, sehingga hasil analisis lama (session_state) otomatis
    dianggap kedaluwarsa dan tidak tercampur dengan data baru."""
    return (uploaded_name, n_rows, id_col, name_col, lat_col, lon_col, y_col, tuple(sorted(x_cols)))


# ================================================================
# 4. SIDEBAR — UPLOAD, PEMETAAN KOLOM, PARAMETER
# ================================================================

with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")

    st.subheader("1️⃣ Data")
    uploaded = st.file_uploader("Unggah data (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])
    boundary_file = st.file_uploader("Batas wilayah opsional (.geojson)", type=["geojson", "json"])

if uploaded is None:
    st.info(
        "👈 Unggah data penelitian Anda di sidebar (format Excel/CSV) untuk memulai. "
        "Data minimal berisi: kolom ID titik, nama lokasi, latitude, longitude, "
        "variabel target (Y), dan kovariat prediktor (X) — sesuai desain SGWRF Anda."
    )
    st.stop()

# --- load file ---
try:
    if uploaded.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded)
    else:
        xls = pd.ExcelFile(uploaded)
        sheet = xls.sheet_names[0]
        if len(xls.sheet_names) > 1:
            with st.sidebar:
                sheet = st.selectbox("Pilih sheet", xls.sheet_names)
        raw_df = pd.read_excel(uploaded, sheet_name=sheet)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

raw_df.columns = [str(c).strip() for c in raw_df.columns]
all_cols = list(raw_df.columns)
num_cols = numeric_columns(raw_df)

with st.sidebar:
    st.subheader("2️⃣ Pemetaan Kolom")
    guess_id = guess_column(all_cols, ["id", "kode", "station"]) or all_cols[0]
    guess_name = guess_column(all_cols, ["name", "nama", "lokasi", "location"]) or all_cols[0]
    guess_lat = guess_column(all_cols, ["lat", "lintang"]) or (num_cols[0] if num_cols else all_cols[0])
    guess_lon = guess_column(all_cols, ["lon", "lng", "bujur"]) or (num_cols[0] if num_cols else all_cols[0])
    guess_y = guess_column(all_cols, ["pm25", "pm2.5", "target", "_y", "y_"]) or (num_cols[-1] if num_cols else all_cols[0])

    id_col = st.selectbox("Kolom ID titik", all_cols, index=all_cols.index(guess_id))
    name_col = st.selectbox("Kolom nama lokasi", all_cols, index=all_cols.index(guess_name))
    lat_col = st.selectbox("Kolom latitude", num_cols or all_cols,
                            index=(num_cols or all_cols).index(guess_lat))
    lon_col = st.selectbox("Kolom longitude", num_cols or all_cols,
                            index=(num_cols or all_cols).index(guess_lon))
    y_col = st.selectbox("Kolom variabel target (Y)", num_cols or all_cols,
                          index=(num_cols or all_cols).index(guess_y))

    default_x = [c for c in num_cols if c not in (lat_col, lon_col, y_col, id_col)]
    x_cols = st.multiselect("Kolom kovariat prediktor (X)", num_cols, default=default_x)

    if not x_cols:
        st.error("Pilih minimal satu kovariat X.")
        st.stop()

    x_labels = {c: c.replace("_", " ").title() for c in x_cols}

    st.subheader("3️⃣ Parameter Bandwidth")
    n_points = len(raw_df.dropna(subset=[id_col, name_col, lat_col, lon_col, y_col] + x_cols))
    default_min_k = min(5, max(2, n_points - 1))
    min_k, max_k = st.slider("Rentang k (jumlah tetangga lokal)", 2, max(3, n_points - 1),
                              (default_min_k, min(max(default_min_k + 3, 5), n_points - 1)))
    n_combo = (max_k - min_k + 1) ** 2
    bw_search_default_idx = 1 if n_combo > 150 else 0
    bw_search = st.radio("Metode pencarian bandwidth", ["grid", "random"], horizontal=True,
                          index=bw_search_default_idx)
    n_random_bw = st.number_input("Jumlah iterasi (mode random)", 20, 2000, 200, step=20,
                                   disabled=(bw_search != "random"))
    leave_target_out = st.checkbox("Leave-target-out (LOO) saat validasi", value=True)

    # Panduan performa otomatis — menyesuaikan saat jumlah titik (n_points)
    # bertambah, karena kompleksitas SGWRF tumbuh terhadap n dan k.
    est_fits = n_combo * n_points if bw_search == "grid" else int(n_random_bw) * n_points
    if n_points > 60 or est_fits > 20000:
        st.warning(
            f"⏱️ Data Anda memiliki **{n_points} titik** dengan estimasi ~{est_fits:,} pelatihan model "
            "pada tahap optimasi bandwidth. Ini bisa memakan waktu cukup lama. Disarankan: gunakan mode "
            "**random** dengan iterasi lebih kecil, kurangi rentang k, atau kurangi jumlah pohon RF tahap CV."
        )

    st.subheader("4️⃣ Parameter Random Forest")
    rf_cv_trees = st.slider("Jumlah pohon (tahap optimasi/CV)", 20, 300, 80, step=10)
    rf_final_trees = st.slider("Jumlah pohon (model final)", 100, 1000, 400, step=50)
    n_perm_repeats = st.slider("Pengulangan permutation importance", 5, 50, 15, step=5)
    seed = st.number_input("Random seed", 0, 99999, 2026)

    st.subheader("5️⃣ Analisis Tambahan")
    run_baseline_flag = st.checkbox("Jalankan model baseline (RF Global, GWR, GWRF, SGWR-Ridge)", value=True)

    st.markdown("---")
    run_button = st.button("🚀 Jalankan Analisis SGWRF", use_container_width=True)

cfg = dict(min_k=min_k, max_k=max_k, bw_search=bw_search, n_random_bw=int(n_random_bw),
           leave_target_out=leave_target_out, rf_cv_trees=rf_cv_trees, rf_final_trees=rf_final_trees,
           n_perm_repeats=n_perm_repeats, seed=int(seed))

RF_PARAM_GRID = [
    {"n_estimators": rf_cv_trees, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
    {"n_estimators": rf_cv_trees, "max_features": 0.7, "min_samples_leaf": 1, "max_depth": None},
    {"n_estimators": rf_cv_trees, "max_features": 1.0, "min_samples_leaf": 1, "max_depth": None},
    {"n_estimators": rf_cv_trees, "max_features": "sqrt", "min_samples_leaf": 2, "max_depth": None},
]

# --- boundary geojson (opsional) ---
boundary = None
if boundary_file is not None:
    try:
        gj = json.load(boundary_file)
        boundary = boundary_lines_from_geojson(gj)
    except Exception as e:
        st.sidebar.warning(f"Gagal membaca GeoJSON batas wilayah: {e}")

# ================================================================
# 5. VALIDASI & PEMBERSIHAN DATA
# ================================================================

required = [id_col, name_col, lat_col, lon_col, y_col] + x_cols
df = raw_df[required].copy()
for c in [lat_col, lon_col, y_col] + x_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
n_before = len(df)
df = df.dropna(subset=required).reset_index(drop=True)
n_after = len(df)

# ----------------------------------------------------------------
# Deteksi otomatis perubahan data (mis. jumlah titik bertambah, file
# diganti, atau pemetaan kolom diubah). Jika berbeda dari analisis
# terakhir, hasil lama di-reset agar tidak ada mismatch dimensi/data
# saat pipeline dijalankan ulang — dashboard selalu mengikuti data
# yang sedang aktif, berapa pun jumlah titiknya.
# ----------------------------------------------------------------
fp = data_fingerprint(uploaded.name, n_after, id_col, name_col, lat_col, lon_col, y_col, x_cols)
if st.session_state.get("sgwrf_fp") != fp:
    if "sgwrf" in st.session_state:
        del st.session_state["sgwrf"]
    st.session_state["sgwrf_fp"] = fp
    st.session_state["sgwrf_data_changed"] = True
else:
    st.session_state["sgwrf_data_changed"] = False

if st.session_state.get("sgwrf_prev_n") is not None and st.session_state["sgwrf_prev_n"] != n_after:
    st.info(
        f"📌 Jumlah titik data terdeteksi berubah: {st.session_state['sgwrf_prev_n']} → {n_after} titik. "
        "Rentang parameter (misalnya k tetangga lokal) sudah menyesuaikan otomatis. "
        "Tekan **Jalankan Analisis SGWRF** di sidebar untuk memproses data terbaru."
    )
st.session_state["sgwrf_prev_n"] = n_after

tab_data, tab_map, tab_bw, tab_rf, tab_local, tab_baseline, tab_download = st.tabs(
    ["📊 Data & Eksplorasi", "🗺️ Peta Interaktif", "🎯 Optimasi Bandwidth",
     "🌲 Optimasi Random Forest", "📈 Model Lokal SGWRF", "⚖️ Perbandingan Baseline", "📥 Unduh Hasil"]
)

# ---------------- TAB: DATA & EKSPLORASI ----------------
with tab_data:
    st.subheader("Ringkasan Data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi awal", n_before)
    c2.metric("Observasi valid", n_after)
    c3.metric("Jumlah kovariat (X)", len(x_cols))
    c4.metric("Duplikat koordinat", int(df.duplicated(subset=[lat_col, lon_col]).sum()))

    if n_after < 8:
        st.error("Jumlah titik valid terlalu sedikit (<8) untuk analisis lokal SGWRF. "
                  "Periksa kembali data atau pemetaan kolom Anda.")
    note(
        f"Dari **{n_before} baris** data mentah, **{n_after} titik** valid dan siap dianalisis "
        f"(baris dengan nilai kosong pada kolom kunci otomatis dibuang). Jika Anda menambah titik "
        f"observasi baru pada file dan mengunggahnya kembali, dashboard akan otomatis mendeteksi "
        f"jumlah titik baru ini, menyesuaikan rentang parameter, dan mereset hasil analisis lama."
    )

    st.dataframe(df, use_container_width=True, height=320)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df[[y_col] + x_cols].describe().T, use_container_width=True)
    note(
        f"Tabel ini merangkum sebaran nilai tiap variabel: rata-rata (mean), sebaran (std), serta nilai "
        f"minimum–maksimum. Perhatikan variabel dengan **std besar relatif terhadap mean** — variabel "
        f"tersebut punya variasi tinggi antar titik dan berpotensi menjadi kovariat yang berpengaruh "
        f"secara spasial dalam model SGWRF."
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Korelasi antar Variabel")
        corr = df[[y_col] + x_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              aspect="auto", title="Matriks Korelasi (Pearson)")
        fig_corr.update_layout(height=460)
        st.plotly_chart(fig_corr, use_container_width=True)
        y_corr = corr[y_col].drop(y_col).abs().sort_values(ascending=False)
        top_var = y_corr.index[0]
        top_val = corr.loc[top_var, y_col]
        arah = "positif (searah)" if top_val > 0 else "negatif (berlawanan arah)"
        note(
            f"Warna **merah** = korelasi positif, **biru** = korelasi negatif; makin pekat warnanya, makin "
            f"kuat hubungan liniernya. Kovariat yang paling berkorelasi dengan **{y_col}** saat ini adalah "
            f"**{x_labels.get(top_var, top_var)}** (r = {top_val:.2f}, hubungan {arah}). Perlu diingat ini "
            f"hanya korelasi **global/linear** — SGWRF justru dibuat untuk menangkap hubungan yang "
            f"**berubah-ubah antar lokasi**, yang tidak terlihat dari matriks ini."
        )
    with cc2:
        st.subheader("Sebaran Variabel Target")
        fig_hist = px.histogram(df, x=y_col, nbins=20, title=f"Distribusi {y_col}", marginal="box")
        fig_hist.update_layout(height=460)
        st.plotly_chart(fig_hist, use_container_width=True)
        skew = float(df[y_col].skew())
        bentuk = "menjulur ke kanan (beberapa titik bernilai tinggi/outlier atas)" if skew > 0.5 else (
            "menjulur ke kiri (beberapa titik bernilai rendah/outlier bawah)" if skew < -0.5 else
            "relatif simetris")
        note(
            f"Histogram menunjukkan sebaran nilai **{y_col}** di seluruh titik: rata-rata "
            f"{df[y_col].mean():.2f}, rentang {df[y_col].min():.2f}–{df[y_col].max():.2f}. Bentuk "
            f"sebarannya **{bentuk}**. Boxplot di atas histogram membantu mengenali titik-titik outlier "
            f"yang mungkin memengaruhi bandwidth lokal di sekitarnya."
        )

    st.subheader("Eksplorasi Hubungan X vs Y")
    xsel = st.selectbox("Pilih kovariat untuk diplot terhadap Y", x_cols)
    fig_scatter = px.scatter(df, x=xsel, y=y_col, hover_name=name_col, trendline="ols",
                              title=f"{x_labels.get(xsel, xsel)} vs {y_col}")
    st.plotly_chart(fig_scatter, use_container_width=True)
    r_xy = float(df[[xsel, y_col]].corr().iloc[0, 1])
    kekuatan = "kuat" if abs(r_xy) >= 0.6 else ("sedang" if abs(r_xy) >= 0.3 else "lemah")
    note(
        f"Garis putus-putus adalah garis tren OLS (regresi linear sederhana). Hubungan **{xsel}** dengan "
        f"**{y_col}** secara global tergolong **{kekuatan}** (r = {r_xy:.2f}). Titik yang menyebar jauh "
        f"dari garis tren mengindikasikan lokasi tempat hubungan ini **tidak berlaku secara global** — "
        f"inilah heterogenitas spasial yang coba ditangkap secara lokal oleh model SGWRF pada tab "
        f"berikutnya."
    )

if n_after < 8:
    st.stop()

df["point_no"] = np.arange(1, len(df) + 1)

# ---------------- TAB: PETA AWAL (selalu tersedia, sebelum run) ----------------
with tab_map:
    st.subheader("Peta Titik Observasi")
    st.caption("Peta menyesuaikan otomatis ke lokasi & rentang data yang diunggah (basemap OpenStreetMap).")
    fig0 = make_continuous_map(df, lat_col, lon_col, y_col, name_col, id_col,
                                f"Sebaran Titik & Nilai {y_col}", boundary=boundary)
    st.plotly_chart(fig0, use_container_width=True)
    st.markdown(
        '<div class="sgwrf-note">💡 Klik-tahan untuk geser, scroll untuk zoom, hover untuk detail titik. '
        'Klik ikon kamera di pojok kanan atas grafik untuk mengunduh sebagai gambar. Peta otomatis '
        're-center &amp; re-zoom mengikuti wilayah data Anda — cocok untuk lokasi mana pun, tidak '
        'terbatas Jabodetabek.</div>',
        unsafe_allow_html=True,
    )
    note(
        f"Warna titik menunjukkan nilai **{y_col}**: makin gelap/pekat, makin tinggi nilainya. Peta ini "
        f"menampilkan **{n_after} titik** — perhatikan apakah nilai tinggi cenderung mengelompok secara "
        f"geografis (mengindikasikan adanya pola spasial) atau tersebar acak. Pola pengelompokan seperti "
        f"ini adalah alasan utama mengapa pendekatan geographically-weighted (SGWRF) lebih tepat "
        f"dibanding model global biasa."
    )

# ================================================================
# 6. JALANKAN PIPELINE SGWRF (saat tombol ditekan)
# ================================================================

if run_button:
    X, Z, y, coords, d_geo, d_attr, scaler = prepare_matrices(df, x_cols, y_col, lat_col, lon_col)

    prog = st.progress(0.0, text="Memulai optimasi bandwidth...")
    def cb_bw(frac, text):
        prog.progress(min(frac, 1.0), text=text)
    best_bw, bw_results, elapsed_bw = optimize_bandwidth(X, y, d_geo, d_attr, cfg, progress_cb=cb_bw)
    prog.progress(1.0, text=f"Optimasi bandwidth selesai ({elapsed_bw:.1f}s)")

    W, Wg, Wa, bg_local, ba_local = combined_weights(d_geo, d_attr, *best_bw)

    prog2 = st.progress(0.0, text="Optimasi hyperparameter Random Forest...")
    def cb_rf(frac, text):
        prog2.progress(min(frac, 1.0), text=text)
    best_rf, rf_results = optimize_rf(X, y, W, cfg, RF_PARAM_GRID, progress_cb=cb_rf)
    prog2.progress(1.0, text="Optimasi Random Forest selesai")

    prog3 = st.progress(0.0, text="Melatih model lokal SGWRF per titik...")
    def cb_local(frac, text):
        prog3.progress(min(frac, 1.0), text=text)
    pred, local_r2, local_mae, local_rmse, importances = train_local_models(
        X, y, W, df, name_col, x_cols, x_labels, best_rf, cfg, progress_cb=cb_local
    )
    prog3.progress(1.0, text="Model lokal selesai")

    overall = metrics_dict(y, pred)
    intercepts, coefs = local_regression_diagnostic(X, y, W)
    results = build_results(df, id_col, name_col, lat_col, lon_col, y_col, x_cols, x_labels,
                             y, pred, local_r2, local_mae, local_rmse, importances, bg_local, ba_local)

    baseline_results = None
    if run_baseline_flag:
        prog4 = st.progress(0.0, text="Menjalankan model baseline pembanding...")
        def cb_base(frac, text):
            prog4.progress(min(frac, 1.0), text=text)
        baseline_results = run_baselines(X, y, d_geo, d_attr, best_bw, cfg, progress_cb=cb_base)
        prog4.progress(1.0, text="Baseline selesai")

    st.session_state["sgwrf"] = dict(
        results=results, bw_results=bw_results, rf_results=rf_results, best_bw=best_bw,
        best_rf=best_rf, overall=overall, importances=importances, intercepts=intercepts,
        coefs=coefs, baseline_results=baseline_results, X=X, y=y, x_cols=x_cols,
        x_labels=x_labels, name_col=name_col, id_col=id_col,
    )
    st.success("✅ Analisis SGWRF selesai! Lihat hasil pada tab lainnya.")

# ================================================================
# 7. TAMPILKAN HASIL (jika sudah pernah dijalankan)
# ================================================================

state = st.session_state.get("sgwrf")

if state is None:
    for t in (tab_bw, tab_rf, tab_local, tab_baseline, tab_download):
        with t:
            st.info("⬅️ Atur parameter di sidebar lalu tekan **Jalankan Analisis SGWRF** untuk melihat hasil di sini.")
else:
    results = state["results"]
    bw_results = state["bw_results"]
    rf_results = state["rf_results"]
    best_bw = state["best_bw"]
    best_rf = state["best_rf"]
    overall = state["overall"]
    importances = state["importances"]
    intercepts, coefs = state["intercepts"], state["coefs"]
    baseline_results = state["baseline_results"]

    # -------- peta hasil (tambahan di tab_map) --------
    with tab_map:
        st.markdown("---")
        st.subheader("Peta Hasil Model SGWRF")
        m1, m2 = st.columns(2)
        with m1:
            fig_pred = make_continuous_map(results, lat_col, lon_col, "pred_sgwrf", name_col, id_col,
                                            "Prediksi PM/Target Lokal — SGWRF", colorscale="YlOrRd", boundary=boundary)
            st.plotly_chart(fig_pred, use_container_width=True)
            note(
                f"Ini adalah nilai **{y_col} hasil prediksi model SGWRF** di tiap titik (bukan data "
                f"aktual). Bandingkan pola warnanya dengan peta 'Sebaran Titik & Nilai {y_col}' di atas — "
                f"jika polanya mirip, model berhasil menangkap pola spasial data aktual dengan baik."
            )
        with m2:
            fig_resid = make_continuous_map(results, lat_col, lon_col, "residual", name_col, id_col,
                                             "Residual (Aktual − Prediksi)", colorscale="RdBu", boundary=boundary)
            st.plotly_chart(fig_resid, use_container_width=True)
            n_over = int((results["residual"] < 0).sum())
            n_under = int((results["residual"] > 0).sum())
            note(
                f"Residual = nilai aktual dikurangi prediksi. Warna **biru** berarti model **terlalu tinggi "
                f"menaksir (overestimate)**, warna **merah** berarti model **terlalu rendah menaksir "
                f"(underestimate)**. Saat ini {n_over} titik overestimate dan {n_under} titik underestimate. "
                f"Titik dengan warna sangat pekat layak dicek lebih lanjut — bisa jadi outlier data atau "
                f"area dengan dinamika lokal yang belum tertangkap kovariat yang ada."
            )

        m3, m4 = st.columns(2)
        with m3:
            fig_dom = make_categorical_map(results, lat_col, lon_col, "dominant_variable", name_col, id_col,
                                            "Variabel Dominan Lokal SGWRF", boundary=boundary)
            st.plotly_chart(fig_dom, use_container_width=True)
            n_dom_vars = results["dominant_variable"].nunique()
            note(
                f"Setiap titik diwarnai menurut kovariat **paling berpengaruh secara lokal** di titik "
                f"tersebut (berdasarkan permutation importance model RF lokal). Saat ini teridentifikasi "
                f"**{n_dom_vars} variabel dominan berbeda** di antara {len(results)} titik. Klik nama "
                f"variabel pada legenda untuk menyalakan/mematikan tampilannya. Semakin beragam variabel "
                f"dominannya, semakin kuat bukti **heterogenitas spasial** — inti yang membedakan SGWRF "
                f"dari model global (misalnya RF biasa) yang mengasumsikan satu set pengaruh yang sama "
                f"untuk semua lokasi."
            )
        with m4:
            fig_strength = make_continuous_map(results, lat_col, lon_col, "dominant_importance", name_col, id_col,
                                                "Kekuatan Variabel Dominan per Titik", colorscale="Viridis", boundary=boundary)
            st.plotly_chart(fig_strength, use_container_width=True)
            note(
                f"Menunjukkan **seberapa dominan** variabel utama tersebut dibanding kovariat lain di "
                f"titik yang sama (dalam %). Warna cerah/kuning = satu variabel sangat mendominasi "
                f"(pengaruh kovariat lain kecil); warna gelap/ungu = pengaruh lebih terbagi rata antar "
                f"beberapa kovariat di titik tersebut."
            )

    # -------- tab bandwidth --------
    with tab_bw:
        st.subheader("Hasil Optimasi Bandwidth Adaptif (CV)")
        b1, b2, b3 = st.columns(3)
        b1.metric("k geografis optimum", best_bw[0])
        b2.metric("k atribut optimum", best_bw[1])
        b3.metric("RMSE CV minimum", f"{bw_results['RMSE_CV'].min():.4f}")

        pivot = bw_results.pivot(index="k_geo", columns="k_attr", values="RMSE_CV")
        fig_heat = px.imshow(pivot, color_continuous_scale="Viridis", aspect="auto",
                              labels=dict(x="k atribut", y="k geografis", color="RMSE CV"),
                              title="Peta Panas RMSE-CV untuk Kombinasi Bandwidth")
        fig_heat.update_traces(hovertemplate="k_atribut=%{x}<br>k_geo=%{y}<br>RMSE=%{z:.4f}<extra></extra>")
        st.plotly_chart(fig_heat, use_container_width=True)
        note(
            f"Setiap sel adalah RMSE hasil validasi silang **leave-target-out** untuk satu kombinasi "
            f"bandwidth. Warna **gelap/ungu** = error kecil (kombinasi lebih baik), warna **terang/kuning** "
            f"= error besar. Kombinasi terbaik yang dipilih otomatis: **k_geo = {best_bw[0]}** (jumlah "
            f"tetangga terdekat secara geografis) dan **k_atribut = {best_bw[1]}** (jumlah tetangga "
            f"terdekat secara karakteristik/atribut). k kecil membuat model sangat lokal (risiko overfit "
            f"pada data sedikit), k besar membuat model mendekati model global (kurang menangkap "
            f"heterogenitas spasial)."
        )

        st.dataframe(bw_results, use_container_width=True, height=300)

    # -------- tab RF --------
    with tab_rf:
        st.subheader("Optimasi Hyperparameter Random Forest")
        st.json(best_rf)
        fig_rf = px.bar(rf_results.reset_index(), x="index", y="RMSE_CV",
                         hover_data=["n_estimators", "max_features", "min_samples_leaf"],
                         title="RMSE-CV per Konfigurasi Random Forest",
                         labels={"index": "Konfigurasi"})
        st.plotly_chart(fig_rf, use_container_width=True)
        note(
            f"Setiap batang adalah satu kombinasi hyperparameter Random Forest (jumlah pohon, "
            f"`max_features`, `min_samples_leaf`) yang diuji lewat validasi silang lokal. Batang "
            f"**terendah** dipilih sebagai konfigurasi final: `max_features={best_rf.get('max_features')}`, "
            f"`min_samples_leaf={best_rf.get('min_samples_leaf')}`. Konfigurasi ini lalu dipakai untuk "
            f"melatih model lokal final dengan jumlah pohon yang lebih besar (lihat pengaturan sidebar) "
            f"agar hasil akhir lebih stabil."
        )
        st.dataframe(rf_results, use_container_width=True)

    # -------- tab model lokal --------
    with tab_local:
        st.subheader("Kinerja Global Model SGWRF")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("RMSE", f"{overall['RMSE']:.4f}")
        g2.metric("MAE", f"{overall['MAE']:.4f}")
        g3.metric("MAPE", f"{overall['MAPE']:.2f}%")
        g4.metric("R²", f"{overall['R2']:.4f}")
        kualitas = "sangat baik" if overall["R2"] >= 0.8 else ("cukup baik" if overall["R2"] >= 0.6 else
                    ("moderat" if overall["R2"] >= 0.4 else "masih lemah — pertimbangkan menambah kovariat atau titik data"))
        note(
            f"**R² = {overall['R2']:.4f}** berarti model SGWRF menjelaskan sekitar "
            f"**{max(overall['R2'], 0)*100:.1f}%** variasi {y_col} antar titik (kualitas model tergolong "
            f"**{kualitas}**). **RMSE** dan **MAE** menunjukkan rata-rata besar kesalahan prediksi dalam "
            f"satuan asli {y_col}; **MAPE** menyatakannya dalam persentase relatif terhadap nilai aktual."
        )

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_avp = px.scatter(results, x=y_col, y="pred_sgwrf", hover_name=name_col,
                                  title="Aktual vs Prediksi", trendline="ols")
            vmin = float(min(results[y_col].min(), results["pred_sgwrf"].min()))
            vmax = float(max(results[y_col].max(), results["pred_sgwrf"].max()))
            fig_avp.add_shape(type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
                               line=dict(color="gray", dash="dash"))
            st.plotly_chart(fig_avp, use_container_width=True)
            note(
                "Garis putus-putus abu-abu adalah garis ideal (prediksi = aktual). Makin dekat titik-titik "
                "ke garis ini, makin akurat model. Titik yang jauh dari garis adalah lokasi dengan error "
                "prediksi besar — cek apakah lokasi tersebut juga muncul menonjol pada peta residual."
            )
        with cc2:
            fig_res_hist = px.histogram(results, x="residual", nbins=15, title="Distribusi Residual")
            st.plotly_chart(fig_res_hist, use_container_width=True)
            mean_resid = float(results["residual"].mean())
            note(
                f"Idealnya residual tersebar **di sekitar 0** tanpa bias sistematis. Rata-rata residual "
                f"saat ini **{mean_resid:.3f}** — nilai mendekati 0 menandakan model tidak bias secara "
                f"konsisten ke arah over/underestimate. Sebaran yang melebar menandakan variasi error "
                f"cukup besar antar titik."
            )

        st.subheader("Hasil Model Lokal per Titik")
        show_cols = ["point_no", name_col, y_col, "pred_sgwrf", "residual",
                     "dominant_variable", "dominant_importance", "local_train_R2",
                     "bandwidth_geo_k", "bandwidth_attr_k"]
        st.dataframe(results[show_cols], use_container_width=True, height=350)

        st.subheader("Rata-rata Variable Importance (Seluruh Lokasi)")
        mean_imp = importances.mean(axis=0)
        imp_df = pd.DataFrame({
            "Variabel": [state["x_labels"][c] for c in state["x_cols"]],
            "Importance (%)": mean_imp * 100,
        }).sort_values("Importance (%)", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance (%)", y="Variabel", orientation="h",
                          title="Rata-rata Kepentingan Variabel (Permutation Importance)",
                          color="Importance (%)", color_continuous_scale="Blues")
        st.plotly_chart(fig_imp, use_container_width=True)
        top_global = imp_df.iloc[-1]
        note(
            f"Ini **rata-rata** kepentingan tiap kovariat di seluruh titik (bukan per-lokasi). Secara "
            f"keseluruhan, **{top_global['Variabel']}** paling berpengaruh terhadap {y_col} "
            f"({top_global['Importance (%)']:.1f}%). Namun karena SGWRF bersifat lokal, urutan ini bisa "
            f"berbeda-beda di tiap titik — lihat peta panas di bawah dan peta 'Variabel Dominan Lokal "
            f"SGWRF' untuk detail per lokasi."
        )

        st.subheader("Peta Panas Variable Importance Lokal (Titik × Variabel)")
        vi_cols = [f"VI_{c}" for c in state["x_cols"]]
        vi_matrix = results[vi_cols].to_numpy() * 100
        fig_vi_heat = px.imshow(
            vi_matrix, x=[state["x_labels"][c] for c in state["x_cols"]],
            y=[f"#{n}" for n in results["point_no"]],
            color_continuous_scale="Magma", aspect="auto",
            labels=dict(x="Variabel", y="Titik", color="Importance (%)"),
            title="Variable Importance Lokal per Titik (%)",
        )
        st.plotly_chart(fig_vi_heat, use_container_width=True)
        note(
            "Setiap baris adalah satu titik, setiap kolom satu kovariat. Warna **terang/kuning** berarti "
            "kovariat tersebut sangat penting **di titik itu saja**; warna **gelap** berarti kurang "
            "berpengaruh di titik itu. Jika satu kolom terang merata di semua baris, variabel itu penting "
            "secara global. Jika warna terang tersebar tidak merata (berbeda-beda tiap baris), itu bukti "
            "kuat heterogenitas spasial — kekuatan utama pendekatan SGWRF dibanding model global."
        )

        st.subheader("Variabel Dominan per Titik")
        for _, r in results.iterrows():
            st.markdown(f"- **Titik {int(r['point_no'])} — {r[name_col]}** → "
                        f"{r['dominant_variable']} ({r['dominant_importance']*100:.2f}%)")

        with st.expander("📐 Persamaan Regresi Lokal Diagnostik (Weighted Ridge — bukan koefisien SGWRF)"):
            st.markdown(
                '<div class="sgwrf-note">Persamaan ini adalah <b>Weighted Ridge lokal</b> untuk interpretasi '
                'tambahan. Model SGWRF utamanya tetap Random Forest dan tidak memiliki koefisien beta linear. '
                'Gunakan permutation importance sebagai ukuran kepentingan lokal SGWRF.</div>',
                unsafe_allow_html=True,
            )
            for i in range(len(results)):
                terms = []
                for j, c in enumerate(state["x_cols"]):
                    sign = "+" if coefs[i, j] >= 0 else "-"
                    terms.append(f" {sign} {abs(coefs[i, j]):.4f}·Z({state['x_labels'][c]})")
                eq = f"Ŷ_{i+1} = {intercepts[i]:.4f}" + "".join(terms)
                st.markdown(f"**[{i+1:02d}] {results.loc[i, name_col]}**")
                st.code(eq, language="text")

    # -------- tab baseline --------
    with tab_baseline:
        if baseline_results is None:
            st.info("Model baseline tidak dijalankan. Aktifkan opsi di sidebar lalu jalankan ulang analisis.")
        else:
            st.subheader("Perbandingan SGWRF vs Model Baseline")
            st.dataframe(baseline_results, use_container_width=True)

            fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("RMSE (lebih rendah lebih baik)", "R² (lebih tinggi lebih baik)"))
            fig_bar.add_trace(go.Bar(x=baseline_results["Model"], y=baseline_results["RMSE"],
                                      marker_color="#e74c3c", name="RMSE"), row=1, col=1)
            fig_bar.add_trace(go.Bar(x=baseline_results["Model"], y=baseline_results["R2"],
                                      marker_color="#27ae60", name="R2"), row=1, col=2)
            fig_bar.update_layout(height=430, showlegend=False, title="Ringkasan Kinerja Model")
            st.plotly_chart(fig_bar, use_container_width=True)
            sgwrf_rank = int((baseline_results["RMSE"] < overall["RMSE"]).sum())
            posisi = "lebih baik dari seluruh" if sgwrf_rank == 0 else f"lebih baik dari {len(baseline_results)-sgwrf_rank} dari {len(baseline_results)}"
            note(
                f"Model utama **SGWRF** memiliki RMSE **{overall['RMSE']:.4f}** dan R² **{overall['R2']:.4f}** "
                f"(lihat tab Model Lokal SGWRF) — dibandingkan baseline di atas, SGWRF saat ini {posisi} "
                f"model pembanding. Jika SGWRF **tidak** mengungguli GWRF/RF_global secara meyakinkan, "
                f"pertimbangkan menambah kovariat relevan, menambah titik observasi, atau memperluas "
                f"rentang pencarian bandwidth di sidebar."
            )

            st.markdown(
                '<div class="sgwrf-note">Baseline: <b>RF_global</b> (RF tanpa pembobotan lokal), '
                '<b>GWR_Ridge</b> (Ridge terboboti geografis), <b>GWRF</b> (RF terboboti geografis saja), '
                '<b>SGWR_Ridge</b> (Ridge terboboti geografis × atribut). Bandingkan dengan model utama '
                'SGWRF (RF terboboti geografis × atribut) pada tab Model Lokal SGWRF.</div>',
                unsafe_allow_html=True,
            )

    # -------- tab unduh --------
    with tab_download:
        st.subheader("Unduh Seluruh Hasil Analisis")

        def to_csv_bytes(d):
            return d.to_csv(index=False).encode("utf-8")

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("⬇️ Hasil Lokal SGWRF (.csv)", to_csv_bytes(results),
                                "hasil_lokal_sgwrf.csv", "text/csv", use_container_width=True)
            st.download_button("⬇️ Optimasi Bandwidth (.csv)", to_csv_bytes(bw_results),
                                "hasil_optimasi_bandwidth.csv", "text/csv", use_container_width=True)
        with dl2:
            st.download_button("⬇️ Optimasi Random Forest (.csv)", to_csv_bytes(rf_results),
                                "hasil_optimasi_rf.csv", "text/csv", use_container_width=True)
            vi_export = pd.DataFrame(importances, columns=state["x_cols"])
            vi_export.insert(0, "point_no", results["point_no"].to_numpy())
            st.download_button("⬇️ Local Variable Importance (.csv)", to_csv_bytes(vi_export),
                                "local_variable_importance.csv", "text/csv", use_container_width=True)
        with dl3:
            if baseline_results is not None:
                st.download_button("⬇️ Perbandingan Model Baseline (.csv)", to_csv_bytes(baseline_results),
                                    "perbandingan_model.csv", "text/csv", use_container_width=True)
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                results.to_excel(writer, sheet_name="hasil_lokal_sgwrf", index=False)
                bw_results.to_excel(writer, sheet_name="optimasi_bandwidth", index=False)
                rf_results.to_excel(writer, sheet_name="optimasi_rf", index=False)
                vi_export.to_excel(writer, sheet_name="variable_importance", index=False)
                if baseline_results is not None:
                    baseline_results.to_excel(writer, sheet_name="perbandingan_baseline", index=False)
            st.download_button("⬇️ Semua Hasil (Excel, multi-sheet)", excel_buf.getvalue(),
                                "hasil_sgwrf_lengkap.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)

st.markdown("---")
st.caption("Dashboard SGWRF — dibangun mengikuti pipeline: preprocessing & standarisasi → "
           "jarak geografis (haversine) & atribut (Euclidean) → kernel Gaussian ganda → "
           "optimasi bandwidth adaptif (CV) → optimasi hyperparameter RF → model lokal RF per titik → "
           "permutation importance lokal → model baseline pembanding.")
