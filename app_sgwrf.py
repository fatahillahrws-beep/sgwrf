# ================================================================
# SGWRF INTERACTIVE DASHBOARD
# Semi-parametric Geographically Weighted Random Forest
# ----------------------------------------------------------------
# Mengikuti persamaan pada proposal (2.3)-(2.38):
#   (2.3)-(2.5)   Standarisasi kovariat (Z-score)
#   (2.10)        Jarak geografis Euclidean d_ij^G
#   (2.11)        Kernel Gaussian geografis w_ij^G = exp[-(d^G)^2 / 2b_g^2]
#   (2.18)-(2.19) Jarak atribut d_ij = mean_k |z_ik - z_jk|
#   (2.20)        Similarity weight w_ij^S = exp(-d_ij^2)
#   (2.24)        W_GS = alpha * W_G + gamma * W_S,  gamma = 1 - alpha
#   (2.28)-(2.33) k_geo, alpha/gamma, DAN hyperparameter RF dicari SEKALIGUS
#                 (joint search) berdasarkan RMSE LOOCV; AICc hanya
#                 diagnostik tambahan (ENP = trace(S), proxy), BUKAN
#                 kriteria pemilihan
#   (2.35)        Treewise permutation variable importance
#   (2.36)-(2.38) RMSE, MAPE, R2 (evaluasi utama memakai LOOCV)
#
# Model baseline (semua LOOCV):
#   - RF (global)  : RF_PARAM_GRID di-tuning ulang tanpa bobot spasial
#   - GWR          : hanya k_geo yang dituning (regresi linear terboboti W_G)
#   - GWRF         : k_geo x RF_PARAM_GRID dituning bersama (RF terboboti W_G)
#   - SGWR         : k_geo x alpha dituning bersama (regresi linear terboboti W_GS)
#   - SGWRF        : memakai hasil pencarian bersama k_geo x alpha x RF
#                    (preds sudah tersedia dari tahap joint search utama,
#                    TIDAK dihitung ulang)
#
# Data TIDAK di-hardcode — pengguna mengunggah file sendiri (xlsx/csv),
# memetakan kolom, mengatur parameter, lalu menjalankan seluruh pipeline.
# Peta & grafik interaktif (Plotly) otomatis menyesuaikan data yang
# diunggah, termasuk saat jumlah titik observasi bertambah.
#
# Jalankan dengan:
#   pip install -r requirements.txt
#   streamlit run sgwrf_dashboard.py
# ================================================================

import io
import json
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Paksa semua grafik Plotly (px & go) memakai template terang secara
# global — mencegah latar chart menjadi hitam/gelap saat Streamlit
# memakai tema Dark (paper/plot background tidak lagi ikut warna gelap
# bawaan, dan font otomatis gelap agar tetap terbaca).
pio.templates.default = "plotly_white"

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
:root, .stApp{
    --sgwrf-primary:#0f4c81;
    --sgwrf-accent:#f39c12;
    --sgwrf-bg:#f5f7fa;
    /* Override variabel tema Streamlit sendiri (dipakai banyak widget
       bawaan) supaya konsisten LIGHT walau pengguna memilih tema
       "Dark" di menu Settings Streamlit atau OS-nya memakai dark mode. */
    --background-color:#ffffff !important;
    --secondary-background-color:#f5f7fa !important;
    --text-color:#1a2733 !important;
    --primary-color:#0f4c81 !important;
}

/* ------------------------------------------------------------
   FIX TEMA GELAP: paksa area utama & sidebar tetap terang + teks
   gelap, apa pun preferensi tema Streamlit/OS pengguna. Beberapa
   selector fallback disertakan karena testid Streamlit bisa berbeda
   antar versi/deployment (mis. Streamlit Community Cloud).
   ------------------------------------------------------------ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
.stApp,
.appview-container,
section.main{
    background-color:#ffffff !important;
}
.main .block-container,
.main .block-container p,
.main .block-container span,
.main .block-container label,
.main .block-container li,
.main .block-container div[data-testid="stMarkdownContainer"]{
    color:#1a2733 !important;
    opacity:1 !important;
}
h1,h2,h3,h4,h5,h6{color:var(--sgwrf-primary) !important;opacity:1 !important;}
div[data-testid="stMetric"] *{
    color:#1a2733 !important;
}
div[data-testid="stMetricValue"]{
    color:var(--sgwrf-primary) !important;
    font-weight:700;
}
/* Kartu grafik Plotly: paksa latar putih supaya tidak ada kotak hitam
   di sekitar judul/chart saat tema Dark aktif. */
div[data-testid="stPlotlyChart"]{
    background:#ffffff !important;border-radius:10px;overflow:hidden;
}

.main .block-container{padding-top:1.4rem;padding-bottom:2.5rem;max-width:1400px;}
div[data-testid="stMetric"]{
    background:white;border:1px solid #e3e8ee;border-radius:12px;
    padding:0.7rem 0.9rem;box-shadow:0 1px 3px rgba(15,76,129,0.08);
}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
    background-color:#0f4c81 !important;
}
/* Teks label, judul, dan keterangan di sidebar dibuat terang agar kontras
   dengan latar biru gelap. */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([data-baseweb] *),
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small{
    color:#eef4fb !important;
    opacity:1 !important;
}
/* Kotak input/dropdown/file-uploader tetap berlatar terang, sehingga
   teks di DALAM kotak tersebut harus gelap agar terbaca jelas. */
section[data-testid="stSidebar"] div[data-baseweb="select"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="popover"],
section[data-testid="stSidebar"] div[role="listbox"],
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"]{
    background:#ffffff !important;border-radius:8px;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-baseweb="popover"] *,
section[data-testid="stSidebar"] div[role="listbox"] *,
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzoneInstructions"] *,
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] *,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea{
    color:#1a2733 !important;
    opacity:1 !important;
}
/* Placeholder / teks bantu ("Choose an option", nama file terunggah, dsb.) */
section[data-testid="stSidebar"] div[data-baseweb="select"] div[class*="placeholder"]{
    color:#5a6b7a !important;
}
/* Tag terpilih pada multiselect (mis. daftar kovariat X) */
section[data-testid="stSidebar"] span[data-baseweb="tag"]{
    background-color:var(--sgwrf-accent) !important;
}
section[data-testid="stSidebar"] span[data-baseweb="tag"] *{
    color:#20242b !important;
}
section[data-testid="stSidebar"] .stButton>button{
    background:var(--sgwrf-accent);color:#20242b !important;font-weight:700;
    border:none;border-radius:8px;
}
section[data-testid="stSidebar"] [data-testid="stAlert"],
section[data-testid="stSidebar"] [data-testid="stAlert"] *{
    color:#1a2733 !important;
}

/* ---------------- TABS ---------------- */
.stTabs [data-baseweb="tab-list"]{gap:4px;}
.stTabs [data-baseweb="tab"]{
    background:#eef2f7 !important;border-radius:8px 8px 0 0;padding:8px 16px;font-weight:600;
}
.stTabs [data-baseweb="tab"] *{
    color:#1a2733 !important;
    opacity:1 !important;
}
.stTabs [aria-selected="true"]{background:var(--sgwrf-primary) !important;}
.stTabs [aria-selected="true"] *{color:#ffffff !important;}

/* ---------------- KOTAK CUSTOM (spesifisitas tinggi agar selalu menang) ---------------- */
div.sgwrf-banner{
    background:linear-gradient(90deg,#0f4c81,#1c7ed6) !important;
    color:white !important;padding:1.1rem 1.4rem;border-radius:14px;margin-bottom:1.1rem;
}
div.sgwrf-banner h1{color:#ffffff !important;margin:0;font-size:1.55rem;opacity:1 !important;}
div.sgwrf-banner p{color:#dbe9fb !important;margin:0.2rem 0 0 0;font-size:0.92rem;opacity:1 !important;}
div.sgwrf-note{
    background:#fff8e6 !important;border-left:4px solid var(--sgwrf-accent);
    padding:0.6rem 0.9rem;border-radius:6px;font-size:0.88rem;color:#5c4813 !important;
}
div.sgwrf-interpret{
    background:#eef6ff !important;border-left:4px solid #1c7ed6;color:#123a5c !important;
    padding:0.55rem 0.9rem;border-radius:6px;font-size:0.87rem;margin:0.35rem 0 1.1rem 0;
}
div.sgwrf-interpret b{color:#0f4c81 !important;}
div.sgwrf-eq{
    background:#f4f6fb !important;border:1px dashed #9db3c9;border-radius:8px;
    padding:0.5rem 0.8rem;font-family:"Courier New",monospace;font-size:0.85rem;color:#0f4c81 !important;
}
div.sgwrf-warn{
    background:#fdecea !important;border-left:4px solid #e74c3c;color:#7a1f14 !important;
    padding:0.6rem 0.9rem;border-radius:6px;font-size:0.88rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="sgwrf-banner">
    <h1>🛰️ SGWRF Interactive Dashboard</h1>
    <p>Semi-parametric Geographically Weighted Random Forest — jarak geografis Euclidean +
    kernel Gaussian adaptif, jarak atribut (mean |Z<sub>i</sub>-Z<sub>j</sub>|) + similarity weight,
    kombinasi aditif W<sub>GS</sub> = α·W<sub>G</sub> + γ·W<sub>S</sub> dengan k<sub>geo</sub>, α,
    DAN hyperparameter Random Forest dicari SEKALIGUS lewat LOOCV (AICc sebagai diagnostik
    tambahan), plus treewise permutation importance. Unggah data Anda sendiri — peta &amp;
    grafik menyesuaikan otomatis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# 1. FUNGSI INTI — sesuai Persamaan (2.3)-(2.38) pada proposal
# ================================================================

def geographic_distance_euclidean(coords):
    """Persamaan (2.10): d_ij^G = sqrt[(u_i-u_j)^2 + (v_i-v_j)^2]."""
    return cdist(coords, coords, metric="euclidean")


def gaussian_kernel(distance, bandwidth):
    """Persamaan (2.11): w_ij^G = exp[-(d_ij^G)^2 / (2 b_g^2)]."""
    bandwidth = np.maximum(np.asarray(bandwidth, dtype=float), 1e-12)
    return np.exp(-(distance ** 2) / (2.0 * bandwidth ** 2))


def adaptive_bandwidth_matrix(d, k):
    """Bandwidth geografis adaptif: b_g(i) = jarak ke tetangga ke-k terdekat."""
    d = np.array(d, copy=True, dtype=float)
    np.fill_diagonal(d, np.inf)
    k = int(np.clip(k, 1, d.shape[0] - 1))
    kth = np.partition(d, kth=k - 1, axis=1)[:, k - 1]
    return np.maximum(kth, 1e-12)


def attribute_pairwise_distance(Z):
    """Persamaan (2.18)-(2.19): d_ijk = |z_ik - z_jk|, d_ij = mean_k d_ijk."""
    d_pairwise = np.abs(Z[:, None, :] - Z[None, :, :])
    return d_pairwise.mean(axis=2)


def similarity_weight(d_attr):
    """Persamaan (2.20): w_ij^S = exp(-d_ij^2)."""
    return np.exp(-(d_attr ** 2))


def build_weight_components(d_geo, d_attr, k_geo, alpha):
    """Persamaan (2.11), (2.20), (2.24):
    W_GS = alpha * W_G + gamma * W_S,  gamma = 1 - alpha (ADITIF, tanpa
    bandwidth atribut, tanpa normalisasi baris)."""
    bg_local = adaptive_bandwidth_matrix(d_geo, k_geo)
    Wg = gaussian_kernel(d_geo, bg_local[:, None])
    Ws = similarity_weight(d_attr)
    gamma = 1.0 - alpha
    Wgs = alpha * Wg + gamma * Ws
    return Wgs, Wg, Ws, bg_local, gamma


def prepare_matrices(df, x_cols, y_col, lat_col, lon_col):
    X = df[x_cols].to_numpy(float)
    y = df[y_col].to_numpy(float)
    coords = df[[lat_col, lon_col]].to_numpy(float)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)                        # (2.3)-(2.5)
    d_geo = geographic_distance_euclidean(coords)        # (2.10)
    d_attr = attribute_pairwise_distance(Z)              # (2.18)-(2.19)
    return X, Z, y, coords, d_geo, d_attr, scaler


def candidate_geo_bandwidths(n, min_k, max_k):
    max_k = (n - 1) if max_k is None else min(max_k, n - 1)
    min_k = max(2, min(min_k, max_k))
    return list(range(min_k, max_k + 1))


# ---------------- 1a. Normalisasi parameter RF (defensif, meniru script) ----------------

def normalize_max_features(value, n_features):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"sqrt", "log2"}:
            return v
        if v in {"none", "null", "all", "auto"}:
            return None
        try:
            num = float(v)
            if 0.0 < num <= 1.0:
                return float(num)
            if num.is_integer() and int(num) >= 1:
                return int(num)
        except ValueError:
            pass
        return "sqrt"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        value = int(value)
        return min(max(value, 1), n_features)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not (0.0 < value <= 1.0):
            return 1.0
        return value
    return "sqrt"


def normalize_rf_params(params, n_features):
    out = dict(params)
    out["n_estimators"] = int(out.get("n_estimators", 100))
    out["min_samples_leaf"] = int(out.get("min_samples_leaf", 1))
    md = out.get("max_depth", None)
    out["max_depth"] = int(md) if md is not None else None
    out["max_features"] = normalize_max_features(out.get("max_features", "sqrt"), n_features)
    return out


def rf_fit_predict(X_train, y_train, X_test, sample_weight, params, random_state):
    params = normalize_rf_params(params, X_train.shape[1])
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        max_depth=params["max_depth"],
        random_state=int(random_state),
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, model.predict(X_test)


def local_prediction_for_target(i, X, y, W, params, random_state, leave_target_out=True):
    w = np.asarray(W[i], dtype=float).copy()
    if leave_target_out:
        w[i] = 0.0
    train_mask = w > 1e-12
    if train_mask.sum() < 4:
        train_mask[:] = True
        if leave_target_out:
            train_mask[i] = False
    model, pred = rf_fit_predict(X[train_mask], y[train_mask], X[i:i + 1], w[train_mask], params, random_state)
    return model, float(pred[0]), train_mask, w


def predict_local_rf_cv(X, y, W, params, base_seed, seed_offset):
    """LOOCV untuk Random Forest lokal berbobot — dipakai di seluruh tahap
    optimasi/seleksi maupun baseline (GWRF)."""
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        _, pred, _, _ = local_prediction_for_target(i, X, y, W, params, base_seed + seed_offset + i, True)
        preds[i] = pred
    return preds


def predict_weighted_linear_cv(X, y, W):
    """Baseline GWR/SGWR: regresi linear terboboti dengan LOOCV."""
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        w = W[i].copy()
        w[i] = 0.0
        mask = w > 1e-12
        if mask.sum() < 3:
            mask[:] = True
            mask[i] = False
        model = LinearRegression()
        model.fit(X[mask], y[mask], sample_weight=w[mask])
        preds[i] = float(model.predict(X[i:i + 1])[0])
    return preds


def predict_global_rf_cv(X, y, params, base_seed, seed_offset):
    """LOOCV Random Forest GLOBAL (tanpa pembobotan spasial) — baseline RF."""
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        pnorm = normalize_rf_params(params, X.shape[1])
        model = RandomForestRegressor(
            n_estimators=pnorm["n_estimators"], max_features=pnorm["max_features"],
            min_samples_leaf=pnorm["min_samples_leaf"], max_depth=pnorm["max_depth"],
            random_state=base_seed + seed_offset + i, n_jobs=-1,
        )
        model.fit(X[mask], y[mask])
        preds[i] = float(model.predict(X[i:i + 1])[0])
    return preds


# ---------------- 1b. AICc (diagnostik tambahan, bukan kriteria seleksi) ----------------

def normalized_weight_hat_proxy(W):
    row_sum = W.sum(axis=1, keepdims=True)
    return W / np.maximum(row_sum, 1e-15)


def effective_number_parameters(W):
    return float(np.trace(normalized_weight_hat_proxy(W)))


def aicc_from_rss(rss, n, enp):
    """Persamaan (2.29): AICc = 2n ln(RSS/n) + n ln(2π) + n + 2(ENP+1)(ENP+2)/(n-ENP-2)."""
    rss = max(float(rss), 1e-15)
    denom = n - enp - 2
    if denom <= 0:
        return np.inf
    return (2.0 * n * np.log(rss / n) + n * np.log(2.0 * np.pi) + n
            + (2.0 * (enp + 1.0) * (enp + 2.0)) / denom)


def metrics_dict(y, pred):
    """Persamaan (2.36)-(2.38) + RSS (2.30)."""
    err = y - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(y) < 1e-12, np.nan, np.abs(y))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    r2 = float(r2_score(y, pred))
    rss = float(np.sum(err ** 2))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "RSS": rss}


def cv_score_weight_matrix(W, X, y, params, base_seed, seed_offset):
    preds = predict_local_rf_cv(X, y, W, params, base_seed, seed_offset)
    m = metrics_dict(y, preds)
    return m["RSS"], m["RMSE"], preds, m


# ---------------- 1c. Pencarian BERSAMA k_geo x alpha x RF (kriteria utama: RMSE-LOOCV) ----------------

def joint_optimize_sgwrf(X, y, d_geo, d_attr, cfg, rf_param_grid, progress_cb=None):
    """Pencarian ekspansif k_geo x alpha x hyperparameter RF (Persamaan
    2.28-2.33). Kriteria utama = RMSE LOOCV; AICc dihitung sebagai
    diagnostik tambahan saja, BUKAN kriteria pemilihan utama. Hanya
    prediksi kombinasi TERBAIK yang disimpan (mengikuti script asli),
    bukan seluruh kombinasi, agar hemat memori."""
    k_candidates = candidate_geo_bandwidths(len(y), cfg["min_k_geo"], cfg["max_k_geo"])
    alpha_candidates = np.round(np.linspace(cfg["alpha_min"], cfg["alpha_max"], cfg["n_alpha"]), 4)

    rows = []
    best = None
    t0 = time.time()
    total = len(rf_param_grid) * len(k_candidates) * len(alpha_candidates)
    iteration = 0

    for rf_id, base_params in enumerate(rf_param_grid, 1):
        params = normalize_rf_params(base_params, X.shape[1])
        for k_geo in k_candidates:
            for alpha in alpha_candidates:
                iteration += 1
                alpha = float(alpha)
                W, Wg, Ws, bg_local, gamma = build_weight_components(d_geo, d_attr, k_geo, alpha)
                rss, rmse, preds, m = cv_score_weight_matrix(
                    W, X, y, params, cfg["seed"],
                    rf_id * 100000 + k_geo * 1000 + int(round(alpha * 1000)),
                )
                enp = effective_number_parameters(W)
                aicc = aicc_from_rss(rss, len(y), enp)
                row = {
                    "iteration": iteration, "rf_config": rf_id, "k_geo": int(k_geo),
                    "alpha": alpha, "gamma": float(gamma),
                    "n_estimators": int(params["n_estimators"]), "max_features": params["max_features"],
                    "min_samples_leaf": int(params["min_samples_leaf"]), "max_depth": params["max_depth"],
                    "RSS_CV": rss, "RMSE_CV": rmse, "MAE_CV": m["MAE"], "R2_CV": m["R2"], "MAPE_CV": m["MAPE"],
                    "ENP_proxy": enp, "AICc_diagnostic": aicc,
                }
                rows.append(row)
                if best is None or rmse < best["RMSE_CV"]:
                    best = row.copy()
                    best["preds"] = preds.copy()

                if progress_cb and (iteration % max(1, total // 200) == 0 or iteration == total):
                    progress_cb(iteration / total,
                                f"[{iteration}/{total}] RF#{rf_id} k={k_geo} α={alpha:.3f} — "
                                f"RMSE_CV={rmse:.4f} | terbaik={best['RMSE_CV']:.4f}")

    results = pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)
    return best, results, time.time() - t0


# ---------------- 1d. Model lokal final + treewise importance (2.35) ----------------

def local_treewise_variable_importance(model, X_train, y_train, sample_weight, seed):
    """Persamaan (2.35): VI_ik = (1/B) * sum_b (E_ib,k^perm - E_ib)."""
    n_train, p = X_train.shape
    B = len(model.estimators_)
    if n_train < 2:
        return np.zeros(p)
    w = np.maximum(np.asarray(sample_weight, dtype=float), 0.0)
    if np.sum(w) <= 0:
        w = np.ones(n_train)
    rng = np.random.default_rng(seed)
    importance = np.zeros(p, dtype=float)
    for tree in model.estimators_:
        base_pred = tree.predict(X_train)
        base_err = float(np.average((y_train - base_pred) ** 2, weights=w))
        for k in range(p):
            X_perm = X_train.copy()
            perm_idx = rng.permutation(n_train)
            X_perm[:, k] = X_perm[perm_idx, k]
            perm_pred = tree.predict(X_perm)
            perm_err = float(np.average((y_train - perm_pred) ** 2, weights=w))
            importance[k] += perm_err - base_err
    importance /= max(B, 1)
    importance = np.maximum(importance, 0.0)
    if importance.sum() > 0:
        importance /= importance.sum()
    return importance


def train_local_models(X, y, W, df, name_col, x_cols, x_labels, sgwrf_params, cfg, progress_cb=None):
    """Model final per titik: TIDAK leave-target-out (bobot titik sendiri
    ikut serta bila w_ii > 0). Jumlah pohon FINAL memakai cfg['rf_final_trees']
    (independen dari n_estimators tahap pencarian LOOCV), sedangkan
    max_features/min_samples_leaf/max_depth memakai hasil pencarian bersama."""
    n = len(y)
    p = X.shape[1]
    preds = np.full(n, np.nan)
    local_r2 = np.full(n, np.nan)
    local_mae = np.full(n, np.nan)
    local_rmse = np.full(n, np.nan)
    importances = np.zeros((n, p))

    final_params = normalize_rf_params({
        "n_estimators": cfg["rf_final_trees"],
        "max_features": sgwrf_params.get("max_features", "sqrt"),
        "min_samples_leaf": sgwrf_params.get("min_samples_leaf", 1),
        "max_depth": sgwrf_params.get("max_depth", None),
    }, p)

    for i in range(n):
        w = W[i].copy()
        train_mask = w > 1e-12
        if train_mask.sum() < 4:
            train_mask[:] = True

        model = RandomForestRegressor(
            n_estimators=final_params["n_estimators"], max_features=final_params["max_features"],
            min_samples_leaf=final_params["min_samples_leaf"], max_depth=final_params["max_depth"],
            random_state=cfg["seed"] + i, n_jobs=-1,
        )
        model.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
        preds[i] = float(model.predict(X[i:i + 1])[0])

        train_pred = model.predict(X[train_mask])
        local_rmse[i] = float(np.sqrt(mean_squared_error(y[train_mask], train_pred, sample_weight=w[train_mask])))
        local_mae[i] = float(mean_absolute_error(y[train_mask], train_pred, sample_weight=w[train_mask]))
        try:
            local_r2[i] = float(r2_score(y[train_mask], train_pred, sample_weight=w[train_mask]))
        except Exception:
            local_r2[i] = np.nan

        importances[i, :] = local_treewise_variable_importance(
            model, X[train_mask], y[train_mask], w[train_mask], cfg["seed"] + i
        )

        if progress_cb:
            top_idx = int(np.argmax(importances[i]))
            progress_cb((i + 1) / n, f"Titik {i+1}/{n} — {df.loc[i, name_col]} | dominan: {x_labels[x_cols[top_idx]]}")

    return preds, local_r2, local_mae, local_rmse, importances


# ---------------- 1e. Baseline: RF, GWR, GWRF, SGWR (masing-masing dituning sendiri), SGWRF (reuse) ----------------

def optimize_global_rf(X, y, cfg, rf_param_grid, progress_cb=None):
    """RF (global): RF_PARAM_GRID di-tuning ulang TANPA bobot spasial, LOOCV."""
    rows, best = [], None
    total = len(rf_param_grid)
    for j, params in enumerate(rf_param_grid, 1):
        preds = predict_global_rf_cv(X, y, params, cfg["seed"], 500000 + j * 1000)
        m = metrics_dict(y, preds)
        row = {"rf_config": j, **normalize_rf_params(params, X.shape[1]), **{f"{k}_CV": v for k, v in m.items()}}
        rows.append(row)
        if best is None or m["RMSE"] < best["RMSE_CV"]:
            best = row.copy()
            best["preds"] = preds.copy()
            best["RMSE_CV"] = m["RMSE"]
        if progress_cb:
            progress_cb(j / total, f"RF global config {j}/{total} — RMSE_CV={m['RMSE']:.4f}")
    return best, pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)


def tune_gwr_k(X, y, d_geo, k_candidates, progress_cb=None):
    """GWR: hanya k_geo yang dituning (regresi linear terboboti W_G), LOOCV."""
    rows, best = [], None
    total = len(k_candidates)
    for idx, k in enumerate(k_candidates, 1):
        bg = adaptive_bandwidth_matrix(d_geo, k)
        Wg = gaussian_kernel(d_geo, bg[:, None])
        pred = predict_weighted_linear_cv(X, y, Wg)
        m = metrics_dict(y, pred)
        row = {"k_geo": int(k), **{f"{a}_CV": b for a, b in m.items()}}
        rows.append(row)
        if best is None or m["RMSE"] < best["RMSE_CV"]:
            best = row.copy()
            best["preds"] = pred.copy()
            best["RMSE_CV"] = m["RMSE"]
        if progress_cb:
            progress_cb(idx / total, f"GWR k_geo={k} ({idx}/{total}) — RMSE_CV={m['RMSE']:.4f}")
    return best, pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)


def tune_gwrf(X, y, d_geo, k_candidates, rf_param_grid, cfg, progress_cb=None):
    """GWRF: k_geo x RF_PARAM_GRID dituning BERSAMA (RF terboboti W_G saja), LOOCV."""
    rows, best = [], None
    total = len(rf_param_grid) * len(k_candidates)
    counter = 0
    for j, raw_params in enumerate(rf_param_grid, 1):
        params = normalize_rf_params(raw_params, X.shape[1])
        for k in k_candidates:
            counter += 1
            bg = adaptive_bandwidth_matrix(d_geo, k)
            Wg = gaussian_kernel(d_geo, bg[:, None])
            pred = predict_local_rf_cv(X, y, Wg, params, cfg["seed"], 600000 + j * 1000 + k)
            m = metrics_dict(y, pred)
            row = {"rf_config": j, "k_geo": k, **params, **{f"{a}_CV": b for a, b in m.items()}}
            rows.append(row)
            if best is None or m["RMSE"] < best["RMSE_CV"]:
                best = row.copy()
                best["preds"] = pred.copy()
                best["RMSE_CV"] = m["RMSE"]
            if progress_cb and (counter % max(1, total // 50) == 0 or counter == total):
                progress_cb(counter / total, f"GWRF RF#{j} k={k} ({counter}/{total}) — RMSE_CV={m['RMSE']:.4f}")
    return best, pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)


def tune_sgwr(X, y, d_geo, d_attr, k_candidates, alpha_candidates, progress_cb=None):
    """SGWR: k_geo x alpha dituning BERSAMA (regresi linear terboboti W_GS), LOOCV."""
    rows, best = [], None
    total = len(k_candidates) * len(alpha_candidates)
    counter = 0
    for k in k_candidates:
        for alpha in alpha_candidates:
            counter += 1
            alpha = float(alpha)
            Wgs, *_ = build_weight_components(d_geo, d_attr, k, alpha)
            pred = predict_weighted_linear_cv(X, y, Wgs)
            m = metrics_dict(y, pred)
            row = {"k_geo": k, "alpha": alpha, "gamma": 1 - alpha, **{f"{a}_CV": b for a, b in m.items()}}
            rows.append(row)
            if best is None or m["RMSE"] < best["RMSE_CV"]:
                best = row.copy()
                best["preds"] = pred.copy()
                best["RMSE_CV"] = m["RMSE"]
            if progress_cb and (counter % max(1, total // 50) == 0 or counter == total):
                progress_cb(counter / total, f"SGWR k={k} α={alpha:.3f} ({counter}/{total}) — RMSE_CV={m['RMSE']:.4f}")
    return best, pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)


def run_baselines(X, y, d_geo, d_attr, cfg, rf_param_grid, k_candidates, alpha_candidates,
                   sgwrf_best_row, progress_cb=None):
    """Semua baseline dievaluasi LOOCV. RF & GWRF di-tuning ulang secara
    independen; GWR hanya menuning k_geo; SGWR menuning k_geo x alpha;
    SGWRF memakai ulang hasil pencarian bersama utama (tanpa hitung ulang)."""

    def sub(base, span):
        def cb(frac, text):
            if progress_cb:
                progress_cb(base + span * frac, text)
        return cb

    rf_best, rf_results = optimize_global_rf(X, y, cfg, rf_param_grid, progress_cb=sub(0.00, 0.15))
    if progress_cb: progress_cb(0.15, "Baseline RF (global) selesai")

    gwr_best, gwr_results = tune_gwr_k(X, y, d_geo, k_candidates, progress_cb=sub(0.15, 0.10))
    if progress_cb: progress_cb(0.25, "Baseline GWR selesai")

    gwrf_best, gwrf_results = tune_gwrf(X, y, d_geo, k_candidates, rf_param_grid, cfg, progress_cb=sub(0.25, 0.45))
    if progress_cb: progress_cb(0.70, "Baseline GWRF selesai")

    sgwr_best, sgwr_results = tune_sgwr(X, y, d_geo, d_attr, k_candidates, alpha_candidates, progress_cb=sub(0.70, 0.30))
    if progress_cb: progress_cb(1.0, "Baseline SGWR selesai")

    rows = [
        {"Model": "RF (global, di-tuning ulang)", **metrics_dict(y, rf_best["preds"])},
        {"Model": "GWR (k_geo dituning)", **metrics_dict(y, gwr_best["preds"])},
        {"Model": "GWRF (k_geo × RF dituning)", **metrics_dict(y, gwrf_best["preds"])},
        {"Model": "SGWR (k_geo × α dituning)", **metrics_dict(y, sgwr_best["preds"])},
        {"Model": "SGWRF (hasil pencarian bersama utama)", **metrics_dict(y, sgwrf_best_row["preds"])},
    ]
    baseline_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return baseline_df, rf_results, gwr_results, gwrf_results, sgwr_results


def build_results(df, id_col, name_col, lat_col, lon_col, y_col, x_cols, x_labels,
                   y, pred_in, pred_cv, local_r2, local_mae, local_rmse, importances,
                   bg_local, alpha, gamma):
    out = df[[id_col, name_col, lat_col, lon_col, y_col]].copy()
    out.insert(0, "point_no", np.arange(1, len(out) + 1))
    out["pred_sgwrf"] = pred_in
    out["residual"] = y - pred_in
    out["abs_error"] = np.abs(y - pred_in)
    out["pred_sgwrf_cv"] = pred_cv
    out["residual_cv"] = y - pred_cv
    out["abs_error_cv"] = np.abs(y - pred_cv)
    out["local_train_R2"] = local_r2
    out["local_train_RMSE"] = local_rmse
    out["local_train_MAE"] = local_mae
    out["bandwidth_geo_local"] = bg_local
    out["alpha"] = alpha
    out["gamma"] = gamma
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
        sub_df = df[df[cat_col] == c]
        fig.add_trace(go.Scattermapbox(
            lat=sub_df[lat_col], lon=sub_df[lon_col], mode="markers+text",
            text=sub_df["point_no"].astype(str) if "point_no" in sub_df.columns else None,
            textfont=dict(size=10, color="black"),
            marker=dict(size=17, color=color_map[c], opacity=0.92),
            customdata=np.stack([sub_df[name_col], sub_df[id_col]], axis=-1),
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
# 3. AUTO-DETEKSI PEMETAAN KOLOM & UTILITAS UI
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


def build_rf_param_grid(rf_cv_trees):
    """Meniru grid 8-konfigurasi pada script terbaru, diskalakan mengikuti
    slider 'jumlah pohon tahap CV' di sidebar."""
    n = rf_cv_trees
    return [
        {"n_estimators": n, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": n, "max_features": 0.7, "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": n, "max_features": 1.0, "min_samples_leaf": 1, "max_depth": None},
        {"n_estimators": n, "max_features": "sqrt", "min_samples_leaf": 2, "max_depth": None},
        {"n_estimators": n, "max_features": 0.7, "min_samples_leaf": 2, "max_depth": None},
        {"n_estimators": n, "max_features": 1.0, "min_samples_leaf": 2, "max_depth": None},
        {"n_estimators": n, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": 5},
        {"n_estimators": n, "max_features": 0.7, "min_samples_leaf": 2, "max_depth": 5},
    ]


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

    n_points = len(raw_df.dropna(subset=[id_col, name_col, lat_col, lon_col, y_col] + x_cols))

    st.subheader("3️⃣ Pencarian Bersama k_geo × α × RF (LOOCV)")
    st.caption("k_geo, α, DAN hyperparameter RF (8 konfigurasi) dicari SEKALIGUS dalam satu grid "
               "(Persamaan 2.28-2.33). Kriteria utama = RMSE LOOCV; AICc hanya diagnostik tambahan.")
    default_min_k = 2
    default_max_k = min(8, max(3, n_points - 1))
    min_k_geo, max_k_geo = st.slider("Rentang k_geo (tetangga geografis terdekat)", 2,
                                      max(3, n_points - 1), (default_min_k, default_max_k))
    alpha_min, alpha_max = st.slider("Rentang alpha (kontribusi bobot geografis)", 0.01, 1.0, (0.025, 1.0), step=0.005)
    n_alpha = st.slider("Jumlah titik grid alpha", 5, 40, 10, step=1,
                         help="Script asli memakai 40 titik grid (0.025-1.0). Karena RF kini ikut "
                              "dicari bersama k_geo & alpha (bukan lagi terpisah), total kombinasi "
                              "= k_geo × alpha × 8 konfigurasi RF — nilai lebih kecil di sini "
                              "mempercepat komputasi secara signifikan.")

    n_k_geo = max_k_geo - min_k_geo + 1
    n_rf_grid = 8
    total_combo = n_k_geo * n_alpha * n_rf_grid
    est_search = total_combo * n_points
    st.markdown(
        f'<div class="sgwrf-warn">⏱️ Total kombinasi pencarian utama: '
        f'<b>{n_k_geo} (k_geo) × {n_alpha} (α) × {n_rf_grid} (RF) = {total_combo:,}</b>, '
        f'≈ <b>{est_search:,} pelatihan model</b> RF berbobot (LOOCV) untuk {n_points} titik data. '
        'Jika baseline diaktifkan, GWRF (k_geo×RF) dan SGWR (k_geo×α) menambah waktu komputasi lagi. '
        'Persempit rentang k_geo / kurangi titik grid alpha / kurangi jumlah pohon jika terasa lambat.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("4️⃣ Parameter Random Forest")
    rf_cv_trees = st.slider("Jumlah pohon (tahap pencarian/CV, berlaku utk 8 konfigurasi RF)", 20, 200, 60, step=10)
    rf_final_trees = st.slider("Jumlah pohon (model final)", 100, 1000, 500, step=50,
                                help="Treewise permutation importance (Pers. 2.35) dihitung untuk SETIAP "
                                     "pohon di model final — makin banyak pohon, makin lama waktu komputasi.")
    seed = st.number_input("Random seed", 0, 99999, 2026)

    st.subheader("5️⃣ Analisis Tambahan")
    run_baseline_flag = st.checkbox(
        "Jalankan model baseline (RF, GWR, GWRF, SGWR, SGWRF)", value=True,
        help="RF di-tuning ulang (8 konfigurasi), GWR menuning k_geo, GWRF menuning k_geo×RF (8 "
             "konfigurasi), SGWR menuning k_geo×α — semuanya di luar pencarian utama, sehingga "
             "menambah waktu komputasi cukup signifikan. SGWRF memakai ulang hasil pencarian utama "
             "(tanpa komputasi tambahan)."
    )

    st.markdown("---")
    run_button = st.button("🚀 Jalankan Analisis SGWRF", use_container_width=True)

cfg = dict(min_k_geo=min_k_geo, max_k_geo=max_k_geo, alpha_min=float(alpha_min), alpha_max=float(alpha_max),
           n_alpha=int(n_alpha), rf_cv_trees=rf_cv_trees, rf_final_trees=rf_final_trees, seed=int(seed))

RF_PARAM_GRID = build_rf_param_grid(rf_cv_trees)

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
        "Rentang parameter (misalnya k_geo) sudah menyesuaikan otomatis. "
        "Tekan **Jalankan Analisis SGWRF** di sidebar untuk memproses data terbaru."
    )
st.session_state["sgwrf_prev_n"] = n_after

tab_data, tab_map, tab_joint, tab_rf_summary, tab_local, tab_baseline, tab_download = st.tabs(
    ["📊 Data & Eksplorasi", "🗺️ Peta Interaktif", "🎯 Pencarian Bersama (k_geo × α × RF)",
     "🌲 Ringkasan Konfigurasi RF", "📈 Model Lokal SGWRF", "🆚 Perbandingan Baseline", "📥 Unduh Hasil"]
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
    k_candidates = candidate_geo_bandwidths(len(y), cfg["min_k_geo"], cfg["max_k_geo"])
    alpha_candidates = np.round(np.linspace(cfg["alpha_min"], cfg["alpha_max"], cfg["n_alpha"]), 4)

    # --- 1) Pencarian BERSAMA k_geo x alpha x RF (kriteria: RMSE LOOCV) ---
    prog = st.progress(0.0, text="Pencarian bersama k_geo × alpha × RF (LOOCV)...")
    def cb_joint(frac, text):
        prog.progress(min(frac, 1.0), text=text)
    best_sgwrf, joint_results, elapsed_j = joint_optimize_sgwrf(
        X, y, d_geo, d_attr, cfg, RF_PARAM_GRID, progress_cb=cb_joint
    )
    best_k_geo = int(best_sgwrf["k_geo"])
    best_alpha = float(best_sgwrf["alpha"])
    best_gamma = 1.0 - best_alpha
    sgwrf_params = {
        "n_estimators": int(best_sgwrf["n_estimators"]), "max_features": best_sgwrf["max_features"],
        "min_samples_leaf": int(best_sgwrf["min_samples_leaf"]), "max_depth": best_sgwrf["max_depth"],
    }
    prog.progress(1.0, text=f"Pencarian selesai ({elapsed_j:.1f}s) — k*={best_k_geo}, α*={best_alpha:.3f}")

    Wgs, Wg, Ws, bg_local, gamma_check = build_weight_components(d_geo, d_attr, best_k_geo, best_alpha)
    pred_cv = best_sgwrf["preds"]  # LOOCV prediksi kombinasi terbaik (langsung dari pencarian bersama)

    # --- 2) Model lokal final (in-sample, self-weight ikut serta) ---
    prog3 = st.progress(0.0, text="Melatih model lokal SGWRF final per titik...")
    def cb_local(frac, text):
        prog3.progress(min(frac, 1.0), text=text)
    pred_in, local_r2, local_mae, local_rmse, importances = train_local_models(
        X, y, Wgs, df, name_col, x_cols, x_labels, sgwrf_params, cfg, progress_cb=cb_local
    )
    prog3.progress(1.0, text="Model lokal final selesai")

    overall_cv = metrics_dict(y, pred_cv)
    overall_in = metrics_dict(y, pred_in)

    results = build_results(df, id_col, name_col, lat_col, lon_col, y_col, x_cols, x_labels,
                             y, pred_in, pred_cv, local_r2, local_mae, local_rmse, importances,
                             bg_local, best_alpha, best_gamma)

    baseline_results, rf_global_results, gwr_results, gwrf_results, sgwr_results = None, None, None, None, None
    if run_baseline_flag:
        prog4 = st.progress(0.0, text="Menjalankan & men-tuning model baseline pembanding...")
        def cb_base(frac, text):
            prog4.progress(min(frac, 1.0), text=text)
        baseline_results, rf_global_results, gwr_results, gwrf_results, sgwr_results = run_baselines(
            X, y, d_geo, d_attr, cfg, RF_PARAM_GRID, k_candidates, alpha_candidates,
            best_sgwrf, progress_cb=cb_base
        )
        prog4.progress(1.0, text="Baseline selesai")

    st.session_state["sgwrf"] = dict(
        results=results, joint_results=joint_results,
        best_k_geo=best_k_geo, best_alpha=best_alpha, best_gamma=best_gamma, sgwrf_params=sgwrf_params,
        overall_cv=overall_cv, overall_in=overall_in, importances=importances,
        baseline_results=baseline_results, rf_global_results=rf_global_results,
        gwr_results=gwr_results, gwrf_results=gwrf_results, sgwr_results=sgwr_results,
        x_cols=x_cols, x_labels=x_labels, name_col=name_col, id_col=id_col,
    )
    st.success("✅ Analisis SGWRF selesai! Lihat hasil pada tab lainnya.")

# ================================================================
# 7. TAMPILKAN HASIL (jika sudah pernah dijalankan)
# ================================================================

state = st.session_state.get("sgwrf")

if state is None:
    for t in (tab_joint, tab_rf_summary, tab_local, tab_baseline, tab_download):
        with t:
            st.info("⬅️ Atur parameter di sidebar lalu tekan **Jalankan Analisis SGWRF** untuk melihat hasil di sini.")
else:
    results = state["results"]
    joint_results = state["joint_results"]
    best_k_geo = state["best_k_geo"]
    best_alpha = state["best_alpha"]
    best_gamma = state["best_gamma"]
    sgwrf_params = state["sgwrf_params"]
    overall_cv = state["overall_cv"]
    overall_in = state["overall_in"]
    importances = state["importances"]
    baseline_results = state["baseline_results"]
    rf_global_results = state["rf_global_results"]
    gwr_results = state["gwr_results"]
    gwrf_results = state["gwrf_results"]
    sgwr_results = state["sgwr_results"]

    # -------- peta hasil (tambahan di tab_map) --------
    with tab_map:
        st.markdown("---")
        st.subheader("Peta Hasil Model SGWRF (berbasis prediksi LOOCV)")
        m1, m2 = st.columns(2)
        with m1:
            fig_pred = make_continuous_map(results, lat_col, lon_col, "pred_sgwrf_cv", name_col, id_col,
                                            "Prediksi LOOCV — SGWRF", colorscale="YlOrRd", boundary=boundary)
            st.plotly_chart(fig_pred, use_container_width=True)
            note(
                f"Ini adalah nilai **{y_col} hasil prediksi LOOCV** model SGWRF di tiap titik (bobot titik "
                f"itu sendiri dinolkan saat memprediksinya, sehingga tidak bias/optimistik). Bandingkan "
                f"pola warnanya dengan peta 'Sebaran Titik & Nilai {y_col}' di atas — jika polanya mirip, "
                f"model berhasil menangkap pola spasial data aktual dengan baik."
            )
        with m2:
            fig_resid = make_continuous_map(results, lat_col, lon_col, "residual_cv", name_col, id_col,
                                             "Residual LOOCV (Aktual − Prediksi)", colorscale="RdBu", boundary=boundary)
            st.plotly_chart(fig_resid, use_container_width=True)
            n_over = int((results["residual_cv"] < 0).sum())
            n_under = int((results["residual_cv"] > 0).sum())
            note(
                f"Residual LOOCV = nilai aktual dikurangi prediksi LOOCV. Warna **biru** berarti model "
                f"**terlalu tinggi menaksir (overestimate)**, warna **merah** berarti model **terlalu rendah "
                f"menaksir (underestimate)**. Saat ini {n_over} titik overestimate dan {n_under} titik "
                f"underestimate. Titik dengan warna sangat pekat layak dicek lebih lanjut — bisa jadi "
                f"outlier data atau area dengan dinamika lokal yang belum tertangkap kovariat yang ada."
            )

        m3, m4 = st.columns(2)
        with m3:
            fig_dom = make_categorical_map(results, lat_col, lon_col, "dominant_variable", name_col, id_col,
                                            "Variabel Dominan Lokal SGWRF", boundary=boundary)
            st.plotly_chart(fig_dom, use_container_width=True)
            n_dom_vars = results["dominant_variable"].nunique()
            note(
                f"Setiap titik diwarnai menurut kovariat **paling berpengaruh secara lokal** di titik "
                f"tersebut, berdasarkan **treewise permutation importance** (Persamaan 2.35) dari model RF "
                f"lokal final. Saat ini teridentifikasi **{n_dom_vars} variabel dominan berbeda** di antara "
                f"{len(results)} titik. Klik nama variabel pada legenda untuk menyalakan/mematikan "
                f"tampilannya. Semakin beragam variabel dominannya, semakin kuat bukti **heterogenitas "
                f"spasial** — inti yang membedakan SGWRF dari model global (misalnya RF biasa) yang "
                f"mengasumsikan satu set pengaruh yang sama untuk semua lokasi."
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

    # -------- tab pencarian bersama k_geo x alpha x RF --------
    with tab_joint:
        st.subheader("Pencarian Bersama k_geo × α × RF (Kriteria Utama: RMSE LOOCV)")
        st.markdown(
            '<div class="sgwrf-eq">W_GS = α·W_G + γ·W_S, γ=1-α (2.24) &nbsp;|&nbsp; '
            '(k*, α*, RF*) = argmin RMSE_CV(k_geo, α, RF) (2.32) &nbsp;|&nbsp; AICc = diagnostik tambahan (2.29)</div>',
            unsafe_allow_html=True,
        )
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("k_geo optimum (k*)", best_k_geo)
        s2.metric("alpha optimum (α*)", f"{best_alpha:.3f}")
        s3.metric("gamma optimum (γ*)", f"{best_gamma:.3f}")
        s4.metric("RMSE LOOCV minimum", f"{joint_results['RMSE_CV'].min():.4f}")
        st.json(sgwrf_params)

        best_rf_id = int(joint_results.loc[joint_results["RMSE_CV"].idxmin(), "rf_config"])
        pivot = joint_results[joint_results["rf_config"] == best_rf_id].pivot_table(
            index="k_geo", columns="alpha", values="RMSE_CV", aggfunc="min"
        )
        fig_heat = px.imshow(pivot, color_continuous_scale="Viridis", aspect="auto",
                              labels=dict(x="alpha", y="k_geo", color="RMSE LOOCV"),
                              title=f"RMSE-LOOCV untuk k_geo × alpha (pada konfigurasi RF#{best_rf_id} terbaik)")
        fig_heat.update_traces(hovertemplate="alpha=%{x}<br>k_geo=%{y}<br>RMSE=%{z:.4f}<extra></extra>")
        st.plotly_chart(fig_heat, use_container_width=True)
        note(
            f"Setiap sel adalah RMSE hasil validasi silang **LOOCV** untuk satu kombinasi (k_geo, α), "
            f"diiris pada konfigurasi RF terbaik (**RF#{best_rf_id}**) agar heatmap tetap 2D dan mudah "
            f"dibaca — walau pencarian sesungguhnya 3D (k_geo × α × RF). Warna **gelap/ungu** = error "
            f"kecil (kombinasi lebih baik), warna **terang/kuning** = error besar. Kombinasi terbaik "
            f"keseluruhan: **k_geo* = {best_k_geo}**, **α* = {best_alpha:.3f}** (γ* = {best_gamma:.3f}), "
            f"dengan hyperparameter RF seperti pada kotak di atas."
        )

        fig_aicc = px.line(
            joint_results[(joint_results["k_geo"] == best_k_geo) & (joint_results["rf_config"] == best_rf_id)].sort_values("alpha"),
            x="alpha", y="AICc_diagnostic", markers=True,
            title=f"AICc vs Alpha pada k_geo*={best_k_geo}, RF#{best_rf_id} (diagnostik tambahan)"
        )
        fig_aicc.add_vline(x=best_alpha, line_dash="dash", line_color="#e74c3c", annotation_text=f"α*={best_alpha:.3f}")
        st.plotly_chart(fig_aicc, use_container_width=True)
        note(
            "AICc menyeimbangkan kecocokan model (RSS) dengan kompleksitas efektif (ENP = trace(S), "
            "proksi karena Random Forest tidak memiliki hat-matrix linear eksplisit). Kurva ini murni "
            "**diagnostik tambahan** — keputusan k_geo*/α*/RF* di dashboard ini sepenuhnya berdasarkan "
            "RMSE LOOCV terkecil, bukan AICc."
        )

        with st.expander("📋 Tabel lengkap seluruh kombinasi k_geo × alpha × RF (diurutkan RMSE_CV)"):
            st.dataframe(joint_results, use_container_width=True, height=350)

    # -------- tab ringkasan konfigurasi RF (diagregasi dari hasil pencarian bersama) --------
    with tab_rf_summary:
        st.subheader("Ringkasan Konfigurasi RF dari Pencarian Bersama")
        st.caption(
            "Karena hyperparameter RF kini dicari BERSAMA k_geo & alpha (bukan tahap terpisah), tab ini "
            "hanya meringkas hasil pencarian bersama tersebut per konfigurasi RF — tidak ada komputasi "
            "tambahan."
        )
        rf_summary = joint_results.groupby(
            ["rf_config", "n_estimators", "max_features", "min_samples_leaf", "max_depth"], as_index=False
        )["RMSE_CV"].min().sort_values("RMSE_CV")
        fig_rf = px.bar(rf_summary, x="rf_config", y="RMSE_CV",
                         hover_data=["n_estimators", "max_features", "min_samples_leaf", "max_depth"],
                         title="RMSE-LOOCV Terbaik per Konfigurasi RF (8 kombinasi, diambil dari seluruh k_geo×alpha)",
                         labels={"rf_config": "Konfigurasi RF"})
        st.plotly_chart(fig_rf, use_container_width=True)
        best_rf_row = rf_summary.iloc[0]
        note(
            f"Setiap batang adalah RMSE-LOOCV **terbaik** (RMSE minimum di seluruh kombinasi k_geo×alpha) "
            f"untuk satu dari **8 konfigurasi** hyperparameter RF (`max_features`, `min_samples_leaf`, "
            f"`max_depth`). Batang **terendah** — konfigurasi **RF#{int(best_rf_row['rf_config'])}** "
            f"(`max_features={best_rf_row['max_features']}`, `min_samples_leaf={int(best_rf_row['min_samples_leaf'])}`, "
            f"`max_depth={best_rf_row['max_depth']}`) — adalah konfigurasi yang terpilih sebagai bagian "
            f"dari kombinasi (k_geo*, α*, RF*) optimal keseluruhan."
        )
        st.dataframe(rf_summary, use_container_width=True)

    # -------- tab model lokal --------
    with tab_local:
        st.subheader("Kinerja Model SGWRF — Evaluasi Utama (LOOCV)")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("RMSE (LOOCV)", f"{overall_cv['RMSE']:.4f}")
        g2.metric("MAE (LOOCV)", f"{overall_cv['MAE']:.4f}")
        g3.metric("MAPE (LOOCV)", f"{overall_cv['MAPE']:.2f}%")
        g4.metric("R² (LOOCV)", f"{overall_cv['R2']:.4f}")
        kualitas = "sangat baik" if overall_cv["R2"] >= 0.8 else ("cukup baik" if overall_cv["R2"] >= 0.6 else
                    ("moderat" if overall_cv["R2"] >= 0.4 else "masih lemah — pertimbangkan menambah kovariat atau titik data"))
        note(
            f"Metrik ini dihitung dari prediksi **leave-one-out cross-validation (LOOCV)** milik kombinasi "
            f"(k_geo*, α*, RF*) terbaik dari tahap pencarian bersama — bobot titik itu sendiri dibuat nol "
            f"saat memprediksinya, sehingga hasilnya tidak bias/optimistik dan menjadi **metrik evaluasi "
            f"utama** untuk model SGWRF. **R² = {overall_cv['R2']:.4f}** berarti model menjelaskan sekitar "
            f"**{max(overall_cv['R2'], 0)*100:.1f}%** variasi {y_col} pada data yang tidak dilihat model "
            f"saat memprediksi titik tersebut (kualitas model tergolong **{kualitas}**)."
        )

        with st.expander("📎 Diagnostik in-sample (bobot titik sendiri ikut serta — cenderung optimistik)"):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("RMSE (in-sample)", f"{overall_in['RMSE']:.4f}")
            d2.metric("MAE (in-sample)", f"{overall_in['MAE']:.4f}")
            d3.metric("MAPE (in-sample)", f"{overall_in['MAPE']:.2f}%")
            d4.metric("R² (in-sample)", f"{overall_in['R2']:.4f}")
            st.caption(
                "Model final SGWRF dilatih TANPA menolkan bobot titik itu sendiri (sesuai desain: semua "
                "titik menjadi penyumbang informasi lokal), dan memakai jumlah pohon FINAL (bisa berbeda "
                "dari jumlah pohon tahap pencarian). Karena itu nilai in-sample biasanya lebih baik "
                "daripada LOOCV. Gunakan metrik LOOCV di atas sebagai acuan evaluasi utama."
            )

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_avp = px.scatter(results, x=y_col, y="pred_sgwrf_cv", hover_name=name_col,
                                  title="Aktual vs Prediksi (LOOCV)", trendline="ols")
            vmin = float(min(results[y_col].min(), results["pred_sgwrf_cv"].min()))
            vmax = float(max(results[y_col].max(), results["pred_sgwrf_cv"].max()))
            fig_avp.add_shape(type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
                               line=dict(color="gray", dash="dash"))
            st.plotly_chart(fig_avp, use_container_width=True)
            note(
                "Garis putus-putus abu-abu adalah garis ideal (prediksi = aktual). Makin dekat titik-titik "
                "ke garis ini, makin akurat model. Titik yang jauh dari garis adalah lokasi dengan error "
                "prediksi LOOCV besar — cek apakah lokasi tersebut juga muncul menonjol pada peta residual."
            )
        with cc2:
            fig_res_hist = px.histogram(results, x="residual_cv", nbins=15, title="Distribusi Residual (LOOCV)")
            st.plotly_chart(fig_res_hist, use_container_width=True)
            mean_resid = float(results["residual_cv"].mean())
            note(
                f"Idealnya residual tersebar **di sekitar 0** tanpa bias sistematis. Rata-rata residual "
                f"LOOCV saat ini **{mean_resid:.3f}** — nilai mendekati 0 menandakan model tidak bias secara "
                f"konsisten ke arah over/underestimate. Sebaran yang melebar menandakan variasi error "
                f"cukup besar antar titik."
            )

        st.subheader("Hasil Model Lokal per Titik")
        show_cols = ["point_no", name_col, y_col, "pred_sgwrf_cv", "residual_cv", "pred_sgwrf",
                     "dominant_variable", "dominant_importance", "local_train_R2",
                     "bandwidth_geo_local", "alpha", "gamma"]
        st.dataframe(results[show_cols], use_container_width=True, height=350)
        note(
            "Kolom `pred_sgwrf_cv`/`residual_cv` adalah hasil **LOOCV** (evaluasi utama), sedangkan "
            "`pred_sgwrf` adalah prediksi **in-sample** dari model final. Kolom `bandwidth_geo_local` "
            "adalah bandwidth geografis adaptif b_g(i) di titik tersebut (dalam satuan derajat koordinat, "
            "mengikuti Persamaan 2.10), sementara `alpha`/`gamma` adalah bobot α*/γ* global hasil pencarian "
            "bersama k_geo×α×RF yang berlaku untuk seluruh titik."
        )

        st.subheader("Rata-rata Variable Importance (Seluruh Lokasi)")
        mean_imp = importances.mean(axis=0)
        imp_df = pd.DataFrame({
            "Variabel": [state["x_labels"][c] for c in state["x_cols"]],
            "Importance (%)": mean_imp * 100,
        }).sort_values("Importance (%)", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance (%)", y="Variabel", orientation="h",
                          title="Rata-rata Kepentingan Variabel (Treewise Permutation Importance)",
                          color="Importance (%)", color_continuous_scale="Blues")
        st.plotly_chart(fig_imp, use_container_width=True)
        top_global = imp_df.iloc[-1]
        note(
            f"Dihitung dengan **treewise permutation importance** (Persamaan 2.35): untuk setiap pohon di "
            f"forest, kovariat dipermutasi satu per satu dan kenaikan weighted-MSE-nya dirata-ratakan ke "
            f"seluruh pohon. Ini **rata-rata** kepentingan tiap kovariat di seluruh titik (bukan "
            f"per-lokasi). Secara keseluruhan, **{top_global['Variabel']}** paling berpengaruh terhadap "
            f"{y_col} ({top_global['Importance (%)']:.1f}%). Namun karena SGWRF bersifat lokal, urutan ini "
            f"bisa berbeda-beda di tiap titik — lihat peta panas di bawah dan peta 'Variabel Dominan Lokal "
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

        with st.expander("🧮 Rincian Bobot Lokal (b_g, α*, γ*) per Titik"):
            st.markdown(
                '<div class="sgwrf-note">Bandwidth geografis b_g(i) bersifat <b>lokal/adaptif</b> (berbeda '
                'tiap titik, mengikuti kepadatan tetangganya), sedangkan α* dan γ* adalah bobot '
                '<b>global</b> hasil pencarian bersama k_geo×α×RF yang sama untuk semua titik pada '
                'Persamaan (2.24). Model SGWRF utamanya adalah Random Forest lokal berbobot — bukan model '
                'linear, sehingga tidak memiliki koefisien beta seperti GWR klasik. Ukuran kepentingan '
                'variabel yang sah adalah treewise permutation importance (VI_ik) pada tabel di atas.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                results[["point_no", name_col, "bandwidth_geo_local", "alpha", "gamma"]],
                use_container_width=True, height=280,
            )

    # -------- tab baseline --------
    with tab_baseline:
        if baseline_results is None:
            st.info("Model baseline tidak dijalankan. Aktifkan opsi di sidebar lalu jalankan ulang analisis.")
        else:
            st.subheader("Perbandingan SGWRF vs Model Baseline (semua LOOCV)")
            st.dataframe(baseline_results, use_container_width=True)

            fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("RMSE (lebih rendah lebih baik)", "R² (lebih tinggi lebih baik)"))
            fig_bar.add_trace(go.Bar(x=baseline_results["Model"], y=baseline_results["RMSE"],
                                      marker_color="#e74c3c", name="RMSE"), row=1, col=1)
            fig_bar.add_trace(go.Bar(x=baseline_results["Model"], y=baseline_results["R2"],
                                      marker_color="#27ae60", name="R2"), row=1, col=2)
            fig_bar.update_layout(height=430, showlegend=False, title="Ringkasan Kinerja Model (semua LOOCV)")
            st.plotly_chart(fig_bar, use_container_width=True)
            sgwrf_rank = int((baseline_results["RMSE"] < overall_cv["RMSE"]).sum())
            posisi = "lebih baik dari seluruh" if sgwrf_rank == 0 else f"lebih baik dari {len(baseline_results)-sgwrf_rank} dari {len(baseline_results)}"
            note(
                f"Model utama **SGWRF** (tab Model Lokal SGWRF) memiliki RMSE LOOCV **{overall_cv['RMSE']:.4f}** "
                f"dan R² LOOCV **{overall_cv['R2']:.4f}** — dibandingkan baseline di atas, SGWRF saat ini "
                f"{posisi} model pembanding. Baris **SGWRF** pada tabel ini memakai ulang hasil pencarian "
                f"bersama utama (tanpa komputasi tambahan), sehingga nilainya identik dengan metrik LOOCV di "
                f"tab Model Lokal SGWRF. Model lain (**RF**, **GWR**, **GWRF**, **SGWR**) masing-masing "
                f"di-tuning secara independen pada ruang pencariannya sendiri, sehingga perbandingannya "
                f"adil — tiap model dievaluasi pada kondisi terbaiknya sendiri. Jika SGWRF **tidak** "
                f"mengungguli GWRF/RF secara meyakinkan, pertimbangkan menambah kovariat relevan, menambah "
                f"titik observasi, atau memperluas rentang pencarian k_geo/alpha di sidebar."
            )

            st.markdown(
                '<div class="sgwrf-note">Baseline: <b>RF</b> (Random Forest tanpa pembobotan lokal, hanya '
                '8 konfigurasi RF yang dituning, LOOCV), <b>GWR</b> (regresi linear terboboti W_G/geografis, '
                'hanya k_geo yang dituning, LOOCV), <b>GWRF</b> (RF terboboti W_G, k_geo × 8 konfigurasi RF '
                'dituning bersama, LOOCV), <b>SGWR</b> (regresi linear terboboti W_GS = α·W_G+γ·W_S, k_geo × '
                'α dituning bersama, LOOCV), <b>SGWRF</b> (hasil terbaik dari pencarian bersama k_geo × α × '
                'RF pada tab utama).</div>',
                unsafe_allow_html=True,
            )

            with st.expander("📋 Detail tuning RF (global) — 8 konfigurasi, LOOCV"):
                st.dataframe(rf_global_results, use_container_width=True, height=280)
            with st.expander("📋 Detail tuning GWR — rentang k_geo, LOOCV"):
                st.dataframe(gwr_results, use_container_width=True, height=280)
            with st.expander("📋 Detail tuning GWRF — k_geo × 8 konfigurasi RF, LOOCV"):
                st.dataframe(gwrf_results, use_container_width=True, height=280)
            with st.expander("📋 Detail tuning SGWR — k_geo × alpha, LOOCV"):
                st.dataframe(sgwr_results, use_container_width=True, height=280)

    # -------- tab unduh --------
    with tab_download:
        st.subheader("Unduh Seluruh Hasil Analisis")

        def to_csv_bytes(d):
            return d.to_csv(index=False).encode("utf-8")

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("⬇️ Hasil Lokal SGWRF (.csv)", to_csv_bytes(results),
                                "hasil_lokal_sgwrf.csv", "text/csv", use_container_width=True)
            st.download_button("⬇️ Pencarian Bersama k_geo×α×RF (.csv)", to_csv_bytes(joint_results),
                                "hasil_optimasi_k_alpha_RF_LOOCV.csv", "text/csv", use_container_width=True)
        with dl2:
            vi_export = pd.DataFrame(importances, columns=state["x_cols"])
            vi_export.insert(0, "point_no", results["point_no"].to_numpy())
            st.download_button("⬇️ Local Variable Importance (.csv)", to_csv_bytes(vi_export),
                                "local_variable_importance.csv", "text/csv", use_container_width=True)
            if baseline_results is not None:
                st.download_button("⬇️ Perbandingan Model Baseline (.csv)", to_csv_bytes(baseline_results),
                                    "perbandingan_model.csv", "text/csv", use_container_width=True)
        with dl3:
            if rf_global_results is not None:
                st.download_button("⬇️ Tuning RF Global (.csv)", to_csv_bytes(rf_global_results),
                                    "hasil_tuning_rf_global_LOOCV.csv", "text/csv", use_container_width=True)
            if gwr_results is not None:
                st.download_button("⬇️ Tuning GWR (.csv)", to_csv_bytes(gwr_results),
                                    "hasil_tuning_gwr_LOOCV.csv", "text/csv", use_container_width=True)
            if gwrf_results is not None:
                st.download_button("⬇️ Tuning GWRF (.csv)", to_csv_bytes(gwrf_results),
                                    "hasil_tuning_gwrf_LOOCV.csv", "text/csv", use_container_width=True)
            if sgwr_results is not None:
                st.download_button("⬇️ Tuning SGWR (.csv)", to_csv_bytes(sgwr_results),
                                    "hasil_tuning_sgwr_LOOCV.csv", "text/csv", use_container_width=True)

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="hasil_lokal_sgwrf", index=False)
            joint_results.to_excel(writer, sheet_name="optimasi_k_alpha_RF_LOOCV", index=False)
            vi_export.to_excel(writer, sheet_name="variable_importance", index=False)
            if baseline_results is not None:
                baseline_results.to_excel(writer, sheet_name="perbandingan_baseline", index=False)
            if rf_global_results is not None:
                rf_global_results.to_excel(writer, sheet_name="tuning_rf_global", index=False)
            if gwr_results is not None:
                gwr_results.to_excel(writer, sheet_name="tuning_gwr", index=False)
            if gwrf_results is not None:
                gwrf_results.to_excel(writer, sheet_name="tuning_gwrf", index=False)
            if sgwr_results is not None:
                sgwr_results.to_excel(writer, sheet_name="tuning_sgwr", index=False)
        st.download_button("⬇️ Semua Hasil (Excel, multi-sheet)", excel_buf.getvalue(),
                            "hasil_sgwrf_lengkap.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)

st.markdown("---")
st.caption(
    "Dashboard SGWRF — pipeline: standarisasi Z-score (2.3-2.5) → jarak geografis Euclidean (2.10) + "
    "kernel Gaussian adaptif (2.11) → jarak atribut mean|Z_i-Z_j| (2.18-2.19) + similarity weight (2.20) "
    "→ kombinasi aditif W_GS = α·W_G + γ·W_S (2.24) → pencarian BERSAMA k_geo, alpha, DAN hyperparameter "
    "RF (8 konfigurasi) via LOOCV, AICc sebagai diagnostik tambahan (2.28-2.33) → model lokal RF final "
    "per titik → treewise permutation importance (2.35) → evaluasi LOOCV (2.36-2.38) → model baseline "
    "pembanding (RF, GWR, GWRF, SGWR masing-masing dituning independen; SGWRF memakai ulang hasil "
    "pencarian utama)."
)
