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
#   (2.28)-(2.33) Optimasi alpha/gamma via AICc (ENP = trace(S), proxy)
#   (2.35)        Treewise permutation variable importance
#   (2.36)-(2.38) RMSE, MAPE, R2 (evaluasi utama memakai LOOCV)
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
from plotly.subplots import make_subplots

from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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
    color:#eef4fb;
}
/* Kotak input/dropdown/file-uploader tetap berlatar terang, sehingga
   teks di DALAM kotak tersebut harus gelap agar terbaca jelas. */
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-baseweb="popover"] *,
section[data-testid="stSidebar"] div[role="listbox"] *,
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] *,
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzoneInstructions"] *,
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] *,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea{
    color:#1a2733 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
    background:#ffffff;border-radius:8px;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"]{
    background:#ffffff;border-radius:10px;
}
/* Placeholder / teks bantu ("Choose an option", nama file terunggah, dsb.) */
section[data-testid="stSidebar"] div[data-baseweb="select"] div[class*="placeholder"]{
    color:#5a6b7a !important;
}
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
.sgwrf-eq{
    background:#f4f6fb;border:1px dashed #9db3c9;border-radius:8px;
    padding:0.5rem 0.8rem;font-family:"Courier New",monospace;font-size:0.85rem;color:#0f4c81;
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
    kombinasi aditif W<sub>GS</sub> = α·W<sub>G</sub> + γ·W<sub>S</sub> yang dioptimasi lewat AICc,
    model Random Forest lokal, dan treewise permutation importance. Unggah data Anda sendiri —
    peta &amp; grafik menyesuaikan otomatis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================================================================
# 1. FUNGSI INTI — sesuai Persamaan (2.3)-(2.38) pada proposal
# ================================================================

def geographic_distance_euclidean(coords):
    """Persamaan (2.10): d_ij^G = sqrt[(u_i-u_j)^2 + (v_i-v_j)^2].
    Koordinat (lat, lon) dipakai langsung sebagai (u, v) sesuai draft."""
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
    W_GS = alpha * W_G + gamma * W_S,  gamma = 1 - alpha.
    Kombinasi ADITIF (bukan perkalian), tanpa bandwidth atribut."""
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
    Z = scaler.fit_transform(X)                      # (2.3)-(2.5)
    d_geo = geographic_distance_euclidean(coords)      # (2.10)
    d_attr = attribute_pairwise_distance(Z)            # (2.18)-(2.19)
    return X, Z, y, coords, d_geo, d_attr, scaler


def candidate_geo_bandwidths(n, min_k, max_k):
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


def local_prediction_for_target(i, X, y, W, params, random_state, leave_target_out=True):
    w = W[i].copy()
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
    """Prediksi LOOCV Random Forest lokal — dipakai untuk evaluasi utama
    (menghindari kebocoran data karena target ikut menjadi data latih)."""
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


# ---------------- 1a. Optimasi bandwidth geografis (k_geo) ----------------

def cv_score_geo_bandwidth(k_geo, X, y, d_geo, d_attr, alpha, params, leave_target_out, base_seed):
    W, *_ = build_weight_components(d_geo, d_attr, k_geo, alpha)
    preds = np.full(len(y), np.nan)
    for i in range(len(y)):
        _, pred, _, _ = local_prediction_for_target(i, X, y, W, params, base_seed + i, leave_target_out)
        preds[i] = pred
    residual = y - preds
    return float(np.sum(residual ** 2)), float(np.sqrt(np.mean(residual ** 2))), preds


def optimize_geographic_bandwidth(X, y, d_geo, d_attr, alpha_initial, cfg, progress_cb=None):
    candidates = candidate_geo_bandwidths(len(y), cfg["min_k_geo"], cfg["max_k_geo"])
    cv_params = {"n_estimators": cfg["rf_cv_trees"], "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None}
    rows, best = [], None
    t0 = time.time()
    for no, k_geo in enumerate(candidates, 1):
        rss, rmse, _ = cv_score_geo_bandwidth(k_geo, X, y, d_geo, d_attr, alpha_initial, cv_params,
                                               cfg["leave_target_out"], cfg["seed"])
        row = {"iteration": no, "k_geo": int(k_geo), "alpha_awal": alpha_initial, "RSS_CV": rss, "RMSE_CV": rmse}
        rows.append(row)
        if best is None or rmse < best["RMSE_CV"]:
            best = row.copy()
        if progress_cb:
            progress_cb(no / len(candidates), f"k_geo={k_geo} — RMSE_CV={rmse:.4f}")
    res = pd.DataFrame(rows).sort_values("RMSE_CV").reset_index(drop=True)
    return int(best["k_geo"]), res, time.time() - t0


# ---------------- 1b. Optimasi alpha/gamma via AICc ----------------

def normalized_weight_hat_proxy(W):
    """Proxy S untuk ENP = trace(S). RF tidak punya hat matrix linear
    eksplisit; proxy = normalized local weight matrix, sesuai catatan
    metodologis pada script asli."""
    row_sum = W.sum(axis=1, keepdims=True)
    return W / np.maximum(row_sum, 1e-15)


def effective_number_parameters(W):
    return float(np.trace(normalized_weight_hat_proxy(W)))


def aicc_from_rss(rss, n, enp):
    """Persamaan (2.29):
    AICc = 2n ln(RSS/n) + n ln(2*pi) + n + 2(ENP+1)(ENP+2)/(n-ENP-2)."""
    rss = max(float(rss), 1e-15)
    denom = n - enp - 2
    if denom <= 0:
        return np.inf
    return (2.0 * n * np.log(rss / n) + n * np.log(2.0 * np.pi) + n
            + (2.0 * (enp + 1.0) * (enp + 2.0)) / denom)


def optimize_alpha(X, y, d_geo, d_attr, k_geo, cfg, progress_cb=None):
    alpha_grid = np.linspace(cfg["alpha_min"], cfg["alpha_max"], cfg["n_alpha"])
    cv_params = {"n_estimators": cfg["rf_cv_trees"], "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None}
    rows, best = [], None
    t0 = time.time()
    for no, alpha in enumerate(alpha_grid, 1):
        alpha = float(alpha)
        gamma = 1.0 - alpha
        W, *_ = build_weight_components(d_geo, d_attr, k_geo, alpha)
        preds = np.full(len(y), np.nan)
        for i in range(len(y)):
            _, pred, _, _ = local_prediction_for_target(
                i, X, y, W, cv_params, cfg["seed"] + 10000 + no * 100 + i, cfg["leave_target_out"]
            )
            preds[i] = pred
        residual = y - preds
        rss = float(np.sum(residual ** 2))
        enp = effective_number_parameters(W)
        aicc = aicc_from_rss(rss, len(y), enp)
        row = {"iteration": no, "alpha": alpha, "gamma": gamma, "RSS": rss, "ENP": enp,
               "AICc": aicc, "RMSE_CV": float(np.sqrt(np.mean(residual ** 2)))}
        rows.append(row)
        if best is None or aicc < best["AICc"]:
            best = row.copy()
        if progress_cb:
            progress_cb(no / len(alpha_grid), f"alpha={alpha:.3f} — AICc={aicc:.2f}")
    results = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    return float(best["alpha"]), float(best["gamma"]), results, time.time() - t0


# ---------------- 1c. Optimasi hyperparameter Random Forest ----------------

def optimize_rf(X, y, W, cfg, param_grid, progress_cb=None):
    rows, best = [], None
    total = len(param_grid)
    for j, params in enumerate(param_grid, 1):
        p = params.copy()
        p["n_estimators"] = min(p["n_estimators"], cfg["rf_cv_trees"])
        preds = np.full(len(y), np.nan)
        for i in range(len(y)):
            _, pred, _, _ = local_prediction_for_target(
                i, X, y, W, p, cfg["seed"] + 20000 + j * 100 + i, cfg["leave_target_out"]
            )
            preds[i] = pred
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        rows.append({**p, "RMSE_CV": rmse})
        if best is None or rmse < best["RMSE_CV"]:
            best = rows[-1].copy()
        if progress_cb:
            progress_cb(j / total, f"RF config {j}/{total} — RMSE_CV={rmse:.4f}")
    return best, pd.DataFrame(rows)


# ---------------- 1d. Model lokal final + treewise importance (2.35) ----------------

def local_treewise_variable_importance(model, X_train, y_train, sample_weight, seed):
    """Persamaan (2.35): VI_ik = (1/B) * sum_b (E_ib,k^perm - E_ib).
    Untuk tiap pohon di forest: hitung weighted MSE sebelum & sesudah
    kovariat k dipermutasi, lalu rata-ratakan selisihnya ke semua pohon."""
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


def train_local_models(X, y, W, df, name_col, x_cols, x_labels, rf_final, cfg, progress_cb=None):
    """Model final per titik: TIDAK leave-target-out (bobot titik sendiri
    ikut serta bila w_ii > 0), sesuai script terbaru — semua titik menjadi
    penyumbang informasi lokal untuk model final. Evaluasi utama tetap
    memakai prediksi LOOCV terpisah (lihat predict_local_rf_cv)."""
    n = len(y)
    p = X.shape[1]
    preds = np.full(n, np.nan)
    local_r2 = np.full(n, np.nan)
    local_mae = np.full(n, np.nan)
    local_rmse = np.full(n, np.nan)
    importances = np.zeros((n, p))

    final_params = {
        "n_estimators": cfg["rf_final_trees"],
        "max_features": rf_final.get("max_features", "sqrt"),
        "min_samples_leaf": rf_final.get("min_samples_leaf", 1),
        "max_depth": rf_final.get("max_depth", None),
    }

    for i in range(n):
        w = W[i].copy()
        train_mask = w > 1e-12
        if train_mask.sum() < 4:
            train_mask[:] = True

        model = RandomForestRegressor(
            n_estimators=final_params["n_estimators"],
            max_features=final_params["max_features"],
            min_samples_leaf=final_params["min_samples_leaf"],
            max_depth=final_params["max_depth"],
            random_state=cfg["seed"] + i,
            n_jobs=-1,
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


def run_baselines(X, y, d_geo, d_attr, k_geo, alpha, cfg, progress_cb=None):
    """Model pembanding: RF global, GWR (linear terboboti geografis),
    GWRF (RF terboboti geografis), SGWR (linear terboboti W_GS),
    SGWRF (RF terboboti W_GS — LOOCV, sanity-check untuk model utama)."""
    rows = []
    steps, step = 5, 0

    rf = RandomForestRegressor(n_estimators=cfg["rf_final_trees"], max_features="sqrt",
                                min_samples_leaf=1, random_state=cfg["seed"], n_jobs=-1)
    rf.fit(X, y)
    rows.append({"Model": "RF (global)", **metrics_dict(y, rf.predict(X))})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: RF global")

    bg_local = adaptive_bandwidth_matrix(d_geo, k_geo)
    Wg = gaussian_kernel(d_geo, bg_local[:, None])

    p_gwr = predict_weighted_linear_cv(X, y, Wg)
    rows.append({"Model": "GWR", **metrics_dict(y, p_gwr)})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: GWR")

    rf_params = {"n_estimators": cfg["rf_cv_trees"], "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None}
    p_gwrf = predict_local_rf_cv(X, y, Wg, rf_params, cfg["seed"], 40000)
    rows.append({"Model": "GWRF", **metrics_dict(y, p_gwrf)})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: GWRF")

    Wgs, *_ = build_weight_components(d_geo, d_attr, k_geo, alpha)
    p_sgwr = predict_weighted_linear_cv(X, y, Wgs)
    rows.append({"Model": "SGWR", **metrics_dict(y, p_sgwr)})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: SGWR")

    p_sgwrf = predict_local_rf_cv(X, y, Wgs, rf_params, cfg["seed"], 41000)
    rows.append({"Model": "SGWRF (LOOCV)", **metrics_dict(y, p_sgwrf)})
    step += 1
    if progress_cb: progress_cb(step / steps, "Baseline: SGWRF (LOOCV)")

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


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

    st.subheader("3️⃣ Bandwidth Geografis (k)")
    default_min_k = min(5, max(2, n_points - 1))
    min_k_geo, max_k_geo = st.slider("Rentang k tetangga terdekat geografis", 2, max(3, n_points - 1),
                                      (default_min_k, min(max(default_min_k + 5, 6), n_points - 1)))
    leave_target_out = st.checkbox("Leave-target-out (LOO) saat validasi/optimasi", value=True,
                                    help="Dipakai pada tahap optimasi k_geo, alpha, RF, dan baseline. "
                                         "Model final tetap dilatih menyertakan bobot titik itu sendiri "
                                         "sesuai desain SGWRF; evaluasi utama tetap memakai prediksi LOOCV.")

    st.subheader("4️⃣ Bobot α · γ (Persamaan 2.24 & AICc)")
    alpha_min, alpha_max = st.slider("Rentang alpha (kontribusi bobot geografis)", 0.01, 1.0, (0.05, 1.0), step=0.01)
    n_alpha = st.slider("Jumlah titik grid alpha", 5, 40, 20, step=1)

    n_k_geo = max_k_geo - min_k_geo + 1
    est_bw = n_k_geo * n_points
    est_alpha = n_alpha * n_points
    if n_points > 60 or (est_bw + est_alpha) > 15000:
        st.warning(
            f"⏱️ Data Anda memiliki **{n_points} titik**. Estimasi ~{est_bw:,} pelatihan model pada "
            f"optimasi bandwidth geografis dan ~{est_alpha:,} pelatihan pada optimasi alpha. Ini bisa "
            "memakan waktu cukup lama. Disarankan: persempit rentang k, kurangi jumlah titik grid alpha, "
            "atau kurangi jumlah pohon RF tahap CV di bawah."
        )

    st.subheader("5️⃣ Parameter Random Forest")
    rf_cv_trees = st.slider("Jumlah pohon (tahap optimasi/CV)", 20, 300, 80, step=10)
    rf_final_trees = st.slider("Jumlah pohon (model final)", 100, 800, 300, step=50,
                                help="Treewise permutation importance (Pers. 2.35) dihitung untuk SETIAP "
                                     "pohon di model final — makin banyak pohon, makin lama waktu komputasi.")
    seed = st.number_input("Random seed", 0, 99999, 2026)

    st.subheader("6️⃣ Analisis Tambahan")
    run_baseline_flag = st.checkbox("Jalankan model baseline (RF, GWR, GWRF, SGWR, SGWRF-LOOCV)", value=True)

    st.markdown("---")
    run_button = st.button("🚀 Jalankan Analisis SGWRF", use_container_width=True)

cfg = dict(min_k_geo=min_k_geo, max_k_geo=max_k_geo, alpha_min=float(alpha_min), alpha_max=float(alpha_max),
           n_alpha=int(n_alpha), leave_target_out=leave_target_out, rf_cv_trees=rf_cv_trees,
           rf_final_trees=rf_final_trees, seed=int(seed))

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

tab_data, tab_map, tab_bw, tab_alpha, tab_rf, tab_local, tab_baseline, tab_download = st.tabs(
    ["📊 Data & Eksplorasi", "🗺️ Peta Interaktif", "🎯 Bandwidth Geografis", "⚖️ Optimasi α (AICc)",
     "🌲 Optimasi Random Forest", "📈 Model Lokal SGWRF", "🆚 Perbandingan Baseline", "📥 Unduh Hasil"]
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

    # --- 1) Bandwidth geografis (k_geo), alpha awal = 0.5 sebagai skala awal ---
    prog = st.progress(0.0, text="Optimasi bandwidth geografis (k_geo)...")
    def cb_bw(frac, text):
        prog.progress(min(frac, 1.0), text=text)
    best_k_geo, bw_results, elapsed_bw = optimize_geographic_bandwidth(
        X, y, d_geo, d_attr, 0.5, cfg, progress_cb=cb_bw
    )
    prog.progress(1.0, text=f"Bandwidth geografis selesai ({elapsed_bw:.1f}s) — k*={best_k_geo}")

    # --- 2) Optimasi alpha/gamma via AICc ---
    prog_a = st.progress(0.0, text="Optimasi alpha & gamma (AICc)...")
    def cb_a(frac, text):
        prog_a.progress(min(frac, 1.0), text=text)
    best_alpha, best_gamma, alpha_results, elapsed_a = optimize_alpha(
        X, y, d_geo, d_attr, best_k_geo, cfg, progress_cb=cb_a
    )
    prog_a.progress(1.0, text=f"Optimasi alpha selesai ({elapsed_a:.1f}s) — α*={best_alpha:.3f}")

    Wgs, Wg, Ws, bg_local, gamma_check = build_weight_components(d_geo, d_attr, best_k_geo, best_alpha)

    # --- 3) Optimasi hyperparameter RF ---
    prog2 = st.progress(0.0, text="Optimasi hyperparameter Random Forest...")
    def cb_rf(frac, text):
        prog2.progress(min(frac, 1.0), text=text)
    best_rf, rf_results = optimize_rf(X, y, Wgs, cfg, RF_PARAM_GRID, progress_cb=cb_rf)
    prog2.progress(1.0, text="Optimasi Random Forest selesai")

    # --- 4) Model lokal final (in-sample, self-weight ikut serta) ---
    prog3 = st.progress(0.0, text="Melatih model lokal SGWRF final per titik...")
    def cb_local(frac, text):
        prog3.progress(min(frac, 1.0), text=text)
    pred_in, local_r2, local_mae, local_rmse, importances = train_local_models(
        X, y, Wgs, df, name_col, x_cols, x_labels, best_rf, cfg, progress_cb=cb_local
    )
    prog3.progress(1.0, text="Model lokal final selesai")

    # --- 5) Prediksi LOOCV (evaluasi utama, tanpa kebocoran data) ---
    prog3b = st.progress(0.0, text="Menghitung prediksi LOOCV (evaluasi utama)...")
    final_rf_cv_params = {
        "n_estimators": cfg["rf_cv_trees"],
        "max_features": best_rf.get("max_features", "sqrt"),
        "min_samples_leaf": best_rf.get("min_samples_leaf", 1),
        "max_depth": best_rf.get("max_depth", None),
    }
    pred_cv = predict_local_rf_cv(X, y, Wgs, final_rf_cv_params, cfg["seed"], 50000)
    prog3b.progress(1.0, text="Prediksi LOOCV selesai")

    overall_cv = metrics_dict(y, pred_cv)
    overall_in = metrics_dict(y, pred_in)

    results = build_results(df, id_col, name_col, lat_col, lon_col, y_col, x_cols, x_labels,
                             y, pred_in, pred_cv, local_r2, local_mae, local_rmse, importances,
                             bg_local, best_alpha, best_gamma)

    baseline_results = None
    if run_baseline_flag:
        prog4 = st.progress(0.0, text="Menjalankan model baseline pembanding...")
        def cb_base(frac, text):
            prog4.progress(min(frac, 1.0), text=text)
        baseline_results = run_baselines(X, y, d_geo, d_attr, best_k_geo, best_alpha, cfg, progress_cb=cb_base)
        prog4.progress(1.0, text="Baseline selesai")

    st.session_state["sgwrf"] = dict(
        results=results, bw_results=bw_results, alpha_results=alpha_results, rf_results=rf_results,
        best_k_geo=best_k_geo, best_alpha=best_alpha, best_gamma=best_gamma, best_rf=best_rf,
        overall_cv=overall_cv, overall_in=overall_in, importances=importances,
        baseline_results=baseline_results, x_cols=x_cols, x_labels=x_labels,
        name_col=name_col, id_col=id_col,
    )
    st.success("✅ Analisis SGWRF selesai! Lihat hasil pada tab lainnya.")

# ================================================================
# 7. TAMPILKAN HASIL (jika sudah pernah dijalankan)
# ================================================================

state = st.session_state.get("sgwrf")

if state is None:
    for t in (tab_bw, tab_alpha, tab_rf, tab_local, tab_baseline, tab_download):
        with t:
            st.info("⬅️ Atur parameter di sidebar lalu tekan **Jalankan Analisis SGWRF** untuk melihat hasil di sini.")
else:
    results = state["results"]
    bw_results = state["bw_results"]
    alpha_results = state["alpha_results"]
    rf_results = state["rf_results"]
    best_k_geo = state["best_k_geo"]
    best_alpha = state["best_alpha"]
    best_gamma = state["best_gamma"]
    best_rf = state["best_rf"]
    overall_cv = state["overall_cv"]
    overall_in = state["overall_in"]
    importances = state["importances"]
    baseline_results = state["baseline_results"]

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

    # -------- tab bandwidth geografis --------
    with tab_bw:
        st.subheader("Optimasi Bandwidth Geografis Adaptif (k_geo)")
        st.markdown(
            '<div class="sgwrf-eq">b_g(i) = jarak ke tetangga ke-k terdekat &nbsp;|&nbsp; '
            'w_ij^G = exp[-(d_ij^G)² / (2·b_g²)] — Persamaan (2.10)-(2.11)</div>',
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        b1.metric("k geografis optimum (k*)", best_k_geo)
        b2.metric("RMSE CV minimum", f"{bw_results['RMSE_CV'].min():.4f}")

        fig_bw = px.line(bw_results.sort_values("k_geo"), x="k_geo", y="RMSE_CV", markers=True,
                          title="RMSE-CV vs Bandwidth Geografis (k)")
        fig_bw.add_vline(x=best_k_geo, line_dash="dash", line_color="#e74c3c",
                          annotation_text=f"k*={best_k_geo}")
        st.plotly_chart(fig_bw, use_container_width=True)
        note(
            f"Setiap titik pada grafik adalah RMSE hasil validasi silang **leave-target-out** untuk satu "
            f"nilai k (jumlah tetangga geografis terdekat yang dipakai membentuk bandwidth adaptif b_g). "
            f"Titik terendah (garis merah putus-putus) adalah **k* = {best_k_geo}** yang dipilih otomatis. "
            f"k kecil membuat model sangat lokal (risiko overfit pada data sedikit), k besar membuat "
            f"bandwidth melebar sehingga model mendekati model global (kurang menangkap heterogenitas "
            f"spasial). Nilai k_geo ini dipakai tetap pada tahap optimasi alpha berikutnya."
        )
        st.dataframe(bw_results, use_container_width=True, height=280)

    # -------- tab alpha / AICc --------
    with tab_alpha:
        st.subheader("Optimasi Bobot α · γ via AICc")
        st.markdown(
            '<div class="sgwrf-eq">W_GS = α·W_G + γ·W_S,  γ = 1-α — Persamaan (2.24)<br>'
            'AICc = 2n·ln(RSS/n) + n·ln(2π) + n + 2(ENP+1)(ENP+2)/(n-ENP-2) — Persamaan (2.29)</div>',
            unsafe_allow_html=True,
        )
        a1, a2, a3 = st.columns(3)
        a1.metric("alpha optimum (α*)", f"{best_alpha:.3f}")
        a2.metric("gamma optimum (γ*)", f"{best_gamma:.3f}")
        a3.metric("AICc minimum", f"{alpha_results['AICc'].min():.2f}")

        fig_a = px.line(alpha_results.sort_values("alpha"), x="alpha", y="AICc", markers=True,
                         title="Kurva AICc terhadap Alpha")
        fig_a.add_vline(x=best_alpha, line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"α*={best_alpha:.3f}")
        st.plotly_chart(fig_a, use_container_width=True)
        note(
            f"**α* = {best_alpha:.3f}** berarti bobot geografis (W_G, kedekatan lokasi) berkontribusi "
            f"**{best_alpha*100:.1f}%**, sedangkan bobot kemiripan atribut/kovariat (W_S) berkontribusi "
            f"**{best_gamma*100:.1f}%** (γ* = {best_gamma:.3f}) dalam pembentukan bobot gabungan W_GS. "
            f"AICc dipilih karena menyeimbangkan **kecocokan model** (RSS kecil) dengan **kompleksitas "
            f"efektif model** (ENP = trace(S) — proksi jumlah parameter efektif, karena Random Forest "
            f"tidak memiliki hat-matrix linear eksplisit seperti pada GWR klasik). Titik terendah pada "
            f"kurva (garis merah) adalah kombinasi α-γ terbaik."
        )

        fig_enp = px.line(alpha_results.sort_values("alpha"), x="alpha", y="ENP", markers=True,
                           title="ENP (Kompleksitas Efektif Model) terhadap Alpha")
        st.plotly_chart(fig_enp, use_container_width=True)
        note(
            "ENP (*Effective Number of Parameters*) menggambarkan seberapa fleksibel/kompleks model pada "
            "tiap nilai alpha. ENP yang naik seiring perubahan alpha menunjukkan model menjadi lebih "
            "fleksibel (berisiko overfit); AICc menghukum kenaikan ENP yang tidak diimbangi penurunan RSS "
            "yang sepadan."
        )
        st.dataframe(alpha_results, use_container_width=True, height=280)

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
            f"`max_features`, `min_samples_leaf`) yang diuji lewat validasi silang lokal menggunakan bobot "
            f"gabungan W_GS (α*, γ* dari tab sebelumnya). Batang **terendah** dipilih sebagai konfigurasi "
            f"final: `max_features={best_rf.get('max_features')}`, "
            f"`min_samples_leaf={best_rf.get('min_samples_leaf')}`. Konfigurasi ini lalu dipakai untuk "
            f"melatih model lokal final dengan jumlah pohon yang lebih besar (lihat pengaturan sidebar) "
            f"agar hasil akhir lebih stabil."
        )
        st.dataframe(rf_results, use_container_width=True)

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
            f"Metrik ini dihitung dari prediksi **leave-one-out cross-validation (LOOCV)**: bobot titik itu "
            f"sendiri dibuat nol saat memprediksinya, sehingga hasilnya tidak bias/optimistik dan menjadi "
            f"**metrik evaluasi utama** untuk model SGWRF. **R² = {overall_cv['R2']:.4f}** berarti model "
            f"menjelaskan sekitar **{max(overall_cv['R2'], 0)*100:.1f}%** variasi {y_col} pada data yang "
            f"tidak dilihat model saat memprediksi titik tersebut (kualitas model tergolong **{kualitas}**)."
        )

        with st.expander("📎 Diagnostik in-sample (bobot titik sendiri ikut serta — cenderung optimistik)"):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("RMSE (in-sample)", f"{overall_in['RMSE']:.4f}")
            d2.metric("MAE (in-sample)", f"{overall_in['MAE']:.4f}")
            d3.metric("MAPE (in-sample)", f"{overall_in['MAPE']:.2f}%")
            d4.metric("R² (in-sample)", f"{overall_in['R2']:.4f}")
            st.caption(
                "Model final SGWRF dilatih TANPA menolkan bobot titik itu sendiri (sesuai desain: semua "
                "titik menjadi penyumbang informasi lokal). Karena itu nilai in-sample biasanya lebih "
                "baik daripada LOOCV. Gunakan metrik LOOCV di atas sebagai acuan evaluasi utama."
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
            "mengikuti Persamaan 2.10), sementara `alpha`/`gamma` adalah bobot α*/γ* global hasil optimasi "
            "AICc yang berlaku untuk seluruh titik."
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
                '<b>global</b> hasil optimasi AICc yang sama untuk semua titik pada Persamaan (2.24). '
                'Model SGWRF utamanya adalah Random Forest lokal berbobot — bukan model linear, sehingga '
                'tidak memiliki koefisien beta seperti GWR klasik. Ukuran kepentingan variabel yang sah '
                'adalah treewise permutation importance (VI_ik) pada tabel di atas.</div>',
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
            st.subheader("Perbandingan SGWRF vs Model Baseline")
            st.dataframe(baseline_results, use_container_width=True)

            fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("RMSE (lebih rendah lebih baik)", "R² (lebih tinggi lebih baik)"))
            fig_bar.add_trace(go.Bar(x=baseline_results["Model"], y=baseline_results["RMSE"],
                                      marker_color="#e74c3c", name="RMSE"), row=1, col=1)
            fig_bar.add_trace(go.Bar(x=baseline_results["Model"], y=baseline_results["R2"],
                                      marker_color="#27ae60", name="R2"), row=1, col=2)
            fig_bar.update_layout(height=430, showlegend=False, title="Ringkasan Kinerja Model (semua LOOCV kecuali RF global)")
            st.plotly_chart(fig_bar, use_container_width=True)
            sgwrf_rank = int((baseline_results["RMSE"] < overall_cv["RMSE"]).sum())
            posisi = "lebih baik dari seluruh" if sgwrf_rank == 0 else f"lebih baik dari {len(baseline_results)-sgwrf_rank} dari {len(baseline_results)}"
            note(
                f"Model utama **SGWRF** (tab Model Lokal SGWRF) memiliki RMSE LOOCV **{overall_cv['RMSE']:.4f}** "
                f"dan R² LOOCV **{overall_cv['R2']:.4f}** — dibandingkan baseline di atas, SGWRF saat ini "
                f"{posisi} model pembanding. Baris **SGWRF (LOOCV)** pada tabel baseline memakai bobot dan "
                f"skema LOOCV yang sama seperti model utama, sehingga nilainya seharusnya mendekati metrik "
                f"LOOCV di tab Model Lokal SGWRF (perbedaan kecil dapat terjadi karena penyesuaian "
                f"parameter/seed). Jika SGWRF **tidak** mengungguli GWRF/RF secara meyakinkan, pertimbangkan "
                f"menambah kovariat relevan, menambah titik observasi, atau memperluas rentang pencarian "
                f"bandwidth/alpha di sidebar."
            )

            st.markdown(
                '<div class="sgwrf-note">Baseline: <b>RF</b> (Random Forest tanpa pembobotan lokal), '
                '<b>GWR</b> (regresi linear terboboti W_G/geografis, LOOCV), <b>GWRF</b> (RF terboboti '
                'W_G, LOOCV), <b>SGWR</b> (regresi linear terboboti W_GS = α·W_G+γ·W_S, LOOCV), '
                '<b>SGWRF (LOOCV)</b> (RF terboboti W_GS, LOOCV — setara model utama). Semua bobot memakai '
                'k_geo* dan α*/γ* hasil optimasi pada tab sebelumnya.</div>',
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
            st.download_button("⬇️ Optimasi Bandwidth Geografis (.csv)", to_csv_bytes(bw_results),
                                "hasil_optimasi_bandwidth_geografis.csv", "text/csv", use_container_width=True)
        with dl2:
            st.download_button("⬇️ Optimasi Alpha / AICc (.csv)", to_csv_bytes(alpha_results),
                                "hasil_optimasi_alpha_AICc.csv", "text/csv", use_container_width=True)
            st.download_button("⬇️ Optimasi Random Forest (.csv)", to_csv_bytes(rf_results),
                                "hasil_optimasi_rf.csv", "text/csv", use_container_width=True)
        with dl3:
            vi_export = pd.DataFrame(importances, columns=state["x_cols"])
            vi_export.insert(0, "point_no", results["point_no"].to_numpy())
            st.download_button("⬇️ Local Variable Importance (.csv)", to_csv_bytes(vi_export),
                                "local_variable_importance.csv", "text/csv", use_container_width=True)
            if baseline_results is not None:
                st.download_button("⬇️ Perbandingan Model Baseline (.csv)", to_csv_bytes(baseline_results),
                                    "perbandingan_model.csv", "text/csv", use_container_width=True)

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="hasil_lokal_sgwrf", index=False)
            bw_results.to_excel(writer, sheet_name="optimasi_bandwidth_geo", index=False)
            alpha_results.to_excel(writer, sheet_name="optimasi_alpha_AICc", index=False)
            rf_results.to_excel(writer, sheet_name="optimasi_rf", index=False)
            vi_export.to_excel(writer, sheet_name="variable_importance", index=False)
            if baseline_results is not None:
                baseline_results.to_excel(writer, sheet_name="perbandingan_baseline", index=False)
        st.download_button("⬇️ Semua Hasil (Excel, multi-sheet)", excel_buf.getvalue(),
                            "hasil_sgwrf_lengkap.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)

st.markdown("---")
st.caption(
    "Dashboard SGWRF — pipeline: standarisasi Z-score (2.3-2.5) → jarak geografis Euclidean (2.10) + "
    "kernel Gaussian adaptif (2.11) → jarak atribut mean|Z_i-Z_j| (2.18-2.19) + similarity weight (2.20) "
    "→ kombinasi aditif W_GS = α·W_G + γ·W_S (2.24) → optimasi k_geo & alpha/gamma via AICc (2.28-2.33) → "
    "optimasi hyperparameter RF → model lokal RF final per titik → treewise permutation importance (2.35) "
    "→ evaluasi LOOCV (2.36-2.38) → model baseline pembanding (RF, GWR, GWRF, SGWR, SGWRF)."
)
