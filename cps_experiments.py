#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 cps_experiments.py  --  Section V experiment engine (E1 ... E8)

 Paper : "Distribution-Free Stealthy False Data Injection Attacks ..."  (L-CSS)
 Plan  : experiment_plan.md
 Body  : 1_body_rev1.tex

 Everything is organised in numbered sections; the companion notebook
 (local_notebook.ipynb) walks the *same* sections in the *same* order and
 calls the *same* functions, so a result obtained on a laptop and a result
 obtained on OSC differ only in (T_dep, n_mc).

   S0  Configuration
   S1  Plant, DARE, regimes, vectorised simulation
   S2  Attack action, remote-estimator divergence, damage metrics
   S3  Scores  (B1/B2/B3, P, A1-A4)
   S4  Thresholds  (Gaussian inverse / naive / conformal)
   S5  Detectors   (Gmag: chi2, windowed chi2, CUSUM;  Gsgn: acf1, LB, runs)
   S6  Autoencoder (torch, trained once per dataset)
   S7  Experiments E1 ... E8
   S8  CLI

 Usage
 -----
   python cps_experiments.py --exp E0             # pretrain the 4 AEs
   python cps_experiments.py --exp E1             # budget accuracy
   python cps_experiments.py --exp all --quick    # smoke test, ~2 min

 Every experiment writes results/E<k>.csv (tidy long format, one row per
 replicate) and results/E<k>_meta.json.  Aggregation, LaTeX tables and the
 two paper figures are produced by collect_results.py.
=============================================================================
"""

# =============================================================================
# S0.  Imports and configuration
# =============================================================================
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.signal import lfilter
from scipy.stats import norm

# torch is only needed for scheduler P; imported lazily in S6 so that every
# other experiment runs on a machine without it.
_TORCH_OK = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:                                        # pragma: no cover
    _TORCH_OK = False


@dataclass
class Config:
    """Single source of truth for every experiment.  Overridden from the CLI."""

    # ---- reproducibility -----------------------------------------------
    seed: int = 0

    # ---- stream lengths (Sec. V-A protocol) ----------------------------
    T_burn: int = 4_000          # discarded filter transient
    N_cal: int = 5_000           # calibration record  (N in Thm. 4 / Thm. 6)
    T_dep: int = 50_000          # deployment record   (T in Thm. 4)
    T_train: int = 60_000        # AE / PCA / AR training stream (disjoint)
    n_mc: int = 200              # Monte-Carlo seeds, budget experiments
    n_mc_impact: int = 50        # Monte-Carlo seeds, damage experiments (E7/E8)

    # ---- budget grid, identical everywhere  (m7) ------------------------
    budgets: tuple = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)

    # ---- scores ---------------------------------------------------------
    L: int = 50                  # score window length (AE, PCA, energy)
    ar_order: int = 5            # A2: AR(p) on magnitudes
    pca_rank: int = 3            # A1: linear / PCA autoencoder rank
    ae_score: str = "window"     # "window" = ||u - AE(u)||^2 (eq. 16); or "last"

    # ---- autoencoder ----------------------------------------------------
    hidden_dim: int = 128
    latent_dim: int = 1
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    n_epochs: int = 50
    early_stop: int = 10
    infer_batch: int = 8192

    # ---- detectors ------------------------------------------------------
    fa_target: float = 0.01      # nominal false-alarm rate of every detector
    chi2_window: int = 50        # W for the windowed chi2 detector
    cusum_drift: float = 1.5     # CUSUM slack b (units of nominal z^2/S = 1)
    sgn_windows: tuple = (500, 2000, 5000, 20_000)   # E4 test-window lengths
    n_null_windows: int = 400    # nominal windows used to calibrate Gsgn

    # ---- E2 / E5 / E6 ---------------------------------------------------
    e2_N_grid: tuple = (100, 200, 500, 1000, 5000)
    e2_budgets: tuple = (0.02, 0.10)
    e2_reps: int = 2_000
    e5_f_grid: tuple = (0.0, 0.5, -0.5, 0.7, -0.7, 0.9, -0.9, 1.0, -1.0)
    e5_N_grid: tuple = (200, 500, 1000, 5000)
    e5_reps: int = 2_000
    e5_alpha: float = 0.05
    e6_N_grid: tuple = (200, 500, 1000, 5000, 20_000)
    e6_T_truth: int = 2_000_000  # long stream defining the "true" beta

    # ---- paths ----------------------------------------------------------
    out_dir: str = "results"
    fig_dir: str = "figures"
    ckpt_dir: str = "models_checkpoint"
    device: str = "auto"

    # ---- switches -------------------------------------------------------
    skip_ae: bool = False        # drop scheduler P (structure-only test run)
    train_ae: bool = True

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if _TORCH_OK and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def dirs(self):
        for d in (self.out_dir, self.fig_dir, self.ckpt_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


CFG = Config()          # module-level default, convenient in the notebook


def quick(cfg: Config) -> Config:
    """Shrink everything so the whole suite runs in a couple of minutes."""
    cfg.T_burn, cfg.N_cal, cfg.T_dep, cfg.T_train = 500, 2_000, 8_000, 12_000
    cfg.n_mc, cfg.n_mc_impact = 12, 8
    cfg.n_epochs, cfg.early_stop = 4, 3
    cfg.e2_reps, cfg.e5_reps = 300, 300
    cfg.e2_N_grid, cfg.e5_N_grid = (100, 500, 1000), (200, 1000)
    cfg.e6_N_grid, cfg.e6_T_truth = (200, 1000, 5000), 200_000
    cfg.sgn_windows, cfg.n_null_windows = (500, 2000), 120
    return cfg


# =============================================================================
# S1.  Plant P1, DARE, regimes, vectorised simulation
# =============================================================================
#   x_{k+1} = A x_k + w_k          w_k ~ (0, Q)
#   y_k     = C x_k + v_k          v_k ~ (0, R_k)      (Sec. II-A, eq. 1)
#
#   Sensor-side steady-state KF (Assumption 2: never sees the remote estimate)
#       p_k      = xhat^s_{k|k-1}          p_0 = 0
#       z_k      = y_k - C p_k                                       (eq. 3)
#       xhat^s_k = p_k + K z_k
#       p_{k+1}  = A(I-KC) p_k + A K y_k   <-- LTI recursion, vectorisable
# -----------------------------------------------------------------------------

A_P1 = np.array([[0.95, 0.02], [0.00, 0.90]])
C_P1 = np.array([[1.0, 0.0]])
Q_P1 = 0.01 * np.eye(2)
R_P1 = 0.05

# Gaussian mixture N2: p=0.05, variance matched to R, kexc(v) = 17.0
P_MIX, RATIO_MIX = 0.05, 25.0
S1_MIX = R_P1 / ((1 - P_MIX) + P_MIX * RATIO_MIX)
S2_MIX = RATIO_MIX * S1_MIX
VAR_MIX = (1 - P_MIX) * S1_MIX + P_MIX * S2_MIX               # == R_P1
KEXC_MIX = 3 * ((1 - P_MIX) * S1_MIX**2 + P_MIX * S2_MIX**2) / VAR_MIX**2 - 3

# ---- the four datasets of Sec. V-A ------------------------------------------
#   noise    : marginal law of v_k
#   R_design : variance the *defender's* filter is designed with
#   tv       : time-varying true variance R_k = R (1 + 0.5 sin(2 pi k / 2000))
REGIMES = {
    "N1": dict(noise="gauss", R_design=R_P1,       tv=False,
               label="Gaussian"),
    "N2": dict(noise="mix",   R_design=R_P1,       tv=False,
               label="heavy-tailed"),
    "MM": dict(noise="gauss", R_design=4 * R_P1,   tv=False,
               label="model mismatch"),
    "TV": dict(noise="gauss", R_design=R_P1,       tv=True,
               label="time-varying"),
}


def solve_kf(A, C, Q, R_scalar):
    """Steady-state prior covariance, innovation covariance, gain."""
    R = np.atleast_2d(float(R_scalar))
    Pbar = solve_discrete_are(A.T, C.T, Q, R)
    S = C @ Pbar @ C.T + R
    K = Pbar @ C.T @ np.linalg.inv(S)
    return Pbar, float(S[0, 0]), K


def lin_rec(M, U):
    """
    s_k = M s_{k-1} + u_k with s_{-1} = 0, for U of shape (T, n).

    Diagonalises M and runs one first-order IIR per mode, which turns an
    O(T) Python loop into vectorised filtering.  Falls back to the explicit
    loop if M is (numerically) defective.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    T, n = U.shape
    try:
        lam, V = np.linalg.eig(M)
        Vi = np.linalg.inv(V)
        if np.linalg.cond(V) > 1e8:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:                        # pragma: no cover
        S = np.zeros((T, n))
        s = np.zeros(n)
        for k in range(T):
            s = M @ s + U[k]
            S[k] = s
        return S
    G = U @ Vi.T                                          # modal input
    Y = np.empty_like(G, dtype=complex)
    for i, l in enumerate(lam):
        Y[:, i] = lfilter([1.0], [1.0, -l], G[:, i].astype(complex))
    return np.real(Y @ V.T)


def draw_noise(rng, T, noise, R_seq):
    """v_k with unit-shape law `noise` scaled to per-step variance R_seq."""
    if noise == "gauss":
        v = rng.standard_normal(T)
        return v * np.sqrt(R_seq)
    # two-component zero-mean mixture, unit variance, then rescaled
    heavy = rng.random(T) < P_MIX
    sd = np.where(heavy, math.sqrt(S2_MIX), math.sqrt(S1_MIX))
    v = rng.standard_normal(T) * sd / math.sqrt(VAR_MIX)   # unit variance
    return v * np.sqrt(R_seq)


def simulate_stream(regime, T, seed, cfg=CFG, burn=None):
    """
    One nominal (attack-free) realisation of plant + sensor-side KF.

    Returns a dict with, all post-burn-in and of length T:
        z    (T,)   transmitted innovation                        eq. (3)
        y    (T,)   measurement
        es   (T,2)  sensor-side posterior error x_k - xhat^s_k
        S_design    innovation covariance the model-aware party believes
        S_emp       realised Var(z)  (differs from S_design under MM / TV)
        K, A, C     filter data
    """
    rg = REGIMES[regime]
    burn = cfg.T_burn if burn is None else burn
    Tt = T + burn
    _, S_des, K = solve_kf(A_P1, C_P1, Q_P1, rg["R_design"])
    Acl = A_P1 @ (np.eye(2) - K @ C_P1)

    rng = np.random.default_rng(seed)
    k_idx = np.arange(Tt)
    R_seq = (R_P1 * (1 + 0.5 * np.sin(2 * np.pi * k_idx / 2000.0))
             if rg["tv"] else np.full(Tt, R_P1))

    w = rng.multivariate_normal(np.zeros(2), Q_P1, size=Tt)
    v = draw_noise(rng, Tt, rg["noise"], R_seq)

    x = lin_rec(A_P1, w)                                  # x_k = A x_{k-1}+w_k
    y = (C_P1 @ x.T).ravel() + v

    s = lin_rec(Acl, y[:, None] * (A_P1 @ K).ravel()[None, :])
    p = np.vstack([np.zeros((1, 2)), s[:-1]])             # p_k = xhat^s_{k|k-1}
    z = y - (C_P1 @ p.T).ravel()
    xs_post = p + z[:, None] * K.ravel()[None, :]
    es = x - xs_post

    sl = slice(burn, None)
    return dict(z=z[sl], y=y[sl], es=es[sl], v=v[sl],
                S_design=S_des, S_emp=float(np.var(z[sl])),
                K=K, A=A_P1, C=C_P1, regime=regime)


# =============================================================================
# S2.  Attack action, remote divergence, damage metrics
# =============================================================================
#   z^c_k = (1 - 2 gamma_k) z_k                                    (eq. 11)
#   d_k   = xhat^s_k - xhat^a_k = A d_{k-1} + 2 gamma_k K z_k
#   e^a_k = x_k - xhat^a_k = e^s_k + d_k
# -----------------------------------------------------------------------------

def divergence(z, gamma, K, A=A_P1):
    """d_k, the sensor-vs-remote estimate divergence, driven by the flip."""
    u = (2.0 * (gamma.astype(float) * z))[:, None] * K.ravel()[None, :]
    return lin_rec(A, u)


def damage_metrics(z, es, gamma, K, k0=0, A=A_P1):
    """
    D3-clean split of the remote mean-square error:
        MSE = tr(Cov(e^a)) + ||E[e^a]||^2 .
    Everything is evaluated on k >= k0 so that all schedulers share the
    same denominator regardless of their score warm-up.
    """
    d = divergence(z, gamma, K, A)
    ea = es + d
    ea = ea[k0:]
    mu = ea.mean(axis=0)
    mse = float(np.mean(np.sum(ea * ea, axis=1)))
    bias_sq = float(mu @ mu)
    nom = es[k0:]
    gk = gamma[k0:]
    zk = np.abs(z[k0:])
    fired = zk[gk] if gk.any() else np.array([0.0])
    runs = 1 + int(np.sum(gk[1:] != gk[:-1])) if len(gk) > 1 else 1
    return dict(
        mse=mse,
        cov_trace=mse - bias_sq,
        bias_sq=bias_sq,
        mse_nominal=float(np.mean(np.sum(nom * nom, axis=1))),
        peak_div=float(np.max(np.linalg.norm(d[k0:], axis=1))),
        rate=float(np.mean(gk)),
        n_fire=int(np.sum(gk)),
        # why one scheduler beats another, in one number: the divergence is
        # driven by 2 gamma_k K z_k, so the energy captured at firing instants
        # is the whole story for a memoryless driving term.
        mean_abs_z_fired=float(np.mean(fired)),
        mean_z2_fired=float(np.mean(fired ** 2)),
        fire_clustering=float(np.sum(gk) / max(runs / 2.0, 1.0)),
    )


def guo_steady_state(Gbar, beta, S, K, A=A_P1):
    """
    tr(P^a_inf) driver of [Guo2023, Thm. 2] written through the divergence
    recursion:   D = A D A' + 4 Psi(Gbar,beta) K S K',
                 Psi = 1 - (1-Gbar)(1-beta).
    Returns (Psi, tr(D_inf)).
    """
    Psi = 1.0 - (1.0 - Gbar) * (1.0 - beta)
    Wm = 4.0 * Psi * (K @ K.T) * S
    D = solve_discrete_lyapunov(A, Wm)
    return float(Psi), float(np.trace(D))


# =============================================================================
# S3.  Scores  --  every score is a function of magnitudes only (Thm. 2)
# =============================================================================
#   B1/B2  zeta_k = |z_k| / sqrt(S_design)     model-based, whitened
#   B3     |z_k|                               model-free, memoryless
#   P      ||u_k - AE(u_k)||^2                 eq. (16), u_k = |z_{k-L+1:k}|
#   A1     PCA (linear AE) reconstruction error on the same windows
#   A2     AR(p) one-step prediction error on |z|
#   A3     windowed energy  sum |z_j|^2
#   A4     | |z_k| - |z_{k-1}| |
#
# Convention: every score has length T with np.nan on indices where it is
# undefined.  All firing rules and all rates are evaluated on k >= cfg.L.
# -----------------------------------------------------------------------------

SCHEDULERS = ["B1", "B2", "B3", "P", "A1", "A2", "A3", "A4"]
ABLATIONS = ["B2", "B3", "A1", "A2", "A3", "A4", "P"]


def _nanpad(T, L):
    s = np.full(T, np.nan)
    return s


def windows_of(mag, L):
    """(T-L+1, L) sliding-window view of a magnitude series -- no copy."""
    return np.lib.stride_tricks.sliding_window_view(mag, L)


def score_whitened(z, S_design):
    return np.abs(z) / math.sqrt(S_design)


def score_magnitude(z):
    return np.abs(z)


def score_diff(z):
    s = _nanpad(len(z), 1)
    m = np.abs(z)
    s[1:] = np.abs(m[1:] - m[:-1])
    return s


def score_energy(z, L):
    s = _nanpad(len(z), L)
    e = z ** 2
    cs = np.concatenate([[0.0], np.cumsum(e)])
    s[L - 1:] = cs[L:] - cs[:-L]
    return s


def fit_ar(mag_train, p):
    """Least-squares AR(p) on nominal magnitudes (frozen afterwards)."""
    X = np.lib.stride_tricks.sliding_window_view(mag_train, p)[:-1]
    yv = mag_train[p:]
    X1 = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(X1, yv, rcond=None)
    return coef


def score_ar(z, coef, p, L):
    s = _nanpad(len(z), L)
    m = np.abs(z)
    X = np.lib.stride_tricks.sliding_window_view(m, p)[:-1]
    X1 = np.hstack([X, np.ones((len(X), 1))])
    pred = X1 @ coef
    s[p:] = (m[p:] - pred) ** 2
    s[:L] = np.nan                      # common warm-up
    return s


def fit_pca(mag_train, L, rank):
    """Linear (PCA) autoencoder on magnitude windows: mean + top-`rank` PCs."""
    Xw = windows_of(mag_train, L)
    mu = Xw.mean(axis=0)
    Xc = Xw - mu
    # economical SVD on a subsample keeps memory bounded
    idx = np.linspace(0, len(Xc) - 1, min(len(Xc), 20_000)).astype(int)
    _, _, Vt = np.linalg.svd(Xc[idx], full_matrices=False)
    return dict(mu=mu, W=Vt[:rank].T, L=L)


def score_pca(z, model):
    L = model["L"]
    s = _nanpad(len(z), L)
    Xw = windows_of(np.abs(z), L) - model["mu"]
    recon = (Xw @ model["W"]) @ model["W"].T
    s[L - 1:] = np.sum((Xw - recon) ** 2, axis=1)
    return s


def fit_scorers(train_stream, cfg=CFG, tag="N1", train_ae_flag=True):
    """
    Fit every data-driven score ONCE per dataset (Sec. V-A: the autoencoder is
    trained once per dataset, not per budget).  Returns a frozen dict.
    """
    mag = np.abs(train_stream["z"])
    sc = dict(L=cfg.L,
              ar=fit_ar(mag, cfg.ar_order),
              pca=fit_pca(mag, cfg.L, cfg.pca_rank),
              S_design=train_stream["S_design"],
              tag=tag)
    if not cfg.skip_ae:
        sc["ae"] = train_or_load_ae(mag, tag=tag, cfg=cfg,
                                    train_flag=train_ae_flag)
    return sc


def compute_scores(z, scorers, cfg=CFG, which=None):
    """dict scheduler -> score array (length len(z), NaN before cfg.L)."""
    which = which or SCHEDULERS
    out = {}
    for name in which:
        if name in ("B1", "B2"):
            s = score_whitened(z, scorers["S_design"]).copy()
        elif name == "B3":
            s = score_magnitude(z).copy()
        elif name == "A1":
            s = score_pca(z, scorers["pca"])
        elif name == "A2":
            s = score_ar(z, scorers["ar"], cfg.ar_order, cfg.L)
        elif name == "A3":
            s = score_energy(z, cfg.L)
        elif name == "A4":
            s = score_diff(z)
        elif name == "P":
            if cfg.skip_ae or "ae" not in scorers:
                continue
            s = ae_scores(scorers["ae"], np.abs(z), cfg)
        else:                                             # pragma: no cover
            raise KeyError(name)
        s = np.asarray(s, dtype=float).copy()
        s[:cfg.L] = np.nan                                # common warm-up k0=L
        out[name] = s
    return out


# =============================================================================
# S4.  Thresholds and firing rules
# =============================================================================

def thr_conformal(s_cal, Gbar):
    """eq. (17):  eps = s_( ceil((N+1)(1-Gbar)) ).  Finite-sample, Thm. 6."""
    s = np.sort(s_cal[np.isfinite(s_cal)])
    N = len(s)
    k = int(math.ceil((N + 1) * (1.0 - Gbar)))
    k = min(max(k, 1), N)
    return float(s[k - 1])


def thr_naive(s_cal, Gbar):
    """The index the conformal correction replaces: ceil(N(1-Gbar))."""
    s = np.sort(s_cal[np.isfinite(s_cal)])
    N = len(s)
    k = int(math.ceil(N * (1.0 - Gbar)))
    k = min(max(k, 1), N)
    return float(s[k - 1])


def thr_gaussian(Gbar):
    """[Guo2023, Thm. 1] for m=1:  Gamma = 2 Q(eps)  =>  eps = Phi^-1(1-G/2)."""
    return float(norm.ppf(1.0 - Gbar / 2.0))


def fire_threshold(s, eps, k0):
    g = np.zeros(len(s), dtype=bool)
    valid = np.isfinite(s)
    g[valid] = s[valid] > eps
    g[:k0] = False
    return g


def fire_top_rate(s, rate, k0):
    """
    Matched-*realised*-rate firing (M9): fire on the top ceil(rate * n_valid)
    scores of the deployment stream itself.  Used by E7/E8 so that every
    scheduler fires exactly the same number of times.
    """
    g = np.zeros(len(s), dtype=bool)
    idx = np.where(np.isfinite(s))[0]
    idx = idx[idx >= k0]
    if rate <= 0 or len(idx) == 0:
        return g
    n_fire = int(math.ceil(rate * len(idx)))
    n_fire = min(n_fire, len(idx))
    order = idx[np.argsort(s[idx])[::-1]]
    g[order[:n_fire]] = True
    return g


# =============================================================================
# S5.  Detectors
# =============================================================================
#  Gmag  (magnitude-measurable -- Thm. 2 applies, FAR preserved exactly)
#     memoryless chi2 :  g_k = z_k^2 / S
#     windowed  chi2 :  g_k = sum_{j=k-W+1}^{k} z_j^2 / S
#     CUSUM          :  g_k = max(0, g_{k-1} + z_k^2/S - b)
#
#  Gsgn  (sign-sensitive -- outside the guarantee, evaluated empirically)
#     lag-1 autocorrelation, Ljung-Box(10), runs test
#
#  Every threshold is calibrated EMPIRICALLY on nominal data to fa_target.
#  The Gaussian/asymptotic nulls are unusable under N2 and we report that.
# -----------------------------------------------------------------------------

def det_chi2(z, S):
    return z ** 2 / S


def det_chi2_window(z, S, W):
    e = z ** 2 / S
    cs = np.concatenate([[0.0], np.cumsum(e)])
    g = cs[1:] - np.concatenate([np.zeros(min(W, len(e))), cs[:max(0, len(e) - W)]])
    return g


def det_cusum(z, S, b):
    e = z ** 2 / S - b
    g = np.empty(len(e))
    acc = 0.0
    for k in range(len(e)):                # inherently sequential, O(T), cheap
        acc = max(0.0, acc + e[k])
        g[k] = acc
    return g


GMAG = {
    "chi2": lambda z, S, cfg: det_chi2(z, S),
    "chi2_win": lambda z, S, cfg: det_chi2_window(z, S, cfg.chi2_window),
    "cusum": lambda z, S, cfg: det_cusum(z, S, cfg.cusum_drift),
}


def stat_acf1(x):
    x = x - x.mean()
    den = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / den) if den > 0 else 0.0


def stat_ljungbox(x, nlags=10):
    x = x - x.mean()
    den = np.sum(x * x)
    if den <= 0:
        return 0.0
    T = len(x)
    q = 0.0
    for l in range(1, nlags + 1):
        r = np.sum(x[l:] * x[:-l]) / den
        q += r * r / (T - l)
    return float(T * (T + 2) * q)


def stat_runs(x):
    s = np.sign(x)
    s = s[s != 0]
    n = len(s)
    if n < 2:
        return 0.0
    npos = int(np.sum(s > 0))
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return 0.0
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2.0 * npos * nneg / n + 1.0
    var = (mu - 1.0) * (mu - 2.0) / (n - 1.0)
    return float((runs - mu) / math.sqrt(var)) if var > 0 else 0.0


GSGN = {"acf1": stat_acf1, "ljungbox": stat_ljungbox, "runs": stat_runs}
GSGN_TWOSIDED = {"acf1": True, "ljungbox": False, "runs": True}


def calibrate_threshold(scores, fa):
    """Empirical (1-fa) quantile -- the only null we trust off-Gaussian."""
    return float(np.quantile(scores[np.isfinite(scores)], 1.0 - fa))


def block_stats(x, W, fn, two_sided):
    """Statistic evaluated on consecutive non-overlapping windows of length W."""
    n = len(x) // W
    if n == 0:
        return np.array([])
    v = np.array([fn(x[i * W:(i + 1) * W]) for i in range(n)])
    return np.abs(v) if two_sided else v


# =============================================================================
# S6.  Autoencoder (scheduler P) -- trained once per dataset
# =============================================================================

if _TORCH_OK:

    class LSTMAutoencoder(nn.Module):
        """Seq2seq LSTM AE on magnitude windows; identical topology to the
        prototype used for the submitted version (hidden 128, latent 1)."""

        def __init__(self, input_dim=1, hidden_dim=128, latent_dim=1, num_layers=1):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc_enc = nn.Linear(hidden_dim, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, hidden_dim)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.out = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            _, (h, c) = self.encoder(x)
            zl = self.fc_enc(h[-1])
            dec_in = self.fc_dec(zl).unsqueeze(1).repeat(1, x.size(1), 1)
            dec_h, _ = self.decoder(dec_in, (h, c))
            return self.out(dec_h), zl


def _standardise(mag):
    mu, sd = float(mag.mean()), float(mag.std() + 1e-12)
    return mu, sd


def train_or_load_ae(mag_train, tag, cfg=CFG, train_flag=True):
    """
    Returns a frozen dict {model, mu, sd, L, mode}.  Checkpoint
    models_checkpoint/AE_<tag>.pth is reused when train_flag is False.
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available: run with --skip_ae or install torch")
    cfg.dirs()
    dev = cfg.resolved_device()
    ckpt = Path(cfg.ckpt_dir) / f"AE_{tag}.pth"
    meta = Path(cfg.ckpt_dir) / f"AE_{tag}_meta.json"

    mu, sd = _standardise(mag_train)
    model = LSTMAutoencoder(1, cfg.hidden_dim, cfg.latent_dim).to(dev)

    if (not train_flag) and ckpt.exists():
        st = json.loads(meta.read_text())
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()
        print(f"[AE {tag}] loaded {ckpt.name} (val={st['best_val']:.6g})")
        return dict(model=model, mu=st["mu"], sd=st["sd"], L=cfg.L, tag=tag)

    Xw = windows_of((mag_train - mu) / sd, cfg.L).astype(np.float32)[:, :, None]
    n_val = max(1, int(0.15 * len(Xw)))
    Xtr, Xva = Xw[:-n_val], Xw[-n_val:]
    tl = DataLoader(TensorDataset(torch.tensor(Xtr)), batch_size=cfg.batch_size,
                    shuffle=True, drop_last=True)
    vl = DataLoader(TensorDataset(torch.tensor(Xva)), batch_size=512, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=5, factor=0.5)
    crit = nn.MSELoss()
    best, bad = float("inf"), 0
    print(f"[AE {tag}] training on {len(Xtr)} windows, device={dev}")
    for ep in range(cfg.n_epochs):
        model.train()
        for (xb,) in tl:
            xb = xb.to(dev)
            rec, _ = model(xb)
            loss = crit(rec, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vls = [crit(model(xb.to(dev))[0], xb.to(dev)).item() for (xb,) in vl]
        v = float(np.mean(vls))
        sch.step(v)
        flag = ""
        if v < best - 1e-12:
            best, bad = v, 0
            torch.save(model.state_dict(), ckpt)
            flag = " *"
        else:
            bad += 1
        print(f"  [AE {tag}] epoch {ep+1:03d}/{cfg.n_epochs} val={v:.6f}{flag}")
        if bad >= cfg.early_stop:
            print(f"  [AE {tag}] early stop at epoch {ep+1}")
            break
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    meta.write_text(json.dumps(dict(mu=mu, sd=sd, best_val=best, L=cfg.L,
                                    hidden=cfg.hidden_dim, latent=cfg.latent_dim,
                                    epochs=cfg.n_epochs, lr=cfg.lr,
                                    optimizer="AdamW", early_stop=cfg.early_stop)))
    return dict(model=model, mu=mu, sd=sd, L=cfg.L, tag=tag)


def ae_scores(ae, mag, cfg=CFG):
    """
    s_k = || u_k - AE(u_k) ||^2  in physical units, eq. (16).
    Windows are a stride view, streamed in batches -- never materialised.
    """
    dev = cfg.resolved_device()
    L = ae["L"]
    Xw = windows_of((mag - ae["mu"]) / ae["sd"], L)
    n = len(Xw)
    out = np.empty(n, dtype=float)
    model = ae["model"]
    with torch.no_grad():
        for i in range(0, n, cfg.infer_batch):
            xb = torch.tensor(np.ascontiguousarray(Xw[i:i + cfg.infer_batch],
                                                   dtype=np.float32)[:, :, None]).to(dev)
            rec, _ = model(xb)
            err = (xb - rec).squeeze(-1).cpu().numpy() * ae["sd"]   # physical
            out[i:i + len(err)] = (np.sum(err ** 2, axis=1) if cfg.ae_score == "window"
                                   else err[:, -1] ** 2)
    s = np.full(len(mag), np.nan)
    s[L - 1:] = out
    return s


# =============================================================================
# S7.  Experiments
# =============================================================================
#  Every experiment writes a tidy CSV, one row per replicate, so that nothing
#  is aggregated before collect_results.py sees it (M10: per-budget numbers
#  with CIs, never pooled).
# -----------------------------------------------------------------------------

import csv

_SEED_BASE = dict(train=10_000, cal=200_000, dep=400_000, chunk=600_000,
                  truth=800_000, aux=900_000)
_REG_OFF = {"N1": 0, "N2": 1_000_000, "MM": 2_000_000, "TV": 3_000_000}


def seed_of(regime, kind, i=0, cfg=CFG):
    return cfg.seed + _REG_OFF[regime] + _SEED_BASE[kind] + i


def save_rows(rows, name, cfg=CFG, meta=None):
    cfg.dirs()
    path = Path(cfg.out_dir) / f"{name}.csv"
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if meta is not None:
        (Path(cfg.out_dir) / f"{name}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str))
    print(f"  -> {path}  ({len(rows)} rows)")
    return path


def eff_sample_size(indicator, nlags):
    """
    Effective calibration sample size of a *serially dependent* score,
    N_eff = N / (1 + 2 sum_l rho_l).  The i.i.d. calibration floor quoted in
    Sec. V-A, sqrt(G(1-G)/N), must be read with N replaced by N_eff for the
    windowed score, whose consecutive windows overlap by L-1 samples.
    """
    x = np.asarray(indicator, dtype=float)
    x = x - x.mean()
    den = float(np.sum(x * x))
    if den <= 0:
        return float(len(x))
    tot = 0.0
    for l in range(1, int(nlags) + 1):
        r = float(np.sum(x[l:] * x[:-l]) / den)
        if r <= 0:
            break
        tot += r
    return float(len(x) / (1.0 + 2.0 * tot))


def _dataset(regime, cfg, train_ae_flag=False, verbose=True):
    """Training stream + frozen scorers for one dataset."""
    tr = simulate_stream(regime, cfg.T_train, seed_of(regime, "train", cfg=cfg), cfg)
    if verbose:
        kz = float(((tr["z"] - tr["z"].mean()) ** 4).mean() / tr["z"].var() ** 2 - 3)
        print(f"[{regime}] S_design={tr['S_design']:.5f} S_emp={tr['S_emp']:.5f} "
              f"kexc(z)={kz:.2f}")
    sc = fit_scorers(tr, cfg, tag=regime, train_ae_flag=train_ae_flag)
    return tr, sc


# ------------------------------------------------------------------ E0 -----
def run_E0(cfg=CFG):
    """Pretrain and freeze one autoencoder per dataset (Sec. V-A)."""
    meta = {}
    for regime in REGIMES:
        tr = simulate_stream(regime, cfg.T_train, seed_of(regime, "train", cfg=cfg), cfg)
        kz = float(((tr["z"] - tr["z"].mean()) ** 4).mean() / tr["z"].var() ** 2 - 3)
        print(f"[E0] {regime}: S_design={tr['S_design']:.5f} "
              f"S_emp={tr['S_emp']:.5f} kexc(z)={kz:.3f}")
        train_or_load_ae(np.abs(tr["z"]), tag=regime, cfg=cfg, train_flag=True)
        meta[regime] = dict(S_design=tr["S_design"], S_emp=tr["S_emp"], kexc_z=kz)
    cfg.dirs()
    (Path(cfg.out_dir) / "E0_datasets.json").write_text(json.dumps(meta, indent=2))
    return meta


# ------------------------------------------------------------------ E1 -----
def run_E1(cfg=CFG, regimes=("N1", "N2")):
    """
    Budget accuracy (Thm. 4, Thm. 6; M5, M10, D1, m7).

    B1 is expected to track under N1 and to depart under N2; B2, B3 and P are
    expected to be statistically indistinguishable under both, because budget
    accuracy is supplied by the calibration and not by the score.  For m=1,
    |z|/sqrt(S) and |z| induce the same firing set, so B2 == B3 exactly.
    """
    names = ["B1", "B2", "B3"] + ([] if cfg.skip_ae else ["P"])
    rows = []
    for regime in regimes:
        _, sc = _dataset(regime, cfg)
        for i in range(cfg.n_mc):
            cal = simulate_stream(regime, cfg.N_cal + cfg.L,
                                  seed_of(regime, "cal", i, cfg), cfg)
            dep = simulate_stream(regime, cfg.T_dep + cfg.L,
                                  seed_of(regime, "dep", i, cfg), cfg)
            s_cal = compute_scores(cal["z"], sc, cfg, which=names)
            s_dep = compute_scores(dep["z"], sc, cfg, which=names)
            for name in names:
                for G in cfg.budgets:
                    eps = (thr_gaussian(G) if name == "B1"
                           else thr_conformal(s_cal[name], G))
                    g = fire_threshold(s_dep[name], eps, cfg.L)
                    rate = float(np.mean(g[cfg.L:]))
                    rows.append(dict(exp="E1", regime=regime, scheduler=name,
                                     budget=G, seed=i, eps=eps,
                                     realized_rate=rate,
                                     signed_err_pp=100.0 * (rate - G),
                                     rel_err=(rate - G) / G,
                                     n_eff_cal=eff_sample_size(
                                         s_cal[name][cfg.L:] > eps, cfg.L)))
            if (i + 1) % max(1, cfg.n_mc // 10) == 0:
                print(f"  [E1 {regime}] {i+1}/{cfg.n_mc}")
    return save_rows(rows, "E1_budget_accuracy", cfg,
                     meta=dict(N_cal=cfg.N_cal, T_dep=cfg.T_dep, n_mc=cfg.n_mc,
                               budgets=list(cfg.budgets), schedulers=names,
                               calib_floor={str(G): math.sqrt(G * (1 - G) / cfg.N_cal)
                                            for G in cfg.budgets}))


# ------------------------------------------------------------------ E2 -----
def run_E2(cfg=CFG, regimes=("N1", "N2"), T_test=20_000):
    """
    Conformal versus naive threshold index (Thm. 6; M8).  No rate claim is
    made (M6): we report the realised exceedance rate and the fraction of
    calibration draws that violate the budget, at each N.
    """
    rows = []
    pool_T = max(200_000, 40 * max(cfg.e2_N_grid) + T_test)
    for regime in regimes:
        st = simulate_stream(regime, pool_T, seed_of(regime, "truth", cfg=cfg), cfg)
        s_pool = np.abs(st["z"])                     # score B3, m = 1
        rng = np.random.default_rng(seed_of(regime, "aux", 1, cfg))
        for N in cfg.e2_N_grid:
            for G in cfg.e2_budgets:
                for rep in range(cfg.e2_reps):
                    i0 = rng.integers(0, len(s_pool) - N - T_test - 1)
                    s_cal = s_pool[i0:i0 + N]
                    s_te = s_pool[i0 + N:i0 + N + T_test]
                    for rule, thr in (("conformal", thr_conformal),
                                      ("naive", thr_naive)):
                        eps = thr(s_cal, G)
                        rows.append(dict(exp="E2", regime=regime, N=N, budget=G,
                                         rule=rule, rep=rep,
                                         exceed_rate=float(np.mean(s_te > eps))))
            print(f"  [E2 {regime}] N={N} done")
    return save_rows(rows, "E2_conformal_index", cfg,
                     meta=dict(T_test=T_test, reps=cfg.e2_reps,
                               N_grid=list(cfg.e2_N_grid),
                               budgets=list(cfg.e2_budgets)))


# ------------------------------------------------------------------ E3 -----
def run_E3(cfg=CFG, regimes=("N1", "N2"), n_seeds=20):
    """
    Gmag invariance (Thm. 2) -- IMPLEMENTATION CHECK, NOT CONFIRMATION.

    Theorem 2 makes attacked-FAR = nominal-FAR an algebraic identity, so this
    table only certifies that the pipeline realises it.  To show the attack is
    nevertheless non-trivial we report the remote MSE alongside: the FAR is
    unchanged while the remote estimate degrades.
    """
    sched = "B3" if cfg.skip_ae else "P"
    rows = []
    for regime in regimes:
        _, sc = _dataset(regime, cfg)
        for i in range(n_seeds):
            cal = simulate_stream(regime, cfg.N_cal + cfg.L,
                                  seed_of(regime, "cal", i, cfg), cfg)
            dep = simulate_stream(regime, cfg.T_dep + cfg.L,
                                  seed_of(regime, "dep", i, cfg), cfg)
            S = cal["S_design"]
            s_cal = compute_scores(cal["z"], sc, cfg, which=[sched])[sched]
            s_dep = compute_scores(dep["z"], sc, cfg, which=[sched])[sched]
            thr = {d: calibrate_threshold(f(cal["z"], S, cfg), cfg.fa_target)
                   for d, f in GMAG.items()}
            for G in cfg.budgets:
                eps = thr_conformal(s_cal, G)
                g = fire_threshold(s_dep, eps, cfg.L)
                zc = np.where(g, -dep["z"], dep["z"])          # eq. (11)
                dm = damage_metrics(dep["z"], dep["es"], g, dep["K"], k0=cfg.L)
                for dname, f in GMAG.items():
                    far_n = float(np.mean(f(dep["z"], S, cfg)[cfg.L:] > thr[dname]))
                    far_a = float(np.mean(f(zc, S, cfg)[cfg.L:] > thr[dname]))
                    rows.append(dict(exp="E3", regime=regime, scheduler=sched,
                                     budget=G, seed=i, detector=dname,
                                     far_nominal=far_n, far_attacked=far_a,
                                     far_diff=far_a - far_n,
                                     realized_rate=dm["rate"],
                                     mse_attacked=dm["mse"],
                                     mse_nominal=dm["mse_nominal"]))
        print(f"  [E3 {regime}] done")
    return save_rows(rows, "E3_gmag_invariance", cfg,
                     meta=dict(fa_target=cfg.fa_target, W=cfg.chi2_window,
                               cusum_drift=cfg.cusum_drift, scheduler=sched,
                               note="Thm. 2 is an identity; this is an "
                                    "implementation check (M2c)."))


# ------------------------------------------------------------------ E4 -----
def run_E4(cfg=CFG, regimes=("N1", "N2")):
    """
    Gsgn exposure (Remark on the scope boundary; M3).  This is the whiteness
    evaluation the reviewer asked for, and it is adverse under N2 by design:
    we report it rather than claim immunity.
    """
    Wmax = max(cfg.sgn_windows)
    # "SGN" is a deliberately sign-DEPENDENT control, gamma_k = 1{z_k > eps}.
    # It is never used as a proposed scheduler; it is here to show that
    # sign-invariance of the firing rule is a necessary design constraint,
    # since a one-sided rule renders the received marginal one-sided and is
    # caught by Gsgn even under N1.
    scheds = ["B3"] + ([] if cfg.skip_ae else ["P"]) + ["SGN"]
    rows = []

    def _score(zs):
        out = compute_scores(zs, sc, cfg, which=[s for s in scheds if s != "SGN"])
        sg = np.array(zs, dtype=float).copy()
        sg[:cfg.L] = np.nan
        out["SGN"] = sg
        return out

    for regime in regimes:
        _, sc = _dataset(regime, cfg)
        cal = simulate_stream(regime, cfg.N_cal + cfg.L,
                              seed_of(regime, "cal", 0, cfg), cfg)
        s_cal = _score(cal["z"])

        # -- independent nominal chunks: empirical null for every Gsgn test --
        chunks = [simulate_stream(regime, Wmax + cfg.L,
                                  seed_of(regime, "chunk", j, cfg), cfg)
                  for j in range(cfg.n_null_windows)]
        z_nom = [c["z"][cfg.L:] for c in chunks]
        thr, far_nom = {}, {}
        for tname, fn in GSGN.items():
            for W in cfg.sgn_windows:
                v = np.concatenate([block_stats(zz, W, fn, GSGN_TWOSIDED[tname])
                                    for zz in z_nom])
                thr[(tname, W)] = float(np.quantile(v, 1 - cfg.fa_target))
                far_nom[(tname, W)] = float(np.mean(v > thr[(tname, W)]))

        # score every chunk ONCE for every scheduler; scores do not depend on
        # the budget, only the threshold does
        all_chunk_scores = [_score(c["z"]) for c in chunks]
        for sched in scheds:
            s_chunks = [d[sched] for d in all_chunk_scores]
            for G in cfg.budgets:
                eps = thr_conformal(s_cal[sched], G)
                z_att, rates, acf_a, acf_n = [], [], [], []
                for c, s in zip(chunks, s_chunks):
                    g = fire_threshold(s, eps, cfg.L)
                    zc = np.where(g, -c["z"], c["z"])[cfg.L:]
                    z_att.append(zc)
                    rates.append(float(np.mean(g[cfg.L:])))
                    acf_a.append(stat_acf1(zc))
                    acf_n.append(stat_acf1(c["z"][cfg.L:]))
                for tname, fn in GSGN.items():
                    for W in cfg.sgn_windows:
                        v = np.concatenate([block_stats(zz, W, fn,
                                                        GSGN_TWOSIDED[tname])
                                            for zz in z_att])
                        rows.append(dict(
                            exp="E4", regime=regime, scheduler=sched, budget=G,
                            test=tname, W=W,
                            power=float(np.mean(v > thr[(tname, W)])),
                            far_nominal=far_nom[(tname, W)],
                            threshold=thr[(tname, W)],
                            realized_rate=float(np.mean(rates)),
                            acf1_attacked=float(np.mean(acf_a)),
                            acf1_nominal=float(np.mean(acf_n))))
            print(f"  [E4 {regime}] scheduler {sched} done")
    return save_rows(rows, "E4_gsgn_exposure", cfg,
                     meta=dict(windows=list(cfg.sgn_windows),
                               n_chunks=cfg.n_null_windows,
                               fa_target=cfg.fa_target,
                               null="empirical (asymptotic nulls are "
                                    "unusable under heavy tails)"))


# ------------------------------------------------------------------ E5 -----
def run_E5(cfg=CFG, regimes=("N1", "N2")):
    """
    Second-moment insufficiency (Prop. 1).  Sweep the feasible set of the
    covariance-matching constraint,  z^a = f z + b,  sigma_b = S(1-f^2):
    the variance is pinned to S while kexc(z^a) = f^4 kexc(z).

    The kurtosis test is calibrated EMPIRICALLY; the normal approximation for
    sample kurtosis has a false-alarm rate near 0.68 under N2 and we report it.
    """
    def kexc(x):
        xc = x - x.mean()
        return float(np.mean(xc ** 4) / np.mean(xc ** 2) ** 2 - 3.0)

    rows_law, rows_pow = [], []
    for regime in regimes:
        st = simulate_stream(regime, cfg.e6_T_truth, seed_of(regime, "truth", 2, cfg), cfg)
        z = st["z"]
        S = float(np.var(z))
        k_z = kexc(z)
        rng = np.random.default_rng(seed_of(regime, "aux", 5, cfg))
        for f in cfg.e5_f_grid:
            sb = S * (1 - f ** 2)
            za = f * z + rng.normal(0.0, math.sqrt(max(sb, 0.0)), len(z))
            rows_law.append(dict(exp="E5a", regime=regime, f=f,
                                 var_ratio=float(np.var(za) / S),
                                 kexc_emp=kexc(za),
                                 kexc_pred=f ** 4 * k_z, kexc_z=k_z))
        # ---- power of an empirically calibrated kurtosis test ----------
        for N in cfg.e5_N_grid:
            i0 = rng.integers(0, len(z) - N - 1, size=cfg.e5_reps)
            null = np.array([kexc(z[i:i + N]) for i in i0])
            lo = float(np.quantile(null, cfg.e5_alpha / 2))
            hi = float(np.quantile(null, 1 - cfg.e5_alpha / 2))
            far_normal = float(np.mean(np.abs(null) > norm.ppf(1 - cfg.e5_alpha / 2)
                                       * math.sqrt(24.0 / N)))
            for f in cfg.e5_f_grid:
                if abs(f) == 1.0:
                    continue
                sb = S * (1 - f ** 2)
                v = []
                for i in i0:
                    seg = f * z[i:i + N] + rng.normal(0, math.sqrt(sb), N)
                    v.append(kexc(seg))
                v = np.array(v)
                rows_pow.append(dict(exp="E5b", regime=regime, f=f, N=N,
                                     alpha=cfg.e5_alpha,
                                     power=float(np.mean((v < lo) | (v > hi))),
                                     far_empirical=cfg.e5_alpha,
                                     far_normal_approx=far_normal,
                                     null_lo=lo, null_hi=hi))
            print(f"  [E5 {regime}] N={N} done")
    save_rows(rows_law, "E5a_kurtosis_law", cfg,
              meta=dict(f_grid=list(cfg.e5_f_grid), T=cfg.e6_T_truth))
    return save_rows(rows_pow, "E5b_kurtosis_power", cfg,
                     meta=dict(alpha=cfg.e5_alpha, reps=cfg.e5_reps,
                               N_grid=list(cfg.e5_N_grid)))


# ------------------------------------------------------------------ E6 -----
def run_E6(cfg=CFG, regimes=("N1", "N2")):
    """
    Distribution-free degradation (Prop. 2).  Compare, per budget:
        beta_true  -- long-run truth,
        beta_G     -- the Gaussian truncated moment of [Guo2023, Lem. 1],
        beta_hat_N -- eq. (18),
    then propagate each through the divergence recursion to a predicted
    tr(P^a_inf), and compare against a simulated attacked run.
    """
    rows, rows_conv = [], []
    for regime in regimes:
        st = simulate_stream(regime, cfg.e6_T_truth, seed_of(regime, "truth", cfg=cfg), cfg)
        z, S = st["z"], st["S_design"]
        K = st["K"]
        m2 = float(np.mean(z ** 2))

        # simulated nominal / attacked reference (B3 with empirical threshold)
        dep = simulate_stream(regime, cfg.T_dep + cfg.L,
                              seed_of(regime, "dep", 0, cfg), cfg)
        for G in cfg.budgets:
            eps = float(np.quantile(np.abs(z), 1.0 - G))
            keep = np.abs(z) <= eps
            beta_true = 1.0 - float(np.mean(z[keep] ** 2)) / m2

            e_g = thr_gaussian(G)
            beta_G = float((2.0 / math.sqrt(2 * math.pi)) * e_g *
                           math.exp(-e_g ** 2 / 2) / (1.0 - 2.0 * norm.sf(e_g)))

            cal = simulate_stream(regime, cfg.N_cal + cfg.L,
                                  seed_of(regime, "cal", 0, cfg), cfg)
            zc_ = cal["z"][cfg.L:]
            eps_N = float(np.quantile(np.abs(zc_), 1.0 - G))
            kp = np.abs(zc_) <= eps_N
            beta_hat = 1.0 - float(np.mean(zc_[kp] ** 2)) / float(np.mean(zc_ ** 2))

            psi_t, trD_t = guo_steady_state(G, beta_true, S, K)
            psi_g, trD_g = guo_steady_state(G, beta_G, S, K)
            psi_h, trD_h = guo_steady_state(G, beta_hat, S, K)

            g = fire_threshold(np.abs(dep["z"]), eps, cfg.L)
            dm = damage_metrics(dep["z"], dep["es"], g, dep["K"], k0=cfg.L)
            excess_sim = dm["mse"] - dm["mse_nominal"]

            rows.append(dict(
                exp="E6", regime=regime, budget=G,
                eps_emp=eps, eps_gauss=e_g * math.sqrt(S),
                beta_true=beta_true, beta_gauss=beta_G, beta_hat=beta_hat,
                rel_err_gauss=(beta_G - beta_true) / beta_true,
                rel_err_hat=(beta_hat - beta_true) / beta_true,
                psi_true=psi_t, psi_gauss=psi_g, psi_hat=psi_h,
                psi_ratio_true_over_gauss=psi_t / psi_g,
                trD_true=trD_t, trD_gauss=trD_g, trD_hat=trD_h,
                trD_rel_err_gauss=(trD_g - trD_t) / trD_t,
                trD_rel_err_hat=(trD_h - trD_t) / trD_t,
                excess_mse_sim=excess_sim, realized_rate=dm["rate"],
                mse_nominal=dm["mse_nominal"]))

            # convergence of beta_hat_N (Prop. 2)
            rng = np.random.default_rng(seed_of(regime, "aux", 7, cfg))
            for N in cfg.e6_N_grid:
                for rep in range(40):
                    i0 = int(rng.integers(0, len(z) - N - 1))
                    seg = z[i0:i0 + N]
                    e_ = float(np.quantile(np.abs(seg), 1.0 - G))
                    k_ = np.abs(seg) <= e_
                    if k_.sum() < 2:
                        continue
                    b_ = 1.0 - float(np.mean(seg[k_] ** 2)) / float(np.mean(seg ** 2))
                    rows_conv.append(dict(exp="E6c", regime=regime, budget=G,
                                          N=N, rep=rep, beta_hat=b_,
                                          beta_true=beta_true))
        print(f"  [E6 {regime}] done")
    save_rows(rows_conv, "E6c_beta_convergence", cfg)
    return save_rows(rows, "E6_degradation", cfg,
                     meta=dict(T_truth=cfg.e6_T_truth, N_cal=cfg.N_cal))


# ------------------------------------------------------------------ E7/E8 --
def _matched_rate(regimes, cfg, name):
    """
    Instant selection at MATCHED REALISED FIRING RATE (M9).  Every scheduler
    fires exactly ceil(rate * n_valid) times on the same stream, so any
    difference is selection quality and not under-firing.

    Reported per D3: MSE split into tr(Cov(e^a)) and ||E[e^a]||^2, plus the
    excess over nominal and the excess per firing.
    """
    names = [n for n in ABLATIONS if not (n == "P" and cfg.skip_ae)]
    rows = []
    for regime in regimes:
        _, sc = _dataset(regime, cfg)
        for i in range(cfg.n_mc_impact):
            dep = simulate_stream(regime, cfg.T_dep + cfg.L,
                                  seed_of(regime, "dep", 5_000 + i, cfg), cfg)
            scores = compute_scores(dep["z"], sc, cfg, which=names)
            for rate in cfg.budgets:
                for nm in names:
                    g = fire_top_rate(scores[nm], rate, cfg.L)
                    dm = damage_metrics(dep["z"], dep["es"], g, dep["K"], k0=cfg.L)
                    exc = dm["mse"] - dm["mse_nominal"]
                    rows.append(dict(
                        exp=name, regime=regime, scheduler=nm, target_rate=rate,
                        seed=i, realized_rate=dm["rate"], n_fire=dm["n_fire"],
                        mse=dm["mse"], cov_trace=dm["cov_trace"],
                        bias_sq=dm["bias_sq"], mse_nominal=dm["mse_nominal"],
                        excess_mse=exc,
                        excess_per_firing=exc / max(dm["rate"], 1e-12),
                        peak_div=dm["peak_div"],
                        mean_abs_z_fired=dm["mean_abs_z_fired"],
                        mean_z2_fired=dm["mean_z2_fired"],
                        fire_clustering=dm["fire_clustering"]))
            if (i + 1) % max(1, cfg.n_mc_impact // 5) == 0:
                print(f"  [{name} {regime}] {i+1}/{cfg.n_mc_impact}")
    return rows


def run_E7(cfg=CFG):
    """E7 -- the decisive experiment: does the AE beat the memoryless score
    on the ideal LTI plant?  If it ties on N1, that is the honest outcome and
    Remark 3 already predicts it."""
    rows = _matched_rate(("N1", "N2"), cfg, "E7")
    return save_rows(rows, "E7_matched_rate", cfg,
                     meta=dict(schedulers=ABLATIONS, T_dep=cfg.T_dep,
                               n_mc=cfg.n_mc_impact,
                               rates=list(cfg.budgets),
                               note="matched realised rate, not matched budget"))


def run_E8(cfg=CFG):
    """E8 -- where the windowed score can earn its place: model mismatch and
    a time-varying operating point, i.e. where the innovation stops being
    white.  Merge with E7 for Table II."""
    rows = _matched_rate(("MM", "TV"), cfg, "E8")
    return save_rows(rows, "E8_regimes", cfg,
                     meta=dict(schedulers=ABLATIONS, regimes=["MM", "TV"],
                               n_mc=cfg.n_mc_impact))


EXPERIMENTS = {"E0": run_E0, "E1": run_E1, "E2": run_E2, "E3": run_E3,
               "E4": run_E4, "E5": run_E5, "E6": run_E6, "E7": run_E7,
               "E8": run_E8}
ARRAY_ORDER = ["E7", "E8", "E1", "E2", "E4", "E6", "E3", "E5"]   # plan, Sec. 5


# =============================================================================
# S8.  CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Section V experiments E1-E8")
    p.add_argument("--exp", default="E1",
                   help="E0..E8 | all | array:<idx>  (array order: "
                        + ",".join(ARRAY_ORDER) + ")")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test sizes")
    p.add_argument("--skip_ae", action="store_true",
                   help="drop scheduler P (no torch needed)")
    p.add_argument("--train_ae", type=int, default=0,
                   help="1 = retrain the AE inside this job")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_mc", type=int, default=None)
    p.add_argument("--n_mc_impact", type=int, default=None)
    p.add_argument("--N_cal", type=int, default=None)
    p.add_argument("--T_dep", type=int, default=None)
    p.add_argument("--T_train", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--n_epochs", type=int, default=None)
    p.add_argument("--ae_score", default=None, choices=[None, "window", "last"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fig_dir", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--device", default=None)
    return p


def config_from_args(a) -> Config:
    cfg = Config()
    if a.quick:
        cfg = quick(cfg)
    for k in ("seed", "n_mc", "n_mc_impact", "N_cal", "T_dep", "T_train", "L",
              "n_epochs", "ae_score", "out_dir", "fig_dir", "ckpt_dir", "device"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    cfg.skip_ae = bool(a.skip_ae) or not _TORCH_OK
    cfg.train_ae = bool(a.train_ae)
    cfg.dirs()
    return cfg


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    global CFG
    CFG = cfg
    np.random.seed(cfg.seed)
    if _TORCH_OK:
        torch.manual_seed(cfg.seed)

    if a.exp.startswith("array:"):
        todo = [ARRAY_ORDER[int(a.exp.split(":")[1])]]
    elif a.exp == "all":
        todo = ["E0"] + ARRAY_ORDER
    else:
        todo = [a.exp]

    print("=" * 72)
    print(f"cps_experiments  |  exp={todo}  device={cfg.resolved_device()}  "
          f"skip_ae={cfg.skip_ae}")
    print(json.dumps({k: v for k, v in asdict(cfg).items()
                      if k in ("N_cal", "T_dep", "T_train", "n_mc",
                               "n_mc_impact", "L", "budgets", "seed")},
                     default=str))
    print("=" * 72)
    for name in todo:
        t0 = time.time()
        print(f"\n### {name} ###")
        EXPERIMENTS[name](cfg)
        print(f"### {name} finished in {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
