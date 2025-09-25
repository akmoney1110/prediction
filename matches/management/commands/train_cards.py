#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Card trainer (yellow/red/total):
- Pulls labels from MatchStats: yellows / reds / cards
- Optional Poisson GLM using goals μ features (if available); else safe fallbacks
- NB dispersion with grouped shrinkage (league x season)
- Totals via independent convolution or Gaussian copula (ρ)
- Optional isotonic calibration for totals and team lines
- If train has no labels but val does, borrow val for fitting (not for calibration)

Usage (example):
export DJANGO_SETTINGS_MODULE=prediction.settings
python manage.py train_cards \
  --card-type yellow \
  --leagues 39 \
  --train-seasons 2021-2023 \
  --val-seasons 2024 \
  --test-seasons 2025 \
  --outdir artifacts/cards \
  --goals-artifact artifacts/goals/artifacts.goals.json \
  --rho-grid auto
"""

# ---------------- Django bootstrap (must be first!) ----------------
import os
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "prediction.settings"
import django
django.setup()
# -------------------------------------------------------------------

import json, argparse, math, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
np.seterr(all="ignore")

# Models (after django.setup())
from matches.models import Match, MatchStats
try:
    from matches.models import MLTrainingMatch  # type: ignore
    HAVE_MLTRAIN = True
except Exception:
    MLTrainingMatch = None  # type: ignore
    HAVE_MLTRAIN = False

from django.core.management.base import BaseCommand

from sklearn.isotonic import IsotonicRegression

# Optional PoissonRegressor
try:
    from sklearn.linear_model import PoissonRegressor
    HAVE_POISSON = True
except Exception:
    HAVE_POISSON = False

# Optional SciPy bits
try:
    from scipy.special import gammaln
    from scipy.stats import kendalltau
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from scipy.special import erf as _erf_u
    _HAVE_ERF_UFUNC = True
except Exception:
    _erf_u = None
    _HAVE_ERF_UFUNC = False


# ============================= utils =============================
def expand_seasons(arg: str) -> List[int]:
    out: List[int] = []
    for seg in str(arg).split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "-" in seg:
            a, b = seg.split("-", 1)
            a, b = int(a), int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(seg))
    return sorted(set(out))

def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]

@dataclass
class GoalsArtifact:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

def load_goals_artifact(path: Optional[str]) -> Optional[GoalsArtifact]:
    if not path:
        return None
    with open(path, "r") as f:
        art = json.load(f)
    return GoalsArtifact(
        mean=np.array(art["scaler_mean"], float),
        scale=np.array(art["scaler_scale"], float),
        coef=np.array(art["poisson_coef"], float),
        intercept=float(art["poisson_intercept"]),
    )

def mu_from_features(x: np.ndarray, art: GoalsArtifact) -> float:
    xs = (x - art.mean) / art.scale
    return float(np.exp(art.intercept + xs.dot(art.coef)))

# goals feature builder (import path tolerant)
try:
    from prediction.train_goals import build_oriented_features  # type: ignore
except Exception:
    try:
        from train_goals import build_oriented_features  # type: ignore
    except Exception:
        build_oriented_features = None  # type: ignore

def _phi_arr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if _HAVE_ERF_UFUNC:
        return 0.5 * (1.0 + _erf_u(z / math.sqrt(2.0)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def _to_py(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(v) for v in obj]
    return obj


# ======================= field helpers =======================
def stat_field_for_card(which: str) -> str:
    w = (which or "").strip().lower()
    if w in ("yellow", "yellows"):
        return "yellows"
    if w in ("red", "reds"):
        return "reds"
    if w in ("total", "cards", "all"):
        return "cards"
    raise ValueError(f"Unknown --card-type: {which}")

def default_priors(which: str) -> Tuple[float, float]:
    """Reasonable priors in case GLM falls back (per-team means)."""
    w = (which or "").strip().lower()
    if w in ("yellow", "yellows"):
        return 2.0, 1.8      # (home, away)
    if w in ("red", "reds"):
        return 0.15, 0.15
    # totals/cards
    return 2.2, 2.0


# ======================= data loading =======================
def load_rows(leagues: List[int], seasons: List[int], which: str) -> pd.DataFrame:
    """
    Load rows and attach *card* labels from MatchStats:
      - yellows / reds / cards depending on --card-type
      - tries MatchStats only (these are not materialized in Match)
      - keeps MLTrainingMatch structure if present for feature extraction / kickoff alignment
    """
    stat_field = stat_field_for_card(which)
    recs = []
    missing_both = 0

    if HAVE_MLTRAIN:
        base_iter = (MLTrainingMatch.objects
                     .filter(league_id__in=leagues, season__in=seasons)
                     .order_by("kickoff_utc")
                     .only("fixture_id","league_id","season","kickoff_utc","stats10_json")
                     .iterator())
        for r in base_iter:
            ch = None; ca = None
            # We need home/away team ids to orient stats
            try:
                m = Match.objects.only("home_id","away_id").get(pk=int(r.fixture_id))
                stats = list(MatchStats.objects
                             .filter(match_id=int(r.fixture_id))
                             .values("team_id", stat_field))
                by_team = {
                    int(s["team_id"]): s.get(stat_field)
                    for s in stats if s.get(stat_field) is not None
                }
                ch = by_team.get(int(m.home_id))
                ca = by_team.get(int(m.away_id))
            except Match.DoesNotExist:
                pass
            except Exception:
                pass

            if ch is None and ca is None:
                missing_both += 1

            recs.append({
                "fixture_id": int(r.fixture_id),
                "league_id": int(r.league_id),
                "season": int(r.season),
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "stats10_json": getattr(r, "stats10_json", None),
                "cards_home": None if ch is None else int(ch),
                "cards_away": None if ca is None else int(ca),
            })
    else:
        base_iter = (Match.objects
                     .filter(league_id__in=leagues, season__in=seasons)
                     .order_by("kickoff_utc")
                     .only("id","league_id","season","kickoff_utc","home_id","away_id","raw_result_json")
                     .iterator())
        for m in base_iter:
            ch = None; ca = None
            try:
                stats = list(MatchStats.objects
                             .filter(match_id=int(m.id))
                             .values("team_id", stat_field))
                by_team = {
                    int(s["team_id"]): s.get(stat_field)
                    for s in stats if s.get(stat_field) is not None
                }
                ch = by_team.get(int(m.home_id))
                ca = by_team.get(int(m.away_id))
            except Exception:
                pass

            if ch is None and ca is None:
                missing_both += 1

            recs.append({
                "fixture_id": int(m.id),
                "league_id": int(m.league_id),
                "season": int(m.season),
                "kickoff_utc": m.kickoff_utc.isoformat(),
                "stats10_json": (m.raw_result_json or {}).get("stats10_json"),
                "cards_home": None if ch is None else int(ch),
                "cards_away": None if ca is None else int(ca),
            })

    df = pd.DataFrame(recs)
    if len(df) > 0 and missing_both:
        print(f"[WARN] No card labels (both sides) for {missing_both}/{len(df)} rows.")
    return df


# =================== goals μ attachment (optional) ===================
def attach_goal_mus(df: pd.DataFrame, art: Optional[GoalsArtifact]) -> pd.DataFrame:
    muH, muA = [], []
    fails = 0
    for _, r in df.iterrows():
        if art is None or build_oriented_features is None:
            muH.append(np.nan); muA.append(np.nan)
            continue
        try:
            xh, xa, _ = build_oriented_features({"stats10_json": r.get("stats10_json")})
            muH.append(mu_from_features(xh, art))
            muA.append(mu_from_features(xa, art))
        except Exception:
            muH.append(np.nan); muA.append(np.nan); fails += 1
    out = df.copy()
    out["mu_goals_home"] = muH
    out["mu_goals_away"] = muA
    if fails:
        print(f"[WARN] build_oriented_features failed on {fails} rows (set μ to NaN).")
    return out


# =================== mapping: goals μ -> cards μ ===================
def _label_side_stats(df: pd.DataFrame, side: str, tag: str, col_base: str):
    col = f"{col_base}_{side}"
    s = pd.to_numeric(df[col], errors="coerce")
    mask = s.notna()
    vals = s[mask].astype(float).to_numpy()
    if vals.size == 0:
        print(f"[STAT] {tag}.{side}: N=0 (no labels)")
        return
    print(f"[STAT] {tag}.{side}: N={vals.size}, mean={np.mean(vals):.3f}, std={np.std(vals, ddof=1) if vals.size>1 else 0:.3f}, "
          f"min={np.min(vals):.0f}, p50={np.median(vals):.1f}, max={np.max(vals):.0f}")
    vc = pd.Series(vals).value_counts().sort_index()
    head = vc.iloc[:11] if vc.index.min() >= 0 and vc.index.max() <= 20 else vc.head(10)
    print(f"[STAT] {tag}.{side} value_counts (head): " + ", ".join(f"{int(k)}:{int(v)}" for k, v in head.items()))

def fit_card_means(train: pd.DataFrame,
                   df_val: Optional[pd.DataFrame],
                   which: str,
                   prior_home: float,
                   prior_away: float,
                   force_glm: bool = False,
                   skip_anchor: bool = False) -> Dict[str, Any]:
    """Fit mapping from goals μ to *card* μ (or fall back to constants)."""
    eps = 1e-6
    col_base = "cards"

    ch = pd.to_numeric(train[f"{col_base}_home"], errors="coerce").to_numpy(dtype=float)
    ca = pd.to_numeric(train[f"{col_base}_away"], errors="coerce").to_numpy(dtype=float)
    msk_h = np.isfinite(ch)
    msk_a = np.isfinite(ca)
    mean_obs_h = float(np.nanmean(ch[msk_h])) if msk_h.any() else np.nan
    mean_obs_a = float(np.nanmean(ca[msk_a])) if msk_a.any() else np.nan

    labels_ok_home = (np.isfinite(mean_obs_h) and 0.0 <= mean_obs_h <= 15.0)
    labels_ok_away = (np.isfinite(mean_obs_a) and 0.0 <= mean_obs_a <= 15.0)

    muH = pd.to_numeric(train["mu_goals_home"], errors="coerce").to_numpy(dtype=float)
    muA = pd.to_numeric(train["mu_goals_away"], errors="coerce").to_numpy(dtype=float)
    good_mu = np.isfinite(muH) & np.isfinite(muA)
    mu_ok = bool(good_mu.any())

    # If we don't trust μ features or not forcing GLM, use constant means = observed means (anchored)
    if (not mu_ok or not (labels_ok_home and labels_ok_away)) and not force_glm:
        mH = float(mean_obs_h) if np.isfinite(mean_obs_h) else prior_home
        mA = float(mean_obs_a) if np.isfinite(mean_obs_a) else prior_away
        print(f"[WARN] Using simple constants for card means: home≈{mH:.3f}, away≈{mA:.3f}")
        return {
            "kind": "constant",
            "means": {"home": mH, "away": mA},
            "card_type": which.lower(),
        }

    # --- GLM fit (Poisson regression on log μ goals -> card rate)
    def _fit_side(y_col, own_mu_col, opp_mu_col, prior):
        y = pd.to_numeric(train[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(pd.to_numeric(train[own_mu_col], errors="coerce")) & np.isfinite(pd.to_numeric(train[opp_mu_col], errors="coerce"))
        print(f"[DBG] GLM usable rows for {y_col}: {int(mask.sum())}/{len(mask)}")
        if mask.sum() < 50:
            intercept = math.log(max(eps, prior))
            return {"intercept": intercept, "b1": 0.0, "b2": 0.0}, np.array([])

        y = y[mask]
        x1 = np.log(np.clip(pd.to_numeric(train.loc[mask, own_mu_col], errors="coerce").to_numpy(dtype=float), eps, None))
        x2 = np.log(np.clip(pd.to_numeric(train.loc[mask, opp_mu_col], errors="coerce").to_numpy(dtype=float), eps, None))
        X = np.vstack([x1, x2]).T

        if HAVE_POISSON:
            alphas = [1e-6, 1e-4, 1e-3, 1e-2]
            rng = np.random.default_rng(42)
            idx = rng.permutation(X.shape[0])
            folds = np.array_split(idx, 3)
            best_alpha, best_loss = alphas[0], float("inf")
            for a in alphas:
                fold_losses = []
                for kfold in range(3):
                    val = folds[kfold]
                    trn = np.concatenate([folds[i] for i in range(3) if i != kfold])
                    pr = PoissonRegressor(alpha=a, max_iter=5000)
                    pr.fit(X[trn], y[trn])
                    yhat = np.clip(np.exp(pr.intercept_ + X[val] @ pr.coef_), 1e-9, None)
                    loss = np.mean(yhat - y[val] * np.log(yhat + 1e-12))
                    fold_losses.append(loss)
                m = float(np.mean(fold_losses))
                if m < best_loss:
                    best_loss, best_alpha = m, a
            pr = PoissonRegressor(alpha=best_alpha, max_iter=5000)
            pr.fit(X, y)
            coef = {"intercept": float(pr.intercept_), "b1": float(pr.coef_[0]), "b2": float(pr.coef_[1]), "alpha": best_alpha}
        else:
            z = np.log(np.clip(y, eps, None))
            Xls = np.hstack([np.ones((X.shape[0], 1)), X])
            beta_hat, *_ = np.linalg.lstsq(Xls, z, rcond=None)
            coef = {"intercept": float(beta_hat[0]), "b1": float(beta_hat[1]), "b2": float(beta_hat[2])}

        z_hat = coef["intercept"] + coef["b1"] * x1 + coef["b2"] * x2
        mu_hat = np.exp(z_hat)
        return coef, mu_hat

    coef_home, mu_hat_h = _fit_side("cards_home", "mu_goals_home", "mu_goals_away", prior_home)
    coef_away, mu_hat_a = _fit_side("cards_away", "mu_goals_away", "mu_goals_home", prior_away)

    def _scale(obs_mean, mu_hat, tag, side_ok: bool):
        if skip_anchor or not side_ok or (mu_hat.size == 0):
            s = 1.0; why = "skip"
        else:
            pred_bar = float(mu_hat.mean())
            s = 1.0 if pred_bar <= 1e-9 else float(np.clip(obs_mean / pred_bar, 0.5, 2.5))
            why = "fit" if pred_bar > 0 else "pred≈0"
        print(f"[DBG] anchor {tag}: obs≈{(obs_mean if np.isfinite(obs_mean) else float('nan')):.2f}, "
              f"pred_bar≈{(mu_hat.mean() if mu_hat.size else float('nan')):.2f}, scale={s:.3f} ({why})")
        return s

    s_home = _scale(mean_obs_h, mu_hat_h, "home", labels_ok_home)
    s_away = _scale(mean_obs_a, mu_hat_a, "away", labels_ok_away)

    return {
        "kind": "glm",
        "home": coef_home,
        "away": coef_away,
        "scale": {"home": s_home, "away": s_away},
        "fallback_means": {"home": prior_home, "away": prior_away},
        "card_type": which.lower(),
    }

def predict_means_for_row(row: pd.Series, mapping: Dict[str, Any], mean_floor: float = 0.05) -> Tuple[float, float]:
    """Module-level so inner closures can call it safely."""
    eps = 1e-6
    def _clip(x):
        # cards are smaller scale than corners; keep a conservative cap
        return float(np.clip(x, mean_floor, 8.0))

    muH = float(row.get("mu_goals_home", np.nan))
    muA = float(row.get("mu_goals_away", np.nan))

    kind = mapping.get("kind", "constant")

    if kind == "glm":
        ch = mapping["home"]; sh = float(mapping["scale"]["home"])
        ca = mapping["away"]; sa = float(mapping["scale"]["away"])
        zH = ch["intercept"] + ch["b1"] * math.log(max(eps, muH)) + ch["b2"] * math.log(max(eps, muA))
        zA = ca["intercept"] + ca["b1"] * math.log(max(eps, muA)) + ca["b2"] * math.log(max(eps, muH))
        mH = _clip(math.exp(zH) * sh)
        mA = _clip(math.exp(zA) * sa)
        return mH, mA

    if kind == "constant":
        baseH = float(mapping["means"]["home"])
        baseA = float(mapping["means"]["away"])
        return _clip(baseH), _clip(baseA)

    baseH = float(mapping.get("fallback_means", {}).get("home", 2.0))
    baseA = float(mapping.get("fallback_means", {}).get("away", 1.8))
    return _clip(baseH), _clip(baseA)


# =================== NB dispersion + distributions ===================
def estimate_nb_k(y: np.ndarray, clip_lo=3.0, clip_hi=1000.0) -> float:
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 50:
        return 150.0
    mu = float(y.mean())
    var = float(y.var(ddof=1)) if y.size > 1 else mu + mu**2/400.0
    if var <= mu + 1e-9:
        return clip_hi
    k = (mu*mu) / (var - mu)
    return float(np.clip(k, clip_lo, clip_hi))

def estimate_nb_k_grouped(df: pd.DataFrame, ycol: str, clip_lo=3.0, clip_hi=1000.0):
    y_all = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    k_global = estimate_nb_k(y_all, clip_lo, clip_hi)
    k_map: Dict[Tuple[int,int], float] = {}
    for (lg, ssn), grp in df.groupby(["league_id","season"]):
        y = pd.to_numeric(grp[ycol], errors="coerce").to_numpy(dtype=float)
        k_loc = estimate_nb_k(y, clip_lo, clip_hi)
        n = np.isfinite(y).sum()
        w = max(0.0, min(1.0, n / 400.0))
        # harmonic shrink
        k_shrunk = 1.0 / ((w / max(k_loc, 1e-9)) + ((1.0 - w) / max(k_global, 1e-9)))
        k_map[(int(lg), int(ssn))] = float(np.clip(k_shrunk, clip_lo, clip_hi))
    return float(k_global), k_map

from math import log, exp
def nb_pmf_vec(mu: float, k: float, nmax: int) -> np.ndarray:
    mu = max(1e-9, float(mu)); k = max(1e-9, float(k))
    p = mu / (k + mu)
    if HAVE_SCIPY:
        out = np.empty(nmax+1, float)
        qk = k * log(1.0 - p)
        for y in range(nmax+1):
            logc = gammaln(y + k) - gammaln(k) - gammaln(y + 1)
            out[y] = exp(logc + qk + y * log(p))
        s = out.sum()
        if s > 0: out /= s
        return out
    q = 1.0 - p
    pmf = np.zeros(nmax+1, dtype=float)
    pmf[0] = q**k
    coef = 1.0
    for y in range(1, nmax+1):
        coef *= (k + y - 1) / y
        pmf[y] = coef * (q**k) * (p**y)
    s = pmf.sum()
    if s > 0: pmf /= s
    return pmf

def conv_sum(pmf_a: np.ndarray, pmf_b: np.ndarray, nmax: int) -> np.ndarray:
    out = np.zeros(nmax+1, dtype=float)
    for i, pa in enumerate(pmf_a):
        if pa == 0: continue
        jmax = min(nmax - i, len(pmf_b) - 1)
        out[i:i+jmax+1] += pa * pmf_b[:jmax+1]
    s = out.sum()
    if s > 0:
        out /= s
    return out

def diff_distribution(pmf_h: np.ndarray, pmf_a: np.ndarray, dmin: int, dmax: int) -> Tuple[np.ndarray,int]:
    D = np.arange(dmin, dmax+1)
    out = np.zeros_like(D, dtype=float)
    for i, ph in enumerate(pmf_h):
        if ph == 0: continue
        for j, pa in enumerate(pmf_a):
            if pa == 0: continue
            d = i - j
            if dmin <= d <= dmax:
                out[d - dmin] += ph * pa
    s = out.sum()
    if s > 0: out /= s
    return out, dmin


# =================== Gaussian copula (optional) ===================
def totals_pmf_copula(pmfH: np.ndarray, pmfA: np.ndarray, rho: float,
                      sims: int, seed: int, nmax: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    Z = rng.multivariate_normal(mean=[0.0, 0.0],
                                cov=[[1.0, rho], [rho, 1.0]],
                                size=int(sims))
    U = _phi_arr(Z)
    cdfH = np.cumsum(pmfH)
    cdfA = np.cumsum(pmfA)
    h = np.searchsorted(cdfH, U[:, 0], side="left")
    a = np.searchsorted(cdfA, U[:, 1], side="left")
    h = np.clip(h, 0, nmax)
    a = np.clip(a, 0, nmax)
    tot = h + a
    tot = np.clip(tot, 0, nmax)
    pmfT = np.bincount(tot, minlength=nmax+1).astype(float)
    pmfT /= pmfT.sum()
    return pmfT, h, a


# =================== ρ from labels (optional, SciPy) ===================
def tau_to_rho_gaussian(tau: float) -> float:
    return float(np.sin(np.pi * tau / 2.0))

def estimate_rho_from_labels(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if not HAVE_SCIPY:
        return None, None
    m = df["cards_home"].notna() & df["cards_away"].notna()
    if not m.any(): return None, None
    h = pd.to_numeric(df.loc[m, "cards_home"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df.loc[m, "cards_away"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(h) & np.isfinite(a)
    h, a = h[mask], a[mask]
    if len(h) < 200: return None, None
    tau, _ = kendalltau(h, a)
    if not np.isfinite(tau): return None, None
    rho0 = np.clip(tau_to_rho_gaussian(tau), -0.35, 0.35)
    return float(rho0), float(tau)


# =================== calibration (totals & team lines) ===================
def _calib_balance_report(y, p, tag):
    N = int(len(y))
    pos = int(np.sum(y==1)); neg = N - pos
    if N == 0:
        print(f"[CAL] {tag}: N=0")
        return
    print(f"[CAL] {tag}: N={N}, positives={pos} ({pos/N:0.1%}), negatives={neg} ({neg/N:0.1%}), p-mean={np.mean(p):.3f}")

def fit_iso_curve(p: np.ndarray, y: np.ndarray) -> Optional[Dict[str, List[float]]]:
    y = y.astype(int)
    if len(y) < 200 or len(np.unique(y)) < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, y)
    return {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}

def apply_iso_scalar(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve: return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve["x"], float)
    y = np.array(curve["y"], float)
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))


# =============================== command ===============================
class Command(BaseCommand):
    help = "Train card models (yellow/red/total). Saves artifacts + test CSV."

    def add_arguments(self, ap):
        ap.add_argument("--leagues", type=int, nargs="+", required=True)
        ap.add_argument("--card-type", choices=["yellow","red","total"], required=True)
        ap.add_argument("--train-seasons", required=True)
        ap.add_argument("--val-seasons", required=True)
        ap.add_argument("--test-seasons", required=True)
        ap.add_argument("--outdir", required=True)

        ap.add_argument("--family", choices=["nb"], default="nb")
        ap.add_argument("--goals-artifact", default=None)

        # Cards are lower scoring than corners; defaults reflect typical ranges
        ap.add_argument("--total-lines", default="3.5,4.5,5.5")
        ap.add_argument("--team-lines",  default="1.5,2.5")
        ap.add_argument("--handicap-lines", default="-1.5,-1,-0.5,0,0.5,1,1.5")

        ap.add_argument("--nmax", type=int, default=15)
        ap.add_argument("--force-glm", action="store_true")
        ap.add_argument("--skip-anchor", action="store_true")
        ap.add_argument("--mean-floor", type=float, default=0.05)

        ap.add_argument("--rho-grid", default="0.0")   # "auto" or a float
        ap.add_argument("--sims", type=int, default=20000)
        ap.add_argument("--k-min", type=float, default=3.0)
        ap.add_argument("--k-max", type=float, default=1000.0)

    def handle(self, *args, **opts):
        if opts["family"] != "nb":
            raise ValueError("Only 'nb' family is supported at the moment.")

        random.seed(123); np.random.seed(123)

        leagues = list(opts["leagues"])
        which = str(opts["card_type"])
        train_seasons = expand_seasons(opts["train_seasons"])
        val_seasons   = expand_seasons(opts["val_seasons"])
        test_seasons  = expand_seasons(opts["test_seasons"])
        outdir = opts["outdir"]
        os.makedirs(outdir, exist_ok=True)

        totals_lines = parse_float_list(opts["total_lines"])
        team_lines   = parse_float_list(opts["team_lines"])
        hcp_lines    = parse_float_list(opts["handicap_lines"])

        print("────────────────────────────────────────────────────────────────")
        print(f"Cards Trainer: {which} | Leagues={leagues}  Train={train_seasons}  Val={val_seasons}  Test={test_seasons}")
        print(f"Lines totals={totals_lines} team={team_lines} (nmax={opts['nmax']})")
        print("────────────────────────────────────────────────────────────────")

        # -------- Load data
        df_train = load_rows(leagues, train_seasons, which)
        df_val   = load_rows(leagues, val_seasons, which)
        df_test  = load_rows(leagues, test_seasons, which)

        for col in ("cards_home","cards_away"):
            df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
            df_val[col]   = pd.to_numeric(df_val[col], errors="coerce")
            df_test[col]  = pd.to_numeric(df_test[col], errors="coerce")

        # Attach goals μ (optional, for GLM)
        art = load_goals_artifact(opts["goals_artifact"])
        df_train = attach_goal_mus(df_train, art)
        df_val   = attach_goal_mus(df_val, art)
        df_test  = attach_goal_mus(df_test, art)

        def _label_count(df):
            m = df["cards_home"].notna() & df["cards_away"].notna()
            zeros = int(((df["cards_home"]==0) & (df["cards_away"]==0)).sum())
            return int(m.sum()), len(df), zeros

        nlab_tr, nt_tr, z_tr = _label_count(df_train)
        nlab_va, nt_va, z_va = _label_count(df_val)
        nlab_te, nt_te, z_te = _label_count(df_test)
        print(f"[INFO] Labeled pairs: train={nlab_tr}/{nt_tr} (0+0={z_tr}), "
              f"val={nlab_va}/{nt_va} (0+0={z_va}), test={nlab_te}/{nt_te} (0+0={z_te})")

        # Borrow val for fitting if train has no labels
        df_fit = df_train
        borrowed_from_val = False
        if nlab_tr == 0 and nlab_va > 0:
            df_fit = pd.concat([df_train, df_val], ignore_index=True)
            borrowed_from_val = True
            print("[INFO] No labeled training rows — temporarily borrowing validation labels for model fitting only.")

        # quick stats
        def _pair_eq_share(df):
            m = df["cards_home"].notna() & df["cards_away"].notna()
            return float((df.loc[m, "cards_home"] == df.loc[m, "cards_away"]).mean()) if m.any() else 0.0
        print(f"[STAT] H==A share: train={_pair_eq_share(df_train):.1%}, val={_pair_eq_share(df_val):.1%}, test={_pair_eq_share(df_test):.1%}")
        _label_side_stats(df_train, "home", "train", "cards"); _label_side_stats(df_train, "away", "train", "cards")
        _label_side_stats(df_val,   "home", "val",   "cards"); _label_side_stats(df_val,   "away", "val",   "cards")

        # Priors depend on card type
        priorH, priorA = default_priors(which)

        # -------- Fit mapping
        mapping = fit_card_means(
            df_fit,
            df_val,
            which=which,
            prior_home=priorH, prior_away=priorA,
            force_glm=bool(opts["force_glm"]),
            skip_anchor=bool(opts["skip_anchor"]),
        )

        # -------- NB dispersion from df_fit (grouped + shrink)
        kH_global, kH_map = estimate_nb_k_grouped(df_fit, "cards_home",
                                              clip_lo=opts["k_min"], clip_hi=opts["k_max"])
        kA_global, kA_map = estimate_nb_k_grouped(df_fit, "cards_away",
                                              clip_lo=opts["k_min"], clip_hi=opts["k_max"])
        print(f"[INFO] NB k: home_global={kH_global:.1f} groups={len(kH_map)} | away_global={kA_global:.1f} groups={len(kA_map)}")

        def k_for(row, side):
            key = (int(row["league_id"]), int(row["season"]))
            return kH_map.get(key, kH_global) if side=="H" else kA_map.get(key, kA_global)

        def _pmfs_for_row(r):
            mH, mA = predict_means_for_row(r, mapping, mean_floor=opts["mean_floor"])
            pmfH = nb_pmf_vec(mH, k_for(r, "H"), opts["nmax"])
            pmfA = nb_pmf_vec(mA, k_for(r, "A"), opts["nmax"])
            return pmfH, pmfA

        # -------- ρ tuning
        def _val_totals_logloss(rho: float) -> float:
            if df_val.empty:
                return 1e9
            eps = 1e-12
            mask = df_val["cards_home"].notna() & df_val["cards_away"].notna()
            if not mask.any():
                return 1e9
            loss = 0.0; n = 0
            for _, r in df_val.loc[mask].iterrows():
                pmfH, pmfA = _pmfs_for_row(r)
                tot = conv_sum(pmfH, pmfA, opts["nmax"]) if abs(rho) < 1e-9 else totals_pmf_copula(pmfH, pmfA, rho=rho, sims=opts["sims"], seed=int(r["fixture_id"]), nmax=opts["nmax"])[0]
                line_losses = []
                for L in totals_lines:
                    thr = math.floor(L + 1e-9) + 1
                    p_over = float(tot[thr:].sum())
                    y = int((int(r["cards_home"]) + int(r["cards_away"])) > math.floor(L + 1e-9))
                    p = min(max(p_over, eps), 1.0 - eps)
                    line_losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
                loss += float(np.mean(line_losses)); n += 1
            return loss / max(1, n)

        chosen_rho = 0.0
        tau_raw = None
        if str(opts["rho_grid"]).lower() == "auto":
            rho0, tau_raw = estimate_rho_from_labels(df_fit)
            if rho0 is not None:
                base = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
                grid = list(np.clip(rho0 + base, -0.35, 0.35)) + [0.0]
            else:
                grid = [-0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
            scores = [(r, _val_totals_logloss(r)) for r in grid]
            chosen_rho, best = min(scores, key=lambda t: t[1])
            print(f"[INFO] Chosen rho={chosen_rho:.3f} (val totals logloss={best:.4f})"
                  f"{'' if tau_raw is None else f', tau≈{tau_raw:.3f}'}")
        else:
            try:
                chosen_rho = float(opts["rho_grid"])
            except Exception:
                chosen_rho = 0.0
            if abs(chosen_rho) < 1e-9:
                print("[INFO] Using independent marginals (rho=0.0)")

        # -------- Calibration (use true validation split only)
        cal_totals: Dict[str, Dict[str, List[float]]] = {}
        cal_team_home: Dict[str, Dict[str, List[float]]] = {}
        cal_team_away: Dict[str, Dict[str, List[float]]] = {}

        mask_val = df_val["cards_home"].notna() & df_val["cards_away"].notna()
        if mask_val.any():
            rows_val = df_val.loc[mask_val]
            preds_totals, preds_team = [], []

            for _, r in rows_val.iterrows():
                pmfH, pmfA = _pmfs_for_row(r)
                tot = conv_sum(pmfH, pmfA, opts["nmax"]) if abs(chosen_rho) < 1e-9 else totals_pmf_copula(pmfH, pmfA, rho=chosen_rho, sims=opts["sims"], seed=int(r["fixture_id"]), nmax=opts["nmax"])[0]
                rowT = {"cards_home": int(r["cards_home"]), "cards_away": int(r["cards_away"])}
                for L in totals_lines:
                    thr = math.floor(L + 1e-9) + 1
                    rowT[f"p_over_{L}"] = float(tot[thr:].sum())
                preds_totals.append(rowT)

                cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)
                rowTeam = {"cards_home": int(r["cards_home"]), "cards_away": int(r["cards_away"])}
                for L in team_lines:
                    thr = math.floor(L + 1e-9) + 1
                    idxH = min(thr-1, opts["nmax"]); idxA = min(thr-1, opts["nmax"])
                    rowTeam[f"pH_over_{L}"] = float(1.0 - cdfH[idxH])
                    rowTeam[f"pA_over_{L}"] = float(1.0 - cdfA[idxA])
                preds_team.append(rowTeam)

            dfp = pd.DataFrame(preds_totals)
            for L in totals_lines:
                y = (dfp["cards_home"].values + dfp["cards_away"].values > math.floor(L + 1e-9)).astype(int)
                p = dfp[f"p_over_{L}"].values.astype(float)
                p = p + (np.arange(p.size) % 997) * 1e-12
                _calib_balance_report(y, p, f"totals L={L}")
                curve = fit_iso_curve(p, y)
                if curve:
                    cal_totals[str(L)] = curve

            dft = pd.DataFrame(preds_team)
            for L in team_lines:
                yH = (dft["cards_home"].values > math.floor(L + 1e-9)).astype(int)
                pH = dft[f"pH_over_{L}"].values.astype(float)
                pH = pH + (np.arange(pH.size) % 997) * 1e-12
                _calib_balance_report(yH, pH, f"team HOME L={L}")
                curveH = fit_iso_curve(pH, yH)
                if curveH:
                    cal_team_home[str(L)] = curveH

                yA = (dft["cards_away"].values > math.floor(L + 1e-9)).astype(int)
                pA = dft[f"pA_over_{L}"].values.astype(float)
                pA = pA + (np.arange(pA.size) % 997) * 1e-12
                _calib_balance_report(yA, pA, f"team AWAY L={L}")
                curveA = fit_iso_curve(pA, yA)
                if curveA:
                    cal_team_away[str(L)] = curveA
            if cal_totals:
                print(f"[INFO] Totals calibration curves learned for lines: {sorted(cal_totals.keys())}")
            if cal_team_home or cal_team_away:
                print(f"[INFO] Team calibration curves learned (home={sorted(cal_team_home.keys())}, away={sorted(cal_team_away.keys())})")
        else:
            print("[WARN] Skipping calibration (no labeled validation rows).")

        # -------- Predictions (TEST)
        rows_out = []
        for _, r in df_test.iterrows():
            pmfH, pmfA = _pmfs_for_row(r)
            tot = conv_sum(pmfH, pmfA, opts["nmax"]) if abs(chosen_rho) < 1e-9 else totals_pmf_copula(pmfH, pmfA, rho=chosen_rho, sims=opts["sims"], seed=int(r["fixture_id"]), nmax=opts["nmax"])[0]

            out = {
                "fixture_id": int(r["fixture_id"]),
                "league_id": int(r["league_id"]),
                "season": int(r["season"]),
                "kickoff_utc": r["kickoff_utc"],
            }

            mH, mA = predict_means_for_row(r, mapping, mean_floor=opts["mean_floor"])
            out["m_cards_home"] = float(mH)
            out["m_cards_away"] = float(mA)

            # totals markets
            for L in totals_lines:
                thr = math.floor(L + 1e-9) + 1
                p_over = float(tot[thr:].sum())
                p_over = apply_iso_scalar(p_over, cal_totals.get(str(L)))
                out[f"totals_over_{L}"]  = p_over
                out[f"totals_under_{L}"] = float(1.0 - p_over)

            # team totals
            cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)
            for L in team_lines:
                thr = math.floor(L + 1e-9) + 1
                idxH = min(thr-1, opts["nmax"]); idxA = min(thr-1, opts["nmax"])
                pH_over = float(1.0 - cdfH[idxH])
                pA_over = float(1.0 - cdfA[idxA])
                pH_over = apply_iso_scalar(pH_over, cal_team_home.get(str(L)))
                pA_over = apply_iso_scalar(pA_over, cal_team_away.get(str(L)))
                out[f"home_over_{L}"] = pH_over
                out[f"home_under_{L}"] = float(1.0 - pH_over)
                out[f"away_over_{L}"] = pA_over
                out[f"away_under_{L}"] = float(1.0 - pA_over)

            # handicaps on card difference (kept for completeness)
            dmin, dmax = -opts["nmax"], opts["nmax"]
            diff, d0 = diff_distribution(pmfH, pmfA, dmin, dmax)
            for h in hcp_lines:
                if float(h).is_integer():
                    h_int = int(h)
                    p_win  = float(diff[(h_int+1 - d0):].sum()) if (h_int+1 - d0) < len(diff) else 0.0
                    p_push = float(diff[(h_int - d0)]) if (dmin <= h_int <= dmax) else 0.0
                    p_lose = float(1.0 - p_win - p_push)
                else:
                    thr = math.floor(h + 1e-9)
                    start = (thr+1 - d0)
                    p_win  = float(diff[start:].sum()) if (start < len(diff)) else 0.0
                    p_push = 0.0
                    p_lose = float(1.0 - p_win)
                out[f"hcp_home_{h}_win"]  = p_win
                out[f"hcp_home_{h}_push"] = p_push
                out[f"hcp_home_{h}_lose"] = p_lose

            rows_out.append(out)

        preds = pd.DataFrame(rows_out)

        # -------- Validation metrics (for reference)
        metrics = {"val": {}}
        def _compute_val_metrics():
            mask = df_val["cards_home"].notna() & df_val["cards_away"].notna()
            if not mask.any(): return
            ll=0.0; nll=0
            bs=0.0; nbs=0
            for _, r in df_val.loc[mask].iterrows():
                pmfH, pmfA = _pmfs_for_row(r)
                tot = conv_sum(pmfH, pmfA, opts["nmax"]) if abs(chosen_rho)<1e-9 else totals_pmf_copula(pmfH, pmfA, rho=chosen_rho, sims=opts["sims"], seed=int(r["fixture_id"]), nmax=opts["nmax"])[0]
                for L in totals_lines:
                    thr = math.floor(L+1e-9)+1
                    p_over = float(tot[thr:].sum())
                    y = int((int(r["cards_home"])+int(r["cards_away"])) > math.floor(L+1e-9))
                    p = min(max(p_over,1e-12), 1.0-1e-12)
                    ll += -(y*math.log(p)+(1-y)*math.log(1-p)); nll+=1
                    bs += (p - y)**2; nbs+=1
            if nll>0: metrics["val"]["totals_logloss"] = ll/nll
            if nbs>0: metrics["val"]["totals_brier"] = bs/nbs

        _compute_val_metrics()

        # -------- Save artifacts + predictions
        artifacts = {
            "type": f"cards_{which}_nb_v1",
            "family": "nb",
            "card_type": which,
            "mapping": mapping,
            "mapping_kind": mapping.get("kind", "constant"),
            "dispersion": {
                "k_home_global": float(kH_global),
                "k_away_global": float(kA_global),
                "k_home_map": {f"{lg}-{ss}": float(v) for (lg,ss), v in kH_map.items()},
                "k_away_map": {f"{lg}-{ss}": float(v) for (lg,ss), v in kA_map.items()},
            },
            "lines": {"totals": totals_lines, "team": team_lines, "handicap": hcp_lines},
            "nmax": int(opts["nmax"]),
            "rho": float(chosen_rho),
            "tau_raw": None if tau_raw is None else float(tau_raw),
            "totals_calibration": {k: {"x": [float(xx) for xx in v["x"]], "y": [float(yy) for yy in v["y"]]} for k, v in ({} if not cal_totals else cal_totals).items()},
            "team_calibration": {
                "home": {k: {"x": [float(xx) for xx in v["x"]], "y": [float(yy) for yy in v["y"]]} for k, v in ({} if not cal_team_home else cal_team_home).items()},
                "away": {k: {"x": [float(xx) for xx in v["x"]], "y": [float(yy) for yy in v["y"]]} for k, v in ({} if not cal_team_away else cal_team_away).items()},
            },
            "mean_floor": float(opts["mean_floor"]),
            "training": {
                "leagues": leagues,
                "train_seasons": train_seasons,
                "val_seasons": val_seasons,
                "test_seasons": test_seasons,
                "borrowed_val_for_fit": bool(borrowed_from_val),
            },
            "metrics": metrics,
            "notes": "Cards (yellow/red/total) trained from MatchStats; GLM optional; grouped NB; isotonic calibration applied.",
        }

        os.makedirs(outdir, exist_ok=True)
        art_path = os.path.join(outdir, f"artifacts.cards.{which}.json")
        with open(art_path, "w") as f:
            json.dump(_to_py(artifacts), f, indent=2)

        pr_path = os.path.join(outdir, f"preds_test.cards.{which}.csv")
        preds.to_csv(pr_path, index=False)

        if not preds.empty:
            print("[DBG] sample means (test head):",
                  f'home≈{preds["m_cards_home"].head().mean():.2f},',
                  f'away≈{preds["m_cards_away"].head().mean():.2f}')
            print("[DBG] unique mean counts:",
                  f'H={preds["m_cards_home"].nunique()} A={preds["m_cards_away"].nunique()}')

        print(f"[INFO] Saved artifacts to {art_path}")
        print(f"[INFO] Wrote test predictions to {pr_path}")
        print("[INFO] Done.")
