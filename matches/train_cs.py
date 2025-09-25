#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct Score trainer:
- Robust FT goals extractor (works with varying Match fields or JSON)
- μ from goals artifact features (safe fallbacks + anchoring)
- Poisson or Negative Binomial marginals (shrunk dispersion per league-season)
- Independent or Gaussian-copula join (ρ tuned on validation via totals logloss)
- Outputs CS matrix + totals + handicap; stores training artifact
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
np.seterr(all="ignore")

from matches.models import Match
try:
    from matches.models import MLTrainingMatch  # optional
    HAVE_MLTRAIN = True
except Exception:
    MLTrainingMatch = None  # type: ignore
    HAVE_MLTRAIN = False

# ---------- optional sklearn/scipy ----------
try:
    from sklearn.linear_model import PoissonRegressor
    HAVE_POISSON = True
except Exception:
    HAVE_POISSON = False

try:
    from scipy.special import gammaln
    from scipy.stats import kendalltau
    from scipy.special import erf as _erf_ufunc
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    gammaln = None
    kendalltau = None
    _erf_ufunc = None


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

def load_goals_artifact(path: Optional[str]) -> GoalsArtifact:
    if not path:
        raise ValueError("--goals-artifact is required for CS training")
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

# feature builder (import path tolerant)
try:
    from prediction.train_goals import build_oriented_features  # type: ignore
except Exception:
    try:
        from train_goals import build_oriented_features  # type: ignore
    except Exception:
        build_oriented_features = None  # type: ignore


def _phi_arr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if _erf_ufunc is not None:
        return 0.5 * (1.0 + _erf_ufunc(z / math.sqrt(2.0)))
    # slow fallback
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def _to_py(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(v) for v in obj]
    return obj


# ======================= FT goals extraction =======================
def _safe_int_0_20(x: Any) -> Optional[int]:
    try:
        v = int(x)
        if 0 <= v <= 20:
            return v
    except Exception:
        pass
    return None

_HOME_FT_ATTRS = [
    "home_ft_goals", "ft_home_goals", "home_goals_ft", "home_fulltime_goals",
    "fulltime_home", "home_ft", "home_score_ft", "home_score", "home_goals",
]
_AWAY_FT_ATTRS = [
    "away_ft_goals", "ft_away_goals", "away_goals_ft", "away_fulltime_goals",
    "fulltime_away", "away_ft", "away_score_ft", "away_score", "away_goals",
]

def _extract_ft_from_match_fields(m: Match) -> Tuple[Optional[int], Optional[int]]:
    h = a = None
    for name in _HOME_FT_ATTRS:
        if hasattr(m, name):
            h = _safe_int_0_20(getattr(m, name))
            if h is not None:
                break
    for name in _AWAY_FT_ATTRS:
        if hasattr(m, name):
            a = _safe_int_0_20(getattr(m, name))
            if a is not None:
                break
    return h, a

def _has_home_away_pair(d: dict) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(d, dict):
        return None, None
    for hk, ak in (("home", "away"), ("homeTeam", "awayTeam"), ("localteam", "visitorteam")):
        if hk in d and ak in d:
            h = _safe_int_0_20(d[hk])
            a = _safe_int_0_20(d[ak])
            if h is not None and a is not None:
                return h, a
    return None, None

def _extract_ft_from_json(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(obj, (dict, list)):
        return None, None

    if isinstance(obj, dict):
        # goals: {home, away}
        if "goals" in obj and isinstance(obj["goals"], dict):
            h, a = _has_home_away_pair(obj["goals"])
            if h is not None and a is not None:
                return h, a
        # score.fulltime / score.ft
        if "score" in obj and isinstance(obj["score"], dict):
            for full_key in ("fulltime", "ft"):
                block = obj["score"].get(full_key)
                if isinstance(block, dict):
                    h, a = _has_home_away_pair(block)
                    if h is not None and a is not None:
                        return h, a
        # teams.{home,away}.goals
        if "teams" in obj and isinstance(obj["teams"], dict):
            th = obj["teams"].get("home"); ta = obj["teams"].get("away")
            if isinstance(th, dict) and isinstance(ta, dict):
                h = _safe_int_0_20(th.get("goals"))
                a = _safe_int_0_20(ta.get("goals"))
                if h is not None and a is not None:
                    return h, a

        preferred = ("fulltime", "ft", "score", "goals", "result", "final")
        for k, v in obj.items():
            if isinstance(v, dict) and any(pk in str(k).lower() for pk in preferred):
                h, a = _extract_ft_from_json(v)
                if h is not None and a is not None:
                    return h, a
        for v in obj.values():
            if isinstance(v, (dict, list)):
                h, a = _extract_ft_from_json(v)
                if h is not None and a is not None:
                    return h, a

    else:  # list
        for v in obj:
            h, a = _extract_ft_from_json(v)
            if h is not None and a is not None:
                return h, a

    return None, None

def _get_ft_score_for_fixture(fixture_id: int, stats_like: Any = None) -> Tuple[Optional[int], Optional[int]]:
    h = a = None
    try:
        m = Match.objects.only("id", "raw_result_json").get(pk=int(fixture_id))
    except Match.DoesNotExist:
        m = None

    if m is not None:
        hh, aa = _extract_ft_from_match_fields(m)
        if hh is not None and aa is not None:
            return hh, aa
        if getattr(m, "raw_result_json", None):
            hh, aa = _extract_ft_from_json(m.raw_result_json)
            if hh is not None and aa is not None:
                return hh, aa

    if stats_like:
        hh, aa = _extract_ft_from_json(stats_like)
        if hh is not None and aa is not None:
            return hh, aa

    return None, None

def load_rows(leagues: List[int], seasons: List[int]) -> pd.DataFrame:
    recs = []
    missing = 0

    if HAVE_MLTRAIN:
        base = (MLTrainingMatch.objects
                .filter(league_id__in=leagues, season__in=seasons)
                .order_by("kickoff_utc")
                .only("fixture_id","league_id","season","kickoff_utc","stats10_json"))
        it = base.iterator()
        for r in it:
            h, a = _get_ft_score_for_fixture(int(r.fixture_id), stats_like=getattr(r, "stats10_json", None))
            if h is None or a is None:
                missing += 1
            recs.append({
                "fixture_id": int(r.fixture_id),
                "league_id": int(r.league_id),
                "season": int(r.season),
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "stats10_json": getattr(r, "stats10_json", None),
                "home_ft": None if h is None else int(h),
                "away_ft": None if a is None else int(a),
            })
    else:
        base = (Match.objects
                .filter(league_id__in=leagues, season__in=seasons)
                .order_by("kickoff_utc")
                .only("id","league_id","season","kickoff_utc","raw_result_json"))
        it = base.iterator()
        for m in it:
            h, a = _get_ft_score_for_fixture(int(m.id), stats_like=getattr(m, "raw_result_json", None))
            if h is None or a is None:
                missing += 1
            recs.append({
                "fixture_id": int(m.id),
                "league_id": int(m.league_id),
                "season": int(m.season),
                "kickoff_utc": m.kickoff_utc.isoformat(),
                "stats10_json": getattr(m, "raw_result_json", None),
                "home_ft": None if h is None else int(h),
                "away_ft": None if a is None else int(a),
            })

    df = pd.DataFrame(recs)
    if len(df):
        print("[INFO] Loading rows (goals labels)...")
        if missing:
            print(f"[WARN] Missing FT score for {missing}/{len(df)} rows.")
    return df


# =================== goals μ attachment ===================
def attach_goal_mus(df: pd.DataFrame, art: GoalsArtifact) -> pd.DataFrame:
    muH, muA = [], []
    fails = 0
    for _, r in df.iterrows():
        if build_oriented_features is None:
            muH.append(np.nan); muA.append(np.nan); continue
        try:
            xh, xa, _ = build_oriented_features({"stats10_json": r.get("stats10_json")})
            muH.append(mu_from_features(xh, art))
            muA.append(mu_from_features(xa, art))
        except Exception:
            muH.append(np.nan); muA.append(np.nan); fails += 1
    out = df.copy()
    out["mu_home"] = muH
    out["mu_away"] = muA
    if fails:
        print(f"[WARN] build_oriented_features failed on {fails} rows (μ set NaN).")
    return out

def anchor_mus(df: pd.DataFrame, train_mask: np.ndarray) -> Tuple[float,float]:
    """Scale μ so train μ means match observed train means (protects against global drift)."""
    tr = df.loc[train_mask]
    m = tr["home_ft"].notna() & tr["away_ft"].notna() & tr["mu_home"].notna() & tr["mu_away"].notna()
    if not m.any():
        return 1.0, 1.0
    obs_h = float(tr.loc[m, "home_ft"].mean())
    obs_a = float(tr.loc[m, "away_ft"].mean())
    pmu_h = float(tr.loc[m, "mu_home"].mean())
    pmu_a = float(tr.loc[m, "mu_away"].mean())
    s_h = 1.0 if pmu_h <= 1e-9 else float(np.clip(obs_h / pmu_h, 0.5, 2.0))
    s_a = 1.0 if pmu_a <= 1e-9 else float(np.clip(obs_a / pmu_a, 0.5, 2.0))
    print(f"[DBG] anchor μ: obs_h≈{obs_h:.3f}, pred_h≈{pmu_h:.3f} → scale_h={s_h:.3f} | "
          f"obs_a≈{obs_a:.3f}, pred_a≈{pmu_a:.3f} → scale_a={s_a:.3f}")
    return s_h, s_a


# =================== NB dispersion + distributions ===================
def estimate_nb_k(y: np.ndarray, clip_lo=3.0, clip_hi=1000.0) -> float:
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 50:
        return 200.0
    mu = float(y.mean())
    var = float(y.var(ddof=1)) if y.size > 1 else mu + mu**2/400.0
    if var <= mu + 1e-9:
        return clip_hi
    k = (mu * mu) / (var - mu)
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
        k_shrunk = 1.0 / ((w / max(k_loc, 1e-9)) + ((1.0 - w) / max(k_global, 1e-9)))
        k_map[(int(lg), int(ssn))] = float(np.clip(k_shrunk, clip_lo, clip_hi))
    return float(k_global), k_map

from math import log, exp

def pois_pmf_vec(mu: float, nmax: int) -> np.ndarray:
    mu = max(1e-12, float(mu))
    out = np.zeros(nmax+1, float)
    out[0] = math.exp(-mu)
    for k in range(1, nmax+1):
        out[k] = out[k-1] * (mu / k)
    s = out.sum()
    if s > 0: out /= s
    return out

def nb_pmf_vec(mu: float, k: float, nmax: int) -> np.ndarray:
    mu = max(1e-12, float(mu)); k = max(1e-12, float(k))
    p = mu / (k + mu)
    if _HAVE_SCIPY:
        out = np.empty(nmax+1, float)
        qk = k * log(1.0 - p)
        for y in range(nmax+1):
            logc = gammaln(y + k) - gammaln(k) - gammaln(y + 1)
            out[y] = exp(logc + qk + y * log(p))
        s = out.sum()
        if s > 0: out /= s
        return out
    # fallback
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

def copula_joint_from_marginals(pmfH: np.ndarray, pmfA: np.ndarray, rho: float, sims: int, seed: int, nmax: int) -> np.ndarray:
    """Return joint PMF matrix via Gaussian copula simulation."""
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    Z = rng.multivariate_normal([0.0,0.0], [[1.0,rho],[rho,1.0]], size=int(sims))
    U = _phi_arr(Z)
    cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)
    i = np.searchsorted(cdfH, U[:,0], side="left")
    j = np.searchsorted(cdfA, U[:,1], side="left")
    i = np.clip(i, 0, nmax); j = np.clip(j, 0, nmax)
    M = np.zeros((nmax+1, nmax+1), float)
    np.add.at(M, (i, j), 1.0)
    M /= M.sum()
    return M

def conv_totals(pmfH: np.ndarray, pmfA: np.ndarray, nmax: int) -> np.ndarray:
    out = np.zeros(nmax+1, float)
    for i, ph in enumerate(pmfH):
        if ph == 0: continue
        jmax = min(nmax - i, len(pmfA) - 1)
        out[i:i+jmax+1] += ph * pmfA[:jmax+1]
    s = out.sum()
    if s > 0: out /= s
    return out

def totals_from_joint(J: np.ndarray, nmax: int) -> np.ndarray:
    """Totals PMF from a (nmax+1 x nmax+1) joint goals matrix."""
    tot = np.zeros(nmax + 1, float)
    for i in range(nmax + 1):
        jmax = min(nmax, nmax - i)
        tot[i:i + jmax + 1] += J[i, :jmax + 1]
    return tot


# =================== ρ estimation ===================
def tau_to_rho_gaussian(tau: float) -> float:
    return float(np.sin(np.pi * tau / 2.0))

def estimate_rho_from_labels(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if not _HAVE_SCIPY:
        return None, None
    m = df["home_ft"].notna() & df["away_ft"].notna()
    if not m.any(): return None, None
    h = pd.to_numeric(df.loc[m, "home_ft"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df.loc[m, "away_ft"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(h) & np.isfinite(a)
    h, a = h[mask], a[mask]
    if len(h) < 200: return None, None
    tau, _ = kendalltau(h, a)
    if not np.isfinite(tau): return None, None
    rho0 = np.clip(tau_to_rho_gaussian(tau), -0.35, 0.35)
    return float(rho0), float(tau)


# =============================== main ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", type=int, nargs="+", required=True)
    ap.add_argument("--train-seasons", required=True)
    ap.add_argument("--val-seasons", required=True)
    ap.add_argument("--test-seasons", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--goals-artifact", required=True)

    ap.add_argument("--family", choices=["poiss","nb"], default="poiss")
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--sims", type=int, default=100000)
    ap.add_argument("--rho-grid", default="auto")
    ap.add_argument("--totals-lines", default="1.5,2.5,3.5")
    ap.add_argument("--handicap-lines", default="-1,-0.5,0,0.5,1")

    ap.add_argument("--k-min", type=float, default=3.0)
    ap.add_argument("--k-max", type=float, default=1000.0)
    ap.add_argument("--mean-floor", type=float, default=0.2)

    args = ap.parse_args()
    random.seed(123); np.random.seed(123)

    leagues = args.leagues
    train_seasons = expand_seasons(args.train_seasons)
    val_seasons   = expand_seasons(args.val_seasons)
    test_seasons  = expand_seasons(args.test_seasons)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    totals_lines = parse_float_list(args.totals_lines)
    hcp_lines    = parse_float_list(args.handicap_lines)

    # -------- Load data
    df_train = load_rows(leagues, train_seasons)
    df_val   = load_rows(leagues, val_seasons)
    df_test  = load_rows(leagues, test_seasons)

    # Attach μ from goals artifact
    art = load_goals_artifact(args.goals_artifact)
    df_train = attach_goal_mus(df_train, art)
    df_val   = attach_goal_mus(df_val, art)
    df_test  = attach_goal_mus(df_test, art)

    # Basic train/val label counts
    m_tr = df_train["home_ft"].notna() & df_train["away_ft"].notna()
    m_va = df_val["home_ft"].notna() & df_val["away_ft"].notna()
    m_te = df_test["home_ft"].notna() & df_test["away_ft"].notna()
    print(f"[INFO] Labeled pairs: train={int(m_tr.sum())}/{len(df_train)}, "
          f"val={int(m_va.sum())}/{len(df_val)}, test={int(m_te.sum())}/{len(df_test)}")

    # μ anchoring (protects against calibration drift)
    s_h, s_a = anchor_mus(
        pd.concat([df_train, df_val], ignore_index=True),
        train_mask=np.r_[np.ones(len(df_train), dtype=bool), np.zeros(len(df_val), dtype=bool)]
    )

    def _predict_mus(row):
        mh = float(row.get("mu_home", np.nan)); ma = float(row.get("mu_away", np.nan))
        if not np.isfinite(mh) or not np.isfinite(ma):
            mh, ma = 1.3, 1.2  # safe fallback
        mh = float(np.clip(mh * s_h, args.mean_floor, 8.0))
        ma = float(np.clip(ma * s_a, args.mean_floor, 8.0))
        return mh, ma

    # -------- NB dispersion for 'nb' family
    if args.family == "nb":
        kH_global, kH_map = estimate_nb_k_grouped(df_train.loc[m_tr], "home_ft", clip_lo=args.k_min, clip_hi=args.k_max)
        kA_global, kA_map = estimate_nb_k_grouped(df_train.loc[m_tr], "away_ft", clip_lo=args.k_min, clip_hi=args.k_max)
        print(f"[INFO] NB k (shrunk): home_global={kH_global:.1f}, groups={len(kH_map)}; away_global={kA_global:.1f}, groups={len(kA_map)}")
    else:
        kH_global, kA_global = None, None
        kH_map, kA_map = {}, {}

    def k_for(row, side):
        if args.family != "nb":
            return None
        key = (int(row["league_id"]), int(row["season"]))
        return (kH_map.get(key, kH_global) if side=="H" else kA_map.get(key, kA_global))

    def pmf_side(mu, k, nmax):
        if args.family == "poiss":
            return pois_pmf_vec(mu, nmax)
        return nb_pmf_vec(mu, float(k), nmax)

    # -------- ρ tuning on validation (totals logloss)
    def _val_totals_logloss(rho: float) -> float:
        mask = df_val["home_ft"].notna() & df_val["away_ft"].notna()
        if not mask.any(): return 1e9
        eps = 1e-12; loss=0.0; n=0
        for _, r in df_val.loc[mask].iterrows():
            mh, ma = _predict_mus(r)
            ph = pmf_side(mh, k_for(r,"H"), args.nmax)
            pa = pmf_side(ma, k_for(r,"A"), args.nmax)
            if abs(rho) < 1e-9:
                tot = conv_totals(ph, pa, args.nmax)
            else:
                J = copula_joint_from_marginals(ph, pa, rho=rho, sims=args.sims, seed=int(r["fixture_id"]), nmax=args.nmax)
                tot = totals_from_joint(J, args.nmax)
            for L in totals_lines:
                thr = math.floor(L + 1e-9) + 1
                p_over = float(tot[thr:].sum())
                y = int((int(r["home_ft"]) + int(r["away_ft"])) > math.floor(L + 1e-9))
                p = min(max(p_over, eps), 1.0 - eps)
                loss += -(y * math.log(p) + (1 - y) * math.log(1 - p)); n += 1
        return loss / max(1, n)

    chosen_rho = 0.0; tau_raw = None
    if str(args.rho_grid).lower() == "auto":
        rho0, tau_raw = estimate_rho_from_labels(df_train)
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
            chosen_rho = float(args.rho_grid)
        except Exception:
            chosen_rho = 0.0
        if abs(chosen_rho) < 1e-9:
            print("[INFO] Using independent marginals (rho=0.0)")

    # -------- Predictions (TEST)
    rows_out = []
    for _, r in df_test.iterrows():
        mh, ma = _predict_mus(r)
        ph = pmf_side(mh, k_for(r,"H"), args.nmax)
        pa = pmf_side(ma, k_for(r,"A"), args.nmax)

        if abs(chosen_rho) < 1e-9:
            J = np.outer(ph, pa)  # (nmax+1, nmax+1)
        else:
            J = copula_joint_from_marginals(ph, pa, rho=chosen_rho, sims=args.sims, seed=int(r["fixture_id"]), nmax=args.nmax)

        out: Dict[str, Any] = {
            "fixture_id": int(r["fixture_id"]),
            "league_id": int(r["league_id"]),
            "season": int(r["season"]),
            "kickoff_utc": r["kickoff_utc"],
            "m_home": float(mh),
            "m_away": float(ma),
        }

        # correct score grid
        for i in range(args.nmax+1):
            for j in range(args.nmax+1):
                out[f"cs_{i}_{j}"] = float(J[i, j])

        # totals
        tot = totals_from_joint(J, args.nmax)
        for L in totals_lines:
            thr = math.floor(L + 1e-9) + 1
            p_over = float(tot[thr:].sum())
            out[f"totals_over_{L}"] = p_over
            out[f"totals_under_{L}"] = float(1.0 - p_over)

        # handicap (home - away)
        D = np.zeros(2*args.nmax+1, float)  # from -nmax..+nmax
        offset = args.nmax
        for i in range(args.nmax+1):
            for j in range(args.nmax+1):
                D[i - j + offset] += J[i, j]

        for h in hcp_lines:
            if float(h).is_integer():
                h_int = int(h)
                p_win  = float(D[(h_int+1)+offset:].sum()) if (h_int+1+offset) < len(D) else 0.0
                p_push = float(D[h_int + offset]) if (-args.nmax <= h_int <= args.nmax) else 0.0
                p_lose = float(1.0 - p_win - p_push)
            else:
                thr = math.floor(h + 1e-9)
                start = (thr+1) + offset
                p_win  = float(D[start:].sum()) if start < len(D) else 0.0
                p_push = 0.0
                p_lose = float(1.0 - p_win)
            out[f"hcp_home_{h}_win"]  = p_win
            out[f"hcp_home_{h}_push"] = p_push
            out[f"hcp_home_{h}_lose"] = p_lose

        rows_out.append(out)

    preds = pd.DataFrame(rows_out)

    # -------- simple val metrics (totals brier/logloss)
    metrics = {"val": {}}
    try:
        mask = df_val["home_ft"].notna() & df_val["away_ft"].notna()
        if mask.any():
            ll=0.0; nll=0; bs=0.0; nbs=0
            for _, r in df_val.loc[mask].iterrows():
                mh, ma = _predict_mus(r)
                ph = pmf_side(mh, k_for(r,"H"), args.nmax)
                pa = pmf_side(ma, k_for(r,"A"), args.nmax)
                if abs(chosen_rho) < 1e-9:
                    tot = conv_totals(ph, pa, args.nmax)
                else:
                    J = copula_joint_from_marginals(ph, pa, rho=chosen_rho, sims=args.sims, seed=int(r["fixture_id"]), nmax=args.nmax)
                    tot = totals_from_joint(J, args.nmax)
                for L in totals_lines:
                    thr = math.floor(L + 1e-9) + 1
                    p_over = float(tot[thr:].sum())
                    y = int((int(r["home_ft"]) + int(r["away_ft"])) > math.floor(L + 1e-9))
                    p = min(max(p_over,1e-12), 1.0-1e-12)
                    ll += -(y*math.log(p)+(1-y)*math.log(1-p)); nll+=1
                    bs += (p - y)**2; nbs+=1
            if nll > 0: metrics["val"]["totals_logloss"] = ll/nll
            if nbs > 0: metrics["val"]["totals_brier"] = bs/nbs
    except Exception:
        pass

    # -------- Save artifacts + predictions
        # -------- Save artifacts + predictions
    artifacts = {
        "type": "cs_v1",
        "family": args.family,
        "rho": float(chosen_rho),
        "tau_raw": None if tau_raw is None else float(tau_raw),
        "nmax": int(args.nmax),
        "lines": {"totals": totals_lines, "handicap": hcp_lines},
        "dispersion": {
            "k_home_global": None if kH_global is None else float(kH_global),
            "k_away_global": None if kA_global is None else float(kA_global),
            "k_home_map": {f"{lg}-{ss}": v for (lg, ss), v in (kH_map.items() if kH_map else [])},
            "k_away_map": {f"{lg}-{ss}": v for (lg, ss), v in (kA_map.items() if kA_map else [])},
        },
        "mu_anchor_scale": {"home": float(s_h), "away": float(s_a)},
        "mean_floor": float(args.mean_floor),
        "training": {
            "leagues": leagues,
            "train_seasons": train_seasons,
            "val_seasons": val_seasons,
            "test_seasons": test_seasons,
        },
        "metrics": metrics,
    }

    os.makedirs(outdir, exist_ok=True)
    art_path = os.path.join(outdir, "artifacts.cs.json")
    with open(art_path, "w") as f:
        json.dump(_to_py(artifacts), f, indent=2)

    pr_path = os.path.join(outdir, "preds_test.cs.csv")
    preds.to_csv(pr_path, index=False)

    if not preds.empty:
        headJ = [c for c in preds.columns if c.startswith("cs_")][:10]
        print("[DBG] sample μ (head):", preds[["m_home","m_away"]].head().to_dict("records"))
        print("[DBG] sample CS cells:", headJ)

    print(f"[INFO] Saved artifacts to {art_path}")
    print(f"[INFO] Wrote test predictions to {pr_path}")
    print("[INFO] Done.")


    os.makedirs(outdir, exist_ok=True)
    art_path = os.path.join(outdir, "artifacts.cs.json")
    with open(art_path, "w") as f:
        json.dump(_to_py(artifacts), f, indent=2)

    pr_path = os.path.join(outdir, "preds_test.cs.csv")
    preds.to_csv(pr_path, index=False)

    if not preds.empty:
        headJ = [c for c in preds.columns if c.startswith("cs_")][:10]
        print("[DBG] sample μ (head):", preds[["m_home","m_away"]].head().to_dict("records"))
        print("[DBG] sample CS cells:", headJ)

    print(f"[INFO] Saved artifacts to {art_path}")
    print(f"[INFO] Wrote test predictions to {pr_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
