# prediction/matches/management/commands/train_corners.py
from __future__ import annotations

"""
Train a corners totals model (negative binomial) and write an artifact JSON.

Highlights
- Robust labels extraction from Match / MatchStats / stats JSON (finished matches only)
- Map goals μ -> corners μ using GLM (PoissonRegressor if available) or a tuned heuristic
- Dispersion (NB k) with league-season grouped shrinkage
- Optional Gaussian copula for correlation tuning (rho), or independent sums
- Optional isotonic calibration for totals & team totals (validation split only)
- Safe fallbacks everywhere + concise diagnostics

Usage
  export DJANGO_SETTINGS_MODULE=prediction.settings
  python manage.py train_corners \
    --leagues 39 \
    --train-seasons 2021-2023 \
    --val-seasons 2024 \
    --test-seasons 2025 \
    --outdir artifacts/corners \
    --goals-artifact artifacts/goals/artifacts.goals.json \
    --rho-grid auto \
    --write-test-csv
"""

import os, json, argparse, math, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
np.seterr(all="ignore")

from django.core.management.base import BaseCommand

# Django models (Django is already bootstrapped by manage.py)
from matches.models import Match, MatchStats
try:
    from matches.models import MLTrainingMatch  # type: ignore
    HAVE_MLTRAIN = True
except Exception:
    MLTrainingMatch = None  # type: ignore
    HAVE_MLTRAIN = False

# Optional libs
try:
    from sklearn.isotonic import IsotonicRegression
    HAVE_ISO = True
except Exception:
    IsotonicRegression = None
    HAVE_ISO = False

try:
    from sklearn.linear_model import PoissonRegressor
    HAVE_POISSON = True
except Exception:
    HAVE_POISSON = False

try:
    from scipy.special import gammaln
    from scipy.stats import kendalltau
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    gammaln = None
    kendalltau = None

try:
    from scipy.special import erf as _erf_u
    HAVE_ERF = True
except Exception:
    HAVE_ERF = False
    _erf_u = None


# ---------------- utils ----------------
def expand_seasons(arg: str) -> List[int]:
    out: List[int] = []
    for seg in str(arg).split(","):
        seg = seg.strip()
        if not seg: continue
        if "-" in seg:
            a, b = seg.split("-", 1)
            out.extend(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
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

def _to_py(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_py(v) for v in obj]
    return obj

def _phi_arr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if HAVE_ERF:
        return 0.5 * (1.0 + _erf_u(z / math.sqrt(2.0)))
    # fallback
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


# -------- goals feature builder (optional) --------
try:
    from prediction.train_goals import build_oriented_features  # type: ignore
except Exception:
    try:
        from train_goals import build_oriented_features  # type: ignore
    except Exception:
        build_oriented_features = None  # type: ignore

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


# -------- robust totals extraction --------
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _has_corner_word(s: str) -> bool:
    t = _norm(s)
    return ("corner" in t) or ("cornerkicks" in t)

def _int0_30(x) -> Optional[int]:
    try:
        v = int(x)
        if 0 <= v <= 30: return v
    except Exception:
        pass
    return None

_HOME_KEYS = ("home","hometeam","home_team","localteam","local_team")
_AWAY_KEYS = ("away","awayteam","away_team","visitorteam","visitor_team")

def _extract_from_statistics(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(obj, dict): return (None, None)
    stats = None
    for k, v in obj.items():
        if _norm(k) in ("statistics","matchstatistics","teamstatistics","stats","statistic"):
            if isinstance(v, list): stats = v; break
    if not isinstance(stats, list): return (None, None)
    h, a = None, None
    for item in stats:
        if not isinstance(item, dict): continue
        label = item.get("type") or item.get("name") or item.get("title") or item.get("label")
        if label and _has_corner_word(label):
            for hk in _HOME_KEYS + ("homescore","homevalue","home_value"):
                hv = _int0_30(item.get(hk))
                if hv is not None: h = hv; break
            for ak in _AWAY_KEYS + ("awayscore","awayvalue","away_value"):
                av = _int0_30(item.get(ak))
                if av is not None: a = av; break
            if (h is None or a is None) and isinstance(item.get("stats"), dict):
                hv = _int0_30(item["stats"].get("home"))
                av = _int0_30(item["stats"].get("away"))
                h = hv if hv is not None else h
                a = av if av is not None else a
            if h is not None or a is not None:
                return (h, a)
    # {"name":"Corner Kicks","value":{"home":5,"away":4}}
    for item in stats:
        if not isinstance(item, dict): continue
        if any(_has_corner_word(k) for k in item.keys()):
            val = item.get("value") or item.get("values") or item.get("stats")
            if isinstance(val, dict):
                hv = _int0_30(val.get("home"))
                av = _int0_30(val.get("away"))
                if hv is not None or av is not None: return (hv, av)
    return (None, None)

def _extract_from_home_away_blocks(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(obj, dict): return (None, None)
    def _get_block(d: Dict[str, Any], keys) -> Optional[Dict[str, Any]]:
        for k, v in d.items():
            if _norm(k) in [_norm(x) for x in keys] and isinstance(v, dict):
                return v
        return None
    home_block = _get_block(obj, _HOME_KEYS)
    away_block = _get_block(obj, _AWAY_KEYS)
    if not isinstance(home_block, dict) or not isinstance(away_block, dict):
        return (None, None)
    def _find_total(d: Dict[str, Any]) -> Optional[int]:
        for kk, vv in d.items():
            if _has_corner_word(kk):
                val = _int0_30(vv)
                if val is not None: return val
        for vv in d.values():
            if isinstance(vv, dict):
                for k2, v2 in vv.items():
                    if _has_corner_word(k2):
                        val = _int0_30(v2)
                        if val is not None: return val
        return None
    return _find_total(home_block), _find_total(away_block)

def _extract_any_pair(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(obj, dict):
        if any(_has_corner_word(k) for k in obj.keys()):
            hv = None; av = None
            for hk in _HOME_KEYS + ("home",):
                if hk in obj:
                    hv = _int0_30(obj.get(hk)); break
            for ak in _AWAY_KEYS + ("away",):
                if ak in obj:
                    av = _int0_30(obj.get(ak)); break
            if hv is not None or av is not None: return (hv, av)
            for v in obj.values():
                if isinstance(v, dict):
                    hv = _int0_30(v.get("home"))
                    av = _int0_30(v.get("away"))
                    if hv is not None or av is not None: return (hv, av)
        for v in obj.values():
            hh, aa = _extract_any_pair(v)
            if hh is not None or aa is not None: return (hh, aa)
    elif isinstance(obj, list):
        for v in obj:
            hh, aa = _extract_any_pair(v)
            if hh is not None or aa is not None: return (hh, aa)
    return (None, None)

def extract_corner_totals(stats_like: Any) -> Tuple[Optional[int], Optional[int]]:
    if stats_like is None: return (None, None)
    h, a = _extract_from_statistics(stats_like)
    if h is not None or a is not None: return (h, a)
    def _dfs_blocks(x: Any):
        if isinstance(x, dict):
            hh, aa = _extract_from_home_away_blocks(x)
            if hh is not None or aa is not None: return (hh, aa)
            for v in x.values():
                r = _dfs_blocks(v)
                if r != (None, None): return r
        elif isinstance(x, list):
            for v in x:
                r = _dfs_blocks(v)
                if r != (None, None): return r
        return (None, None)
    h, a = _dfs_blocks(stats_like)
    if h is not None or a is not None: return (h, a)
    return _extract_any_pair(stats_like)


# ---------------- data loading ----------------
_FINISHED_STATUSES = {"FT", "AET", "PEN", "AFTER_PEN"}
def _is_finished_status(s: Optional[str]) -> bool:
    return (s is not None) and (str(s).upper() in _FINISHED_STATUSES)

def load_rows(leagues: List[int], seasons: List[int]) -> pd.DataFrame:
    recs = []
    missing_both = 0

    if HAVE_MLTRAIN:
        base_iter = (MLTrainingMatch.objects
                     .filter(league_id__in=leagues, season__in=seasons)
                     .order_by("kickoff_utc")
                     .only("fixture_id","league_id","season","kickoff_utc","stats10_json")
                     .iterator())
        for r in base_iter:
            ch = None; ca = None; stats_exists = False
            finished = False; m = None
            try:
                m = Match.objects.only("id","home_id","away_id","status","corners_home","corners_away").get(pk=int(r.fixture_id))
                finished = _is_finished_status(getattr(m, "status", None))
            except Match.DoesNotExist:
                pass

            if finished and m:
                if m.corners_home is not None: ch = int(m.corners_home)
                if m.corners_away is not None: ca = int(m.corners_away)
                if ch is None or ca is None:
                    try:
                        stats = list(MatchStats.objects.filter(match_id=int(m.id)).values("team_id","corners"))
                        stats_exists = len(stats) > 0
                        if stats_exists:
                            by_team = {int(s["team_id"]): s["corners"] for s in stats if s["corners"] is not None}
                            if ch is None and m.home_id in by_team: ch = int(by_team[m.home_id])
                            if ca is None and m.away_id in by_team: ca = int(by_team[m.away_id])
                    except Exception:
                        pass
                if (ch is None or ca is None) and getattr(r, "stats10_json", None) is not None:
                    h2, a2 = extract_corner_totals(r.stats10_json)
                    if ch is None and h2 is not None: ch = int(h2)
                    if ca is None and a2 is not None: ca = int(a2)
                if (ch == 0 and ca == 0) and not stats_exists:
                    ch, ca = None, None
            else:
                ch, ca = None, None

            if ch is None and ca is None: missing_both += 1
            recs.append({
                "fixture_id": int(r.fixture_id),
                "league_id": int(r.league_id),
                "season": int(r.season),
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "stats10_json": getattr(r, "stats10_json", None),
                "corners_home": None if ch is None else int(ch),
                "corners_away": None if ca is None else int(ca),
            })
    else:
        base_iter = (Match.objects
                     .filter(league_id__in=leagues, season__in=seasons)
                     .order_by("kickoff_utc")
                     .only("id","league_id","season","kickoff_utc","status",
                           "home_id","away_id","corners_home","corners_away","raw_result_json")
                     .iterator())
        for m in base_iter:
            finished = _is_finished_status(getattr(m, "status", None))
            ch = None; ca = None; stats_exists = False
            if finished:
                if m.corners_home is not None: ch = int(m.corners_home)
                if m.corners_away is not None: ca = int(m.corners_away)
                if ch is None or ca is None:
                    try:
                        stats = list(MatchStats.objects.filter(match_id=int(m.id)).values("team_id","corners"))
                        stats_exists = len(stats) > 0
                        if stats_exists:
                            by_team = {int(s["team_id"]): s["corners"] for s in stats if s["corners"] is not None}
                            if ch is None and m.home_id in by_team: ch = int(by_team[m.home_id])
                            if ca is None and m.away_id in by_team: ca = int(by_team[m.away_id])
                    except Exception:
                        pass
                if (ch is None or ca is None) and (m.raw_result_json is not None):
                    h2, a2 = extract_corner_totals(m.raw_result_json)
                    if ch is None and h2 is not None: ch = int(h2)
                    if ca is None and a2 is not None: ca = int(a2)
                if (ch == 0 and ca == 0) and not stats_exists:
                    ch, ca = None, None
            else:
                ch, ca = None, None

            if ch is None and ca is None: missing_both += 1
            recs.append({
                "fixture_id": int(m.id),
                "league_id": int(m.league_id),
                "season": int(m.season),
                "kickoff_utc": m.kickoff_utc.isoformat(),
                "stats10_json": (m.raw_result_json or {}).get("stats10_json"),
                "corners_home": None if ch is None else int(ch),
                "corners_away": None if ca is None else int(ca),
            })

    df = pd.DataFrame(recs)
    if len(df) > 0 and missing_both:
        print(f"[WARN] No finished-match corner totals found for {missing_both}/{len(df)} rows.")
    return df


# ---------------- goals μ attachment ----------------
def attach_goal_mus(df: pd.DataFrame, art: Optional[GoalsArtifact]) -> pd.DataFrame:
    muH, muA, fails = [], [], 0
    for _, r in df.iterrows():
        if art is None or build_oriented_features is None:
            muH.append(np.nan); muA.append(np.nan); continue
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


# ---------------- mapping: goals μ -> corners μ ----------------
def _label_side_stats(df: pd.DataFrame, side: str, tag: str):
    col = f"corners_{side}"
    s = pd.to_numeric(df[col], errors="coerce")
    mask = s.notna()
    vals = s[mask].astype(float).to_numpy()
    if vals.size == 0:
        print(f"[STAT] {tag}.{side}: N=0 (no labels)")
        return
    print(f"[STAT] {tag}.{side}: N={vals.size}, mean={np.mean(vals):.3f}, std={np.std(vals, ddof=1) if vals.size>1 else 0:.3f}, "
          f"min={np.min(vals):.0f}, p50={np.median(vals):.1f}, max={np.max(vals):.0f}")

def estimate_nb_k(y: np.ndarray, clip_lo=5.0, clip_hi=1000.0) -> float:
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 50: return 150.0
    mu = float(y.mean())
    var = float(y.var(ddof=1)) if y.size > 1 else mu + mu**2/400.0
    if var <= mu + 1e-9: return clip_hi
    k = (mu * mu) / (var - mu)
    return float(np.clip(k, clip_lo, clip_hi))

def estimate_nb_k_grouped(df: pd.DataFrame, ycol: str, clip_lo=5.0, clip_hi=1000.0):
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

def nb_pmf_vec(mu: float, k: float, nmax: int) -> np.ndarray:
    mu = max(1e-9, float(mu)); k = max(1e-9, float(k))
    p = mu / (k + mu)
    if HAVE_SCIPY and gammaln is not None:
        out = np.empty(nmax+1, float)
        qk = k * math.log(1.0 - p)
        for y in range(nmax+1):
            logc = gammaln(y + k) - gammaln(k) - gammaln(y + 1)
            out[y] = math.exp(logc + qk + y * math.log(p))
        s = out.sum()
        if s > 0: out /= s
        return out
    # fallback stable recurrence
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
    if s > 0: out /= s
    return out

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

def tau_to_rho_gaussian(tau: float) -> float:
    return float(np.sin(np.pi * tau / 2.0))

def estimate_rho_from_labels(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if not HAVE_SCIPY or kendalltau is None:
        return None, None
    m = df["corners_home"].notna() & df["corners_away"].notna()
    if not m.any(): return None, None
    h = pd.to_numeric(df.loc[m, "corners_home"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df.loc[m, "corners_away"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(h) & np.isfinite(a)
    h, a = h[mask], a[mask]
    if len(h) < 200: return None, None
    tau, _ = kendalltau(h, a)
    if not np.isfinite(tau): return None, None
    rho0 = np.clip(tau_to_rho_gaussian(tau), -0.35, 0.35)
    return float(rho0), float(tau)

def fit_iso_curve(p: np.ndarray, y: np.ndarray) -> Optional[Dict[str, List[float]]]:
    if not HAVE_ISO: return None
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


# ---------------- Django command ----------------
class Command(BaseCommand):
    help = "Train a corners totals model and save artifacts."

    def add_arguments(self, parser):
        parser.add_argument("--leagues", type=int, nargs="+", required=True)
        parser.add_argument("--train-seasons", required=True)
        parser.add_argument("--val-seasons", required=True)
        parser.add_argument("--test-seasons", required=True)
        parser.add_argument("--outdir", required=True)

        parser.add_argument("--family", choices=["nb"], default="nb")
        parser.add_argument("--goals-artifact", default=None)

        parser.add_argument("--total-lines", default="7.5,8.5,9.5,10.5")
        parser.add_argument("--team-lines", default="3.5,4.5")
        parser.add_argument("--nmax", type=int, default=30)

        parser.add_argument("--force-glm", action="store_true")
        parser.add_argument("--skip-anchor", action="store_true")
        parser.add_argument("--mean-floor", type=float, default=0.8)

        parser.add_argument("--rho-grid", default="0.0", help="'auto' or numeric (e.g. 0.1)")
        parser.add_argument("--sims", type=int, default=20000)

        parser.add_argument("--k-min", type=float, default=5.0)
        parser.add_argument("--k-max", type=float, default=1000.0)

        parser.add_argument("--write-test-csv", action="store_true")

    def handle(self, *args, **opts):
        if opts["family"] != "nb":
            raise ValueError("Only 'nb' is supported.")

        random.seed(123); np.random.seed(123)

        leagues = list(opts["leagues"])
        train_seasons = expand_seasons(opts["train_seasons"])
        val_seasons   = expand_seasons(opts["val_seasons"])
        test_seasons  = expand_seasons(opts["test_seasons"])
        outdir = opts["outdir"]
        os.makedirs(outdir, exist_ok=True)

        totals_lines = parse_float_list(opts["total_lines"])
        team_lines   = parse_float_list(opts["team_lines"])
        nmax         = int(opts["nmax"])
        mean_floor   = float(opts["mean_floor"])
        sims         = int(opts["sims"])
        k_min        = float(opts["k_min"])
        k_max        = float(opts["k_max"])

        # Load rows
        self.stdout.write("[INFO] Loading rows (corners totals labels from finished matches)...")
        df_train = load_rows(leagues, train_seasons)
        df_val   = load_rows(leagues, val_seasons)
        df_test  = load_rows(leagues, test_seasons)

        for col in ("corners_home","corners_away"):
            df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
            df_val[col]   = pd.to_numeric(df_val[col], errors="coerce")
            df_test[col]  = pd.to_numeric(df_test[col], errors="coerce")

        # Attach goals μ (optional)
        art_goals = load_goals_artifact(opts.get("goals_artifact"))
        df_train = attach_goal_mus(df_train, art_goals)
        df_val   = attach_goal_mus(df_val,   art_goals)
        df_test  = attach_goal_mus(df_test,  art_goals)

        def _label_count(df):
            m = df["corners_home"].notna() & df["corners_away"].notna()
            zeros = int(((df["corners_home"]==0) & (df["corners_away"]==0)).sum())
            return int(m.sum()), len(df), zeros

        nlab_tr, nt_tr, z_tr = _label_count(df_train)
        nlab_va, nt_va, z_va = _label_count(df_val)
        nlab_te, nt_te, z_te = _label_count(df_test)
        self.stdout.write(f"[INFO] Labeled corner pairs: train={nlab_tr}/{nt_tr} (0+0={z_tr}), "
                          f"val={nlab_va}/{nt_va} (0+0={z_va}), test={nlab_te}/{nt_te} (0+0={z_te})")

        # Borrow val for fit if train has 0 labels
        df_fit = df_train
        borrowed_from_val = False
        if nlab_tr == 0 and nlab_va > 0:
            df_fit = pd.concat([df_train, df_val], ignore_index=True)
            borrowed_from_val = True
            self.stdout.write("[INFO] No labeled training rows — borrowing validation for fitting only.")

        # Quick label stats
        def _pair_eq_share(df):
            m = df["corners_home"].notna() & df["corners_away"].notna()
            return float((df.loc[m, "corners_home"] == df.loc[m, "corners_away"]).mean()) if m.any() else 0.0
        self.stdout.write(f"[STAT] H==A share: train={_pair_eq_share(df_train):.1%}, "
                          f"val={_pair_eq_share(df_val):.1%}, test={_pair_eq_share(df_test):.1%}")
        _label_side_stats(df_train, "home", "train")
        _label_side_stats(df_train, "away", "train")
        _label_side_stats(df_val,   "home", "val")
        _label_side_stats(df_val,   "away", "val")

        # ---- fit mapping (GLM or heuristic) ----
        def fit_corner_means(train: pd.DataFrame,
                             df_val: Optional[pd.DataFrame],
                             prior_home: float = 5.2,
                             prior_away: float = 4.8,
                             force_glm: bool = False,
                             skip_anchor: bool = False) -> Dict[str, Any]:
            eps = 1e-6
            ch = pd.to_numeric(train["corners_home"], errors="coerce").to_numpy(dtype=float)
            ca = pd.to_numeric(train["corners_away"], errors="coerce").to_numpy(dtype=float)
            msk_h = np.isfinite(ch); msk_a = np.isfinite(ca)
            mean_obs_h = float(np.nanmean(ch[msk_h])) if msk_h.any() else np.nan
            mean_obs_a = float(np.nanmean(ca[msk_a])) if msk_a.any() else np.nan
            labels_ok_home = (np.isfinite(mean_obs_h) and 1.0 <= mean_obs_h <= 20.0)
            labels_ok_away = (np.isfinite(mean_obs_a) and 1.0 <= mean_obs_a <= 20.0)
            labels_ok = labels_ok_home and labels_ok_away

            muH = pd.to_numeric(train["mu_goals_home"], errors="coerce").to_numpy(dtype=float)
            muA = pd.to_numeric(train["mu_goals_away"], errors="coerce").to_numpy(dtype=float)
            good_mu = np.isfinite(muH) & np.isfinite(muA)
            muH_med = float(np.nanmedian(muH[good_mu])) if good_mu.any() else np.nan
            muA_med = float(np.nanmedian(muA[good_mu])) if good_mu.any() else np.nan
            self.stdout.write(f"[DBG] goals μ medians: home≈{(muH_med if np.isfinite(muH_med) else float('nan')):.3f} "
                              f"away≈{(muA_med if np.isfinite(muA_med) else float('nan')):.3f}  "
                              f"(NaN share={(1.0 - good_mu.mean() if good_mu.size else 1.0):.2%})")
            mu_ok = (np.isfinite(muH_med) and np.isfinite(muA_med) and 0.3 <= muH_med <= 5.0 and 0.3 <= muA_med <= 5.0)

            # Heuristic mapping if labels or μ are bad (and not forced GLM)
            if (not labels_ok or not mu_ok) and not force_glm:
                mu_sum = pd.to_numeric(train["mu_goals_home"], errors="coerce").to_numpy(dtype=float) + \
                         pd.to_numeric(train["mu_goals_away"], errors="coerce").to_numpy(dtype=float)
                mu_bar = float(np.nanmedian(mu_sum[np.isfinite(mu_sum)])) / 2.0 if np.isfinite(mu_sum).any() else 1.4
                alpha_grid = [0.45, 0.55, 0.60, 0.65, 0.70, 0.75]
                beta_grid  = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
                best=(0.60,0.60); best_loss=float("inf")

                def _nb_k_safe(y):
                    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
                    y = y[np.isfinite(y)]
                    if len(y) < 50: return 400.0
                    mu = float(y.mean()); var = float(y.var(ddof=1)) if len(y) > 1 else mu + mu**2/400.0
                    if var <= mu + 1e-9: return 1000.0
                    k = (mu*mu) / (var - mu)
                    return float(np.clip(k, 50.0, 1000.0))

                kH = _nb_k_safe(train["corners_home"].values)
                kA = _nb_k_safe(train["corners_away"].values)

                def val_loss(alpha, beta):
                    if df_val is None or df_val.empty: return 0.0
                    mask = df_val["corners_home"].notna() & df_val["corners_away"].notna()
                    if not mask.any(): return 0.0
                    loss=0.0; n=0
                    for _, r in df_val.loc[mask].iterrows():
                        muH0, muA0 = float(r.get("mu_goals_home", np.nan)), float(r.get("mu_goals_away", np.nan))
                        denom = max(1e-6, mu_bar*(1.0+beta))
                        mH = max(0.8, prior_home * ((muH0 + beta*muA0)/denom)**alpha)
                        mA = max(0.8, prior_away * ((muA0 + beta*muH0)/denom)**alpha)
                        pmfH = nb_pmf_vec(mH, kH, 30); pmfA = nb_pmf_vec(mA, kA, 30)
                        tot = conv_sum(pmfH, pmfA, 30)
                        for L in totals_lines:
                            thr = math.floor(L+1e-9)+1
                            p_over = float(tot[thr:].sum())
                            y = int((int(r["corners_home"])+int(r["corners_away"])) > math.floor(L + 1e-9))
                            p = min(max(p_over,1e-12), 1.0-1e-12)
                            loss += -(y*math.log(p)+(1-y)*math.log(1-p)); n+=1
                    return loss/max(1,n)

                for a in alpha_grid:
                    for b in beta_grid:
                        L = val_loss(a,b)
                        if L < best_loss:
                            best_loss, best = L, (a,b)
                alpha, beta = best
                self.stdout.write(f"[WARN] Using heuristic mapping (α={alpha:.2f}, β={beta:.2f}, mu_bar={mu_bar:.3f})")
                return {"kind": "heuristic",
                        "params": {"alpha": alpha, "beta": beta, "mu_bar": mu_bar,
                                   "prior_home": 5.2, "prior_away": 4.8}}

            # GLM
            def _fit_side(y_col, own_mu_col, opp_mu_col, prior):
                y = pd.to_numeric(train[y_col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(y) & np.isfinite(pd.to_numeric(train[own_mu_col], errors="coerce")) \
                                 & np.isfinite(pd.to_numeric(train[opp_mu_col], errors="coerce"))
                if mask.sum() < 50:
                    return {"intercept": math.log(prior), "b1": 0.0, "b2": 0.0}, np.array([])
                y = y[mask]
                eps = 1e-6
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
                            trn = np.concatenate([folds[i] for i in range(3) if i!=kfold])
                            pr = PoissonRegressor(alpha=a, max_iter=5000)
                            pr.fit(X[trn], y[trn])
                            yhat = np.clip(np.exp(pr.intercept_ + X[val] @ pr.coef_), 1e-9, None)
                            loss = np.mean(yhat - y[val]*np.log(yhat+1e-12))
                            fold_losses.append(loss)
                        m = float(np.mean(fold_losses))
                        if m < best_loss:
                            best_loss, best_alpha = m, a
                    pr = PoissonRegressor(alpha=best_alpha, max_iter=5000)
                    pr.fit(X, y)
                    coef = {"intercept": float(pr.intercept_), "b1": float(pr.coef_[0]), "b2": float(pr.coef_[1]), "alpha": best_alpha}
                else:
                    z = np.log(np.clip(y, 1e-6, None))
                    Xls = np.hstack([np.ones((X.shape[0], 1)), X])
                    beta_hat, *_ = np.linalg.lstsq(Xls, z, rcond=None)
                    coef = {"intercept": float(beta_hat[0]), "b1": float(beta_hat[1]), "b2": float(beta_hat[2])}

                mu_hat = np.exp(coef["intercept"] + coef["b1"]*x1 + coef["b2"]*x2)
                return coef, mu_hat

            coef_home, mu_hat_h = _fit_side("corners_home", "mu_goals_home", "mu_goals_away", prior=5.2)
            coef_away, mu_hat_a = _fit_side("corners_away", "mu_goals_away", "mu_goals_home", prior=4.8)

            def _scale(obs_mean, mu_hat, tag):
                if opts["skip_anchor"] or mu_hat.size == 0 or not np.isfinite(obs_mean):
                    s = 1.0; why = "skip"
                else:
                    pred_bar = float(mu_hat.mean())
                    s = 1.0 if pred_bar <= 1e-9 else float(np.clip(obs_mean / pred_bar, 0.5, 2.5))
                    why = "fit" if pred_bar > 0 else "pred≈0"
                self.stdout.write(f"[DBG] anchor {tag}: obs≈{(obs_mean if np.isfinite(obs_mean) else float('nan')):.2f}, "
                                  f"pred_bar≈{(mu_hat.mean() if mu_hat.size else float('nan')):.2f}, scale={s:.3f} ({why})")
                return s

            mean_obs_h = float(pd.to_numeric(df_train["corners_home"], errors="coerce").mean())
            mean_obs_a = float(pd.to_numeric(df_train["corners_away"], errors="coerce").mean())
            s_home = _scale(mean_obs_h, mu_hat_h, "home")
            s_away = _scale(mean_obs_a, mu_hat_a, "away")

            return {"kind": "glm",
                    "home": coef_home, "away": coef_away,
                    "scale": {"home": s_home, "away": s_away},
                    "fallback_means": {"home": 5.2, "away": 4.8}}

        mapping = fit_corner_means(df_fit, df_val,
                                   prior_home=5.2, prior_away=4.8,
                                   force_glm=bool(opts["force_glm"]),
                                   skip_anchor=bool(opts["skip_anchor"]))

        # ---- dispersion ----
        kH_global, kH_map = estimate_nb_k_grouped(df_fit, "corners_home", clip_lo=k_min, clip_hi=k_max)
        kA_global, kA_map = estimate_nb_k_grouped(df_fit, "corners_away", clip_lo=k_min, clip_hi=k_max)
        self.stdout.write(f"[INFO] NB k (shrunk): home_global={kH_global:.1f} groups={len(kH_map)}; "
                          f"away_global={kA_global:.1f} groups={len(kA_map)}")

        def k_for(row, side):
            key = (int(row["league_id"]), int(row["season"]))
            return kH_map.get(key, kH_global) if side=="H" else kA_map.get(key, kA_global)

        def predict_means(row: pd.Series) -> Tuple[float, float]:
            eps = 1e-6
            def _clip(x): return float(np.clip(x, mean_floor, 15.0))
            muH = float(row.get("mu_goals_home", np.nan))
            muA = float(row.get("mu_goals_away", np.nan))
            kind = mapping.get("kind", "heuristic")
            if kind == "glm":
                ch = mapping["home"]; sh = float(mapping["scale"]["home"])
                ca = mapping["away"]; sa = float(mapping["scale"]["away"])
                zH = ch["intercept"] + ch["b1"]*math.log(max(eps, muH)) + ch["b2"]*math.log(max(eps, muA))
                zA = ca["intercept"] + ca["b1"]*math.log(max(eps, muA)) + ca["b2"]*math.log(max(eps, muH))
                return _clip(math.exp(zH)*sh), _clip(math.exp(zA)*sa)
            if kind == "heuristic":
                p = mapping["params"]
                alpha = float(p["alpha"]); beta = float(p["beta"]); mu_bar = float(p["mu_bar"])
                baseH = float(p["prior_home"]); baseA = float(p["prior_away"])
                denom = max(eps, mu_bar * (1.0 + beta))
                mH = _clip(baseH * ((muH + beta * muA) / denom) ** alpha)
                mA = _clip(baseA * ((muA + beta * muH) / denom) ** alpha)
                return mH, mA
            baseH = float(mapping.get("fallback_means", {}).get("home", 5.2))
            baseA = float(mapping.get("fallback_means", {}).get("away", 4.8))
            return _clip(baseH), _clip(baseA)

        # ---- rho tuning ----
        def _val_totals_logloss(rho: float) -> float:
            if df_val.empty: return 1e9
            mask = df_val["corners_home"].notna() & df_val["corners_away"].notna()
            if not mask.any(): return 1e9
            eps = 1e-12; loss=0.0; n=0
            for _, r in df_val.loc[mask].iterrows():
                mH, mA = predict_means(r)
                pmfH = nb_pmf_vec(mH, k_for(r,"H"), nmax)
                pmfA = nb_pmf_vec(mA, k_for(r,"A"), nmax)
                tot = conv_sum(pmfH, pmfA, nmax) if abs(rho) < 1e-9 \
                    else totals_pmf_copula(pmfH, pmfA, rho=rho, sims=sims, seed=int(r["fixture_id"]), nmax=nmax)[0]
                line_losses=[]
                for L in totals_lines:
                    thr = math.floor(L+1e-9)+1
                    p_over = float(tot[thr:].sum())
                    y = int((int(r["corners_home"])+int(r["corners_away"])) > math.floor(L + 1e-9))
                    p = min(max(p_over,eps), 1.0-eps)
                    line_losses.append(-(y*math.log(p)+(1-y)*math.log(1-p)))
                loss += float(np.mean(line_losses)); n+=1
            return loss/max(1,n)

        rho_grid_opt = str(opts["rho_grid"]).lower()
        chosen_rho = 0.0; tau_raw = None
        if rho_grid_opt == "auto":
            rho0, tau_raw = estimate_rho_from_labels(df_fit)
            if rho0 is not None:
                base = np.array([-0.10,-0.05,0.0,0.05,0.10])
                grid = list(np.clip(rho0 + base, -0.35, 0.35)) + [0.0]
            else:
                grid = [-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25]
            scores = [(r, _val_totals_logloss(r)) for r in grid]
            chosen_rho, best = min(scores, key=lambda t: t[1])
            self.stdout.write(f"[INFO] Chosen rho={chosen_rho:.3f} (val totals logloss={best:.4f})"
                              f"{'' if tau_raw is None else f', tau≈{tau_raw:.3f}'}")
        else:
            try:
                chosen_rho = float(opts["rho_grid"])
            except Exception:
                chosen_rho = 0.0
            if abs(chosen_rho) < 1e-9:
                self.stdout.write("[INFO] Using independent marginals (rho=0.0)")

        # ---- calibration (validation only) ----
        cal_totals: Dict[str, Dict[str, List[float]]] = {}
        cal_team_home: Dict[str, Dict[str, List[float]]] = {}
        cal_team_away: Dict[str, Dict[str, List[float]]] = {}

        mask_val = df_val["corners_home"].notna() & df_val["corners_away"].notna()
        if mask_val.any() and HAVE_ISO:
            preds_totals, preds_team = [], []
            for _, r in df_val.loc[mask_val].iterrows():
                mH, mA = predict_means(r)
                pmfH = nb_pmf_vec(mH, k_for(r,"H"), nmax)
                pmfA = nb_pmf_vec(mA, k_for(r,"A"), nmax)
                tot = conv_sum(pmfH, pmfA, nmax) if abs(chosen_rho) < 1e-9 \
                    else totals_pmf_copula(pmfH, pmfA, rho=chosen_rho, sims=sims, seed=int(r["fixture_id"]), nmax=nmax)[0]
                rowT = {"corners_home": int(r["corners_home"]), "corners_away": int(r["corners_away"])}
                for L in totals_lines:
                    thr = math.floor(L+1e-9)+1
                    rowT[f"p_over_{L}"] = float(tot[thr:].sum())
                preds_totals.append(rowT)

                cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)
                rowTeam = {"corners_home": int(r["corners_home"]), "corners_away": int(r["corners_away"])}
                for L in team_lines:
                    thr = math.floor(L+1e-9)+1
                    idxH = min(thr-1, nmax); idxA = min(thr-1, nmax)
                    rowTeam[f"pH_over_{L}"] = float(1.0 - cdfH[idxH])
                    rowTeam[f"pA_over_{L}"] = float(1.0 - cdfA[idxA])
                preds_team.append(rowTeam)

            dfp = pd.DataFrame(preds_totals)
            for L in totals_lines:
                y = (dfp["corners_home"].values + dfp["corners_away"].values > math.floor(L + 1e-9)).astype(int)
                p = dfp[f"p_over_{L}"].values.astype(float)
                p = p + (np.arange(p.size) % 997) * 1e-12
                curve = fit_iso_curve(p, y)
                if curve: cal_totals[str(L)] = curve

            dft = pd.DataFrame(preds_team)
            for L in team_lines:
                # home
                yH = (dft["corners_home"].values > math.floor(L+1e-9)).astype(int)
                pH = dft[f"pH_over_{L}"].values.astype(float)
                pH = pH + (np.arange(pH.size) % 997) * 1e-12
                curveH = fit_iso_curve(pH, yH)
                if curveH: cal_team_home[str(L)] = curveH
                # away
                yA = (dft["corners_away"].values > math.floor(L+1e-9)).astype(int)
                pA = dft[f"pA_over_{L}"].values.astype(float)
                pA = pA + (np.arange(pA.size) % 997) * 1e-12
                curveA = fit_iso_curve(pA, yA)
                if curveA: cal_team_away[str(L)] = curveA
            if cal_totals:
                self.stdout.write(f"[INFO] Totals calibration learned: {sorted(cal_totals.keys())}")
            if cal_team_home or cal_team_away:
                self.stdout.write(f"[INFO] Team calibration learned "
                                  f"(home={sorted(cal_team_home.keys())}, away={sorted(cal_team_away.keys())})")
        else:
            self.stdout.write("[WARN] Skipping calibration (no labeled validation rows or isotonic unavailable).")

        # ---- predictions on TEST (for sanity / optional CSV) ----
        def predict_means_for_row(r: pd.Series) -> Tuple[float, float]:
            # same as earlier small closure, duplicated for clarity
            eps = 1e-6
            def _clip(x): return float(np.clip(x, mean_floor, 15.0))
            muH = float(r.get("mu_goals_home", np.nan))
            muA = float(r.get("mu_goals_away", np.nan))
            kind = mapping.get("kind", "heuristic")
            if kind == "glm":
                ch = mapping["home"]; sh = float(mapping["scale"]["home"])
                ca = mapping["away"]; sa = float(mapping["scale"]["away"])
                zH = ch["intercept"] + ch["b1"]*math.log(max(eps, muH)) + ch["b2"]*math.log(max(eps, muA))
                zA = ca["intercept"] + ca["b1"]*math.log(max(eps, muA)) + ca["b2"]*math.log(max(eps, muH))
                return _clip(math.exp(zH)*sh), _clip(math.exp(zA)*sa)
            if kind == "heuristic":
                p = mapping["params"]
                alpha = float(p["alpha"]); beta = float(p["beta"]); mu_bar = float(p["mu_bar"])
                baseH = float(p["prior_home"]); baseA = float(p["prior_away"])
                denom = max(eps, mu_bar*(1.0+beta))
                mH = _clip(baseH * ((muH + beta*muA)/denom)**alpha)
                mA = _clip(baseA * ((muA + beta*muH)/denom)**alpha)
                return mH, mA
            baseH = float(mapping.get("fallback_means", {}).get("home", 5.2))
            baseA = float(mapping.get("fallback_means", {}).get("away", 4.8))
            return _clip(baseH), _clip(baseA)

        rows_out = []
        for _, r in df_test.iterrows():
            mH, mA = predict_means_for_row(r)
            pmfH = nb_pmf_vec(mH, kH_map.get((int(r["league_id"]),int(r["season"])), kH_global), nmax)
            pmfA = nb_pmf_vec(mA, kA_map.get((int(r["league_id"]),int(r["season"])), kA_global), nmax)
            tot = conv_sum(pmfH, pmfA, nmax) if abs(chosen_rho) < 1e-9 \
                else totals_pmf_copula(pmfH, pmfA, rho=chosen_rho, sims=sims, seed=int(r["fixture_id"]), nmax=nmax)[0]

            out = {
                "fixture_id": int(r["fixture_id"]),
                "league_id": int(r["league_id"]),
                "season": int(r["season"]),
                "kickoff_utc": r["kickoff_utc"],
                "m_corners_home": float(mH),
                "m_corners_away": float(mA),
            }

            for L in totals_lines:
                thr = math.floor(L+1e-9)+1
                p_over = float(tot[thr:].sum())
                if str(L) in {}: pass  # placeholder
                out[f"totals_over_{L}"]  = p_over
                out[f"totals_under_{L}"] = float(1.0 - p_over)

            cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)
            for L in team_lines:
                thr = math.floor(L+1e-9)+1
                idxH = min(thr-1, nmax); idxA = min(thr-1, nmax)
                pH_over = float(1.0 - cdfH[idxH])
                pA_over = float(1.0 - cdfA[idxA])
                out[f"home_over_{L}"] = pH_over
                out[f"home_under_{L}"] = float(1.0 - pH_over)
                out[f"away_over_{L}"] = pA_over
                out[f"away_under_{L}"] = float(1.0 - pA_over)

            rows_out.append(out)

        preds = pd.DataFrame(rows_out)

        # ---- metrics (val) quick
        metrics = {"val": {}}
        def _compute_val_metrics():
            mask = df_val["corners_home"].notna() & df_val["corners_away"].notna()
            if not mask.any(): return
            ll=0.0; nll=0
            bs=0.0; nbs=0
            for _, r in df_val.loc[mask].iterrows():
                mH, mA = predict_means(r)
                pmfH = nb_pmf_vec(mH, k_for(r,"H"), nmax)
                pmfA = nb_pmf_vec(mA, k_for(r,"A"), nmax)
                tot = conv_sum(pmfH, pmfA, nmax) if abs(chosen_rho)<1e-9 \
                    else totals_pmf_copula(pmfH, pmfA, rho=chosen_rho, sims=sims, seed=int(r["fixture_id"]), nmax=nmax)[0]
                for L in totals_lines:
                    thr = math.floor(L+1e-9)+1
                    p_over = float(tot[thr:].sum())
                    y = int((int(r["corners_home"])+int(r["corners_away"])) > math.floor(L+1e-9))
                    p = min(max(p_over,1e-12), 1.0-1e-12)
                    ll += -(y*math.log(p)+(1-y)*math.log(1-p)); nll+=1
                    bs += (p - y)**2; nbs+=1
            if nll>0: metrics["val"]["totals_logloss"] = ll/nll
            if nbs>0: metrics["val"]["totals_brier"] = bs/nbs

        _compute_val_metrics()

        # ---- save artifacts (+ optional test CSV)
        artifacts = {
            "type": "corners_nb_totals_v1",
            "family": "nb",
            "mapping": mapping,
            "mapping_kind": mapping.get("kind", "heuristic"),
            "dispersion": {
                "k_home_global": kH_global,
                "k_away_global": kA_global,
                "k_home_map": {f"{lg}-{ss}": v for (lg,ss), v in kH_map.items()},
                "k_away_map": {f"{lg}-{ss}": v for (lg,ss), v in kA_map.items()},
            },
            "lines": {"totals": totals_lines, "team": team_lines},
            "nmax": nmax,
            "rho": float(chosen_rho),
            "tau_raw": None if tau_raw is None else float(tau_raw),
            "totals_calibration": {},   # filled in below if we learned
            "team_calibration": {"home": {}, "away": {}},
            "mean_floor": float(mean_floor),
            "training": {
                "leagues": leagues,
                "train_seasons": train_seasons,
                "val_seasons": val_seasons,
                "test_seasons": test_seasons,
                "borrowed_val_for_fit": borrowed_from_val,
            },
            "metrics": metrics,
            "notes": "Finished-only labels; robust extractor; grouped NB dispersion with shrinkage.",
        }

        # put calibration (if available)
        if HAVE_ISO and 'cal_totals' in locals():
            artifacts["totals_calibration"] = _to_py(cal_totals)
            artifacts["team_calibration"] = {"home": _to_py(cal_team_home), "away": _to_py(cal_team_away)}

        art_path = os.path.join(outdir, "artifacts.corners.json")
        with open(art_path, "w") as f:
            json.dump(_to_py(artifacts), f, indent=2)
        self.stdout.write(f"[INFO] Saved artifacts to {art_path}")

        if bool(opts.get("write_test_csv")) and not preds.empty:
            pr_path = os.path.join(outdir, "preds_test.corners.csv")
            preds.to_csv(pr_path, index=False)
            self.stdout.write(f"[INFO] Wrote test predictions to {pr_path}")
