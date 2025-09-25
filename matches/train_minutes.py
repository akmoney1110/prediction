#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minute model trainer (flexible & fast)

What you get:
- First-goal CDF from a smoothed baseline hazard (1..TMAX) with optional 2H bump
- Match-specific CDF via BP-aligned survival scaling + mu-shape beta
- Time-varying home-share p_H(t) via small logistic (features: log_mu_ratio, mu_tot, half)
- Closed-form Last-to-Score (no simulation)
- Isotonic calibration for the CDF and FTS(home)

Key flex knobs (CLI):
  --tmax 90
  --bands "1-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90"
  --ma-window 5
  --second-half-start 46
  --goal-any-prior 0.92 (fallback when data is sparse)
  --beta-grid "-0.6:0.6:0.06"
  --use-2h-bump

Usage:
  export DJANGO_SETTINGS_MODULE=prediction.settings
  python -m matches.train_minutes \
    --goals-artifact artifacts/goals/artifacts.goals.json \
    --leagues 39 \
    --train-seasons 2021-2023 \
    --val-seasons 2024 \
    --test-seasons 2025 \
    --outdir artifacts/minutes \
    --tmax 90 \
    --bands "1-15,16-30,31-45,46-60,61-75,76-90" \
    --use-2h-bump \
    --prompt
"""

import os, json, argparse, math, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
np.seterr(all="ignore")

# --- Django setup ---
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    raise RuntimeError("DJANGO_SETTINGS_MODULE not set. e.g., export DJANGO_SETTINGS_MODULE=prediction.settings")
import django
django.setup()

from matches.models import MLTrainingMatch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

# import feature builder + BP grid
try:
    from prediction.train_goals import build_oriented_features, bp_grid_pmf  # type: ignore
except Exception:
    from train_goals import build_oriented_features, bp_grid_pmf  # type: ignore


# ----------------------------- utils -----------------------------
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

def parse_grid(arg: str, default: List[float]) -> List[float]:
    s = str(arg).strip()
    if not s:
        return default
    if ":" in s:
        a, b, step = s.split(":")
        a, b, step = float(a), float(b), float(step)
        n = int(round((b - a) / step)) + 1
        return [a + i * step for i in range(max(n, 1))]
    return [float(x) for x in s.split(",") if x.strip()]

def parse_bands(arg: str, tmax: int) -> List[Tuple[int,int]]:
    """
    arg: "1-10,11-20,21-30,...,81-90"
    Returns a validated list of inclusive (L,R) with 1 <= L <= R <= tmax
    """
    raw = [seg.strip() for seg in str(arg).split(",") if seg.strip()]
    out: List[Tuple[int,int]] = []
    for seg in raw:
        if "-" not in seg:
            raise ValueError(f"Bad band segment: '{seg}' (use 'L-R')")
        l, r = seg.split("-", 1)
        L, R = int(l), int(r)
        if not (1 <= L <= R <= tmax):
            raise ValueError(f"Band {L}-{R} outside 1..{tmax}")
        out.append((L, R))
    # sanity: disjoint & ordered
    for i in range(1, len(out)):
        if out[i][0] <= out[i-1][1]:
            raise ValueError("Bands must be strictly increasing & non-overlapping")
    return out

@dataclass
class GoalsArtifact:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    max_goals: int
    c_bp: float

def load_goals_artifact(path: str) -> GoalsArtifact:
    with open(path, "r") as f:
        art = json.load(f)
    return GoalsArtifact(
        mean=np.array(art["scaler_mean"], float),
        scale=np.array(art["scaler_scale"], float),
        coef=np.array(art["poisson_coef"], float),
        intercept=float(art["poisson_intercept"]),
        max_goals=int(art["max_goals"]),
        c_bp=float(art.get("bp_c", 0.0)),
    )

def mu_from_features(x: np.ndarray, art: GoalsArtifact) -> float:
    xs = (x - art.mean) / art.scale
    return float(np.exp(art.intercept + xs.dot(art.coef)))


# ----------------------------- data -----------------------------
def _first_last_goal_from_minutes(mj: Dict[str, List[int]], tmax: int) -> Tuple[Optional[int], Optional[str],
                                                                                Optional[int], Optional[str]]:
    """Extract first/last goal minute & side; clamp to tmax."""
    if not mj:
        return None, None, None, None

    def clean(v):
        mm = []
        for x in (v or []):
            try:
                xi = int(x)
            except Exception:
                continue
            if 1 <= xi <= 120:
                mm.append(min(xi, tmax))
        return sorted(mm)

    H = clean(mj.get("goal_minutes_home", []))
    A = clean(mj.get("goal_minutes_away", []))
    ev = [(m, "H") for m in H] + [(m, "A") for m in A]
    if not ev:
        return None, None, None, None
    ev.sort(key=lambda t: (t[0], 0 if t[1] == "H" else 1))
    return ev[0][0], ev[0][1], ev[-1][0], ev[-1][1]

def load_rows(leagues: List[int], seasons: List[int], tmax: int) -> pd.DataFrame:
    qs = (MLTrainingMatch.objects
          .filter(league_id__in=leagues, season__in=seasons)
          .order_by("kickoff_utc")
          .only("fixture_id","league_id","season","kickoff_utc","stats10_json","minute_labels_json"))
    recs = []
    for r in qs.iterator():
        mj = r.minute_labels_json or {}
        fmin, fside, lmin, lside = _first_last_goal_from_minutes(mj, tmax)
        recs.append({
            "fixture_id": r.fixture_id,
            "league_id": r.league_id,
            "season": r.season,
            "kickoff_utc": r.kickoff_utc.isoformat(),
            "stats10_json": r.stats10_json,
            "minute_labels_json": mj,
            "first_min": fmin,
            "first_side": fside,  # "H"/"A" or None
            "last_min": lmin,
            "last_side": lside,   # "H"/"A" or None
        })
    return pd.DataFrame(recs)

def build_mu(df: pd.DataFrame, art: GoalsArtifact) -> pd.DataFrame:
    muH, muA = [], []
    for _, r in df.iterrows():
        xh, xa, _ = build_oriented_features({"stats10_json": r["stats10_json"]})
        muH.append(mu_from_features(xh, art))
        muA.append(mu_from_features(xa, art))
    df = df.copy()
    df["mu_home"] = muH
    df["mu_away"] = muA
    df["mu_tot"] = df["mu_home"] + df["mu_away"]
    eps = 1e-6
    df["log_mu_ratio"] = np.log((df["mu_home"] + eps) / (df["mu_away"] + eps))
    return df


# ----------------------------- first-goal CDF -----------------------------
def _estimate_hazard_from_df(df: pd.DataFrame, tmax: int, ma_window: int) -> np.ndarray:
    """Unsmoothed discrete hazard h[1..tmax] from first_min, then moving average + clips."""
    n = len(df)
    h = np.zeros(tmax + 1, dtype=float)
    survivors = n
    counts = df["first_min"].value_counts(dropna=False).to_dict()
    for t in range(1, tmax + 1):
        at_risk = max(survivors, 1e-9)
        e_t = float(counts.get(t, 0.0))
        h[t] = e_t / at_risk
        survivors -= e_t
    # Moving average
    w = max(1, int(ma_window))
    hs = h.copy()
    for t in range(1, tmax + 1):
        lo, hi = max(1, t - (w // 2)), min(tmax, t + (w // 2))
        hs[t] = float(h[lo:hi+1].mean())
    # Clips
    hs[hs < 1e-5] = 1e-5
    hs[hs > 0.25] = 0.25
    return hs

def estimate_baseline_hazard_first(train_df: pd.DataFrame,
                                   fallback_df: Optional[pd.DataFrame],
                                   tmax: int,
                                   ma_window: int,
                                   min_events: int,
                                   goal_any_prior: float) -> Tuple[np.ndarray, int]:
    train_events = int(train_df["first_min"].notna().sum())
    if train_events >= min_events:
        return _estimate_hazard_from_df(train_df, tmax, ma_window), train_events
    if fallback_df is not None and int(fallback_df["first_min"].notna().sum()) > 0:
        print(f"[WARN] Train first-goal events too few ({train_events}); using fallback dataset for baseline hazard.")
        return _estimate_hazard_from_df(fallback_df, tmax, ma_window), int(fallback_df["first_min"].notna().sum())
    print(f"[WARN] No usable first-goal events; using uniform fallback hazard.")
    lam = -math.log(max(1e-9, 1 - goal_any_prior)) / max(1, tmax)
    h = np.zeros(tmax + 1, float); h[1:] = lam
    return h, 0

def survival_from_h(h: np.ndarray) -> np.ndarray:
    tmax = h.shape[0] - 1
    S = np.ones(tmax + 1, float)
    for t in range(1, tmax + 1):
        S[t] = S[t-1] * (1.0 - h[t])
    return S

def band_probs_from_cdf(F: np.ndarray, bands: List[Tuple[int,int]]) -> Dict[str, float]:
    pm = {}
    for (L, R) in bands:
        pm[f"{L:02d}-{R:02d}"] = float(F[R] - F[L - 1])
    pm["no_goal"] = float(1.0 - F[-1])
    s = sum(pm.values())
    if s > 0:
        for k in pm:
            pm[k] /= s
    return pm

def hazard_from_cdf(F: np.ndarray) -> np.ndarray:
    tmax = F.shape[0] - 1
    h = np.zeros(tmax + 1, float)
    for t in range(1, tmax + 1):
        num = max(float(F[t] - F[t - 1]), 0.0)
        den = max(1.0 - float(F[t - 1]), 1e-9)
        h[t] = num / den
    return h

def bp_no_goal_prob(mu_h: float, mu_a: float, c_bp: float, G: int = 10) -> float:
    lam12 = c_bp * min(mu_h, mu_a)
    l1 = max(1e-12, mu_h - lam12)
    l2 = max(1e-12, mu_a - lam12)
    grid = bp_grid_pmf(l1, l2, lam12, G)
    return float(grid[0, 0])

def match_gamma_from_targets(S_base_T: float, S_target_T: float) -> float:
    S_base_T = float(np.clip(S_base_T, 1e-12, 1.0 - 1e-12))
    S_target_T = float(np.clip(S_target_T, 1e-12, 1.0 - 1e-12))
    return max(1e-6, math.log(S_target_T) / math.log(S_base_T))

def cdf_from_base_and_gamma(S_base: np.ndarray, gamma: float) -> np.ndarray:
    S = np.clip(S_base, 1e-9, 1.0)
    return 1.0 - np.power(S, float(gamma))

def apply_2h_bump(h: np.ndarray, theta: float, second_half_start: int) -> np.ndarray:
    tmax = h.shape[0] - 1
    s2 = max(1, min(tmax + 1, int(second_half_start)))
    h2 = h.copy()
    h2[s2:tmax+1] = np.clip(h2[s2:tmax+1] * math.exp(theta), 1e-6, 0.8)
    return h2

def tune_theta_2h(df_train: pd.DataFrame, h_base: np.ndarray, second_half_start: int) -> float:
    tmax = h_base.shape[0] - 1
    grid = np.linspace(-0.35, 0.35, 29)
    best = (1e99, 0.0)
    for th in grid:
        h_b = apply_2h_bump(h_base, th, second_half_start)
        S_b = survival_from_h(h_b)
        nll = 0.0
        for _, r in df_train.iterrows():
            fm = r["first_min"]
            if pd.isna(fm):
                p = max(1.0 - float(S_b[tmax]), 1e-12)
            else:
                t = int(fm)
                p = max(float((1.0 - S_b[t-1]) * h_b[t]), 1e-12)
            nll -= math.log(p)
        if nll < best[0]:
            best = (nll, float(th))
    return best[1]

def tune_beta_mu(df_train_with_mu: pd.DataFrame, S_base: np.ndarray,
                 art: GoalsArtifact, beta_grid: List[float]) -> Tuple[float, float]:
    df = df_train_with_mu.copy()
    df["first_min"] = pd.to_numeric(df["first_min"], errors="coerce").astype("Int64")
    mu_bar = float(np.median(df["mu_tot"].values))
    best = (1e99, 0.0)
    S_T = float(S_base[-1])
    # precompute S_target once per row
    S_targets = df.apply(
        lambda r: bp_no_goal_prob(float(r["mu_home"]), float(r["mu_away"]), art.c_bp, art.max_goals), axis=1
    ).values
    for beta in beta_grid:
        nll = 0.0
        for (idx, r) in df.iterrows():
            S_t = float(S_targets[idx])
            gamma0 = match_gamma_from_targets(S_T, S_t)
            gamma = gamma0 * ((float(r["mu_tot"]) / mu_bar) ** beta)
            F = cdf_from_base_and_gamma(S_base, gamma)
            fm = r["first_min"]
            if pd.isna(fm):
                p = max(1.0 - float(F[-1]), 1e-12)
            else:
                t = int(fm)
                p = max(float(F[t] - F[t-1]), 1e-12)
            nll -= math.log(p)
        if nll < best[0]:
            best = (nll, float(beta))
    return best[1], mu_bar


# ----------------------------- scorer share p_H(t) -----------------------------
def build_goal_events_for_share(df_with_mu: pd.DataFrame, tmax: int, second_half_start: int) -> pd.DataFrame:
    rows = []
    for _, r in df_with_mu.iterrows():
        mj = r["minute_labels_json"] or {}
        H = [int(x) for x in (mj.get("goal_minutes_home") or []) if 1 <= int(x) <= tmax]
        A = [int(x) for x in (mj.get("goal_minutes_away") or []) if 1 <= int(x) <= tmax]
        for m in H:
            rows.append({
                "log_mu_ratio": float(r["log_mu_ratio"]),
                "mu_tot": float(r["mu_tot"]),
                "is_2h": 1 if m >= second_half_start else 0,
                "y_home": 1
            })
        for m in A:
            rows.append({
                "log_mu_ratio": float(r["log_mu_ratio"]),
                "mu_tot": float(r["mu_tot"]),
                "is_2h": 1 if m >= second_half_start else 0,
                "y_home": 0
            })
    return pd.DataFrame(rows)

class TinyLogit:
    """Sigmoid(intercept + coef · x)."""
    def __init__(self, intercept: float, coefs: List[float]):
        self.intercept_ = np.array([intercept], dtype=float)
        self.coef_ = np.array([coefs], dtype=float)  # shape (1, k)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.intercept_[0] + np.dot(X, self.coef_[0].T)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.vstack([1.0 - p, p]).T

def fit_share_logit_by_half(train_df_mu: pd.DataFrame, tmax: int, second_half_start: int):
    """p_H(t) = sigmoid(b0 + b1*log_mu_ratio + b2*mu_tot + b3*is_2h)."""
    ev = build_goal_events_for_share(train_df_mu, tmax, second_half_start)
    if ev.empty:
        return TinyLogit(0.0, [0.0, 0.0, 0.0])
    X = ev[["log_mu_ratio","mu_tot","is_2h"]].values.astype(float)
    y = ev["y_home"].astype(int).values
    n_pos = int(y.sum()); n = int(len(y)); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        p = (n_pos + 1.0) / (n + 2.0)
        return TinyLogit(float(np.log(p / max(1e-9, 1 - p))), [0.0, 0.0, 0.0])
    clf = LogisticRegression(C=2.0, solver="lbfgs", class_weight="balanced", max_iter=300)
    clf.fit(X, y)
    return clf

def p_home_minute(clf, log_mu_ratio: float, mu_tot: float, second_half_start: int, tmax: int) -> Tuple[float, float]:
    """Return (p_H_1H, p_H_2H) for reporting; per-minute vector is built downstream."""
    X1 = np.array([[log_mu_ratio, mu_tot, 0.0]], float)
    X2 = np.array([[log_mu_ratio, mu_tot, 1.0]], float)
    ph1 = float(clf.predict_proba(X1)[:,1][0])
    ph2 = float(clf.predict_proba(X2)[:,1][0])
    return ph1, ph2


# ----------------------------- isotonic helpers -----------------------------
def iso_fit_curve(p: np.ndarray, y: np.ndarray) -> Optional[Dict[str, List[float]]]:
    if len(y) < 50 or len(np.unique(y)) < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, y)
    return {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}

def iso_apply_scalar(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve:
        return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve["x"], float); y = np.array(curve["y"], float)
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))

def fit_cdf_isotonic(df_val: pd.DataFrame, S_base: np.ndarray,
                     beta_mu: float, mu_bar: float, art: GoalsArtifact) -> Optional[Dict[str, List[float]]]:
    Xp, Yb = [], []
    S_T = float(S_base[-1])
    for _, r in df_val.iterrows():
        mu_h, mu_a = float(r["mu_home"]), float(r["mu_away"])
        S_target = bp_no_goal_prob(mu_h, mu_a, art.c_bp, G=art.max_goals)
        gamma0 = match_gamma_from_targets(S_T, S_target)
        gamma = gamma0 * ((float(r["mu_tot"]) / mu_bar) ** beta_mu)
        F = cdf_from_base_and_gamma(S_base, gamma)
        fm = r["first_min"]
        for t in range(1, len(F)):
            Xp.append(float(F[t]))
            Yb.append(0 if pd.isna(fm) else (1 if int(fm) <= t else 0))
    Xp = np.array(Xp, float); Yb = np.array(Yb, int)
    if len(np.unique(Yb)) < 2:
        return None
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(Xp, Yb)
    return {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}


# ----------------------------- LTS: closed form (no sims) -----------------------------
def last_scorer_probs_from_h(h: np.ndarray, pH1: float, pH2: float, second_half_start: int) -> Dict[str, float]:
    """
    Given hazard h[1..T], piecewise pH (1H/2H):
      last_home = sum_t pH(t) * h[t] * prod_{u>t}(1 - h[u])
      last_away = sum_t (1-pH(t)) * h[t] * prod_{u>t}(1 - h[u])
      last_none = prod_{u>=1}(1 - h[u])
    """
    T = h.shape[0] - 1
    one_minus = 1.0 - h
    R = np.ones(T + 2, float)  # suffix products
    for t in range(T, 0, -1):
        R[t] = R[t+1] * one_minus[t]
    p_none = float(R[1])
    p_last_H = 0.0; p_last_A = 0.0
    for t in range(1, T+1):
        pH_t = pH1 if t < second_half_start else pH2
        inc = float(h[t] * R[t+1])
        p_last_H += pH_t * inc
        p_last_A += (1.0 - pH_t) * inc
    s = p_last_H + p_last_A + p_none
    if s > 0:
        p_last_H /= s; p_last_A /= s; p_none /= s
    return {"last_home": p_last_H, "last_away": p_last_A, "last_none": p_none}


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals-artifact", required=True)
    ap.add_argument("--leagues", type=int, nargs="+", required=True)
    ap.add_argument("--train-seasons", required=True)
    ap.add_argument("--val-seasons", required=True)
    ap.add_argument("--test-seasons", required=True)
    ap.add_argument("--outdir", required=True)

    # Flex knobs (minutes & bands)
    ap.add_argument("--tmax", type=int, default=90, help="Last minute included in regulation (e.g., 90, 95).")
    ap.add_argument("--bands", type=str,
                    default="1-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90",
                    help="Comma-separated inclusive bands like '1-15,16-30,31-45,46-60,61-75,76-90'.")

    # Hazard smoothing / shape
    ap.add_argument("--ma-window", type=int, default=5, help="Moving-average window for baseline hazard.")
    ap.add_argument("--min-events", type=int, default=200, help="Minimum first-goal events to fit a baseline.")
    ap.add_argument("--goal-any-prior", type=float, default=0.92, help="Fallback prior P(any goal by TMAX).")
    ap.add_argument("--use-2h-bump", action="store_true")
    ap.add_argument("--second-half-start", type=int, default=46, help="Minute at which 2nd half starts.")

    # Beta-mu tuning
    ap.add_argument("--beta-grid", default="-0.6:0.6:0.06",
                    help="Grid for beta_mu; either 'start:end:step' or comma list")

    # Repro / UX
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prompt", action="store_true", help="Print a human-friendly plan/config summary.")

    args = ap.parse_args()

    # deterministic
    random.seed(args.seed); np.random.seed(args.seed)

    leagues = args.leagues
    train_seasons = expand_seasons(args.train_seasons)
    val_seasons   = expand_seasons(args.val_seasons)
    test_seasons  = expand_seasons(args.test_seasons)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    tmax = int(args.tmax)
    bands = parse_bands(args.bands, tmax)
    ma_window = int(args.ma_window)
    second_half_start = int(args.second_half_start)
    goal_any_prior = float(args.goal_any_prior)

    art = load_goals_artifact(args.goals_artifact)

    if args.prompt:
        print("────────────────────────────────────────────────────────────────")
        print("Minute Model: config summary")
        print(f"• Leagues: {leagues}   Train: {train_seasons}  Val: {val_seasons}  Test: {test_seasons}")
        print(f"• Minutes: 1..{tmax} (2H starts at {second_half_start})")
        print(f"• Bands: {[f'{L:02d}-{R:02d}' for (L,R) in bands]} + ['no_goal']")
        print(f"• Hazard: MA-window={ma_window}  Fallback P(any_goal_by_TMAX)={goal_any_prior:.2f}")
        print(f"• 2H bump: {'ON' if args.use_2h_bump else 'OFF'}")
        print("────────────────────────────────────────────────────────────────")

    print("[INFO] Loading rows...")
    df_train_raw = load_rows(leagues, train_seasons, tmax)
    df_val_raw   = load_rows(leagues, val_seasons,   tmax)
    df_test_raw  = load_rows(leagues, test_seasons,  tmax)
    print(f"[INFO] Train rows: {len(df_train_raw)}  | Val: {len(df_val_raw)} | Test: {len(df_test_raw)}")

    # attach mu features
    df_train = build_mu(df_train_raw, art)
    df_val   = build_mu(df_val_raw, art)
    df_test  = build_mu(df_test_raw, art)

    # ---- baseline hazard & optional 2H bump ----
    h_tot, used_events = estimate_baseline_hazard_first(
        df_train_raw, fallback_df=df_val_raw, tmax=tmax,
        ma_window=ma_window, min_events=args.min_events,
        goal_any_prior=goal_any_prior
    )
    if args.use_2h_bump:
        theta_2h = tune_theta_2h(df_train, h_tot, second_half_start)
        h_tot = apply_2h_bump(h_tot, theta_2h, second_half_start)
    else:
        theta_2h = 0.0

    S_base = survival_from_h(h_tot)
    bands_preview = band_probs_from_cdf(1.0 - S_base + 0.0, bands)
    print(f"[INFO] First-goal events used for baseline: {used_events}")
    print("[INFO] Example TFG bands from baseline:", bands_preview)

    # ---- tune beta_mu (shape flex via mu_tot) ----
    beta_grid = parse_grid(args.beta_grid, [-0.6 + 0.06*i for i in range(21)])
    beta_mu, mu_bar = tune_beta_mu(df_train, S_base, art, beta_grid)
    print(f"[INFO] Chosen beta_mu={beta_mu:.3f} (mu_bar={mu_bar:.3f})")

    # ---- scorer share by half ----
    share_clf = fit_share_logit_by_half(df_train, tmax, second_half_start)

    # ---- isotonic: CDF ----
    cal_cdf = fit_cdf_isotonic(df_val, S_base, beta_mu, mu_bar, art)
    if cal_cdf:
        Xp = []; Yp = []
        S_T = float(S_base[-1])
        for _, r in df_val.iterrows():
            mu_h, mu_a = float(r["mu_home"]), float(r["mu_away"])
            S_target = bp_no_goal_prob(mu_h, mu_a, art.c_bp, G=art.max_goals)
            gamma0 = match_gamma_from_targets(S_T, S_target)
            gamma = gamma0 * ((float(r["mu_tot"]) / mu_bar) ** beta_mu)
            F = cdf_from_base_and_gamma(S_base, gamma)
            # apply calibration
            for t in range(1, len(F)):
                F[t] = iso_apply_scalar(float(F[t]), cal_cdf)
            fm = r["first_min"]
            for t in range(1, len(F)):
                Xp.append(float(F[t]))
                Yp.append(0 if pd.isna(fm) else (1 if int(fm) <= t else 0))
        Xp = np.array(Xp, float); Yp = np.array(Yp, int)
        if len(np.unique(Yp)) >= 2:
            print(f"[INFO] Val CDF (isotonic):     logloss={log_loss(Yp, np.vstack([1-Xp,Xp]).T):.4f}  "
                  f"brier={brier_score_loss(Yp, Xp):.4f}")

    # ---- compute raw FTS on validation (for isotonic on final FTS) ----
    fts_raw_val, y_fts_val = [], []
    S_T = float(S_base[-1])
    for _, r in df_val.iterrows():
        mu_h, mu_a = float(r["mu_home"]), float(r["mu_away"])
        S_target = bp_no_goal_prob(mu_h, mu_a, art.c_bp, G=art.max_goals)
        gamma0 = match_gamma_from_targets(S_T, S_target)
        gamma = gamma0 * ((float(r["mu_tot"]) / mu_bar) ** beta_mu)
        F = cdf_from_base_and_gamma(S_base, gamma)
        if cal_cdf:
            for t in range(1, len(F)):
                F[t] = iso_apply_scalar(float(F[t]), cal_cdf)
        ph1, ph2 = p_home_minute(share_clf, float(r["log_mu_ratio"]), float(r["mu_tot"]), second_half_start, tmax)
        # FTS(home) = sum_t (F[t]-F[t-1]) * pH(t)
        pH_t = np.array([ph1]*(second_half_start-1) + [ph2]*(tmax-(second_half_start-1)), float)
        dF = (F[1:tmax+1] - F[0:tmax])
        p_fts_home_raw = float((dF * pH_t).sum())
        if (r["first_min"] is not None) and (r["first_side"] in ("H","A")):
            fts_raw_val.append(p_fts_home_raw)
            y_fts_val.append(1 if r["first_side"] == "H" else 0)

    cal_fts_home = None
    if len(y_fts_val) >= 50 and len(np.unique(y_fts_val)) >= 2:
        cal_fts_home = iso_fit_curve(np.array(fts_raw_val, float), np.array(y_fts_val, int))
        ph_cal = np.array([iso_apply_scalar(p, cal_fts_home) for p in np.array(fts_raw_val, float)])
        print(f"[INFO] Val FTS (calibrated):   logloss={log_loss(y_fts_val, np.vstack([1-ph_cal, ph_cal]).T):.4f}  "
              f"brier={brier_score_loss(y_fts_val, ph_cal):.4f}")
    else:
        print("[WARN] Skipping FTS calibration (insufficient labelled validation rows).")

    # ---- Build predictions for test ----
    preds = []
    for _, r in df_test.iterrows():
        mu_h, mu_a = float(r["mu_home"]), float(r["mu_away"])
        # match CDF
        S_target = bp_no_goal_prob(mu_h, mu_a, art.c_bp, G=art.max_goals)
        gamma0 = match_gamma_from_targets(float(S_base[-1]), S_target)
        gamma = gamma0 * ((float(r["mu_tot"]) / mu_bar) ** beta_mu)
        F = cdf_from_base_and_gamma(S_base, gamma)
        if cal_cdf:
            for t in range(1, len(F)):
                F[t] = iso_apply_scalar(float(F[t]), cal_cdf)
        h_match = hazard_from_cdf(F)

        # minute share by half
        ph1, ph2 = p_home_minute(share_clf, float(r["log_mu_ratio"]), float(r["mu_tot"]), second_half_start, tmax)
        pH_t = np.array([ph1]*(second_half_start-1) + [ph2]*(tmax-(second_half_start-1)), float)

        # FTS
        dF = (F[1:tmax+1] - F[0:tmax])
        p_fts_home = float((dF * pH_t).sum())
        if cal_fts_home:
            p_fts_home = iso_apply_scalar(p_fts_home, cal_fts_home)
        p_goal_any = float(F[-1])
        p_fts_none = float(1.0 - p_goal_any)
        p_fts_away = float(p_goal_any - p_fts_home)
        s = p_fts_home + p_fts_away + p_fts_none
        if s > 0:
            p_fts_home /= s; p_fts_away /= s; p_fts_none /= s

        # LTS (closed form)
        lts = last_scorer_probs_from_h(h_match, ph1, ph2, second_half_start)

        # Bands
        bands_probs = band_probs_from_cdf(F, bands)

        row = {
            "fixture_id": int(r["fixture_id"]),
            "league_id": int(r["league_id"]),
            "season": int(r["season"]),
            "kickoff_utc": r["kickoff_utc"],
            "mu_home": float(r["mu_home"]),
            "mu_away": float(r["mu_away"]),
            "fts_home": float(p_fts_home),
            "fts_away": float(p_fts_away),
            "fts_none": float(p_fts_none),
            "lts_home": float(lts["last_home"]),
            "lts_away": float(lts["last_away"]),
            "lts_none": float(lts["last_none"]),
        }
        for (L, R) in bands:
            row[f"tfg_{L:02d}_{R:02d}"] = float(bands_probs[f"{L:02d}-{R:02d}"])
        row["tfg_no_goal"] = float(bands_probs["no_goal"])
        preds.append(row)

    pr = pd.DataFrame(preds)

    # quick sanity prints
    for i in range(min(3, len(pr))):
        rid = int(pr.iloc[i]["fixture_id"])
        fts_sum = float(pr.iloc[i][["fts_home","fts_away","fts_none"]].sum())
        tfg_cols = [f"tfg_{L:02d}_{R:02d}" for (L,R) in bands] + ["tfg_no_goal"]
        tfg_sum = float(pr.iloc[i][tfg_cols].sum())
        print(f"[CHECK] fixture {rid}: FTS sum={fts_sum:.6f}, TFG sum={tfg_sum:.6f}")

    # ---- Save artifacts + predictions ----
    h_base = hazard_from_cdf(1.0 - S_base)  # shape (T+1,), h_base[0] is 0 by construction

    artifacts = {
        "type": "minutes_survival_v3_flex",
        "tmax": tmax,
        "bands": [{"lo": L, "hi": R} for (L, R) in bands],
        "baseline": {
        # index 0 intentionally left as None to make indices 1..T align with minutes
        "hazard_1_T": [None] + [float(x) for x in h_base[1:].tolist()],
        "survival_0_T": [float(x) for x in S_base.tolist()],
        "theta_2h": float(theta_2h),
        "ma_window": int(ma_window),
        "second_half_start": int(second_half_start),
        "goal_any_prior": float(goal_any_prior),
        },
        "shape_mapping": {
            "beta_mu": float(beta_mu),
            "mu_bar": float(mu_bar),
            "note": "F_match(t) = 1 - S_base(t) ** (gamma0 * (mu_tot/mu_bar)**beta_mu); gamma0 aligns no-goal to BP.",
        },
        "bp": {"c_bp": float(art.c_bp), "max_goals": int(art.max_goals)},
        "share_model": {
            "kind": "sklearn_logit" if hasattr(share_clf, "classes_") else "tiny_logit",
            "coef": [float(v) for v in getattr(share_clf, "coef_", np.array([[0,0,0]])).ravel().tolist()],
            "intercept": float(getattr(share_clf, "intercept_", np.array([0.0]))[0]),
            "features": ["log_mu_ratio", "mu_tot", "is_2h"],
        },
        "calibration": {"cdf": cal_cdf, "fts_home": cal_fts_home},
        "training": {
            "leagues": leagues,
            "train_seasons": train_seasons,
            "val_seasons": val_seasons,
            "test_seasons": test_seasons,
            "min_events_for_hazard": int(args.min_events),
            "use_2h_bump": bool(args.use_2h_bump),
            "beta_grid": beta_grid,
            "seed": int(args.seed),
        },
    }

    os.makedirs(outdir, exist_ok=True)
    art_path = os.path.join(outdir, "artifacts.minutes.json")
    with open(art_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    pr_path = os.path.join(outdir, "preds_test.minutes.csv")
    pr.to_csv(pr_path, index=False)
    print(f"[INFO] Saved artifacts to {art_path}")
    print(f"[INFO] Wrote test predictions to {pr_path}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
