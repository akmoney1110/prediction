#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train half-time/full-time (HT/FT) markets by reusing the team-goals model.

Usage (example)
---------------
export DJANGO_SETTINGS_MODULE=prediction.settings

python -m prediction.train_ht_ft \
  --goals-artifact artifacts/goals/artifacts.goals.json \
  --leagues 61 \
  --train-seasons 2020-2023 \
  --val-seasons 2024 \
  --test-seasons 2025 \
  --outdir artifacts/htft \
  --use-half-corr
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np


# --- Django setup ---
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    raise RuntimeError(
        "DJANGO_SETTINGS_MODULE not set. e.g. export DJANGO_SETTINGS_MODULE=prediction.settings"
    )

import django
django.setup()

from django.db.models import Q
from matches.models import MLTrainingMatch, Match

# Reuse functions from goals trainer
try:
    from prediction.train_goals import build_oriented_features, bp_grid_pmf
except Exception:
    # fallback if script is invoked from the same directory
    from train_goals import build_oriented_features, bp_grid_pmf

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss


# -------------------------- Utilities --------------------------

def expand_seasons(arg: str) -> List[int]:
    """
    "2020-2023,2025" -> [2020,2021,2022,2023,2025]
    """
    parts: List[int] = []
    for seg in str(arg).split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "-" in seg:
            a, b = seg.split("-", 1)
            a = int(a); b = int(b)
            parts.extend(range(min(a, b), max(a, b) + 1))
        else:
            parts.append(int(seg))
    return sorted(set(parts))


def safe_getattr_any(obj, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def iso_fit(y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, List[float]]:
    """
    Fit isotonic regression and export thresholds for JSON.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob, y_true)
    return {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}


def iso_apply_scalar(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve:
        return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve["x"], float)
    y = np.array(curve["y"], float)
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))


def one_x_two_from_grid(grid: np.ndarray) -> Tuple[float, float, float]:
    """
    Given PMF over (home_goals, away_goals), return (p_home, p_draw, p_away).
    """
    i = np.arange(grid.shape[0])[:, None]
    j = np.arange(grid.shape[1])[None, :]
    p_home = float(grid[i > j].sum())
    p_draw = float(grid[i == j].sum())
    p_away = float(grid[i < j].sum())
    return p_home, p_draw, p_away


def conv2d_grids(g1: np.ndarray, g2: np.ndarray, G: int) -> np.ndarray:
    """
    Convolution of two 2D PMFs with truncation to [0..G]x[0..G].
    """
    H = np.zeros((G + 1, G + 1), dtype=float)
    a, b = g1.shape
    c, d = g2.shape
    for i1 in range(a):
        for j1 in range(b):
            w = g1[i1, j1]
            if w == 0.0:
                continue
            i_max = min(G, i1 + c - 1)
            j_max = min(G, j1 + d - 1)
            H[i1:i_max + 1, j1:j_max + 1] += w * g2[: i_max - i1 + 1, : j_max - j1 + 1]
    s = H.sum()
    if s > 0:
        H /= s
    return H


# -------------------------- Data loading --------------------------

@dataclass
class Row:
    fixture_id: int
    league_id: int
    season: int
    kickoff_utc: str  # ISO string
    stats10_json: dict
    y_ht_home: Optional[int]
    y_ht_away: Optional[int]
    y_ft_home: Optional[int]
    y_ft_away: Optional[int]

import pandas as pd
from typing import List, Dict, Tuple, Optional
import pandas as pd
from matches.models import MLTrainingMatch, Match

# Candidate field names for HT goals on Match
_HT_NUM_PAIRS: List[Tuple[str, str]] = [
    ("goals_home_ht", "goals_away_ht"),
    ("ht_home_goals", "ht_away_goals"),
    ("halftime_home_goals", "halftime_away_goals"),
    ("goals_ht_home", "goals_ht_away"),
]
# Sometimes HT score is stored as a string like "1-0" / "1:0" / "1–0"
_HT_STR_FIELDS: List[str] = [
    "ht_score", "score_ht", "half_time_score", "score_ht_string",
]

def _parse_ht_from_match_values(vals: Dict[str, object]) -> Tuple[Optional[int], Optional[int]]:
    """
    Given a dict of Match field -> value (from .values()), return (HT_home, HT_away)
    using numeric fields if present, else parsing string score fields.
    """
    # 1) numeric pairs
    for h, a in _HT_NUM_PAIRS:
        yh = vals.get(h, None)
        ya = vals.get(a, None)
        if yh is not None and ya is not None:
            try:
                return int(yh), int(ya)
            except Exception:
                pass

    # 2) string score like "1-0", "1:0", "1–0"
    for fld in _HT_STR_FIELDS:
        s = vals.get(fld, None)
        if isinstance(s, str) and s.strip():
            norm = s.strip().replace("–", "-").replace("−", "-").replace(":", "-")
            if "-" in norm:
                L, R = norm.split("-", 1)
                try:
                    return int(L.strip()), int(R.strip())
                except Exception:
                    continue

    return None, None


def load_dataset(leagues: List[int], seasons: List[int]) -> pd.DataFrame:
    """
    Load rows from MLTrainingMatch; prefer HT labels stored on MLTrainingMatch.
    Fallback to Match.* half-time fields only if MLTrainingMatch HT is missing.
    Returns a Pandas DataFrame with:
      fixture_id, league_id, season, kickoff_utc (ISO), stats10_json,
      y_ht_home, y_ht_away, y_ft_home, y_ft_away
    """
    qs = (
        MLTrainingMatch.objects
        .filter(league_id__in=leagues, season__in=seasons)
        .only(
            "fixture_id", "league_id", "season", "kickoff_utc", "stats10_json",
            "y_ht_home", "y_ht_away", "y_home_goals_90", "y_away_goals_90"
        )
        .order_by("kickoff_utc")
    )

    rows = list(qs)
    if not rows:
        return pd.DataFrame([])

    # Determine which fixtures still need Match fallback for HT
    need_match_ids = [
        r.fixture_id for r in rows
        if (r.y_ht_home is None or r.y_ht_away is None)
    ]

    # Build a values() dict for only the fields that actually exist on Match
    match_vals_by_id: Dict[int, Dict[str, object]] = {}
    if need_match_ids:
        # Discover available fields on Match to avoid FieldError with .values()
        match_field_names = {f.name for f in Match._meta.get_fields()}
        num_keys = [k for pair in _HT_NUM_PAIRS for k in pair if k in match_field_names]
        str_keys = [k for k in _HT_STR_FIELDS if k in match_field_names]
        wanted = ["id"] + num_keys + str_keys

        if len(wanted) > 1:  # at least "id" + something
            for mv in Match.objects.filter(id__in=need_match_ids).values(*wanted):
                match_vals_by_id[mv["id"]] = mv
        else:
            # No known HT fields exist on Match; fallback will be no-op
            match_vals_by_id = {}

    # Build output rows
    out: List[Dict[str, object]] = []
    for r in rows:
        # Prefer MLTrainingMatch HT labels
        yhh = int(r.y_ht_home) if r.y_ht_home is not None else None
        yah = int(r.y_ht_away) if r.y_ht_away is not None else None

        # Fallback to Match.* only if needed
        if (yhh is None or yah is None) and (r.fixture_id in match_vals_by_id):
            mh, ma = _parse_ht_from_match_values(match_vals_by_id[r.fixture_id])
            if yhh is None:
                yhh = mh
            if yah is None:
                yah = ma

        out.append({
            "fixture_id": r.fixture_id,
            "league_id": r.league_id,
            "season": r.season,
            "kickoff_utc": (r.kickoff_utc.isoformat() if hasattr(r.kickoff_utc, "isoformat") else str(r.kickoff_utc)),
            "stats10_json": (r.stats10_json or {}),
            "y_ht_home": yhh,
            "y_ht_away": yah,
            "y_ft_home": (int(r.y_home_goals_90) if r.y_home_goals_90 is not None else None),
            "y_ft_away": (int(r.y_away_goals_90) if r.y_away_goals_90 is not None else None),
        })

    return pd.DataFrame(out)


# -------------------------- Model pieces --------------------------

@dataclass
class GoalsArtifact:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    c_bp: float
    max_goals: int
    cal_1x2: Optional[dict]  # {home:{x:[],y:[]}, draw:{..}, away:{..}} or None


def load_goals_artifact(path: str) -> GoalsArtifact:
    with open(path, "r") as f:
        art = json.load(f)
    mean = np.array(art["scaler_mean"], float)
    scale = np.array(art["scaler_scale"], float)
    coef = np.array(art["poisson_coef"], float)
    b = float(art["poisson_intercept"])
    c = float(art["bp_c"])
    G = int(art["max_goals"])
    cal_1x2 = (art.get("calibration", {}) or {}).get("onextwo")
    return GoalsArtifact(mean, scale, coef, b, c, G, cal_1x2)


def mu_from_features(x: np.ndarray, art: GoalsArtifact) -> float:
    xs = (x - art.mean) / art.scale
    return float(np.exp(art.intercept + xs.dot(art.coef)))


def ht_grid(mu_h1: float, mu_a1: float, c_bp: float, G: int) -> np.ndarray:
    lam12 = c_bp * min(mu_h1, mu_a1)
    l1 = max(1e-12, mu_h1 - lam12)
    l2 = max(1e-12, mu_a1 - lam12)
    return bp_grid_pmf(l1, l2, lam12, G)


def sh_grid(mu_h2: float, mu_a2: float, c_bp: float, G: int) -> np.ndarray:
    return ht_grid(mu_h2, mu_a2, c_bp, G)


def build_ft_grid_from_halves(g1: np.ndarray, g2: np.ndarray, G: int) -> np.ndarray:
    return conv2d_grids(g1, g2, G)


def build_ft_grid_halfcorr(
    g1: np.ndarray,          # 1H grid
    mu1_h: float, mu1_a: float,
    mu2_h: float, mu2_a: float,
    c_bp: float, rho: float, G: int
) -> np.ndarray:
    """
    Simple "game-state" correlation:
    For each (gh1,ga1), scale 2H means as:
      mu2h' = mu2_h * (1 + rho * (gh1 - mu1_h)/max(mu1_h, eps))
      mu2a' = mu2_a * (1 + rho * (ga1 - mu1_a)/max(mu1_a, eps))
    Then mix BP( mu2h', mu2a' ) weighted by P1(gh1,ga1).
    """
    eps = 1e-6
    H = np.zeros((G + 1, G + 1), dtype=float)
    for gh1 in range(g1.shape[0]):
        for ga1 in range(g1.shape[1]):
            p = g1[gh1, ga1]
            if p <= 0.0:
                continue
            scale_h = (1.0 + rho * (gh1 - mu1_h) / max(mu1_h, eps))
            scale_a = (1.0 + rho * (ga1 - mu1_a) / max(mu1_a, eps))
            mu2h_p = max(1e-12, mu2_h * max(0.2, scale_h))
            mu2a_p = max(1e-12, mu2_a * max(0.2, scale_a))
            g2 = sh_grid(mu2h_p, mu2a_p, c_bp, G)
            imax = min(G, gh1 + G)
            jmax = min(G, ga1 + G)
            H[gh1:imax+1, ga1:jmax+1] += p * g2[:imax-gh1+1, :jmax-ga1+1]
    s = H.sum()
    if s > 0:
        H /= s
    return H


def htft_joint_matrix(g1: np.ndarray, gFT: np.ndarray) -> np.ndarray:
    """
    Approximate joint P(HT result i, FT result j) with outer(pHT, pFT).
    i,j in [H(0),D(1),A(2)].
    """
    G = g1.shape[0] - 1
    i = np.arange(G+1)[:, None]; j = np.arange(G+1)[None, :]
    mH = (i > j); mD = (i == j); mA = (i < j)
    pHT = np.array([g1[mH].sum(), g1[mD].sum(), g1[mA].sum()])

    k = np.arange(G+1)[:, None]; l = np.arange(G+1)[None, :]
    nH = (k > l); nD = (k == l); nA = (k < l)
    pFT = np.array([gFT[nH].sum(), gFT[nD].sum(), gFT[nA].sum()])

    J = np.outer(pHT, pFT)
    s = J.sum()
    if s > 0:
        J /= s
    return J


# -------------------------- Training --------------------------

def fit_pi1_constants(
    df_train: pd.DataFrame,
    art: GoalsArtifact,
) -> Tuple[float, float]:
    """
    Learn π1_home, π1_away by minimizing NLL of HT goals with μ1 = π1 * μ_total.
    Grid over [0.30..0.60] with 0.01 step.
    """
    muH_tot: List[float] = []
    muA_tot: List[float] = []
    yH1: List[int] = []
    yA1: List[int] = []

    for _, r in df_train.iterrows():
        if r.get("y_ht_home") is None or r.get("y_ht_away") is None:
            continue
        xh, xa, _ = build_oriented_features({"stats10_json": r["stats10_json"]})
        muH_tot.append(mu_from_features(xh, art))
        muA_tot.append(mu_from_features(xa, art))
        yH1.append(int(r["y_ht_home"]))
        yA1.append(int(r["y_ht_away"]))

    if not yH1:
        # fallback typical split if no HT labels present
        return 0.45, 0.45

    muH_tot = np.array(muH_tot, float)
    muA_tot = np.array(muA_tot, float)
    yH1 = np.array(yH1, int)
    yA1 = np.array(yA1, int)

    grid = np.arange(0.30, 0.601, 0.01)
    best = (float("inf"), 0.45, 0.45)
    # precompute log-factorials via lgamma
    logfacH = np.array([math.lgamma(int(k)+1) for k in yH1], float)
    logfacA = np.array([math.lgamma(int(k)+1) for k in yA1], float)

    for pH in grid:
        lamH = np.clip(pH * muH_tot, 1e-12, None)
        nllH = float((lamH - yH1 * np.log(lamH) + logfacH).sum())
        for pA in grid:
            lamA = np.clip(pA * muA_tot, 1e-12, None)
            nllA = float((lamA - yA1 * np.log(lamA) + logfacA).sum())
            nll = nllH + nllA
            if nll < best[0]:
                best = (nll, float(pH), float(pA))
    _, pi1_home, pi1_away = best
    return pi1_home, pi1_away


def tune_half_rho(
    df_val: pd.DataFrame, art: GoalsArtifact,
    pi1_home: float, pi1_away: float,
    rho_grid: List[float]
) -> float:
    """
    Tune rho on validation by maximizing likelihood of FT actual scores under FT grid.
    """
    best = (float("inf"), 0.0)
    for rho in rho_grid:
        nll = 0.0
        n = 0
        for _, r in df_val.iterrows():
            if r.get("y_ft_home") is None or r.get("y_ft_away") is None:
                continue
            xh, xa, _ = build_oriented_features({"stats10_json": r["stats10_json"]})
            muH = mu_from_features(xh, art)
            muA = mu_from_features(xa, art)
            mu1h, mu1a = pi1_home * muH, pi1_away * muA
            mu2h, mu2a = (1.0 - pi1_home) * muH, (1.0 - pi1_away) * muA
            g1 = ht_grid(mu1h, mu1a, art.c_bp, art.max_goals)
            gFT = build_ft_grid_halfcorr(g1, mu1h, mu1a, mu2h, mu2a, art.c_bp, rho, art.max_goals)
            i = min(int(r["y_ft_home"]), art.max_goals)
            j = min(int(r["y_ft_away"]), art.max_goals)
            p = max(float(gFT[i, j]), 1e-12)
            nll -= math.log(p)
            n += 1
        if n > 0 and nll < best[0]:
            best = (nll, float(rho))
    return float(best[1])


def evaluate_half_markets(
    df: pd.DataFrame, art: GoalsArtifact,
    pi1_home: float, pi1_away: float,
    use_half_corr: bool, rho: float,
    cal_ht_1x2: Optional[dict], cal_gbh: Optional[dict]
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Return metrics and a dataframe of predictions for each row.
    """
    recs: List[dict] = []
    y_ht_1x2: List[int] = []
    p_ht_1x2: List[List[float]] = []
    y_gbh: List[int] = []
    p_gbh: List[float] = []

    for _, r in df.iterrows():
        xh, xa, _ = build_oriented_features({"stats10_json": r["stats10_json"]})
        muH = mu_from_features(xh, art)
        muA = mu_from_features(xa, art)

        mu1h, mu1a = pi1_home * muH, pi1_away * muA
        mu2h, mu2a = (1.0 - pi1_home) * muH, (1.0 - pi1_away) * muA

        g1 = ht_grid(mu1h, mu1a, art.c_bp, art.max_goals)

        if use_half_corr:
            gFT = build_ft_grid_halfcorr(g1, mu1h, mu1a, mu2h, mu2a, art.c_bp, rho, art.max_goals)
        else:
            g2 = sh_grid(mu2h, mu2a, art.c_bp, art.max_goals)
            gFT = build_ft_grid_from_halves(g1, g2, art.max_goals)

        # HT 1X2 (raw)
        ph, pd, pa = one_x_two_from_grid(g1)

        # Calibrate HT 1X2
        if cal_ht_1x2:
            ph = iso_apply_scalar(ph, cal_ht_1x2["home"])
            pd = iso_apply_scalar(pd, cal_ht_1x2["draw"])
            pa = iso_apply_scalar(pa, cal_ht_1x2["away"])
            s = ph + pd + pa
            if s > 0:
                ph, pd, pa = ph/s, pd/s, pa/s

        # FT 1X2 (from half model) + reuse goals 1X2 calibration
        fph, fpd, fpa = one_x_two_from_grid(gFT)
        if art.cal_1x2:
            fph = iso_apply_scalar(fph, art.cal_1x2["home"])
            fpd = iso_apply_scalar(fpd, art.cal_1x2["draw"])
            fpa = iso_apply_scalar(fpa, art.cal_1x2["away"])
            s2 = fph + fpd + fpa
            if s2 > 0:
                fph, fpd, fpa = fph/s2, fpd/s2, fpa/s2

        # GBH
        p_h1_zero = float(g1[0, 0])  # P(no 1H goals)
        if use_half_corr:
            # P(goal in 2H | 1H state) mixture
            p2_zero = 0.0
            for gh1 in range(g1.shape[0]):
                for ga1 in range(g1.shape[1]):
                    p = g1[gh1, ga1]
                    if p <= 0:
                        continue
                    eps = 1e-6
                    scale_h = (1.0 + rho * (gh1 - mu1h) / max(mu1h, eps))
                    scale_a = (1.0 + rho * (ga1 - mu1a) / max(mu1a, eps))
                    mu2h_p = max(1e-12, mu2h * max(0.2, scale_h))
                    mu2a_p = max(1e-12, mu2a * max(0.2, scale_a))
                    g2p = sh_grid(mu2h_p, mu2a_p, art.c_bp, art.max_goals)
                    p2_zero += p * float(g2p[0, 0])
            p_gbh_i = (1 - p_h1_zero) * (1 - p2_zero)
        else:
            g2 = sh_grid(mu2h, mu2a, art.c_bp, art.max_goals)
            p_gbh_i = (1 - p_h1_zero) * (1 - float(g2[0, 0]))

        # Calibrate GBH
        if cal_gbh:
            p_gbh_i = iso_apply_scalar(p_gbh_i, cal_gbh["gbh"])

        # HT/FT joint (approx)
        J = htft_joint_matrix(g1, gFT)

        rec = {
            "fixture_id": r["fixture_id"],
            "kickoff_utc": r["kickoff_utc"],
            "league_id": r["league_id"],
            "season": r["season"],
            "mu_home": muH, "mu_away": muA,
            "mu1_home": mu1h, "mu1_away": mu1a,
            "mu2_home": mu2h, "mu2_away": mu2a,
            "ht_p_home": ph, "ht_p_draw": pd, "ht_p_away": pa,
            "ft_p_home": fph, "ft_p_draw": fpd, "ft_p_away": fpa,
            "p_gbh": p_gbh_i,
            "p_HT_H__FT_H": float(J[0, 0]),
            "p_HT_H__FT_D": float(J[0, 1]),
            "p_HT_H__FT_A": float(J[0, 2]),
            "p_HT_D__FT_H": float(J[1, 0]),
            "p_HT_D__FT_D": float(J[1, 1]),
            "p_HT_D__FT_A": float(J[1, 2]),
            "p_HT_A__FT_H": float(J[2, 0]),
            "p_HT_A__FT_D": float(J[2, 1]),
            "p_HT_A__FT_A": float(J[2, 2]),
            "y_ht_home": r.get("y_ht_home"),
            "y_ht_away": r.get("y_ht_away"),
            "y_ft_home": r.get("y_ft_home"),
            "y_ft_away": r.get("y_ft_away"),
        }
        recs.append(rec)

        # Metrics prep
        if rec["y_ht_home"] is not None and rec["y_ht_away"] is not None:
            y_res = 0 if rec["y_ht_home"] > rec["y_ht_away"] else (1 if rec["y_ht_home"] == rec["y_ht_away"] else 2)
            y_ht_1x2.append(y_res)
            p_ht_1x2.append([ph, pd, pa])

        if (rec["y_ht_home"] is not None and rec["y_ht_away"] is not None and
            rec["y_ft_home"] is not None and rec["y_ft_away"] is not None):
            gbh_true = ((rec["y_ht_home"] + rec["y_ht_away"]) > 0) and \
                       (((rec["y_ft_home"] - rec["y_ht_home"]) + (rec["y_ft_away"] - rec["y_ht_away"])) > 0)
            y_gbh.append(1 if gbh_true else 0)
            p_gbh.append(p_gbh_i)
    import pandas as pd
    preds = pd.DataFrame(recs)

    metrics: Dict[str, float] = {}
    # HT 1X2 metrics
    if p_ht_1x2:
        P = np.array(p_ht_1x2, float)
        Y = np.array(y_ht_1x2, int)
        metrics["ht_1x2_logloss"] = float(log_loss(Y, P, labels=[0, 1, 2]))
        # macro Brier
        bH = brier_score_loss((Y == 0).astype(int), P[:, 0])
        bD = brier_score_loss((Y == 1).astype(int), P[:, 1])
        bA = brier_score_loss((Y == 2).astype(int), P[:, 2])
        metrics["ht_1x2_brier_macro"] = float((bH + bD + bA) / 3.0)

    # GBH metrics
    if p_gbh:
        yb = np.array(y_gbh, int)
        pb = np.array(p_gbh, float)
        metrics["gbh_logloss"] = float(log_loss(yb, np.vstack([1 - pb, pb]).T))
        metrics["gbh_brier"] = float(brier_score_loss(yb, pb))

    return metrics, preds


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals-artifact", required=True, help="Path to artifacts.goals.json produced by train_goals.py")
    ap.add_argument("--leagues", type=int, nargs="+", required=True)
    ap.add_argument("--train-seasons", required=True)
    ap.add_argument("--val-seasons", required=True)
    ap.add_argument("--test-seasons", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--use-half-corr", action="store_true",
                    help="Enable simple game-state correlation between halves; tunes rho on val")
    ap.add_argument("--rho-grid", default="0.00,0.05,0.10,0.15,0.20,0.25,0.30",
                    help="Comma list for rho grid when --use-half-corr")
    args = ap.parse_args()

    leagues = args.leagues
    train_seasons = expand_seasons(args.train_seasons)
    val_seasons   = expand_seasons(args.val_seasons)
    test_seasons  = expand_seasons(args.test_seasons)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    art = load_goals_artifact(args.goals_artifact)

    print("[INFO] Loading data...")
    df_train = load_dataset(leagues, train_seasons)
    df_val   = load_dataset(leagues, val_seasons)
    df_test  = load_dataset(leagues, test_seasons)

    # Filter rows that actually have HT labels for training/val calibration
    df_train_ht = df_train.dropna(subset=["y_ht_home", "y_ht_away"])
    df_val_ht   = df_val.dropna(subset=["y_ht_home", "y_ht_away"])

    print(f"[INFO] Train rows: {len(df_train)} (HT-labelled: {len(df_train_ht)})")
    print(f"[INFO] Val rows:   {len(df_val)} (HT-labelled: {len(df_val_ht)})")
    print(f"[INFO] Test rows:  {len(df_test)}")

    print("[INFO] Fitting π1 split...")
    pi1_home, pi1_away = fit_pi1_constants(df_train_ht, art)
    print(f"[INFO] Learned splits: pi1_home={pi1_home:.3f}  pi1_away={pi1_away:.3f}  (pi2 = 1-pi1)")

    rho = 0.0
    if args.use_half_corr:
        rho_grid = [float(x) for x in str(args.rho_grid).split(",") if x.strip()]
        print("[INFO] Tuning half correlation rho on validation...")
        rho = tune_half_rho(df_val, art, pi1_home, pi1_away, rho_grid)
        print(f"[INFO] Chosen rho = {rho:.3f}")

    # --------- Calibration on validation ---------
    print("[INFO] Calibrating HT 1X2 and GBH on validation...")
    # First, get raw val preds (no HT calibration yet; GBH uncali)
    _m_val_raw, preds_val_raw = evaluate_half_markets(
        df_val, art, pi1_home, pi1_away,
        use_half_corr=args.use_half_corr, rho=rho,
        cal_ht_1x2=None, cal_gbh=None
    )

    # Build labels for HT calib
    mask_ht = preds_val_raw["y_ht_home"].notna() & preds_val_raw["y_ht_away"].notna()
    cal_ht = None
    if mask_ht.any():
        y_res = np.where(
            preds_val_raw.loc[mask_ht, "y_ht_home"].values > preds_val_raw.loc[mask_ht, "y_ht_away"].values, 0,
            np.where(preds_val_raw.loc[mask_ht, "y_ht_home"].values == preds_val_raw.loc[mask_ht, "y_ht_away"].values, 1, 2)
        )
        P_ht = preds_val_raw.loc[mask_ht, ["ht_p_home", "ht_p_draw", "ht_p_away"]].values
        if len(y_res) >= 20:
            cal_ht = {
                "home": iso_fit(P_ht[:, 0], (y_res == 0).astype(int)),
                "draw": iso_fit(P_ht[:, 1], (y_res == 1).astype(int)),
                "away": iso_fit(P_ht[:, 2], (y_res == 2).astype(int)),
            }

    # GBH calibration
    mask_gbh = preds_val_raw[["y_ht_home", "y_ht_away", "y_ft_home", "y_ft_away"]].notna().all(axis=1)
    cal_gbh = None
    if mask_gbh.any():
        y_gbh = ((preds_val_raw.loc[mask_gbh, "y_ht_home"] + preds_val_raw.loc[mask_gbh, "y_ht_away"] > 0) &
                 ((preds_val_raw.loc[mask_gbh, "y_ft_home"] - preds_val_raw.loc[mask_gbh, "y_ht_home"]) +
                  (preds_val_raw.loc[mask_gbh, "y_ft_away"] - preds_val_raw.loc[mask_gbh, "y_ht_away"]) > 0)).astype(int).values
        p_gbh = preds_val_raw.loc[mask_gbh, "p_gbh"].values
        if len(y_gbh) >= 20:
            cal_gbh = {"gbh": iso_fit(p_gbh, y_gbh)}

    # --------- Final evaluation (val/test) with calibration ---------
    metrics_val, _   = evaluate_half_markets(df_val, art, pi1_home, pi1_away, args.use_half_corr, rho, cal_ht, cal_gbh)
    metrics_test, pr = evaluate_half_markets(df_test, art, pi1_home, pi1_away, args.use_half_corr, rho, cal_ht, cal_gbh)

    print("[INFO] Validation metrics:", json.dumps(metrics_val, indent=2))
    print("[INFO] Test metrics:", json.dumps(metrics_test, indent=2))

    # --------- Save artifacts ---------
    out_art = {
        "type": "htft",
        "goals_artifact_used": args.goals_artifact,
        "pi1_home": pi1_home,
        "pi1_away": pi1_away,
        "use_half_corr": bool(args.use_half_corr),
        "rho": rho,
        "bp_c": art.c_bp,
        "max_goals": art.max_goals,
        "calibration": {
            "ht_onextwo": cal_ht,
            "gbh": cal_gbh,
        },
        "metrics": {
            **{f"val_{k}": v for k, v in metrics_val.items()},
            **{f"test_{k}": v for k, v in metrics_test.items()},
        },
    }
    art_path = os.path.join(outdir, "artifacts.htft.json")
    with open(art_path, "w") as f:
        json.dump(out_art, f, indent=2)
    print(f"[INFO] Saved artifacts to {art_path}")

    # Also save test predictions (handy to inspect)
    pr_out = os.path.join(outdir, "preds_test.htft.csv")
    pr.to_csv(pr_out, index=False)
    print(f"[INFO] Wrote test predictions to {pr_out}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
