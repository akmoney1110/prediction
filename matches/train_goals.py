# matches/management/commands/train_goals.py
"""
Train a goals model and derive betting markets (1X2, DC, AH, O/U 0.5–4.5, BTTS, Odd/Even, Team Totals).

Simplified, transparent flow:
1) Load finished matches for (leagues, seasons) buckets: TRAIN / VAL / TEST.
2) Build oriented features from stats10_json with clear H/A mapping (+ a home_flag).
3) Train a single PoissonRegressor for team goals (predicts μ for the oriented "team").
4) Tune the bivariate-Poisson coupling λ12 = c * min(μ_home, μ_away) on validation.
5) From the BP grid, derive required markets:
   - 1X2 + Double Chance (1X, 12, X2)
   - Asian Handicaps (win/push/lose) for lines like -2…+2 step 0.5, incl. 0
   - Over/Under totals 0.5…4.5
   - BTTS, Odd/Even
   - Team totals (home/away) 0.5…4.5
6) (Optional) Isotonic calibration for 1X2, Over(1.5), BTTS on validation.
7) Save transparent artifacts (JSON) with: features kept, scaler params, reg coef, bp_c, metrics.

Usage:
  python manage.py train_goals \
    --leagues 39,61 \
    --train-seasons 2016-2023 \
    --val-seasons 2024 \
    --test-seasons 2025 \
    --outdir artifacts/goals \
    --max-goals 10
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from django.core.management.base import BaseCommand
from django.db.models import Q

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from matches.models import MLTrainingMatch


# --------------------------- tiny helpers ---------------------------

def _tofloat(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def parse_year_list(s: str) -> List[int]:
    s = str(s).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


# --------------------------- bivariate Poisson ---------------------------

def bp_logpmf(h: int, a: int, lam1: float, lam2: float, lam12: float) -> float:
    if lam1 < 0 or lam2 < 0 or lam12 < 0:
        return -np.inf
    base = -(lam1 + lam2 + lam12)
    m = min(h, a)
    terms = []
    for k in range(m + 1):
        terms.append(
            base
            + (h - k) * math.log(lam1 + 1e-12)
            + (a - k) * math.log(lam2 + 1e-12)
            + k * math.log(lam12 + 1e-12)
            - (gammaln(h - k + 1) + gammaln(a - k + 1) + gammaln(k + 1))
        )
    return float(logsumexp(terms))

def bp_grid(l1: float, l2: float, l12: float, max_goals: int) -> np.ndarray:
    """Return normalized P(H=h, A=a) on [0..max_goals]^2."""
    G = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            G[h, a] = math.exp(bp_logpmf(h, a, l1, l2, l12))
    s = G.sum()
    if not np.isfinite(s) or s <= 0:
        G[0, 0] = 1.0
        s = 1.0
    return G / s


# --------------------------- markets from grid ---------------------------

def derive_markets(G: np.ndarray,
                   totals_lines: List[float],
                   team_totals_lines: List[float],
                   ah_lines: List[float]) -> Dict[str, float]:
    """
    Compute:
      - 1X2, DC(1X,12,X2), BTTS, Odd/Even
      - O/U for totals_lines (0.5..4.5 recommended)
      - Team totals O/U for team_totals_lines
      - AH win/push/lose for ah_lines (home side, e.g., -1.0, +0.5, etc.)
    """
    H, A = np.indices(G.shape)
    T = H + A
    D = H - A

    out: Dict[str, float] = {}

    # 1X2 + DC
    p_home = float(G[H > A].sum())
    p_draw = float(G[H == A].sum())
    p_away = float(G[H < A].sum())
    out.update({
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_1x": float(p_home + p_draw),
        "p_12": float(p_home + p_away),
        "p_x2": float(p_draw + p_away),
    })

    # BTTS, Odd/Even
    out["p_btts"] = float(G[(H > 0) & (A > 0)].sum())
    out["p_odd_total"] = float(G[(T % 2) == 1].sum())
    out["p_even_total"] = float(G[(T % 2) == 0].sum())

    # O/U (match totals)
    for L in totals_lines:
        out[f"p_over_{L:g}"] = float(G[T > L].sum())
        out[f"p_under_{L:g}"] = 1.0 - out[f"p_over_{L:g}"]

    # Team totals
    for L in team_totals_lines:
        out[f"p_home_over_{L:g}"] = float(G[H > L].sum())
        out[f"p_home_under_{L:g}"] = 1.0 - out[f"p_home_over_{L:g}"]
        out[f"p_away_over_{L:g}"] = float(G[A > L].sum())
        out[f"p_away_under_{L:g}"] = 1.0 - out[f"p_away_over_{L:g}"]

    # Asian handicaps (home side perspective)
    # For integer lines: return win/push/lose (3-way). For half lines: just win/lose.
    eps = 1e-9
    for line in ah_lines:
        if abs(line - round(line)) < eps:  # integer line => push possible
            win = float(G[D > -line].sum())
            push = float(G[D == -line].sum())
            lose = float(1.0 - win - push)
        else:
            win = float(G[D > -line].sum())
            push = 0.0
            lose = float(1.0 - win)
        out[f"ah_home_{line:+g}_win"] = win
        out[f"ah_home_{line:+g}_push"] = push
        out[f"ah_home_{line:+g}_lose"] = lose

    return out


# --------------------------- simple oriented features ---------------------------

TEAM_KEYS    = ["gf","ga","cs","shots","sot","shots_in_box","xg","conv","sot_pct","poss","corners","cards"]
DERIVED_KEYS = ["xg_per_shot","sot_rate","box_share","save_rate","xg_diff"]
ALLOWED_KEYS = ["shots_allowed","sot_allowed","shots_in_box_allowed","xga"]

def build_oriented_features(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Minimal but robust feature builder with correct home/away orientation.
    Reads row["stats10_json"] which may contain:
      - shots (team), shots_opp (opponent)
      - derived.home/away, allowed.home/away
      - situational {h_*, a_*}
      - elo {home, away} (optional)
      - gelo {exp_home_goals, exp_away_goals} (optional)

    Returns (xh, xa, names) with a symmetry breaker 'home_flag'.
    """
    js = row.get("stats10_json") or {}
    if isinstance(js, str):
        try:
            js = json.loads(js)
        except Exception:
            js = {}

    shots_h = js.get("shots") or {}
    shots_a = js.get("shots_opp") or {}

    derived   = js.get("derived") or {}
    drv_h     = derived.get("home") or {}
    drv_a     = derived.get("away") or {}

    allowed   = js.get("allowed") or {}
    allow_h   = allowed.get("home") or {}
    allow_a   = allowed.get("away") or {}

    situ      = js.get("situational") or {}

    elo_block = js.get("elo") or {}
    elo_h = elo_block.get("home") if isinstance(elo_block, dict) else None
    elo_a = elo_block.get("away") if isinstance(elo_block, dict) else None
    elo_diff = None
    if elo_h is not None and elo_a is not None:
        try: elo_diff = float(elo_h) - float(elo_a)
        except Exception: elo_diff = None

    gelo_block = js.get("gelo") or {}
    g_mu_h = gelo_block.get("exp_home_goals") if isinstance(gelo_block, dict) else None
    g_mu_a = gelo_block.get("exp_away_goals") if isinstance(gelo_block, dict) else None

    names: List[str] = []
    xh: List[float] = []
    xa: List[float] = []

    def _get(d, k, default=0.0): return _tofloat(d.get(k, default), default)

    # team vs opp (home POV for xh)
    for k in TEAM_KEYS:
        names.append(f"team_{k}")
        xh.append(_get(shots_h, k)); xa.append(_get(shots_a, k))
    for k in TEAM_KEYS:
        names.append(f"opp_{k}")
        xh.append(_get(shots_a, k)); xa.append(_get(shots_h, k))
    for k in DERIVED_KEYS:
        names.append(f"teamdrv_{k}")
        xh.append(_get(drv_h, k)); xa.append(_get(drv_a, k))
    for k in DERIVED_KEYS:
        names.append(f"oppdrv_{k}")
        xh.append(_get(drv_a, k)); xa.append(_get(drv_h, k))
    for k in ALLOWED_KEYS:
        names.append(f"team_allowed_{k}")
        xh.append(_get(allow_h, k)); xa.append(_get(allow_a, k))
    for k in ALLOWED_KEYS:
        names.append(f"opp_allowed_{k}")
        xh.append(_get(allow_a, k)); xa.append(_get(allow_h, k))

    # diffs (anti-symmetric)
    for k in TEAM_KEYS:
        names.append(f"diff_{k}")
        v = _get(shots_h, k) - _get(shots_a, k)
        xh.append(v); xa.append(-v)
    for k in DERIVED_KEYS:
        names.append(f"diffdrv_{k}")
        v = _get(drv_h, k) - _get(drv_a, k)
        xh.append(v); xa.append(-v)
    for k in ALLOWED_KEYS:
        names.append(f"diff_allowed_{k}")
        v = _get(allow_h, k) - _get(allow_a, k)
        xh.append(v); xa.append(-v)

    # situational diffs
    r_h = _tofloat(situ.get("h_rest_days", 0.0)); r_a = _tofloat(situ.get("a_rest_days", 0.0))
    m7_h = _tofloat(situ.get("h_matches_7d", 0.0)); m7_a = _tofloat(situ.get("a_matches_7d", 0.0))
    m14_h = _tofloat(situ.get("h_matches_14d", 0.0)); m14_a = _tofloat(situ.get("a_matches_14d", 0.0))
    for name, diff in [
        ("rest_days_diff", r_h - r_a),
        ("matches_7d_diff", m7_h - m7_a),
        ("matches_14d_diff", m14_h - m14_a),
    ]:
        names.append(name); xh.append(diff); xa.append(-diff)

    # elo/gelo diffs (optional)
    if elo_diff is not None:
        names.append("elo_diff"); xh.append(elo_diff); xa.append(-elo_diff)
    if (g_mu_h is not None) and (g_mu_a is not None):
        try:
            gd = float(g_mu_h) - float(g_mu_a)
        except Exception:
            gd = 0.0
        names.append("gelo_mu_diff"); xh.append(gd); xa.append(-gd)

    # presence flags + venue breaker
    names.append("has_gelo"); xh.append(1.0 if (g_mu_h is not None and g_mu_a is not None) else 0.0); xa.append(xh[-1])
    names.append("home_flag"); xh.append(+1.0); xa.append(-1.0)

    # finalize
    xh = np.asarray(xh, float); xa = np.asarray(xa, float)
    return xh, xa, names


# --------------------------- calibration wrappers ---------------------------

@dataclass
class Iso:
    f: Optional[IsotonicRegression] = None
    def fit(self, p_hat: np.ndarray, y: np.ndarray):
        p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
        self.f = IsotonicRegression(out_of_bounds="clip")
        self.f.fit(p_hat, y)
    def predict(self, p_hat: np.ndarray) -> np.ndarray:
        if self.f is None: return p_hat
        p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
        return np.asarray(self.f.predict(p_hat))

@dataclass
class OneXTwoIso:
    home: Iso; draw: Iso; away: Iso
    def fit(self, p: np.ndarray, y_idx: np.ndarray):
        self.home.fit(p[:,0], (y_idx==0).astype(float))
        self.draw.fit(p[:,1], (y_idx==1).astype(float))
        self.away.fit(p[:,2], (y_idx==2).astype(float))
    def predict(self, p: np.ndarray) -> np.ndarray:
        ph = self.home.predict(p[:,0]); pd = self.draw.predict(p[:,1]); pa = self.away.predict(p[:,2])
        stack = np.vstack([ph,pd,pa]).T
        s = stack.sum(axis=1, keepdims=True); s[s<=0]=1.0
        return stack/s


# --------------------------- core training ---------------------------

@dataclass
class Artifacts:
    feature_names: List[str]
    kept_idx: List[int]
    scaler_mean: List[float]
    scaler_scale: List[float]
    coef: List[float]
    intercept: float
    bp_c: float
    max_goals: int
    onextwo_cal: Optional[Dict[str, Any]]
    over15_cal: Optional[Dict[str, Any]]
    btts_cal: Optional[Dict[str, Any]]
    config: Dict[str, Any]
    metrics: Dict[str, Any]

def brier_multiclass(y_idx: np.ndarray, p: np.ndarray) -> float:
    n, k = p.shape
    oh = np.zeros_like(p)
    oh[np.arange(n), y_idx] = 1.0
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))

def _iso_export(iso: IsotonicRegression | None):
    if iso is None:
        return None
    if hasattr(iso, "X_thresholds_") and hasattr(iso, "y_thresholds_"):
        return {
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_],
            "increasing": True
        }
    return None

def load_df(league_ids: List[int], seasons: List[int]) -> pd.DataFrame:
    qs = (
        MLTrainingMatch.objects
        .filter(league_id__in=league_ids, season__in=seasons)
        .filter(~Q(y_home_goals_90=None), ~Q(y_away_goals_90=None))
        .order_by("kickoff_utc")
        .values("league_id","season","kickoff_utc","home_team_id","away_team_id",
                "y_home_goals_90","y_away_goals_90","stats10_json")
    )
    rows = list(qs)
    for r in rows:
        js = r.get("stats10_json")
        if isinstance(js, str):
            try: r["stats10_json"] = json.loads(js)
            except Exception: r["stats10_json"] = {}
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matches found for given filters.")
    return df

def make_oriented(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, names_ref = [], [], None
    for _, r in df.iterrows():
        xh, xa, names = build_oriented_features({"stats10_json": r.get("stats10_json") or {}})
        if names_ref is None: names_ref = names
        X.append(xh); y.append(float(r["y_home_goals_90"]))
        X.append(xa); y.append(float(r["y_away_goals_90"]))
    return np.vstack(X), np.array(y, float), (names_ref or [])

def evaluate_block(df: pd.DataFrame, mu_h: np.ndarray, mu_a: np.ndarray, c: float, max_goals: int):
    yH = df["y_home_goals_90"].values.astype(int)
    yA = df["y_away_goals_90"].values.astype(int)
    idxs = np.where(yH > yA, 0, np.where(yH == yA, 1, 2))

    p1x2, pov15, pbtts = [], [], []
    for mh, ma in zip(mu_h, mu_a):
        lam12 = c * min(mh, ma)
        l1 = max(1e-9, mh - lam12); l2 = max(1e-9, ma - lam12)
        G = bp_grid(l1, l2, lam12, max_goals)
        H, A = np.indices(G.shape)
        p_home = float(G[H > A].sum())
        p_draw = float(G[H == A].sum())
        p_away = 1.0 - p_home - p_draw
        p1x2.append([p_home, p_draw, p_away])
        pov15.append(float(G[(H + A) > 1].sum()))
        pbtts.append(float(G[(H > 0) & (A > 0)].sum()))
    p1x2 = np.asarray(p1x2); pov15 = np.asarray(pov15); pbtts = np.asarray(pbtts)

    m = {
        "1x2_logloss": float(log_loss(idxs, p1x2, labels=[0,1,2])),
        "1x2_brier": float(brier_multiclass(idxs, p1x2)),
        "over15_logloss": float(log_loss(((yH+yA)>1).astype(int),
                                         np.vstack([1-pov15, pov15]).T, labels=[0,1])),
        "btts_logloss": float(log_loss(((yH>0)&(yA>0)).astype(int),
                                       np.vstack([1-pbtts, pbtts]).T, labels=[0,1])),
    }
    return m, idxs, p1x2, pov15, pbtts


# --------------------------- management command ---------------------------

class Command(BaseCommand):
    help = "Train goals model and export 1X2/DC/AH/O-U/BTTS/Odd-Even/Team Totals."

    def add_arguments(self, parser):
        parser.add_argument("--leagues", type=str, required=True, help="Comma list, e.g. '39,61'")
        parser.add_argument("--train-seasons", type=str, required=True, help="Range or list, e.g. '2016-2023'")
        parser.add_argument("--val-seasons", type=str, required=True, help="'2024'")
        parser.add_argument("--test-seasons", type=str, required=True, help="'2025'")
        parser.add_argument("--outdir", type=str, default="artifacts/goals")
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--max-goals", type=int, default=10)
        parser.add_argument("--no-calibrate", action="store_true")
        parser.add_argument("--seed", type=int, default=1337)

    def handle(self, *args, **opts):
        np.random.seed(int(opts["seed"]))

        leagues = [int(x) for x in str(opts["leagues"]).split(",")]
        train_seasons = parse_year_list(opts["train_seasons"])
        val_seasons   = parse_year_list(opts["val_seasons"])
        test_seasons  = parse_year_list(opts["test_seasons"])
        outdir = Path(opts["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
        alpha = float(opts["alpha"]); max_goals = int(opts["max_goals"])
        calibrate = not bool(opts["no_calibrate"])

        totals_lines = [0.5,1.5,2.5,3.5,4.5]
        team_totals_lines = [0.5,1.5,2.5,3.5,4.5]
        ah_lines = [-2.0,-1.5,-1.0,-0.5,0.0,+0.5,+1.0,+1.5,+2.0]

        self.stdout.write(self.style.NOTICE("Loading datasets..."))
        df_tr = load_df(leagues, train_seasons)
        df_va = load_df(leagues, val_seasons)
        df_te = load_df(leagues, test_seasons)

        X_tr, y_tr, feat_names = make_oriented(df_tr)
        X_va, y_va, _ = make_oriented(df_va)
        X_te, y_te, _ = make_oriented(df_te)

        # prune dead features (train-only), reuse mask
        zero_var = (X_tr.std(axis=0) < 1e-12)
        zero_rate = (np.mean(np.isclose(X_tr, 0.0), axis=0) > 0.98)
        keep_mask = ~(zero_var | zero_rate)
        kept_idx = np.where(keep_mask)[0].tolist()
        kept_names = [feat_names[i] for i in kept_idx]
        X_tr = X_tr[:, keep_mask]; X_va = X_va[:, keep_mask]; X_te = X_te[:, keep_mask]

        # scale
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_trs = scaler.fit_transform(X_tr)
        X_vas = scaler.transform(X_va)
        X_tes = scaler.transform(X_te)

        # poisson(team goals)
        self.stdout.write(self.style.NOTICE("Training PoissonRegressor..."))
        pr = PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-8)
        pr.fit(X_trs, y_tr)

        def predict_mu(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            mu_h, mu_a = [], []
            for _, r in df.iterrows():
                xh, xa, _ = build_oriented_features({"stats10_json": r.get("stats10_json") or {}})
                xh = scaler.transform(xh[keep_mask].reshape(1,-1))
                xa = scaler.transform(xa[keep_mask].reshape(1,-1))
                h = float(pr.predict(xh)[0]); a = float(pr.predict(xa)[0])
                mu_h.append(max(1e-6, min(8.0, h)))
                mu_a.append(max(1e-6, min(8.0, a)))
            return np.array(mu_h), np.array(mu_a)

        mu_h_va, mu_a_va = predict_mu(df_va)
        mu_h_te, mu_a_te = predict_mu(df_te)

        # tune coupling c on validation (NLL)
        self.stdout.write(self.style.NOTICE("Tuning BP coupling c on validation..."))
        yH_va = df_va["y_home_goals_90"].values.astype(int)
        yA_va = df_va["y_away_goals_90"].values.astype(int)

        def nll_for_c(c: float) -> float:
            l12 = c * np.minimum(mu_h_va, mu_a_va)
            l1 = np.maximum(1e-9, mu_h_va - l12)
            l2 = np.maximum(1e-9, mu_a_va - l12)
            ll = 0.0
            for h, a, _l1, _l2, _l12 in zip(yH_va, yA_va, l1, l2, l12):
                ll += bp_logpmf(int(h), int(a), float(_l1), float(_l2), float(_l12))
            return -ll

        cs = np.linspace(0.0, 0.6, 31)
        nlls = [nll_for_c(float(c)) for c in cs]
        c_best = float(cs[int(np.argmin(nlls))])
        self.stdout.write(self.style.SUCCESS(f"Chosen c = {c_best:.3f}"))

        # metrics + (optional) calibration on 1X2, Over(1.5), BTTS
        self.stdout.write(self.style.NOTICE("Evaluating validation/test..."))
        m_va, y1x2_va, p1x2_va, pov15_va, pbtts_va = evaluate_block(df_va, mu_h_va, mu_a_va, c_best, max_goals)
        m_te, y1x2_te, p1x2_te, pov15_te, pbtts_te = evaluate_block(df_te, mu_h_te, mu_a_te, c_best, max_goals)

        onextwo_cal = over15_cal = btts_cal = None
        if calibrate:
            self.stdout.write(self.style.NOTICE("Calibrating (1X2, Over1.5, BTTS) on validation..."))
            cal_1x2 = OneXTwoIso(Iso(), Iso(), Iso()); cal_1x2.fit(p1x2_va, y1x2_va)
            p1x2_te_cal = cal_1x2.predict(p1x2_te)

            cal_over = Iso(); cal_over.fit(pov15_va, ((df_va["y_home_goals_90"] + df_va["y_away_goals_90"]) > 1).astype(float).values)
            pov15_te_cal = cal_over.predict(pov15_te)

            cal_btts = Iso(); cal_btts.fit(pbtts_va, (((df_va["y_home_goals_90"] > 0) & (df_va["y_away_goals_90"] > 0))).astype(float).values)
            pbtts_te_cal = cal_btts.predict(pbtts_te)

            m_te.update({
                "1x2_logloss_cal": float(log_loss(y1x2_te, p1x2_te_cal, labels=[0,1,2])),
                "1x2_brier_cal": float(brier_multiclass(y1x2_te, p1x2_te_cal)),
                "over15_logloss_cal": float(log_loss(((df_te["y_home_goals_90"] + df_te["y_away_goals_90"]) > 1).astype(int),
                                                     np.vstack([1 - pov15_te_cal, pov15_te_cal]).T, labels=[0,1])),
                "btts_logloss_cal": float(log_loss((((df_te["y_home_goals_90"] > 0) & (df_te["y_away_goals_90"] > 0))).astype(int),
                                                    np.vstack([1 - pbtts_te_cal, pbtts_te_cal]).T, labels=[0,1])),
            })
            onextwo_cal = {
                "home": _iso_export(cal_1x2.home.f),
                "draw": _iso_export(cal_1x2.draw.f),
                "away": _iso_export(cal_1x2.away.f),
            }
            over15_cal = _iso_export(cal_over.f)
            btts_cal = _iso_export(cal_btts.f)

        # pack artifacts
        art = {
            "feature_names": kept_names,
            "kept_feature_idx": kept_idx,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "poisson_coef": pr.coef_.tolist(),
            "poisson_intercept": float(pr.intercept_),
            "bp_c": c_best,
            "max_goals": max_goals,
            "onextwo_cal": onextwo_cal,
            "over15_cal": over15_cal,
            "btts_cal": btts_cal,
            "config": {
                "leagues": leagues,
                "train_seasons": train_seasons,
                "val_seasons": val_seasons,
                "test_seasons": test_seasons,
                "alpha": alpha,
                "calibrate": calibrate,
                "totals_lines": totals_lines,
                "team_totals_lines": team_totals_lines,
                "ah_lines": ah_lines,
                "seed": int(opts["seed"]),
            },
            "metrics": {
                **{f"val_{k}": v for k, v in m_va.items()},
                **{f"test_{k}": v for k, v in m_te.items()},
            },
        }

        out_json = outdir / "artifacts.goals.json"
        with open(out_json, "w") as f:
            json.dump(art, f, indent=2)
        self.stdout.write(self.style.SUCCESS(f"Saved artifacts → {out_json}"))

        # also dump a sample TEST markets CSV (one row per match, covering all requested markets)
        self.stdout.write(self.style.NOTICE("Building test markets preview..."))
        rows = []
        for (_, r), mh, ma in zip(df_te.iterrows(), mu_h_te, mu_a_te):
            lam12 = c_best * min(mh, ma)
            G = bp_grid(max(1e-9, mh - lam12), max(1e-9, ma - lam12), lam12, max_goals)
            mk = derive_markets(G, totals_lines, team_totals_lines, ah_lines)
            rows.append({
                "league": r["league_id"], "season": r["season"],
                "home_team_id": r["home_team_id"], "away_team_id": r["away_team_id"],
                "yH": int(r["y_home_goals_90"]), "yA": int(r["y_away_goals_90"]),
                "mu_home": float(mh), "mu_away": float(ma),
                **mk
            })
        df_out = pd.DataFrame(rows)
        out_csv = outdir / "test_markets_preview.csv"
        df_out.to_csv(out_csv, index=False)
        self.stdout.write(self.style.SUCCESS(f"Saved markets preview → {out_csv}"))

        self.stdout.write(self.style.SUCCESS("Done."))
