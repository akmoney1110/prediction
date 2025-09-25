# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Train a Poisson goals model and export bookmaker-style markets.

Exports:
- feature_names (kept), kept_feature_idx, scaler mean/scale, GLM coef & intercept
- tuned bivariate-Poisson coupling (bp_c)
- optional isotonic calibrators for 1X2, Over(1.5), BTTS
- compatibility: top-level {onextwo_cal, over_cal, btts_cal} AND nested {"calibration":{"onextwo":...}}

Also writes:
- val/test market previews
- validation c-grid NLL
- validation reliability-by-decile tables (raw; calibrated optional)

Run example:
  python manage.py train_goals \
    --leagues 39,61,140 \
    --train-seasons 2020-2023 \
    --val-seasons 2024 \
    --test-seasons 2025 \
    --outdir artifacts/goals \
    --alpha-grid 0.1,0.3,1,3
"""
import json, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from django.core.management.base import BaseCommand, CommandParser
from django.db.models import Q

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from scipy.special import gammaln, logsumexp

from matches.models import MLTrainingMatch


# --------------------------- tiny utils ---------------------------

def _poisson_nll(y: np.ndarray, mu: np.ndarray, eps: float = 1e-9) -> float:
    mu = np.clip(mu, eps, None)
    # up to constant terms; good for model/alpha selection
    return float(np.sum(mu - y * np.log(mu)))

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

def brier_multiclass(y_true_idx: np.ndarray, p: np.ndarray) -> float:
    n, k = p.shape
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y_true_idx] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))

def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    q = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "bin": q})
    g = df.groupby("bin", observed=True)
    return g.agg(n=("y", "size"), avg_p=("p", "mean"), emp_rate=("y", "mean")).reset_index()


# --------------------------- bivariate Poisson ---------------------------

def bp_logpmf(h: int, a: int, lam1: float, lam2: float, lam12: float) -> float:
    if lam1 < 0 or lam2 < 0 or lam12 < 0:
        return -np.inf
    m = min(h, a)
    base = -(lam1 + lam2 + lam12)
    terms = []
    for k in range(m + 1):
        t = (base
             + (h - k) * math.log(lam1 + 1e-12)
             + (a - k) * math.log(lam2 + 1e-12)
             + k * math.log(lam12 + 1e-12)
             - (gammaln(h - k + 1) + gammaln(a - k + 1) + gammaln(k + 1)))
        terms.append(t)
    return float(logsumexp(terms))

def bp_grid_pmf(l1: float, l2: float, l12: float, max_goals: int) -> np.ndarray:
    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            grid[h, a] = math.exp(bp_logpmf(h, a, l1, l2, l12))
    s = grid.sum()
    if s <= 0:
        grid[0, 0] = 1.0
        s = 1.0
    return grid / s


# --------------------------- feature builder (parity with predictor) ---------------------------

TEAM_KEYS    = ["gf", "ga", "cs", "shots", "sot", "shots_in_box", "xg", "conv", "sot_pct", "poss", "corners", "cards"]
DERIVED_KEYS = ["xg_per_shot", "sot_rate", "box_share", "save_rate", "xg_diff"]
ALLOWED_KEYS = ["shots_allowed", "sot_allowed", "shots_in_box_allowed", "xga"]
SITU_KEYS    = ["h_rest_days", "a_rest_days", "h_matches_14d", "a_matches_14d", "h_matches_7d", "a_matches_7d"]
CROSS_KEYS   = ["home_xgps_minus_away_sot_allow_rate", "away_xgps_minus_home_sot_allow_rate"]

def build_oriented_features(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build oriented (home, away) feature vectors from stats10_json.

    Supports both schemas:
      A) Nested shots:
         stats10_json["shots"] = {"home": {...}, "away": {...}}
      B) Flat shots (slim builder):
         stats10_json["shots"] = <home_dict>
         stats10_json["shots_opp"] = <away_dict>
    """
    js = row.get("stats10_json") or {}
    if isinstance(js, str):
        try:
            js = json.loads(js)
        except Exception:
            js = {}

    # shots: accept nested or flat+shots_opp
    shots_obj     = js.get("shots", {}) or {}
    shots_opp_obj = js.get("shots_opp", {}) or {}
    if isinstance(shots_obj, dict) and ("home" in shots_obj or "away" in shots_obj):
        shots_home = shots_obj.get("home", {}) or {}
        shots_away = shots_obj.get("away", {}) or {}
    else:
        shots_home = shots_obj if isinstance(shots_obj, dict) else {}
        shots_away = shots_opp_obj if isinstance(shots_opp_obj, dict) else {}

    # allowed + derived (nested)
    allowed = js.get("allowed", {}) or {}
    allowed_home = allowed.get("home", {}) or {}
    allowed_away = allowed.get("away", {}) or {}

    derived = js.get("derived", {}) or {}
    derived_home = derived.get("home", {}) or {}
    derived_away = derived.get("away", {}) or {}

    # cross & situational
    cross = js.get("cross", {}) or {}
    situ  = js.get("situational", {}) or {}

    # optional ELO-ish fields
    elo_home = js.get("elo_home")
    elo_away = js.get("elo_away")
    elo_diff = None
    if elo_home is not None and elo_away is not None:
        try:
            elo_diff = float(elo_home) - float(elo_away)
        except Exception:
            elo_diff = None
    gelo_home_mu = js.get("gelo_mu_home")
    gelo_away_mu = js.get("gelo_mu_away")

    feats_home: List[float] = []
    feats_away: List[float] = []
    names: List[str] = []

    def _nz(v, d=0.0) -> float:
        try:
            vv = float(v)
            return vv if np.isfinite(vv) else float(d)
        except Exception:
            return float(d)

    def _get(d: Dict, key: str, default=0.0) -> float:
        return _nz(d.get(key, default), default)

    def add_pair(k_list: List[str], source_home: Dict, source_away: Dict, prefix: str):
        for k in k_list:
            names.append(f"{prefix}_{k}")
            feats_home.append(_get(source_home, k, 0.0))
            feats_away.append(_get(source_away, k, 0.0))

    # team/offense
    add_pair(TEAM_KEYS, shots_home, shots_away, "team")
    add_pair(TEAM_KEYS, shots_away, shots_home, "opp")

    # derived
    add_pair(DERIVED_KEYS, derived_home, derived_away, "teamdrv")
    add_pair(DERIVED_KEYS, derived_away, derived_home, "oppdrv")

    # allowed (orientation matters)
    add_pair(ALLOWED_KEYS, allowed_home, allowed_away, "team_allowed")
    add_pair(ALLOWED_KEYS, allowed_away, allowed_home, "opp_allowed")

    # cross (signed; mirrored for away)
    for k in CROSS_KEYS:
        names.append(k)
        v = _nz(cross.get(k, 0.0))
        feats_home.append(v)
        feats_away.append(-v)

    # situational (scalars duplicated for both sides)
    for k in SITU_KEYS:
        names.append(k)
        v = _nz(situ.get(k, 0.0))
        feats_home.append(v)
        feats_away.append(v)

    if elo_diff is not None:
        names.append("elo_diff")
        feats_home.append(elo_diff)
        feats_away.append(-elo_diff)
    if gelo_home_mu is not None and gelo_away_mu is not None:
        try:
            ghd = float(gelo_home_mu) - float(gelo_away_mu)
            names.append("gelo_mu_diff")
            feats_home.append(ghd)
            feats_away.append(-ghd)
        except Exception:
            pass

    return np.array(feats_home, dtype=float), np.array(feats_away, dtype=float), names


# --------------------------- markets from grid ---------------------------

def _derive_markets_from_grid(grid: np.ndarray,
                              totals_lines: List[float],
                              team_totals_lines: List[float],
                              ah_lines: List[float]) -> Dict[str, float]:
    H, A = np.indices(grid.shape)
    total = H + A
    diff = H - A

    out: Dict[str, float] = {}
    # 1X2
    ph = float(grid[H > A].sum()); pd = float(grid[H == A].sum()); pa = float(grid[H < A].sum())
    out.update(dict(p_home=ph, p_draw=pd, p_away=pa))
    # DC
    out["p_1x"] = ph + pd; out["p_12"] = ph + pa; out["p_x2"] = pd + pa
    # BTTS
    out["p_btts"] = float(grid[(H > 0) & (A > 0)].sum())
    # Odd/Even (with back-compat aliases)
    out["p_odd_total"]  = float(grid[(total % 2) == 1].sum())
    out["p_even_total"] = float(grid[(total % 2) == 0].sum())
    out["p_odd"]  = out["p_odd_total"]
    out["p_even"] = out["p_even_total"]
    # Totals
    for L in totals_lines:
        p_over = float(grid[total > L].sum())
        out[f"p_over_{L:g}"] = p_over
    # Team Totals
    for L in team_totals_lines:
        out[f"p_home_over_{L:g}"] = float(grid[H > L].sum())
        out[f"p_away_over_{L:g}"] = float(grid[A > L].sum())
    # AH (home quoted)
    eps = 1e-9
    for line in ah_lines:
        if abs(line - round(line)) < eps:  # integer line
            win  = float(grid[diff > -line].sum())
            push = float(grid[diff == -line].sum())
            lose = 1.0 - win - push
        else:  # half
            win  = float(grid[diff > -line].sum())
            push = 0.0
            lose = 1.0 - win
        out[f"ah_home_{line:g}_win"]  = win
        out[f"ah_home_{line:g}_push"] = push
        out[f"ah_home_{line:g}_lose"] = lose
    return out


# --------------------------- data ---------------------------

def _load_matches(league_ids: List[int], seasons: List[int]) -> pd.DataFrame:
    qs = (
        MLTrainingMatch.objects
        .filter(league_id__in=league_ids, season__in=seasons)
        .filter(~Q(y_home_goals_90=None), ~Q(y_away_goals_90=None))
        .order_by("kickoff_utc")
        .values("league_id","season","kickoff_utc",
                "home_team_id","away_team_id",
                "y_home_goals_90","y_away_goals_90",
                "stats10_json")
    )
    rows = list(qs)
    for r in rows:
        js = r.get("stats10_json")
        if isinstance(js, str):
            try:
                r["stats10_json"] = json.loads(js)
            except Exception:
                r["stats10_json"] = {}
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No finished MLTrainingMatch rows found for those filters.")
    return df


# --------------------------- command ---------------------------

class Command(BaseCommand):
    help = "Train goals model and export artifacts + previews."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--leagues", type=str, required=True, help="e.g. '39' or '39,61,140'")
        parser.add_argument("--train-seasons", type=str, required=True, help="e.g. '2020-2023'")
        parser.add_argument("--val-seasons", type=str, required=True, help="e.g. '2024'")
        parser.add_argument("--test-seasons", type=str, required=True, help="e.g. '2025'")
        parser.add_argument("--outdir", type=str, default="artifacts/goals")
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--alpha-grid", type=str, default="", help="Comma-separated L2 values to try; picks best by val Poisson NLL.")
        parser.add_argument("--max-goals", type=int, default=10)
        parser.add_argument("--c-max", type=float, default=0.6, help="max c in grid search")
        parser.add_argument("--c-steps", type=int, default=31, help="#points in c grid (>=5)")
        parser.add_argument("--no-calibrate", action="store_true", help="disable isotonic calibration")
        parser.add_argument("--seed", type=int, default=1337)

        parser.add_argument("--totals", type=str, default="0.5,1.5,2.5,3.5,4.5")
        parser.add_argument("--team-totals", type=str, default="0.5,1.5,2.5")
        parser.add_argument("--ah", type=str, default="-2,-1.5,-1,-0.5,0,0.5,1,1.5,2")

    def handle(self, *args, **opts):
        leagues = [int(x) for x in str(opts["leagues"]).split(",") if x.strip()]
        train_seasons = parse_year_list(opts["train_seasons"])
        val_seasons   = parse_year_list(opts["val_seasons"])
        test_seasons  = parse_year_list(opts["test_seasons"])
        outdir        = Path(opts["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
        alpha_cli     = float(opts["alpha"])
        alpha_grid    = [float(x) for x in str(opts.get("alpha_grid","")).split(",") if x.strip()]
        max_goals     = int(opts["max_goals"])
        c_max         = float(opts["c_max"])
        c_steps       = max(5, int(opts["c_steps"]))
        calibrate     = not bool(opts["no_calibrate"])
        seed          = int(opts["seed"])

        totals_lines = [float(x) for x in str(opts["totals"]).split(",") if x.strip()]
        team_totals_lines = [float(x) for x in str(opts["team_totals"]).split(",") if x.strip()]
        ah_lines = [float(x) for x in str(opts["ah"]).split(",") if x.strip()]

        np.random.seed(seed)

        # 1) Data
        self.stdout.write("Loading datasets…")
        df_tr  = _load_matches(leagues, train_seasons)
        df_val = _load_matches(leagues, val_seasons)
        df_te  = _load_matches(leagues, test_seasons)

        # 2) Oriented design
        def make_xy(df: pd.DataFrame):
            X_list, y_list, names_ref = [], [], None
            for _, r in df.iterrows():
                row_stats = {"stats10_json": r.get("stats10_json") or {}}
                xh, xa, names = build_oriented_features(row_stats)
                if names_ref is None:
                    names_ref = names
                X_list.append(xh); y_list.append(float(r["y_home_goals_90"]))
                X_list.append(xa); y_list.append(float(r["y_away_goals_90"]))
            return np.vstack(X_list), np.array(y_list, float), names_ref or []

        X_tr_raw, y_tr, feat_names = make_xy(df_tr)
        X_val_raw, y_val, _ = make_xy(df_val)
        X_te_raw, y_te, _ = make_xy(df_te)

        # 3) Prune dead features (train only)
        zero_var  = (X_tr_raw.std(axis=0) < 1e-12)
        zero_rate = (np.mean(np.isclose(X_tr_raw, 0.0), axis=0) > 0.98)
        keep_mask = ~(zero_var | zero_rate)
        kept_idx = np.where(keep_mask)[0].tolist()
        kept_names = [feat_names[i] for i in kept_idx]

        X_tr = X_tr_raw[:, keep_mask]
        X_val = X_val_raw[:, keep_mask]
        X_te = X_te_raw[:, keep_mask]

        # 4) GLM Poisson (optional alpha grid)
        self.stdout.write("Training PoissonRegressor…")

        best_alpha = alpha_cli
        best_model = None
        best_scaler = None
        best_val_nll = float("inf")

        def _fit_and_score(alpha_val: float):
            sc = StandardScaler()
            X_trs_local = sc.fit_transform(X_tr)
            model = PoissonRegressor(alpha=alpha_val, max_iter=2000, tol=1e-8)
            model.fit(X_trs_local, y_tr)

            def _predict_mu_pair(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
                mus_h, mus_a = [], []
                for _, r in df.iterrows():
                    xh, xa, _ = build_oriented_features({"stats10_json": r.get("stats10_json") or {}})
                    xh = sc.transform(xh[keep_mask].reshape(1, -1))
                    xa = sc.transform(xa[keep_mask].reshape(1, -1))
                    mu_h = float(np.clip(model.predict(xh)[0], 1e-6, 8.0))
                    mu_a = float(np.clip(model.predict(xa)[0], 1e-6, 8.0))
                    mus_h.append(mu_h); mus_a.append(mu_a)
                return np.array(mus_h), np.array(mus_a)

            mu_h_val_loc, mu_a_val_loc = _predict_mu_pair(df_val)
            yH = df_val["y_home_goals_90"].values.astype(float)
            yA = df_val["y_away_goals_90"].values.astype(float)
            nll = _poisson_nll(yH, mu_h_val_loc) + _poisson_nll(yA, mu_a_val_loc)
            return nll, model, sc

        if alpha_grid:
            self.stdout.write(f"Alpha grid: {alpha_grid}")
            for a in alpha_grid:
                nll, model, sc = _fit_and_score(a)
                self.stdout.write(f"  alpha={a:g} → val Poisson NLL={nll:.3f}")
                if nll < best_val_nll:
                    best_val_nll = nll
                    best_alpha = a
                    best_model = model
                    best_scaler = sc
            pr = best_model
            scaler = best_scaler
            self.stdout.write(f"Chosen alpha = {best_alpha:g}")
        else:
            scaler = StandardScaler()
            X_trs = scaler.fit_transform(X_tr)
            pr = PoissonRegressor(alpha=alpha_cli, max_iter=2000, tol=1e-8)
            pr.fit(X_trs, y_tr)

        # sanity on dimensions to catch drift early
        assert len(kept_names) == len(kept_idx) == len(scaler.mean_) == len(pr.coef_), \
            "kept/features/scaler/coef length mismatch"

        def predict_mus(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            mus_h, mus_a = [], []
            for _, r in df.iterrows():
                row_stats = {"stats10_json": r.get("stats10_json") or {}}
                xh, xa, _ = build_oriented_features(row_stats)
                xh = scaler.transform(xh[keep_mask].reshape(1, -1))
                xa = scaler.transform(xa[keep_mask].reshape(1, -1))
                mu_h = float(np.clip(pr.predict(xh)[0], 1e-6, 8.0))
                mu_a = float(np.clip(pr.predict(xa)[0], 1e-6, 8.0))
                mus_h.append(mu_h); mus_a.append(mu_a)
            return np.array(mus_h), np.array(mus_a)

        # Predict μ for all splits (for metrics & previews)
        mu_h_tr,  mu_a_tr  = predict_mus(df_tr)
        mu_h_val, mu_a_val = predict_mus(df_val)
        mu_h_te,  mu_a_te  = predict_mus(df_te)

        # 5) Tune BP coupling c on validation
        self.stdout.write("Tuning BP coupling c on validation…")
        yH_val = df_val["y_home_goals_90"].values.astype(int)
        yA_val = df_val["y_away_goals_90"].values.astype(int)

        def nll_for_c(c: float) -> float:
            l12 = c * np.minimum(mu_h_val, mu_a_val)
            l1 = np.maximum(1e-9, mu_h_val - l12)
            l2 = np.maximum(1e-9, mu_a_val - l12)
            ll = 0.0
            for h, a, _l1, _l2, _l12 in zip(yH_val, yA_val, l1, l2, l12):
                ll += bp_logpmf(int(h), int(a), float(_l1), float(_l2), float(_l12))
            return -ll

        cs = np.linspace(0.0, float(c_max), int(c_steps))
        nlls = [nll_for_c(float(c)) for c in cs]
        best_c = float(cs[int(np.argmin(nlls))])
        pd.DataFrame({"c": cs, "nll": nlls}).to_csv(outdir / "val_c_grid.csv", index=False)
        self.stdout.write(f"Chosen c = {best_c:.3f}")

        # 6) Metrics (+ optional calibration) on train/val/test
        def evaluate(df: pd.DataFrame, mu_h: np.ndarray, mu_a: np.ndarray, label: str):
            yH = df["y_home_goals_90"].values.astype(int)
            yA = df["y_away_goals_90"].values.astype(int)
            idxs, p1x2, pov15, pbtts = [], [], [], []
            for mh, ma, h, a in zip(mu_h, mu_a, yH, yA):
                lam12 = best_c * min(mh, ma)
                lam1 = max(1e-9, mh - lam12)
                lam2 = max(1e-9, ma - lam12)
                grid = bp_grid_pmf(l1=lam1, l2=lam2, l12=lam12, max_goals=max_goals)
                H, A = np.indices(grid.shape)
                total = H + A
                p_home = float(grid[H > A].sum())
                p_draw = float(grid[H == A].sum())
                p_away = float(grid[H < A].sum())
                p1x2.append([p_home, p_draw, p_away])
                pov15.append(float(grid[total > 1.5].sum()))
                pbtts.append(float(grid[(H > 0) & (A > 0)].sum()))
                idxs.append(0 if h > a else 1 if h == a else 2)

            idxs = np.array(idxs, int)
            p1x2 = np.array(p1x2, float)
            pov15 = np.array(pov15, float)
            pbtts = np.array(pbtts, float)

            y_over15 = ((yH + yA) > 1).astype(int)
            y_btts = ((yH > 0) & (yA > 0)).astype(int)

            return {
                f"{label}_1x2_logloss": float(log_loss(idxs, p1x2, labels=[0, 1, 2])),
                f"{label}_1x2_brier": float(brier_multiclass(idxs, p1x2)),
                f"{label}_over15_logloss": float(log_loss(y_over15, np.vstack([1 - pov15, pov15]).T)),
                f"{label}_btts_logloss": float(log_loss(y_btts,  np.vstack([1 - pbtts, pbtts]).T)),
            }, (idxs, p1x2, pov15, pbtts)

        self.stdout.write("Evaluating train/validation/test…")
        m_train, _ = evaluate(df_tr,  mu_h_tr,  mu_a_tr,  "train")
        m_val,   (yidx_val, p1x2_val, pov15_val, pbtts_val) = evaluate(df_val, mu_h_val, mu_a_val, "val")
        m_test,  (yidx_te,  p1x2_te,  pov15_te,  pbtts_te ) = evaluate(df_te,  mu_h_te,  mu_a_te,  "test")

        onextwo_cal = over_cal = btts_cal = None
        if calibrate and len(df_val) >= 200:
            self.stdout.write("Calibrating (1X2, Over 1.5, BTTS) on validation…")
            iso_h = IsotonicRegression(out_of_bounds="clip")
            iso_d = IsotonicRegression(out_of_bounds="clip")
            iso_a = IsotonicRegression(out_of_bounds="clip")
            iso_h.fit(np.clip(p1x2_val[:, 0], 1e-6, 1-1e-6), (yidx_val == 0).astype(float))
            iso_d.fit(np.clip(p1x2_val[:, 1], 1e-6, 1-1e-6), (yidx_val == 1).astype(float))
            iso_a.fit(np.clip(p1x2_val[:, 2], 1e-6, 1-1e-6), (yidx_val == 2).astype(float))

            Pte = np.vstack([
                iso_h.predict(np.clip(p1x2_te[:, 0], 1e-6, 1-1e-6)),
                iso_d.predict(np.clip(p1x2_te[:, 1], 1e-6, 1-1e-6)),
                iso_a.predict(np.clip(p1x2_te[:, 2], 1e-6, 1-1e-6)),
            ]).T
            Pte = Pte / np.clip(Pte.sum(axis=1, keepdims=True), 1e-9, None)

            iso_over = IsotonicRegression(out_of_bounds="clip")
            y_over_val = ((df_val["y_home_goals_90"] + df_val["y_away_goals_90"]) > 1).astype(int).values
            iso_over.fit(np.clip(pov15_val, 1e-6, 1-1e-6), y_over_val)
            over_te = iso_over.predict(np.clip(pov15_te, 1e-6, 1-1e-6))

            iso_btts = IsotonicRegression(out_of_bounds="clip")
            y_btts_val = (((df_val["y_home_goals_90"] > 0) & (df_val["y_away_goals_90"] > 0))).astype(int).values
            iso_btts.fit(np.clip(pbtts_val, 1e-6, 1-1e-6), y_btts_val)
            btts_te = iso_btts.predict(np.clip(pbtts_te, 1e-6, 1-1e-6))

            m_test.update({
                "test_1x2_logloss_cal": float(log_loss(yidx_te, Pte, labels=[0, 1, 2])),
                "test_1x2_brier_cal": float(brier_multiclass(yidx_te, Pte)),
                "test_over15_logloss_cal": float(log_loss(
                    ((df_te["y_home_goals_90"] + df_te["y_away_goals_90"]) > 1).astype(int),
                    np.vstack([1 - over_te, over_te]).T
                )),
                "test_btts_logloss_cal": float(log_loss(
                    (((df_te["y_home_goals_90"] > 0) & (df_te["y_away_goals_90"] > 0))).astype(int),
                    np.vstack([1 - btts_te, btts_te]).T
                )),
            })

            def _iso_export(iso: IsotonicRegression):
                x = getattr(iso, "X_thresholds_", getattr(iso, "X_", None))
                y = getattr(iso, "y_thresholds_", getattr(iso, "y_", None))
                if x is None or y is None:
                    return None
                return {"x": [float(v) for v in x], "y": [float(v) for v in y]}

            onextwo_cal = {"home": _iso_export(iso_h), "draw": _iso_export(iso_d), "away": _iso_export(iso_a)}
            over_cal = _iso_export(iso_over)
            btts_cal = _iso_export(iso_btts)
        else:
            self.stdout.write("Calibration skipped (insufficient validation size or --no-calibrate).")

        # 7) Persist artifacts (include kept_feature_idx; persist chosen alpha)
        chosen_alpha = best_alpha if alpha_grid else alpha_cli
        intercept_mu = float(np.exp(pr.intercept_))
        metrics = {
            **m_train, **m_val, **m_test,
            "intercept_mu": intercept_mu,
            "intercept_share_train": float(np.mean(np.isclose(
                np.concatenate([mu_h_tr, mu_a_tr]), intercept_mu, atol=1e-6))),
            "intercept_share_val": float(np.mean(np.isclose(
                np.concatenate([mu_h_val, mu_a_val]), intercept_mu, atol=1e-6))),
            "intercept_share_test": float(np.mean(np.isclose(
                np.concatenate([mu_h_te, mu_a_te]), intercept_mu, atol=1e-6))),
        }
        payload = {
            "feature_names": kept_names,
            "kept_feature_idx": kept_idx,                      # for audit parity
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "poisson_coef": pr.coef_.tolist(),
            "poisson_intercept": float(pr.intercept_),
            "bp_c": best_c,
            "max_goals": max_goals,
            # calibrators — top-level (predict_markets) AND nested (other trainers)
            "onextwo_cal": onextwo_cal,
            "over_cal": over_cal,
            "btts_cal": btts_cal,
            "calibration": {"onextwo": onextwo_cal, "over15": over_cal, "btts": btts_cal},
            "config": {
                "leagues": leagues,
                "train_seasons": train_seasons, "val_seasons": val_seasons, "test_seasons": test_seasons,
                "alpha": chosen_alpha,
                "alpha_grid": alpha_grid,
                "max_goals": max_goals, "c_max": float(opts["c_max"]), "c_steps": int(opts["c_steps"]),
                "totals_lines": totals_lines, "team_totals_lines": team_totals_lines, "ah_lines": ah_lines,
                "calibrate": calibrate, "seed": seed,
            },
            "metrics": metrics,
        }
        art_path = outdir / "artifacts.goals.json"
        with open(art_path, "w") as f:
            json.dump(payload, f, indent=2)
        self.stdout.write(f"Saved artifacts → {art_path}")

        # 8) Build previews for VAL + TEST
        def build_preview(df: pd.DataFrame, mu_h: np.ndarray, mu_a: np.ndarray, name: str):
            recs = []
            for i, r in df.reset_index(drop=True).iterrows():
                mh = float(mu_h[i]); ma = float(mu_a[i])
                lam12 = best_c * min(mh, ma)
                l1 = max(1e-9, mh - lam12); l2 = max(1e-9, ma - lam12)
                grid = bp_grid_pmf(l1, l2, lam12, max_goals)
                mk = _derive_markets_from_grid(grid, totals_lines, team_totals_lines, ah_lines)
                rec = dict(
                    league=r["league_id"], season=r["season"],
                    home_team_id=r["home_team_id"], away_team_id=r["away_team_id"],
                    yH=int(r["y_home_goals_90"]), yA=int(r["y_away_goals_90"]),
                    mu_home=mh, mu_away=ma,
                ); rec.update(mk)
                recs.append(rec)
            dfp = pd.DataFrame.from_records(recs)
            dfp.to_csv(outdir / f"{name}_markets_preview.csv", index=False)

        self.stdout.write("Building markets previews…")
        build_preview(df_val, mu_h_val, mu_a_val, "val")
        build_preview(df_te,  mu_h_te,  mu_a_te,  "test")

        # 9) Diagnostics: reliability (VAL only; RAW)
        if len(df_val) >= 1:
            rel_dir = outdir / "reliability"
            rel_dir.mkdir(exist_ok=True, parents=True)

            # 1X2 components (val)
            reliability_table(p1x2_val[:, 0], (yidx_val == 0).astype(int)).to_csv(rel_dir / "val_reliability_1x2_home.csv", index=False)
            reliability_table(p1x2_val[:, 1], (yidx_val == 1).astype(int)).to_csv(rel_dir / "val_reliability_1x2_draw.csv", index=False)
            reliability_table(p1x2_val[:, 2], (yidx_val == 2).astype(int)).to_csv(rel_dir / "val_reliability_1x2_away.csv", index=False)

            # Over 1.5 (val)
            y_over_val = ((df_val["y_home_goals_90"] + df_val["y_away_goals_90"]) > 1).astype(int).values
            reliability_table(pov15_val, y_over_val).to_csv(rel_dir / "val_reliability_over15.csv", index=False)

            # BTTS (val)
            y_btts_val = (((df_val["y_home_goals_90"] > 0) & (df_val["y_away_goals_90"] > 0))).astype(int).values
            reliability_table(pbtts_val, y_btts_val).to_csv(rel_dir / "val_reliability_btts.csv", index=False)

        self.stdout.write("Done.")
