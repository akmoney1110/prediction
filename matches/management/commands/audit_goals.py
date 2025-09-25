# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Audit a trained goals model end-to-end:
- Rebuild μ_home/μ_away from saved scaler + coefficients (using kept_feature_idx)
- Recompute markets (1X2, DC, AH, O/U, BTTS, Odd/Even, Team Totals)
- Run sanity checks (prob sums, identities, pushes)
- Compute raw + calibrated metrics (if calibrators were saved)
- Reliability-by-decile tables (1X2 components, Over1.5, BTTS)
- Save detailed CSVs and a compact metrics.json

USAGE:
  python manage.py audit_goals \
    --artifacts artifacts/goals/artifacts.goals.json \
    --val-seasons 2024 \
    --test-seasons 2025 \
    --outdir artifacts/goals_audit

Optionally include training seasons:
  --train-seasons 2020-2023
"""
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from django.db.models import Q
from sklearn.metrics import log_loss
from scipy.special import gammaln, logsumexp

from matches.models import MLTrainingMatch

# --------------------------- helpers ---------------------------

def _tofloat(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _brier_multiclass(y_true_idx: np.ndarray, p: np.ndarray) -> float:
    n, k = p.shape
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y_true_idx] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))

def _iso_predict_from_points(x: np.ndarray, table: Optional[Dict[str, Any]]) -> np.ndarray:
    if not table: return x
    xs = np.asarray(table.get("x", []), dtype=float)
    ys = np.asarray(table.get("y", []), dtype=float)
    if xs.size == 0 or ys.size == 0 or xs.size != ys.size: return x
    x = np.clip(x, xs.min(), xs.max())
    return np.interp(x, xs, ys)

def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    q = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "bin": q})
    g = df.groupby("bin", observed=True)
    return g.agg(n=("y", "size"), avg_p=("p","mean"), emp_rate=("y","mean")).reset_index()

# --------------------------- BP ---------------------------

def _bp_logpmf(h: int, a: int, lam1: float, lam2: float, lam12: float) -> float:
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

def _bp_grid(l1: float, l2: float, l12: float, max_goals: int) -> np.ndarray:
    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            grid[h, a] = math.exp(_bp_logpmf(h, a, l1, l2, l12))
    s = grid.sum()
    if s <= 0:
        grid[0, 0] = 1.0
        s = 1.0
    return grid / s

# --------------------------- feature builder (parity with trainer) ---------------------------

TEAM_KEYS    = ["gf", "ga", "cs", "shots", "sot", "shots_in_box", "xg", "conv", "sot_pct", "poss", "corners", "cards"]
DERIVED_KEYS = ["xg_per_shot", "sot_rate", "box_share", "save_rate", "xg_diff"]
ALLOWED_KEYS = ["shots_allowed", "sot_allowed", "shots_in_box_allowed", "xga"]
SITU_KEYS    = ["h_rest_days", "a_rest_days", "h_matches_14d", "a_matches_14d", "h_matches_7d", "a_matches_7d"]
CROSS_KEYS   = ["home_xgps_minus_away_sot_allow_rate", "away_xgps_minus_home_sot_allow_rate"]

def build_oriented_features(row_stats: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    js = row_stats.get("stats10_json") or {}
    if isinstance(js, str):
        try: js = json.loads(js)
        except Exception: js = {}

    shots_obj = js.get("shots", {}) or {}
    shots_opp_obj = js.get("shots_opp", {}) or {}
    if isinstance(shots_obj, dict) and ("home" in shots_obj or "away" in shots_obj):
        shots_h = shots_obj.get("home", {}) or {}
        shots_a = shots_obj.get("away", {}) or {}
    else:
        shots_h = shots_obj if isinstance(shots_obj, dict) else {}
        shots_a = shots_opp_obj if isinstance(shots_opp_obj, dict) else {}

    allowed = js.get("allowed", {}) or {}
    allow_h = allowed.get("home", {}) or {}
    allow_a = allowed.get("away", {}) or {}

    drv = js.get("derived", {}) or {}
    drv_h = drv.get("home", {}) or {}
    drv_a = drv.get("away", {}) or {}

    cross = js.get("cross", {}) or {}
    situ  = js.get("situational", {}) or {}

    elo_h = js.get("elo_home"); elo_a = js.get("elo_away")
    elo_diff = None
    if elo_h is not None and elo_a is not None:
        try: elo_diff = float(elo_h) - float(elo_a)
        except Exception: elo_diff = None
    g_h = js.get("gelo_mu_home"); g_a = js.get("gelo_mu_away")

    feats_h: List[float] = []; feats_a: List[float] = []; names: List[str] = []

    def _nz(v, d=0.0) -> float:
        try:
            vv = float(v); return vv if np.isfinite(vv) else float(d)
        except Exception:
            return float(d)

    def _get(d: Dict, key: str, default=0.0) -> float:
        return _nz(d.get(key, default), default)

    def add_pair(keys: List[str], src_h: Dict, src_a: Dict, prefix: str):
        for k in keys:
            names.append(f"{prefix}_{k}")
            feats_h.append(_get(src_h, k, 0.0))
            feats_a.append(_get(src_a, k, 0.0))

    # team/opp
    add_pair(TEAM_KEYS, shots_h, shots_a, "team")
    add_pair(TEAM_KEYS, shots_a, shots_h, "opp")

    # derived
    add_pair(DERIVED_KEYS, drv_h, drv_a, "teamdrv")
    add_pair(DERIVED_KEYS, drv_a, drv_h, "oppdrv")

    # allowed
    add_pair(ALLOWED_KEYS, allow_h, allow_a, "team_allowed")
    add_pair(ALLOWED_KEYS, allow_a, allow_h, "opp_allowed")

    # cross (signed & mirrored)
    for k in CROSS_KEYS:
        names.append(k)
        v = _nz(cross.get(k, 0.0))
        feats_h.append(v); feats_a.append(-v)

    # situational (same value on both)
    for k in SITU_KEYS:
        names.append(k)
        v = _nz(situ.get(k, 0.0))
        feats_h.append(v); feats_a.append(v)

    if elo_diff is not None:
        names.append("elo_diff"); feats_h.append(elo_diff); feats_a.append(-elo_diff)
    if g_h is not None and g_a is not None:
        try:
            gdiff = float(g_h) - float(g_a)
            names.append("gelo_mu_diff"); feats_h.append(gdiff); feats_a.append(-gdiff)
        except Exception:
            pass

    return np.asarray(feats_h, float), np.asarray(feats_a, float), names

# --------------------------- markets ---------------------------

def derive_markets(grid: np.ndarray,
                   totals_lines: List[float],
                   team_totals_lines: List[float],
                   ah_lines: List[float]) -> Dict[str, float]:
    H, A = np.indices(grid.shape)
    total = H + A
    diff  = H - A

    out: Dict[str, float] = {}
    ph = float(grid[H > A].sum()); pd = float(grid[H == A].sum()); pa = float(grid[H < A].sum())
    out.update(dict(p_home=ph, p_draw=pd, p_away=pa))
    out["p_1x"] = ph + pd; out["p_12"] = ph + pa; out["p_x2"] = pd + pa
    out["p_btts"] = float(grid[(H > 0) & (A > 0)].sum())
    out["p_odd_total"]  = float(grid[(total % 2) == 1].sum())
    out["p_even_total"] = float(grid[(total % 2) == 0].sum())
    # legacy aliases
    out["p_odd"]  = out["p_odd_total"]
    out["p_even"] = out["p_even_total"]
    for L in totals_lines:
        p_over = float(grid[total > L].sum())
        out[f"p_over_{L:g}"] = p_over
        out[f"p_under_{L:g}"] = 1.0 - p_over
    for L in team_totals_lines:
        pho = float(grid[H > L].sum()); pao = float(grid[A > L].sum())
        out[f"p_home_over_{L:g}"] = pho; out[f"p_home_under_{L:g}"] = 1.0 - pho
        out[f"p_away_over_{L:g}"] = pao; out[f"p_away_under_{L:g}"] = 1.0 - pao
    eps = 1e-9
    for line in ah_lines:
        if abs(line - round(line)) < eps:
            win  = float(grid[diff > -line].sum())
            push = float(grid[diff == -line].sum())
            lose = 1.0 - win - push
        else:
            win  = float(grid[diff > -line].sum())
            push = 0.0
            lose = 1.0 - win
        key = f"ah_home_{line:g}"
        out[f"{key}_win"]  = win
        out[f"{key}_push"] = push
        out[f"{key}_lose"] = lose
    return out

# --------------------------- DB + artifacts ---------------------------

def _load_df(league_ids: List[int], seasons: List[int]) -> pd.DataFrame:
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
            try: r["stats10_json"] = json.loads(js)
            except Exception: r["stats10_json"] = {}
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No completed matches for those filters.")
    return df

@dataclass
class Art:
    feat_names: List[str]
    kept_idx: List[int]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    c: float
    max_goals: int
    totals: List[float]
    team_totals: List[float]
    ah: List[float]
    oneX2_iso: Optional[Dict[str, Dict[str, Any]]]
    over15_iso: Optional[Dict[str, Any]]
    btts_iso: Optional[Dict[str, Any]]

def _load_artifacts(path: str) -> Art:
    with open(path, "r") as f: j = json.load(f)
    cfg = j.get("config", {})
    kept_idx = j.get("kept_feature_idx", list(range(len(j["feature_names"]))))
    return Art(
        feat_names=j["feature_names"],
        kept_idx=kept_idx,
        mean=np.asarray(j["scaler_mean"], float),
        scale=np.asarray(j["scaler_scale"], float),
        coef=np.asarray(j["poisson_coef"], float),
        intercept=float(j["poisson_intercept"]),
        c=float(j["bp_c"]),
        max_goals=int(j["max_goals"]),
        totals=[float(x) for x in cfg.get("totals_lines", [0.5,1.5,2.5,3.5,4.5])],
        team_totals=[float(x) for x in cfg.get("team_totals_lines", [0.5,1.5,2.5])],
        ah=[float(x) for x in cfg.get("ah_lines", [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])],
        oneX2_iso=j.get("onextwo_cal") or j.get("onextwo_iso") or j.get("calibration",{}).get("onextwo"),
        over15_iso=j.get("over_cal") or j.get("bovers15_cal") or j.get("calibration",{}).get("over15"),
        btts_iso=j.get("btts_cal") or j.get("calibration",{}).get("btts"),
    )

def _glm_predict_mu(x_raw: np.ndarray, art: Art) -> float:
    # apply kept_idx -> standardize -> GLM -> exp
    x = x_raw[art.kept_idx]
    denom = np.where(art.scale == 0, 1.0, art.scale)
    xs = (x - art.mean) / denom
    mu = math.exp(float(np.dot(art.coef, xs) + art.intercept))
    return max(1e-6, min(8.0, mu))

# --------------------------- audit logic ---------------------------

def _predict_partition(df: pd.DataFrame, art: Art) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mus_h, mus_a = [], []
    for _, r in df.iterrows():
        xh, xa, _ = build_oriented_features({"stats10_json": r.get("stats10_json") or {}})
        mus_h.append(_glm_predict_mu(xh, art))
        mus_a.append(_glm_predict_mu(xa, art))
    mu_h = np.asarray(mus_h, float); mu_a = np.asarray(mus_a, float)

    # markets
    recs = []
    for i, r in df.reset_index(drop=True).iterrows():
        mh = float(mu_h[i]); ma = float(mu_a[i])
        lam12 = art.c * min(mh, ma)
        l1 = max(1e-9, mh - lam12); l2 = max(1e-9, ma - lam12)
        grid = _bp_grid(l1, l2, lam12, art.max_goals)
        mk = derive_markets(grid, art.totals, art.team_totals, art.ah)
        rec = dict(
            league=r["league_id"], season=r["season"],
            home_team_id=r["home_team_id"], away_team_id=r["away_team_id"],
            yH=int(r["y_home_goals_90"]), yA=int(r["y_away_goals_90"]),
            mu_home=mh, mu_away=ma,
        ); rec.update(mk)
        recs.append(rec)
    preds = pd.DataFrame.from_records(recs)

    stats = {}
    desc_h = pd.Series(mu_h).describe(); desc_a = pd.Series(mu_a).describe()
    for k, v in desc_h.items(): stats[f"mu_home_{k}"] = float(v)
    for k, v in desc_a.items(): stats[f"mu_away_{k}"] = float(v)
    # intercept share (for quick drift check)
    mu0 = float(math.exp(art.intercept))
    stats["share_mu_home_eq_intercept"] = float(np.mean(np.isclose(mu_h, mu0, atol=1e-9)))
    stats["share_mu_away_eq_intercept"] = float(np.mean(np.isclose(mu_a, mu0, atol=1e-9)))
    return preds, stats

def _sanity_checks(preds: pd.DataFrame, art: Art) -> Dict[str, float]:
    errs: Dict[str, float] = {}
    s = preds[["p_home","p_draw","p_away"]].sum(axis=1).values
    errs["sum_1x2_max_abs_err"] = float(np.max(np.abs(s - 1.0)))
    dc1 = preds["p_1x"].values - (preds["p_home"].values + preds["p_draw"].values)
    dc2 = preds["p_12"].values - (preds["p_home"].values + preds["p_away"].values)
    dc3 = preds["p_x2"].values - (preds["p_draw"].values + preds["p_away"].values)
    errs["dc_identity_max_abs_err"] = float(np.max(np.abs(np.concatenate([dc1,dc2,dc3]))))
    for L in art.totals:
        ko, ku = f"p_over_{L:g}", f"p_under_{L:g}"
        if ko in preds and ku in preds:
            errs[f"ou_{L:g}_comp_max_abs_err"] = float(np.max(np.abs(preds[ko].values + preds[ku].values - 1.0)))
    if "p_odd_total" in preds and "p_even_total" in preds:
        errs["odd_even_comp_max_abs_err"] = float(np.max(np.abs(preds["p_odd_total"].values + preds["p_even_total"].values - 1.0)))
    for L in art.team_totals:
        for side in ["home","away"]:
            ko, ku = f"p_{side}_over_{L:g}", f"p_{side}_under_{L:g}"
            if ko in preds and ku in preds:
                errs[f"{side}_tt_{L:g}_comp_max_abs_err"] = float(np.max(np.abs(preds[ko].values + preds[ku].values - 1.0)))
    for line in art.ah:
        key = f"ah_home_{line:g}"
        cols = [f"{key}_win", f"{key}_push", f"{key}_lose"]
        if all(c in preds for c in cols):
            errs[f"{key}_sum_max_abs_err"] = float(np.max(np.abs(preds[cols].sum(axis=1).values - 1.0)))
    return errs

def _metrics_block(preds: pd.DataFrame, label: str, art: Art, outdir: Path) -> Dict[str, float]:
    m: Dict[str, float] = {}

    # 1X2 RAW
    y_idx = np.where(preds["yH"].values > preds["yA"].values, 0,
                     np.where(preds["yH"].values == preds["yA"].values, 1, 2))
    P = preds[["p_home","p_draw","p_away"]].values
    m[f"{label}_1x2_logloss_raw"] = float(log_loss(y_idx, P, labels=[0,1,2]))
    m[f"{label}_1x2_brier_raw"] = _brier_multiclass(y_idx, P)

    # reliability (RAW)
    rel_dir = outdir / "reliability"; rel_dir.mkdir(exist_ok=True, parents=True)
    reliability_table(P[:,0], (y_idx==0).astype(int)).to_csv(rel_dir / f"{label}_rel_1x2_home.csv", index=False)
    reliability_table(P[:,1], (y_idx==1).astype(int)).to_csv(rel_dir / f"{label}_rel_1x2_draw.csv", index=False)
    reliability_table(P[:,2], (y_idx==2).astype(int)).to_csv(rel_dir / f"{label}_rel_1x2_away.csv", index=False)

    # calibrated 1X2 (if saved)
    if art.oneX2_iso:
        ph = _iso_predict_from_points(P[:,0], art.oneX2_iso.get("home"))
        pd = _iso_predict_from_points(P[:,1], art.oneX2_iso.get("draw"))
        pa = _iso_predict_from_points(P[:,2], art.oneX2_iso.get("away"))
        Pc = np.vstack([ph,pd,pa]).T
        Pc = Pc / np.clip(Pc.sum(axis=1, keepdims=True), 1e-9, None)
        m[f"{label}_1x2_logloss_cal"] = float(log_loss(y_idx, Pc, labels=[0,1,2]))
        m[f"{label}_1x2_brier_cal"]  = _brier_multiclass(y_idx, Pc)

    # Over 1.5
    if "p_over_1.5" in preds:
        y_over15 = ((preds["yH"]+preds["yA"]) > 1).astype(int).values
        po = preds["p_over_1.5"].values
        m[f"{label}_over15_logloss_raw"] = float(log_loss(y_over15, np.vstack([1-po, po]).T, labels=[0,1]))
        reliability_table(po, y_over15).to_csv(rel_dir / f"{label}_rel_over15.csv", index=False)
        if art.over15_iso:
            poc = _iso_predict_from_points(po, art.over15_iso)
            m[f"{label}_over15_logloss_cal"] = float(log_loss(y_over15, np.vstack([1-poc, poc]).T, labels=[0,1]))

    # BTTS
    if "p_btts" in preds:
        y_btts = ((preds["yH"]>0) & (preds["yA"]>0)).astype(int).values
        pb = preds["p_btts"].values
        m[f"{label}_btts_logloss_raw"] = float(log_loss(y_btts, np.vstack([1-pb, pb]).T, labels=[0,1]))
        reliability_table(pb, y_btts).to_csv(rel_dir / f"{label}_rel_btts.csv", index=False)
        if art.btts_iso:
            pbc = _iso_predict_from_points(pb, art.btts_iso)
            m[f"{label}_btts_logloss_cal"] = float(log_loss(y_btts, np.vstack([1-pbc, pbc]).T, labels=[0,1]))

    return m

# --------------------------- Django command ---------------------------

class Command(BaseCommand):
    help = "Audit a trained goals model (markets + sanity + metrics + reliability tables)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--artifacts", type=str, required=True, help="Path to artifacts.goals.json")
        parser.add_argument("--outdir", type=str, default="artifacts/goals_audit")
        parser.add_argument("--train-seasons", type=str, default="")
        parser.add_argument("--val-seasons", type=str, default="")
        parser.add_argument("--test-seasons", type=str, default="")
        parser.add_argument("--leagues", type=str, default="")

    def _years(self, s: str) -> List[int]:
        s = str(s).strip()
        if not s: return []
        if "-" in s:
            a,b = s.split("-",1); return list(range(int(a), int(b)+1))
        if "," in s:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        return [int(s)]

    def handle(self, *args, **opts):
        artifacts_path = opts["artifacts"]
        outdir = Path(opts["outdir"]); outdir.mkdir(parents=True, exist_ok=True)

        art = _load_artifacts(artifacts_path)
        cfg = {}
        try:
            with open(artifacts_path,"r") as f: cfg = json.load(f).get("config", {})
        except Exception:
            pass

        leagues = [int(x) for x in str(opts["leagues"]).split(",") if x.strip()] or cfg.get("leagues", [])
        if not leagues: raise RuntimeError("No leagues provided and none found in artifacts.")

        parts = {
            "train": self._years(opts["train_seasons"]) or cfg.get("train_seasons", []),
            "val":   self._years(opts["val_seasons"])   or cfg.get("val_seasons", []),
            "test":  self._years(opts["test_seasons"])  or cfg.get("test_seasons", []),
        }

        self.stdout.write("Loading datasets…\n")
        metrics_all: Dict[str, float] = {}

        for label, seasons in parts.items():
            if not seasons: continue

            df = _load_df(leagues, seasons)
            preds, mu_stats = _predict_partition(df, art)

            # print μ summary + sample
            self.stdout.write(self.style.HTTP_INFO(
                f"== μ summary ({label}) ==\n{pd.DataFrame({'mu_home':preds['mu_home'],'mu_away':preds['mu_away']}).describe()}\n"
            ))
            sample_cols = ["league","season","home_team_id","away_team_id","yH","yA",
                           "mu_home","mu_away","p_home","p_draw","p_away","p_over_1.5","p_btts","p_odd_total","p_even_total"]
            self.stdout.write(self.style.HTTP_INFO(f"== Sample rows ({label}) ==\n{preds.sample(n=min(3,len(preds)), random_state=42)[sample_cols]}\n"))

            # sanity checks
            errs = _sanity_checks(preds, art)
            self.stdout.write(self.style.HTTP_INFO(
                f"[{label}] sanity max errors → " + "  ".join([f"{k}:{v:.2e}" for k,v in errs.items()]) + "\n"
            ))

            # metrics + reliability tables
            metr = _metrics_block(preds, label, art, outdir)
            metrics_all.update(metr)

            # save preds
            preds.to_csv(outdir / f"{label}_preds.csv", index=False)

            # enrich metrics
            uniq_pairs = len(preds[["mu_home","mu_away"]].round(6).drop_duplicates())
            metrics_all[f"{label}_unique_mu_pairs"] = float(uniq_pairs)
            metrics_all.update({f"{label}_{k}": v for k,v in mu_stats.items()})
            metrics_all.update({f"{label}_{k}": v for k,v in errs.items()})

        with open(outdir / "metrics.json", "w") as f: json.dump(metrics_all, f, indent=2)
        self.stdout.write(self.style.SUCCESS(f"Audit artifacts written → {outdir}"))
