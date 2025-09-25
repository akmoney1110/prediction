# -*- coding: utf-8 -*-
"""
Predict bookmaker-style pre-match markets for upcoming fixtures using a trained goals artifact.

Outputs (per match):
- 1X2 (raw + calibrated) and Double Chance (1X/12/X2)
- Asian Handicaps for requested lines (home quoted; win/push/lose; quarter lines supported)
- Totals Over/Under for requested lines (monotonicity enforced)
- BTTS (raw + calibrated)
- Odd/Even total goals
- Team Totals Over/Under for requested lines (monotonicity enforced)
- Diagnostics: μ_home, μ_away, aligned-feature coverage and zero-fraction

Usage:
  python manage.py predict_market \
    --artifacts artifacts/goals/artifacts.goals.json \
    --leagues 39,61 \
    --from-date 2025-09-01 --to-date 2025-10-01 \
    --out artifacts/preds/predictions.csv --debug
"""
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from django.db.models import Q
from scipy.special import gammaln, logsumexp

from matches.models import MLTrainingMatch

EPS = 1e-9

# --------------------------- Bi-variate Poisson ---------------------------

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

# --------------------------- Markets from grid ---------------------------

def _derive_markets_from_grid(
    grid: np.ndarray,
    totals_lines: List[float],
    team_totals_lines: List[float],
) -> Dict[str, float]:
    H, A = np.indices(grid.shape)
    total = H + A
    diff = H - A

    out: Dict[str, float] = {}
    # 1X2
    ph = float(grid[H > A].sum())
    pd = float(np.trace(grid))
    pa = float(grid[H < A].sum())
    out["p_home"] = ph; out["p_draw"] = pd; out["p_away"] = pa
    # DC
    out["p_1x"] = ph + pd; out["p_12"] = ph + pa; out["p_x2"] = pd + pa
    # BTTS
    out["p_btts"] = float(grid[(H > 0) & (A > 0)].sum())
    # Odd/Even
    out["p_odd_total"]  = float(grid[(total % 2) == 1].sum())
    out["p_even_total"] = float(grid[(total % 2) == 0].sum())
    # Totals
    for L in totals_lines:
        p_over = float(grid[total > L].sum())
        out[f"p_over_{L:g}"]  = p_over
        out[f"p_under_{L:g}"] = 1.0 - p_over
    # Team Totals
    for L in team_totals_lines:
        pho = float(grid[H > L].sum()); pao = float(grid[A > L].sum())
        out[f"p_home_over_{L:g}"] = pho; out[f"p_home_under_{L:g}"] = 1.0 - pho
        out[f"p_away_over_{L:g}"] = pao; out[f"p_away_under_{L:g}"] = 1.0 - pao

    # helper for AH half/integer lines
    def _ah_half_or_int(line: float) -> Tuple[float, float, float]:
        if abs(line - round(line)) < 1e-9:  # integer
            win  = float(grid[diff > -line].sum())
            push = float(grid[diff == -line].sum())
            lose = 1.0 - win - push
        else:  # half
            win  = float(grid[diff > -line].sum())
            push = 0.0
            lose = 1.0 - win
        return win, push, lose

    out["_ah_half_or_int"] = _ah_half_or_int
    return out

def _quarter_ah_probs(ah_half_or_int_fn, line: float) -> Tuple[float, float, float]:
    lo = math.floor(line * 2) / 2.0
    hi = math.ceil(line * 2) / 2.0
    w1, p1, l1 = ah_half_or_int_fn(lo)
    w2, p2, l2 = ah_half_or_int_fn(hi)
    return 0.5 * (w1 + w2), 0.5 * (p1 + p2), 0.5 * (l1 + l2)

def _is_quarter_line(L: float, eps: float = 1e-9) -> bool:
    frac = abs(L - math.floor(L))
    return abs(frac - 0.25) < eps or abs(frac - 0.75) < eps

def _enforce_monotone_over(rows: Dict[str, float], totals_lines: List[float]) -> None:
    """In-place: enforce p_over is non-increasing as the totals line increases."""
    lines_sorted = sorted(set(totals_lines))
    max_allowed = 1.0
    for L in lines_sorted:
        ko = f"p_over_{L:g}"; ku = f"p_under_{L:g}"
        if ko in rows:
            p = rows[ko]
            p_new = float(np.clip(min(p, max_allowed), EPS, 1 - EPS))
            rows[ko] = p_new
            rows[ku] = float(np.clip(1.0 - p_new, EPS, 1 - EPS))
            max_allowed = p_new

def _enforce_monotone_team_totals(rows: Dict[str, float], lines: List[float], side: str) -> None:
    """In-place: enforce monotonicity for team totals for 'home' or 'away'."""
    lines_sorted = sorted(set(lines))
    max_allowed = 1.0
    for L in lines_sorted:
        ko = f"p_{side}_over_{L:g}"; ku = f"p_{side}_under_{L:g}"
        if ko in rows:
            p = rows[ko]
            p_new = float(np.clip(min(p, max_allowed), EPS, 1 - EPS))
            rows[ko] = p_new
            rows[ku] = float(np.clip(1.0 - p_new, EPS, 1 - EPS))
            max_allowed = p_new

# --------------------------- Calibration helpers ---------------------------

def _apply_iso_curve(iso: Optional[Dict[str, Any]], p: np.ndarray) -> np.ndarray:
    if not iso: return p
    x = np.asarray((iso or {}).get("x", []), dtype=float)
    y = np.asarray((iso or {}).get("y", []), dtype=float)
    if x.size == 0 or y.size == 0: return p
    return np.interp(np.clip(p, 1e-6, 1-1e-6), x, y, left=y[0], right=y[-1])

def _apply_onextwo_cal(onextwo: Optional[Dict[str, Any]], p3: np.ndarray) -> np.ndarray:
    if not onextwo: return p3
    ph = _apply_iso_curve(onextwo.get("home"), p3[:,0])
    pd = _apply_iso_curve(onextwo.get("draw"), p3[:,1])
    pa = _apply_iso_curve(onextwo.get("away"), p3[:,2])
    stack = np.vstack([ph, pd, pa]).T
    s = stack.sum(axis=1, keepdims=True); s[s <= 0] = 1.0
    return stack / s

# --------------------------- Feature builder (MATCHES TRAINER) ---------------------------

TEAM_KEYS    = ["gf","ga","cs","shots","sot","shots_in_box","xg","conv","sot_pct","poss","corners","cards"]
DERIVED_KEYS = ["xg_per_shot","sot_rate","box_share","save_rate","xg_diff"]
ALLOWED_KEYS = ["shots_allowed","sot_allowed","shots_in_box_allowed","xga"]
SITU_KEYS    = ["h_rest_days","a_rest_days","h_matches_14d","a_matches_14d","h_matches_7d","a_matches_7d"]
CROSS_KEYS   = ["home_xgps_minus_away_sot_allow_rate","away_xgps_minus_home_sot_allow_rate"]

def _tofloat(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _synth_js_from_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback minimal stats if stats10_json is missing."""
    def _nz(v, d=0.0):
        try: return float(v) if v is not None else float(d)
        except Exception: return float(d)
    return {
        "shots": {
            "home": {
                "gf": _nz(row.get("h_gf10")), "ga": _nz(row.get("h_ga10")),
                "sot": _nz(row.get("h_sot10")), "poss": _nz(row.get("h_poss10")),
                "corners": _nz(row.get("h_corners_for10")), "cards": _nz(row.get("h_cards_for10")),
            },
            "away": {
                "gf": _nz(row.get("a_gf10")), "ga": _nz(row.get("a_ga10")),
                "sot": _nz(row.get("a_sot10")), "poss": _nz(row.get("a_poss10")),
                "corners": _nz(row.get("a_corners_for10")), "cards": _nz(row.get("a_cards_for10")),
            },
        },
        "derived": {"home": {}, "away": {}},
        "allowed": {"home": {}, "away": {}},
        "situational": {
            "h_rest_days": _nz(row.get("h_rest_days")), "a_rest_days": _nz(row.get("a_rest_days")),
            "h_matches_7d": 0.0, "a_matches_7d": 0.0,
            "h_matches_14d": _nz(row.get("h_matches_14d")), "a_matches_14d": _nz(row.get("a_matches_14d")),
        },
        # ELO/GELO defaults (trainer accepts both schemas)
        "elo": {"home": None, "away": None},
        "gelo": {"exp_home_goals": None, "exp_away_goals": None},
    }

def _build_oriented_features(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    EXACTLY mirrors trainer build_oriented_features:
      - team/opp      for TEAM_KEYS
      - teamdrv/oppdrv for DERIVED_KEYS
      - team_allowed/opp_allowed for ALLOWED_KEYS
      - CROSS_KEYS (signed; mirrored for away)
      - SITU_KEYS (same scalar for both sides)
      - elo_diff, gelo_mu_diff (if present; supports both json schemas)
    """
    js = row.get("stats10_json") or {}
    if isinstance(js, str):
        try: js = json.loads(js)
        except Exception: js = {}
    if not js:
        js = _synth_js_from_columns(row)

    # shots: accept nested {"home","away"} OR flat+shots_opp (trainer covered both)
    shots_obj     = js.get("shots", {}) or {}
    shots_opp_obj = js.get("shots_opp", {}) or {}
    if isinstance(shots_obj, dict) and ("home" in shots_obj or "away" in shots_obj):
        shots_home = shots_obj.get("home", {}) or {}
        shots_away = shots_obj.get("away", {}) or {}
    else:
        shots_home = shots_obj if isinstance(shots_obj, dict) else {}
        shots_away = shots_opp_obj if isinstance(shots_opp_obj, dict) else {}

    allowed = js.get("allowed", {}) or {}
    allowed_home = allowed.get("home", {}) or {}
    allowed_away = allowed.get("away", {}) or {}

    derived = js.get("derived", {}) or {}
    derived_home = derived.get("home", {}) or {}
    derived_away = derived.get("away", {}) or {}

    cross = js.get("cross", {}) or {}
    situ  = js.get("situational", {}) or {}

    # ELO/GELO: support both schemas
    elo_home = js.get("elo_home", None)
    elo_away = js.get("elo_away", None)
    if elo_home is None or elo_away is None:
        elo_dic = js.get("elo", {}) or {}
        elo_home = elo_dic.get("home", elo_home)
        elo_away = elo_dic.get("away", elo_away)

    gelo_home_mu = js.get("gelo_mu_home", None)
    gelo_away_mu = js.get("gelo_mu_away", None)
    if gelo_home_mu is None or gelo_away_mu is None:
        gelo_dic = js.get("gelo", {}) or {}
        gelo_home_mu = gelo_dic.get("exp_home_goals", gelo_home_mu)
        gelo_away_mu = gelo_dic.get("exp_away_goals", gelo_away_mu)

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

    def add_pair(keys: List[str], src_h: Dict, src_a: Dict, prefix: str):
        for k in keys:
            names.append(f"{prefix}_{k}")
            feats_home.append(_get(src_h, k, 0.0))
            feats_away.append(_get(src_a, k, 0.0))

    # team/offense
    add_pair(TEAM_KEYS, shots_home, shots_away, "team")
    add_pair(TEAM_KEYS, shots_away, shots_home, "opp")
    # derived
    add_pair(DERIVED_KEYS, derived_home, derived_away, "teamdrv")
    add_pair(DERIVED_KEYS, derived_away, derived_home, "oppdrv")
    # allowed
    add_pair(ALLOWED_KEYS, allowed_home, allowed_away, "team_allowed")
    add_pair(ALLOWED_KEYS, allowed_away, allowed_home, "opp_allowed")
    # cross (signed; mirrored for away)
    for k in CROSS_KEYS:
        names.append(k)
        v = _nz(cross.get(k, 0.0))
        feats_home.append(v)
        feats_away.append(-v)
    # situational (same scalar to both)
    for k in SITU_KEYS:
        names.append(k)
        v = _nz(situ.get(k, 0.0))
        feats_home.append(v)
        feats_away.append(v)

    # elo/gelo diffs
    try:
        if elo_home is not None and elo_away is not None:
            elo_diff = float(elo_home) - float(elo_away)
            names.append("elo_diff")
            feats_home.append(elo_diff)
            feats_away.append(-elo_diff)
    except Exception:
        pass

    try:
        if gelo_home_mu is not None and gelo_away_mu is not None:
            gdiff = float(gelo_home_mu) - float(gelo_away_mu)
            names.append("gelo_mu_diff")
            feats_home.append(gdiff)
            feats_away.append(-gdiff)
    except Exception:
        pass

    return np.array(feats_home, dtype=float), np.array(feats_away, dtype=float), names

# --------------------------- Alignment & standardization ---------------------------

def _align_with_ref(x_vec: np.ndarray, names: List[str], ref_names: List[str]) -> Tuple[np.ndarray, float, float, List[str]]:
    """Return vector aligned to ref_names, coverage fraction, zero fraction, and list of missing names."""
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros(len(ref_names), dtype=float)
    present = 0
    missing: List[str] = []
    for j, name in enumerate(ref_names):
        i = idx.get(name, None)
        if i is not None:
            out[j] = float(x_vec[i]); present += 1
        else:
            missing.append(name)
    coverage = present / max(1, len(ref_names))
    zero_frac = float(np.mean(out == 0.0))
    return out, coverage, zero_frac, missing

def _standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_safe = np.where(scale == 0.0, 1.0, scale)
    return (x - mean) / scale_safe

# --------------------------- Command ---------------------------

class Command(BaseCommand):
    help = "Predict 1X2/DC/AH/OU/BTTS/Odd-Even/Team Totals for upcoming MLTrainingMatch fixtures."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--artifacts", type=str, required=True, help="Path to artifacts.goals.json")
        parser.add_argument("--leagues", type=str, required=True, help="e.g. '39' or '39,61,140'")
        parser.add_argument("--from-date", type=str, default=None, help="inclusive ISO date (YYYY-MM-DD)")
        parser.add_argument("--to-date", type=str, default=None, help="exclusive ISO date (YYYY-MM-DD)")
        parser.add_argument("--out", type=str, default="predictions.csv")
        parser.add_argument("--totals", type=str, default="0.5,1.5,2.5,3.5,4.5")
        parser.add_argument("--team-totals", type=str, default="0.5,1.5,2.5")
        parser.add_argument("--ah", type=str, default="-1.5,-1.0,-0.5,0.0,0.5,1.0")
        parser.add_argument("--min-coverage", type=float, default=0.10, help="skip if < this fraction of features align")
        parser.add_argument("--max-goals", type=int, default=None, help="override grid cap; default = artifact")
        parser.add_argument("--debug", action="store_true")

    def handle(self, *args, **opts):
        # --- load artifacts
        with open(Path(opts["artifacts"]), "r") as f:
            art = json.load(f)

        ref_names: List[str] = art["feature_names"]                 # kept names (post-prune)
        mean = np.asarray(art["scaler_mean"], dtype=float)
        scale = np.asarray(art["scaler_scale"], dtype=float)
        coef = np.asarray(art["poisson_coef"], dtype=float)
        intercept = float(art["poisson_intercept"])
        bp_c = float(art["bp_c"])
        max_goals = int(opts["max_goals"]) if opts["max_goals"] is not None else int(art["max_goals"])

        # calibrators (top-level or nested)
        cal = art.get("calibration") or {}
        onextwo_cal = art.get("onextwo_cal") or cal.get("onextwo") or None
        over_cal    = art.get("over_cal")     or cal.get("over15")  or None
        btts_cal    = art.get("btts_cal")     or cal.get("btts")    or None

        # validate vector sizes
        n = len(ref_names)
        if not (n == len(mean) == len(scale) == len(coef)):
            raise RuntimeError(f"Artifact arrays mismatch: feature_names={n}, mean={len(mean)}, scale={len(scale)}, coef={len(coef)}")

        # args
        totals_lines = [float(x) for x in str(opts["totals"]).split(",") if x.strip()]
        team_totals_lines = [float(x) for x in str(opts["team_totals"]).split(",") if x.strip()]
        ah_lines = [float(x) for x in str(opts["ah"]).split(",") if x.strip()]
        leagues = [int(x) for x in str(opts["leagues"]).split(",") if x.strip()]

        # --- load upcoming fixtures
        qs = (
            MLTrainingMatch.objects
            .filter(league_id__in=leagues)
            .filter(Q(y_home_goals_90=None) | Q(y_away_goals_90=None))
            .order_by("kickoff_utc")
            .values(
                "league_id","season","kickoff_utc",
                "home_team_id","away_team_id","stats10_json",
                # fallback columns used by _synth_js_from_columns
                "h_gf10","a_gf10","h_ga10","a_ga10",
                "h_sot10","a_sot10",
                "h_poss10","a_poss10",
                "h_corners_for10","a_corners_for10",
                "h_cards_for10","a_cards_for10",
                "h_rest_days","a_rest_days",
                "h_matches_14d","a_matches_14d",
            )
        )
        if opts["from_date"]: qs = qs.filter(kickoff_utc__date__gte=opts["from_date"])
        if opts["to_date"]:   qs = qs.filter(kickoff_utc__date__lt=opts["to_date"])

        rows = list(qs)
        for r in rows:
            js = r.get("stats10_json")
            if isinstance(js, str):
                try: r["stats10_json"] = json.loads(js)
                except Exception: r["stats10_json"] = {}

        if not rows:
            self.stdout.write(self.style.WARNING("No upcoming MLTrainingMatch rows for the given filters."))
            Path(opts["out"]).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([]).to_csv(opts["out"], index=False)
            return

        min_cov = float(opts["min_coverage"])
        out_rows: List[Dict[str, Any]] = []

        # diag accumulators
        n_skipped = 0
        mu_h_sum = mu_a_sum = cov_sum = zfrac_sum = 0.0
        oneX2_errs = []; dc_errs = []; ou_errs = []; ah_errs = []

        for r in rows:
            xh, xa, names = _build_oriented_features(r)

            xh_aligned, cov_h, zfrac_h, missing_h = _align_with_ref(xh, names, ref_names)
            xa_aligned, cov_a, zfrac_a, missing_a = _align_with_ref(xa, names, ref_names)
            cov = 0.5 * (cov_h + cov_a)
            zfrac = 0.5 * (zfrac_h + zfrac_a)

            if cov < min_cov:
                n_skipped += 1
                if opts["debug"]:
                    self.stdout.write(
                        f"SKIP low coverage: {r['home_team_id']} vs {r['away_team_id']} "
                        f"@ {r['kickoff_utc']} cov={cov:.2f} "
                        f"| missing(any)≈{len(set(missing_h) | set(missing_a))}"
                    )
                continue

            # standardize + GLM μ
            xh_s = _standardize(xh_aligned, mean, scale)
            xa_s = _standardize(xa_aligned, mean, scale)
            mu_h = math.exp(float(intercept + np.dot(xh_s, coef)))
            mu_a = math.exp(float(intercept + np.dot(xa_s, coef)))
            mu_h = float(np.clip(mu_h, 1e-6, 8.0))
            mu_a = float(np.clip(mu_a, 1e-6, 8.0))

            # BP grid
            lam12 = bp_c * min(mu_h, mu_a)
            l1 = max(EPS, mu_h - lam12); l2 = max(EPS, mu_a - lam12)
            grid = _bp_grid(l1, l2, lam12, max_goals)

            # markets from grid
            mk = _derive_markets_from_grid(grid, totals_lines, team_totals_lines)
            ah_half_or_int = mk.pop("_ah_half_or_int")

            # AH (home quoted; quarter supported)
            for L in ah_lines:
                if _is_quarter_line(L):
                    w, p, l = _quarter_ah_probs(ah_half_or_int, L)
                else:
                    w, p, l = ah_half_or_int(L)
                mk[f"ah_home_{L:g}_win"]  = float(np.clip(w, EPS, 1 - EPS))
                mk[f"ah_home_{L:g}_push"] = float(np.clip(p, EPS, 1 - EPS))
                mk[f"ah_home_{L:g}_lose"] = float(np.clip(l, EPS, 1 - EPS))

            # enforce monotonicity
            _enforce_monotone_over(mk, totals_lines)
            _enforce_monotone_team_totals(mk, team_totals_lines, "home")
            _enforce_monotone_team_totals(mk, team_totals_lines, "away")

            # 1X2 calibration
            p1x2_raw = np.array([[mk["p_home"], mk["p_draw"], mk["p_away"]]], dtype=float)
            p1x2_cal = _apply_onextwo_cal(onextwo_cal, p1x2_raw)[0]

            # Over 1.5 / BTTS calibration
            p_over15_raw = float(mk.get("p_over_1.5", np.nan))
            p_over15_cal = float(_apply_iso_curve(over_cal, np.array([p_over15_raw]))[0]) if np.isfinite(p_over15_raw) else np.nan
            p_btts_cal = float(_apply_iso_curve(btts_cal, np.array([mk["p_btts"]]))[0]) if btts_cal else mk["p_btts"]

            # record
            rec = {
                "league_id": r["league_id"], "season": r["season"],
                "kickoff_utc": str(r["kickoff_utc"]),
                "home_team_id": r["home_team_id"], "away_team_id": r["away_team_id"],
                "mu_home": mu_h, "mu_away": mu_a,
                # 1X2 raw + calibrated
                "p_home": float(p1x2_raw[0,0]), "p_draw": float(p1x2_raw[0,1]), "p_away": float(p1x2_raw[0,2]),
                "p_home_cal": float(p1x2_cal[0]), "p_draw_cal": float(p1x2_cal[1]), "p_away_cal": float(p1x2_cal[2]),
                # DC
                "p_1x": mk["p_1x"], "p_12": mk["p_12"], "p_x2": mk["p_x2"],
                # BTTS
                "p_btts": mk["p_btts"], "p_btts_cal": p_btts_cal,
                # Odd/Even
                "p_odd_total": mk["p_odd_total"], "p_even_total": mk["p_even_total"],
                # Over 1.5
                "p_over_1.5": p_over15_raw, "p_over_1.5_cal": p_over15_cal,
                # diagnostics
                "feat_cov": cov, "feat_zero_frac": zfrac,
            }

            # Totals & Team Totals
            for L in totals_lines:
                rec[f"p_over_{L:g}"]  = mk[f"p_over_{L:g}"]
                rec[f"p_under_{L:g}"] = mk[f"p_under_{L:g}"]
            for L in team_totals_lines:
                rec[f"p_home_over_{L:g}"]  = mk[f"p_home_over_{L:g}"]
                rec[f"p_home_under_{L:g}"] = mk[f"p_home_under_{L:g}"]
                rec[f"p_away_over_{L:g}"]  = mk[f"p_away_over_{L:g}"]
                rec[f"p_away_under_{L:g}"] = mk[f"p_away_under_{L:g}"]
            # AH
            for L in ah_lines:
                rec[f"ah_home_{L:g}_win"]  = mk[f"ah_home_{L:g}_win"]
                rec[f"ah_home_{L:g}_push"] = mk[f"ah_home_{L:g}_push"]
                rec[f"ah_home_{L:g}_lose"] = mk[f"ah_home_{L:g}_lose"]

            out_rows.append(rec)

            # diagnostics
            mu_h_sum += mu_h; mu_a_sum += mu_a
            cov_sum  += cov;   zfrac_sum += zfrac

            # per-row sanity deltas
            oneX2_errs.append(abs(rec["p_home"] + rec["p_draw"] + rec["p_away"] - 1.0))
            dc_errs.append(max(
                abs(rec["p_1x"] - (rec["p_home"] + rec["p_draw"])),
                abs(rec["p_12"] - (rec["p_home"] + rec["p_away"])),
                abs(rec["p_x2"] - (rec["p_draw"] + rec["p_away"])),
            ))
            ou_errs.extend([abs(rec[f"p_over_{L:g}"] + rec[f"p_under_{L:g}"] - 1.0) for L in totals_lines])
            ah_errs.extend([abs(rec[f"ah_home_{L:g}_win"] + rec[f"ah_home_{L:g}_push"] + rec[f"ah_home_{L:g}_lose"] - 1.0) for L in ah_lines])

            if opts["debug"]:
                self.stdout.write(
                    f"{r['home_team_id']} vs {r['away_team_id']} @ {r['kickoff_utc']} | "
                    f"cov={cov:.2f} z0={zfrac:.2f} μ=({mu_h:.2f},{mu_a:.2f}) | "
                    f"1X2cal=({rec['p_home_cal']:.2f},{rec['p_draw_cal']:.2f},{rec['p_away_cal']:.2f}) "
                    f"OU2.5={rec.get('p_over_2.5', float('nan')):.2f} BTTS(cal)={rec['p_btts_cal']:.2f}"
                )

        # --- write CSV
        out_path = Path(opts["out"]); out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(out_path, index=False)

        # summary
        n_out = len(out_rows)
        if n_out > 0:
            cov_mean = cov_sum / n_out; zfrac_mean = zfrac_sum / n_out
            mu_h_mean = mu_h_sum / n_out; mu_a_mean = mu_a_sum / n_out
            s1 = max(oneX2_errs) if oneX2_errs else 0.0
            s2 = max(dc_errs)    if dc_errs    else 0.0
            s3 = max(ou_errs)    if ou_errs    else 0.0
            s4 = max(ah_errs)    if ah_errs    else 0.0
        else:
            cov_mean = zfrac_mean = mu_h_mean = mu_a_mean = float("nan")
            s1 = s2 = s3 = s4 = 0.0

        self.stdout.write(self.style.SUCCESS(
            f"Saved {n_out} rows → {out_path} | skipped_low_cov={n_skipped} | "
            f"cov={cov_mean:.2f} z0={zfrac_mean:.2f} mu_h={mu_h_mean:.2f} mu_a={mu_a_mean:.2f} | "
            f"sanity(max): 1x2={s1:.2e}  dc={s2:.2e}  ou={s3:.2e}  ah={s4:.2e}"
        ))
