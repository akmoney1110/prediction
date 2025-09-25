# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Predict bookmaker-style 1X2 + Double Chance markets using a trained goals artifact.

Sources of μ (home/away):
  - Preferred: ART  → artifact + Match.stats10_json or Match.raw_result_json["stats10_json"]
  - Fallback:  DB   → MatchPrediction.lambda_home / lambda_away

Will write one PredictedMarket row per outcome:
  - ONE_X_TWO: specifier in {"HOME","DRAW","AWAY"}
  - DOUBLE_CHANCE: specifier in {"HOME_OR_DRAW","HOME_OR_AWAY","DRAW_OR_AWAY"}

Fields set: league_id, match, kickoff_utc, market_code, specifier, p_model, fair_odds,
            lambda_home, lambda_away. (book_odds/edge left untouched)

Example:
  python manage.py predict1x2DCmarket \
    --league-id 39 \
    --days 7 \
    --artifact artifacts/goals/artifacts.goals.json \
    --delete-first \
    --use-calibrated \
    --write \
    --audit /tmp/audit_1x2dc.csv \
    --verbose
"""
import json, math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from django.db.models import Q

from matches.models import MatchPrediction, PredictedMarket, Match


# ---------------- Utils ----------------

def _safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def _get_match_stats10(m: Match) -> Optional[Dict[str, Any]]:
    # Prefer concrete Match.stats10_json if present
    js = _safe_json_loads(getattr(m, "stats10_json", None))
    if js:
        return js
    raw = getattr(m, "raw_result_json", None)
    if isinstance(raw, dict):
        js = _safe_json_loads(raw.get("stats10_json"))
        if js:
            return js
    return None

def _iso_apply(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve or "x" not in curve or "y" not in curve:
        return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve["x"], float)
    y = np.array(curve["y"], float)
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))

def _bp_grid_from_components(l1: float, l2: float, l12: float, max_goals: int) -> np.ndarray:
    # Standard bivariate Poisson via series
    H = int(max_goals) + 1
    A = int(max_goals) + 1
    P = np.zeros((H, A), dtype=float)
    base = math.exp(-(l1 + l2 + l12))
    from math import factorial
    for i in range(H):
        for j in range(A):
            s = 0.0
            m = i if i < j else j
            for k in range(m + 1):
                s += (l1 ** (i - k)) / factorial(i - k) * \
                     (l2 ** (j - k)) / factorial(j - k) * \
                     (l12 ** k) / factorial(k)
            P[i, j] = base * s
    S = P.sum()
    if not np.isfinite(S) or S <= 0:
        P[:] = 0.0
        P[0, 0] = 1.0
    else:
        P /= S
    return P

def _one_x_two_from_grid(P: np.ndarray) -> Tuple[float, float, float]:
    H, A = np.indices(P.shape)
    pH = float(P[(H > A)].sum())
    pD = float(np.trace(P))
    pA = float(P[(H < A)].sum())
    s = pH + pD + pA
    if s > 0:
        pH, pD, pA = pH / s, pD / s, pA / s
    return pH, pD, pA

def _dc_from_grid(P: np.ndarray) -> Tuple[float, float, float]:
    pH, pD, pA = _one_x_two_from_grid(P)
    return (pH + pD, pH + pA, pD + pA)  # 1X, 12, X2


# --------------- Local feature builder (robust to schema) ---------------

TEAM_KEYS    = ["gf", "ga", "cs", "shots", "sot", "shots_in_box", "xg", "conv", "sot_pct", "poss", "corners", "cards"]
DERIVED_KEYS = ["xg_per_shot", "sot_rate", "box_share", "save_rate", "xg_diff"]
ALLOWED_KEYS = ["shots_allowed", "sot_allowed", "shots_in_box_allowed", "xga"]
SITU_KEYS    = ["h_rest_days", "a_rest_days", "h_matches_14d", "a_matches_14d", "h_matches_7d", "a_matches_7d"]
CROSS_KEYS   = ["home_xgps_minus_away_sot_allow_rate", "away_xgps_minus_home_sot_allow_rate"]

@dataclass
class GoalsArtifact:
    kept_idx: List[int]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    max_goals: int
    bp_c: float
    cal_onextwo: Optional[Dict[str, Dict[str, List[float]]]]  # {"home":{x..y..},"draw":...,"away":...}

def _build_oriented_features(stats10_json: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    js = stats10_json or {}

    shots     = js.get("shots") or {}
    shots_opp = js.get("shots_opp") or {}
    if isinstance(shots, dict) and ("home" in shots or "away" in shots):
        sh_home = shots.get("home") or {}
        sh_away = shots.get("away") or {}
    else:
        # slim schema
        sh_home = shots if isinstance(shots, dict) else {}
        sh_away = shots_opp if isinstance(shots_opp, dict) else {}

    allowed = js.get("allowed") or {}
    al_home = allowed.get("home") or {}
    al_away = allowed.get("away") or {}

    derived = js.get("derived") or {}
    dv_home = derived.get("home") or {}
    dv_away = derived.get("away") or {}

    cross = js.get("cross") or {}
    situ  = js.get("situational") or {}

    elo_home = js.get("elo_home"); elo_away = js.get("elo_away")
    gelo_home_mu = js.get("gelo_mu_home"); gelo_away_mu = js.get("gelo_mu_away")

    def _nz(v, d=0.0) -> float:
        try:
            z = float(v);  return z if np.isfinite(z) else float(d)
        except Exception:
            return float(d)

    names: List[str] = []
    fh: List[float] = []
    fa: List[float] = []

    def add_pair(keys, H, A, prefix):
        nonlocal names, fh, fa
        for k in keys:
            names.append(f"{prefix}_{k}")
            fh.append(_nz(H.get(k, 0.0)))
            fa.append(_nz(A.get(k, 0.0)))

    # team (own) / opponent mirroring
    add_pair(TEAM_KEYS, sh_home, sh_away, "team")
    add_pair(TEAM_KEYS, sh_away, sh_home, "opp")
    add_pair(DERIVED_KEYS, dv_home, dv_away, "teamdrv")
    add_pair(DERIVED_KEYS, dv_away, dv_home, "oppdrv")
    add_pair(ALLOWED_KEYS, al_home, al_away, "team_allowed")
    add_pair(ALLOWED_KEYS, al_away, al_home, "opp_allowed")

    for k in CROSS_KEYS:
        names.append(k)
        v = _nz(cross.get(k, 0.0))
        fh.append(v); fa.append(-v)

    for k in SITU_KEYS:
        names.append(k)
        v = _nz(situ.get(k, 0.0))
        fh.append(v); fa.append(v)

    if (elo_home is not None) and (elo_away is not None):
        try:
            names.append("elo_diff")
            d = float(elo_home) - float(elo_away)
            fh.append(d); fa.append(-d)
        except Exception:
            pass

    if (gelo_home_mu is not None) and (gelo_away_mu is not None):
        try:
            names.append("gelo_mu_diff")
            d = float(gelo_home_mu) - float(gelo_away_mu)
            fh.append(d); fa.append(-d)
        except Exception:
            pass

    return np.array(fh, float), np.array(fa, float), names


def _load_artifact(path: str) -> GoalsArtifact:
    with open(path, "r") as f:
        art = json.load(f)
    kept_idx = art.get("kept_feature_idx") or list(range(len(art["scaler_mean"])))
    return GoalsArtifact(
        kept_idx=kept_idx,
        mean=np.array(art["scaler_mean"], float),
        scale=np.array(art["scaler_scale"], float),
        coef=np.array(art["poisson_coef"], float),
        intercept=float(art["poisson_intercept"]),
        max_goals=int(art.get("max_goals", 10)),
        bp_c=float(art.get("bp_c", 0.0)),
        cal_onextwo=(art.get("calibration", {}) or {}).get("onextwo") \
                     or art.get("onextwo_cal") or None,
    )


# ---------------- Command ----------------

class Command(BaseCommand):
    help = "Price 1X2 and Double Chance using goals artifact (+ DB fallback)."

    def add_arguments(self, p: CommandParser) -> None:
        p.add_argument("--league-id", type=int, required=True)
        p.add_argument("--days", type=int, default=7)
        p.add_argument("--artifact", type=str, required=True)
        p.add_argument("--delete-first", action="store_true")
        p.add_argument("--write", action="store_true")
        p.add_argument("--use-calibrated", action="store_true")
        p.add_argument("--c", type=float, default=None, help="Override c (lam12=c*min). Default=artifact bp_c.")
        p.add_argument("--c-taper", type=float, default=0.60, help="exp(-c_taper*|μH-μA|) taper.")
        p.add_argument("--max-goals", type=int, default=None, help="Override max goals for grid.")
        p.add_argument("--audit", type=str, default=None)
        p.add_argument("--ids", nargs="+", type=int, default=None)
        p.add_argument("--verbose", action="store_true")

    def handle(self, *args, **opt):
        lg = int(opt["league_id"])
        days = int(opt["days"])
        art_path = str(opt["artifact"])
        delete_first = bool(opt["delete_first"])
        do_write = bool(opt["write"])
        use_cal = bool(opt["use_calibrated"])
        c_override = opt.get("c")
        c_taper = float(opt["c_taper"])
        max_goals_override = opt.get("max_goals")
        audit_path = opt.get("audit")
        ids = opt.get("ids")
        verbose = bool(opt["verbose"])

        # Load artifact
        art = _load_artifact(art_path)

        if max_goals_override is not None:
            art.max_goals = int(max_goals_override)
        if c_override is not None:
            art.bp_c = float(c_override)

        # Pull upcoming fixtures
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)

        qs = (
            MatchPrediction.objects
            .filter(league_id=lg,
                    kickoff_utc__gte=now,
                    kickoff_utc__lte=upto,
                    match__status__in=["NS","PST","TBD"])
            .select_related("match", "match__home", "match__away")
            .order_by("kickoff_utc")
        )
        if ids:
            qs = qs.filter(id__in=ids)

        # Optional cleanup
        if delete_first:
            n_deleted, _ = PredictedMarket.objects.filter(
                match__in=[r.match for r in qs],
                market_code__in=["ONE_X_TWO","DOUBLE_CHANCE"]
            ).delete()
            self.stdout.write(f"Deleted {n_deleted} existing ONE_X_TWO / DOUBLE_CHANCE rows.")

        # Prepare scaler vectors
        mean = np.array(art.mean, float)
        scale = np.array(art.scale, float)
        coef = np.array(art.coef, float)
        kept = np.array(art.kept_idx, int)

        # Sanity checks
        if not (len(mean) == len(scale) == len(coef)):
            raise RuntimeError("Artifact scaler/coef length mismatch.")
        if kept.size and kept.max() >= len(mean):
            raise RuntimeError("kept_feature_idx out of bounds for artifact vectors.")

        # Audit store
        audit_rows: List[List[Any]] = []
        header = [
            "mp_id","match_id","kickoff","home","away",
            "reason","mu_home","mu_away","c_eff",
            "p_home_raw","p_draw_raw","p_away_raw",
            "p_home_cal","p_draw_cal","p_away_cal"
        ]
        if audit_path:
            audit_rows.append(header)

        n_wrote = 0
        for mp in qs:
            m: Match = mp.match
            hn = getattr(m.home, "name", str(m.home_id))
            an = getattr(m.away, "name", str(m.away_id))

            # Try artifact path
            reason = "DB"
            mu_h = mu_a = None

            js = _get_match_stats10(m)
            if js is None:
                reason = "NOSTATS"
            else:
                # Build oriented features
                try:
                    xh, xa, names = _build_oriented_features(js)
                    # Align with artifact-kept subset
                    if kept.size:
                        xh = xh[kept]; xa = xa[kept]
                    # Standardize
                    xhs = (xh - mean) / np.where(scale == 0, 1.0, scale)
                    xas = (xa - mean) / np.where(scale == 0, 1.0, scale)
                    # Predict Poisson mean
                    mu_h = float(np.clip(np.exp(art.intercept + float(np.dot(xhs, coef))), 1e-6, 8.0))
                    mu_a = float(np.clip(np.exp(art.intercept + float(np.dot(xas, coef))), 1e-6, 8.0))
                    reason = "ART"
                except Exception as e:
                    # Feature incompatibility → fall back
                    reason = f"NOART:{type(e).__name__}"

            if reason != "ART":
                # DB fallback
                mu_h = float(np.clip(getattr(mp, "lambda_home", 1.20) or 1.20, 0.05, 8.0))
                mu_a = float(np.clip(getattr(mp, "lambda_away", 1.20) or 1.20, 0.05, 8.0))

            # Coupling with taper
            d = abs(mu_h - mu_a)
            taper = math.exp(-c_taper * d) if c_taper > 0 else 1.0
            lam12 = max(0.0, float(art.bp_c)) * float(min(mu_h, mu_a)) * float(taper)
            l1 = max(1e-9, mu_h - lam12)
            l2 = max(1e-9, mu_a - lam12)

            P = _bp_grid_from_components(l1, l2, lam12, art.max_goals)
            pH_raw, pD_raw, pA_raw = _one_x_two_from_grid(P)

            # optional isotonic calibration
            pH_cal = pH_raw; pD_cal = pD_raw; pA_cal = pA_raw
            if use_cal and art.cal_onextwo and reason == "ART":
                pH_cal = _iso_apply(pH_raw, art.cal_onextwo.get("home"))
                pD_cal = _iso_apply(pD_raw, art.cal_onextwo.get("draw"))
                pA_cal = _iso_apply(pA_raw, art.cal_onextwo.get("away"))
                S = pH_cal + pD_cal + pA_cal
                if S > 1e-12:
                    pH_cal /= S; pD_cal /= S; pA_cal /= S

            # DC from the same grid; calibrate by recompute from calibrated 1X2? We keep raw-grid DC.
            p1x, p12, px2 = _dc_from_grid(P)

            if verbose:
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an}\n"
                    f"  {reason} → μ=({mu_h:.2f},{mu_a:.2f}) | c={art.bp_c:.3f} taper={taper:.3f}\n"
                    f"  1X2_raw=({pH_raw:.3f},{pD_raw:.3f},{pA_raw:.3f})"
                    + (f"  1X2_cal=({pH_cal:.3f},{pD_cal:.3f},{pA_cal:.3f})" if use_cal and reason=="ART" else "")
                )

            if audit_path:
                audit_rows.append([
                    mp.id, m.id, m.kickoff_utc.isoformat(), hn, an,
                    reason, mu_h, mu_a, lam12,
                    pH_raw, pD_raw, pA_raw,
                    pH_cal if use_cal and reason=="ART" else "", 
                    pD_cal if use_cal and reason=="ART" else "", 
                    pA_cal if use_cal and reason=="ART" else "",
                ])

            if do_write:
                # ONE_X_TWO
                rows = [
                    ("ONE_X_TWO","HOME", pH_cal if (use_cal and reason=="ART") else pH_raw),
                    ("ONE_X_TWO","DRAW", pD_cal if (use_cal and reason=="ART") else pD_raw),
                    ("ONE_X_TWO","AWAY", pA_cal if (use_cal and reason=="ART") else pA_raw),
                    ("DOUBLE_CHANCE","HOME_OR_DRAW", p1x),
                    ("DOUBLE_CHANCE","HOME_OR_AWAY", p12),
                    ("DOUBLE_CHANCE","DRAW_OR_AWAY", px2),
                ]
                objs = []
                for market_code, spec, p in rows:
                    p = float(np.clip(p, 1e-9, 1.0))
                    fair = float(1.0 / p)
                    objs.append(PredictedMarket(
                        league_id=lg,
                        match=m,
                        kickoff_utc=m.kickoff_utc,
                        market_code=market_code,
                        specifier=spec,
                        p_model=p,
                        fair_odds=fair,
                        lambda_home=mu_h,
                        lambda_away=mu_a,
                    ))
                PredictedMarket.objects.bulk_create(objs)
                n_wrote += 1

        if audit_path:
            outp = Path(audit_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", newline="") as f:
                w = csv.writer(f)
                for row in audit_rows:
                    w.writerow(row)
            self.stdout.write(f"Wrote {len(audit_rows)-1} rows → {audit_path}")

        if do_write:
            self.stdout.write(self.style.SUCCESS(f"Wrote/updated {n_wrote} matches (1X2 + DC)."))
        else:
            self.stdout.write("Dry-run. Use --write to persist.")
