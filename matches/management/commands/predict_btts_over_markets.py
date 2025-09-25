# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Price BTTS and Totals (Over/Under) markets using the goals artifact (train_goals).

- Prefers μ from artifact + Match.stats10_json (or Match.raw_result_json["stats10_json"])
- Falls back to DB lambdas (MatchPrediction.lambda_home / lambda_away) if needed
- Uses artifact's BP coupling (c * min(μH, μA)) with optional taper
- Optional isotonic calibration:
    * BTTS: "calibration.btts" or top-level "btts_cal"
    * Over 1.5: "calibration.over15" or top-level "over_cal"
- Writes PredictedMarket rows:
    * market_code="BTTS", specifier in {"YES","NO"}
    * market_code="GOALS_TOTALS", specifier like "OVER_1.5","UNDER_1.5", etc.
      (Adjust names if your app expects different codes.)

Example:
  python manage.py predict_btts_totals_markets \
    --league-id 39 \
    --days 7 \
    --artifact artifacts/goals/artifacts.goals.json \
    --lines 0.5,1.5,2.5,3.5 \
    --delete-first \
    --use-calibrated \
    --write \
    --audit /tmp/audit_btts_totals.csv \
    --verbose
"""
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from django.core.management.base import BaseCommand, CommandParser

from matches.models import MatchPrediction, PredictedMarket, Match


# ---------------- helpers ----------------

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

def _totals_over_prob(P: np.ndarray, line: float) -> float:
    H, A = np.indices(P.shape)
    total = H + A
    return float(P[(total > line)].sum())

def _btts_yes_prob(P: np.ndarray) -> float:
    H, A = np.indices(P.shape)
    return float(P[(H > 0) & (A > 0)].sum())


# --------------- robust oriented features (same as trainer) ---------------

TEAM_KEYS    = ["gf", "ga", "cs", "shots", "sot", "shots_in_box", "xg", "conv", "sot_pct", "poss", "corners", "cards"]
DERIVED_KEYS = ["xg_per_shot", "sot_rate", "box_share", "save_rate", "xg_diff"]
ALLOWED_KEYS = ["shots_allowed", "sot_allowed", "shots_in_box_allowed", "xga"]
SITU_KEYS    = ["h_rest_days", "a_rest_days", "h_matches_14d", "a_matches_14d", "h_matches_7d", "a_matches_7d"]
CROSS_KEYS   = ["home_xgps_minus_away_sot_allow_rate", "away_xgps_minus_home_sot_allow_rate"]

def _build_oriented_features(stats10_json: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    js = stats10_json or {}

    shots     = js.get("shots") or {}
    shots_opp = js.get("shots_opp") or {}
    if isinstance(shots, dict) and ("home" in shots or "away" in shots):
        sh_home = shots.get("home") or {}
        sh_away = shots.get("away") or {}
    else:
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


# ---------------- artifact ----------------

@dataclass
class GoalsArtifact:
    kept_idx: List[int]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    max_goals: int
    bp_c: float
    cal_over15: Optional[Dict[str, List[float]]]
    cal_btts: Optional[Dict[str, List[float]]]

def _load_artifact(path: str) -> GoalsArtifact:
    with open(path, "r") as f:
        art = json.load(f)
    kept_idx = art.get("kept_feature_idx") or list(range(len(art["scaler_mean"])))
    # calibration lookups (both nested and legacy top-level)
    cal = art.get("calibration", {}) or {}
    over15 = cal.get("over15") or art.get("over_cal")
    btts = cal.get("btts") or art.get("btts_cal")
    return GoalsArtifact(
        kept_idx=kept_idx,
        mean=np.array(art["scaler_mean"], float),
        scale=np.array(art["scaler_scale"], float),
        coef=np.array(art["poisson_coef"], float),
        intercept=float(art["poisson_intercept"]),
        max_goals=int(art.get("max_goals", 10)),
        bp_c=float(art.get("bp_c", 0.0)),
        cal_over15=over15,
        cal_btts=btts,
    )


# ---------------- command ----------------

class Command(BaseCommand):
    help = "Price BTTS and Totals (Over/Under) using goals artifact (+ DB fallback)."

    def add_arguments(self, p: CommandParser) -> None:
        p.add_argument("--league-id", type=int, required=True)
        p.add_argument("--days", type=int, default=7)
        p.add_argument("--artifact", type=str, required=True)
        p.add_argument("--lines", type=str, default="", help="Comma-separated totals lines (e.g. '0.5,1.5,2.5'). If empty, use artifact config if present, else defaults.")
        p.add_argument("--delete-first", action="store_true")
        p.add_argument("--write", action="store_true")
        p.add_argument("--use-calibrated", action="store_true", help="Apply isotonic for BTTS and Over(1.5) when available from artifact.")
        p.add_argument("--c", type=float, default=None, help="Override c; default=artifact bp_c")
        p.add_argument("--c-taper", type=float, default=0.60, help="exp(-c_taper*|μH-μA|) taper.")
        p.add_argument("--max-goals", type=int, default=None, help="Override max goals grid.")
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

        art = _load_artifact(art_path)
        if c_override is not None:
            art.bp_c = float(c_override)
        if max_goals_override is not None:
            art.max_goals = int(max_goals_override)

        # totals lines
        lines_cli = [float(x) for x in str(opt.get("lines","")).split(",") if str(x).strip()]
        if lines_cli:
            totals_lines = lines_cli
        else:
            cfg = (_safe_json_loads(open(art_path).read()) or {}).get("config", {}) if art_path else {}
            totals_lines = cfg.get("totals_lines") or [0.5, 1.5, 2.5, 3.5]

        # fetch fixtures
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

        # cleanup existing
        if delete_first:
            n_deleted, _ = PredictedMarket.objects.filter(
                match__in=[r.match for r in qs],
                market_code__in=["BTTS","GOALS_TOTALS"]
            ).delete()
            self.stdout.write(f"Deleted {n_deleted} existing BTTS / GOALS_TOTALS rows.")

        mean = np.array(art.mean, float)
        scale = np.array(art.scale, float)
        coef = np.array(art.coef, float)
        kept = np.array(art.kept_idx, int)
        if not (len(mean) == len(scale) == len(coef)):
            raise RuntimeError("Artifact scaler/coef length mismatch.")
        if kept.size and kept.max() >= len(mean):
            raise RuntimeError("kept_feature_idx out of bounds.")

        # audit
        audit_rows: List[List[Any]] = []
        if audit_path:
            audit_rows.append([
                "mp_id","match_id","kickoff","home","away","reason",
                "mu_home","mu_away","lam12","p_btts_yes_raw","p_btts_yes_cal",
                *[f"p_over_{L:g}" for L in totals_lines],
                "notes"
            ])

        n_wrote = 0
        for mp in qs:
            m: Match = mp.match
            hn = getattr(m.home, "name", str(m.home_id))
            an = getattr(m.away, "name", str(m.away_id))

            reason = "DB"
            mu_h = mu_a = None

            js = _get_match_stats10(m)
            if js is None:
                reason = "NOSTATS"
            else:
                try:
                    xh, xa, _ = _build_oriented_features(js)
                    if kept.size:
                        xh = xh[kept]; xa = xa[kept]
                    xs_h = (xh - mean) / np.where(scale == 0, 1.0, scale)
                    xs_a = (xa - mean) / np.where(scale == 0, 1.0, scale)
                    mu_h = float(np.clip(np.exp(art.intercept + float(np.dot(xs_h, coef))), 1e-6, 8.0))
                    mu_a = float(np.clip(np.exp(art.intercept + float(np.dot(xs_a, coef))), 1e-6, 8.0))
                    reason = "ART"
                except Exception as e:
                    reason = f"NOART:{type(e).__name__}"

            if reason != "ART":
                mu_h = float(np.clip(getattr(mp, "lambda_home", 1.20) or 1.20, 0.05, 8.0))
                mu_a = float(np.clip(getattr(mp, "lambda_away", 1.20) or 1.20, 0.05, 8.0))

            d = abs(mu_h - mu_a)
            taper = math.exp(-c_taper * d) if c_taper > 0 else 1.0
            lam12 = max(0.0, float(art.bp_c)) * float(min(mu_h, mu_a)) * float(taper)
            l1 = max(1e-9, mu_h - lam12)
            l2 = max(1e-9, mu_a - lam12)
            P = _bp_grid_from_components(l1, l2, lam12, art.max_goals)

            # BTTS
            p_btts_yes_raw = _btts_yes_prob(P)
            p_btts_yes_cal = p_btts_yes_raw
            if use_cal and art.cal_btts and reason == "ART":
                p_btts_yes_cal = _iso_apply(p_btts_yes_raw, art.cal_btts)
            p_btts_yes = p_btts_yes_cal if (use_cal and reason == "ART") else p_btts_yes_raw
            p_btts_no  = float(1.0 - p_btts_yes)

            # Totals
            p_over_map = {}
            for L in totals_lines:
                pov = _totals_over_prob(P, float(L))
                # Only O1.5 has calibrator; apply if present
                if use_cal and art.cal_over15 and abs(L - 1.5) < 1e-6 and reason == "ART":
                    pov = _iso_apply(pov, art.cal_over15)
                p_over_map[float(L)] = float(np.clip(pov, 1e-9, 1.0-1e-9))

            if verbose:
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an}\n"
                    f"  {reason} → μ=({mu_h:.2f},{mu_a:.2f}) | c={art.bp_c:.3f} taper={taper:.3f}\n"
                    f"  BTTS_yes={p_btts_yes:.3f}  "
                    + " ".join([f"O{L:g}={p_over_map[L]:.3f}" for L in totals_lines])
                )

            if audit_path:
                audit_rows.append([
                    mp.id, m.id, m.kickoff_utc.isoformat(), hn, an, reason,
                    mu_h, mu_a, lam12, _btts_yes_prob(P), 
                    (p_btts_yes if (use_cal and reason=="ART") else ""),
                    *[p_over_map[L] for L in totals_lines],
                    ""
                ])

            if do_write:
                objs = []

                # BTTS YES/NO
                for spec, p in (("YES", p_btts_yes), ("NO", p_btts_no)):
                    p = float(np.clip(p, 1e-9, 1.0))
                    objs.append(PredictedMarket(
                        league_id=lg,
                        match=m,
                        kickoff_utc=m.kickoff_utc,
                        market_code="BTTS",
                        specifier=spec,
                        p_model=p,
                        fair_odds=float(1.0/p),
                        lambda_home=mu_h,
                        lambda_away=mu_a,
                    ))

                # Totals Over/Under for each line
                for L in totals_lines:
                    pov = p_over_map[L]
                    pun = float(1.0 - pov)
                    objs.append(PredictedMarket(
                        league_id=lg, match=m, kickoff_utc=m.kickoff_utc,
                        market_code="GOALS_TOTALS", specifier=f"OVER_{L:g}",
                        p_model=float(np.clip(pov, 1e-9, 1.0-1e-9)),
                        fair_odds=float(1.0/np.clip(pov, 1e-9, 1.0)),
                        lambda_home=mu_h, lambda_away=mu_a,
                    ))
                    objs.append(PredictedMarket(
                        league_id=lg, match=m, kickoff_utc=m.kickoff_utc,
                        market_code="GOALS_TOTALS", specifier=f"UNDER_{L:g}",
                        p_model=float(np.clip(pun, 1e-9, 1.0-1e-9)),
                        fair_odds=float(1.0/np.clip(pun, 1e-9, 1.0)),
                        lambda_home=mu_h, lambda_away=mu_a,
                    ))

                PredictedMarket.objects.bulk_create(objs)
                n_wrote += 1

        # audit write
        if audit_path:
            outp = Path(audit_path); outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", newline="") as f:
                w = csv.writer(f)
                for row in audit_rows:
                    w.writerow(row)
            self.stdout.write(f"Wrote {len(audit_rows)-1} rows → {audit_path}")

        if do_write:
            self.stdout.write(self.style.SUCCESS(f"Wrote/updated {n_wrote} matches (BTTS + GOALS_TOTALS)."))
        else:
            self.stdout.write("Dry-run. Use --write to persist.")
