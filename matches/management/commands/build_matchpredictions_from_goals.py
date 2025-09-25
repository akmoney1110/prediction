# -*- coding: utf-8 -*-
"""
Rebuild MatchPrediction rows (λ_home, λ_away) for upcoming fixtures
directly from the goals artifact, using the same oriented feature pipeline
you already validated.

This fixes "Spurs/Wolves" and similar flips at the SOURCE by writing
correctly oriented lambdas tied to the Match home/away FKs.

Usage:
  python manage.py build_matchpredictions_from_goals \
      --artifacts artifacts/goals/artifacts.goals.json \
      --leagues 39,61 \
      --from-date 2025-09-01 --to-date 2025-10-01 \
      --min-coverage 0.10 --debug
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from django.core.management.base import BaseCommand, CommandParser
from django.db.models import Q

from matches.models import (
    Match,
    MatchPrediction,
    MLTrainingMatch,   # same model your working pipeline uses for features
)

EPS = 1e-9


# --------------------------- tiny numeric helpers ---------------------------

def _tofloat(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


# --------------------- synth fallback for stats10_json ----------------------

def _synth_js_from_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """If stats10_json is missing, build a minimal structure from MLTrainingMatch columns."""
    def _nz(v, d=0.0):
        try:
            return float(v) if v is not None else float(d)
        except Exception:
            return float(d)

    return {
        "shots": {
            "home": {
                "gf": _nz(row.get("h_gf10")),
                "ga": _nz(row.get("h_ga10")),
                "sot": _nz(row.get("h_sot10")),
                "poss": _nz(row.get("h_poss10")),
                "corners": _nz(row.get("h_corners_for10")),
                "cards": _nz(row.get("h_cards_for10")),
            },
            "away": {
                "gf": _nz(row.get("a_gf10")),
                "ga": _nz(row.get("a_ga10")),
                "sot": _nz(row.get("a_sot10")),
                "poss": _nz(row.get("a_poss10")),
                "corners": _nz(row.get("a_corners_for10")),
                "cards": _nz(row.get("a_cards_for10")),
            },
        },
        "derived": {"home": {}, "away": {}},
        "allowed": {"home": {}, "away": {}},
        "situational": {
            "h_rest_days": _nz(row.get("h_rest_days")),
            "a_rest_days": _nz(row.get("a_rest_days")),
            "h_matches_7d": 0.0,
            "a_matches_7d": 0.0,
            "h_matches_14d": _nz(row.get("h_matches_14d")),
            "a_matches_14d": _nz(row.get("a_matches_14d")),
        },
        "elo": {"home": 0.0, "away": 0.0},
        "gelo": {"exp_home_goals": None, "exp_away_goals": None},
    }


# --------------------------- oriented features ------------------------------

def _build_oriented_features(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return (xh, xa, names) such that:
      - xh is the feature vector for the HOME team perspective
      - xa is the feature vector for the AWAY team perspective
    """
    js = row.get("stats10_json") or {}
    if isinstance(js, str):
        try:
            js = json.loads(js)
        except Exception:
            js = {}
    if not js:
        js = _synth_js_from_columns(row)

    shots = js.get("shots") or {}
    shots_h = shots.get("home", {}) or {}
    shots_a = shots.get("away", {}) or {}

    drv = js.get("derived") or {}
    drv_h = drv.get("home", {}) or {}
    drv_a = drv.get("away", {}) or {}

    allowed = js.get("allowed") or {}
    allow_h = allowed.get("home", {}) or {}
    allow_a = allowed.get("away", {}) or {}

    situ = js.get("situational") or {}
    h_rest = _tofloat(situ.get("h_rest_days", 0.0))
    a_rest = _tofloat(situ.get("a_rest_days", 0.0))
    h_m7 = _tofloat(situ.get("h_matches_7d", 0.0))
    a_m7 = _tofloat(situ.get("a_matches_7d", 0.0))
    h_m14 = _tofloat(situ.get("h_matches_14d", 0.0))
    a_m14 = _tofloat(situ.get("a_matches_14d", 0.0))

    elo = js.get("elo") or {}
    elo_h = _tofloat(elo.get("home", 0.0))
    elo_a = _tofloat(elo.get("away", 0.0))

    gelo = js.get("gelo") or {}
    g_h = gelo.get("exp_home_goals", None)
    g_a = gelo.get("exp_away_goals", None)
    gdiff = 0.0
    if g_h is not None and g_a is not None:
        try:
            gdiff = float(g_h) - float(g_a)
        except Exception:
            gdiff = 0.0

    def g(d, k, default=0.0):
        return _tofloat(d.get(k, default), default)

    base_keys = ["gf", "ga", "shots", "sot", "shots_in_box", "xg", "poss", "corners"]
    drv_keys = ["xg_per_shot", "sot_rate", "box_share", "save_rate", "xg_diff"]
    allow_keys = ["shots_allowed", "sot_allowed", "shots_in_box_allowed", "xga"]

    feats_h: List[float] = []
    feats_a: List[float] = []
    names: List[str] = []

    # team vs opp (home POV), then mirror for away POV
    for k in base_keys:
        names.append(f"team_{k}")
        feats_h.append(g(shots_h, k))
        feats_a.append(g(shots_a, k))
    for k in base_keys:
        names.append(f"opp_{k}")
        feats_h.append(g(shots_a, k))
        feats_a.append(g(shots_h, k))

    for k in drv_keys:
        names.append(f"teamdrv_{k}")
        feats_h.append(g(drv_h, k))
        feats_a.append(g(drv_a, k))
    for k in drv_keys:
        names.append(f"oppdrv_{k}")
        feats_h.append(g(drv_a, k))
        feats_a.append(g(drv_h, k))

    for k in allow_keys:
        names.append(f"team_allowed_{k}")
        feats_h.append(g(allow_h, k))
        feats_a.append(g(allow_a, k))
    for k in allow_keys:
        names.append(f"opp_allowed_{k}")
        feats_h.append(g(allow_a, k))
        feats_a.append(g(allow_h, k))

    # anti-symmetric diffs
    for k in base_keys:
        names.append(f"diff_{k}")
        v = g(shots_h, k) - g(shots_a, k)
        feats_h.append(v)
        feats_a.append(-v)
    for k in drv_keys:
        names.append(f"diffdrv_{k}")
        v = g(drv_h, k) - g(drv_a, k)
        feats_h.append(v)
        feats_a.append(-v)
    for k in allow_keys:
        names.append(f"diff_allowed_{k}")
        v = g(allow_h, k) - g(allow_a, k)
        feats_h.append(v)
        feats_a.append(-v)

    # situational diffs
    for name, diff in [
        ("rest_days_diff", h_rest - a_rest),
        ("matches_7d_diff", h_m7 - a_m7),
        ("matches_14d_diff", h_m14 - a_m14),
    ]:
        names.append(name)
        feats_h.append(diff)
        feats_a.append(-diff)

    # elo / gelo diffs
    names.append("elo_diff")
    feats_h.append(elo_h - elo_a)
    feats_a.append(elo_a - elo_h)

    names.append("gelo_mu_diff")
    feats_h.append(gdiff)
    feats_a.append(-gdiff)

    # venue flag
    names.append("home_flag")
    feats_h.append(+1.0)
    feats_a.append(-1.0)

    return np.asarray(feats_h, float), np.asarray(feats_a, float), names


# --------------------- align + standardize to artifact ----------------------

def _align_with_ref(x_vec: np.ndarray, names: List[str], ref_names: List[str]) -> Tuple[np.ndarray, float, float]:
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros(len(ref_names), dtype=float)
    present = 0
    for j, name in enumerate(ref_names):
        i = idx.get(name, None)
        if i is not None:
            out[j] = float(x_vec[i])
            present += 1
    coverage = present / max(1, len(ref_names))
    zero_frac = float(np.mean(out == 0.0))
    return out, coverage, zero_frac


def _standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_safe = np.where(scale == 0.0, 1.0, scale)
    return (x - mean) / scale_safe


# ------------------------------- command ------------------------------------

class Command(BaseCommand):
    help = "Write correctly oriented λ_home/λ_away into MatchPrediction from goals artifact (fixes swapped source)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--artifacts", type=str, required=True, help="Path to artifacts.goals.json")
        parser.add_argument("--leagues", type=str, required=True, help="e.g. '39' or '39,61,140'")
        parser.add_argument("--from-date", type=str, default=None, help="inclusive ISO date (YYYY-MM-DD)")
        parser.add_argument("--to-date", type=str, default=None, help="exclusive ISO date (YYYY-MM-DD)")
        parser.add_argument("--min-coverage", type=float, default=0.10, help="skip if < this fraction of features align")
        parser.add_argument("--max-goals", type=int, default=None, help="(optional) cap for internal grid sanity")
        parser.add_argument("--debug", action="store_true")

    def handle(self, *args, **opts):
        # --- load artifact
        art_path = Path(opts["artifacts"])
        with open(art_path, "r") as f:
            art = json.load(f)

        ref_names: List[str] = art["feature_names"]
        mean = np.asarray(art["scaler_mean"], dtype=float)
        scale = np.asarray(art["scaler_scale"], dtype=float)
        coef = np.asarray(art["poisson_coef"], dtype=float)
        intercept = float(art["poisson_intercept"])

        # (optional) for sanity checks; not needed to compute μ
        max_goals = int(opts["max_goals"]) if opts["max_goals"] is not None else int(art.get("max_goals", 8))

        # sanity on vector sizes
        n = len(ref_names)
        if not (n == len(mean) == len(scale) == len(coef)):
            raise RuntimeError("Artifact arrays mismatch (feature_names / scaler / coef).")

        leagues = [int(x) for x in str(opts["leagues"]).split(",") if x.strip()]
        min_cov = float(opts["min_coverage"])

        # --- upcoming fixtures from MLTrainingMatch (same source you trained on)
        qs = (
            MLTrainingMatch.objects
            .filter(league_id__in=leagues)
            .filter(Q(y_home_goals_90=None) | Q(y_away_goals_90=None))
            .order_by("kickoff_utc")
            .values(
                "league_id", "season", "kickoff_utc",
                "home_team_id", "away_team_id", "stats10_json",
                # fallback columns used by _synth_js_from_columns
                "h_gf10", "a_gf10", "h_ga10", "a_ga10",
                "h_sot10", "a_sot10",
                "h_poss10", "a_poss10",
                "h_corners_for10", "a_corners_for10",
                "h_cards_for10", "a_cards_for10",
                "h_rest_days", "a_rest_days",
                "h_matches_14d", "a_matches_14d",
            )
        )
        if opts["from_date"]:
            qs = qs.filter(kickoff_utc__date__gte=opts["from_date"])
        if opts["to_date"]:
            qs = qs.filter(kickoff_utc__date__lt=opts["to_date"])

        rows = list(qs)
        for r in rows:
            js = r.get("stats10_json")
            if isinstance(js, str):
                try:
                    r["stats10_json"] = json.loads(js)
                except Exception:
                    r["stats10_json"] = {}

        if not rows:
            self.stdout.write(self.style.WARNING("No upcoming MLTrainingMatch rows in window."))
            return

        wrote = 0
        skipped_cov = 0
        # Loop: compute μ_home/μ_away using oriented features, then write to MatchPrediction
        for r in rows:
            xh, xa, names = _build_oriented_features(r)
            xh_aligned, cov_h, zf_h = _align_with_ref(xh, names, ref_names)
            xa_aligned, cov_a, zf_a = _align_with_ref(xa, names, ref_names)
            cov = 0.5 * (cov_h + cov_a)

            if cov < min_cov:
                skipped_cov += 1
                if opts["debug"]:
                    self.stdout.write(f"SKIP low coverage: {r['home_team_id']} vs {r['away_team_id']} @ {r['kickoff_utc']} cov={cov:.2f}")
                continue

            xh_s = _standardize(xh_aligned, mean, scale)
            xa_s = _standardize(xa_aligned, mean, scale)
            mu_h = math.exp(float(intercept + np.dot(xh_s, coef)))
            mu_a = math.exp(float(intercept + np.dot(xa_s, coef)))
            mu_h = float(np.clip(mu_h, 1e-6, 8.0))
            mu_a = float(np.clip(mu_a, 1e-6, 8.0))

            # Map to actual Match row and write safely
            m = (
                Match.objects
                .filter(
                    league_id=r["league_id"],
                    home_id=r["home_team_id"],   # NOTE: Match has FKs named 'home'/'away'
                    away_id=r["away_team_id"],
                    kickoff_utc=r["kickoff_utc"],
                )
                .select_related("home", "away")
                .first()
            )
            if not m:
                # Try loose-time match (within 2 hours) if exact timestamp differs after ingestion
                m = (
                    Match.objects
                    .filter(
                        league_id=r["league_id"],
                        home_id=r["home_team_id"],
                        away_id=r["away_team_id"],
                        kickoff_utc__gte=r["kickoff_utc"] - np.timedelta64(2, "h"),
                        kickoff_utc__lte=r["kickoff_utc"] + np.timedelta64(2, "h"),
                    )
                    .select_related("home", "away")
                    .order_by("kickoff_utc")
                    .first()
                )
            if not m:
                if opts["debug"]:
                    self.stdout.write(f"MISS Match row for {r['league_id']} {r['home_team_id']} vs {r['away_team_id']} @ {r['kickoff_utc']}")
                continue

            # Upsert MatchPrediction with CORRECT orientation
            obj, _created = MatchPrediction.objects.update_or_create(
                match=m,
                defaults={
                    "league_id": r["league_id"],
                    "kickoff_utc": r["kickoff_utc"],
                    "lambda_home": mu_h,
                    "lambda_away": mu_a,
                    # If you also keep running means, variance, etc., set them here as needed.
                },
            )
            wrote += 1

            if opts["debug"]:
                hn = getattr(m.home, "name", str(m.home_id))
                an = getattr(m.away, "name", str(m.away_id))
                self.stdout.write(
                    f"{m.id} | {hn} vs {an} | μ=({mu_h:.2f},{mu_a:.2f}) cov={cov:.2f} z0={0.5*(zf_h+zf_a):.2f}"
                )

        self.stdout.write(self.style.SUCCESS(
            f"Wrote/updated {wrote} MatchPrediction rows | skipped_low_cov={skipped_cov}"
        ))
