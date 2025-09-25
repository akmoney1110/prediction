from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logsumexp
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from matches.models import Match, ModelVersion, PredictedMarket, FeaturesSnapshot

# the feature list used in training
CS_FEATURES: List[str] = [
    "h_gf10","h_ga10","h_gd10","h_sot10","h_conv10","h_sot_pct10","h_poss10",
    "h_corners_for10","h_cards_for10","h_clean_sheets10",
    "a_gf10","a_ga10","a_gd10","a_sot10","a_conv10","a_sot_pct10","a_poss10",
    "a_corners_for10","a_cards_for10","a_clean_sheets10",
    "h_home_gf10","a_away_gf10",
    "h_rest_days","a_rest_days","h_matches_14d","a_matches_14d",
    "d_gf10","d_sot10","d_rest_days","h_stats_missing","a_stats_missing",
]

def _predict_proba_statsmodels(res, X: np.ndarray) -> np.ndarray:
    """
    Stable softmax for MNLogit fitted result.
    res.params: [k_exog, J-1]; we add the baseline column of zeros.
    X is already add-constant'ed.
    """
    B = np.asarray(res.params, dtype=float)         # [k_exog, J-1]
    scores = X @ B                                  # [N, J-1]
    scores_full = np.concatenate([scores, np.zeros((scores.shape[0], 1))], axis=1)
    lse = logsumexp(scores_full, axis=1, keepdims=True)
    P = np.exp(scores_full - lse)
    # sanitize
    P = np.clip(P, 1e-9, 1 - 1e-9)
    P /= P.sum(axis=1, keepdims=True)
    return P

def _add_constant(Z: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)

def _load_latest_bundle(league_id: int):
    mv = (ModelVersion.objects
          .filter(kind="market:CS", league_id=league_id)
          .order_by("-trained_until", "-id")
          .first())
    if not mv:
        raise RuntimeError("No CS ModelVersion found. Run train_cs first.")
    bundle = joblib.load(mv.file_home)
    # Optional per-class isotonic calibrators:
    cal = None
    try:
        cal_json = mv.calibration_json or {}
        if isinstance(cal_json, str):
            cal_json = json.loads(cal_json)
        if cal_json.get("file"):
            cal = joblib.load(cal_json["file"])
    except Exception:
        cal = None
    return mv, bundle, (cal or {})

def _apply_per_class_calibration(P: np.ndarray, labels: List[str], calibrators: Dict[str, object]) -> np.ndarray:
    if not calibrators:
        return P
    P2 = P.copy()
    for j, lbl in enumerate(labels):
        ir = calibrators.get(lbl)
        if ir is None:
            continue
        try:
            pj = np.asarray(P[:, j], dtype=float)
            P2[:, j] = np.clip(ir.predict(pj), 1e-9, 1 - 1e-9)
        except Exception:
            pass
    # renormalize per row
    s = P2.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    P2 = P2 / s
    return P2

def _fetch_features_for_match(m, want: List[str]) -> Dict[str, float]:
    """
    Try several sources:
    1) Your builder (if available): matches.management.commands.predict_markets._build_feature_map_for_match
    2) Latest FeaturesSnapshot(ts_mode='T24' or any) -> features_json
    Fallback: zeros for missing keys.
    """
    fmap: Dict[str, float] = {}

    # 1) Try to import your existing builder
    if not fmap:
        try:
            from matches.management.commands.predict_markets import _build_feature_map_for_match  # type: ignore
            # The builder expects a manifest in your goals pipeline; we only need keys
            dummy_manifest = type("MF", (), {"feature_order": []})
            fmap = _build_feature_map_for_match(m, dummy_manifest)  # returns a big dict
        except Exception:
            fmap = {}

    # 2) Try FeaturesSnapshot
    if not fmap:
        snap = (FeaturesSnapshot.objects
                .filter(match=m)
                .order_by("-created_at")
                .first())
        if snap and isinstance(snap.features_json, dict):
            fmap = dict(snap.features_json)

    # Fallback: zeros; keep only required keys
    out = {}
    for k in want:
        v = fmap.get(k, 0.0)
        try:
            out[k] = float(v)
        except Exception:
            out[k] = 0.0
    return out

class Command(BaseCommand):
    help = "Score Correct Score (CS) probabilities for upcoming fixtures and write PredictedMarket('CS', 'i-j')."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--top-n", type=int, default=12,
                            help="Only store top-N CS outcomes per match (keeps rows manageable). Use 0 for all.")
        parser.add_argument("--ts-mode", type=str, default="T24",
                            help="If using FeaturesSnapshot, prefer this ts mode (informational only).")

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        days = int(opts["days"])
        top_n = int(opts["top_n"])

        mv, bundle, calibrators = _load_latest_bundle(league_id)

        feat_order: List[str] = list(bundle["feature_order"])
        mean = np.asarray(bundle["zscaler_mean"], dtype=float)
        std  = np.asarray(bundle["zscaler_std"], dtype=float)
        clip = float(bundle.get("z_clip", 6.0))
        present_codes: List[int] = list(bundle["present_codes"])
        full_labels: List[str] = list(bundle["full_labels"])
        present_labels: List[str] = [full_labels[c] for c in present_codes]
        res = bundle["model"]

        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)
        fixtures = (Match.objects
                    .filter(league_id=league_id,
                            kickoff_utc__gte=now, kickoff_utc__lte=upto)
                    .exclude(status__in=["FT","AET","PEN"])
                    .order_by("kickoff_utc"))

        if not fixtures.exists():
            self.stdout.write("No upcoming fixtures in window.")
            return

        wrote = 0
        for m in fixtures:
            # Build row in the exact training feature order
            fmap = _fetch_features_for_match(m, feat_order)
            x = np.array([fmap.get(k, 0.0) for k in feat_order], dtype=float)[None, :]  # [1, F]

            # z-scale + clip in z-space
            std_safe = np.where(std > 1e-12, std, 1.0)
            Z = (x - mean) / std_safe
            if np.isfinite(clip) and clip > 0:
                Z = np.clip(Z, -clip, clip)

            X = _add_constant(Z)  # [1, F+1] to match MNLogit exog with const
            P_compact = _predict_proba_statsmodels(res, X)  # [1, J_present]

            # optional per-class isotonic calibration
            P_compact = _apply_per_class_calibration(P_compact, present_labels, calibrators)

            # Map compact probs back to full label grid indices
            P_full = np.zeros((1, len(full_labels)), dtype=float)
            for j, code in enumerate(present_codes):
                P_full[0, code] = P_compact[0, j]

            # Decide which outcomes to store
            probs = P_full[0]
            order = np.argsort(-probs)
            if top_n and top_n > 0:
                order = order[:top_n]

            for idx in order:
                lbl = full_labels[idx]   # e.g. "2-1"
                p   = float(np.clip(probs[idx], 1e-9, 1 - 1e-9))
                PredictedMarket.objects.update_or_create(
                    match=m,
                    market_code="CS",
                    specifier=lbl,
                    defaults={
                        "league": m.league,
                        "kickoff_utc": m.kickoff_utc,
                        "p_model": p,
                        "fair_odds": float(1.0 / p),
                        "lambda_home": None,
                        "lambda_away": None,
                    },
                )
                wrote += 1

        self.stdout.write(self.style.SUCCESS(
            f"Wrote/updated {wrote} CS rows for league {league_id}"
        ))
