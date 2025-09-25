# matches/management/commands/refresh_goal_lambdas.py
from __future__ import annotations
import json, math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction
from matches.models import MatchPrediction, PredictedMarket

EPS = 1e-9

@dataclass
class GoalsArtifact:
    feature_names: Optional[List[str]]
    kept_feature_idx: Optional[List[int]]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float

def _load_art(path: str) -> GoalsArtifact:
    with open(path, "r") as f:
        art = json.load(f)
    return GoalsArtifact(
        feature_names=art.get("feature_names"),
        kept_feature_idx=art.get("kept_feature_idx"),
        scaler_mean=np.array(art["scaler_mean"], float),
        scaler_scale=np.array(art["scaler_scale"], float),
        coef=np.array(art["poisson_coef"], float),
        intercept=float(art["poisson_intercept"]),
    )

# feature builder import (same as trainer)
try:
    from prediction.train_goals import build_oriented_features  # type: ignore
except Exception:
    try:
        from train_goals import build_oriented_features  # type: ignore
    except Exception:
        build_oriented_features = None  # type: ignore

def _get_stats10(match_obj) -> Any:
    js = getattr(match_obj, "raw_result_json", None)
    if isinstance(js, dict) and "stats10_json" in js:
        return js["stats10_json"]
    return getattr(match_obj, "stats10_json", None)

def _vec_by_names(stats: Any, target_names: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if build_oriented_features is None:
        return None
    try:
        xh, xa, names = build_oriented_features({"stats10_json": stats})
        name2idx = {n:i for i,n in enumerate(names)}
        def pick(xside):
            vals = []
            for n in target_names:
                j = name2idx.get(n)
                vals.append(float(xside[j]) if j is not None else 0.0)
            return np.array(vals, float)
        return pick(xh), pick(xa)
    except Exception:
        return None

def _mus_from_art(stats: Any, art: GoalsArtifact) -> Tuple[Optional[float], Optional[float], str]:
    # prefer robust name alignment
    if art.feature_names:
        out = _vec_by_names(stats, art.feature_names)
        if out is not None:
            vh, va = out
            xs_h = (vh - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
            xs_a = (va - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
            mu_h = float(np.clip(math.exp(art.intercept + xs_h.dot(art.coef)), 1e-6, 8.0))
            mu_a = float(np.clip(math.exp(art.intercept + xs_a.dot(art.coef)), 1e-6, 8.0))
            return mu_h, mu_a, "artifact_names"
    # fallback to kept_feature_idx
    if art.kept_feature_idx and build_oriented_features is not None:
        try:
            xh, xa, _ = build_oriented_features({"stats10_json": stats})
            vh = np.array(xh, float)[art.kept_feature_idx]
            va = np.array(xa, float)[art.kept_feature_idx]
            if vh.shape[0] == art.scaler_mean.shape[0]:
                xs_h = (vh - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
                xs_a = (va - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
                mu_h = float(np.clip(math.exp(art.intercept + xs_h.dot(art.coef)), 1e-6, 8.0))
                mu_a = float(np.clip(math.exp(art.intercept + xs_a.dot(art.coef)), 1e-6, 8.0))
                return mu_h, mu_a, "artifact_idx"
        except Exception:
            pass
    return None, None, "artifact_failed"

class Command(BaseCommand):
    help = "Dry-run (or write) λ_home/λ_away in MatchPrediction from goals artifact. Fixes compressed/old DB lambdas."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=10)
        parser.add_argument("--artifact", type=str, required=True)
        parser.add_argument("--write", action="store_true", help="Persist changes.")
        parser.add_argument("--wipe-markets", action="store_true", help="Delete PredictedMarket for updated matches.")
        parser.add_argument("--verbose", action="store_true")

    def handle(self, *args, **opts):
        lg  = int(opts["league_id"]); days = int(opts["days"])
        art = _load_art(str(opts["artifact"]))
        write = bool(opts["write"]); wipe = bool(opts["wipe_markets"]); verbose = bool(opts["verbose"])

        now = datetime.now(timezone.utc); upto = now + timedelta(days=days)
        qs = (MatchPrediction.objects
              .filter(league_id=lg,
                      kickoff_utc__gte=now, kickoff_utc__lte=upto,
                      match__status__in=["NS","PST","TBD"])
              .select_related("match")
              .order_by("kickoff_utc"))

        if not qs.exists():
            self.stdout.write("No rows in range.")
            return

        diffs = 0; updates = []
        for mp in qs:
            m = mp.match
            s10 = _get_stats10(m)
            if s10 is None:
                if verbose:
                    self.stdout.write(f"{mp.id} | no stats10_json → skip (keeps DB λ)")
                continue

            muH, muA, src = _mus_from_art(s10, art)
            if muH is None or muA is None:
                if verbose:
                    self.stdout.write(f"{mp.id} | artifact alignment failed → skip")
                continue

            lh_db = float(getattr(mp, "lambda_home", 0.0) or 0.0)
            la_db = float(getattr(mp, "lambda_away", 0.0) or 0.0)
            dH = muH - lh_db; dA = muA - la_db
            if verbose:
                hn = getattr(m.home, "name", str(getattr(m, "home_id", "?")))
                an = getattr(m.away, "name", str(getattr(m, "away_id", "?")))
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an} | DB=({lh_db:.2f},{la_db:.2f}) → ART({muH:.2f},{muA:.2f}) [{src}] Δ=({dH:+.2f},{dA:+.2f})"
                )

            # mark for update if material diffs
            if abs(dH) > 1e-6 or abs(dA) > 1e-6:
                diffs += 1
                updates.append((mp.id, muH, muA))

        self.stdout.write(f"Would update {len(updates)} rows (diffs detected in {diffs}).")
        if not write:
            self.stdout.write("Dry-run only. Use --write to persist.")
            return

        with transaction.atomic():
            for mp_id, muH, muA in updates:
                mp = MatchPrediction.objects.select_for_update().get(id=mp_id)
                mp.lambda_home = float(muH)
                mp.lambda_away = float(muA)
                mp.save()
                if wipe:
                    PredictedMarket.objects.filter(match=mp.match).delete()

        self.stdout.write(self.style.SUCCESS(
            f"Updated {len(updates)} MatchPrediction rows"
            + (" and wiped related PredictedMarket rows" if wipe else "")
        ))
