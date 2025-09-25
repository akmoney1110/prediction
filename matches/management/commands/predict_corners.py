# prediction/matches/management/commands/predict_corners.py

import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand
from matches.models import Match, ModelVersion, MatchPrediction

def _finite(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else float(d)
    except Exception:
        return float(d)

def _load_meta(mv: ModelVersion):
    """
    Load the corners meta (expected feature order).
    Tries mv.calibration_json['meta_file'] first; otherwise derives from file_home; then globs.
    """
    meta_path = None

    # 1) calibration_json path
    cj = getattr(mv, "calibration_json", None)
    if isinstance(cj, dict):
        meta_path = cj.get("meta_file")

    # 2) derive from model filename timestamp
    if not meta_path:
        p = Path(mv.file_home)
        # expect gbdt_corners_home_{league}_{TS}.joblib
        stem = p.stem
        parts = stem.split("_")
        TS = parts[-1] if parts else None
        if TS:
            candidate = p.parent / f"corners_meta_{mv.league_id}_{TS}.json"
            if candidate.exists():
                meta_path = str(candidate)

    # 3) last-resort glob in same dir
    if not meta_path:
        p = Path(mv.file_home)
        cand = sorted(p.parent.glob(f"corners_meta_{mv.league_id}_*.json"))
        if cand:
            meta_path = str(cand[-1])

    if not meta_path:
        raise RuntimeError("Corners meta file not found. Re-train saving expected_features.")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    feats = meta.get("expected_features")
    if not feats or not isinstance(feats, list):
        raise RuntimeError("Corners meta missing expected_features.")
    return feats


def _build_feature_row(m: Match, feature_names: list[str]) -> np.ndarray:
    """
    Build one row in the exact order of `feature_names`.
    The last two names must be 'home_team_id' and 'away_team_id' (ints).
    Everything else is numeric floats.
    """
    row = []
    for name in feature_names:
        if name == "home_team_id":
            row.append(int(getattr(m, "home_id")))
        elif name == "away_team_id":
            row.append(int(getattr(m, "away_id")))
        else:
            row.append(_finite(getattr(m, name, 0.0), 0.0))
    return np.array(row, dtype=float)

class Command(BaseCommand):
    help = "Predict corner lambdas (home/away) for upcoming fixtures using trained corners model."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        days = int(opts["days"])

        # 1) Load latest corners model + meta
        mv = (ModelVersion.objects
              .filter(kind="corners", league_id=league_id)
              .order_by("-trained_until", "-id")
              .first())
        if not mv:
            self.stderr.write("No ModelVersion(kind='corners') found. Run train_corners first.")
            return

        home_model = joblib.load(mv.file_home)
        away_model = joblib.load(mv.file_away)
        feature_names = _load_meta(mv)   # ordered list; len must match pipeline expectation

        # 2) Upcoming fixtures
        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)
        fixtures = (Match.objects
                    .filter(league_id=league_id, kickoff_utc__gte=now, kickoff_utc__lte=upto)
                    .exclude(status__in=["FT","AET","PEN"])
                    .order_by("kickoff_utc"))

        if not fixtures.exists():
            self.stdout.write("No fixtures in window.")
            return

        n = 0
        for m in fixtures:
            row = _build_feature_row(m, feature_names)   # length == model’s expected raw input length
            X = row.reshape(1, -1)

            lam_h = float(home_model.predict(X)[0])
            lam_a = float(away_model.predict(X)[0])

            MatchPrediction.objects.update_or_create(
                match=m,
                defaults={
                    "league_id": m.league_id,
                    "season": m.season,
                    "kickoff_utc": m.kickoff_utc,
                    "lambda_corners_home": lam_h,
                    "lambda_corners_away": lam_a,
                }
            )
            n += 1

        self.stdout.write(self.style.SUCCESS(
            f"Wrote corner λ for {n} fixtures (league {league_id})"
        ))
