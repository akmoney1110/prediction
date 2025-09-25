# prediction/matches/management/commands/predict_cards.py
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand

from matches.models import Match, ModelVersion, MatchPrediction

PRED_CARDS_VERSION = "cards-predict v2 (ts-bound meta + snapshots)"


# ------------------------- utils -------------------------

def _finite(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else float(d)
    except Exception:
        return float(d)


def _load_meta_for_model(mv: ModelVersion, family: str) -> list[str]:
    """
    Bind the meta to THIS model's timestamp.
    Example model filename: rf_cards_home_L39_20250828T134250Z.joblib
    -> meta: cards_meta_39_20250828T134250Z.json
    """
    model_path = Path(mv.file_home)
    stem = model_path.stem  # e.g., rf_cards_home_L39_20250828T134250Z
    parts = stem.split("_")
    ts = parts[-1] if parts else None
    meta_path = model_path.parent / f"{family}_meta_{mv.league_id}_{ts}.json"
    if not meta_path.exists():
        # fallback to latest matching meta in same dir
        cands = sorted(model_path.parent.glob(f"{family}_meta_{mv.league_id}_*.json"))
        if not cands:
            raise RuntimeError(f"{family.capitalize()} meta file not found for model timestamp.")
        meta_path = cands[-1]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    raw = meta.get("expected_features") or []
    feats, seen = [], set()
    for name in raw:
        s = str(name).strip()
        if s and s not in seen:
            seen.add(s)
            feats.append(s)

    if not feats:
        raise RuntimeError(f"{family.capitalize()} meta missing expected_features.")
    return feats


def _latest_snapshot_dict(m: Match) -> dict:
    """
    Get the most recent engineered-feature snapshot for a match, if any.
    Supports common attribute names: data/json/payload/content/blob and stringified JSON.
    """
    rel = getattr(m, "feature_snapshots", None)
    if not rel:
        return {}
    try:
        snap = rel.order_by("-id").first() or rel.last()
    except Exception:
        snap = None
    if not snap:
        return {}
    for attr in ("data", "json", "payload", "content", "blob"):
        if hasattr(snap, attr):
            val = getattr(snap, attr)
            if isinstance(val, str):
                try:
                    return json.loads(val) or {}
                except Exception:
                    return {}
            if isinstance(val, dict):
                return val
    return {}


def _build_pool(m: Match) -> dict:
    """
    Merge snapshot features + match scalar ids into one dict.
    """
    pool = {}
    snap = _latest_snapshot_dict(m)
    if isinstance(snap, dict):
        pool.update(snap)
    # normalize IDs (both aliases available to the row-builder)
    pool["home_id"] = getattr(m, "home_id", None)
    pool["away_id"] = getattr(m, "away_id", None)
    # if caller expects explicit *_team_id, keep them too
    pool.setdefault("home_team_id", pool.get("home_id"))
    pool.setdefault("away_team_id", pool.get("away_id"))
    return pool


def _row_df_from_pool(pool: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Build exactly one-row DataFrame with ALL expected columns (in the training order).
    - Maps team-id aliases
    - Fills defaults for missing values
    """
    row = {}
    for name in feature_names:
        if name in ("home_team_id", "home_id"):
            row[name] = int(pool.get("home_id") or pool.get("home_team_id") or 0)
        elif name in ("away_team_id", "away_id"):
            row[name] = int(pool.get("away_id") or pool.get("away_team_id") or 0)
        elif name == "league_cluster":
            row[name] = pool.get("league_cluster", "") or ""
        else:
            row[name] = _finite(pool.get(name, 0.0), 0.0)
    return pd.DataFrame([row], columns=feature_names)


# ------------------------- command -------------------------

class Command(BaseCommand):
    help = "Predict total (and optional yellow) cards λ for upcoming fixtures and store in MatchPrediction."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--debug", action="store_true")

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        days = int(opts["days"])
        debug = bool(opts.get("debug"))

        if debug:
            print(f"[debug] {PRED_CARDS_VERSION}")

        # --- Load latest CARDS models + meta (required) ---
        cards_mv = (ModelVersion.objects
                    .filter(kind="cards", league_id=league_id)
                    .order_by("-trained_until", "-id")
                    .first())
        if not cards_mv:
            self.stderr.write("[warn] No ModelVersion(kind='cards') found. Run train_cards first.")
            return

        try:
            cards_home = joblib.load(cards_mv.file_home)
            cards_away = joblib.load(cards_mv.file_away)
            cards_feats = _load_meta_for_model(cards_mv, "cards")
            if debug:
                print(f"[debug] cards model_home={cards_mv.file_home}")
                print(f"[debug] cards feats: {len(cards_feats)}")
        except Exception as e:
            self.stderr.write(f"[warn] Could not load 'cards' model/meta: {e}")
            return

        # --- Load latest YELLOWS models + meta (optional) ---
        yell_home = yell_away = None
        yell_feats = None
        yell_mv = (ModelVersion.objects
                   .filter(kind="yellows", league_id=league_id)
                   .order_by("-trained_until", "-id")
                   .first())
        if yell_mv:
            try:
                yell_home = joblib.load(yell_mv.file_home)
                yell_away = joblib.load(yell_mv.file_away)
                yell_feats = _load_meta_for_model(yell_mv, "yellows")
                if debug:
                    print(f"[debug] yellows model_home={yell_mv.file_home}")
                    print(f"[debug] yellows feats: {len(yell_feats)}")
            except Exception as e:
                self.stderr.write(f"[warn] Could not load 'yellows' model/meta: {e}")

        # --- Fixtures window ---
        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)
        fixtures = (Match.objects
                    .filter(league_id=league_id,
                            kickoff_utc__gte=now,
                            kickoff_utc__lte=upto)
                    .exclude(status__in=["FT", "AET", "PEN"])
                    .order_by("kickoff_utc"))

        if not fixtures.exists():
            self.stdout.write("No fixtures in window.")
            return

        wrote = 0
        for m in fixtures:
            try:
                pool = _build_pool(m)

                # --- total CARDS ---
                Xc = _row_df_from_pool(pool, cards_feats)
                if debug:
                    print(f"[debug] match={m.id} cards_df shape={Xc.shape}")
                lam_ch = float(cards_home.predict(Xc)[0])
                lam_ca = float(cards_away.predict(Xc)[0])
                lam_ch = float(np.clip(lam_ch, 0.01, 12.0))
                lam_ca = float(np.clip(lam_ca, 0.01, 12.0))

                updates = {
                    "league_id": m.league_id,
                    "season": m.season,
                    "kickoff_utc": m.kickoff_utc,
                    "lambda_cards_home": lam_ch,
                    "lambda_cards_away": lam_ca,
                }

                # --- YELLOWS (optional) ---
                if yell_home is not None and yell_away is not None and yell_feats:
                    Xy = _row_df_from_pool(pool, yell_feats)
                    if debug:
                        print(f"[debug] match={m.id} yellows_df shape={Xy.shape}")
                    lam_yh = float(yell_home.predict(Xy)[0])
                    lam_ya = float(yell_away.predict(Xy)[0])
                    updates["lambda_yellows_home"] = float(np.clip(lam_yh, 0.01, 12.0))
                    updates["lambda_yellows_away"] = float(np.clip(lam_ya, 0.01, 12.0))

                MatchPrediction.objects.update_or_create(match=m, defaults=updates)
                wrote += 1

            except Exception as e:
                self.stderr.write(f"[warn] cards predict failed for match {m.id}: {e}")

        kinds = ["cards"] + (["yellows"] if (yell_home and yell_away and yell_feats) else [])
        self.stdout.write(self.style.SUCCESS(
            f"Wrote {'/'.join(kinds)} λ for {wrote} fixtures (league {league_id})"
        ))
