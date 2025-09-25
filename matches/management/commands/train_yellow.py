from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.conf import settings

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from matches.models import Match, ModelVersion


# Reuse the same feature set you used for cards training
CANDIDATE_FEATURES = [
    "h_gf10","a_gf10","h_ga10","a_ga10",
    "h_sot10","a_sot10","h_sot_pct10","a_sot_pct10",
    "h_conv10","a_conv10",
    "h_poss10","a_poss10",
    "h_clean_sheets10","a_clean_sheets10",
    "h_matches_14d","a_matches_14d",
    "h_rest_days","a_rest_days",
    "h_cards_for10","a_cards_for10",
    "h_corners_for10","a_corners_for10",
    "d_sot10","d_rest_days",
    "a_stats_missing","h_stats_missing",
    "home_id","away_id",
    "league_cluster",
]


def _safe_float(x, d=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else float(d)
    except Exception:
        return float(d)


def _available_fields(model) -> set[str]:
    names = set()
    for f in model._meta.get_fields():
        if hasattr(f, "attname"):
            names.add(f.attname)
        if hasattr(f, "name"):
            names.add(f.name)
    return names


def _build_feature_frame(qs, feature_names: list[str]) -> pd.DataFrame:
    rows = list(qs.values(*feature_names))
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=feature_names)
    if df.empty:
        return df
    num_cols = [c for c in feature_names if c != "league_cluster"]
    for c in num_cols:
        df[c] = df[c].apply(lambda v: _safe_float(v, 0.0))
    if "league_cluster" in feature_names:
        if "league_cluster" not in df.columns:
            df["league_cluster"] = ""
        df["league_cluster"] = df["league_cluster"].astype("category")
    return df


def _make_pipeline(numeric_cols, cat_cols):
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols) if cat_cols else ("cat", "drop", []),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("prep", ct), ("reg", model)])


def _extract_yellows_from_json_blob(blob: dict, keys_home: list[str], keys_away: list[str]):
    if not isinstance(blob, dict):
        return (None, None)

    def _probe(d: dict, keys: list[str]):
        for k in keys:
            if k in d:
                try:
                    return _safe_float(d[k], None)
                except Exception:
                    pass
        return None

    yh = _probe(blob, keys_home)
    ya = _probe(blob, keys_away)

    if (yh is None or ya is None):
        for container in ("stats", "statistics", "summary", "teams"):
            sub = blob.get(container)
            if isinstance(sub, dict):
                if yh is None:
                    yh = _probe(sub, keys_home)
                if ya is None:
                    ya = _probe(sub, keys_away)
    return (yh, ya)


class Command(BaseCommand):
    help = "Train yellow-card regressors (home/away). Source can be 'cards' (proxy) or 'json' (extract)."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--seasons", type=int, nargs="+", required=True)
        parser.add_argument("--kfold", type=int, default=5)
        parser.add_argument("--source", choices=["cards", "json"], default="cards",
                            help="Label source: 'cards' uses cards_home/away as yellow proxy; 'json' extracts from JSON.")
        parser.add_argument("--json-home-keys", nargs="*", default=[
            "yellows_home","home_yellows","home_yellow_cards","home_cards_yellow",
            "home_yellow","y_home_yellows","yellowsH","H_yellows"
        ])
        parser.add_argument("--json-away-keys", nargs="*", default=[
            "yellows_away","away_yellows","away_yellow_cards","away_cards_yellow",
            "away_yellow","y_away_yellows","yellowsA","A_yellows"
        ])

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        seasons = list(opts["seasons"])
        n_folds = int(opts["kfold"])
        source = opts["source"]
        keys_home = list(opts["json_home_keys"])
        keys_away = list(opts["json_away_keys"])

        try:
            base_dir = Path(settings.BASE_DIR)
        except Exception:
            base_dir = Path(__file__).resolve().parents[5]
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        fields_available = _available_fields(Match)
        feats = [f for f in CANDIDATE_FEATURES if f in fields_available]
        if not feats:
            self.stderr.write("No usable features available on Match.")
            return

        qs = Match.objects.filter(league_id=league_id, season__in=seasons)
        if not qs.exists():
            self.stderr.write("No matches for given league/seasons.")
            return

        # --- build labels
        label_rows = []
        if source == "cards":
            qs_lab = qs.filter(Q(cards_home__isnull=False) & Q(cards_away__isnull=False))
            for r in qs_lab.values("id", "cards_home", "cards_away"):
                label_rows.append({
                    "id": r["id"],
                    "yellows_home": _safe_float(r["cards_home"], 0.0),
                    "yellows_away": _safe_float(r["cards_away"], 0.0),
                })
        else:  # json
            # try to parse from raw_result_json first; fall back to stats relation if present
            for m in qs.only("id", "raw_result_json"):
                yh = ya = None
                blob = m.raw_result_json
                if isinstance(blob, str):
                    try:
                        blob = json.loads(blob)
                    except Exception:
                        blob = None
                if isinstance(blob, dict):
                    yh, ya = _extract_yellows_from_json_blob(blob, keys_home, keys_away)

                if (yh is None or ya is None) and hasattr(m, "stats") and m.stats:
                    for attr in ("data", "json", "payload", "content", "blob"):
                        if hasattr(m.stats, attr):
                            val = getattr(m.stats, attr)
                            if isinstance(val, str):
                                try:
                                    val = json.loads(val)
                                except Exception:
                                    val = None
                            if isinstance(val, dict):
                                yh2, ya2 = _extract_yellows_from_json_blob(val, keys_home, keys_away)
                                yh = yh if yh is not None else yh2
                                ya = ya if ya is not None else ya2

                if yh is not None and ya is not None:
                    label_rows.append({
                        "id": m.id,
                        "yellows_home": _safe_float(yh, 0.0),
                        "yellows_away": _safe_float(ya, 0.0),
                    })

        lab = pd.DataFrame(label_rows)
        if lab.empty:
            msg = "Could not derive yellow-card labels"
            if source == "json":
                msg += " from JSON; check your key names or use --source cards"
            self.stderr.write(msg + ".")
            return

        # --- features aligned to labels
        X = _build_feature_frame(qs.filter(id__in=lab["id"].tolist()), feats)
        if X.empty:
            self.stderr.write("Feature frame empty after alignment.")
            return

        # bring id into X to merge
        ids = list(qs.filter(id__in=lab["id"].tolist()).values_list("id", flat=True))
        X_ = X.copy()
        X_.insert(0, "id", ids)
        df = pd.merge(X_, lab, on="id", how="inner")
        if df.empty:
            self.stderr.write("No overlap between features and labels.")
            return

        target_h = df["yellows_home"].astype(float)
        target_a = df["yellows_away"].astype(float)
        Xall = df[feats].copy()

        cat_cols = ["league_cluster"] if "league_cluster" in feats else []
        num_cols = [c for c in feats if c not in cat_cols]

        pipe_h = _make_pipeline(num_cols, cat_cols)
        pipe_a = _make_pipeline(num_cols, cat_cols)

        if len(df) >= n_folds:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            mae_h, mae_a = [], []
            for i, (tr, te) in enumerate(kf.split(Xall), 1):
                ph = _make_pipeline(num_cols, cat_cols)
                pa = _make_pipeline(num_cols, cat_cols)
                ph.fit(Xall.iloc[tr], target_h.iloc[tr])
                pa.fit(Xall.iloc[tr], target_a.iloc[tr])
                pred_h = ph.predict(Xall.iloc[te])
                pred_a = pa.predict(Xall.iloc[te])
                mae_h.append(mean_absolute_error(target_h.iloc[te], pred_h))
                mae_a.append(mean_absolute_error(target_a.iloc[te], pred_a))
                self.stdout.write(f"Fold {i}: MAE yellows H={mae_h[-1]:.3f} A={mae_a[-1]:.3f}")
        else:
            self.stdout.write("Not enough rows for CV; skipping folds.")

        pipe_h.fit(Xall, target_h)
        pipe_a.fit(Xall, target_a)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        mh = models_dir / f"rf_yellows_home_L{league_id}_{ts}.joblib"
        ma = models_dir / f"rf_yellows_away_L{league_id}_{ts}.joblib"
        meta = models_dir / f"yellows_meta_{league_id}_{ts}.json"

        import joblib
        joblib.dump(pipe_h, mh)
        joblib.dump(pipe_a, ma)

        with open(meta, "w") as f:
            json.dump({
                "kind": "yellows",
                "league_id": league_id,
                "timestamp": ts,
                "expected_features": feats,
                "numeric": [c for c in feats if c != "league_cluster"],
                "categorical": ["league_cluster"] if "league_cluster" in feats else [],
            }, f, indent=2)

        try:
            ModelVersion.objects.create(
                kind="yellows",
                league_id=league_id,
                name=f"yellows L{league_id} {ts}",
                file_home=str(mh),
                file_away=str(ma),
                trained_until=datetime.now(timezone.utc),
            )
        except Exception as e:
            self.stderr.write(f"[warn] Could not write ModelVersion: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"Saved yellows models: {mh.name}, {ma.name}\nMeta: {meta.name}"
        ))
