# prediction/matches/management/commands/check_features.py
# Enhanced feature QA & comparison utility
# - Confirms compact MLTrainingMatch fields are populated and sane
# - Audits extended MatchStats coverage
# - Verifies JSON aggregate keys and finiteness
# - Optional: correlation to targets, pairwise feature correlation, decile-bucket analysis
# - Optional: denominator-drift sampler comparing on-the-fly 10-match means vs stored features
#
# Usage examples:
#   python manage.py check_features \
#       --league-id 72 --seasons 2024 \
#       --with-corr --with-bins --export features_72_2024.csv \
#       --days 7 --sample 5 --drift-sample 15
#
# Notes:
# - Spearman/Pearson use numpy; Spearman also uses scipy if available (falls back if missing).
# - Pairwise feature correlation highlights multicollinearity.
# - Decile-bucket summaries show monotonic signal vs targets.

import csv
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    from scipy.stats import spearmanr as _spearmanr  # optional
except Exception:  # pragma: no cover
    _spearmanr = None

from django.core.management.base import BaseCommand
from django.db.models import Avg, Min, Max, Q, Count, BooleanField

from matches.models import Match, MatchStats, MLTrainingMatch

# ===== existing compact table features you already check =====
FEATS = [
    "h_gf10","a_gf10","d_gf10",
    "h_ga10","a_ga10",
    "h_gd10","a_gd10",
    "h_sot10","a_sot10","d_sot10",
    "h_sot_pct10","a_sot_pct10",
    "h_conv10","a_conv10",
    "h_poss10","a_poss10",
    "h_clean_sheets10","a_clean_sheets10",
    "h_corners_for10","a_corners_for10",
    "h_cards_for10","a_cards_for10",
    "h_rest_days","a_rest_days","d_rest_days",
    "h_matches_14d","a_matches_14d",
    "h_stats_missing","a_stats_missing",
]

TARGETS = ["y_home_goals_90", "y_away_goals_90"]

# Derived target helpers
DERIVED = {
    "goal_diff": lambda r: _safe_float(r.get("y_home_goals_90")) - _safe_float(r.get("y_away_goals_90")),
    "total_goals": lambda r: _safe_float(r.get("y_home_goals_90")) + _safe_float(r.get("y_away_goals_90")),
    "home_win": lambda r: 1.0 if _safe_float(r.get("y_home_goals_90")) > _safe_float(r.get("y_away_goals_90")) else 0.0,
    "over_2_5": lambda r: 1.0 if (_safe_float(r.get("y_home_goals_90")) + _safe_float(r.get("y_away_goals_90"))) > 2.5 else 0.0,
}

# ===== MatchStats extended fields to verify =====
MATCHSTATS_FIELDS = [
    # core
    "shots","sot","possession_pct","pass_acc_pct","corners","cards","xg","yellows","reds",
    # extended
    "shots_off","shots_blocked","shots_in_box","shots_out_box",
    "fouls","offsides","saves","passes_total","passes_accurate",
]

# ===== keys expected inside JSON aggregates =====
JSON_KEYS = [
    "gf","ga","cs","shots","sot","shots_off","shots_blocked","shots_in_box","shots_out_box",
    "fouls","offsides","saves","passes_total","passes_accurate","pass_acc","poss",
    "corners","cards","yellows","reds","xg","conv","sot_pct"
]


def _safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _avg(a):
    return float(np.mean(a)) if a else 0.0


def _form_for_team(team_id, upto_dt, lookback=10):
    qs = (
        Match.objects
        .filter(Q(home_id=team_id) | Q(away_id=team_id), kickoff_utc__lt=upto_dt, status__in=["FT","AET","PEN"])
        .order_by("-kickoff_utc")[:lookback]
    )
    gf, ga, sot, shots, poss, corners, cards, cs = [], [], [], [], [], [], [], []
    for m in qs:
        is_home = (m.home_id == team_id)
        if is_home:
            gf.append(m.goals_home or 0); ga.append(m.goals_away or 0)
            cs.append(1.0 if (m.goals_away or 0)==0 else 0.0)
        else:
            gf.append(m.goals_away or 0); ga.append(m.goals_home or 0)
            cs.append(1.0 if (m.goals_home or 0)==0 else 0.0)
        st = MatchStats.objects.filter(match=m, team_id=team_id).first()
        if st:
            sot.append(st.sot or 0)
            shots.append(st.shots or 0)
            poss.append(st.possession_pct or 0)
            corners.append(st.corners or 0)
            cards.append((st.yellows or 0) + (st.reds or 0))
    shots_avg = _avg(shots)
    return {
        "gf": _avg(gf),
        "ga": _avg(ga),
        "gd": _avg(gf) - _avg(ga),
        "sot": _avg(sot),
        "shots": shots_avg,
        "sot_pct": (_avg(sot) / shots_avg) if shots_avg>0 else 0.0,
        "conv": (_avg(gf) / shots_avg) if shots_avg>0 else 0.0,
        "poss": _avg(poss),
        "corners": _avg(corners),
        "cards": _avg(cards),
        "cs_rate": _avg(cs),
        "count": len(qs),
    }


def _rest_days(team_id, upto_dt):
    last = (
        Match.objects
        .filter(Q(home_id=team_id) | Q(away_id=team_id), kickoff_utc__lt=upto_dt, status__in=["FT","AET","PEN"])
        .order_by("-kickoff_utc")
        .first()
    )
    if not last:
        return 7.0
    d = (upto_dt - last.kickoff_utc)
    return max(0.0, d.days + d.seconds/86400.0)


def _matches_in_14d(team_id, upto_dt):
    since = upto_dt - timedelta(days=14)
    return (
        Match.objects
        .filter(Q(home_id=team_id) | Q(away_id=team_id), kickoff_utc__lt=upto_dt, kickoff_utc__gte=since, status__in=["FT","AET","PEN"])
        .count()
    )


def _build_feature_vector(m: Match):
    dt = m.kickoff_utc
    H = _form_for_team(m.home_id, dt, lookback=10)
    A = _form_for_team(m.away_id, dt, lookback=10)
    h_rest = _rest_days(m.home_id, dt)
    a_rest = _rest_days(m.away_id, dt)
    feats = {
        "h_gf10": H["gf"], "a_gf10": A["gf"], "d_gf10": H["gf"]-A["gf"],
        "h_ga10": H["ga"], "a_ga10": A["ga"],
        "h_gd10": H["gd"], "a_gd10": A["gd"],
        "h_sot10": H["sot"], "a_sot10": A["sot"], "d_sot10": H["sot"]-A["sot"],
        "h_sot_pct10": H["sot_pct"], "a_sot_pct10": A["sot_pct"],
        "h_conv10": H["conv"], "a_conv10": A["conv"],
        "h_poss10": H["poss"], "a_poss10": A["poss"],
        "h_clean_sheets10": H["cs_rate"], "a_clean_sheets10": A["cs_rate"],
        "h_corners_for10": H["corners"], "a_corners_for10": A["corners"],
        "h_cards_for10": H["cards"], "a_cards_for10": A["cards"],
        "h_rest_days": h_rest, "a_rest_days": a_rest, "d_rest_days": h_rest - a_rest,
        "h_matches_14d": float(_matches_in_14d(m.home_id, dt)),
        "a_matches_14d": float(_matches_in_14d(m.away_id, dt)),
        "h_stats_missing": 1.0 if H["count"] < 3 else 0.0,
        "a_stats_missing": 1.0 if A["count"] < 3 else 0.0,
    }
    X = np.array([_safe_float(feats.get(f, 0.0)) for f in FEATS], dtype=float)
    return feats, X


class Command(BaseCommand):
    help = "Verify + compare features (compact fields, MatchStats, JSON). Optional correlation, bins, and drift checks."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--seasons", type=int, nargs="*", default=None)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--sample", type=int, default=3, help="How many upcoming matches to print features for")
        parser.add_argument("--with-corr", action="store_true", help="Compute Pearson/Spearman corr vs targets & derived")
        parser.add_argument("--with-bins", action="store_true", help="Bucket features by deciles and print target means")
        parser.add_argument("--export", type=str, default=None, help="Export per-row features & targets CSV")
        parser.add_argument("--drift-sample", type=int, default=0, help="If >0, sample rows to compare stored vs on-the-fly features")

    # --------------------------- utilities ---------------------------
    def _qs_rows(self, qs, fields):
        for r in qs.values(*fields):
            yield {k: r.get(k) for k in fields}

    def _fetch_train_rows(self, league_id, seasons):
        fields = ["fixture_id", "kickoff_utc"] + FEATS + TARGETS + ["stats10_json", "stats5_json", "home_team_id", "away_team_id"]
        qs = MLTrainingMatch.objects.filter(league_id=league_id)
        if seasons:
            qs = qs.filter(season__in=seasons)
        return list(self._qs_rows(qs, fields))

    def _pearson(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan")
        x = x[mask]; y = y[mask]
        if x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def _spearman(self, x, y):
        if _spearmanr is None:
            return float("nan")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan")
        rho, p = _spearmanr(x[mask], y[mask])
        try:
            return float(rho)
        except Exception:
            return float("nan")

    def _deciles(self, arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return []
        qs = [np.quantile(arr, q) for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
        return qs

    # --------------------------- handle ---------------------------
    def handle(self, *args, **opts):
        league_id = opts["league_id"]
        seasons = opts["seasons"]
        days = opts["days"]
        sample_n = opts["sample"]
        with_corr = opts["with_corr"]
        with_bins = opts["with_bins"]
        export_path = opts["export"]
        drift_sample_n = opts["drift_sample"]

        # ---------- TRAIN rows summary ----------
        rows = self._fetch_train_rows(league_id, seasons)
        total = len(rows)
        self.stdout.write(f"[TRAIN] MLTrainingMatch rows: {total}")

        miss_targets = {t: sum(1 for r in rows if r.get(t) is None) for t in TARGETS}
        self.stdout.write(f"[TRAIN] Missing targets: {miss_targets}")

        # Feature coverage
        self.stdout.write("[TRAIN] Feature coverage (nulls / avg / min / max):")
        for f in FEATS:
            vals = [r.get(f) for r in rows]
            nulls = sum(1 for v in vals if v is None)
            finite = [float(v) for v in vals if v is not None and math.isfinite(_safe_float(v))]
            avg = np.mean(finite) if finite else float("nan")
            mn = np.min(finite) if finite else float("nan")
            mx = np.max(finite) if finite else float("nan")
            self.stdout.write(f"  - {f}: nulls={nulls}/{total}, avg={avg:.4f}, min={mn:.4f}, max={mx:.4f}")

        # ---------- STATS coverage ----------
        self.stdout.write("\n[STATS] Extended MatchStats coverage within league/season:")
        stats_qs = MatchStats.objects.filter(match__league_id=league_id)
        if seasons:
            stats_qs = stats_qs.filter(match__season__in=seasons)
        stats_total = stats_qs.count()
        self.stdout.write(f"  rows: {stats_total}")

        match_stats_counts = stats_qs.values('match_id').annotate(c=Count('id')).order_by()
        both = sum(1 for r in match_stats_counts if r['c'] >= 2)
        uniq_matches = len({r['match_id'] for r in match_stats_counts})
        pct_both = (both / uniq_matches) if uniq_matches else 0.0
        self.stdout.write(f"  fixtures with both team rows: {both}/{uniq_matches} ({pct_both:.1%})")

        for name in MATCHSTATS_FIELDS:
            nn = stats_qs.filter(**{f"{name}__isnull": False}).count()
            pct = (nn / stats_total) if stats_total else 0.0
            self.stdout.write(f"  - {name}: non-null {nn}/{stats_total} ({pct:.1%})")

        # ---------- JSON coverage ----------
        has_json10 = sum(1 for r in rows if r.get("stats10_json") is not None)
        has_json5  = sum(1 for r in rows if r.get("stats5_json") is not None)
        self.stdout.write(f"\n[JSON] MLTrainingMatch JSON aggregates present: stats10_json={has_json10}/{total}, stats5_json={has_json5}/{total}")

        def _json_key_coverage(field_name):
            present = defaultdict(int)
            finite = defaultdict(int)
            nside = 0
            for r in rows:
                blob = r.get(field_name) or {}
                for side in ("shots", "shots_opp"):
                    d = blob.get(side) or {}
                    if d:
                        nside += 1
                    for k in JSON_KEYS:
                        if k in d:
                            present[k] += 1
                            v = d.get(k)
                            try:
                                vv = float(v)
                                if math.isfinite(vv):
                                    finite[k] += 1
                            except Exception:
                                pass
            return nside, present, finite

        n10, present10, finite10 = _json_key_coverage("stats10_json")
        self.stdout.write("[JSON] stats10_json key coverage (counts across both sides per row):")
        for k in JSON_KEYS:
            self.stdout.write(f"  - {k}: present={present10[k]}, finite={finite10[k]} (checked in {n10} side-dicts)")

        n5, present5, finite5 = _json_key_coverage("stats5_json")
        self.stdout.write("[JSON] stats5_json key coverage (counts across both sides per row):")
        for k in JSON_KEYS:
            self.stdout.write(f"  - {k}: present={present5[k]}, finite={finite5[k]} (checked in {n5} side-dicts)")

        # ---------- Optional: correlations vs targets ----------
        if with_corr:
            self.stdout.write("\n[CORR] Pearson/Spearman vs targets & derived:")
            # Build arrays
            feats_mat = {f: [] for f in FEATS}
            tars = {t: [] for t in TARGETS}
            der = {k: [] for k in DERIVED.keys()}
            for r in rows:
                for f in FEATS:
                    feats_mat[f].append(_safe_float(r.get(f)))
                for t in TARGETS:
                    tars[t].append(_safe_float(r.get(t), default=float("nan")))
                # derived
                for k, fn in DERIVED.items():
                    try:
                        der[k].append(fn(r))
                    except Exception:
                        der[k].append(float("nan"))

            def _print_corr_block(target_name, target_arr):
                arr = np.asarray(target_arr, dtype=float)
                mask = np.isfinite(arr)
                n_eff = int(mask.sum())
                self.stdout.write(f"  Target: {target_name} (n={n_eff})")
                # collect and sort by |pearson|
                rows_out = []
                for f, x in feats_mat.items():
                    x_arr = np.asarray(x, dtype=float)
                    mask2 = mask & np.isfinite(x_arr)
                    if mask2.sum() < 30:
                        pr = float("nan"); sr = float("nan")
                    else:
                        pr = self._pearson(x_arr[mask2], arr[mask2])
                        sr = self._spearman(x_arr[mask2], arr[mask2])
                    rows_out.append((f, pr, sr))
                # sort by absolute Pearson
                rows_out.sort(key=lambda t: (0 if math.isnan(t[1]) else -abs(t[1])))
                for f, pr, sr in rows_out[:20]:  # top 20
                    self.stdout.write(f"    - {f:22s} pearson={pr: .3f}  spearman={sr: .3f}")

            # real targets
            for t in TARGETS:
                _print_corr_block(t, tars[t])
            # derived
            for k in DERIVED.keys():
                _print_corr_block(k, der[k])

            # Pairwise feature correlation to spot multicollinearity
            self.stdout.write("\n[CORR] Pairwise feature Pearson |r|>=0.90 (potential multicollinearity):")
            names = FEATS
            M = len(names)
            # build matrix as needed
            F = {f: np.asarray([_safe_float(r.get(f)) for r in rows], dtype=float) for f in names}
            for i in range(M):
                for j in range(i+1, M):
                    xi, xj = F[names[i]], F[names[j]]
                    mask = np.isfinite(xi) & np.isfinite(xj)
                    if mask.sum() < 50:
                        continue
                    r = self._pearson(xi[mask], xj[mask])
                    if not math.isnan(r) and abs(r) >= 0.90:
                        self.stdout.write(f"  - {names[i]} ~ {names[j]} : r={r: .3f}")

        # ---------- Optional: decile-bucket analysis ----------
        if with_bins:
            self.stdout.write("\n[BINS] Decile buckets: mean target by feature decile (goal_diff & total_goals)")
            for f in FEATS:
                vals = np.asarray([_safe_float(r.get(f), default=float("nan")) for r in rows])
                gd = np.asarray([DERIVED["goal_diff"](r) for r in rows], dtype=float)
                tg = np.asarray([DERIVED["total_goals"](r) for r in rows], dtype=float)
                mask = np.isfinite(vals) & np.isfinite(gd) & np.isfinite(tg)
                if mask.sum() < 100:
                    continue
                qs = self._deciles(vals[mask])
                if not qs:
                    continue
                # build bins
                def _bucket(v):
                    for i, q in enumerate(qs):
                        if v <= q:
                            return i
                    return len(qs)
                buckets = defaultdict(lambda: {"n":0, "gd":0.0, "tg":0.0})
                vsub, gdsub, tgsub = vals[mask], gd[mask], tg[mask]
                for v, g1, g2 in zip(vsub, gdsub, tgsub):
                    b = _bucket(v)
                    buckets[b]["n"] += 1
                    buckets[b]["gd"] += g1
                    buckets[b]["tg"] += g2
                self.stdout.write(f"  Feature: {f}")
                for b in range(len(qs)+1):
                    if buckets[b]["n"] == 0:
                        continue
                    mean_gd = buckets[b]["gd"] / buckets[b]["n"]
                    mean_tg = buckets[b]["tg"] / buckets[b]["n"]
                    self.stdout.write(f"    decile {b+1}: n={buckets[b]['n']:4d}  goal_diff={mean_gd: .3f}  total_goals={mean_tg: .3f}")

        # ---------- Optional: export CSV ----------
        if export_path:
            fields = ["fixture_id", "kickoff_utc", "home_team_id", "away_team_id"] + FEATS + TARGETS
            with open(export_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(fields + ["goal_diff","total_goals","home_win","over_2_5"]) 
                for r in rows:
                    row = [r.get(k) for k in fields]
                    d = {k: DERIVED[k](r) for k in DERIVED}
                    w.writerow(row + [d["goal_diff"], d["total_goals"], d["home_win"], d["over_2_5"]])
            self.stdout.write(f"[EXPORT] Wrote {len(rows)} rows to {export_path}")

        # ---------- Optional: denominator-drift sampler ----------
        if drift_sample_n and total:
            self.stdout.write("\n[DRIFT] Comparing stored features vs on-the-fly 10-match means (sample)")
            sample_rows = random.sample(rows, k=min(drift_sample_n, total))
            for r in sample_rows:
                try:
                    # Fetch the live match row to rebuild H/A features
                    m = Match.objects.get(pk=r["fixture_id"])  # the target fixture
                    # Home side comparison using JSON stats10 (home side keyed at 'shots')
                    live_H = _form_for_team(m.home_id, m.kickoff_utc, lookback=10)
                    blob = r.get("stats10_json") or {}
                    stored_H = blob.get("shots") or {}
                    # Pick a few sensitive keys to compare
                    for key, label in [("shots","shots"),("sot","sot"),("poss","poss"),("conv","conv")]:
                        live_v = _safe_float(live_H.get(key), default=float("nan"))
                        stored_v = _safe_float(stored_H.get(key), default=float("nan"))
                        if math.isfinite(live_v) and math.isfinite(stored_v):
                            diff = live_v - stored_v
                            if abs(diff) > 0.15 * (1.0 + abs(stored_v)):
                                self.stdout.write(
                                    f"  [fixture {r['fixture_id']}] home {label}: stored={stored_v:.3f} live={live_v:.3f} diff={diff:+.3f}"
                                )
                except Exception:
                    continue

        # ---------- Upcoming fixtures quick sanity ----------
        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)
        upcoming = (
            Match.objects
            .filter(league_id=league_id, kickoff_utc__gte=now, kickoff_utc__lte=upto)
            .exclude(status__in=["FT","AET","PEN"]).order_by("kickoff_utc")
        )
        self.stdout.write(f"\n[PRED] Upcoming fixtures in window: {upcoming.count()}")

        tested = 0
        for m in upcoming[:sample_n]:
            feats, X = _build_feature_vector(m)
            missing_keys = [k for k in FEATS if k not in feats]
            nonfinite = [k for k in FEATS if not np.isfinite(_safe_float(feats.get(k, 0.0)))]
            self.stdout.write(f"[PRED] Match {m.id}: {m.home} vs {m.away} @ {m.kickoff_utc}")
            self.stdout.write("        Missing keys: " + (", ".join(missing_keys) if missing_keys else "none"))
            self.stdout.write("        Non-finite values: " + (", ".join(nonfinite) if nonfinite else "none"))
            self.stdout.write("        X shape: " + str(X.shape))
            tested += 1

        if tested == 0:
            self.stdout.write("[PRED] No matches to sample in the window.")

        # ---------- SCHEMA check ----------
        fields_map = {f.name: f for f in MLTrainingMatch._meta.get_fields() if hasattr(f, 'attname')}
        fields = set(fields_map.keys())
        missing_in_model = [f for f in FEATS if f not in fields]
        if missing_in_model:
            self.stderr.write(self.style.ERROR(f"[SCHEMA] Missing fields in MLTrainingMatch: {missing_in_model}"))
        else:
            self.stdout.write(self.style.SUCCESS("[SCHEMA] All FEATS present in MLTrainingMatch."))

        for t in TARGETS:
            if t not in fields:
                self.stderr.write(self.style.ERROR(f"[SCHEMA] Missing target field in MLTrainingMatch: {t}"))
