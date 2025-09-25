# -*- coding: utf-8 -*-
"""
Self-consistency audit for MLTrainingMatch rows.

What this checks:
  For each (league, season) pair selected:
    • Count parity of Match vs MLTrainingMatch (per (league, season)).
    • Recompute 'allowed' aggregates from opponent MatchStats rows strictly BEFORE kickoff:
        - Unweighted last-N mean
        - Exponentially-weighted mean (lambda=decay)
      and compare to stored values in stats10_json:
        stats10_json["allowed"]["home"/"away"]
        stats10_json["weighted"]["home_allowed"/"away_allowed"]
      Metrics compared (auto-selected from what we can recompute):
        shots_allowed, sot_allowed, shots_off_allowed, shots_blocked_allowed,
        shots_in_box_allowed, shots_out_box_allowed, fouls_drawn, offsides_against,
        saves_opponent, passes_total_against, passes_accurate_against,
        corners_against, cards_against, yellows_against, reds_against, xga
    • Verify stored meta (last_n, decay) if present against CLI values.

Outputs:
    • A CSV of anomalies (value mismatches, meta mismatches, missing data).
    • Optional CSV of count mismatches with missing/extra fixture_ids.

Usage examples:
    python manage.py audit_ml_rows \
        --season-min 2024 --season-max 2025 \
        --last-n 10 --decay 0.85 \
        --out audit_2024_2025.csv

    python manage.py audit_ml_rows \
        --league-ids 39 61 --season 2025 \
        --sample 500 \
        --tol-abs 1e-6 --tol-rel 0.01 \
        --out audit_l39_l61_2025.csv

    python manage.py audit_ml_rows \
        --season-min 2024 --season-max 2025 \
        --dump-mismatches \
        --mismatch-out mismatches_2024_2025.csv \
        --out /tmp/anomalies.csv
"""
from __future__ import annotations

import csv
import bisect
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count

from matches.models import Match, MatchStats, MLTrainingMatch

log = logging.getLogger(__name__)

# ---------- defaults ----------
LAST_N_DEFAULT = 10
DECAY_DEFAULT = 0.85
TOL_ABS_DEFAULT = 1e-6          # absolute tolerance
TOL_REL_DEFAULT = 1e-2          # relative tolerance (1% default)
SAMPLE_DEFAULT = 0              # 0 = audit all rows
MISMATCH_MAX_IDS_DEFAULT = 200

# Metrics we can recompute from MatchStats and how they map -> "allowed" keys
RECOMP_MAP = {
    # source_field_in_MatchStats : allowed_key_name
    "shots": "shots_allowed",
    "sot": "sot_allowed",
    "shots_off": "shots_off_allowed",
    "shots_blocked": "shots_blocked_allowed",
    "shots_in_box": "shots_in_box_allowed",
    "shots_out_box": "shots_out_box_allowed",
    "fouls": "fouls_drawn",
    "offsides": "offsides_against",
    "saves": "saves_opponent",
    "passes_total": "passes_total_against",
    "passes_accurate": "passes_accurate_against",
    "corners": "corners_against",
    "cards": "cards_against",
    "yellows": "yellows_against",
    "reds": "reds_against",
    "xg": "xga",
}

@dataclass
class TeamSlice:
    times: List  # list[datetime]
    matches: List[Match]

def _exp_weights(n: int, lam: float) -> List[float]:
    # index 0 corresponds to most-recent match (we slice newest-first)
    return [lam ** i for i in range(n)]

def _same_id(a, b) -> bool:
    try:
        return int(a) == int(b)
    except Exception:
        return str(a) == str(b)

def _getattr_num(obj, name: str) -> float:
    v = getattr(obj, name, 0.0)
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0

class Command(BaseCommand):
    help = "Audit MLTrainingMatch stats10_json 'allowed' fields via self-consistency recomputation."

    # ---------- CLI ----------
    def add_arguments(self, p):
        g_target = p.add_mutually_exclusive_group(required=True)
        g_target.add_argument("--season", type=int, help="Single season to audit.")
        g_target.add_argument("--season-min", type=int, help="Lower bound (inclusive) for seasons.")
        p.add_argument("--season-max", type=int, help="Upper bound (inclusive) for seasons. Required with --season-min.")
        p.add_argument("--league-ids", type=int, nargs="*", help="Optional league id filter (one or more).")

        p.add_argument("--sample", type=int, default=SAMPLE_DEFAULT, help="Random sample size per (league, season) pair. 0 = all.")
        p.add_argument("--last-n", type=int, default=LAST_N_DEFAULT, help="Last-N window to recompute.")
        p.add_argument("--decay", type=float, default=DECAY_DEFAULT, help="Exponential decay for weighted means.")
        p.add_argument("--tol-abs", type=float, default=TOL_ABS_DEFAULT, help="Absolute tolerance for float comparisons.")
        p.add_argument("--tol-rel", type=float, default=TOL_REL_DEFAULT, help="Relative tolerance (fraction, e.g., 0.01 = 1%).")
        p.add_argument("--out", type=str, required=True, help="CSV output path for anomalies.")

        # extra reporting: detailed count parity mismatches
        p.add_argument("--dump-mismatches", action="store_true",
                       help="Print detailed (league,season) count mismatches with missing/extra fixture_ids.")
        p.add_argument("--mismatch-out", type=str, default=None,
                       help="Optional CSV path to write mismatch details.")
        p.add_argument("--mismatch-max-ids", type=int, default=MISMATCH_MAX_IDS_DEFAULT,
                       help="Cap how many fixture_ids to include per pair.")

    # ---------- main ----------
    def handle(self, *args, **opt):
        season = opt.get("season")
        season_min = opt.get("season_min")
        season_max = opt.get("season_max")
        league_ids = opt.get("league_ids") or []
        sample = int(opt["sample"])
        last_n = max(1, int(opt["last_n"]))
        decay = float(opt["decay"])
        tol_abs = float(opt["tol_abs"])
        tol_rel = float(opt["tol_rel"])
        out_path = opt["out"]

        if season_min is not None and season_max is None:
            raise CommandError("--season-max is required when using --season-min")
        if season_min is None and season_max is not None:
            raise CommandError("--season-min is required when using --season-max")
        if not (0.0 < decay <= 1.0):
            raise CommandError("--decay must be in (0,1]")

        # Build season list
        seasons = [int(season)] if season is not None else list(range(int(season_min), int(season_max) + 1))

        # ---------- discover pairs we will audit ----------
        match_qs = Match.objects.all()
        if league_ids:
            match_qs = match_qs.filter(league_id__in=league_ids)
        match_pairs_qs = (
            match_qs.filter(season__in=seasons)
                    .values("league_id", "season")
                    .annotate(n=Count("id"))  # Match PK
                    .order_by("league_id", "season")
        )
        match_pairs = list(match_pairs_qs)  # evaluate once

        # Count ML rows to check parity (informational + optional deep dive)
        mltm_qs = MLTrainingMatch.objects.all()
        if league_ids:
            mltm_qs = mltm_qs.filter(league_id__in=league_ids)
        mltm_pairs = list(
            mltm_qs.filter(season__in=seasons)
                   .values("league_id", "season")
                   .annotate(n=Count("fixture_id"))
                   .order_by("league_id", "season")
        )
        have_map = {(p["league_id"], p["season"]): p["n"] for p in mltm_pairs}
        mis = 0
        for p in match_pairs:
            key = (p["league_id"], p["season"])
            want = p["n"]
            have = have_map.get(key, 0)
            if want != have:
                mis += 1
        self.stdout.write(f"Checking count parity...\nPairs checked: {len(match_pairs)}  mismatches: {mis}")

        # Detailed mismatch dump if requested
        if opt.get("dump_mismatches") or opt.get("mismatch_out"):
            self._report_count_mismatches(
                league_ids=league_ids,
                season_min=min(seasons),
                season_max=max(seasons),
                dump=bool(opt.get("dump_mismatches")),
                out_csv=opt.get("mismatch_out"),
                max_ids=int(opt.get("mismatch_max_ids") or MISMATCH_MAX_IDS_DEFAULT),
            )

        # ---------- open anomalies CSV ----------
        fields = [
            "anomaly_type",           # value_mismatch | meta_mismatch | missing_json
            "league_id", "season", "fixture_id", "kickoff_utc",
            "side", "kind",          # home/away | unweighted/weighted
            "metric",
            "got", "expected", "abs_diff", "rel_diff",
            "last_n_cli", "decay_cli",
            "last_n_meta", "decay_meta",
        ]
        out_f = open(out_path, "w", newline="")
        writer = csv.DictWriter(out_f, fieldnames=fields)
        writer.writeheader()

        total_rows = 0
        total_bad = 0
        total_meta = 0
        total_missing = 0

        # ---------- audit each (league, season) independently ----------
        for pair in match_pairs:
            L = pair["league_id"]; S = pair["season"]
            self.stdout.write(f"Auditing league={L} season={S}...")

            # Fixtures (ASC) and their stats, limited to this league/season
            fixtures = list(Match.objects.filter(league_id=L, season=S).order_by("kickoff_utc"))
            if not fixtures:
                continue
            fixture_ids = [m.id for m in fixtures]

            # Per-team timeline (ASC) for window slicing
            by_team_all: Dict[int, List[Match]] = defaultdict(list)
            for m in fixtures:
                by_team_all[m.home_id].append(m)
                by_team_all[m.away_id].append(m)

            slices: Dict[int, TeamSlice] = {}
            for tid, lst in by_team_all.items():
                lst = sorted(lst, key=lambda x: x.kickoff_utc)
                slices[tid] = TeamSlice(times=[mm.kickoff_utc for mm in lst], matches=lst)

            # Stats index: match_id -> List[MatchStats]
            stats_by_match: Dict[int, List[MatchStats]] = defaultdict(list)
            for s in MatchStats.objects.filter(match_id__in=fixture_ids):
                stats_by_match[s.match_id].append(s)

            # helpers
            def last_window(team_id: int, cutoff, n: int) -> List[Match]:
                sl = slices.get(team_id)
                if not sl:
                    return []
                idx = bisect.bisect_left(sl.times, cutoff)  # strictly before cutoff
                start = max(0, idx - n)
                return list(reversed(sl.matches[start:idx]))  # newest-first

            def opponent_row(mm: Match, team_id: int) -> Optional[MatchStats]:
                """
                Return the opponent's stats row for this match, using strict match on the true opponent id.
                If that fails and there are exactly two rows for the participants, use the non-self row.
                Only consider rows for the match's two teams.
                """
                opp_id = mm.away_id if _same_id(mm.home_id, team_id) else mm.home_id
                rows = [r for r in stats_by_match.get(mm.id, [])
                        if _same_id(r.team_id, mm.home_id) or _same_id(r.team_id, mm.away_id)]
                row = next((r for r in rows if _same_id(r.team_id, opp_id)), None)
                if row:
                    return row
                if len(rows) == 2:
                    a, b = rows
                    cand = a if not _same_id(a.team_id, team_id) else b
                    return cand if not _same_id(cand.team_id, team_id) else None
                return None

            def recompute_allowed(team_id: int, cutoff, n: int, weighted: bool) -> Optional[Dict[str, float]]:
                matches_win = last_window(team_id, cutoff, n)
                if not matches_win:
                    return None
                weights = _exp_weights(len(matches_win), decay) if weighted else [1.0] * len(matches_win)
                sums = {allowed_key: 0.0 for allowed_key in RECOMP_MAP.values()}
                denom = 0.0
                for i, mm in enumerate(matches_win):
                    opp = opponent_row(mm, team_id)
                    if not opp:
                        continue
                    wgt = weights[i]
                    for src, allowed_key in RECOMP_MAP.items():
                        sums[allowed_key] += wgt * _getattr_num(opp, src)
                    denom += wgt
                if denom <= 0.0:
                    return None
                return {k: (v / denom) for k, v in sums.items()}

            # ML rows for this pair
            rows = list(
                MLTrainingMatch.objects
                .filter(league_id=L, season=S)
                .only("fixture_id", "kickoff_utc", "home_team_id", "away_team_id", "stats10_json")
                .order_by("kickoff_utc")
            )
            total_rows += len(rows)

            if sample and sample < len(rows):
                random.shuffle(rows)
                rows = rows[:sample]

            # Compare per side
            for r in rows:
                sj = r.stats10_json or {}
                # meta cross-checks (optional in builder)
                meta = sj.get("meta", {}) or {}
                meta_last_n = meta.get("last_n")
                meta_decay = meta.get("decay")
                if (meta_last_n is not None and int(meta_last_n) != last_n) or \
                   (meta_decay is not None and float(meta_decay) != decay):
                    writer.writerow({
                        "anomaly_type": "meta_mismatch",
                        "league_id": L, "season": S, "fixture_id": r.fixture_id, "kickoff_utc": r.kickoff_utc,
                        "side": "", "kind": "",
                        "metric": "",
                        "got": "", "expected": "", "abs_diff": "", "rel_diff": "",
                        "last_n_cli": last_n, "decay_cli": decay,
                        "last_n_meta": meta_last_n, "decay_meta": meta_decay,
                    })
                    total_meta += 1

                sj_allowed = sj.get("allowed", {}) or {}
                sj_weighted = sj.get("weighted", {}) or {}

                got_home_u = sj_allowed.get("home", {}) or {}
                got_away_u = sj_allowed.get("away", {}) or {}
                got_home_w = sj_weighted.get("home_allowed", {}) or {}
                got_away_w = sj_weighted.get("away_allowed", {}) or {}

                # If JSON missing, flag once per side/kind and skip comparisons
                def flag_missing(side: str, kind: str):
                    nonlocal total_missing
                    writer.writerow({
                        "anomaly_type": "missing_json",
                        "league_id": L, "season": S, "fixture_id": r.fixture_id, "kickoff_utc": r.kickoff_utc,
                        "side": side, "kind": kind, "metric": "",
                        "got": "", "expected": "", "abs_diff": "", "rel_diff": "",
                        "last_n_cli": last_n, "decay_cli": decay,
                        "last_n_meta": meta_last_n, "decay_meta": meta_decay,
                    })
                    total_missing += 1

                # HOME side
                exp_u = recompute_allowed(r.home_team_id, r.kickoff_utc, last_n, weighted=False)
                exp_w = recompute_allowed(r.home_team_id, r.kickoff_utc, last_n, weighted=True)

                if exp_u is None:
                    # no opponent rows in window → not an anomaly
                    pass
                else:
                    if not got_home_u:
                        flag_missing("home", "unweighted")
                    else:
                        total_bad += self._compare_and_write(
                            writer, "value_mismatch", L, S, r,
                            side="home", kind="unweighted",
                            got=got_home_u, exp=exp_u,
                            tol_abs=tol_abs, tol_rel=tol_rel,
                            last_n_cli=last_n, decay_cli=decay,
                            last_n_meta=meta_last_n, decay_meta=meta_decay
                        )

                if exp_w is None:
                    pass
                else:
                    if not got_home_w:
                        flag_missing("home", "weighted")
                    else:
                        total_bad += self._compare_and_write(
                            writer, "value_mismatch", L, S, r,
                            side="home", kind="weighted",
                            got=got_home_w, exp=exp_w,
                            tol_abs=tol_abs, tol_rel=tol_rel,
                            last_n_cli=last_n, decay_cli=decay,
                            last_n_meta=meta_last_n, decay_meta=meta_decay
                        )

                # AWAY side
                exp_u = recompute_allowed(r.away_team_id, r.kickoff_utc, last_n, weighted=False)
                exp_w = recompute_allowed(r.away_team_id, r.kickoff_utc, last_n, weighted=True)

                if exp_u is None:
                    pass
                else:
                    if not got_away_u:
                        flag_missing("away", "unweighted")
                    else:
                        total_bad += self._compare_and_write(
                            writer, "value_mismatch", L, S, r,
                            side="away", kind="unweighted",
                            got=got_away_u, exp=exp_u,
                            tol_abs=tol_abs, tol_rel=tol_rel,
                            last_n_cli=last_n, decay_cli=decay,
                            last_n_meta=meta_last_n, decay_meta=meta_decay
                        )

                if exp_w is None:
                    pass
                else:
                    if not got_away_w:
                        flag_missing("away", "weighted")
                    else:
                        total_bad += self._compare_and_write(
                            writer, "value_mismatch", L, S, r,
                            side="away", kind="weighted",
                            got=got_away_w, exp=exp_w,
                            tol_abs=tol_abs, tol_rel=tol_rel,
                            last_n_cli=last_n, decay_cli=decay,
                            last_n_meta=meta_last_n, decay_meta=meta_decay
                        )

        out_f.close()

        # Summary
        self.stdout.write("Audit summary")
        self.stdout.write(f"Total rows checked: {total_rows}")
        self.stdout.write(f"Value mismatches:   {total_bad}")
        self.stdout.write(f"Meta mismatches:    {total_meta}")
        self.stdout.write(f"Missing JSON:       {total_missing}")
        self.stdout.write(f"Wrote anomalies → {out_path}")

    # ---------- helpers ----------
    def _compare_and_write(
        self, writer, anomaly_type: str, league_id: int, season: int, row: MLTrainingMatch,
        side: str, kind: str, got: Dict[str, float], exp: Dict[str, float],
        tol_abs: float, tol_rel: float,
        last_n_cli: int, decay_cli: float,
        last_n_meta: Optional[int], decay_meta: Optional[float],
    ) -> int:
        """
        Compare across the intersection of recomputable metrics
        with both absolute and relative tolerances.
        """
        bad = 0
        for allowed_key in RECOMP_MAP.values():
            if allowed_key not in exp:
                continue
            g = float(got.get(allowed_key, 0.0))
            e = float(exp.get(allowed_key, 0.0))
            abs_diff = abs(g - e)
            # relative diff: |g-e| / max(|e|, tiny)
            denom = max(abs(e), 1e-9)
            rel_diff = abs_diff / denom
            if (abs_diff > tol_abs) and (rel_diff > tol_rel):
                writer.writerow({
                    "anomaly_type": anomaly_type,
                    "league_id": league_id, "season": season,
                    "fixture_id": row.fixture_id, "kickoff_utc": row.kickoff_utc,
                    "side": side, "kind": kind,
                    "metric": allowed_key,
                    "got": f"{g:.9f}", "expected": f"{e:.9f}",
                    "abs_diff": f"{abs_diff:.9f}", "rel_diff": f"{rel_diff:.6f}",
                    "last_n_cli": last_n_cli, "decay_cli": decay_cli,
                    "last_n_meta": last_n_meta, "decay_meta": decay_meta,
                })
                bad += 1
        return bad

    def _report_count_mismatches(self, *, league_ids, season_min, season_max,
                                 dump: bool, out_csv: Optional[str], max_ids: int):
        """
        Compares Match vs MLTrainingMatch counts per (league_id, season) and, for each mismatch,
        computes the exact missing/extra fixture_ids.
        """
        base_match = Match.objects.all()
        base_t = MLTrainingMatch.objects.all()

        if league_ids:
            base_match = base_match.filter(league_id__in=league_ids)
            base_t     = base_t.filter(league_id__in=league_ids)
        if season_min is not None:
            base_match = base_match.filter(season__gte=season_min)
            base_t     = base_t.filter(season__gte=season_min)
        if season_max is not None:
            base_match = base_match.filter(season__lte=season_max)
            base_t     = base_t.filter(season__lte=season_max)

        pairs_match = {
            (p["league_id"], p["season"]): p["n"]
            for p in base_match.values("league_id","season").annotate(n=Count("id")).order_by()
        }
        pairs_t = {
            (p["league_id"], p["season"]): p["n"]
            for p in base_t.values("league_id","season").annotate(n=Count("fixture_id")).order_by()
        }

        keys = sorted(set(pairs_match) | set(pairs_t))
        mismatches = []
        for k in keys:
            nm = pairs_match.get(k, 0)
            nt = pairs_t.get(k, 0)
            if nm != nt:
                league_id, season = k
                mids = set(base_match.filter(league_id=league_id, season=season)
                                      .values_list("id", flat=True))
                tids = set(base_t.filter(league_id=league_id, season=season)
                                   .values_list("fixture_id", flat=True))

                missing = sorted(mids - tids)
                extra   = sorted(tids - mids)

                row = {
                    "league_id": league_id,
                    "season": season,
                    "match_count": nm,
                    "mltm_count": nt,
                    "missing_count": len(missing),
                    "extra_count": len(extra),
                    "missing_fixture_ids": missing[:max_ids],
                    "extra_fixture_ids": extra[:max_ids],
                    "missing_truncated": int(len(missing) > max_ids),
                    "extra_truncated": int(len(extra) > max_ids),
                }
                mismatches.append(row)

        if dump:
            if not mismatches:
                self.stdout.write(self.style.SUCCESS("Count parity OK — no mismatches."))
            else:
                self.stdout.write(self.style.WARNING(f"Found {len(mismatches)} mismatched (league, season) pairs:"))
                for r in mismatches:
                    self.stdout.write(
                        f"  • league={r['league_id']} season={r['season']} "
                        f"Match={r['match_count']} vs MLTM={r['mltm_count']}  "
                        f"missing={r['missing_count']} extra={r['extra_count']}"
                    )
                    if r["missing_fixture_ids"]:
                        ids = ", ".join(map(str, r["missing_fixture_ids"]))
                        tail = " …(truncated)" if r["missing_truncated"] else ""
                        self.stdout.write(f"      missing fixture_ids: {ids}{tail}")
                    if r["extra_fixture_ids"]:
                        ids = ", ".join(map(str, r["extra_fixture_ids"]))
                        tail = " …(truncated)" if r["extra_truncated"] else ""
                        self.stdout.write(f"      extra fixture_ids:   {ids}{tail}")

        if out_csv and mismatches:
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "league_id","season",
                    "match_count","mltm_count",
                    "missing_count","extra_count",
                    "missing_fixture_ids","extra_fixture_ids",
                    "missing_truncated","extra_truncated",
                ])
                for r in mismatches:
                    w.writerow([
                        r["league_id"], r["season"],
                        r["match_count"], r["mltm_count"],
                        r["missing_count"], r["extra_count"],
                        ";".join(map(str, r["missing_fixture_ids"])),
                        ";".join(map(str, r["extra_fixture_ids"])),
                        r["missing_truncated"], r["extra_truncated"],
                    ])
            self.stdout.write(self.style.SUCCESS(f"Wrote mismatch detail → {out_csv}"))
