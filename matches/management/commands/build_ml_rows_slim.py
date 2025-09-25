# -*- coding: utf-8 -*-
"""
A transparent, minimal, strictly as-of builder for MLTrainingMatch.

What it does (and nothing more):
- Loads fixtures for (league_id, season) ordered by kickoff_utc ASC.
- Builds per-team history windows as-of each fixture:
    * last-N per-match means  (unweighted)
    * exp-weighted means      (lambda decay)
    * venue splits (home-only for home team, away-only for away team)
    * simple 'allowed' stats  (opponent rows only; skip when missing)
    * a few derived rates     (xg/shot, SOT rate, box-share, save-rate)
    * simple situational      (rest-days, matches last 7/14 days, b2b flag)
- Writes compact JSON payloads (stats10_json, stats5_json) and a small set of
  scalar columns useful for baseline goal/corners/cards modeling.
- Optionally writes HT labels and minute-bucket JSON if your model has those fields.

Intentionally omitted (for clarity + performance):
- Classic Elo / Goal-Elo (kept as future hooks)
- SoS scaling (kept as future hook)
- Auto-tuned priors, complicated regressions, MOV factors, etc.

Run:
    python manage.py build_ml_rows_slim --league-id 61 --season 2025 \
        --last-n 10 --last-m 5 --decay 0.85 --alpha-prior 4
"""
import bisect
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from django.core.exceptions import FieldError
from django.core.management.base import BaseCommand
from django.db import transaction

from matches.models import Match, MatchStats, MLTrainingMatch

# Optional: granular event model (safe import)
try:
    from matches.models import MatchEvent  # type: ignore
except Exception:  # pragma: no cover
    MatchEvent = None

log = logging.getLogger(__name__)

# ---------- defaults (CLI overridable) ----------
LAST_N_DEFAULT = 10
LAST_M_DEFAULT = 5
DECAY_DEFAULT = 0.85
ALPHA_PRIOR_DEFAULT = 4.0

# ---------- small utils ----------
def _safe_div(a, b, default=0.0):
    try:
        if a is None or b in (None, 0):
            return default
        return float(a) / float(b)
    except Exception:
        return default

def _safe_sub(a, b):
    if a is None or b is None:
        return None
    return a - b

def _exp_weights(n: int, lam: float) -> List[float]:
    # index 0 is the most recent match (we slice newest-first)
    return [lam ** i for i in range(n)]

def _blend(sample: float, n: int, prior: float, alpha: float) -> float:
    # Bayesian shrinkage to stabilize early season
    return ((n * sample) + (alpha * prior)) / (n + alpha) if (n + alpha) else prior

def _same_id(a, b) -> bool:
    """Tolerant equality: handles int/str cross-types."""
    try:
        return int(a) == int(b)
    except Exception:
        return str(a) == str(b)

@dataclass
class TeamSlice:
    times: List  # list[datetime]
    matches: List[Match]  # same length


class Command(BaseCommand):
    help = "Slim, readable ML row builder (last-N, weighted, venue, allowed, derived, situational)."

    def add_arguments(self, p):
        p.add_argument("--league-id", type=int, required=True)
        p.add_argument("--season", type=int, required=True)
        p.add_argument("--last-n", type=int, default=LAST_N_DEFAULT)
        p.add_argument("--last-m", type=int, default=LAST_M_DEFAULT)
        p.add_argument("--decay", type=float, default=DECAY_DEFAULT)
        p.add_argument("--alpha-prior", type=float, default=ALPHA_PRIOR_DEFAULT)
        # Optional tiny QA report (does not change saved rows)
        p.add_argument("--qa", action="store_true", help="Print a short QA summary at the end.")

    # -------------------- main entry --------------------
    def handle(self, *args, **opt):
        league_id = opt["league_id"]
        season    = opt["season"]
        self.LAST_N = max(1, int(opt["last_n"]))
        self.LAST_M = max(1, int(opt["last_m"]))
        self.DECAY  = float(opt["decay"])
        self.ALPHA  = max(0.0, float(opt["alpha_prior"]))
        self.DO_QA  = bool(opt.get("qa"))

        if not (0.0 < self.DECAY <= 1.0):
            raise ValueError("--decay must be in (0,1]")

        fixtures: List[Match] = list(
            Match.objects.filter(league_id=league_id, season=season).order_by("kickoff_utc")
        )
        self.stdout.write(f"[build_ml_rows_slim] Fixtures: {len(fixtures)} (league={league_id}, season={season})")
        if not fixtures:
            return

        # Bulk-load stats (avoid N+1)
        fixture_ids = [m.id for m in fixtures]
        all_stats: List[MatchStats] = list(MatchStats.objects.filter(match_id__in=fixture_ids))

        # Group stats by match id
        self.stats_by_match: Dict[int, List[MatchStats]] = {}
        for s in all_stats:
            self.stats_by_match.setdefault(s.match_id, []).append(s)

        # Build per-team slices (ASC times) for fast slicing
        self.slice_all, self.slice_home, self.slice_away = self._build_slices(fixtures)

        # Simple league priors for shrinkage (current-season labeled).
        # NOTE: We add (home+away) goals to both gf/ga and divide by 2*matches,
        # so these are *league per-team per-match* totals (naming is historical).
        self.league_priors = self._compute_league_priors(fixtures)

        # Detect optional fields on MLTrainingMatch (safe writes)
        tm_fields = {f.name for f in MLTrainingMatch._meta.get_fields()}
        self._has_ht_labels   = {"y_ht_home", "y_ht_away"}.issubset(tm_fields)
        self._has_minute_json = "minute_labels_json" in tm_fields

        # Process fixtures in order (atomic per fixture for safety)
        processed = 0
        for m in fixtures:
            try:
                self._process_fixture(m)
                processed += 1
            except Exception as e:
                log.exception("Failed fixture %s (%s/%s @ %s): %s", m.id, m.league_id, m.season, m.kickoff_utc, e)

        self.stdout.write(self.style.SUCCESS(f"[build_ml_rows_slim] Done. Wrote {processed} rows."))

        # Optional micro QA (does not affect stored rows)
        if self.DO_QA and processed:
            self._qa_summary(league_id, season)

    # -------------------- slices --------------------
    def _build_slices(self, fixtures: List[Match]):
        by_team_all: Dict[int, List[Match]] = {}
        by_team_home: Dict[int, List[Match]] = {}
        by_team_away: Dict[int, List[Match]] = {}

        for m in fixtures:
            by_team_all.setdefault(m.home_id, []).append(m)
            by_team_all.setdefault(m.away_id, []).append(m)
            by_team_home.setdefault(m.home_id, []).append(m)
            by_team_away.setdefault(m.away_id, []).append(m)

        def mk(ts: List[Match]) -> TeamSlice:
            ts_sorted = sorted(ts, key=lambda x: x.kickoff_utc)
            return TeamSlice(times=[x.kickoff_utc for x in ts_sorted], matches=ts_sorted)

        slice_all  = {tid: mk(lst) for tid, lst in by_team_all.items()}
        slice_home = {tid: mk(lst) for tid, lst in by_team_home.items()}
        slice_away = {tid: mk(lst) for tid, lst in by_team_away.items()}
        return slice_all, slice_home, slice_away

    # -------------------- priors --------------------
    def _compute_league_priors(self, fixtures: List[Match]) -> Dict[str, float]:
        """
        Per-team per-match means across labeled games in this season.
        Used only for gentle shrinkage on noisy rates.
        """
        total = {"gf": 0.0, "ga": 0.0, "shots": 0.0, "sot": 0.0, "xg": 0.0, "corners": 0.0, "cards": 0.0}
        n = 0
        for m in fixtures:
            if m.goals_home is None or m.goals_away is None:
                continue
            total["gf"] += (m.goals_home + m.goals_away)
            total["ga"] += (m.goals_home + m.goals_away)
            n += 2
            for r in self.stats_by_match.get(m.id, []):
                total["shots"]   += float(r.shots or 0)
                total["sot"]     += float(r.sot or 0)
                total["xg"]      += float(getattr(r, "xg", 0.0) or 0.0)
                total["corners"] += float(r.corners or 0)
                total["cards"]   += float(r.cards or 0)
        denom = float(n) if n else 1.0
        return {k: total[k] / denom for k in total}

    # -------------------- per-fixture --------------------
    @transaction.atomic
    def _process_fixture(self, m: Match):
        cutoff = m.kickoff_utc
        N = self.LAST_N
        M = self.LAST_M
        lam = self.DECAY
        alpha = self.ALPHA

        # Histories (newest-first windows)
        h10_all   = self._slice_last(m.home_id, cutoff, N)
        a10_all   = self._slice_last(m.away_id, cutoff, N)
        h5_all    = h10_all[:M]
        a5_all    = a10_all[:M]
        h10_home  = self._slice_last(m.home_id, cutoff, N, venue="home")
        a10_away  = self._slice_last(m.away_id, cutoff, N, venue="away")

        # Unweighted means (with shrinkage)
        h10 = self._agg_means(m.home_id, h10_all, alpha)
        a10 = self._agg_means(m.away_id, a10_all, alpha)
        h5  = self._agg_means(m.home_id, h5_all,  alpha)
        a5  = self._agg_means(m.away_id, a5_all,  alpha)

        # Exp-weighted means
        hw = self._agg_weighted(m.home_id, h10_all, lam, alpha)
        aw = self._agg_weighted(m.away_id, a10_all, lam, alpha)

        # Venue splits (unweighted)
        h_home10 = self._agg_means(m.home_id, h10_home, alpha)
        a_away10 = self._agg_means(m.away_id, a10_away, alpha)

        # Allowed stats (skip match if opponent row missing; not in denom)
        h10_allowed = self._agg_allowed(m.home_id, h10_all, cutoff)
        a10_allowed = self._agg_allowed(m.away_id, a10_all, cutoff)
        h5_allowed  = self._agg_allowed(m.home_id, h5_all,  cutoff)
        a5_allowed  = self._agg_allowed(m.away_id, a5_all,  cutoff)
        hw_allowed  = self._agg_allowed_weighted(m.home_id, h10_all, lam, cutoff)
        aw_allowed  = self._agg_allowed_weighted(m.away_id, a10_all, lam, cutoff)

        # Derived rates (few, interpretable)
        def derived(base: Dict[str, float], allowed: Dict[str, float]) -> Dict[str, float]:
            shots = base.get("shots", 0.0)
            sot   = base.get("sot", 0.0)
            xg    = base.get("xg", 0.0)
            inbx  = base.get("shots_in_box", 0.0)
            saves = base.get("saves", 0.0)
            sot_allowed = allowed.get("sot_allowed", 0.0)
            xga        = allowed.get("xga", 0.0)
            return {
                "xg_per_shot": _safe_div(xg, shots),
                "sot_rate":    _safe_div(sot, shots),
                "box_share":   _safe_div(inbx, shots),
                "save_rate":   _safe_div(saves, sot_allowed),
                "xg_diff":     (xg - xga),
            }

        h10_drv, a10_drv = derived(h10, h10_allowed), derived(a10, a10_allowed)
        h5_drv,  a5_drv  = derived(h5,  h5_allowed),  derived(a5,  a5_allowed)
        hw_drv,  aw_drv  = derived(hw,  hw_allowed),  derived(aw,  aw_allowed)

        d10_drv = {k: _safe_sub(h10_drv.get(k), a10_drv.get(k)) for k in h10_drv.keys()}
        d5_drv  = {k: _safe_sub(h5_drv.get(k),  a5_drv.get(k))  for k in h5_drv.keys()}
        dw_drv  = {k: _safe_sub(hw_drv.get(k),  aw_drv.get(k))  for k in hw_drv.keys()}

        # Simple cross features (intuitive combos)
        home_xgps   = h10_drv["xg_per_shot"]
        away_sot_ar = _safe_div(a10_allowed.get("sot_allowed", 0.0), a10_allowed.get("shots_allowed", 0.0))
        cross_h = home_xgps - away_sot_ar

        away_xgps   = a10_drv["xg_per_shot"]
        home_sot_ar = _safe_div(h10_allowed.get("sot_allowed", 0.0), h10_allowed.get("shots_allowed", 0.0))
        cross_a = away_xgps - home_sot_ar

        # Situational (as-of)
        h_rest_days, h_matches_14d = self._rest_and_14d(m.home_id, cutoff)
        a_rest_days, a_matches_14d = self._rest_and_14d(m.away_id, cutoff)
        h_matches_7d = self._matches_in_days(m.home_id, cutoff, 7)
        a_matches_7d = self._matches_in_days(m.away_id, cutoff, 7)
        h_b2b = (h_rest_days is not None and h_rest_days <= 3)
        a_b2b = (a_rest_days is not None and a_rest_days <= 3)

        # Labels (90'): robust backfill per team if Match fields are None
        y_hg, y_ag = m.goals_home, m.goals_away
        y_hc, y_ac = m.corners_home, m.corners_away
        y_hcd, y_acd = m.cards_home, m.cards_away

        if (y_hc is None or y_ac is None) or (y_hcd is None or y_acd is None):
            rows = self.stats_by_match.get(m.id, [])
            hrow = next((r for r in rows if _same_id(r.team_id, m.home_id)), None)
            arow = next((r for r in rows if _same_id(r.team_id, m.away_id)), None)

            # corners
            if y_hc is None and hrow and hrow.corners is not None:
                y_hc = hrow.corners
            if y_ac is None and arow and arow.corners is not None:
                y_ac = arow.corners

            # cards (prefer total 'cards'; else sum yellows+reds if present)
            def _cards_total(row):
                if row is None:
                    return None
                total = getattr(row, "cards", None)
                if total is not None:
                    return total
                y = getattr(row, "yellows", None)
                r = getattr(row, "reds", None)
                if y is None and r is None:
                    return None
                return float(y or 0) + float(r or 0)

            if y_hcd is None:
                y_hcd = _cards_total(hrow)
            if y_acd is None:
                y_acd = _cards_total(arow)

        # HT + minute buckets (optional)
        y_ht_home = y_ht_away = None
        minute_labels = {}
        if self._has_ht_labels or self._has_minute_json:
            y_ht_home, y_ht_away, minute_labels = self._ht_and_minutes(m)

        # Missingness flags (how many stats rows exist for last-N window)
        h_stats_rows_10 = self._count_stats_rows(m.home_id, h10_all)
        a_stats_rows_10 = self._count_stats_rows(m.away_id, a10_all)
        h_stats_missing = bool(h_stats_rows_10 < 3)
        a_stats_missing = bool(a_stats_rows_10 < 3)

        # Compact JSON payloads
        stats10_json = {
            "shots": h10, "shots_opp": a10,
            "allowed": {"home": h10_allowed, "away": a10_allowed},
            "derived": {"home": h10_drv, "away": a10_drv, "delta": d10_drv},
            "weighted": {
                "home": hw, "away": aw,
                "home_allowed": hw_allowed, "away_allowed": aw_allowed,
                "derived": {"home": hw_drv, "away": aw_drv, "delta": dw_drv},
            },
            "venue": {"home_only": h_home10, "away_only": a_away10},
            "situational": {
                "h_rest_days": h_rest_days, "a_rest_days": a_rest_days,
                "h_matches_14d": h_matches_14d, "a_matches_14d": a_matches_14d,
                "h_matches_7d": h_matches_7d, "a_matches_7d": a_matches_7d,
                "h_back_to_back": h_b2b, "a_back_to_back": a_b2b,
            },
            "cross": {
                "home_xgps_minus_away_sot_allow_rate": cross_h,
                "away_xgps_minus_home_sot_allow_rate": cross_a,
            },
            "meta": {
                "n_h10": len(h10_all), "n_a10": len(a10_all),
                "n_h5": len(h5_all), "n_a5": len(a5_all),
                "last_n": N, "last_m": M, "decay": lam, "alpha_prior": alpha,
            },
        }
        stats5_json = {
            "shots": h5, "shots_opp": a5,
            "allowed": {"home": h5_allowed, "away": a5_allowed},
            "derived": {"home": h5_drv, "away": a5_drv, "delta": d5_drv},
            "meta": {"n_h5": len(h5_all), "n_a5": len(a5_all)},
        }

        # Compact scalar deltas you likely use downstream
        d_gf10  = _safe_sub(h10.get("gf"),  a10.get("gf"))
        d_sot10 = _safe_sub(h10.get("sot"), a10.get("sot"))
        d_rest  = _safe_sub(h_rest_days, a_rest_days)

        # Save (safe if optional fields absent)
        defaults = dict(
            league_id=m.league_id, season=m.season, kickoff_utc=m.kickoff_utc,
            home_team_id=m.home_id, away_team_id=m.away_id, ts_cutoff=cutoff,

            # labels (90')
            y_home_goals_90=y_hg, y_away_goals_90=y_ag,
            y_home_corners_90=y_hc, y_away_corners_90=y_ac,
            y_home_cards_90=y_hcd, y_away_cards_90=y_acd,

            # headline aggregates
            h_gf10=h10.get("gf"), h_ga10=h10.get("ga"),
            a_gf10=a10.get("gf"), a_ga10=a10.get("ga"),
            h_home_gf10=h_home10.get("gf"), a_away_gf10=a_away10.get("gf"),

            # small derived set + situational deltas
            d_gf10=d_gf10, d_sot10=d_sot10, d_rest_days=d_rest,
            h_rest_days=h_rest_days, a_rest_days=a_rest_days,
            h_matches_14d=h_matches_14d, a_matches_14d=a_matches_14d,

            # missingness hints
            h_stats_missing=h_stats_missing, a_stats_missing=a_stats_missing,

            # JSON payloads
            stats10_json=stats10_json, stats5_json=stats5_json,
        )
        if self._has_ht_labels:
            defaults["y_ht_home"] = y_ht_home
            defaults["y_ht_away"] = y_ht_away
        if self._has_minute_json:
            defaults["minute_labels_json"] = minute_labels

        MLTrainingMatch.objects.update_or_create(fixture_id=m.id, defaults=defaults)

    # -------------------- helpers: slicing / counts --------------------
    def _slice_last(self, team_id: int, cutoff, limit: int, venue: Optional[str] = None) -> List[Match]:
        sl = self.slice_all.get(team_id)
        if venue == "home":
            sl = self.slice_home.get(team_id, sl)
        elif venue == "away":
            sl = self.slice_away.get(team_id, sl)
        if not sl:
            return []
        idx = bisect.bisect_left(sl.times, cutoff)  # strictly before cutoff
        start = max(0, idx - limit)
        return list(reversed(sl.matches[start:idx]))  # newest-first

    def _count_stats_rows(self, team_id: int, matches: List[Match]) -> int:
        cnt = 0
        for mm in matches:
            rows = self.stats_by_match.get(mm.id, [])
            if any(_same_id(r.team_id, team_id) for r in rows):
                cnt += 1
        return cnt

    # -------------------- helpers: opponent rows (SINGLE source of truth) --------------------
    def _opp_stats_row(self, mm: Match, team_id: int):
        """
        Return stats row for the *actual* opponent team id (strict),
        else fallback only in the safe 'exactly-two-rows' case.
        """
        opp_id = mm.away_id if _same_id(mm.home_id, team_id) else mm.home_id
        rows = self.stats_by_match.get(mm.id, [])
        # 1) strict match to the true opponent id
        row = next((r for r in rows if _same_id(r.team_id, opp_id)), None)
        if row:
            return row
        # 2) safe fallback: if there are exactly two rows, pick the non-self row
        if len(rows) == 2:
            a, b = rows
            cand = a if not _same_id(a.team_id, team_id) else b
            return cand if not _same_id(cand.team_id, team_id) else None
        # 3) otherwise, give up (skip from denominator)
        return None

    # -------------------- helpers: aggregation --------------------
    def _agg_means(self, team_id: int, matches: List[Match], alpha: float) -> Dict[str, float]:
        """
        Per-match means with light shrinkage to league priors.
        Only uses fields that are commonly present in MatchStats.
        """
        if not matches:
            return {k: 0.0 for k in (
                "gf","ga","cs","shots","sot","shots_off","shots_blocked",
                "shots_in_box","shots_out_box","fouls","offsides","saves",
                "passes_total","passes_accurate","pass_acc",
                "poss","corners","cards","yellows","reds","xg","conv","sot_pct"
            )}

        lp = self.league_priors
        n = 0
        gf = ga = cs = 0.0
        shots = sot = shots_off = shots_blocked = shots_in = shots_out = 0.0
        fouls = offsides = saves = 0.0
        passes_total = passes_acc = 0.0
        poss_sum = pass_acc_pct_sum = 0.0
        corners = cards = yellows = reds = 0.0
        xg_sum = 0.0

        for mm in matches:
            if _same_id(mm.home_id, team_id):
                gf += float(mm.goals_home or 0); ga += float(mm.goals_away or 0)
                if (mm.goals_away or 0) == 0: cs += 1.0
            else:
                gf += float(mm.goals_away or 0); ga += float(mm.goals_home or 0)
                if (mm.goals_home or 0) == 0: cs += 1.0

            st = next((r for r in self.stats_by_match.get(mm.id, []) if _same_id(r.team_id, team_id)), None)
            if st:
                shots          += float(st.shots or 0);      sot     += float(st.sot or 0)
                shots_off      += float(getattr(st, "shots_off", 0) or 0)
                shots_blocked  += float(getattr(st, "shots_blocked", 0) or 0)
                shots_in       += float(getattr(st, "shots_in_box", 0) or 0)
                shots_out      += float(getattr(st, "shots_out_box", 0) or 0)
                fouls          += float(getattr(st, "fouls", 0) or 0)
                offsides       += float(getattr(st, "offsides", 0) or 0)
                saves          += float(getattr(st, "saves", 0) or 0)
                passes_total   += float(getattr(st, "passes_total", 0) or 0)
                passes_acc     += float(getattr(st, "passes_accurate", 0) or 0)
                poss_sum       += float(getattr(st, "possession_pct", 0.0) or 0.0)
                pass_acc_pct_sum += float(getattr(st, "pass_acc_pct", 0.0) or 0.0)
                corners        += float(st.corners or 0);    cards   += float(st.cards or 0)
                yellows        += float(getattr(st, "yellows", 0) or 0)
                reds           += float(getattr(st, "reds", 0) or 0)
                xg_sum         += float(getattr(st, "xg", 0.0) or 0.0)
            n += 1

        denom = float(max(1, n))
        out = {
            "gf": gf/denom, "ga": ga/denom, "cs": cs/denom,
            "shots": shots/denom, "sot": sot/denom,
            "shots_off": shots_off/denom, "shots_blocked": shots_blocked/denom,
            "shots_in_box": shots_in/denom, "shots_out_box": shots_out/denom,
            "fouls": fouls/denom, "offsides": offsides/denom, "saves": saves/denom,
            "passes_total": passes_total/denom, "passes_accurate": passes_acc/denom,
            "pass_acc": (pass_acc_pct_sum/denom)/100.0 if n else 0.0,
            "poss": (poss_sum/denom)/100.0 if n else 0.5,
            "corners": corners/denom, "cards": cards/denom,
            "yellows": yellows/denom, "reds": reds/denom,
            "xg": xg_sum/denom,
        }
        out["conv"]    = _safe_div(out["gf"], out["shots"])
        out["sot_pct"] = _safe_div(out["sot"], out["shots"])

        # Gentle shrinkage only on the “rates” that get noisy early
        for k in ("gf","ga","shots","sot","xg","corners","cards"):
            out[k] = _blend(out[k], n, lp.get(k, 0.0), alpha=alpha)
        return out

    def _agg_weighted(self, team_id: int, matches: List[Match], lam: float, alpha: float) -> Dict[str, float]:
        if not matches:
            return self._agg_means(team_id, matches, alpha)
        w = _exp_weights(len(matches), lam)
        lp = self.league_priors

        gf = ga = cs = 0.0
        shots = sot = shots_off = shots_blocked = shots_in = shots_out = 0.0
        fouls = offsides = saves = 0.0
        passes_total = passes_acc = 0.0
        poss_sum = pass_acc_pct_sum = 0.0
        corners = cards = yellows = reds = 0.0
        xg_sum = 0.0
        wtot = 0.0

        for i, mm in enumerate(matches):
            ww = w[i]
            if _same_id(mm.home_id, team_id):
                gf += ww * float(mm.goals_home or 0); ga += ww * float(mm.goals_away or 0)
                if (mm.goals_away or 0) == 0: cs += ww
            else:
                gf += ww * float(mm.goals_away or 0); ga += ww * float(mm.goals_home or 0)
                if (mm.goals_home or 0) == 0: cs += ww

            st = next((r for r in self.stats_by_match.get(mm.id, []) if _same_id(r.team_id, team_id)), None)
            if st:
                shots          += ww * float(st.shots or 0);    sot     += ww * float(st.sot or 0)
                shots_off      += ww * float(getattr(st,"shots_off",0) or 0)
                shots_blocked  += ww * float(getattr(st,"shots_blocked",0) or 0)
                shots_in       += ww * float(getattr(st,"shots_in_box",0) or 0)
                shots_out      += ww * float(getattr(st,"shots_out_box",0) or 0)
                fouls          += ww * float(getattr(st,"fouls",0) or 0)
                offsides       += ww * float(getattr(st,"offsides",0) or 0)
                saves          += ww * float(getattr(st,"saves",0) or 0)
                passes_total   += ww * float(getattr(st,"passes_total",0) or 0)
                passes_acc     += ww * float(getattr(st,"passes_accurate",0) or 0)
                poss_sum       += ww * float(getattr(st,"possession_pct",0.0) or 0.0)
                pass_acc_pct_sum += ww * float(getattr(st,"pass_acc_pct",0.0) or 0.0)
                corners        += ww * float(st.corners or 0);  cards   += ww * float(st.cards or 0)
                yellows        += ww * float(getattr(st,"yellows",0) or 0)
                reds           += ww * float(getattr(st,"reds",0) or 0)
                xg_sum         += ww * float(getattr(st,"xg",0.0) or 0.0)
            wtot += ww

        wtot = wtot or 1.0
        out = {
            "gf": gf/wtot, "ga": ga/wtot, "cs": cs/wtot,
            "shots": shots/wtot, "sot": sot/wtot,
            "shots_off": shots_off/wtot, "shots_blocked": shots_blocked/wtot,
            "shots_in_box": shots_in/wtot, "shots_out_box": shots_out/wtot,
            "fouls": fouls/wtot, "offsides": offsides/wtot, "saves": saves/wtot,
            "passes_total": passes_total/wtot, "passes_accurate": passes_acc/wtot,
            "pass_acc": _safe_div(pass_acc_pct_sum, wtot) * 0.01,
            "poss": _safe_div(poss_sum, wtot) * 0.01 or 0.5,
            "corners": corners/wtot, "cards": cards/wtot,
            "yellows": yellows/wtot, "reds": reds/wtot,
            "xg": xg_sum/wtot,
        }
        out["conv"]    = _safe_div(out["gf"], out["shots"])
        out["sot_pct"] = _safe_div(out["sot"], out["shots"])

        for k in ("gf","ga","shots","sot","xg","corners","cards"):
            # Downweight the 'n' a little so shrinkage still helps with heavy decay
            out[k] = _blend(out[k], int(min(len(matches), 10)), lp.get(k, 0.0), alpha=alpha)
        return out

    def _agg_allowed(self, team_id: int, matches: List[Match], cutoff, weighted_w: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Opponent-facing averages. Skip games with missing opponent row; don't include them in denom.
        If weighted_w provided, use those weights; otherwise unweighted mean.
        """
        if not matches:
            return {k: 0.0 for k in (
                "shots_allowed","sot_allowed","shots_off_allowed","shots_blocked_allowed",
                "shots_in_box_allowed","shots_out_box_allowed","fouls_drawn","offsides_against",
                "saves_opponent","passes_total_against","passes_accurate_against",
                "corners_against","cards_against","yellows_against","reds_against","xga"
            )}

        shots = sot = shots_off = shots_blocked = shots_in = shots_out = 0.0
        fouls = offsides = saves_opp = 0.0
        passes_total = passes_acc = 0.0
        corners = cards = yellows = reds = 0.0
        xga = 0.0
        wtot = 0.0

        for i, mm in enumerate(matches):
            opp = self._opp_stats_row(mm, team_id)
            if not opp:
                continue
            w = 1.0 if weighted_w is None else weighted_w[i]
            shots         += w * float(opp.shots or 0)
            sot           += w * float(opp.sot or 0)
            shots_off     += w * float(getattr(opp, "shots_off", 0) or 0)
            shots_blocked += w * float(getattr(opp, "shots_blocked", 0) or 0)
            shots_in      += w * float(getattr(opp, "shots_in_box", 0) or 0)
            shots_out     += w * float(getattr(opp, "shots_out_box", 0) or 0)
            fouls         += w * float(getattr(opp, "fouls", 0) or 0)
            offsides      += w * float(getattr(opp, "offsides", 0) or 0)
            saves_opp     += w * float(getattr(opp, "saves", 0) or 0)
            passes_total  += w * float(getattr(opp, "passes_total", 0) or 0)
            passes_acc    += w * float(getattr(opp, "passes_accurate", 0) or 0)
            corners       += w * float(opp.corners or 0)
            cards         += w * float(opp.cards or 0)
            yellows       += w * float(getattr(opp, "yellows", 0) or 0)
            reds          += w * float(getattr(opp, "reds", 0) or 0)
            xga           += w * float(getattr(opp, "xg", 0.0) or 0.0)
            wtot          += w

        wtot = wtot or 1.0
        return {
            "shots_allowed": shots/wtot, "sot_allowed": sot/wtot,
            "shots_off_allowed": shots_off/wtot, "shots_blocked_allowed": shots_blocked/wtot,
            "shots_in_box_allowed": shots_in/wtot, "shots_out_box_allowed": shots_out/wtot,
            "fouls_drawn": fouls/wtot, "offsides_against": offsides/wtot, "saves_opponent": saves_opp/wtot,
            "passes_total_against": passes_total/wtot, "passes_accurate_against": passes_acc/wtot,
            "corners_against": corners/wtot, "cards_against": cards/wtot,
            "yellows_against": yellows/wtot, "reds_against": reds/wtot,
            "xga": xga/wtot,
        }

    def _agg_allowed_weighted(self, team_id: int, matches: List[Match], lam: float, cutoff) -> Dict[str, float]:
        if not matches:
            return self._agg_allowed(team_id, matches, cutoff)
        weights = _exp_weights(len(matches), lam)
        return self._agg_allowed(team_id, matches, cutoff, weighted_w=weights)

    # -------------------- helpers: HT + minutes (optional) --------------------
    def _ht_and_minutes(self, m: Match) -> Tuple[Optional[int], Optional[int], Dict[str, List[int]]]:
        y_ht_home, y_ht_away = self._extract_ht_score(m)
        buckets = self._extract_minute_events(m)
        if (y_ht_home is None or y_ht_away is None):
            # If no HT stored but we have first-half goal events, derive HT
            home_first = [x for x in buckets.get("goal_minutes_home", []) if x <= 45]
            away_first = [x for x in buckets.get("goal_minutes_away", []) if x <= 45]
            if home_first or away_first:
                y_ht_home = len(home_first)
                y_ht_away = len(away_first)
        return y_ht_home, y_ht_away, buckets

    def _extract_ht_score(self, m: Match) -> Tuple[Optional[int], Optional[int]]:
        pairs = [("ht_goals_home","ht_goals_away"), ("goals_home_ht","goals_away_ht")]
        for h,a in pairs:
            if hasattr(m,h) and hasattr(m,a):
                try:
                    yh, ya = getattr(m,h), getattr(m,a)
                    if yh is not None and ya is not None:
                        return int(yh), int(ya)
                except Exception:
                    pass
        for fld in ["ht_score","score_ht","half_time_score","score_ht_string"]:
            s = getattr(m,fld,None)
            if isinstance(s,str) and s.strip():
                s2 = s.replace("–","-").replace("−","-").replace(":", "-")
                if "-" in s2:
                    L,R = s2.split("-",1)
                    try:
                        return int(L.strip()), int(R.strip())
                    except Exception:
                        pass
        return (None, None)

    def _extract_minute_events(self, m: Match) -> Dict[str, List[int]]:
        out = {
            "goal_minutes_home":[], "goal_minutes_away":[],
            "corner_minutes_home":[], "corner_minutes_away":[],
            "yellow_minutes_home":[], "yellow_minutes_away":[],
            "red_minutes_home":[], "red_minutes_away":[]
        }
        if MatchEvent is None:
            return out
        # Try common foreign-keys
        q = None
        try:
            q = MatchEvent.objects.filter(match_id=m.id)
        except (FieldError, Exception):
            try:
                q = MatchEvent.objects.filter(match__id=m.id)
            except (FieldError, Exception):
                try:
                    q = MatchEvent.objects.filter(match=m)
                except Exception:
                    q = None
        if q is None:
            return out

        for ev in q.iterator():
            minute = getattr(ev,"minute", None)
            if minute is None:
                minute = getattr(ev, "elapsed", None)
            try:
                minute = int(minute)
            except Exception:
                continue
            if not (1 <= minute <= 120):
                continue
            # side
            side = None
            is_home = getattr(ev,"is_home", None)
            if is_home is not None:
                side = "home" if is_home else "away"
            else:
                tid = getattr(ev, "team_id", None)
                if tid is None:
                    tobj = getattr(ev, "team", None)
                    tid = getattr(tobj, "id", None) if tobj is not None else None
                if _same_id(tid, m.home_id):
                    side = "home"
                elif _same_id(tid, m.away_id):
                    side = "away"
                else:
                    continue
            et = str(getattr(ev, "type", None) or getattr(ev, "event_type", None) or getattr(ev, "code", None) or "").lower()
            detail = str(getattr(ev,"detail","") or "").lower()
            if "goal" in et and not getattr(ev,"is_missed_penalty", False):
                out[f"goal_minutes_{side}"].append(minute)
            elif "corner" in et or "corner" in detail:
                out[f"corner_minutes_{side}"].append(minute)
            elif "yellow" in et or "yellow" in detail or "card" in et or "card" in detail:
                out[f"yellow_minutes_{side}"].append(minute)
            elif "red" in et or "red" in detail:
                out[f"red_minutes_{side}"].append(minute)
        for k in out:
            out[k] = sorted(set(out[k]))
        return out

    # -------------------- helpers: situational --------------------
    def _rest_and_14d(self, team_id: int, cutoff):
        sl = self.slice_all.get(team_id)
        if not sl:
            return (7.0, 0)
        idx = bisect.bisect_left(sl.times, cutoff)
        recent = sl.matches[max(0, idx-6):idx]
        rest_days = None
        m14 = 0
        if recent:
            last = recent[-1].kickoff_utc
            rest_days = (cutoff - last).days
        for mm in recent:
            if (cutoff - mm.kickoff_utc).days <= 14:
                m14 += 1
        return (rest_days if rest_days is not None else 7.0), m14

    def _matches_in_days(self, team_id: int, cutoff, days: int) -> int:
        sl = self.slice_all.get(team_id)
        if not sl:
            return 0
        since = cutoff - timedelta(days=days)
        lo = bisect.bisect_left(sl.times, since)
        hi = bisect.bisect_left(sl.times, cutoff)
        return max(0, hi - lo)

    # -------------------- micro QA summary (optional) --------------------
    def _qa_summary(self, league_id: int, season: int) -> None:
        """
        Small, read-only sanity report so you spot obvious data issues quickly.
        Does not change any stored rows.
        """
        try:
            qs = MLTrainingMatch.objects.filter(league_id=league_id, season=season)
            n = qs.count()
            if not n:
                self.stdout.write("[QA] No rows to summarize.")
                return
            # Pull a tiny sample's JSON to inspect allowed stats health
            some = qs.values_list("stats10_json", flat=True)[:200]
            tot = {
                "shots_allowed": 0.0, "sot_allowed": 0.0, "xga": 0.0,
                "shots": 0.0, "sot": 0.0, "xg": 0.0
            }
            c = 0
            for js in some:
                try:
                    allowed_h = js.get("allowed", {}).get("home", {}) or {}
                    shots = js.get("shots", {}) or {}
                    tot["shots_allowed"] += float(allowed_h.get("shots_allowed", 0.0) or 0.0)
                    tot["sot_allowed"]   += float(allowed_h.get("sot_allowed", 0.0) or 0.0)
                    tot["xga"]           += float(allowed_h.get("xga", 0.0) or 0.0)
                    tot["shots"]         += float(shots.get("shots", 0.0) or 0.0)
                    tot["sot"]           += float(shots.get("sot", 0.0) or 0.0)
                    tot["xg"]            += float(shots.get("xg", 0.0) or 0.0)
                    c += 1
                except Exception:
                    continue
            if c:
                avg = {k: v / c for k, v in tot.items()}
                self.stdout.write("[QA] ~Averages over sample of {} rows: {}".format(
                    c, json.dumps(avg, indent=2)))
        except Exception as e:
            self.stdout.write(f"[QA] Skipped (error: {e})")
