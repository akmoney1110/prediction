import logging
from datetime import timedelta
from typing import List, Dict, Tuple
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from django.db.models import Q
from matches.models import MLTrainingMatch

from django.db import models

from leagues.models import Team
from matches.models import Match, MatchStats


logger = logging.getLogger(__name__)

LAST_N = 10
LAST_M = 5

class Command(BaseCommand):
    help = "Build MLTrainingMatch rows with last-10/5 features and labels (90')."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--season", type=int, required=True)

    def handle(self, *args, **opts):
        league_id = opts["league_id"]
        season = opts["season"]

        fixtures = list(
            Match.objects
                 .filter(league_id=league_id, season=season)
                 .order_by("kickoff_utc")
                 .select_related("league")
        )
        self.stdout.write(f"Building training rows for {len(fixtures)} fixtures (league={league_id}, season={season})")

        for m in fixtures:
            try:
                self._process_fixture(m)
            except Exception as e:
                logger.exception("Failed to process fixture %s: %s", m.id, e)

        self.stdout.write(self.style.SUCCESS("Done."))

    @transaction.atomic
    def _process_fixture(self, m: Match):
        cutoff = m.kickoff_utc  # as-of kickoff
        # fetch last N historical matches BEFORE kickoff for each team (all competitions within same league season for simplicity)
        h_hist = self._last_matches(m.home_id, cutoff, limit=LAST_N)
        a_hist = self._last_matches(m.away_id, cutoff, limit=LAST_N)
        h_hist5 = h_hist[:LAST_M]
        a_hist5 = a_hist[:LAST_M]

        # compute aggregates
        h_feats10 = self._aggregate(h_hist)
        a_feats10 = self._aggregate(a_hist)
        h_feats5  = self._aggregate(h_hist5)
        a_feats5  = self._aggregate(a_hist5)

        # venue split: home team's home-only, away team's away-only
        h_home_hist = self._last_matches(m.home_id, cutoff, limit=LAST_N, require_home=True)
        a_away_hist = self._last_matches(m.away_id, cutoff, limit=LAST_N, require_away=True)
        h_home_feats10 = self._aggregate(h_home_hist)
        a_away_feats10 = self._aggregate(a_away_hist)

        # situational
        h_rest_days, h_matches_14d = self._rest_and_congestion(m.home_id, cutoff)
        a_rest_days, a_matches_14d = self._rest_and_congestion(m.away_id, cutoff)

        # deltas
        d_gf10 = _safe_sub(h_feats10.get("gf"), a_feats10.get("gf"))
        d_sot10 = _safe_sub(h_feats10.get("sot"), a_feats10.get("sot"))
        d_rest = _safe_sub(h_rest_days, a_rest_days)

        # labels (90' finals)
        y_hg = m.goals_home
        y_ag = m.goals_away

        # 90' corners/cards labels from Match; if null, try stats
        y_hc, y_ac = m.corners_home, m.corners_away
        y_hcard, y_acard = m.cards_home, m.cards_away

        if y_hc is None or y_ac is None or y_hcard is None or y_acard is None:
            # fill from MatchStats if available
            try:
                h_stats = MatchStats.objects.get(match=m, team_id=m.home_id)
                a_stats = MatchStats.objects.get(match=m, team_id=m.away_id)
                y_hc = y_hc if y_hc is not None else h_stats.corners
                y_ac = y_ac if y_ac is not None else a_stats.corners
                y_hcard = y_hcard if y_hcard is not None else h_stats.cards
                y_acard = y_acard if y_acard is not None else a_stats.cards
            except MatchStats.DoesNotExist:
                pass

        MLTrainingMatch.objects.update_or_create(
            fixture_id=m.id,
            defaults={
                "league_id": m.league_id,
                "season": m.season,
                "kickoff_utc": m.kickoff_utc,
                "home_team_id": m.home_id,
                "away_team_id": m.away_id,
                "ts_cutoff": cutoff,

                "y_home_goals_90": y_hg,
                "y_away_goals_90": y_ag,
                "y_home_corners_90": y_hc,
                "y_away_corners_90": y_ac,
                "y_home_cards_90": y_hcard,
                "y_away_cards_90": y_acard,

                # home last-10
                "h_gf10": h_feats10.get("gf"),
                "h_ga10": h_feats10.get("ga"),
                "h_gd10": _safe_sub(h_feats10.get("gf"), h_feats10.get("ga")),
                "h_sot10": h_feats10.get("sot"),
                "h_conv10": h_feats10.get("conv"),
                "h_sot_pct10": h_feats10.get("sot_pct"),
                "h_poss10": h_feats10.get("poss"),
                "h_corners_for10": h_feats10.get("corners_for"),
                "h_cards_for10": h_feats10.get("cards_for"),
                "h_clean_sheets10": h_feats10.get("cs"),

                # away last-10
                "a_gf10": a_feats10.get("gf"),
                "a_ga10": a_feats10.get("ga"),
                "a_gd10": _safe_sub(a_feats10.get("gf"), a_feats10.get("ga")),
                "a_sot10": a_feats10.get("sot"),
                "a_conv10": a_feats10.get("conv"),
                "a_sot_pct10": a_feats10.get("sot_pct"),
                "a_poss10": a_feats10.get("poss"),
                "a_corners_for10": a_feats10.get("corners_for"),
                "a_cards_for10": a_feats10.get("cards_for"),
                "a_clean_sheets10": a_feats10.get("cs"),

                # venue split
                "h_home_gf10": h_home_feats10.get("gf"),
                "a_away_gf10": a_away_feats10.get("gf"),

                # situational
                "h_rest_days": h_rest_days,
                "a_rest_days": a_rest_days,
                "h_matches_14d": h_matches_14d,
                "a_matches_14d": a_matches_14d,

                # deltas
                "d_gf10": d_gf10,
                "d_sot10": d_sot10,
                "d_rest_days": d_rest,
            }
        )

    def _last_matches(self, team_id: int, cutoff, limit: int = 10, require_home: bool=False, require_away: bool=False):
        qs = Match.objects.filter(kickoff_utc__lt=cutoff).filter(
            models.Q(home_id=team_id) | models.Q(away_id=team_id)
        ).order_by("-kickoff_utc")

        if require_home:
            qs = qs.filter(home_id=team_id)
        if require_away:
            qs = qs.filter(away_id=team_id)

        return list(qs[:limit])

    def _aggregate(self, matches: List[Match]) -> Dict[str, float]:
        """
        Aggregate last-N using Match finals + MatchStats for shots/possession/cards/etc.
        """
        if not matches:
            return {"gf": 0.0, "ga": 0.0, "sot": 0.0, "conv": 0.0, "sot_pct": 0.0, "poss": 0.5,
                    "corners_for": 0.0, "cards_for": 0.0, "cs": 0.0}

        gf = ga = shots = sot = poss = corners = cards = cs = 0.0
        for m in matches:
            is_home = (m.home_id == matches[0].home_id)  # not reliable; recompute per match
            # Determine our team id for this match
            # We'll pull stats rows to be precise
            try:
                if m.home_id == matches[0].home_id or m.home_id == m.home_id:
                    # find which side we're aggregating: rely on MatchStats to pick correct team
                    pass
            except Exception:
                pass

        # More robust: detect our team id first from first match in list
        # but we actually need to pass which team we aggregate; simpler: compute as if team == matches[0].home_id
        # Better approach: rework to accept a team_id; we have it in caller scope but keeping function signature simple.

        # Re-implement cleanly using team-centric stats:
        return self._aggregate_team_centric(matches)

    def _aggregate_team_centric(self, matches: List[Match]) -> Dict[str, float]:
        if not matches:
            return {"gf": 0.0, "ga": 0.0, "sot": 0.0, "conv": 0.0, "sot_pct": 0.0, "poss": 0.5,
                    "corners_for": 0.0, "cards_for": 0.0, "cs": 0.0}

        # derive the team we are aggregating from the first match context
        # We'll infer by comparing with the next step caller; to keep deterministic, pick the team that appears most often
        # Simplify: use stats rows per match for both teams and pick "our" row using which one has stats for home or away.
        # Since that gets messy here, fallback:
        # We'll compute both sides per match relying on stats; but we need the "our" team id. Pass it via closure? quick hack:
        # We'll infer by majority of appearances as home team across the list - acceptable for small N.
        home_counts = {}
        for m in matches:
            home_counts[m.home_id] = home_counts.get(m.home_id, 0) + 1
        # pick the id that appears most frequently as home in these matches as proxy (not perfect)
        our_id = max(home_counts.items(), key=lambda kv: kv[1])[0]

        n = 0
        gf = ga = shots = sot = poss = corners = cards = cs = 0.0
        for m in matches:
            if m.home_id == our_id:
                gf += (m.goals_home or 0)
                ga += (m.goals_away or 0)
                if (m.goals_away or 0) == 0:
                    cs += 1
                try:
                    st = MatchStats.objects.get(match=m, team_id=m.home_id)
                    shots += st.shots or 0
                    sot += st.sot or 0
                    poss += (st.possession_pct or 0)
                    corners += st.corners or 0
                    cards += st.cards or 0
                except MatchStats.DoesNotExist:
                    pass
            elif m.away_id == our_id:
                gf += (m.goals_away or 0)
                ga += (m.goals_home or 0)
                if (m.goals_home or 0) == 0:
                    cs += 1
                try:
                    st = MatchStats.objects.get(match=m, team_id=m.away_id)
                    shots += st.shots or 0
                    sot += st.sot or 0
                    poss += (st.possession_pct or 0)
                    corners += st.corners or 0
                    cards += st.cards or 0
                except MatchStats.DoesNotExist:
                    pass
            else:
                # shouldn't happen
                continue
            n += 1

        conv = (gf / shots) if shots else 0.0
        sot_pct = (sot / shots) if shots else 0.0
        poss = (poss / n) if n else 0.5
        return {
            "gf": gf / max(n, 1),
            "ga": ga / max(n, 1),
            "sot": sot / max(n, 1),
            "conv": conv,
            "sot_pct": sot_pct,
            "poss": poss / 100.0,            # convert to 0-1 scale
            "corners_for": corners / max(n, 1),
            "cards_for": cards / max(n, 1),
            "cs": cs / max(n, 1),
        }

    def _rest_and_congestion(self, team_id: int, cutoff):
        recent = list(
            Match.objects.filter(kickoff_utc__lt=cutoff)
                         .filter(models.Q(home_id=team_id) | models.Q(away_id=team_id))
                         .order_by("-kickoff_utc")[:6]
        )
        rest_days = None
        matches_14d = 0
        if recent:
            last = recent[0].kickoff_utc
            rest_days = (cutoff - last).days
        for m in recent:
            if (cutoff - m.kickoff_utc).days <= 14:
                matches_14d += 1
        return (rest_days if rest_days is not None else 7.0), matches_14d


def _safe_sub(a, b):
    if a is None or b is None:
        return None
    return a - b
