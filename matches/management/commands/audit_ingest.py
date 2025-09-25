# matches/management/commands/audit_ingest.py
import sys
from typing import List

from django.core.management.base import BaseCommand
from django.db.models import Count, Q, Exists, OuterRef
from django.core.exceptions import FieldError

from matches.models import Match, MatchStats, MatchEvent, Lineup, PlayerSeason, Injury, Transfer


class Command(BaseCommand):
    help = "Audit that league+season ingest is complete (fixtures, stats, events, lineups, corners, players, injuries, transfers)."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--season", type=int, required=True)
        parser.add_argument(
            "--strict-playerseason-injuries",
            action="store_true",
            help="Count injuries only for players who have a PlayerSeason row for this league+season.",
        )
        parser.add_argument("--sample-limit", type=int, default=20)

    def handle(self, *args, **opts):
        lid = int(opts["league_id"])
        season = int(opts["season"])
        strict_ps = bool(opts["strict_playerseason_injuries"])
        sample_limit = int(opts["sample_limit"])

        self.stdout.write(self.style.NOTICE(f"AUDIT league={lid} season={season}"))

        # ---- Fixtures
        total = Match.objects.filter(league_id=lid, season=season).count()
        ft = Match.objects.filter(league_id=lid, season=season, status__iexact="FT").count()
        self.stdout.write(f"Fixtures: {total} | FT: {ft}")

        # ---- Per-team stats (require both teams present)
        stats_per_match = (
            MatchStats.objects.filter(match__league_id=lid, match__season=season)
            .values("match_id")
            .annotate(n=Count("id"))
        )
        with_stats_2 = sum(1 for r in stats_per_match if r["n"] >= 2)
        self.stdout.write(f"Matches with stats for both teams: {with_stats_2} / {total}")

        # ---- Events presence (DB-agnostic)
        with_events = self._count_with_events(lid, season)
        self.stdout.write(f"Matches with ANY events: {with_events} / {total}")

        ft_no_events = self._ft_no_events_ids(lid, season)
        self.stdout.write(f"FT matches missing events: {len(ft_no_events)}")
        if ft_no_events:
            self.stdout.write("  sample ids: " + ", ".join(map(str, ft_no_events[:sample_limit])))

        # ---- Corners
        corn_labeled = Match.objects.filter(
            league_id=lid, season=season,
            corners_home__isnull=False, corners_away__isnull=False
        ).count()
        self.stdout.write(f"Matches with corner totals: {corn_labeled} / {total}")

        # ---- Lineups (both teams)
        lineups_by_match = (
            Lineup.objects.filter(match__league_id=lid, match__season=season)
            .values("match_id")
            .annotate(n=Count("id"))
        )
        with_lineups = sum(1 for r in lineups_by_match if r["n"] >= 2)
        self.stdout.write(f"Matches with both lineups: {with_lineups} / {total}")

        # ---- Players & extras
        players_rows = PlayerSeason.objects.filter(season=season, team__league_id=lid).count()

        # >>> FIXED: Injury counts via fixture_id IN {match ids} <<<
        match_ids_qs = Match.objects.filter(league_id=lid, season=season).values_list("id", flat=True)

        if strict_ps:
            injuries = (
                Injury.objects.annotate(
                    has_ps=Exists(
                        PlayerSeason.objects.filter(
                            player_id=OuterRef("player_id"),
                            team__league_id=lid,
                            season=season,
                        )
                    )
                )
                .filter(fixture_id__in=match_ids_qs, has_ps=True)
                .count()
            )
        else:
            injuries = Injury.objects.filter(fixture_id__in=match_ids_qs).count()

        transfers = (
            Transfer.objects.filter(to_team__league_id=lid).count()
            + Transfer.objects.filter(from_team__league_id=lid).count()
        )
        self.stdout.write(
            f"PlayerSeason rows: {players_rows} | Injuries: {injuries} | Transfers touching league: {transfers}"
        )

        # ---- Missing stats sample
        missing_stats_ids = self._missing_stats_ids(lid, season)
        self.stdout.write(f"Matches missing BOTH teams' stats: {len(missing_stats_ids)}")
        if missing_stats_ids:
            self.stdout.write("  sample ids: " + ", ".join(map(str, missing_stats_ids[:sample_limit])))

        self.stdout.write(self.style.SUCCESS("Audit complete."))

    # ---------------- helpers ----------------

    def _count_with_events(self, league_id: int, season: int) -> int:
        try:
            return (
                Match.objects.filter(league_id=league_id, season=season)
                .filter(Q(raw_result_json__has_key="events") | Q(id__in=MatchEvent.objects.values("match_id")))
                .distinct()
                .count()
            )
        except FieldError:
            return (
                Match.objects.filter(league_id=league_id, season=season)
                .filter(id__in=MatchEvent.objects.values("match_id"))
                .distinct()
                .count()
            )

    def _ft_no_events_ids(self, league_id: int, season: int) -> List[int]:
        ids = []
        qs = Match.objects.filter(league_id=league_id, season=season, status__iexact="FT").only("id", "raw_result_json")
        for m in qs.iterator():
            raw = m.raw_result_json or {}
            ev = raw.get("events")
            if not ev or len(ev or []) == 0:
                ids.append(m.id)
        return ids

    def _missing_stats_ids(self, league_id: int, season: int) -> List[int]:
        missing = []
        qs = Match.objects.filter(league_id=league_id, season=season).values_list("id", flat=True)
        for mid in qs.iterator():
            if MatchStats.objects.filter(match_id=mid).count() < 2:
                missing.append(mid)
        return missing
