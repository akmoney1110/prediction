import logging
from datetime import datetime, timezone
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from leagues.models import League, Team
from matches.models import Match, MatchStats, Lineup
from services import apifootball as api

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Ingest fixtures + stats + lineups + events for one league and one season."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True, help="API-Football league id")
        parser.add_argument("--season", type=int, required=True, help="Season start year, e.g., 2023")

    def handle(self, *args, **opts):
        league_id = opts["league_id"]
        season = opts["season"]

        # Ensure league row exists (you can preseed more metadata elsewhere)
        League.objects.get_or_create(id=league_id, defaults={
            "display_name": f"League {league_id}",
            "country": "Unknown",
            "tier": 1,
            "has_xg": False,
            "has_multi_group": False,
        })

        from datetime import date

        self.stdout.write(self.style.NOTICE(f"Fetching fixtures league={league_id}, season={season}"))
        count_inserted = 0

        data = api.fixtures_by_league_season(league_id, season)
        fixtures = data.get("response", []) or []
        self.stdout.write(f"  Season call returned: {len(fixtures)} fixtures")

        # Fallback: pull by date window if season returns 0
        if not fixtures:
            self.stdout.write(self.style.WARNING("No fixtures via season param; trying date-window fallback..."))
            # Typical European season: Jul 1 → Jul 1 next year
            start = f"{season}-07-01"
            end   = f"{season+1}-07-01"
            data = api.fixtures_by_league_between(league_id, start, end)
            fixtures = data.get("response", []) or []
            self.stdout.write(f"  Date-window returned: {len(fixtures)} fixtures")

        # Optional: chunk by month if still 0 (some plans prefer smaller windows)
        if not fixtures:
            self.stdout.write(self.style.WARNING("Chunking by month (Jul..Jun)..."))
            chunked = []
            months = [("07","31"),("08","31"),("09","30"),("10","31"),("11","30"),("12","31"),
              ("01","31"),("02","29"),("03","31"),("04","30"),("05","31"),("06","30")]
            for m, last in months:
                y = season if int(m) >= 7 else season+1
                frm = f"{y}-{m}-01"
                to  = f"{y}-{m}-{last}"
                d = api.fixtures_by_league_between(league_id, frm, to).get("response", []) or []
                chunked.extend(d)
            # de-dup by fixture id
            seen = set(); dedup = []
            for fx in chunked:
                fid = fx.get("fixture", {}).get("id")
                if fid and fid not in seen:
                    seen.add(fid); dedup.append(fx)
            fixtures = dedup
            self.stdout.write(f"  Monthly chunks returned: {len(fixtures)} fixtures")

        # Upsert what we have
        for fx in fixtures:
            try:
                self._upsert_fixture_block(fx); count_inserted += 1
            except Exception:
                logger.exception("Failed to upsert fixture %s", fx.get("fixture", {}).get("id"))

        self.stdout.write(self.style.SUCCESS(f"Saved/updated {count_inserted} fixtures"))
        self.stdout.write(self.style.NOTICE("Fetching per-fixture details (stats, lineups, events)..."))


        # Now fetch stats/lineups/events for each fixture id
        fixture_ids = list(Match.objects.filter(league_id=league_id, season=season).values_list("id", flat=True))
        for fid in fixture_ids:
            try:
                self._ingest_fixture_details(fid)
            except Exception:
                logger.exception("Failed details for fixture %s", fid)

        self.stdout.write(self.style.SUCCESS("Ingestion complete."))

    @transaction.atomic
    def _upsert_fixture_block(self, fx: dict):
        fixture = fx["fixture"]
        league = fx["league"]
        teams = fx["teams"]
        goals = fx.get("goals", {})

        fixture_id = fixture["id"]
        season = league["season"]
        status = fixture["status"]["short"]
        kickoff_iso = fixture["date"]   # ISO8601
        kickoff_utc = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))

        # teams
        home_team = teams["home"]; away_team = teams["away"]
        home_id = home_team["id"]; away_id = away_team["id"]

        # Make sure Team rows exist
        h_team, _ = Team.objects.get_or_create(id=home_id, defaults={
            "league_id": league["id"],
            "name": home_team["name"],
            "short_name": home_team.get("name", "")[:32],
            "logo_url": home_team.get("logo"),
        })
        a_team, _ = Team.objects.get_or_create(id=away_id, defaults={
            "league_id": league["id"],
            "name": away_team["name"],
            "short_name": away_team.get("name", "")[:32],
            "logo_url": away_team.get("logo"),
        })

        # Match upsert
        m, _created = Match.objects.update_or_create(
            id=fixture_id,
            defaults={
                "league_id": league["id"],
                "season": season,
                "home_id": home_id,
                "away_id": away_id,
                "kickoff_utc": kickoff_utc,
                "status": status,
                "goals_home": goals.get("home"),
                "goals_away": goals.get("away"),
                "raw_result_json": fx,  # store raw for audit
            },
        )

    def _ingest_fixture_details(self, fixture_id: int):
        # --- Statistics (team split) ---
        stats_data = api.fixture_statistics(fixture_id)
        stats_rows = 0
        for entry in stats_data.get("response", []) or []:
            try:
                team = entry["team"]["id"]
                stats_map = {s["type"]: s.get("value") for s in entry.get("statistics", []) or []}

                def _pct(v):
                    if v is None: return None
                    if isinstance(v, (int, float)): return float(v)
                    return float(str(v).replace("%", ""))

                MatchStats.objects.update_or_create(
                    match_id=fixture_id,
                    team_id=team,
                    defaults={
                    "shots": stats_map.get("Total Shots") or stats_map.get("Shots Total"),
                    "sot": stats_map.get("Shots on Goal") or stats_map.get("Shots on Target") or stats_map.get("Shots on target"),
                    "possession_pct": _pct(stats_map.get("Ball Possession") or stats_map.get("Possession")),
                    "pass_acc_pct": _pct(stats_map.get("Passes %") or stats_map.get("Pass Accuracy")),
                    "corners": stats_map.get("Corner Kicks") or stats_map.get("Corners"),
                    "cards": (stats_map.get("Yellow Cards") or 0) + (stats_map.get("Red Cards") or 0),
                    "xg": stats_map.get("expected_goals") or stats_map.get("xG"),
                    "pens_won": None,
                    "pens_conceded": None,
                    "reds": stats_map.get("Red Cards") or stats_map.get("Red cards"),
                    "yellows": stats_map.get("Yellow Cards") or stats_map.get("Yellow cards"),
                },
            )
                stats_rows += 1
            except Exception:
                # keep going; some responses are partial
                continue

        # --- Lineups (optional) ---
        lu_data = api.fixture_lineups(fixture_id)
        for entry in lu_data.get("response", []) or []:
            try:
                Lineup.objects.update_or_create(
                    match_id=fixture_id,
                    team_id=entry["team"]["id"],
                    defaults={
                    "formation": entry.get("formation"),
                    "starters_json": entry.get("startXI") or [],
                    "bench_json": entry.get("substitutes") or [],
                },
            )
            except Exception:
                continue

       
        ev_data = api.fixture_events(fixture_id)
        match = Match.objects.get(pk=fixture_id)
        raw = match.raw_result_json or {}
        raw["events"] = ev_data.get("response", []) or []
        match.raw_result_json = raw

        # If we got per-team stats, hydrate corners/cards on Match (if missing)
        if stats_rows >= 2:
            try:
                h_stats = MatchStats.objects.get(match_id=fixture_id, team_id=match.home_id)
                a_stats = MatchStats.objects.get(match_id=fixture_id, team_id=match.away_id)
                if match.corners_home is None and h_stats.corners is not None and a_stats.corners is not None:
                    match.corners_home = h_stats.corners
                    match.corners_away = a_stats.corners
                if match.cards_home is None and h_stats.cards is not None and a_stats.cards is not None:
                    match.cards_home = h_stats.cards
                    match.cards_away = a_stats.cards
            except MatchStats.DoesNotExist:
                pass

        match.save()
