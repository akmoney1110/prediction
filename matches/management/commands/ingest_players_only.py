from django.core.management.base import BaseCommand
from leagues.models import Team
from matches.management.commands.ingest_new_season import Command as BigCmd

class Command(BaseCommand):
    help = "Ingest players only for a league+season (uses the existing helper)."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--season", type=int, required=True)
        parser.add_argument("--team-id", type=int, help="If set, only this team")

    def handle(self, *args, **opts):
        league_id = opts["league_id"]
        season = opts["season"]
        team_id = opts.get("team_id")

        big = BigCmd()
        big.league_id = league_id

        if team_id:
            self.stdout.write(f"Ingesting players for team {team_id}…")
            big._ingest_players_for_team(team_id=team_id, season=season, league_id=league_id)
        else:
            self.stdout.write(f"Ingesting players for all teams in league {league_id}…")
            for tid in Team.objects.filter(league_id=league_id).values_list("id", flat=True):
                big._ingest_players_for_team(team_id=tid, season=season, league_id=league_id)

        self.stdout.write(self.style.SUCCESS("Players done."))
