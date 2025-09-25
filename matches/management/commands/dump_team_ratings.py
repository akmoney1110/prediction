# prediction/matches/management/commands/dump_team_ratings.py
import json
from dataclasses import asdict, dataclass
from typing import Optional, List

from django.core.management.base import BaseCommand
from django.db.models import Q

from matches.models import TeamRating

# If you have a Team model to resolve names -> ids, import it.
try:
    from matches.models import Team
except Exception:
    Team = None  # command still works if you pass --team-id


@dataclass
class RatingRow:
    league_id: int
    season: int
    team_id: int
    team_name: Optional[str]
    attack: float
    defense: float


def _estimate_league_home_adv(league_id: int, seasons: Optional[List[int]]):
    """Average (home_goals - away_goals) across training data as a quick HFA prior."""
    try:
        from matches.models import MLTrainingMatch
    except Exception:
        return 0.0

    qs = MLTrainingMatch.objects.filter(league_id=league_id)
    if seasons:
        qs = qs.filter(season__in=seasons)
    qs = qs.filter(y_home_goals_90__isnull=False, y_away_goals_90__isnull=False)

    vals = []
    for r in qs.only("y_home_goals_90", "y_away_goals_90"):
        try:
            h = float(r.y_home_goals_90 or 0.0)
            a = float(r.y_away_goals_90 or 0.0)
            vals.append(h - a)
        except Exception:
            pass
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _resolve_team_ids_by_name(name: str, league_id: int, season: int) -> List[int]:
    """Return a list of plausible team_ids for a fuzzy name."""
    if Team is None:
        return []
    # Try exact first, then icontains
    exact = Team.objects.filter(name__iexact=name).values_list("id", flat=True)
    if exact:
        return list(exact)
    fuzzy = Team.objects.filter(name__icontains=name).values_list("id", flat=True)
    return list(fuzzy)


class Command(BaseCommand):
    help = "Dump TeamRating attack/defense for a league/season (optionally a specific team), plus league HFA."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--season", type=int, required=True)
        parser.add_argument("--team", type=str, default=None,
                            help="Team name (will try exact then fuzzy match via Team model)")
        parser.add_argument("--team-id", type=int, default=None,
                            help="Team id (skip name resolution)")
        parser.add_argument("--json", action="store_true", help="Output JSON instead of text table")
        parser.add_argument("--limit", type=int, default=0, help="Limit number of rows (for --all)")

        # convenience switch: if neither --team nor --team-id, we dump all teams for the league/season
        parser.add_argument("--all", action="store_true", help="Dump all team ratings for league/season")

    def handle(self, *args, **opts):
        league_id = opts["league_id"]
        season = opts["season"]
        want_json = bool(opts["json"])
        limit = int(opts["limit"] or 0)

        team_param = opts.get("team")
        team_id_param = opts.get("team_id")
        dump_all = bool(opts.get("all") or (not team_param and not team_id_param))

        # Build queryset
        qs = TeamRating.objects.filter(league_id=league_id, season=season)

        if team_id_param is not None:
            qs = qs.filter(team_id=team_id_param)
        elif team_param:
            ids = _resolve_team_ids_by_name(team_param, league_id, season)
            if not ids:
                self.stderr.write(self.style.WARNING(
                    f"No team match for '{team_param}'. Try --team-id instead."
                ))
                return
            qs = qs.filter(team_id__in=ids)
        else:
            # dump_all path
            pass

        if limit > 0:
            qs = qs.order_by("team_id")[:limit]
        else:
            qs = qs.order_by("team_id")

        rows = list(qs)
        if not rows:
            self.stdout.write("No TeamRating rows found for the given filters.")
            return

        # Optionally fetch names for nicer output
        id_to_name = {}
        if Team is not None:
            ids = [r.team_id for r in rows]
            for tid, name in Team.objects.filter(id__in=ids).values_list("id", "name"):
                id_to_name[int(tid)] = name

        # Compose output
        out = []
        for r in rows:
            out.append(RatingRow(
                league_id=league_id,
                season=season,
                team_id=int(r.team_id),
                team_name=id_to_name.get(int(r.team_id)),
                attack=float(r.attack),
                defense=float(r.defense),
            ))

        # League HFA (quick prior)
        hfa = _estimate_league_home_adv(league_id, [season])

        if want_json:
            payload = {
                "league_id": league_id,
                "season": season,
                "home_field_advantage": hfa,
                "count": len(out),
                "ratings": [asdict(o) for o in out],
            }
            self.stdout.write(json.dumps(payload, indent=2))
            return

        # Pretty text table
        self.stdout.write(f"League {league_id} Season {season} | HFA ~ {hfa:.3f}")
        self.stdout.write(f"{'Team ID':>8}  {'Team':<28}  {'Attack':>8}  {'Defense':>8}")
        self.stdout.write("-" * 55)
        for o in out:
            name = (o.team_name or "").strip()[:28]
            self.stdout.write(f"{o.team_id:>8}  {name:<28}  {o.attack:>8.3f}  {o.defense:>8.3f}")
