# prediction/matches/management/commands/build_all_team_ratings.py

from __future__ import annotations
import logging
from datetime import timezone, datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command
from django.db.models import Count, Q

from matches.models import Match, TeamRating

log = logging.getLogger(__name__)

MIN_MATCHES_DEFAULT = 30          # per (league, season), must have at least this many finished matches
ALPHA_DEFAULT       = 1.0
MAX_ITER_DEFAULT    = 2000
TOL_DEFAULT         = 1e-8

# We’ll call the fitter from your single-season command if available
try:
    # from your build_team_ratings.py
    from matches.management.commands.build_team_ratings import _fit_one_season  # type: ignore
except Exception:  # pragma: no cover
    _fit_one_season = None


class Command(BaseCommand):
    help = "Build team attack/defense ratings for many (league, season) pairs at once."

    def add_arguments(self, p):
        # What to target
        scope = p.add_mutually_exclusive_group(required=False)
        scope.add_argument("--season", type=int, help="Fit only this season.")
        scope.add_argument("--season-min", type=int, help="Lower bound season (inclusive). Use with --season-max.")
        p.add_argument("--season-max", type=int, help="Upper bound season (inclusive). Use with --season-min.")
        p.add_argument("--league-ids", type=int, nargs="*", help="Optional league filter (one or more).")

        # Behavior
        p.add_argument("--only-missing", action="store_true",
                       help="Skip (league,season) where ratings already exist for ALL teams present in that season.")
        p.add_argument("--min-matches", type=int, default=MIN_MATCHES_DEFAULT,
                       help=f"Require at least this many finished matches to fit (default {MIN_MATCHES_DEFAULT}).")
        p.add_argument("--dry-run", action="store_true", help="Just print the planned fits and exit.")

        # Fitter hyper-params (forwarded to the single-season fitter)
        p.add_argument("--alpha", type=float, default=ALPHA_DEFAULT, help="PoissonRegressor alpha (L2 strength).")
        p.add_argument("--max-iter", type=int, default=MAX_ITER_DEFAULT, help="Max iterations.")
        p.add_argument("--tol", type=float, default=TOL_DEFAULT, help="Optimization tolerance.")

    def handle(self, *args, **opt):
        season      = opt.get("season")
        smin        = opt.get("season_min")
        smax        = opt.get("season_max")
        league_ids  = opt.get("league_ids") or []
        only_missing= bool(opt.get("only_missing"))
        min_matches = int(opt.get("min_matches"))
        dry_run     = bool(opt.get("dry_run"))
        alpha       = float(opt.get("alpha"))
        max_iter    = int(opt.get("max_iter"))
        tol         = float(opt.get("tol"))

        if (smin is None) ^ (smax is None):
            raise CommandError("Use --season-min together with --season-max (or use --season for a single year).")

        # discover candidate (league, season) pairs from Match table where enough finished & labeled
        matches_qs = Match.objects.all()
        if league_ids:
            matches_qs = matches_qs.filter(league_id__in=league_ids)

        # finished + labeled (adjust statuses if your data uses different codes)
        finished_q = Q(status__in=["FT", "AET", "PEN"])
        labeled_q  = ~Q(goals_home__isnull=True) & ~Q(goals_away__isnull=True)

        if season is not None:
            matches_qs = matches_qs.filter(season=season)
        else:
            # choose the observed seasons in the data, then clamp to [smin, smax] if provided
            if smin is not None:
                matches_qs = matches_qs.filter(season__gte=int(smin))
            if smax is not None:
                matches_qs = matches_qs.filter(season__lte=int(smax))

        # group by (league, season) with a count of finished labeled matches
        pairs = list(
            matches_qs.filter(finished_q & labeled_q)
                      .values("league_id", "season")
                      .annotate(n=Count("id"))
                      .order_by("league_id", "season")
        )

        # shape helper: distinct teams per (league, season)
        def teams_for_pair(L: int, S: int) -> Set[int]:
            ids = set(
                Match.objects
                .filter(league_id=L, season=S)
                .values_list("home_id", flat=True)
            )
            ids.update(
                Match.objects
                .filter(league_id=L, season=S)
                .values_list("away_id", flat=True)
            )
            return ids

        # filter pairs by min matches and (optionally) only-missing
        plan: List[Tuple[int, int, int, int]] = []  # (league_id, season, n_matches, n_teams)
        for p in pairs:
            L, S, N = int(p["league_id"]), int(p["season"]), int(p["n"])
            if N < min_matches:
                continue
            team_ids = teams_for_pair(L, S)
            n_teams  = len(team_ids)
            if n_teams == 0:
                continue

            if only_missing:
                existing = TeamRating.objects.filter(league_id=L, season=S).values_list("team_id", flat=True)
                have = set(int(t) for t in existing)
                if team_ids.issubset(have):
                    # we already have ratings for all teams in this (league, season)
                    continue

            plan.append((L, S, N, n_teams))

        # pretty print plan
        if not plan:
            self.stdout.write(self.style.WARNING("No (league, season) pairs qualify for fitting."))
            return

        self.stdout.write("Planned fits")
        for L, S, N, T in plan:
            self.stdout.write(f"  • league={L:<4}  season={S}   matches={N:<4}  teams≈{T}")

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry-run enabled — exiting without fitting."))
            return

        # run the fits
        ok = 0
        for L, S, N, T in plan:
            try:
                if _fit_one_season is not None:
                    # use the in-process fitter (fast, no new process)
                    fs = _fit_one_season(L, S, alpha=alpha, max_iter=max_iter, tol=tol)
                    if fs is None:
                        self.stdout.write(self.style.WARNING(
                            f"Skipped league={L} season={S}: insufficient data after checks."
                        ))
                        continue

                    now = datetime.now(timezone.utc)
                    for tid in fs.team_ids:
                        TeamRating.objects.update_or_create(
                            league_id=L, season=S, team_id=int(tid),
                            defaults={
                                "attack": fs.attack[tid],
                                "defense": fs.defense[tid],
                                "last_updated": now,
                            }
                        )

                    self.stdout.write(self.style.SUCCESS(
                        f"[{ok+1}/{len(plan)}] Saved ratings league={L} season={S} "
                        f"(matches={fs.n_matches}, HFA={fs.hfa:.3f})"
                    ))
                else:
                    # fallback: call the single-season command (slower, but works)
                    call_command(
                        "build_team_ratings",
                        league_id=L,
                        seasons=[S],
                        alpha=alpha,
                        max_iter=max_iter,
                        tol=tol,
                        verbosity=0,
                    )
                    self.stdout.write(self.style.SUCCESS(
                        f"[{ok+1}/{len(plan)}] Saved ratings league={L} season={S} (via subcommand)"
                    ))
                ok += 1
            except Exception as e:
                log.exception("Failed league=%s season=%s: %s", L, S, e)
                self.stderr.write(self.style.ERROR(f"✖ Failed league={L} season={S}: {e}"))

        self.stdout.write(self.style.SUCCESS(f"All done: {ok}/{len(plan)} seasons fitted."))
