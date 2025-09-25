# matches/management/commands/build_all_ml_rows.py
"""
Orchestrate ML row building across all (league_id, season) pairs that exist in Match.

Defaults:
- Iterates sequentially (safe, easy to debug).
- Respects your slim builder knobs: last-n, last-m, decay, alpha-prior.
- Skips empty seasons automatically.
- Optional: filter leagues / seasons, dry-run, and "only-missing" skip logic.

Usage examples:
  python manage.py build_all_ml_rows
  python manage.py build_all_ml_rows --league-ids 39 61 140 --season-min 2020 --season-max 2025
  python manage.py build_all_ml_rows --only-missing
  python manage.py build_all_ml_rows --last-n 10 --last-m 5 --decay 0.85 --alpha-prior 4
  python manage.py build_all_ml_rows --exclude-league-ids 233
"""
import time
from typing import Iterable, List, Tuple, Optional

from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.db.models import Count

from matches.models import Match, MLTrainingMatch


def _int_list(arg: Optional[Iterable[str]]) -> Optional[List[int]]:
    if not arg:
        return None
    out = []
    for s in arg:
        try:
            out.append(int(s))
        except Exception:
            continue
    return out or None


class Command(BaseCommand):
    help = "Run build_ml_rows_slim for all (league_id, season) pairs found in Match."

    def add_arguments(self, parser):
        # Filters
        parser.add_argument("--league-ids", nargs="+", help="Only these league IDs")
        parser.add_argument("--exclude-league-ids", nargs="+", help="Skip these league IDs")
        parser.add_argument("--season-min", type=int, help="Lower bound for season (inclusive)")
        parser.add_argument("--season-max", type=int, help="Upper bound for season (inclusive)")
        parser.add_argument("--only-missing", action="store_true",
                            help="Skip (league, season) where MLTrainingMatch count already equals Match count")

        # Slim builder knobs (forwarded)
        parser.add_argument("--last-n", type=int, default=10)
        parser.add_argument("--last-m", type=int, default=5)
        parser.add_argument("--decay", type=float, default=0.85)
        parser.add_argument("--alpha-prior", type=float, default=4.0)

        # Safety
        parser.add_argument("--dry-run", action="store_true", help="List planned work but do not execute")

    def handle(self, *args, **opts):
        league_ids = _int_list(opts.get("league_ids"))
        exclude_ids = _int_list(opts.get("exclude_league_ids"))
        season_min = opts.get("season_min")
        season_max = opts.get("season_max")
        only_missing = bool(opts.get("only_missing"))
        dry = bool(opts.get("dry_run"))

        last_n = int(opts["last_n"])
        last_m = int(opts["last_m"])
        decay = float(opts["decay"])
        alpha = float(opts["alpha_prior"])

        # Build the (league, season) grid from existing matches
        qs = Match.objects.values("league_id", "season")
        if league_ids:
            qs = qs.filter(league_id__in=league_ids)
        if exclude_ids:
            qs = qs.exclude(league_id__in=exclude_ids)
        if season_min is not None:
            qs = qs.filter(season__gte=season_min)
        if season_max is not None:
            qs = qs.filter(season__lte=season_max)

        # Annotate with match counts so we can skip empty ones (shouldn’t happen) and support only-missing
        pairs = (
            Match.objects
            .filter(id__in=Match.objects.values("id"))  # no-op, keeps ORM happy for chaining on same DB
            .values("league_id", "season")
            .annotate(n_matches=Count("id"))
        )
        if league_ids:
            pairs = pairs.filter(league_id__in=league_ids)
        if exclude_ids:
            pairs = pairs.exclude(league_id__in=exclude_ids)
        if season_min is not None:
            pairs = pairs.filter(season__gte=season_min)
        if season_max is not None:
            pairs = pairs.filter(season__lte=season_max)

        # Order for reproducibility
        pairs = pairs.order_by("league_id", "season")

        todo: List[Tuple[int, int, int]] = []
        for row in pairs:
            lid = int(row["league_id"])
            ssn = int(row["season"])
            n_matches = int(row["n_matches"] or 0)
            if n_matches <= 0:
                continue
            if only_missing:
                # Skip if target table already has same number of rows for this league/season
                have = MLTrainingMatch.objects.filter(league_id=lid, season=ssn).count()
                if have >= n_matches:
                    continue
            todo.append((lid, ssn, n_matches))

        if not todo:
            self.stdout.write(self.style.WARNING("Nothing to do (filters too strict or everything already built)."))
            return

        self.stdout.write(self.style.MIGRATE_HEADING("Planned builds"))
        for lid, ssn, n in todo:
            self.stdout.write(f"  • league={lid:<4} season={ssn:<6} matches={n}")

        if dry:
            self.stdout.write(self.style.WARNING("Dry-run enabled — exiting without executing builds."))
            return

        # Execute sequentially (simple & robust)
        start_all = time.time()
        successes = 0
        for i, (lid, ssn, n) in enumerate(todo, 1):
            self.stdout.write(self.style.HTTP_INFO(f"[{i}/{len(todo)}] league={lid} season={ssn} (matches={n})"))
            t0 = time.time()
            try:
                call_command(
                    "build_ml_rows_slim",
                    league_id=lid,
                    season=ssn,
                    last_n=last_n,
                    last_m=last_m,
                    decay=decay,
                    alpha_prior=alpha,
                )
                successes += 1
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"  ✖ Failed league={lid} season={ssn}: {e}"))
            else:
                dt = time.time() - t0
                built = MLTrainingMatch.objects.filter(league_id=lid, season=ssn).count()
                self.stdout.write(self.style.SUCCESS(f"  ✓ Done in {dt:.1f}s → ML rows now: {built}"))

        dt_all = time.time() - start_all
        self.stdout.write(self.style.SUCCESS(f"All done: {successes}/{len(todo)} batches succeeded in {dt_all:.1f}s"))
