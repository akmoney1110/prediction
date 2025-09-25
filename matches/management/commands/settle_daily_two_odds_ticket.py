# matches/management/commands/settle_daily_two_odds_ticket.py
from datetime import datetime, timezone
from django.core.management.base import BaseCommand, CommandError

from matches.utils import settle_two_odds_ticket_for_date, TWO_ODDS_LEAGUE_ID

class Command(BaseCommand):
    help = "Settle the 2-odds daily ticket for a date, or a date range."

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, help="UTC date YYYY-MM-DD")
        parser.add_argument("--start", type=str, help="UTC start date YYYY-MM-DD")
        parser.add_argument("--end", type=str, help="UTC end date YYYY-MM-DD (inclusive)")
        parser.add_argument("--league-id", type=int, default=TWO_ODDS_LEAGUE_ID)
        parser.add_argument("--keep-unknown", action="store_true",
                            help="If set, keep UNKNOWN (don't convert to VOID)")

    def handle(self, *args, **opts):
        keep_unknown = bool(opts["keep_unknown"])
        league_id = int(opts["league_id"])

        def _to_date(s: str):
            y, m, d = map(int, s.split("-"))
            from datetime import date as _d
            return _d(y, m, d)

        # Single date (default to today UTC)
        if opts.get("date") and not (opts.get("start") or opts.get("end")):
            d = _to_date(opts["date"])
            dt = settle_two_odds_ticket_for_date(
                d, league_id=league_id, treat_unknown_as_void=not keep_unknown
            )
            if not dt:
                self.stdout.write(self.style.WARNING(f"No ticket for {d} (league_id={league_id})."))
                return
            self.stdout.write(self.style.SUCCESS(
                f"Settled {d}: status={dt.status} legs={dt.legs}"
            ))
            return

        # Range
        if opts.get("start") and opts.get("end"):
            start = _to_date(opts["start"])
            end = _to_date(opts["end"])
            if end < start:
                raise CommandError("end must be >= start")
            from datetime import timedelta
            d = start
            total = 0
            while d <= end:
                dt = settle_two_odds_ticket_for_date(
                    d, league_id=league_id, treat_unknown_as_void=not keep_unknown
                )
                if dt:
                    self.stdout.write(f"{d}: {dt.status} ({dt.legs} legs)")
                    total += 1
                d += timedelta(days=1)
            self.stdout.write(self.style.SUCCESS(f"Settled {total} ticket(s)."))
            return

        # No args ⇒ today UTC
        d = datetime.now(timezone.utc).date()
        dt = settle_two_odds_ticket_for_date(
            d, league_id=league_id, treat_unknown_as_void=not keep_unknown
        )
        if not dt:
            self.stdout.write(self.style.WARNING(f"No ticket for {d} (league_id={league_id})."))
            return
        self.stdout.write(self.style.SUCCESS(
            f"Settled {d}: status={dt.status} legs={dt.legs}"
        ))
