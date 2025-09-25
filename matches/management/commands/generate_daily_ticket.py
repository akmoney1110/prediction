# matches/management/commands/generate_daily_ticket.py
from datetime import datetime, timezone
from django.core.management.base import BaseCommand

from matches.utils import get_or_create_global_daily_ticket

class Command(BaseCommand):
    help = "Generate (or reuse) the single GLOBAL daily ticket (across all leagues) for the UTC date."

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, default=None, help="UTC date YYYY-MM-DD; default: today")
        parser.add_argument("--legs", type=int, default=5)
        parser.add_argument("--min-p", type=float, default=0.55)
        parser.add_argument("--force", action="store_true")

    def handle(self, *args, **opts):
        legs = int(opts["legs"])
        min_p = float(opts["min_p"])
        force = bool(opts["force"])

        if opts["date"]:
            from datetime import date as _date
            y, m, d = map(int, opts["date"].split("-"))
            ticket_date = _date(y, m, d)
        else:
            ticket_date = datetime.now(timezone.utc).date()

        dt = get_or_create_global_daily_ticket(
            ticket_date=ticket_date,
            legs=legs,
            min_p=min_p,
            force_regenerate=force,
        )

        self.stdout.write(self.style.SUCCESS(
            f"Global daily ticket for {ticket_date}: {dt.legs} legs | prob={dt.acc_probability:.4f} | fair_odds={dt.acc_fair_odds:.2f} | bookish_odds={dt.acc_bookish_odds:.2f}"
        ))
