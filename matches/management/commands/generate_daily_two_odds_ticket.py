# matches/management/commands/generate_daily_two_odds_ticket.py
from datetime import datetime, timezone
from django.core.management.base import BaseCommand, CommandError
from matches.utils import get_or_create_daily_two_odds_ticket, TWO_ODDS_LEAGUE_ID

class Command(BaseCommand):
    help = "Generate (or reuse) the DAILY ~target odds ticket with high-probability, low fair-odds legs."

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, default=None, help="UTC date YYYY-MM-DD; default: today")
        parser.add_argument("--target-odds", type=float, default=3.00)
        # New: preferred flag
        parser.add_argument("--over-tolerance", dest="over_tolerance", type=float, default=None,
                            help="Allowed overshoot fraction, e.g. 0.15 = +15% cap")
        # Legacy aliases (mapped to over_tolerance)
        parser.add_argument("--tolerance", dest="tolerance", type=float, default=None,
                            help="[DEPRECATED] use --over-tolerance")
        parser.add_argument("--tol", dest="tol", type=float, default=None,
                            help="[DEPRECATED] alias for --over-tolerance")

        parser.add_argument("--min-legs", dest="min_legs", type=int, default=2)
        parser.add_argument("--max-legs", dest="max_legs", type=int, default=6)
        parser.add_argument("--min-p", dest="min_p", type=float, default=0.60)
        parser.add_argument("--max-fair-odds", dest="max_fair_odds", type=float, default=1.40)
        parser.add_argument("--attempts", type=int, default=800)
        parser.add_argument("--force", action="store_true")

    def handle(self, *args, **opts):
        # Parse date (UTC)
        if opts["date"]:
            try:
                y, m, d = map(int, opts["date"].split("-"))
                ticket_date = datetime(y, m, d, tzinfo=timezone.utc).date()
            except Exception:
                raise CommandError("Invalid --date. Use YYYY-MM-DD (UTC).")
        else:
            ticket_date = datetime.now(timezone.utc).date()

        # Map legacy flags -> over_tolerance
        over_tol = (
            opts.get("over_tolerance")
            if opts.get("over_tolerance") is not None
            else (opts.get("tol") if opts.get("tol") is not None else opts.get("tolerance"))
        )
        if over_tol is None:
            over_tol = 0.15  # default overshoot cap

        ticket = get_or_create_daily_two_odds_ticket(
            ticket_date=ticket_date,
            target_odds=float(opts["target_odds"]),
            over_tolerance=float(over_tol),          # <-- pass the correct kwarg
            min_p=float(opts["min_p"]),
            max_fair_odds=float(opts["max_fair_odds"]),
            min_legs=int(opts["min_legs"]),
            max_legs=int(opts["max_legs"]),
            attempts=int(opts["attempts"]),
            force_regenerate=bool(opts["force"]),
        )

        self.stdout.write(self.style.SUCCESS(
            f"Daily ticket (league_id={TWO_ODDS_LEAGUE_ID}) for {ticket_date}: "
            f"{ticket.legs} legs | bookish_odds={ticket.acc_bookish_odds:.2f} "
            f"| fair_odds={ticket.acc_fair_odds:.2f} | prob={ticket.acc_probability:.4f}"
        ))
