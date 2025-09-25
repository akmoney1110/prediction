# matches/management/commands/settle_daily_ticket.py
from datetime import datetime, date as _date, timezone
from django.core.management.base import BaseCommand

from matches.utils import settle_ticket_for_date
from matches.models import DailyTicket, Match
from matches.utils import evaluate_selection_outcome, _pull_final_totals

def explain_ticket(date, league_id=0):
    dt = (DailyTicket.objects
          .filter(ticket_date=date, league_id=league_id)
          .order_by('-id')
          .first())
    if not dt:
        print("No ticket found.")
        return

    print(f"Ticket {date} [{dt.status}]")
    for i, sel in enumerate(dt.selections or [], 1):
        mid    = sel.get("match_id")
        market = sel.get("market") or sel.get("market_code")
        spec   = sel.get("specifier")
        res    = sel.get("result")

        m = Match.objects.filter(id=mid).select_related("home","away").first()
        if not m:
            print(f"{i:02d}) #{mid} — match missing → VOID")
            continue

        out = evaluate_selection_outcome(m, market, spec)
        S = _pull_final_totals(m)
        print(
            f"{i:02d}) #{mid} {m.home.name}–{m.away.name} [{m.status}] "
            f"G:{S['gh']}-{S['ga']} C:{S['ch']}-{S['ca']} "
            f"market={market} spec={spec} → stored={res} eval={out}"
        )

class Command(BaseCommand):
    help = "Evaluate today's (or given date) global daily ticket legs and update results + ticket status."

    def add_arguments(self, parser):
        parser.add_argument("--date", type=str, default=None, help="UTC date YYYY-MM-DD; default: today")
        parser.add_argument("--league-id", type=int, default=0, help="DailyTicket.league_id (default 0 = global)")

    def handle(self, *args, **opts):
        league_id = int(opts.get("league_id", 0))
        
        if opts["date"]:
            y, m, d = map(int, opts["date"].split("-"))
            ticket_date = _date(y, m, d)
        else:
            ticket_date = datetime.now(timezone.utc).date()

        dt = settle_ticket_for_date(ticket_date)
        if not dt:
            self.stdout.write(self.style.WARNING(f"No DailyTicket found for {ticket_date}."))
            return

        wins = sum(1 for s in (dt.selections or []) if s.get("result") == "WIN")
        loses = sum(1 for s in (dt.selections or []) if s.get("result") == "LOSE")
        voids = sum(1 for s in (dt.selections or []) if s.get("result") == "VOID")
        pend  = sum(1 for s in (dt.selections or []) if s.get("result") not in {"WIN","LOSE","VOID"})

        self.stdout.write(self.style.SUCCESS(
            f"Settled ticket {ticket_date}: status={dt.status} | WIN={wins} LOSE={loses} VOID={voids} PENDING={pend}"
        ))




