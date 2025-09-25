# matches/management/commands/diag_shots.py
from django.core.management.base import BaseCommand
from django.db import connection
from django.utils import timezone
from django.test import RequestFactory
from django.apps import apps
import json

class Command(BaseCommand):
    help = "Diagnose h_shot / a_shot presence end-to-end (models, DB, data, views)."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, default=None, help="Filter by league_id (optional)")
        parser.add_argument("--days", type=int, default=7, help="Window for time-based checks (default: 7)")

    def handle(self, *args, **opts):
        league_id = opts["league_id"]
        days = int(opts["days"])

        Match = apps.get_model("matches", "Match")
        PredictedMarket = apps.get_model("matches", "PredictedMarket")
        MatchPrediction = apps.get_model("matches", "MatchPrediction")

        self.stdout.write(self.style.NOTICE("=== 1) MODEL FIELD CHECK ==="))
        pm_has_h = hasattr(PredictedMarket, "h_shot")
        pm_has_a = hasattr(PredictedMarket, "a_shot")
        mp_has_h = hasattr(MatchPrediction, "h_shot") if MatchPrediction else False
        mp_has_a = hasattr(MatchPrediction, "a_shot") if MatchPrediction else False
        self.stdout.write(f"PredictedMarket.h_shot: {pm_has_h}")
        self.stdout.write(f"PredictedMarket.a_shot: {pm_has_a}")
        self.stdout.write(f"MatchPrediction.h_shot: {mp_has_h}")
        self.stdout.write(f"MatchPrediction.a_shot: {mp_has_a}")

        self.stdout.write(self.style.NOTICE("\n=== 2) DATABASE COLUMN CHECK (introspection) ==="))
        with connection.cursor() as cur:
            def table_cols(table):
                try:
                    desc = connection.introspection.get_table_description(cur, table)
                    return {c.name for c in desc}
                except Exception:
                    return set()

            # Guess table names (default Django naming)
            pm_table = PredictedMarket._meta.db_table
            mp_table = MatchPrediction._meta.db_table

            pm_cols = table_cols(pm_table)
            mp_cols = table_cols(mp_table)

        self.stdout.write(f"{pm_table} has columns: h_shot={('h_shot' in pm_cols)}, a_shot={('a_shot' in pm_cols)}")
        self.stdout.write(f"{mp_table} has columns: h_shot={('h_shot' in mp_cols)}, a_shot={('a_shot' in mp_cols)}")

        self.stdout.write(self.style.NOTICE("\n=== 3) DATA POPULATION CHECK (recent rows) ==="))
        now = timezone.now()
        start = now - timezone.timedelta(days=days)

        pm_qs = PredictedMarket.objects.filter(kickoff_utc__gte=start, kickoff_utc__lte=now)
        if league_id is not None:
            pm_qs = pm_qs.filter(league_id=league_id)

        pm_total = pm_qs.count()
        pm_h_null = pm_qs.filter(h_shot__isnull=True).count() if pm_has_h else None
        pm_a_null = pm_qs.filter(a_shot__isnull=True).count() if pm_has_a else None

        self.stdout.write(f"PredictedMarket rows (last {days}d): {pm_total}")
        if pm_has_h:
            self.stdout.write(f"  h_shot null: {pm_h_null}  | not-null: {pm_total - pm_h_null}")
        else:
            self.stdout.write("  h_shot: field missing on model or DB.")
        if pm_has_a:
            self.stdout.write(f"  a_shot null: {pm_a_null}  | not-null: {pm_total - pm_a_null}")
        else:
            self.stdout.write("  a_shot: field missing on model or DB.")

        # Sample a few rows to see actual values
        if pm_total and pm_has_h and pm_has_a:
            sample = pm_qs.order_by("-kickoff_utc")[:3]
            self.stdout.write("  Sample latest PredictedMarket rows (id, h_shot, a_shot):")
            for r in sample:
                self.stdout.write(f"    #{r.id}: h={getattr(r, 'h_shot', None)} a={getattr(r, 'a_shot', None)}")

        self.stdout.write(self.style.NOTICE("\n=== 4) VIEW PAYLOAD CHECK (are they in JSON?) ==="))
        # We’ll call your JSON endpoints via RequestFactory with a small window (days=1)
        try:
            from matches.views import upcoming_predictions_json, completed_today_json
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Could not import views: {e}"))
            return

        rf = RequestFactory()
        lid = league_id or 0
        req_up = rf.get(f"/_upcoming/{lid}/1/")
        req_co = rf.get(f"/_completed/{lid}/")

        up = json.loads(upcoming_predictions_json(req_up, lid, 1).content.decode("utf-8"))
        co = json.loads(completed_today_json(req_co, lid).content.decode("utf-8"))

        def probe(payload, name):
            matches = payload.get("matches") or []
            self.stdout.write(f"{name}: {len(matches)} matches")
            if matches:
                first = matches[0]
                keys = list(first.keys())
                has_h = ("h_shot" in first)
                has_a = ("a_shot" in first)
                self.stdout.write(f"  First match keys include h_shot={has_h}, a_shot={has_a}")
                if has_h or has_a:
                    self.stdout.write(f"  Sample values: h={first.get('h_shot')} a={first.get('a_shot')}")
                else:
                    self.stdout.write("  NOTE: Views are not including h_shot/a_shot. Add them to the JSON payload in the view.")

        probe(up, "upcoming_predictions_json")
        probe(co, "completed_today_json")

        self.stdout.write(self.style.NOTICE("\n=== 5) QUICK FIX HINTS ==="))
        if not pm_has_h or "h_shot" not in pm_cols:
            self.stdout.write("- Add h_shot to PredictedMarket model and run migrations.")
        if not pm_has_a or "a_shot" not in pm_cols:
            self.stdout.write("- Add a_shot to PredictedMarket model and run migrations.")
        self.stdout.write("- Ensure your ingestion code sets h_shot/a_shot when creating/updating PredictedMarket rows.")
        self.stdout.write("- In your JSON views, explicitly include these fields in the dict you append per match.")
        self.stdout.write(self.style.SUCCESS("\nDone."))
