# matches/management/commands/diag_mltraining_match.py
from django.core.management.base import BaseCommand
from django.apps import apps
from django.db import connection
from django.utils import timezone
from datetime import timedelta

DEFAULT_FIELDS = ["h_shots10", "a_shots10", "h_shot", "a_shot"]  # checks which actually exist

class Command(BaseCommand):
    help = "Diagnose MLTrainingMatch feature fields (existence, DB columns, null stats, and sample values)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--fields",
            nargs="*",
            default=DEFAULT_FIELDS,
            help="Feature field names to check (default: h_shots10 a_shots10 h_shot a_shot)",
        )
        parser.add_argument("--league-id", type=int, default=None, help="Optional league filter")
        parser.add_argument("--days", type=int, default=365, help="Lookback window for data stats (default 365)")
        parser.add_argument("--sample", type=int, default=5, help="How many sample rows to print (default 5)")

    def handle(self, *args, **opts):
        fields = opts["fields"] or DEFAULT_FIELDS
        days = int(opts["days"])
        sample_n = int(opts["sample"])
        league_id = opts["league_id"]

        MLTM = apps.get_model("matches", "MLTrainingMatch")

        self.stdout.write(self.style.NOTICE("=== 1) MODEL FIELD CHECK ==="))
        present = {}
        for f in fields:
            present[f] = hasattr(MLTM, f)
            self.stdout.write(f"- MLTrainingMatch.{f}: {present[f]}")

        self.stdout.write(self.style.NOTICE("\n=== 2) DATABASE COLUMN CHECK ==="))
        table = MLTM._meta.db_table
        with connection.cursor() as cur:
            cols = {c.name for c in connection.introspection.get_table_description(cur, table)}
        for f in fields:
            self.stdout.write(f"- {table}.{f}: {f in cols}")

        # Warn if any requested feature is missing at model or DB level
        missing_any = [f for f in fields if (not present.get(f) or f not in cols)]
        if missing_any:
            self.stdout.write(self.style.ERROR(
                f"\nMissing fields/columns: {', '.join(missing_any)}"
            ))
            self.stdout.write(
                "If you just added them to models.py, make sure you:\n"
                "  - created a migration (python manage.py makemigrations)\n"
                "  - applied it       (python manage.py migrate)\n"
            )

        self.stdout.write(self.style.NOTICE("\n=== 3) DATA POPULATION (NULL vs NON-NULL) ==="))
        now = timezone.now()
        start = now - timedelta(days=days)
        qs = MLTM.objects.filter(kickoff_utc__gte=start, kickoff_utc__lte=now)
        if league_id is not None:
            qs = qs.filter(league_id=league_id)
        total = qs.count()
        self.stdout.write(f"Rows in last {days} day(s){' (league '+str(league_id)+')' if league_id is not None else ''}: {total}")

        if total == 0:
            self.stdout.write(self.style.WARNING("No rows in the time window; broaden --days or remove filters."))
        else:
            for f in fields:
                if f in cols:
                    nulls = qs.filter(**{f"{f}__isnull": True}).count()
                    nn = total - nulls
                    self.stdout.write(f"- {f}: null={nulls} | not-null={nn}")

            # sample values
            self.stdout.write("\nSample rows (fixture_id, " + ", ".join(fields) + "):")
            for row in qs.order_by("-kickoff_utc").values("fixture_id", *[f for f in fields if f in cols])[:sample_n]:
                row_str = f"  #{row['fixture_id']} " + " ".join([f"{f}={row.get(f)}" for f in fields if f in cols])
                self.stdout.write(row_str)

        self.stdout.write(self.style.NOTICE("\n=== 4) FEATURE EXTRACTION GUARD (copy/paste into your pipeline) ==="))
        self.stdout.write(
            "In the code that builds X for model.predict, ensure you select these columns and in the same order\n"
            "the model was trained on. Add a guard like this around your predict code:\n\n"
            "    REQUIRED = ['h_shots10','a_shots10']  # adjust to what you actually trained on\n"
            "    missing = [c for c in REQUIRED if c not in X.columns]\n"
            "    if missing:\n"
            "        raise ValueError(f\"Missing required features in X: {missing}\")\n\n"
            "    # Optional: if you saved the training feature order, enforce it:\n"
            "    # X = X[TRAIN_FEATURE_ORDER]\n"
            "    # And ensure ColumnTransformer got a DataFrame with matching columns.\n"
        )

        self.stdout.write(self.style.SUCCESS("\nDone."))
