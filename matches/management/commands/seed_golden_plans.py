# subscriptions/management/commands/seed_golden_plans.py
from decimal import Decimal
from django.core.management.base import BaseCommand
from subscriptions.models import SubscriptionPlan

PLANS = [
    # Nigeria pricing
    {"code": "gold-7d-ng",  "region": "NG",   "duration_days": 7,  "currency": "NGN", "amount": Decimal("9000")},
    {"code": "gold-14d-ng", "region": "NG",   "duration_days": 14, "currency": "NGN", "amount": Decimal("15000")},
    {"code": "gold-30d-ng", "region": "NG",   "duration_days": 30, "currency": "NGN", "amount": Decimal("30000")},
    # International pricing
    {"code": "gold-7d-intl",  "region": "INTL", "duration_days": 7,  "currency": "USD", "amount": Decimal("5")},
    {"code": "gold-14d-intl", "region": "INTL", "duration_days": 14, "currency": "USD", "amount": Decimal("10")},
    {"code": "gold-30d-intl", "region": "INTL", "duration_days": 30, "currency": "USD", "amount": Decimal("20")},
]

class Command(BaseCommand):
    help = "Seed Golden Membership plans for NG and INTL regions."

    def handle(self, *args, **options):
        created = 0
        for p in PLANS:
            obj, was_created = SubscriptionPlan.objects.update_or_create(
                code=p["code"],
                defaults={
                    "name": "Golden Membership",
                    "region": p["region"],
                    "duration_days": p["duration_days"],
                    "currency": p["currency"],
                    "amount": p["amount"],
                    "is_active": True,
                },
            )
            created += 1 if was_created else 0
            self.stdout.write(self.style.SUCCESS(f"Upserted {obj}"))
        self.stdout.write(self.style.SUCCESS(f"Done. New rows: {created}"))
