# subscriptions/models.py
from datetime import timedelta
from decimal import Decimal
from django.conf import settings
from django.db import models
from django.utils import timezone


class SubscriptionPlanQuerySet(models.QuerySet):
    def active(self):
        return self.filter(is_active=True)
    def for_region(self, region):  # "NG" or "INTL"
        return self.filter(region=region)


class SubscriptionPlan(models.Model):
    REGION_CHOICES   = (("NG", "Nigeria"), ("INTL", "International"))
    CURRENCY_CHOICES = (("NGN", "Naira"), ("USD", "US Dollar"))

    code          = models.SlugField(max_length=64, unique=True)
    name          = models.CharField(max_length=100, default="Golden Membership")
    duration_days = models.PositiveIntegerField()
    region        = models.CharField(max_length=10, choices=REGION_CHOICES)
    currency      = models.CharField(max_length=10, choices=CURRENCY_CHOICES)
    amount        = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    is_active     = models.BooleanField(default=True)

    created_at    = models.DateTimeField(auto_now_add=True)

    objects = SubscriptionPlanQuerySet.as_manager()

    class Meta:
        ordering = ("region", "duration_days")

    def __str__(self):
        return f"{self.name} {self.duration_days}d [{self.region}] {self.currency} {self.amount}"

    @property
    def price_label(self) -> str:
        # Pretty label for templates
        if self.currency == "NGN":
            # e.g. ₦9,000
            naira = f"₦{int(self.amount):,}"
            return f"{self.duration_days} days • {naira}"
        # USD
        usd = f"${self.amount.normalize():f}".rstrip("0").rstrip(".")
        return f"{self.duration_days} days • {usd}"

    def ends_at_from(self, dt) -> timezone.datetime:
        return dt + timedelta(days=self.duration_days)


class UserSubscription(models.Model):
    user       = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="subscriptions")
    plan       = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT, related_name="user_subscriptions")
    starts_at  = models.DateTimeField(default=timezone.now)
    ends_at    = models.DateTimeField()
    is_active  = models.BooleanField(default=True)
    canceled_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-starts_at",)

    def __str__(self):
        return f"{self.user} -> {self.plan} (active={self.is_active})"

    @property
    def is_current(self) -> bool:
        return self.is_active and self.canceled_at is None and self.ends_at >= timezone.now()

    def save(self, *args, **kwargs):
        if not self.ends_at:
            base = self.starts_at or timezone.now()
            self.ends_at = self.plan.ends_at_from(base)
        super().save(*args, **kwargs)






# subscriptions/models.py
from django.db import models
from django.conf import settings
from django.utils import timezone
from decimal import Decimal
import uuid

# ... keep your SubscriptionPlan and UserSubscription above ...

class Payment(models.Model):
    PROVIDERS = (("paystack","Paystack"), ("stripe","Stripe"))
    STATUSES  = (("PENDING","Pending"), ("SUCCESS","Success"),
                 ("FAILED","Failed"), ("CANCELED","Canceled"))

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="payments")
    plan = models.ForeignKey("subscriptions.SubscriptionPlan", on_delete=models.PROTECT, related_name="payments")

    provider = models.CharField(max_length=16, choices=PROVIDERS)
    status   = models.CharField(max_length=16, choices=STATUSES, default="PENDING")

    amount   = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    currency = models.CharField(max_length=10)  # "NGN" or "USD"
    region   = models.CharField(max_length=10)  # "NG" or "INTL"

    # Reference/ids from provider
    provider_ref = models.CharField(max_length=128, blank=True, default="")  # paystack reference or stripe session/payment id
    meta = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.user} {self.plan} {self.provider} {self.status} {self.amount} {self.currency}"
