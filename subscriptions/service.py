# subscriptions/service.py
from datetime import timedelta
from django.db import transaction
from django.utils import timezone
from .models import SubscriptionPlan, UserSubscription


def country_code_for(user) -> str:
    """
    Try common places a code might live. Returns a 2-letter code like 'NG' or ''.
    """
    for attr in ("country_code", "country"):
        v = getattr(user, attr, None)
        if v:
            return str(v).upper()
    profile = getattr(user, "profile", None)
    if profile:
        for attr in ("country_code", "country"):
            v = getattr(profile, attr, None)
            if v:
                return str(v).upper()
    return ""


def region_for(user) -> str:
    """Map a country to 'NG' vs 'INTL' pricing regions."""
    return "NG" if country_code_for(user) == "NG" else "INTL"


def get_entitled_plans(user):
    """
    Return a queryset of the 3 Golden plans the user should see, based on region.
    """
    return (
        SubscriptionPlan.objects.active()
        .for_region(region_for(user))
        .filter(name__iexact="Golden Membership")
        .order_by("duration_days")
    )


def has_active_subscription(user) -> bool:
    now = timezone.now()
    return UserSubscription.objects.filter(
        user=user, is_active=True, canceled_at__isnull=True, ends_at__gte=now
    ).exists()


@transaction.atomic
def start_subscription(user, plan: SubscriptionPlan, extend=False) -> UserSubscription:
    """
    Activate a plan for the user.
    - extend=False: cancel current active and start from now
    - extend=True : start from current end if exists (stacking)
    """
    now = timezone.now()

    if not extend:
        UserSubscription.objects.filter(
            user=user, is_active=True, canceled_at__isnull=True, ends_at__gte=now
        ).update(is_active=False, canceled_at=now)
        starts_at = now
    else:
        current = (
            UserSubscription.objects.filter(
                user=user, is_active=True, canceled_at__isnull=True, ends_at__gte=now
            )
            .order_by("-ends_at")
            .first()
        )
        starts_at = current.ends_at if current else now

    ends_at = plan.ends_at_from(starts_at)
    return UserSubscription.objects.create(
        user=user, plan=plan, starts_at=starts_at, ends_at=ends_at, is_active=True
    )











# subscriptions/service.py
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from .models import SubscriptionPlan, UserSubscription, Payment
from .payments import PaystackProvider, StripeProvider

# ---------- Country / region helpers ----------
def _extract_country_raw(user) -> str:
    # Look in common places (user and profile)
    for attr in ("country_code", "country", "countryAlpha2", "country_alpha2"):
        val = getattr(user, attr, None)
        if val:
            return str(val)
    profile = getattr(user, "profile", None)
    if profile:
        for attr in ("country_code", "country", "countryAlpha2", "country_alpha2"):
            val = getattr(profile, attr, None)
            if val:
                return str(val)
    return ""

def _normalize_iso2(val: str) -> str:
    t = (val or "").strip().upper()
    if not t:
        return ""
    # Already ISO-2
    if len(t) == 2:
        return t
    # Common fallbacks
    if t in ("NGA", "NIGERIA"):
        return "NG"
    return t  # last resort (if someone stores weird stuff)

def country_code_for(user) -> str:
    return _normalize_iso2(_extract_country_raw(user))

def region_for(user) -> str:
    return "NG" if country_code_for(user) == "NG" else "INTL"

# ---------- Plans list (unchanged if you already had this) ----------
def get_entitled_plans(user):
    region = region_for(user)
    return (
        SubscriptionPlan.objects.active()
        .filter(name__iexact="Golden Membership", region=region)
        .order_by("duration_days")
    )

def has_active_subscription(user) -> bool:
    return UserSubscription.objects.filter(
        user=user, is_active=True, ends_at__gte=timezone.now(), canceled_at__isnull=True
    ).exists()

@transaction.atomic
def start_subscription(user, plan: SubscriptionPlan, extend=False) -> UserSubscription:
    from datetime import timedelta
    now = timezone.now()
    if not extend:
        UserSubscription.objects.filter(
            user=user, is_active=True, ends_at__gte=now, canceled_at__isnull=True
        ).update(is_active=False, canceled_at=now)
        starts_at = now
    else:
        current = (
            UserSubscription.objects
            .filter(user=user, is_active=True, ends_at__gte=now, canceled_at__isnull=True)
            .order_by("-ends_at")
            .first()
        )
        starts_at = current.ends_at if current else now
    ends_at = starts_at + timedelta(days=plan.duration_days)
    return UserSubscription.objects.create(
        user=user, plan=plan, starts_at=starts_at, ends_at=ends_at, is_active=True
    )

# ---------- Provider selection ----------
def provider_for_region(region: str) -> str:
    cfg = getattr(settings, "SUBSCRIPTIONS_PAYMENT", {})
    return (cfg.get(region) or {}).get("provider", "stripe")  # default stripe as safe fallback

@transaction.atomic
def start_checkout(user, plan: SubscriptionPlan, request):
    """
    Create Payment and return provider checkout URL.
    IMPORTANT: trust the PLAN's region (already constrained in the view).
    """
    region = plan.region  # <— trust the plan we fetched with region filter
    provider = provider_for_region(region)

    payment = Payment.objects.create(
        user=user,
        plan=plan,
        provider=provider,
        status="PENDING",
        amount=plan.amount,
        currency=plan.currency,
        region=region,
    )

    if provider == "paystack":
        return PaystackProvider.create_checkout(request, payment)
    elif provider == "stripe":
        return StripeProvider.create_checkout(request, payment)
    raise RuntimeError(f"Unsupported provider: {provider}")
