# subscriptions/context_processors.py
from django.utils import timezone
from .models import UserSubscription

def subscription(request):
    """
    Adds `subscription` dict to every template:
      - subscription.has_active_subscription: bool
      - subscription.current: UserSubscription or None
    """
    data = {"has_active_subscription": False, "current": None}
    u = getattr(request, "user", None)

    if u and u.is_authenticated:
        sub = (
            UserSubscription.objects
            .select_related("plan")
            .filter(
                user=u,
                is_active=True,
                canceled_at__isnull=True,
                ends_at__gte=timezone.now(),
            )
            .order_by("-ends_at")
            .first()
        )
        if sub:
            data["has_active_subscription"] = True
            data["current"] = sub

    return {"subscription": data}
