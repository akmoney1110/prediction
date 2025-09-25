from functools import wraps
from django.conf import settings
from django.shortcuts import redirect
from django.contrib import messages
from .service import has_active_subscription

def subscription_required(view_func):
    """
    Gate premium content. If not subscribed, push to the chooser page.
    """
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            login_url = settings.LOGIN_URL or "/accounts/login/"
            return redirect(f"{login_url}?next={request.get_full_path()}")
        if has_active_subscription(request.user):
            return view_func(request, *args, **kwargs)
        messages.info(request, "You need an active Golden Membership.")
        return redirect("subscriptions:choose")
    return _wrapped
