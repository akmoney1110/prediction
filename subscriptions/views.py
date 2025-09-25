from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

# Create your views here.
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST
from django.contrib import messages

from .models import SubscriptionPlan
from .service import (
    get_entitled_plans,
   
    start_subscription,
)


def _format_amount(amount, currency):
    if currency == "NGN":
        # show like ₦9,000 / ₦15,000 / ₦30,000
        n = int(amount) if float(amount).is_integer() else amount
        return f"₦{n:,}"
    # USD: $5, $10, $20 (no .00)
    if float(amount).is_integer():
        return f"${int(amount)}"
    return f"${amount:,.2f}"


from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from .service import get_entitled_plans, start_subscription, country_code_for, region_for
from .models import SubscriptionPlan


@login_required
def chose(request):
    plans_qs = get_entitled_plans(request.user)
    context = {
        "country": country_code_for(request.user) or "—",
        "plans": plans_qs,  # objects expose .price_label
    }
    return render(request, "choose.html", context)


@login_required
def start(request, code: str):
    if request.method != "POST":
        raise Http404()

    # Ensure users cannot start a plan from the wrong region by typing codes
    plan = get_object_or_404(
        SubscriptionPlan.objects.active(),
        code=code,
        region=region_for(request.user),
        name__iexact="Golden Membership",
    )

    start_subscription(request.user, plan, extend=False)
    messages.success(request, f"{plan.duration_days}-day Golden Membership activated.")
    next_url = request.GET.get("next") or "/"
    return redirect(next_url)









# subscriptions/views.py
import json
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

from .models import SubscriptionPlan, Payment
from .service import get_entitled_plans, country_code_for, region_for, start_subscription, start_checkout
from .payments import PaystackProvider, StripeProvider


@login_required
def choose(request):
    plans_qs = get_entitled_plans(request.user)
    return render(request, "choose.html", {
        "country": country_code_for(request.user) or "—",
        "plans": plans_qs,
    })


@login_required
def checkout(request, code: str):
    if request.method != "POST":
        raise Http404()
    # Only allow a plan from the user's region
    plan = get_object_or_404(
        SubscriptionPlan.objects.active(),
        code=code,
        region=region_for(request.user),
        name__iexact="Golden Membership",
    )
    redirect_url = start_checkout(request.user, plan, request)
    return redirect(redirect_url)


@login_required
def success(request):
    # This page is just a friendly landing. We rely on webhooks to activate.
    messages.success(request, "Payment received (or pending). Your access will be activated shortly if successful.")
    next_url = request.GET.get("next") or "/"
    return redirect(next_url)


@login_required
def cancel(request):
    messages.info(request, "Payment canceled.")
    return redirect(reverse("subscriptions:choose"))


# ------- Webhooks (no auth; verify signatures!) -------

@csrf_exempt
def paystack_webhook(request):
    if not PaystackProvider.verify_webhook(request):
        return HttpResponseBadRequest("Bad signature")

    event = json.loads(request.body.decode("utf-8"))
    if event.get("event") == "charge.success":
        data = event.get("data", {}) or {}
        ref  = data.get("reference")  # we used payment.id as reference
        try:
            payment = Payment.objects.select_related("plan", "user").get(provider="paystack", provider_ref=ref)
        except Payment.DoesNotExist:
            # fallback: find by id in metadata if needed
            pid = (data.get("metadata") or {}).get("payment_id")
            try:
                payment = Payment.objects.select_related("plan", "user").get(id=pid)
            except Payment.DoesNotExist:
                return HttpResponse("ignore", status=200)

        if payment.status != "SUCCESS":
            payment.status = "SUCCESS"
            payment.save(update_fields=["status"])
            # Activate subscription
            start_subscription(payment.user, payment.plan, extend=False)

    elif event.get("event") in {"charge.failed", "charge.dispute"}:
        # Mark failed if we can map a payment
        data = event.get("data", {}) or {}
        ref  = data.get("reference")
        Payment.objects.filter(provider="paystack", provider_ref=ref).update(status="FAILED")

    return HttpResponse("ok", status=200)


@csrf_exempt
def stripe_webhook(request):
    try:
        evt = StripeProvider.parse_webhook(request)
    except Exception:
        return HttpResponseBadRequest("Bad signature")

    # We care about checkout.session.completed and payment_intent.succeeded
    if evt["type"] in {"checkout.session.completed", "payment_intent.succeeded"}:
        obj = evt["data"]["object"]
        meta = obj.get("metadata") or {}
        pid = meta.get("payment_id")
        if not pid:
            # fallback to session id
            ref = obj.get("id")
            qs = Payment.objects.select_related("plan","user").filter(provider="stripe", provider_ref=ref)
        else:
            qs = Payment.objects.select_related("plan","user").filter(id=pid)
        payment = qs.first()
        if payment and payment.status != "SUCCESS":
            payment.status = "SUCCESS"
            if not payment.provider_ref:
                payment.provider_ref = obj.get("id", "")
            payment.save(update_fields=["status","provider_ref"])
            start_subscription(payment.user, payment.plan, extend=False)

    elif evt["type"] in {"payment_intent.payment_failed", "checkout.session.expired"}:
        obj = evt["data"]["object"]
        ref = obj.get("id")
        Payment.objects.filter(provider="stripe", provider_ref=ref).update(status="FAILED")

    return HttpResponse("ok")
