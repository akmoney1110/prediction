# subscriptions/payments.py
import json
import hmac, hashlib
import requests
from decimal import Decimal
from django.conf import settings
from django.urls import reverse
from django.utils.http import urlencode

# Stripe is optional in dev; import guarded
try:
    import stripe
except Exception:  # pragma: no cover
    stripe = None


class PaystackProvider:
    """
    Initialize a transaction and redirect the user to Paystack's authorization_url.
    Verify using webhook (X-Paystack-Signature).
    """
    API_INIT = "https://api.paystack.co/transaction/initialize"

    @staticmethod
    def create_checkout(request, payment):
        assert payment.currency == "NGN", "Paystack expects NGN here"
        amount_kobo = int(Decimal(payment.amount) * 100)

        callback_query = urlencode({"pid": str(payment.id)})
        success_url = request.build_absolute_uri(reverse("subscriptions:success")) + f"?{callback_query}"
        cancel_url  = request.build_absolute_uri(reverse("subscriptions:cancel"))  + f"?{callback_query}"

        payload = {
            "email": request.user.email or "no-email@example.com",
            "amount": amount_kobo,
            "reference": str(payment.id),  # we drive with our UUID
            "callback_url": success_url,   # optional; we still rely on webhook for truth
            "metadata": {
                "payment_id": str(payment.id),
                "user_id": request.user.pk,
                "plan_code": payment.plan.code,
                "cancel_url": cancel_url,
            },
        }
        headers = {"Authorization": f"Bearer {settings.PAYSTACK_SECRET_KEY}", "Content-Type": "application/json"}
        r = requests.post(PaystackProvider.API_INIT, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data.get("status"):
            raise RuntimeError(f"Paystack init failed: {data}")

        auth_url = data["data"]["authorization_url"]
        payment.provider_ref = data["data"]["reference"]
        payment.meta = data["data"]
        payment.save(update_fields=["provider_ref", "meta"])
        return auth_url

    @staticmethod
    def verify_webhook(request):
        # Paystack sends a SHA512 HMAC signature of the raw request body
        secret = settings.PAYSTACK_SECRET_KEY.encode()
        signature = request.headers.get("X-Paystack-Signature", "")
        computed = hmac.new(secret, request.body, hashlib.sha512).hexdigest()
        return hmac.compare_digest(signature, computed)


class StripeProvider:
    """
    Create Checkout Session → redirect. Use webhook for finalization.
    """
    @staticmethod
    def create_checkout(request, payment):
        if stripe is None:
            raise RuntimeError("Stripe not installed. `pip install stripe`")

        stripe.api_key = settings.STRIPE_SECRET_KEY

        success = request.build_absolute_uri(reverse("subscriptions:success")) + f"?pid={payment.id}"
        cancel  = request.build_absolute_uri(reverse("subscriptions:cancel"))  + f"?pid={payment.id}"

        amount_cents = int(Decimal(payment.amount) * 100)

        session = stripe.checkout.Session.create(
            mode="payment",
            success_url=success,
            cancel_url=cancel,
            metadata={
                "payment_id": str(payment.id),
                "user_id": request.user.pk,
                "plan_code": payment.plan.code,
            },
            customer_email=request.user.email or None,
            line_items=[{
                "price_data": {
                    "currency": payment.currency.lower(),  # "usd"
                    "product_data": {"name": f"{payment.plan.name} — {payment.plan.duration_days} days"},
                    "unit_amount": amount_cents,
                },
                "quantity": 1,
            }],
        )
        payment.provider_ref = session.id
        payment.meta = {"checkout_session": session.id}
        payment.save(update_fields=["provider_ref", "meta"])
        return session.url

    @staticmethod
    def parse_webhook(request):
        if stripe is None:
            raise RuntimeError("Stripe not installed")

        sig = request.headers.get("Stripe-Signature", "")
        payload = request.body.decode("utf-8")
        return stripe.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
