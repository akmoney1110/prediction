from django.urls import path
from . import views

app_name = "subscriptions"

urlpatterns = [
    path("choose/", views.choose, name="choose"),
    path("start/<slug:code>/", views.start, name="start"),

    path("checkout/<slug:code>/", views.checkout, name="checkout"),
    path("success/", views.success, name="success"),
    path("cancel/", views.cancel, name="cancel"),

    # webhooks
    path("wh/paystack/", views.paystack_webhook, name="paystack_webhook"),
    path("wh/stripe/", views.stripe_webhook, name="stripe_webhook"),
]
