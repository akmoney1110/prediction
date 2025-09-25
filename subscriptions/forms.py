# -*- coding: utf-8 -*-
from __future__ import annotations

from decimal import Decimal
from typing import Any

from django import forms
from django.core.exceptions import ValidationError
from django.utils import timezone

from .models import SubscriptionPlan, UserSubscription, Payment


class SubscriptionPlanAdminForm(forms.ModelForm):
    class Meta:
        model = SubscriptionPlan
        fields = "__all__"

    def clean_amount(self) -> Decimal:
        amt = self.cleaned_data["amount"]
        if amt < 0:
            raise ValidationError("Amount cannot be negative.")
        if self.cleaned_data.get("currency") == "NGN" and amt != amt.quantize(Decimal("1")):
            # Naira often handled as whole-naira in UI; keep it strict if you prefer
            pass
        return amt


class UserSubscriptionAdminForm(forms.ModelForm):
    class Meta:
        model = UserSubscription
        fields = "__all__"

    def clean(self) -> dict[str, Any]:
        data = super().clean()
        plan: SubscriptionPlan = data.get("plan")
        starts_at = data.get("starts_at")
        ends_at = data.get("ends_at")

        if plan and not ends_at:
            base = starts_at or timezone.now()
            data["ends_at"] = plan.ends_at_from(base)

        # If canceled_at is set, is_active should generally be False
        canceled_at = data.get("canceled_at")
        if canceled_at and data.get("is_active"):
            raise ValidationError("Canceled subscriptions cannot be active.")

        return data


class PaymentAdminForm(forms.ModelForm):
    class Meta:
        model = Payment
        fields = "__all__"

    def clean(self):
        data = super().clean()
        plan: SubscriptionPlan = data.get("plan")
        currency = data.get("currency")
        region = data.get("region")
        amount = data.get("amount")

        # Keep provider/payments consistent with the chosen plan
        if plan:
            if currency and currency != plan.currency:
                raise ValidationError(
                    {"currency": f"Currency must match plan currency ({plan.currency})."}
                )
            if region and region != plan.region:
                raise ValidationError(
                    {"region": f"Region must match plan region ({plan.region})."}
                )
            if amount is not None and amount <= 0:
                raise ValidationError({"amount": "Amount must be > 0."})
            # Optional strictness: ensure admin doesn’t accidentally log the wrong figure
            # if amount and amount != plan.amount:
            #     raise ValidationError({"amount": f"Amount should equal plan amount ({plan.amount})."})

        return data
