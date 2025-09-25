# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Iterable, Optional

from django.contrib import admin, messages
from django.db.models import Count, Q, QuerySet
from django.utils import timezone
from django.utils.html import format_html

from .models import SubscriptionPlan, UserSubscription, Payment
from .forms import (
    SubscriptionPlanAdminForm,
    UserSubscriptionAdminForm,
    PaymentAdminForm,
)

# -------------------------
# Helpers
# -------------------------
def _now():
    return timezone.now()

def _current_subscriptions_q():
    now = _now()
    return Q(is_active=True, canceled_at__isnull=True, ends_at__gte=now)

# -------------------------
# Filters
# -------------------------
class CurrentSubscriptionFilter(admin.SimpleListFilter):
    title = "current?"
    parameter_name = "current"

    def lookups(self, request, model_admin):
        return (("yes", "Yes (active & not expired)"),
                ("no", "No (expired/canceled)"))

    def queryset(self, request, qs: QuerySet[UserSubscription]):
        value = self.value()
        if value == "yes":
            return qs.filter(_current_subscriptions_q())
        if value == "no":
            return qs.exclude(_current_subscriptions_q())
        return qs


# -------------------------
# Actions (SubscriptionPlan)
# -------------------------
@admin.action(description="Activate selected plans")
def activate_plans(modeladmin, request, qs: QuerySet[SubscriptionPlan]):
    updated = qs.update(is_active=True)
    messages.success(request, f"Activated {updated} plan(s).")

@admin.action(description="Deactivate selected plans")
def deactivate_plans(modeladmin, request, qs: QuerySet[SubscriptionPlan]):
    updated = qs.update(is_active=False)
    messages.warning(request, f"Deactivated {updated} plan(s).")

# -------------------------
# Actions (UserSubscription)
# -------------------------
def _extend_days(qs: QuerySet[UserSubscription], days: int) -> int:
    cnt = 0
    for sub in qs:
        sub.ends_at = sub.ends_at + timedelta(days=days)
        sub.is_active = True
        sub.canceled_at = None
        sub.save(update_fields=["ends_at", "is_active", "canceled_at"])
        cnt += 1
    return cnt

@admin.action(description="Extend by 7 days")
def extend_7(modeladmin, request, qs):
    n = _extend_days(qs, 7)
    messages.success(request, f"Extended {n} subscription(s) by 7 days.")

@admin.action(description="Extend by 30 days")
def extend_30(modeladmin, request, qs):
    n = _extend_days(qs, 30)
    messages.success(request, f"Extended {n} subscription(s) by 30 days.")

@admin.action(description="Cancel (set canceled_at=now, inactive)")
def cancel_subscriptions(modeladmin, request, qs: QuerySet[UserSubscription]):
    now = _now()
    updated = qs.update(is_active=False, canceled_at=now)
    messages.warning(request, f"Canceled {updated} subscription(s).")

@admin.action(description="Activate (reactivate without changing ends_at)")
def reactivate_subscriptions(modeladmin, request, qs: QuerySet[UserSubscription]):
    updated = qs.update(is_active=True, canceled_at=None)
    messages.success(request, f"Reactivated {updated} subscription(s).")

# -------------------------
# Actions (Payment)
# -------------------------
@admin.action(description="Mark as SUCCESS")
def mark_success(modeladmin, request, qs: QuerySet[Payment]):
    updated = qs.update(status="SUCCESS")
    messages.success(request, f"Marked {updated} payment(s) as SUCCESS.")

@admin.action(description="Mark as FAILED")
def mark_failed(modeladmin, request, qs: QuerySet[Payment]):
    updated = qs.update(status="FAILED")
    messages.warning(request, f"Marked {updated} payment(s) as FAILED.")

@admin.action(description="Mark as CANCELED")
def mark_canceled(modeladmin, request, qs: QuerySet[Payment]):
    updated = qs.update(status="CANCELED")
    messages.info(request, f"Marked {updated} payment(s) as CANCELED.")

@admin.action(description="Provision subscriptions for SUCCESS payments")
def provision_from_payments(modeladmin, request, qs: QuerySet[Payment]):
    created = 0
    for p in qs.select_related("plan", "user"):
        if p.status != "SUCCESS":
            continue
        # If user already has a current sub for the same plan, extend instead of creating overlap
        now = _now()
        existing: Optional[UserSubscription] = (
            UserSubscription.objects
            .filter(user=p.user, plan=p.plan)
            .order_by("-ends_at")
            .first()
        )
        if existing and existing.ends_at and existing.ends_at >= now and existing.is_active:
            # Extend by plan.duration_days
            existing.ends_at = p.plan.ends_at_from(existing.ends_at)
            existing.save(update_fields=["ends_at"])
            created += 1  # count as handled
            continue

        UserSubscription.objects.create(
            user=p.user,
            plan=p.plan,
            starts_at=now,
            ends_at=p.plan.ends_at_from(now),
            is_active=True,
            canceled_at=None,
        )
        created += 1
    if created:
        messages.success(request, f"Provisioned/extended {created} subscription(s).")
    else:
        messages.info(request, "No subscriptions provisioned.")

# -------------------------
# Admin registrations
# -------------------------
@admin.register(SubscriptionPlan)
class SubscriptionPlanAdmin(admin.ModelAdmin):
    form = SubscriptionPlanAdminForm
    list_display = (
        "name", "code", "region", "currency", "amount",
        "duration_days", "is_active", "price_label", "active_users_count",
        "created_at",
    )
    list_filter = ("region", "currency", "is_active",)
    search_fields = ("code", "name",)
    ordering = ("region", "duration_days", "name")
    actions = (activate_plans, deactivate_plans)
    readonly_fields = ("created_at",)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        # approximate "active users" count: subs that are current for this plan
        now = _now()
        return qs.annotate(
            _active_users=Count(
                "user_subscriptions",
                filter=Q(
                    user_subscriptions__is_active=True,
                    user_subscriptions__canceled_at__isnull=True,
                    user_subscriptions__ends_at__gte=now,
                )
            )
        )

    @admin.display(description="Active users", ordering="_active_users")
    def active_users_count(self, obj: SubscriptionPlan):
        return obj._active_users

    @admin.display(description="Price label")
    def price_label(self, obj: SubscriptionPlan):
        return obj.price_label


@admin.register(UserSubscription)
class UserSubscriptionAdmin(admin.ModelAdmin):
    form = UserSubscriptionAdminForm
    list_display = (
        "user", "plan",
        "starts_at", "ends_at",
        "is_active", "is_current_badge",
        "days_left",
        "created_at",
    )
    list_filter = (CurrentSubscriptionFilter, "is_active", "plan__region", "plan__currency")
    search_fields = (
        "user__username", "user__email",
        "plan__name", "plan__code",
    )
    autocomplete_fields = ("user", "plan")
    ordering = ("-starts_at",)
    readonly_fields = ("created_at",)
    actions = (extend_7, extend_30, cancel_subscriptions, reactivate_subscriptions)

    @admin.display(description="Current?", boolean=False)
    def is_current_badge(self, obj: UserSubscription):
        ok = obj.is_current
        color = "#10B981" if ok else "#ef4444"
        text = "YES" if ok else "NO"
        return format_html('<b style="color:{}">{}</b>', color, text)

    @admin.display(description="Days left")
    def days_left(self, obj: UserSubscription):
        now = _now()
        if obj.ends_at < now:
            return 0
        return (obj.ends_at - now).days


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    form = PaymentAdminForm
    list_display = (
        "id", "user", "plan", "provider", "status",
        "amount", "currency", "region",
        "provider_ref", "created_at", "updated_at",
    )
    list_filter = ("provider", "status", "currency", "region", "plan__code")
    search_fields = (
        "id", "provider_ref",
        "user__username", "user__email",
        "plan__code", "plan__name",
    )
    autocomplete_fields = ("user", "plan")
    readonly_fields = ("created_at", "updated_at")
    actions = (mark_success, mark_failed, mark_canceled, provision_from_payments)

    def save_model(self, request, obj: Payment, form, change):
        """
        If a payment transitions to SUCCESS, auto-provision or extend a subscription.
        """
        pre_status = None
        if change:
            try:
                pre_status = Payment.objects.get(pk=obj.pk).status
            except Payment.DoesNotExist:
                pre_status = None

        super().save_model(request, obj, form, change)

        if obj.status == "SUCCESS" and pre_status != "SUCCESS":
            # Auto-provision on success
            now = _now()
            existing: Optional[UserSubscription] = (
                UserSubscription.objects
                .filter(user=obj.user, plan=obj.plan)
                .order_by("-ends_at")
                .first()
            )
            if existing and existing.is_active and existing.ends_at >= now:
                existing.ends_at = obj.plan.ends_at_from(existing.ends_at)
                existing.save(update_fields=["ends_at"])
                self.message_user(
                    request,
                    f"Extended existing subscription for {obj.user} by {obj.plan.duration_days} day(s).",
                    level=messages.SUCCESS,
                )
            else:
                UserSubscription.objects.create(
                    user=obj.user,
                    plan=obj.plan,
                    starts_at=now,
                    ends_at=obj.plan.ends_at_from(now),
                    is_active=True,
                    canceled_at=None,
                )
                self.message_user(
                    request,
                    f"Created new subscription for {obj.user}.",
                    level=messages.SUCCESS,
                )
