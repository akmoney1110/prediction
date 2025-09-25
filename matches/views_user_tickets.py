# matches/views_user_tickets.py  (new file)
from datetime import timedelta
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponseBadRequest
from django.template.response import TemplateResponse
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.views.decorators.http import require_GET, require_POST

from .models import UserDailyTicket
from .utils_preview import build_two_odds_ticket_preview  # from step 2
from .utils import TWO_ODDS_LEAGUE_ID  # keep your existing constant
from .views import _enrich_selection  # reuse helper you already wrote

def _d(s, default=None):
    d = parse_date(s) if s else None
    return d or default or timezone.now().date()

def _f(q, k, dv):  # float
    try:
        return float(q.get(k, dv))
    except Exception:
        return dv

def _i(q, k, dv):  # int
    try:
        return int(q.get(k, dv))
    except Exception:
        return dv

# ---------- Builder page with range controls ----------
@login_required
def two_odds_builder_page(request):
    ctx = {
        "init_start": (_d(request.GET.get("start"))).isoformat(),
        "init_end": (_d(request.GET.get("end"))).isoformat(),
        "init_target_odds": request.GET.get("target_odds", "2.0"),
        "init_over_tol": request.GET.get("over_tolerance", "0.15"),
        "init_min_legs": request.GET.get("min_legs", "2"),
        "init_max_legs": request.GET.get("max_legs", "6"),
        "init_min_p": request.GET.get("min_p", "0.60"),
        "init_max_fair": request.GET.get("max_fair_odds", "1.60"),
        "init_attempts": request.GET.get("attempts", "500"),
    }
    return TemplateResponse(request, "two_odds_builder_range.html", ctx)

# ---------- HTMX fragment: preview tickets for a range (no DB writes) ----------
@login_required
@require_GET
def two_odds_preview_range_fragment(request):
    start = _d(request.GET.get("start"))
    end = _d(request.GET.get("end"), start)
    if end < start:
        return HttpResponseBadRequest("end must be >= start")
    # safety cap (avoid huge loops)
    if (end - start).days > 60:
        return HttpResponseBadRequest("Range too large (max 60 days).")

    params = dict(
        target_odds=_f(request.GET, "target_odds", 2.0),
        over_tolerance=_f(request.GET, "over_tolerance", 0.15),
        min_legs=_i(request.GET, "min_legs", 2),
        max_legs=_i(request.GET, "max_legs", 6),
        min_p=_f(request.GET, "min_p", 0.60),
        max_fair_odds=_f(request.GET, "max_fair_odds", 1.60),
        attempts=_i(request.GET, "attempts", 500),
    )

    items = []
    d = start
    while d <= end:
        try:
            t = build_two_odds_ticket_preview(ticket_date=d, **params)
            legs = [_enrich_selection(s.copy()) for s in (t.selections or [])]
            items.append({
                "date": d.isoformat(),
                "has_ticket": True,
                "legs": legs,
                "legs_count": t.legs,
                "acc_bookish_odds": f"{t.acc_bookish_odds:.2f}",
                "acc_fair_odds": f"{t.acc_fair_odds:.2f}",
                "acc_probability_pct": f"{(t.acc_probability*100.0):.2f}",
            })
        except ValueError as e:
            items.append({
                "date": d.isoformat(),
                "has_ticket": False,
                "note": str(e),
            })
        d += timedelta(days=1)

    ctx = {"items": items, "brand": "Rollover 2.0"}
    return TemplateResponse(request, "_two_odds_range_list.html", ctx)

# ---------- Save range into per-user table ----------

@require_POST
def save_two_odds_ange_api(request):
    start = _d(request.POST.get("start"))
    end = _d(request.POST.get("end"), start)
    if end < start:
        return HttpResponseBadRequest("end must be >= start")
    if (end - start).days > 60:
        return HttpResponseBadRequest("Range too large (max 60 days).")

    params = dict(
        target_odds=_f(request.POST, "target_odds", 2.0),
        over_tolerance=_f(request.POST, "over_tolerance", 0.15),
        min_legs=_i(request.POST, "min_legs", 2),
        max_legs=_i(request.POST, "max_legs", 6),
        min_p=_f(request.POST, "min_p", 0.60),
        max_fair_odds=_f(request.POST, "max_fair_odds", 1.60),
        attempts=_i(request.POST, "attempts", 500),
    )

    saved, skipped = 0, 0
    d = start
    while d <= end:
        try:
            t = build_two_odds_ticket_preview(ticket_date=d, **params)
            obj, _ = UserDailyTicket.objects.update_or_create(
                user=request.user, ticket_date=d, 
                defaults={
                    "selections": t.selections,
                    "legs": t.legs,
                    "acc_probability": t.acc_probability,
                    "acc_fair_odds": t.acc_fair_odds,
                    "acc_bookish_odds": t.acc_bookish_odds,
                    "status": "pending",     # will be settled later
                    "build_params": params,
                    "source": "custom",
                }
            )
            saved += 1
        except ValueError:
            # optional: persist an 'empty' row so user sees it in history
            UserDailyTicket.objects.update_or_create(
                user=request.user, ticket_date=d,
                defaults={
                    "selections": [],
                    "legs": 0,
                    "acc_probability": None,
                    "acc_fair_odds": None,
                    "acc_bookish_odds": None,
                    "status": "empty",
                    "build_params": params,
                    "source": "custom",
                }
            )
            skipped += 1
        d += timedelta(days=1)

    return JsonResponse({"ok": True, "saved": saved, "empty": skipped})











# matches/views_user_tickets.py
from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpResponseBadRequest
from django.utils import timezone
from django.utils.dateparse import parse_date
from datetime import timedelta

from .models import UserDailyTicket, DailyTicket
from .utils_preview import build_two_odds_ticket_preview

def _d(s, default=None):
    d = parse_date(s) if s else None
    return d or default or timezone.now().date()

def _f(q, k, dv):
    try:
        return float(q.get(k, dv))
    except Exception:
        return dv

def _i(q, k, dv):
    try:
        return int(q.get(k, dv))
    except Exception:
        return dv
@login_required
@require_POST
def save_two_odds_range_api(request):
    start = _d(request.POST.get("start"))
    end = _d(request.POST.get("end"), start)
    if end < start:
        return HttpResponseBadRequest("end must be >= start")
    if (end - start).days > 60:
        return HttpResponseBadRequest("Range too large (max 60 days).")

    params = dict(
        target_odds=_f(request.POST, "target_odds", 2.0),
        over_tolerance=_f(request.POST, "over_tolerance", 0.15),
        min_legs=_i(request.POST, "min_legs", 2),
        max_legs=_i(request.POST, "max_legs", 6),
        min_p=_f(request.POST, "min_p", 0.60),
        max_fair_odds=_f(request.POST, "max_fair_odds", 1.60),
        attempts=_i(request.POST, "attempts", 500),
    )

    saved, skipped = 0, 0
    d = start
    while d <= end:
        try:
            # build preview (no DB write)
            t = build_two_odds_ticket_preview(ticket_date=d, **params)

            # link to base global ticket if you have it
            base_dt = DailyTicket.objects.filter(ticket_date=d, league_id=-1).order_by("-id").first()

            # IMPORTANT: no 'kind' here — lookup is only (user, ticket_date)
            obj, _ = UserDailyTicket.objects.update_or_create(
                user=request.user,
                ticket_date=d,
                defaults={
                    "base_ticket": base_dt,
                    "name": f"2-odds ({params['target_odds']:.2f})",
                    "selections": t.selections,
                    "legs": t.legs,
                    "acc_probability": t.acc_probability,
                    "acc_fair_odds": t.acc_fair_odds,
                    "acc_bookish_odds": t.acc_bookish_odds,
                    "status": "pending",  # will be settled later
                    "filters": {**params, "category": "two_odds"},
                },
            )
            saved += 1
        except ValueError:
            # optionally persist an empty row so the day shows up in history
            UserDailyTicket.objects.update_or_create(
                user=request.user,
                ticket_date=d,
                defaults={
                    "base_ticket": None,
                    "name": f"2-odds ({params['target_odds']:.2f})",
                    "selections": [],
                    "legs": 0,
                    "acc_probability": None,
                    "acc_fair_odds": None,
                    "acc_bookish_odds": None,
                    "status": "empty",
                    "filters": {**params, "category": "two_odds"},
                },
            )
            skipped += 1
        d += timedelta(days=1)

    return JsonResponse({"ok": True, "saved": saved, "empty": skipped})










# matches/views_user_tickets.py
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET
from django.shortcuts import get_object_or_404
from django.template.response import TemplateResponse
from django.utils import timezone
from django.utils.dateparse import parse_date

from .models import UserDailyTicket
from .views import _enrich_selection  # you already have this

@login_required
@require_GET
def my_two_odds_list(request):
    """
    List all saved tickets for the logged-in user, filterable by date range.
    Default: current month to today (UTC).
    """
    today = timezone.now().date()
    start = parse_date(request.GET.get("start") or "") or today.replace(day=1)
    end   = parse_date(request.GET.get("end") or "")   or today
    if end < start:
        start, end = end, start

    qs = (UserDailyTicket.objects
          .filter(user=request.user,
                  ticket_date__gte=start,
                  ticket_date__lte=end)
          .order_by("-ticket_date", "-created_at"))

    ctx = {
        "start": start,
        "end": end,
        "tickets": list(qs),
        "count": qs.count(),
    }
    return TemplateResponse(request, "tickets_two_odds.html", ctx)


@login_required
@require_GET
def my_two_odds_detail(request, pk: int):
    """
    Detail page for a single saved ticket (no settling — just render what was saved).
    """
    udt = get_object_or_404(UserDailyTicket, pk=pk, user=request.user)
    legs = [_enrich_selection(s.copy()) for s in (udt.selections or [])]

    ctx = {
        "brand": "My Saved Ticket",
        "ticket_date": udt.ticket_date.isoformat(),
        "has_ticket": bool(legs),
        "legs": legs,
        "legs_count": udt.legs,
        "acc_bookish_odds": udt.acc_bookish_odds,
        "acc_fair_odds": udt.acc_fair_odds,
        "acc_probability_pct": (
            round((udt.acc_probability or 0) * 100, 2)
            if udt.acc_probability is not None else None
        ),
        "status": udt.status or "pending",
        "status_class": f"status-{udt.status or 'pending'}",
    }
    # Reuse your nice ticket template:
    return TemplateResponse(request, "printable_2odds.html", ctx)





















# ---------- Dashboard widget (user’s saved tickets) ----------

from datetime import timedelta
from django.db.models import Q
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET
from django.shortcuts import render
from .models import UserDailyTicket
from .utils import TWO_ODDS_LEAGUE_ID  # (-1)


from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET
from django.shortcuts import render
from django.utils.dateparse import parse_date
from django.utils import timezone

from .models import UserDailyTicket  # make sure this model exists

@login_required
@require_GET
def my_two_odds_dashboard(request):
    # Default range: current month to today (UTC)
    today = timezone.now().date()
    start_q = request.GET.get("start") or ""
    end_q = request.GET.get("end") or ""

    start = parse_date(start_q) or today.replace(day=1)
    end = parse_date(end_q) or today
    if end < start:
        start, end = end, start  # normalize

    qs = (
        UserDailyTicket.objects
        .filter(user=request.user, ticket_date__gte=start, ticket_date__lte=end)
        .order_by("-ticket_date", "-created_at")
    )

    context = {
        "start": start,
        "end": end,
        "tickets": list(qs),   # always provide a list
        "count": qs.count(),   # provided for convenience
    }
    return render(request, "tickets_two_odds.html", context)
