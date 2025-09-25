# views.py
from .models import DailyTicket
# BEFORE
from datetime import datetime, timedelta, timezone

# AFTER
from datetime import timedelta
from django.utils import timezone


# views.py (add this; leave your existing views untouched)
from datetime import date as _date

from django.http import JsonResponse
from .models import DailyTicket
from datetime import datetime, timedelta, timezone
from django.http import JsonResponse
from django.db.models import Prefetch
from .models import Match, PredictedMarket, MatchPrediction

# views.py
from django.http import JsonResponse
from django.db.models import Prefetch
from django.utils import timezone
from datetime import timedelta
from .models import Match, PredictedMarket, MatchPrediction

# Accept these names as 1X2 “aliases” just in case your model varies
IX2_NAMES = {"1X2", "FT_1X2", "MATCH_ODDS", "FULLTIME_1X2"}
# views.py
def upcoming_preictions_json(request, league_id, days):
    now = timezone.now()
    upto = now + timedelta(days=days)

    # Prefetch markets in the same window
    pm_qs = (PredictedMarket.objects
             .filter(kickoff_utc__gte=now, kickoff_utc__lte=upto)
             .order_by("market_code", "specifier"))

    # Build base match queryset (optionally filter by league)
    match_qs = Match.objects.filter(
        kickoff_utc__gte=now, kickoff_utc__lte=upto
    )

    # Accept 0 or "all" to mean all leagues
    if str(league_id).lower() not in {"0", "all"}:
        match_qs = match_qs.filter(league_id=league_id)

    matches = (match_qs
        .select_related("home", "away", "league")
        .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs))
        .order_by("kickoff_utc"))

    # Pull predictions map (don’t over-filter by league if we're showing all)
    mp_map = {
        mp.match_id: mp
        for mp in MatchPrediction.objects.filter(
            kickoff_utc__gte=now,
            kickoff_utc__lte=upto,
            **({} if str(league_id).lower() in {"0", "all"} else {"league_id": league_id})
        )
    }

    out = []
    for m in matches:
        markets = [{
            "market": pm.market_code,
            "specifier": pm.specifier,
            "p": pm.p_model,
            "fair_odds": (1.0 / pm.p_model) if pm.p_model else None,
        } for pm in m.predicted_markets.all()]

        mp = mp_map.get(m.id)
        lambdas = {}
        if mp:
            def get(name): return getattr(mp, name, None)
            blocks = {
                "goals":   {"home": get("lambda_home"),             "away": get("lambda_away")},
                "corners": {"home": get("lambda_corners_home"),     "away": get("lambda_corners_away")},
                "cards":   {"home": get("lambda_cards_home"),       "away": get("lambda_cards_away")},
                "yellows": {"home": get("lambda_yellows_home"),     "away": get("lambda_yellows_away")},
                "reds":    {"home": get("lambda_reds_home"),        "away": get("lambda_reds_away")},
            }
            for k, v in blocks.items():
                if (v["home"] is not None) or (v["away"] is not None):
                    lambdas[k] = v

        out.append({
            "match_id": m.id,
            "kickoff_utc": m.kickoff_utc.isoformat(),
            "home": getattr(m.home, "name", "") or "",
            "away": getattr(m.away, "name", "") or "",
            "lambdas": lambdas,
            "markets": markets,
            "home_logo": getattr(m.home, "logo_url", "") or "",
            "away_logo": getattr(m.away, "logo_url", "") or "",
            # add league info so the table can group/show it
            "league_id": m.league_id,
            "league_name": getattr(m.league, "name", "") or "",
            "league_logo": getattr(m.league, "logo_url", "") or "",
            "status": m.status or "NS",
        })

    return JsonResponse(
        {"league_id": league_id, "count": len(out), "matches": out},
        json_dumps_params={"indent": 2}
    )




# prediction/matches/views.py
from datetime import datetime, timedelta, timezone
from django.shortcuts import render
from django.db.models import Prefetch, Q, F
from .models import Match, PredictedMarket

# choose which markets can be “top pick”
TOP_MARKET_WHITELIST = {
    "1X2", "OU", "BTTS", "TEAM_TOTAL",
    "OU_CORNERS", "TEAM_TOTAL_CORNERS",
    "CARDS_TOT", "YELLOWS_TOT", "REDS_TOT",
}

def _top_pick(pm_iter):
    """Return (market_code, specifier, p_model) with the highest p_model among allowed markets."""
    best = None
    for pm in pm_iter:
        if pm.market_code not in TOP_MARKET_WHITELIST:
            continue
        if pm.p_model is None:
            continue
        if (best is None) or (pm.p_model > best[2]):
            best = (pm.market_code, pm.specifier, float(pm.p_model))
    return best

def dashboard(request, league_id):
    day = int(request.GET.get("day", 0))
    day = max(0, min(day, 7))  # clamp 0..7
    now = datetime.now(timezone.utc)

    # window for the selected day
    start = (now + timedelta(days=day)).replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)

    # upcoming fixtures for the day (center column)
    pm_qs_for_day = PredictedMarket.objects.filter(
        league_id=league_id,
        kickoff_utc__gte=start,
        kickoff_utc__lt=end,
    ).order_by("-p_model")  # helps when scanning for top pick

    upcoming = (Match.objects
        .filter(
            league_id=league_id,
            kickoff_utc__gte=start,
            kickoff_utc__lt=end,
            status__in=["NS", "TBD", "PST"],
        )
        .select_related("home","away")
        .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs_for_day))
        .order_by("kickoff_utc"))

    # completed matches (today, left column)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end   = today_start + timedelta(days=1)
    completed = (Match.objects
        .filter(
            league_id=league_id,
            kickoff_utc__gte=today_start,
            kickoff_utc__lt=today_end,
            status__in=["FT", "AET", "PEN"],
        )
        .select_related("home","away")
        .order_by("kickoff_utc"))

    # today’s ticket (right column): take top pick from each **today** fixture
    pm_qs_today = PredictedMarket.objects.filter(
        league_id=league_id,
        kickoff_utc__gte=today_start,
        kickoff_utc__lt=today_end,
    ).order_by("-p_model")

    today_fixtures = (Match.objects
        .filter(
            league_id=league_id,
            kickoff_utc__gte=today_start,
            kickoff_utc__lt=today_end,
            status__in=["NS", "TBD", "PST", "1H", "HT", "2H", "LIVE"],  # include live
        )
        .select_related("home","away")
        .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs_today))
        .order_by("kickoff_utc"))

    ticket = []
    any_started = False
    for m in today_fixtures:
        # detect started
        if m.status not in ("NS", "TBD", "PST"):
            any_started = True
        pick = _top_pick(m.predicted_markets.all())
        if pick:
            market, spec, p = pick
            ticket.append({
                "match_id": m.id,
                "koff": m.kickoff_utc,
                "home": m.home.name,
                "away": m.away.name,
                "market": market,
                "specifier": spec,
                "p": p,
                # outcome: if you already compute/store it, replace with that
                "outcome": None,  # "WIN"|"LOSE"|None
            })

    # prepare center col cards with top pick too
    upcoming_ctx = []
    for m in upcoming:
        pick = _top_pick(m.predicted_markets.all())
        upcoming_ctx.append({
            "match_id": m.id,
            "koff": m.kickoff_utc,
            "home": m.home.name,
            "away": m.away.name,
            "status": m.status,
            "top_pick": None if not pick else {
                "market": pick[0], "specifier": pick[1], "p": pick[2]
            }
        })

    # day tabs 0..7
    day_tabs = [{"val": i, "label": "Today" if i==0 else ("Tomorrow" if i==1 else f"{i} days"),
                 "active": (i==day)} for i in range(0, 8)]

    context = {
        "league_id": league_id,
        "day": day,
        "day_tabs": day_tabs,
        "upcoming": upcoming_ctx,
        "completed": completed,
        "ticket": ticket,
        "ticket_locked": not any_started,  # blur until something is live/started
    }
    return render(request, "dashboard.html", context)




# views.py (top of file)
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.http import JsonResponse

# If you already have a real _enrich_selection elsewhere, remove this fallback.
def _enrich_selecion(sel: dict) -> dict:
    """
    Safe fallback. If you already have a version that attaches bookish_odds,
    result, etc, keep using yours. This one just ensures keys exist.
    """
    sel.setdefault("bookish_odds", None)
    sel.setdefault("result", None)  # "WIN" | "LOSE" | None
    return sel


# views.py
from django.utils import timezone
from django.utils.dateparse import parse_datetime

# views.py
from datetime import timezone as dt_timezone
from django.utils import timezone as dj_tz
from django.utils.dateparse import parse_datetime

from datetime import timezone as dt_timezone
from django.utils import timezone as dj_tz
from django.utils.dateparse import parse_datetime

def _enrich_selection(sel: dict) -> dict:
    from .models import Match
    m = Match.objects.select_related("home", "away").filter(id=sel.get("match_id")).first()
    if m:
        sel["home"] = getattr(m.home, "name", sel.get("home", ""))
        sel["away"] = getattr(m.away, "name", sel.get("away", ""))
        sel["home_logo"] = getattr(m.home, "logo_url", sel.get("home_logo", ""))
        sel["away_logo"] = getattr(m.away, "logo_url", sel.get("away_logo", ""))
        sel["status"] = (m.status or "")
        sel["goals_home"] = getattr(m, "goals_home", None)
        sel["goals_away"] = getattr(m, "goals_away", None)

        # Kickoff formatting → UTC label
        dt = m.kickoff_utc
        if isinstance(dt, str):
            dt = parse_datetime(dt)
        if dt:
            if dj_tz.is_naive(dt):
                dt = dj_tz.make_aware(dt, dt_timezone.utc)
            dt_utc = dt.astimezone(dt_timezone.utc)
            sel["kickoff_ts"] = dt_utc.isoformat()
            sel["kickoff_label"] = dt_utc.strftime("%b %d, %H:%M") + " UTC"
        else:
            sel["kickoff_ts"] = sel.get("kickoff_utc")
            sel["kickoff_label"] = sel.get("kickoff_utc", "")
    else:
        sel.setdefault("status", "")
        sel.setdefault("kickoff_label", sel.get("kickoff_utc", ""))

    sel.setdefault("result", None)
    sel.setdefault("p", None)
    sel.setdefault("fair_odds", None)
    sel.setdefault("bookish_odds", None)
    return sel




def daily_ticket_json(request):
    q = request.GET.get("date")
    d_obj = parse_date(q) if q else None
    ticket_date = d_obj or timezone.now().date()

    dt = (DailyTicket.objects
          .filter(ticket_date=ticket_date, league_id=0)  # global
          .order_by("-id")
          .first())

    if not dt:
        return JsonResponse({"ticket": None, "ticket_date": ticket_date.isoformat()}, json_dumps_params={"indent": 2})

    legs = [_enrich_selection(sel.copy()) for sel in (dt.selections or [])]

    payload = {
        "date": ticket_date.isoformat(),
        "legs": dt.legs,
        "status": dt.status,
        "acc_probability": round((dt.acc_probability or 0) * 100, 2),  # %
        "acc_fair_odds": dt.acc_fair_odds,
        "acc_bookish_odds": dt.acc_bookish_odds,
        "selections": legs,
    }
    return JsonResponse({"ticket": payload}, json_dumps_params={"indent": 2})


def daily_2_odds_ticket_json(request):
    q = request.GET.get("date")
    d_obj = parse_date(q) if q else None
    ticket_date = d_obj or timezone.now().date()

    dt = (DailyTicket.objects
          .filter(ticket_date=ticket_date, league_id=-1)  # global
          .order_by("-id")
          .first())

    if not dt:
        return JsonResponse({"ticket": None, "ticket_date": ticket_date.isoformat()}, json_dumps_params={"indent": 2})

    legs = [_enrich_selection(sel.copy()) for sel in (dt.selections or [])]

    payload = {
        "date": ticket_date.isoformat(),
        "legs": dt.legs,
        "status": dt.status,
        "acc_probability": round((dt.acc_probability or 0) * 100, 2),  # %
        "acc_fair_odds": dt.acc_fair_odds,
        "acc_bookish_odds": dt.acc_bookish_odds,
        "selections": legs,
    }
    return JsonResponse({"ticket": payload}, json_dumps_params={"indent": 2})






def daily_tcket_json(request):
    q = request.GET.get("date")
    d_obj = parse_date(q) if q else None
    ticket_date = d_obj or timezone.now().date()

    from .models import DailyTicket  # local import to avoid cycles, if any
    dt = (DailyTicket.objects
          .filter(ticket_date=ticket_date)
          .order_by("-id")
          .first())

    if not dt:
        return JsonResponse({
            "ticket": None,
            "ticket_date": ticket_date.isoformat(),
            "message": "No ticket for this date."
        }, json_dumps_params={"indent": 2})

    selections = []
    for sel in (dt.selections or []):
        selections.append(_enrich_selection(sel.copy()))

    payload = {
        "date": ticket_date.isoformat(),
        "legs": dt.legs,
        "status": dt.status,
        "acc_probability": round(dt.acc_probability * 100, 2),  # percent
        "acc_fair_odds": dt.acc_fair_odds,
        "acc_bookish_odds": dt.acc_bookish_odds,
        "selections": selections,
        
    }
    return JsonResponse({"ticket": payload}, json_dumps_params={"indent": 2})


def daily_ticet_json(request):
    q = request.GET.get("date")
    d_obj = parse_date(q) if q else None
    ticket_date = d_obj or timezone.now().date()

    dt = (DailyTicket.objects
          .filter(ticket_date=ticket_date)
          .order_by("-id")
          .first())

    if not dt:
        return JsonResponse({
            "ticket": None,
            "ticket_date": ticket_date.isoformat(),
            "message": "No ticket for this date."
        }, json_dumps_params={"indent": 2})

    selections = []
    for sel in (dt.selections or []):
        selections.append(_enrich_selection(sel.copy()))

    payload = {
        "date": ticket_date.isoformat(),
        "legs": dt.legs,
        "status": dt.status,
        "acc_probability": round(dt.acc_probability * 100, 2),  # in %
        "acc_fair_odds": dt.acc_fair_odds,
        "acc_bookish_odds": dt.acc_bookish_odds,
        "selections": selections,
    }
    return JsonResponse({"ticket": payload}, json_dumps_params={"indent": 2})




from .models import Match

def _erich_selection(sel):
    """Attach match names and outcome to a stored leg."""
    m = Match.objects.filter(id=sel["match_id"]).select_related("home", "away").first()
    if m:
        sel["home"] = m.home.name
        sel["away"] = m.away.name

        # if match completed, compare result
        if m.status in ["FT", "AET", "PEN"]:
            # ⚽️ Example: settle only OU (over/under) markets
            if sel["market"] == "OU":
                total_goals = (m.home_goals or 0) + (m.away_goals or 0)
                if "over" in sel["specifier"]:
                    th = float(sel["specifier"].split("_")[0])  # e.g. "2.5_over"
                    sel["result"] = "WIN" if total_goals > th else "LOSE"
                elif "under" in sel["specifier"]:
                    th = float(sel["specifier"].split("_")[0])
                    sel["result"] = "WIN" if total_goals < th else "LOSE"
            # you can add BTTS, 1X2, etc. similarly
    return sel








# views.py (add near your other JSON views)
import json
from datetime import datetime, timezone, date as _date
from django.http import JsonResponse
from django.utils.dateparse import parse_date
from .models import DailyTicket

def daily_tickt_json(request):
    """
    Returns the global daily ticket for ?date=YYYY-MM-DD (UTC).
    If no date given, uses today's UTC date.
    """
    q = request.GET.get("date")
    d_obj = parse_date(q) if q else None
    ticket_date = d_obj or datetime.now().date()

    dt = (DailyTicket.objects
          .filter(ticket_date=ticket_date)
          .order_by("-id")
          .first())

    if not dt:
        return JsonResponse({
            "ticket": None,
            "ticket_date": ticket_date.isoformat(),
            "message": "No ticket for this date."
        }, json_dumps_params={"indent": 2})

    # Build a stable payload from DailyTicket JSON
    payload = {
        "date": ticket_date.isoformat(),
        "legs": dt.legs,
        "status": dt.status,
        "acc_probability": dt.acc_probability,
        "acc_fair_odds": dt.acc_fair_odds,
        "acc_bookish_odds": dt.acc_bookish_odds,
        "selections": dt.selections or [],
    }
    return JsonResponse({"ticket": payload}, json_dumps_params={"indent": 2})





# views.py (add this; leave your existing views untouched)
from datetime import date as _date
from django.utils import timezone
from django.http import JsonResponse
from .models import DailyTicket

def daily_tickt_json(request):
    """
    Landing-page JSON for the global daily ticket.
    Query params:
      - date=YYYY-MM-DD  (optional; defaults to today UTC)
      - fallback=1       (optional; if today's missing, return latest available)
    """
    qd = request.GET.get("date")
    if qd:
        try:
            y, m, d = map(int, qd.split("-"))
            day = _date(y, m, d)
        except Exception:
            return JsonResponse({"error": "Invalid date format, use YYYY-MM-DD"}, status=400)
    else:
        day = timezone.now().date()

    # Fetch today’s ticket or latest (if fallback requested)
    t = DailyTicket.objects.filter(ticket_date=day, league_id=0).first()
    if not t and request.GET.get("fallback") == "1":
        t = DailyTicket.objects.order_by("-ticket_date", "-id").first()

    if not t:
        return JsonResponse({"ticket": None, "message": "No ticket available"}, json_dumps_params={"indent": 2})

    # Selections is expected to be a JSON array of legs; pass through with light normalization
    raw_legs = t.selections or []
    legs_out = []
    for leg in raw_legs:
        # Be forgiving about key names
        m = {
            "match_id":          leg.get("match_id"),
            "kickoff_utc":       leg.get("kickoff_utc"),
            "home":              leg.get("home"),
            "away":              leg.get("away"),
            "market":            leg.get("market") or leg.get("market_code"),
            "specifier":         leg.get("specifier"),
            "p":                 float(leg["p"]) if leg.get("p") is not None else None,
            "fair_odds":         float(leg["fair_odds"]) if leg.get("fair_odds") is not None else None,
            "bookish_odds":      float(leg["bookish_odds"]) if leg.get("bookish_odds") is not None else None,
            "result":            leg.get("result"),   # "WIN" | "LOSE" | None, if you set it later
        }
        legs_out.append(m)

    payload = {
        "date":              t.ticket_date.isoformat(),
        "legs":              int(getattr(t, "legs", len(legs_out)) or len(legs_out)),
        "status":            t.status,
        "acc_probability":   float(t.acc_probability or 0.0),
        "acc_fair_odds":     float(t.acc_fair_odds or 0.0),
        "acc_bookish_odds":  float(t.acc_bookish_odds or 0.0),
        "selections":        legs_out,
    }
    return JsonResponse({"ticket": payload}, json_dumps_params={"indent": 2})












# views.py (add below your existing views)
from datetime import timedelta
from django.http import JsonResponse
from django.db.models import Prefetch
from django.utils import timezone
from .models import Match, PredictedMarket, MatchPrediction

def completed_tody_json(request, league_id):
    now = timezone.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)

    pm_qs = (PredictedMarket.objects
             .filter(kickoff_utc__gte=start, kickoff_utc__lt=end)
             .order_by("market_code", "specifier"))

    match_qs = Match.objects.filter(
        kickoff_utc__gte=start,
        kickoff_utc__lt=end,
        status__in=["FT", "AET", "PEN"],
    )
    if str(league_id).lower() not in {"0", "all"}:
        match_qs = match_qs.filter(league_id=league_id)

    matches = (match_qs
        .select_related("home", "away", "league")
        .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs))
        .order_by("kickoff_utc"))

    mp_map = {
        mp.match_id: mp
        for mp in MatchPrediction.objects.filter(
            kickoff_utc__gte=start,
            kickoff_utc__lt=end,
            **({} if str(league_id).lower() in {"0", "all"} else {"league_id": league_id})
        )
    }

    out = []
    for m in matches:
        markets = [{
            "market": pm.market_code,
            "specifier": pm.specifier,
            "p": pm.p_model,
            "fair_odds": (1.0 / pm.p_model) if pm.p_model else None,
        } for pm in m.predicted_markets.all()]

        mp = mp_map.get(m.id)
        lambdas = {}
        if mp:
            def get(name): return getattr(mp, name, None)
            blocks = {
                "goals":   {"home": get("lambda_home"),             "away": get("lambda_away")},
                "corners": {"home": get("lambda_corners_home"),     "away": get("lambda_corners_away")},
                "cards":   {"home": get("lambda_cards_home"),       "away": get("lambda_cards_away")},
                "yellows": {"home": get("lambda_yellows_home"),     "away": get("lambda_yellows_away")},
                "reds":    {"home": get("lambda_reds_home"),        "away": get("lambda_reds_away")},
            }
            for k, v in blocks.items():
                if (v["home"] is not None) or (v["away"] is not None):
                    lambdas[k] = v

        out.append({
            "match_id": m.id,
            "kickoff_utc": m.kickoff_utc.isoformat(),
            "home": getattr(m.home, "name", "") or "",
            "away": getattr(m.away, "name", "") or "",
            "result": {
                "status": m.status,
                "goals_home": getattr(m, "goals_home", None),
                "goals_away": getattr(m, "goals_away", None),
            },
            "lambdas": lambdas,
            "markets": markets,
            "home_logo": getattr(m.home, "logo_url", "") or "",
            "away_logo": getattr(m.away, "logo_url", "") or "",
            "league_id": m.league_id,
            "league_name": getattr(m.league, "name", "") or "",
            "league_logo": getattr(m.league, "logo_url", "") or "",
        })

    return JsonResponse(
        {"league_id": league_id, "count": len(out), "matches": out},
        json_dumps_params={"indent": 2}
    )








## views.py (add this near your other views)
import json
from django.shortcuts import render
from django.utils import timezone

# assuming these three views are in the same file/module
# upcoming_predictions_json(request, league_id, days)
# completed_today_json(request, league_id)
# daily_ticket_json(request)

def combned_dashboard(request, league_id, days):
    """
    Aggregates:
      - upcoming_predictions_json(league_id, days)    -> upcoming fixtures + markets + lambdas
      - completed_today_json(league_id)               -> today's completed matches
      - daily_ticket_json()                           -> global daily ticket for today
    and passes all three payloads to one template.
    """
    # normalize/guard day selection 0..7
    try:
        selected_day = max(0, min(7, int(days)))
    except Exception:
        selected_day = 0

    # build day tabs for the UI
    day_tabs = []
    for i in range(0, 8):
        if i == 0:
            label = "Today"
        elif i == 1:
            label = "Tomorrow"
        else:
            label = f"{i} days"
        day_tabs.append({
            "val": i,
            "label": label,
            "active": (i == selected_day),
        })

    # 1) Upcoming (bytes -> str -> dict)
    up_resp = upcoming_predictions_json(request, league_id, selected_day)
    upcoming = {}
    try:
        upcoming = json.loads(up_resp.content.decode("utf-8"))
    except Exception:
        upcoming = {"league_id": league_id, "count": 0, "matches": []}

    # 2) Completed today
    comp_resp = completed_today_json(request, league_id)
    completed = {}
    try:
        completed = json.loads(comp_resp.content.decode("utf-8"))
    except Exception:
        completed = {"league_id": league_id, "count": 0, "matches": []}

    # 3) Global daily ticket
    ticket_resp = daily_ticket_json(request)
    daily_ticket = {}
    try:
        daily_ticket = json.loads(ticket_resp.content.decode("utf-8"))
    except Exception:
        daily_ticket = {"ticket_date": str(timezone.now().date()), "legs": []}

    context = {
        "league_id": league_id,
        "day": selected_day,
        "day_tabs": day_tabs,
        # full JSON dicts (so template can use anything inside)
        "upcoming_json": upcoming,
        "completed_today_json": completed,
        "daily_ticket_json": daily_ticket,
        # convenience shortcuts (common in templates)
        "upcoming_matches": upcoming.get("matches", []),
        "completed_matches": completed.get("matches", []),
        "ticket": daily_ticket,  # same as daily_ticket_json
    }
    return render(request, "index.html", context)





def daily_ticet_json(request):
    q = request.GET.get("date")
    d_obj = parse_date(q) if q else None
    ticket_date = d_obj or timezone.now().date()

    dt = (DailyTicket.objects
          .filter(ticket_date=ticket_date)
          .order_by("-id")
          .first())

    if not dt:
        return JsonResponse({
            "ticket": None,
            "ticket_date": ticket_date.isoformat(),
            "message": "No ticket for this date."
        }, json_dumps_params={"indent": 2})

    selections = []
    for sel in (dt.selections or []):
        selections.append(_enrich_selection(sel.copy()))

    payload = {
        "date": ticket_date.isoformat(),
        "legs": dt.legs,
        "status": dt.status,
        "acc_probability": round(dt.acc_probability * 100, 2),  # in %
        "acc_fair_odds": dt.acc_fair_odds,
        "acc_bookish_odds": dt.acc_bookish_odds,
        "selections": selections,
    }
    return JsonResponse({"ticket": payload}, json_dumps_params={"indent": 2})





# views.py (your combined page)
import json
from django.shortcuts import render
from django.http import HttpRequest

def combine_dasboard(request: HttpRequest, league_id, days):
    # keep your existing imports to the 3 JSON views
    up_resp = upcoming_predictions_json(request, league_id, days)
    upcoming = json.loads(up_resp.content.decode("utf-8"))

    comp_resp = completed_today_json(request, league_id)
    completed = json.loads(comp_resp.content.decode("utf-8"))

    # Pick date (or today) and call the JSON endpoint with it
    ticket_date = request.GET.get("ticket_date", "")
    if ticket_date:
        # simulate querystring by cloning request.GET
        from django.test import RequestFactory
        rf = RequestFactory()
        req2 = rf.get("/_daily_ticket", {"date": ticket_date})
        ticket_resp = daily_ticket_json(req2)
    else:
        ticket_resp = daily_ticket_json(request)

    daily_ticket = json.loads(ticket_resp.content.decode("utf-8"))

    context = {
        "league_id": league_id,
        "days": days,
        "ticket_date": ticket_date,
        "upcoming_json": upcoming,
        "completed_today_json": completed,
        "daily_ticket_json": daily_ticket,
    }
    return render(request, "index.html", context)





# views.py
from collections import Counter

def _is_all(value) -> bool:
    return str(value).lower() in {"0", "all"}

def upcoming_predictions_json(request, league_id, days):
    now  = timezone.now()
    upto = now + timedelta(days=int(days))

    # Markets for this time window (no league filter here)
    pm_qs = (PredictedMarket.objects
             .filter(kickoff_utc__gte=now, kickoff_utc__lte=upto)
             .order_by("market_code", "specifier"))

    match_qs = Match.objects.filter(kickoff_utc__gte=now, kickoff_utc__lte=upto)
    if not _is_all(league_id):
        match_qs = match_qs.filter(league_id=league_id)

    matches = (match_qs
        .select_related("home", "away", "league")
        .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs))
        .order_by("kickoff_utc"))

    # Predictions (only filter by league if not ALL)
    mp_filter = dict(kickoff_utc__gte=now, kickoff_utc__lte=upto)
    if not _is_all(league_id):
        mp_filter["league_id"] = league_id
    mp_map = {mp.match_id: mp for mp in MatchPrediction.objects.filter(**mp_filter)}

    out = []
    league_counter = Counter()

    for m in matches:
        league_counter[m.league_id] += 1

        markets = [{
            "market": pm.market_code,
            "specifier": pm.specifier,
            "p": pm.p_model,
            "fair_odds": (1.0 / pm.p_model) if pm.p_model else None,
        } for pm in m.predicted_markets.all()]

        mp = mp_map.get(m.id)
        lambdas = {}
        if mp:
            def get(name): return getattr(mp, name, None)
            blocks = {
                "goals":   {"home": get("lambda_home"),         "away": get("lambda_away")},
                "corners": {"home": get("lambda_corners_home"), "away": get("lambda_corners_away")},
                "cards":   {"home": get("lambda_cards_home"),   "away": get("lambda_cards_away")},
                "yellows": {"home": get("lambda_yellows_home"), "away": get("lambda_yellows_away")},
                "reds":    {"home": get("lambda_reds_home"),    "away": get("lambda_reds_away")},
            }
            for k, v in blocks.items():
                if (v["home"] is not None) or (v["away"] is not None):
                    lambdas[k] = v

        out.append({
            "match_id": m.id,
            "kickoff_utc": m.kickoff_utc.isoformat(),
            "home": getattr(m.home, "name", "") or "",
            "away": getattr(m.away, "name", "") or "",
            "status": m.status or "NS",
            "lambdas": lambdas,
            "markets": markets,
            "home_logo": getattr(m.home, "logo_url", "") or "",
            "away_logo": getattr(m.away, "logo_url", "") or "",
            "league_id": m.league_id,
            "league_name": getattr(m.league, "name", "") or "",
            "league_logo": getattr(m.league, "logo_url", "") or "",
        })

    # Optional debug flag ?debug=1
    debug = {}
    if request.GET.get("debug") == "1":
        debug = {
            "leagues_found": len(league_counter),
            "by_league": league_counter.most_common(20),
        }

    return JsonResponse(
        {"league_id": league_id, "count": len(out), "matches": out, "debug": debug},
        json_dumps_params={"indent": 2}
    )


def completed_today_json(request, league_id):
    now   = timezone.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)
    cutoff = timezone.now() - timedelta(hours=24)

    pm_qs = (PredictedMarket.objects
             .filter(kickoff_utc__gte=start, kickoff_utc__lt=end)
             .order_by("market_code", "specifier"))

    match_qs = Match.objects.filter(
        kickoff_utc__gte=cutoff,          # Last 24 hours
        status__in=["FT", "AET", "PEN"]   # Completed status
    ).order_by('-kickoff_utc')            # Most recent first

    if not _is_all(league_id):
        match_qs = match_qs.filter(league_id=league_id)
    print(f"DEBUG: Found {match_qs.count()} matches")
   
    matches = (match_qs
        .select_related("home", "away", "league")
        .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs))
        .order_by("kickoff_utc"))
    
    print(f"DEBUG: Found {match_qs.count()} matches")

    mp_filter = dict(kickoff_utc__gte=start, kickoff_utc__lt=end)
    if not _is_all(league_id):
        mp_filter["league_id"] = league_id
    mp_map = {mp.match_id: mp for mp in MatchPrediction.objects.filter(**mp_filter)}

    out = []
    for m in matches:
        markets = [{
            "market": pm.market_code,
            "specifier": pm.specifier,
            "p": pm.p_model,
            "fair_odds": (1.0 / pm.p_model) if pm.p_model else None,
        } for pm in m.predicted_markets.all()]

        mp = mp_map.get(m.id)
        lambdas = {}
        if mp:
            def get(name): return getattr(mp, name, None)
            blocks = {
                "goals":   {"home": get("lambda_home"),         "away": get("lambda_away")},
                "corners": {"home": get("lambda_corners_home"), "away": get("lambda_corners_away")},
                "cards":   {"home": get("lambda_cards_home"),   "away": get("lambda_cards_away")},
                "yellows": {"home": get("lambda_yellows_home"), "away": get("lambda_yellows_away")},
                "reds":    {"home": get("lambda_reds_home"),    "away": get("lambda_reds_away")},
            }
            for k, v in blocks.items():
                if (v["home"] is not None) or (v["away"] is not None):
                    lambdas[k] = v

        out.append({
    "match_id": m.id,
    "kickoff_utc": m.kickoff_utc.isoformat(),
    "home": getattr(m.home, "name", "") or "",
    "away": getattr(m.away, "name", "") or "",

    # ✅ add flat fields for the frontend
    "status": m.status,
    "goals_home": getattr(m, "goals_home", None),
    "goals_away": getattr(m, "goals_away", None),

    # keep nested for backward-compat if you want
    "result": {
        "status": m.status,
        "goals_home": getattr(m, "goals_home", None),
        "goals_away": getattr(m, "goals_away", None),
    },

    "lambdas": lambdas,
    "markets": markets,
    "home_logo": getattr(m.home, "logo_url", "") or "",
    "away_logo": getattr(m.away, "logo_url", "") or "",
    "league_id": m.league_id,
    "league_name": getattr(m.league, "name", "") or "",
    "league_logo": getattr(m.league, "logo_url", "") or "",
        })


    return JsonResponse(
        {"league_id": league_id, "count": len(out), "matches": out},
        json_dumps_params={"indent": 2}
    )



# views.py
from django.shortcuts import render, get_object_or_404
from django.db.models import Prefetch
from .models import Match, PredictedMarket, MatchPrediction







# views.py
from django.utils.dateparse import parse_date
from django.utils import timezone
from datetime import timedelta

def combned_dashboard(request, league_id, days):
    # --- Resolve selected date (if provided) ---
    qd = request.GET.get("date")  # e.g. 2025-09-02
    today_utc = timezone.now().date()
    if qd:
        sel_date = parse_date(qd)
        if not sel_date:
            sel_date = today_utc
    else:
        # fall back to days offset (existing path param)
        sel_date = today_utc + timedelta(days=int(days))

    # Build a 14-day date rail starting today
    date_rail = []
    for i in range(14):
        d = today_utc + timedelta(days=i)
        date_rail.append({
            "iso": d.isoformat(),               # "2025-09-02"
            "dow": d.strftime("%a"),            # "Tue"
            "day": d.strftime("%-d") if hasattr(d, "strftime") else str(d.day),  # "2" (Linux/Mac) 
            "active": (d == sel_date),
            "url": f"/mega/{league_id}/{i}/?date={d.isoformat()}",
        })

    # Use the selected date when fetching JSON
    # We’ll pass it through to your existing JSON views via RequestFactory so they
    # query by date window instead of `days`.
    from django.test import RequestFactory
    rf = RequestFactory()
    req_up = rf.get("/api/upcoming", {"date": sel_date.isoformat()})
    req_co = rf.get("/api/completed", {"date": sel_date.isoformat()})

    # 1) Upcoming (use your existing function; it ignores GET unless you add support)
    up_resp = upcoming_predictions_json(req_up, league_id, 0)  # days param unused when ?date present (see note below)
    upcoming = json.loads(up_resp.content.decode("utf-8"))

    # 2) Completed for that date
    comp_resp = completed_today_json(req_co, league_id)        # same note
    completed = json.loads(comp_resp.content.decode("utf-8"))

    # 3) Ticket (already supports ?date in your code path)
    req_ticket = rf.get("/_daily_ticket", {"date": sel_date.isoformat()})
    ticket_resp = daily_ticket_json(req_ticket)
    daily_ticket = json.loads(ticket_resp.content.decode("utf-8"))

    context = {
        "league_id": league_id,
        "days": days,
        "selected_date": sel_date.isoformat(),
        "date_rail": date_rail,                       # 👈 new
        "upcoming_json": upcoming,
        "completed_today_json": completed,
        "daily_ticket_json": daily_ticket,
        "league_label": "All Leagues" if str(league_id).lower() in {"0","all"} else f"League {league_id}",
    }
    return render(request, "index.html", context)
















# matches/views.py
import json
from datetime import timedelta
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.shortcuts import render
from django.test import RequestFactory

def combind_dashboard(request, league_id, days):
    # --- keep your existing calls ---
    up_resp = upcoming_predictions_json(request, league_id, days)
    upcoming = json.loads(up_resp.content.decode("utf-8"))

    comp_resp = completed_today_json(request, league_id)
    completed = json.loads(comp_resp.content.decode("utf-8"))

    ticket_date = request.GET.get("ticket_date", "")
    if ticket_date:
        rf = RequestFactory()
        req2 = rf.get("/_daily_ticket", {"date": ticket_date})
        ticket_resp = daily_ticket_json(req2)
        selected_date = ticket_date
    else:
        ticket_resp = daily_ticket_json(request)
        try:
            selected_date = json.loads(ticket_resp.content.decode("utf-8"))["ticket"]["date"]
        except Exception:
            selected_date = timezone.now().date().isoformat()

    daily_ticket = json.loads(ticket_resp.content.decode("utf-8"))

    # --- Build incoming_matches from upcoming["matches"] ---
    def extract_1x2(markets):
        pH = pD = pA = None
        # exact FT 1X2 first
        for m in markets:
            if str(m.get("market", "")).upper() == "1X2":
                s = (m.get("specifier") or "").upper()
                if s == "H": pH = float(m.get("p"))
                elif s == "D": pD = float(m.get("p"))
                elif s == "A": pA = float(m.get("p"))
        # fallback to any 1X2_* if nothing found
        if pH is None and pD is None and pA is None:
            for m in markets:
                if str(m.get("market", "")).upper().startswith("1X2"):
                    s = (m.get("specifier") or "").upper()
                    if s == "H" and pH is None: pH = float(m.get("p"))
                    elif s == "D" and pD is None: pD = float(m.get("p"))
                    elif s == "A" and pA is None: pA = float(m.get("p"))
        best = None
        cands = [(k, v) for k, v in (("H", pH), ("D", pD), ("A", pA)) if v is not None]
        if cands:
            best = max(cands, key=lambda kv: kv[1])[0]
        return pH, pD, pA, best

    incoming_matches = []
    for item in upcoming.get("matches", []):
        markets = item.get("markets", []) or []
        pH, pD, pA, best = extract_1x2(markets)
        # parse kickoff so |date works in template
        dt = parse_datetime(item.get("kickoff_utc") or "")
        incoming_matches.append({
            "id":          item.get("match_id"),
            "kickoff_dt":  dt,  # Django datetime (may be None if parse failed)
            "home":        item.get("home") or "",
            "away":        item.get("away") or "",
            "home_logo":   item.get("home_logo") or "",
            "away_logo":   item.get("away_logo") or "",
            "status":      item.get("status") or "NS",
            "pH": pH, "pD": pD, "pA": pA, "best": best,
        })

    # make sure logos exist in completed too
    for m in completed.get("matches", []):
        m["home_logo"] = m.get("home_logo") or ""
        m["away_logo"] = m.get("away_logo") or ""
    league_label = "All Leagues" if str(league_id).lower() in {"0","all"} else f"League ID: {league_id}"
    context = {
        "league_id": league_id,
        "league_label": league_label,
        "days": days,
        "ticket_date": selected_date,
        "upcoming_json": upcoming,
        "completed_today_json": completed,
        "daily_ticket_json": daily_ticket,
        # NEW: server-rendered list for center column
        "incoming_matches": incoming_matches,
    }
    return render(request, "index.html", context)




# matches/views.py
import json
from datetime import timedelta
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.shortcuts import render
from django.test import RequestFactory

def combid_dashboard(request, league_id, days):
    # --- existing JSON calls (unchanged) ---
    up_resp = upcoming_predictions_json(request, league_id, days)
    upcoming = json.loads(up_resp.content.decode("utf-8"))

    comp_resp = completed_today_json(request, league_id)
    completed = json.loads(comp_resp.content.decode("utf-8"))

    ticket_date = request.GET.get("ticket_date", "")
    if ticket_date:
        rf = RequestFactory()
        req2 = rf.get("/_daily_ticket", {"date": ticket_date})
        ticket_resp = daily_ticket_json(req2)
        selected_ticket_date = ticket_date
    else:
        ticket_resp = daily_ticket_json(request)
        try:
            selected_ticket_date = json.loads(ticket_resp.content.decode("utf-8"))["ticket"]["date"]
        except Exception:
            selected_ticket_date = timezone.now().date().isoformat()

    daily_ticket = json.loads(ticket_resp.content.decode("utf-8"))



    # --- build incoming_matches (server-side) from upcoming JSON ---
    def extract_1x2(markets):
        pH = pD = pA = None
        for m in (markets or []):
            if str(m.get("market", "")).upper() == "1X2":
                s = (m.get("specifier") or "").upper()
                if s == "H": pH = float(m.get("p"))
                elif s == "D": pD = float(m.get("p"))
                elif s == "A": pA = float(m.get("p"))
        # fallback to any 1X2_* if exact not present
        if pH is None and pD is None and pA is None:
            for m in (markets or []):
                if str(m.get("market", "")).upper().startswith("1X2"):
                    s = (m.get("specifier") or "").upper()
                    if s == "H" and pH is None: pH = float(m.get("p"))
                    elif s == "D" and pD is None: pD = float(m.get("p"))
                    elif s == "A" and pA is None: pA = float(m.get("p"))
        best = None
        cands = [(k, v) for k, v in (("H", pH), ("D", pD), ("A", pA)) if v is not None]
        if cands:
            best = max(cands, key=lambda kv: kv[1])[0]
        return pH, pD, pA, best

    incoming_matches = []
    for item in upcoming.get("matches", []):
        dt = parse_datetime(item.get("kickoff_utc") or "")
        pH, pD, pA, best = extract_1x2(item.get("markets", []))
        incoming_matches.append({
            "id":         item.get("match_id"),
            "kickoff_dt": dt,  # may be None if parse failed
            "home":       item.get("home") or "",
            "away":       item.get("away") or "",
            "home_logo":  item.get("home_logo") or "",
            "away_logo":  item.get("away_logo") or "",
            "status":     item.get("status") or "NS",
            "pH": pH, "pD": pD, "pA": pA, "best": best,
        })

    # --- date filter for incoming (UTC calendar day) ---
    date_qs = request.GET.get("date")  # e.g. 2025-09-02
    if date_qs:
        try:
            y, m, d = map(int, date_qs.split("-"))
            target_date = timezone.datetime(y, m, d, tzinfo=timezone.utc).date()
        except Exception:
            target_date = None
        if target_date:
            def same_utc_day(dtobj):
                if not dtobj:
                    return False
                # ensure aware & compare in UTC
                if timezone.is_naive(dtobj):
                    dtobj = timezone.make_aware(dtobj, timezone=timezone.utc)
                return dtobj.astimezone(timezone.utc).date() == target_date
            incoming_matches = [m for m in incoming_matches if same_utc_day(m["kickoff_dt"])]

    # --- make sure logos exist in completed too (safe defaults) ---
    for m in completed.get("matches", []):
        m["home_logo"] = m.get("home_logo") or ""
        m["away_logo"] = m.get("away_logo") or ""

    # --- 7-day calendar rail (links with ?date=YYYY-MM-DD) ---
    today = timezone.now().date()
    # which date is currently selected for incoming rail?
    selected_incoming_date = date_qs or today.isoformat()
    date_rail = []
    for i in range(7):
        d = today + timedelta(days=i)
        date_rail.append({
            "dow": d.strftime("%a"),               # Sun, Mon, ...
            "day": d.strftime("%d"),               # 01..31
            "url": f"/mega/{league_id}/{days}/?date={d.isoformat()}",  # <-- consistent
            "active": (d.isoformat() == selected_incoming_date),
        })

    context = {
        "league_id": league_id,
        "days": days,
        "ticket_date": selected_ticket_date,
        "upcoming_json": upcoming,
        "completed_today_json": completed,
        "daily_ticket_json": daily_ticket,
        "incoming_matches": incoming_matches,   # <— use this in template
        "date_rail": date_rail,                 # <— clickable calendar strip
        "selected_incoming_date": selected_incoming_date,
    }
    return render(request, "index.html", context)







# matches/views.py
import json
from datetime import timedelta
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.shortcuts import render
from django.test import RequestFactory

# NEW









# prediction/matches/views.py  (add near your other imports)
from collections import defaultdict
from datetime import date as _date
from django.shortcuts import get_object_or_404, render
from django.db.models import Q, Prefetch
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from .models import (
    Match, MatchStats, Lineup,
    StandingsRow, PredictedMarket, MatchPrediction
)

FINALS = ("FT", "AET", "PEN")

def _result_for(match: Match, for_home=True):
    """Return 'W','D','L' from the given team's perspective (only if finals exist)."""
    if match.goals_home is None or match.goals_away is None:
        return None
    h, a = match.goals_home, match.goals_away
    if for_home:
        if h > a: return "W"
        if h == a: return "D"
        return "L"
    else:
        if a > h: return "W"
        if a == h: return "D"
        return "L"

# views.py
from collections import defaultdict, deque
from django.db.models import Prefetch, Q
from django.shortcuts import get_object_or_404, render
from django.utils import timezone

from .models import (
    Match, PredictedMarket, MatchPrediction,
    MatchStats, Lineup, StandingsRow
)

# views.py
from collections import defaultdict
from datetime import timedelta

from django.db.models import Q, Prefetch
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from .models import (
    Match, MatchStats, Lineup, StandingsRow, PredictedMarket,
    MatchPrediction, MarketProb
)

def _pct(p):
    try:
        return round(float(p) * 100.0, 1)
    except Exception:
        return None

# prediction/matches/views.py
from collections import defaultdict
from django.shortcuts import get_object_or_404, render
from django.db.models import Q
from django.utils import timezone

from .models import (
    Match, MatchStats, Lineup,
    PredictedMarket, MatchPrediction,
    StandingsRow
)

FINISHED = ("FT", "AET", "PEN")

def _val(obj, *names, default=""):
    """Pick first existing attr/key that is truthy; else default."""
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v not in (None, "", []):
                return v
        if isinstance(obj, dict) and n in obj and obj[n] not in (None, "", []):
            return obj[n]
    return default

# views.py
from collections import defaultdict
from datetime import timezone as dt_timezone
from django.db.models import Q, Prefetch
from django.shortcuts import get_object_or_404, render
from django.utils import timezone

from .models import (
    Match, PredictedMarket, MatchPrediction,
    Lineup, StandingsRow,
)

COMPLETED_STATUSES = ("FT", "AET", "PEN")


def _aware_utc(dt):
    if not dt:
        return None
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc)


def _league_name(league):
    # Tolerant across schema variants
    return getattr(league, "name", None) or getattr(league, "league_name", "") or ""


def _team_logo(team):
    return getattr(team, "logo_url", None) or getattr(team, "logo", "") or ""


def _result_for_team(m: Match, team_id: int):
    if m.goals_home is None or m.goals_away is None:
        return None  # unknown
    if m.home_id == team_id:
        if m.goals_home > m.goals_away: return "W"
        if m.goals_home < m.goals_away: return "L"
        return "D"
    else:
        if m.goals_away > m.goals_home: return "W"
        if m.goals_away < m.goals_home: return "L"
        return "D"


def _last_n_matches_for(team_id: int, upto_dt, n=5):
    qs = (Match.objects
          .filter(
              Q(home_id=team_id) | Q(away_id=team_id),
              status__in=COMPLETED_STATUSES,
              kickoff_utc__lt=upto_dt
          )
          .select_related("home", "away", "league")
          .order_by("-kickoff_utc")[:n])
    out = []
    for m in qs:
        venue = "H" if m.home_id == team_id else "A"
        opp = m.away if venue == "H" else m.home
        out.append({
            "date": _aware_utc(m.kickoff_utc).date().isoformat(),
            "league": _league_name(m.league),
            "venue": venue,
            "opponent": opp.name,
            "opp_logo": _team_logo(opp),
            "score": f"{m.goals_home}–{m.goals_away}" if m.goals_home is not None and m.goals_away is not None else "-",
            "result": _result_for_team(m, team_id),  # W/D/L/None
        })
    return out


def _last_n_h2h(home_id: int, away_id: int, upto_dt, n=5):
    qs = (Match.objects
          .filter(
              Q(home_id=home_id, away_id=away_id) | Q(home_id=away_id, away_id=home_id),
              status__in=COMPLETED_STATUSES,
              kickoff_utc__lt=upto_dt
          )
          .select_related("home", "away", "league")
          .order_by("-kickoff_utc")[:n])
    out = []
    for m in qs:
        out.append({
            "date": _aware_utc(m.kickoff_utc).date().isoformat(),
            "league": _league_name(m.league),
            "home": m.home.name, "away": m.away.name,
            "home_logo": _team_logo(m.home), "away_logo": _team_logo(m.away),
            "score": f"{m.goals_home}–{m.goals_away}" if m.goals_home is not None and m.goals_away is not None else "-",
        })
    return out


def _latest_lineup_for(team_id: int, upto_dt):
    lu = (Lineup.objects
          .select_related("match", "team")
          .filter(team_id=team_id, match__kickoff_utc__lte=upto_dt)
          .order_by("-match__kickoff_utc")
          .first())
    if not lu:
        return None
    return {
        "team_id": lu.team_id,
        "team": lu.team.name,
        "formation": lu.formation or "",
        "starters": lu.starters_json or [],
        "bench": lu.bench_json or [],
        "as_of": _aware_utc(lu.match.kickoff_utc).strftime("%b %d, %H:%M UTC"),
        "xi_strength": lu.xi_strength,
        "xi_changes": lu.xi_changes,
    }


def _normalize_standings_rows(data):
    """
    Return a list of rows with keys:
      team_id, team_name, pos, pts, played, gd, form
    Handles common API-Football shape and simple tables.
    """
    rows = []

    if not data:
        return rows

    # API-Football shape
    try:
        blocks = data.get("response", [])[0]["league"]["standings"][0]
        for r in blocks:
            team = r.get("team", {})
            all_stats = r.get("all", {})
            rows.append({
                "team_id": team.get("id"),
                "team_name": team.get("name", ""),
                "pos": r.get("rank"),
                "pts": r.get("points"),
                "played": all_stats.get("played"),
                "gd": r.get("goalsDiff"),
                "form": r.get("form", ""),
            })
        if rows:
            return rows
    except Exception:
        pass

    # Flat list of dicts?
    if isinstance(data, list):
        for r in data:
            rows.append({
                "team_id": r.get("team_id") or r.get("team", {}).get("id"),
                "team_name": r.get("team_name") or r.get("team", {}).get("name", ""),
                "pos": r.get("pos") or r.get("rank"),
                "pts": r.get("pts") or r.get("points"),
                "played": r.get("played") or r.get("games") or r.get("mp"),
                "gd": r.get("gd") or r.get("goal_diff") or r.get("goalsDiff"),
                "form": r.get("form", ""),
            })
        return rows

    # Dict with 'table' key?
    if isinstance(data, dict) and "table" in data and isinstance(data["table"], list):
        for r in data["table"]:
            rows.append({
                "team_id": r.get("team_id") or r.get("team", {}).get("id"),
                "team_name": r.get("team_name") or r.get("team", {}).get("name", ""),
                "pos": r.get("pos") or r.get("rank"),
                "pts": r.get("pts") or r.get("points"),
                "played": r.get("played") or r.get("games") or r.get("mp"),
                "gd": r.get("gd") or r.get("goal_diff") or r.get("goalsDiff"),
                "form": r.get("form", ""),
            })
        return rows

    return rows



from django.http import Http404
from django.shortcuts import get_object_or_404, render


# views.py
from django.shortcuts import render, get_object_or_404
from django.http import Http404
from django.db.models import Q
from django.utils import timezone
from datetime import timedelta

from .models import Match, PredictedMarket, Lineup, MatchStats, StandingsRow



# views.py
from datetime import timezone as dt_timezone
from django.shortcuts import render, get_object_or_404
from django.db.models import Q
from django.utils import timezone

from .models import Match, Lineup, PredictedMarket

# views.py
from datetime import timezone as dt_timezone
from django.shortcuts import render, get_object_or_404
from django.db.models import Q
from django.utils import timezone

from .models import Match, Lineup, PredictedMarket

# ---------- helpers ----------

def _utc_label(dt):
    try:
        return dt.astimezone(dt_timezone.utc).strftime("%b %d, %H:%M UTC")
    except Exception:
        return dt.isoformat()

def _flatten_player(entry):
    if isinstance(entry, str):
        return {"name": entry, "number": None, "pos": None}
    if not isinstance(entry, dict):
        return {"name": "Player", "number": None, "pos": None}

    name = entry.get("name") or entry.get("player_name") or entry.get("fullname")
    number = entry.get("number") or entry.get("shirt") or entry.get("shirtNumber") or entry.get("jersey")
    pos = entry.get("pos") or entry.get("position") or entry.get("grid") or entry.get("role")

    if not name and isinstance(entry.get("player"), dict):
        p = entry["player"]
        name = p.get("name") or p.get("fullname") or p.get("display") or name
        if number is None: number = p.get("number")
        if pos is None: pos = p.get("pos") or p.get("position")
    return {"name": name or "Player", "number": number, "pos": pos}

def _extract_players(payload):
    if not payload:
        return []
    seq = payload
    if isinstance(payload, dict):
        for k in ("startXI", "starters", "players", "xi", "list"):
            if isinstance(payload.get(k), list):
                seq = payload[k]
                break
        else:
            for v in payload.values():
                if isinstance(v, list):
                    seq = v
                    break
    if not isinstance(seq, list):
        return []
    return [_flatten_player(x) for x in seq]

def _best_lineup_for_team(match_obj, team_id):
    lu = Lineup.objects.filter(match=match_obj, team_id=team_id).first()
    if lu:
        return lu, {"kind": "this_match", "match_id": match_obj.id, "label": "This fixture"}
    fallback = (Lineup.objects
        .select_related("match")
        .filter(
            team_id=team_id,
            match__league_id=match_obj.league_id,
            match__season=match_obj.season,
            match__kickoff_utc__lt=match_obj.kickoff_utc,
        )
        .order_by("-match__kickoff_utc")
        .first())
    if fallback:
        opp = (fallback.match.away.name if fallback.match.home_id == team_id else fallback.match.home.name)
        return fallback, {
            "kind": "previous_match",
            "match_id": fallback.match_id,
            "label": f"From {fallback.match.kickoff_utc.date().isoformat()} vs {opp}",
        }
    return None, {"kind": "none", "match_id": None, "label": "No lineup available"}



from django.db.models.functions import TruncDate
from django.db.models import Q

# --- find recent completed matchdays (dates) for this league/season ---
def _recent_completed_matchdays(league_id, season, before_dt, max_days=10):
    """
    Returns a list of distinct UTC dates (most-recent first) on which there were
    completed matches (FT/AET/PEN) in the same league/season, strictly before `before_dt`.
    """
    qs = (
        Match.objects
        .filter(
            league_id=league_id,
            season=season,
            status__in=["FT", "AET", "PEN"],
            kickoff_utc__lt=before_dt,
        )
        .annotate(md=TruncDate("kickoff_utc"))
        .values_list("md", flat=True)
        .order_by("-md")
        .distinct()
    )
    # Limit to a few days so we don't scan the whole season
    return list(qs[:max_days])


def _load_standings_rolling_back(league_id, season, ref_dt, max_backdays=10):
    """
    Try standings on the most recent completed matchday dates before ref_dt,
    stopping at the first date that yields parsable rows.
    """
    tried_dates = []
    for md in _recent_completed_matchdays(league_id, season, ref_dt, max_days=max_backdays):
        rows, asof, meta = _load_standings_rows(league_id, season, md)
        tried_dates.append(str(md))
        if rows:
            meta = dict(meta or {})
            meta["source"] = "recent_matchday"
            meta["used_ref_date"] = str(md)
            meta["tried_dates"] = tried_dates
            return rows, asof, meta

    return [], None, {"reason": "no_rows_on_recent_matchdays", "tried_dates": tried_dates}



def _norm_row(r):
    """Normalize one standings row dict into a flat structure."""
    if not isinstance(r, dict):
        return None
    team_id = r.get("team_id") or (r.get("team") or {}).get("id") or r.get("id")
    team = r.get("team_name") or (r.get("team") or {}).get("name") or r.get("name")
    rank = r.get("rank") or r.get("position") or r.get("pos")
    all_blk = r.get("all") or {}
    goals_blk = all_blk.get("goals") or {}
    played = r.get("played") or all_blk.get("played") or r.get("mp") or r.get("P")
    win = r.get("win") or all_blk.get("win") or r.get("W")
    draw = r.get("draw") or all_blk.get("draw") or r.get("D")
    lose = r.get("lose") or all_blk.get("lose") or r.get("L")
    gf = r.get("gf") or goals_blk.get("for") or r.get("GF")
    ga = r.get("ga") or goals_blk.get("against") or r.get("GA")
    gd = r.get("gd")
    if gd is None and isinstance(gf, (int, float)) and isinstance(ga, (int, float)):
        gd = gf - ga
    pts = r.get("points") or r.get("pts") or r.get("Pts")
    form = r.get("form") or r.get("recent_form")  # some providers have it
    return {
        "team_id": team_id, "team": team, "rank": rank,
        "played": played, "win": win, "draw": draw, "lose": lose,
        "gf": gf, "ga": ga, "gd": gd, "points": pts, "form": form,
    }

# --- STANDINGS helpers (replace your _dig_standings_list and _load_standings_rows) ---
from typing import Any

from typing import Any

WANT_KEYS = ("team", "team_id", "rank", "position", "points", "pts")

def _looks_like_rows(lst: list, want_keys=WANT_KEYS) -> bool:
    """
    True if `lst` is (or contains) dicts with at least one indicative key.
    We sample up to a few elements to be robust.
    """
    if not isinstance(lst, list) or not lst:
        return False

    # If it's a list of dicts, union keys over first few rows
    sample = [x for x in lst if isinstance(x, dict)][:5]
    if sample:
        keyset = set()
        for d in sample:
            for k in getattr(d, "keys", lambda: [])():
                try:
                    keyset.add(str(k).lower())
                except Exception:
                    pass
        if keyset & set(want_keys):
            return True

    # If it's a list of lists (like [[{...}, {...}], ...]) check inner lists
    inner_lists = [x for x in lst if isinstance(x, list)]
    for inner in inner_lists:
        if _looks_like_rows(inner, want_keys):
            return True

    return False


def _first_rows_from_list(lst: list, want_keys=WANT_KEYS):
    """
    Given any list, return the first sub-list that looks like rows.
    Handles list[dict] and list[list[dict]].
    """
    if not isinstance(lst, list) or not lst:
        return None

    # Case A: the list itself is rows
    if _looks_like_rows(lst, want_keys):
        # If it's list[list[dict]], return the first inner rows list
        if isinstance(lst[0], list):
            for inner in lst:
                if _looks_like_rows(inner, want_keys):
                    return inner
        return lst

    # Case B: search inner lists
    for item in lst:
        if isinstance(item, list) and _looks_like_rows(item, want_keys):
            return item

    return None


def _find_list_of_dicts(obj: Any, want_keys=WANT_KEYS):
    """
    Recursively scan obj to find the first list[dict] that looks like standings rows.
    More tolerant than the original (samples multiple items, handles list-of-lists).
    """
    # Direct list cases first
    if isinstance(obj, list):
        hit = _first_rows_from_list(obj, want_keys)
        if hit is not None:
            return hit
        # Recurse into elements (if previous didn’t find)
        for item in obj:
            found = _find_list_of_dicts(item, want_keys)
            if found is not None:
                return found
        return None

    # Dict: recurse into values
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_list_of_dicts(v, want_keys)
            if found is not None:
                return found
        return None

    # Scalars
    return None


def _dig_standings_list(payload):
    """
    Try known shapes quickly, then fall back to a robust recursive finder.
    Returns a list[dict] (or [] if nothing usable).
    """
    if payload is None:
        return []

    # API-Football common shapes
    try:
        if isinstance(payload, dict) and isinstance(payload.get("response"), list) and payload["response"]:
            first = payload["response"][0]
            if isinstance(first, dict):
                lg = first.get("league")
                if isinstance(lg, dict) and "standings" in lg:
                    st = lg["standings"]
                    # standings can be [[{...}]] or [{...}]
                    if isinstance(st, list):
                        rows = _first_rows_from_list(st)
                        if rows:
                            return rows
    except Exception:
        pass

    # league.standings / league.table
    try:
        if isinstance(payload, dict) and isinstance(payload.get("league"), dict):
            st = payload["league"].get("standings") or payload["league"].get("table")
            if isinstance(st, list):
                rows = _first_rows_from_list(st)
                if rows:
                    return rows
    except Exception:
        pass

    # Top-level common keys
    if isinstance(payload, dict):
        for k in ("standings", "table", "rows", "data", "league_table"):
            st = payload.get(k)
            if isinstance(st, list):
                rows = _first_rows_from_list(st)
                if rows:
                    return rows

    # Already a list? Make sure it actually looks like rows.
    if isinstance(payload, list):
        rows = _first_rows_from_list(payload)
        if rows:
            return rows

    # Last resort: recursive scan anywhere in the payload
    found = _find_list_of_dicts(payload)
    return found or []

def _norm_row(r):
    """(unchanged) Normalize one standings row dict into a flat structure."""
    if not isinstance(r, dict):
        return None
    team_id = r.get("team_id") or (r.get("team") or {}).get("id") or r.get("id")
    team = r.get("team_name") or (r.get("team") or {}).get("name") or r.get("name")
    rank = r.get("rank") or r.get("position") or r.get("pos")
    all_blk = r.get("all") or {}
    goals_blk = all_blk.get("goals") or {}
    played = r.get("played") or all_blk.get("played") or r.get("mp") or r.get("P")
    win = r.get("win") or all_blk.get("win") or r.get("W")
    draw = r.get("draw") or all_blk.get("draw") or r.get("D")
    lose = r.get("lose") or all_blk.get("lose") or r.get("L")
    gf = r.get("gf") or goals_blk.get("for") or r.get("GF")
    ga = r.get("ga") or goals_blk.get("against") or r.get("GA")
    gd = r.get("gd")
    if gd is None and isinstance(gf, (int, float)) and isinstance(ga, (int, float)):
        gd = gf - ga
    pts = r.get("points") or r.get("pts") or r.get("Pts")
    form = r.get("form") or r.get("recent_form")
    return {
        "team_id": team_id, "team": team, "rank": rank,
        "played": played, "win": win, "draw": draw, "lose": lose,
        "gf": gf, "ga": ga, "gd": gd, "points": pts, "form": form,
    }


def _standings_exist_info(league_id, season):
    qs = StandingsSnapshot.objects.filter(league_id=league_id, season=season).order_by("-as_of_date")
    latest = qs.first()
    return {
        "league_id": league_id,
        "season": season,
        "any_standings_exist": qs.exists(),
        "total_standings_snapshots": qs.count(),
        "latest_standings_date": latest.as_of_date if latest else None,
        "latest_standings_exists": latest is not None,
        "all_standings_dates": [s.as_of_date for s in qs] if qs.exists() else [],
    }


def _load_standings_rows(league_id, season, as_of_date):
    """
    Fallback order:
      1) season, on/before as_of_date
      2) season, latest
      3) league (any season), latest
    Returns: (rows, as_of_date, meta)
    """
    base = StandingsSnapshot.objects.filter(league_id=league_id)
    tried = []

    def try_qs(qs, source_label):
        snap = qs.first()
        if not snap or snap.id in tried:
            return None
        tried.append(snap.id)
        rows_raw = _dig_standings_list(snap.json)
        rows = []
        for r in rows_raw or []:
            nr = _norm_row(r)
            if nr:
                rows.append(nr)
        if rows:
            return rows, snap.as_of_date, {"reason": "ok", "source": source_label, "rows": len(rows)}
        # return empty but with meta (we may try another source)
        return [], snap.as_of_date, {"reason": "no_rows_from_parser", "source": source_label}

    # 1) season + on/before
    out = try_qs(
        base.filter(season=season, as_of_date__lte=as_of_date).order_by("-as_of_date"),
        "season_on_or_before"
    )
    if out and out[0]:
        return out

    # 2) season latest
    out2 = try_qs(
        base.filter(season=season).order_by("-as_of_date"),
        "season_latest"
    )
    if out2 and out2[0]:
        return out2

    # 3) league latest (any season)
    out3 = try_qs(
        base.order_by("-as_of_date"),
        "league_latest_any_season"
    )
    if out3 and out3[0]:
        return out3

    # Nothing parsable; still return best meta we have
    meta = {"reason": "no_snapshot_or_unparsable", "tried": len(tried)}
    # best as_of for message
    best_snap = base.order_by("-as_of_date").first()
    best_date = best_snap.as_of_date if best_snap else None
    return [], best_date, meta


def _team_form(team_id: int, league_id: int, season: int, cutoff_dt, n=5):
    """
    Last n finished matches for team in same league/season up to cutoff.
    """
    qs = (Match.objects
          .select_related("home", "away")
          .filter(
              Q(league_id=league_id),
              Q(season=season),
              Q(kickoff_utc__lt=cutoff_dt),
              Q(status__in=["FT", "AET", "PEN"]),
              Q(home_id=team_id) | Q(away_id=team_id),
          )
          .order_by("-kickoff_utc")[:n])

    items = []
    for m in qs:
        res = _result_for_team(m, team_id)
        items.append({
            "id": m.id,
            "opp": (m.away.name if m.home_id == team_id else m.home.name),
            "venue": ("H" if m.home_id == team_id else "A"),
            "score": f"{m.goals_home or 0}-{m.goals_away or 0}",
            "date": m.kickoff_utc.date().isoformat(),
            "result": res,
        })
    summary = "".join([x["result"] for x in items])  # e.g., "WDWDL"
    return {"last": items, "summary": summary}

def _h2h(home_id: int, away_id: int, cutoff_dt, limit=10):
    """
    Recent head-to-head between these clubs in any venue within same league (soft filter).
    """
    qs = (Match.objects
          .select_related("home", "away")
          .filter(
              Q(kickoff_utc__lt=cutoff_dt),
              Q(status__in=["FT", "AET", "PEN"]),
              Q(home_id__in=[home_id, away_id]),
              Q(away_id__in=[home_id, away_id]),
          )
          .order_by("-kickoff_utc")[:limit])

    rows = []
    for m in qs:
        if not ((m.home_id in (home_id, away_id)) and (m.away_id in (home_id, away_id))):
            continue
        # from home team's perspective of THIS fixture (home_id argument)
        if m.goals_home is None or m.goals_away is None:
            res = ""
        else:
            if m.goals_home > m.goals_away:
                res = "HOME" if m.home_id == home_id else "AWAY"
            elif m.goals_home < m.goals_away:
                res = "AWAY" if m.away_id == away_id else "HOME"
            else:
                res = "DRAW"
        rows.append({
            "id": m.id,
            "date": m.kickoff_utc.date().isoformat(),
            "home": m.home.name, "away": m.away.name,
            "score": f"{m.goals_home or 0}-{m.goals_away or 0}",
            "result": res,
        })
    return rows
from typing import Any, Dict, List, Optional

def _extract_players(payload: Any) -> List[Dict[str, Optional[str]]]:
    out: List[Dict[str, Optional[str]]] = []
    if not payload:
        return out

    if isinstance(payload, dict):
        for key in ("startXI", "start11", "starters", "players", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break

    if not isinstance(payload, list):
        return out

    for item in payload:
        p = item.get("player") if isinstance(item, dict) else None
        if isinstance(p, dict):
            pid = p.get("id")
            name = p.get("name") or p.get("fullname") or p.get("full_name") or ""
            number = p.get("number") or p.get("shirt") or p.get("shirtNumber")
            pos = p.get("pos") or p.get("position") or item.get("position")
            photo = p.get("photo") or p.get("image") or None
        else:
            pid = (item.get("id") if isinstance(item, dict) else None)
            name = (item.get("name") if isinstance(item, dict) else None) or ""
            number = (item.get("number") if isinstance(item, dict) else None) or item.get("shirt")
            pos = (item.get("pos") if isinstance(item, dict) else None) or item.get("position")
            photo = (item.get("photo") if isinstance(item, dict) else None) or item.get("image")

        # ⬇️ ADD THIS FALLBACK HERE
        if not photo and pid:
            photo = f"https://media-3.api-sports.io/football/players/{pid}.png"

        out.append({
            "id": pid,
            "name": name,
            "number": number,
            "pos": pos,
            "photo": photo,
        })

    return out






# ---------- view ----------
from .models import Match 

# views.py
from datetime import date
from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from matches.standings import _load_standings_rows

from matches.models import Match, PredictedMarket  # adjust if your app labels differ
from matches.models import StandingsRow            # <-- ADJUST to wherever StandingsRow lives

try:
    from subscriptions.service import has_active_subscription
except Exception:
    def has_active_subscription(user): return False


def _utc_label(dt):
    try:
        # dt is UTC already in your schema
        return dt.strftime("%b %d, %Y • %H:%M UTC")
    except Exception:
        return ""


# views.py
from django.shortcuts import render, get_object_or_404
from django.apps import apps
from django.utils import timezone

from matches.models import Match, PredictedMarket  # adjust if these live elsewhere
def _serialize_standings_row(sr):
    return {
        "team_id": sr.team_id,
        "team_name": getattr(sr.team, "name", ""),
        "rank": sr.rank,
        "played": sr.played,
        "win": sr.win,
        "draw": sr.draw,
        "loss": sr.loss,
        "gf": sr.gf,
        "ga": sr.ga,
        "gd": sr.gd,
        "points": sr.points,
        "group_name": sr.group_name or "",
        "form": sr.form or "",
        "last5": sr.last5_json or {},
    }

def _load_standings_rows_for_league(league_id: int, season: int):
    qs = (
        StandingsRow.objects.filter(league_id=league_id, season=season)
        .select_related("team")
        .order_by("group_name", "rank", "-points", "team__name")
    )
    rows = [_serialize_standings_row(r) for r in qs]
    season_used = season

    if not rows:
        latest_season = (
            StandingsRow.objects.filter(league_id=league_id)
            .order_by("-season")
            .values_list("season", flat=True)
            .first()
        )
        if latest_season and latest_season != season:
            qs = (
                StandingsRow.objects.filter(league_id=league_id, season=latest_season)
                .select_related("team")
                .order_by("group_name", "rank", "-points", "team__name")
            )
            rows = [_serialize_standings_row(r) for r in qs]
            season_used = latest_season

    return rows, {
        "season_requested": season,
        "season_used": season_used,
        "rows_count": len(rows),
    }

def _group_standings(rows):
    groups = {}
    for r in rows:
        key = r["group_name"] or "Table"
        groups.setdefault(key, []).append(r)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: (x["rank"], -x["points"]))
    return [{"name": name, "rows": rows} for name, rows in groups.items()]


from django.db.models import Q
from matches.models import Venue, Transfer, Team  # make sure these are imported


def _serialize_venue_row(v: Venue) -> dict:
    return {
        "id": v.id,
        "name": v.name or "",
        "city": v.city or "",
        "country": v.country or "",
        "capacity": v.capacity,
        "surface": v.surface or "",
        "address": v.address or "",
        "image_url": getattr(v, "image_url", None),
    }


def _venue_for_match(match_obj) -> dict:
    """
    Try (1) Match.venue_id if your model has it,
        (2) raw_result_json['fixture']['venue']['id'] as a fallback.
    """
    vid = getattr(match_obj, "venue_id", None)
    if not vid:
        fx = (match_obj.raw_result_json or {}).get("fixture") or {}
        vid = (fx.get("venue") or {}).get("id")

    vrow = Venue.objects.filter(pk=vid).first() if vid else None
    return {
        "exists": bool(vrow),
        "data": _serialize_venue_row(vrow) if vrow else None,
    }


def _serialize_transfer_row(tr):
    # Pick the best label we can find without assuming fields exist
    label = (
        getattr(tr, "fee_text", None) or
        getattr(tr, "fee", None) or
        getattr(tr, "reason", None) or
        getattr(tr, "type", "")  # usually present
    )

    return {
        "date": tr.date,
        "season": getattr(tr, "season", None),
        "type": label,
        "player_id": getattr(tr, "player_id", None),
        "player_name": getattr(getattr(tr, "player", None), "name", "")
                        or getattr(tr, "player_name", "") or "",
        "from_team_id": getattr(tr, "from_team_id", None),
        "from_team_name": getattr(getattr(tr, "from_team", None), "name", "")
                           or getattr(tr, "from_team_name", "") or "",
        "to_team_id": getattr(tr, "to_team_id", None),
        "to_team_name": getattr(getattr(tr, "to_team", None), "name", "")
                         or getattr(tr, "to_team_name", "") or "",
    }


def _recent_transfers_for_team(team_id: int, limit: int = 10) -> list[dict]:
    """
    Show latest moves either into or out of the team.
    """
    qs = (
        Transfer.objects
        .filter(Q(to_team_id=team_id) | Q(from_team_id=team_id))
        .order_by("-date", "-id")[:limit]
    )
    return [_serialize_transfer_row(t) for t in qs]

# --- Injuries & Trophies helpers (ADD) ---
from django.db.models import Q
from matches.models import Injury, Trophy, PlayerSeason, Player

def _serialize_injury(row: Injury) -> dict:
    return {
        "player_id": row.player_id,
        "player_name": getattr(getattr(row, "player", None), "name", "") or "",  # safe if FK exists
        "type": row.type or "",
        "reason": row.reason or "",
        "date": row.date,  # can be None
    }

def _recent_injuries_for_team(team_id: int, limit: int = 10) -> list[dict]:
    qs = (Injury.objects
          .filter(team_id=team_id)
          .order_by("-date", "-id")[:limit])
    return [_serialize_injury(x) for x in qs]

def _serialize_trophy(tr: Trophy, names: dict[int, str]) -> dict:
    return {
        "player_id": tr.player_id,
        "player_name": names.get(tr.player_id, ""),
        "league": tr.league or "",
        "season": tr.season or "",
        "place": tr.place,          # 1/2/3/etc.
        "country": tr.country or "",
    }

def _team_trophies(team_id: int, season: int, limit: int = 12) -> list[dict]:
    # players who belong to this team in this season
    pids = list(PlayerSeason.objects
                .filter(team_id=team_id, season=season)
                .values_list("player_id", flat=True))
    if not pids:
        return []
    names = dict(Player.objects.filter(id__in=pids).values_list("id", "name"))
    qs = (Trophy.objects
          .filter(player_id__in=pids)
          .order_by("-season", "-id")[:limit])
    return [_serialize_trophy(t, names) for t in qs]

# ----------------- THE VIEW -----------------
# views.py
from django.shortcuts import render, get_object_or_404
from matches.models import Match, StandingsRow  # or use apps.get_model if you prefer

def match_detail(request, pk=None, match_id=None, **kwargs):
    match_pk = pk or match_id or kwargs.get("id")

    match_obj = get_object_or_404(
        Match.objects.select_related("home", "away", "league"),
        pk=match_pk,
    )

    # predictions (unchanged)
    pm = (
        PredictedMarket.objects.filter(match_id=match_obj.id)
        .order_by("market_code", "specifier", "-p_model")
    )
    markets = [
        {
            "market": x.market_code,
            "specifier": x.specifier or "",
            "p": x.p_model,
            "fair_odds": x.fair_odds,
            "book_odds": x.book_odds,
            "edge": x.edge,
        }
        for x in pm
    ]

    # lineups (unchanged)
    home_lu, home_src = _best_lineup_for_team(match_obj, match_obj.home_id)
    away_lu, away_src = _best_lineup_for_team(match_obj, match_obj.away_id)

    # ✅ standings (use THIS match's league & season)
    st_rows, st_meta = _load_standings_rows_for_league(match_obj.league_id, match_obj.season)
    rows_by_team = {r["team_id"]: r for r in st_rows}

    standings_ctx = {
        "exists": bool(st_rows),
        "season_used": st_meta["season_used"],
        "groups": _group_standings(st_rows),
        "rows_count": st_meta["rows_count"],
        # handy lookups/highlights
        "home_row": rows_by_team.get(match_obj.home_id),
        "away_row": rows_by_team.get(match_obj.away_id),
        "highlight_ids": {match_obj.home_id, match_obj.away_id},
    }
    venue_ctx = _venue_for_match(match_obj)

    # Transfers (latest 10 per team; tweak limit if you like)
    home_transfers = _recent_transfers_for_team(match_obj.home_id, limit=10)
    away_transfers = _recent_transfers_for_team(match_obj.away_id, limit=10)
   

    # injuries
    home_injuries = _recent_injuries_for_team(match_obj.home_id, limit=8)
    away_injuries = _recent_injuries_for_team(match_obj.away_id, limit=8)
    print(home_injuries)
    # trophies (for players registered with each team this season)
    home_trophies = _team_trophies(match_obj.home_id, match_obj.season, limit=12)
    away_trophies = _team_trophies(match_obj.away_id, match_obj.season, limit=12)


    # forms (unchanged)
    form_home = _team_form(match_obj.home_id, match_obj.league_id, match_obj.season, match_obj.kickoff_utc, n=5)
    form_away = _team_form(match_obj.away_id, match_obj.league_id, match_obj.season, match_obj.kickoff_utc, n=5)

    # h2h (unchanged)
    h2h_rows = _h2h(match_obj.home_id, match_obj.away_id, match_obj.kickoff_utc, limit=10)

    context = {
        "match": {
            "id": match_obj.id,
            "league_id": match_obj.league_id,
            "league_name": getattr(match_obj.league, "name", getattr(match_obj.league, "league_name", "")),
            "season": match_obj.season,
            "home_id": match_obj.home_id,
            "away_id": match_obj.away_id,
            "home": match_obj.home.name,
            "away": match_obj.away.name,
            "home_logo": getattr(match_obj.home, "logo_url", ""),
            "away_logo": getattr(match_obj.away, "logo_url", ""),
            "kickoff_ts": match_obj.kickoff_utc,
            "kickoff_label": _utc_label(match_obj.kickoff_utc),
            "status": match_obj.status,
            "goals_home": match_obj.goals_home,
            "goals_away": match_obj.goals_away,
        },
        "markets": markets,
        "home_lineup": {
            "exists": bool(home_lu),
            "formation": getattr(home_lu, "formation", "") if home_lu else "",
            "starters": _extract_players(getattr(home_lu, "starters_json", None)) if home_lu else [],
            "bench": _extract_players(getattr(home_lu, "bench_json", None)) if home_lu else [],
            "source": home_src,
        },
        "away_lineup": {
            "exists": bool(away_lu),
            "formation": getattr(away_lu, "formation", "") if away_lu else "",
            "starters": _extract_players(getattr(away_lu, "starters_json", None)) if away_lu else [],
            "bench": _extract_players(getattr(away_lu, "bench_json", None)) if away_lu else [],
            "source": away_src,
        },
        "standings": standings_ctx,
        "form_home": form_home,
        "form_away": form_away,
        "h2h_rows": h2h_rows,
        "rows_by_team": rows_by_team,  # optional quick lookup
        "venue": venue_ctx,
    "transfers": {
        "home": home_transfers,
        "away": away_transfers,
    },
    "injuries_home": home_injuries,
    "injuries_away": away_injuries,
    "trophies_home": home_trophies,
    "trophies_away": away_trophies,
    }
    return render(request, "match_detail.html", context)





# views.py
from django.shortcuts import render, get_object_or_404
from django.db.models import Prefetch
from .models import Match, PredictedMarket, MatchPrediction

def match_deail(request, match_id: int):
    """
    Match details page. Shows all predicted markets for this match + core info.
    """
    # Pull every predicted market for this match, ordered nicely
    pm_qs = (PredictedMarket.objects
             .filter(match_id=match_id)
             .order_by("market_code", "specifier"))

    match_obj = (Match.objects
                 .select_related("home", "away", "league")
                 .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs))
                 .filter(id=match_id)
                 .first())

    if not match_obj:
        return render(request, "match_detail.html", {
            "match": None,
            "markets_by_group": [],
            "prediction": None,
            "title": "Match not found",
            "note": "We couldn’t find this match. (Check that your table is linking with the correct match_id.)",
        }, status=404)

    # If you store Poisson/λ predictions per match, pull them too
    prediction = MatchPrediction.objects.filter(match_id=match_id).first()

    # Normalize markets → group by market_code
    markets = []
    for pm in match_obj.predicted_markets.all():
        # probability
        p = pm.p_model
        try:
            p = float(p) if p is not None else None
        except:
            p = None
        # fair odds
        fair = (1.0/p) if (p not in (None, 0.0)) else None

        markets.append({
            "market":     (pm.market_code or "").upper(),
            "specifier":  pm.specifier or "—",
            "p":          p,
            "p_pct":      f"{round(p*1000)/10:.1f}%" if p is not None else "—",
            "fair_odds":  f"{fair:.2f}" if fair else "—",
        })

    # group for template
    from collections import defaultdict
    groups = defaultdict(list)
    for row in markets:
        groups[row["market"]].append(row)
    markets_by_group = sorted(
        [{"name": k, "rows": v} for k, v in groups.items()],
        key=lambda g: g["name"]
    )

    title = f"{getattr(match_obj.home, 'name', 'Home')} vs {getattr(match_obj.away, 'name', 'Away')}"
    return render(request, "match_detail.html", {
        "match": match_obj,
        "markets_by_group": markets_by_group,
        "prediction": prediction,   # contains lambdas if you have them
        "title": title,
        "note": "",
    })





# matches/views.py
import csv
from io import StringIO
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.utils.dateparse import parse_date

from .models import DailyTicket

TWO_ODDS_LEAGUE_ID = -1  # keep consistent with utils.py

def download_daily_2odds_ticket(request):
    """
    GET /tickets/daily-2odds/download?date=YYYY-MM-DD[&format=csv|json]
    - default format: csv
    - never 404s; returns a small 'no ticket' file when empty
    """
    fmt = (request.GET.get("format") or "csv").lower()
    d_str = request.GET.get("date")
    d_obj = parse_date(d_str) if d_str else None
    ticket_date = d_obj or timezone.now().date()

    dt = (
        DailyTicket.objects
        .filter(ticket_date=ticket_date, league_id=TWO_ODDS_LEAGUE_ID)
        .order_by("-id")
        .first()
    )

    # JSON branch — always return a valid file, even if empty
    if fmt == "json":
        payload = {
            "date": ticket_date.isoformat(),
            "legs": (dt.legs if dt else 0),
            "status": (dt.status if dt else "empty"),
            "acc_probability": (dt.acc_probability if dt else None),
            "acc_fair_odds": (dt.acc_fair_odds if dt else None),
            "acc_bookish_odds": (dt.acc_bookish_odds if dt else None),
            "selections": (dt.selections if dt and dt.selections else []),
            "note": None if dt else "No ticket for this date.",
        }
        resp = JsonResponse(payload, json_dumps_params={"indent": 2})
        resp["Content-Disposition"] = f'attachment; filename="two_odds_{ticket_date}.json"'
        return resp

    # CSV branch — always produce a CSV, even if empty
    fieldnames = [
        "date",
        "leg_index",
        "match_id",
        "league_id",
        "home",
        "away",
        "market",
        "specifier",
        "p",          # 0..1
        "p_pct",      # %
        "fair_odds",
        "kickoff_utc",
        "note",       # extra column for messages
    ]

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    if not dt or not dt.selections:
        writer.writerow({
            "date": ticket_date.isoformat(),
            "leg_index": "NONE",
            "note": "No ticket for this date.",
        })
    else:
        for i, s in enumerate(dt.selections or [], start=1):
            p = s.get("p")
            writer.writerow({
                "date": ticket_date.isoformat(),
                "leg_index": i,
                "match_id": s.get("match_id"),
                "league_id": s.get("league_id"),
                "home": s.get("home"),
                "away": s.get("away"),
                "market": s.get("market"),
                "specifier": s.get("specifier"),
                "p": p if p is not None else "",
                "p_pct": f"{(float(p)*100.0):.1f}" if p is not None else "",
                "fair_odds": s.get("fair_odds"),
                "kickoff_utc": s.get("kickoff_utc"),
                "note": "",
            })

        # Summary row
        writer.writerow({
            "date": ticket_date.isoformat(),
            "leg_index": f"TOTAL({dt.legs})",
            "p": dt.acc_probability if dt.acc_probability is not None else "",
            "p_pct": f"{(dt.acc_probability*100.0):.2f}" if dt.acc_probability is not None else "",
            "fair_odds": dt.acc_fair_odds if dt.acc_fair_odds is not None else "",
            "note": "",
        })

    csv_data = buf.getvalue()
    resp = HttpResponse(csv_data, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="two_odds_{ticket_date}.csv"'
    return resp





# matches/views.py (add)
from datetime import timedelta
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Max

from .models import DailyTicket

TWO_ODDS_LEAGUE_ID = -1  # keep consistent with utils/generator

def rollover_rail_json(request):
    """
    GET /api/rollover/rail?days=30[&settle=1]
    Returns the most recent ticket per day (for the last `days`) as:
      { items: [{dateISO, odds, status}, ...] }
    status ∈ {win, loss, pending, void}
    """
    days = max(1, int(request.GET.get("days", 30)))
    today = timezone.now().date()
    start = today - timedelta(days=days - 1)

    # Pick latest ticket by id for each date
    latest_ids = (
        DailyTicket.objects
        .filter(league_id=TWO_ODDS_LEAGUE_ID, ticket_date__gte=start)
        .values("ticket_date")
        .annotate(last_id=Max("id"))
        .values_list("last_id", flat=True)
    )

    qs = (
        DailyTicket.objects
        .filter(id__in=list(latest_ids))
        .order_by("-ticket_date")  # newest first
    )

    def _norm_status(s: str) -> str:
        s = (s or "").lower()
        if s in {"won", "win"}: return "win"
        if s in {"lost", "loss", "lose"}: return "loss"
        if s in {"void"}: return "void"
        return "pending"

    items = []
    for dt in qs:
        items.append({
            "dateISO": dt.ticket_date.isoformat(),
            "odds": round(float(dt.acc_bookish_odds or 0.0), 2),
            "status": _norm_status(dt.status),
        })

    return JsonResponse({"items": items}, json_dumps_params={"indent": 2})









# matches/views.py
from django.template.response import TemplateResponse
from django.views.decorators.http import require_GET, require_POST
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt  # (optional; prefer using CSRF header)

from .utils import (
    build_two_odds_ticket_preview,
    get_or_create_daily_two_odds_ticket,
    TWO_ODDS_LEAGUE_ID,
)
from .models import DailyTicket, Match

def _parse_float(qd, key, default):
    try:
        return float(qd.get(key, default))
    except Exception:
        return default

def _parse_int(qd, key, default):
    try:
        return int(qd.get(key, default))
    except Exception:
        return default

def _parse_date_or_today(s):
    d = parse_date(s) if s else None
    return d or timezone.now().date()

def _enrich_selection(sel: dict) -> dict:
    from datetime import timezone as dt_timezone
    from django.utils import timezone as dj_tz
    from django.utils.dateparse import parse_datetime

    m = Match.objects.select_related("home", "away").filter(id=sel.get("match_id")).first()
    if m:
        sel["home"] = getattr(m.home, "name", sel.get("home", ""))
        sel["away"] = getattr(m.away, "name", sel.get("away", ""))
        sel["home_logo"] = getattr(m.home, "logo_url", "")
        sel["away_logo"] = getattr(m.away, "logo_url", "")
        sel["status"] = (m.status or "")
        sel["goals_home"] = getattr(m, "goals_home", None)
        sel["goals_away"] = getattr(m, "goals_away", None)

        dt = getattr(m, "kickoff_utc", None) or sel.get("kickoff_utc")
        if isinstance(dt, str):
            dt = parse_datetime(dt)
        if dt:
            if dj_tz.is_naive(dt):
                dt = dj_tz.make_aware(dt, dt_timezone.utc)
            dt_utc = dt.astimezone(dt_timezone.utc)
            sel["kickoff_ts"]   = dt_utc.isoformat()
            sel["kickoff_label"] = dt_utc.strftime("%b %d, %H:%M") + " UTC"
    return sel

# --- Page with form + live preview (HTMX) ---
def two_odds_builder_page(request):
    # initial defaults (you can tweak)
    ctx = {
        "init_date": (_parse_date_or_today(request.GET.get("date"))).isoformat(),
        "init_target_odds": request.GET.get("target_odds", "2.0"),
        "init_over_tol": request.GET.get("over_tolerance", "0.10"),
        "init_min_legs": request.GET.get("min_legs", "2"),
        "init_max_legs": request.GET.get("max_legs", "6"),
        "init_min_p": request.GET.get("min_p", "0.60"),
        "init_max_fair": request.GET.get("max_fair_odds", "1.60"),
        "init_attempts": request.GET.get("attempts", "500"),
    }
    return TemplateResponse(request, "two_odds_builder.html", ctx)

# --- Fragment endpoint HTMX calls to render just the ticket card ---
@require_GET
def two_odds_preview_fragment(request):
    d = _parse_date_or_today(request.GET.get("date"))
    try:
        t = build_two_odds_ticket_preview(
            ticket_date=d,
            target_odds=_parse_float(request.GET, "target_odds", 2.0),
            over_tolerance=_parse_float(request.GET, "over_tolerance", 0.10),
            min_legs=_parse_int(request.GET, "min_legs", 2),
            max_legs=_parse_int(request.GET, "max_legs", 6),
            min_p=_parse_float(request.GET, "min_p", 0.60),
            max_fair_odds=_parse_float(request.GET, "max_fair_odds", 1.60),
            attempts=_parse_int(request.GET, "attempts", 500),
        )
        legs = [_enrich_selection(s.copy()) for s in (t.selections or [])]
        ctx = {
            "brand": "Rollover 2.0",
            "ticket_date": d.isoformat(),
            "has_ticket": True,
            "legs": legs,
            "legs_count": t.legs,
            "acc_bookish_odds": f"{t.acc_bookish_odds:.2f}",
            "acc_fair_odds": f"{t.acc_fair_odds:.2f}",
            "acc_probability_pct": f"{(t.acc_probability*100.0):.2f}",
            "status": "preview",
        }
    except ValueError as e:
        ctx = {
            "brand": "Rollover 2.0",
            "ticket_date": d.isoformat(),
            "has_ticket": False,
            "note": str(e),
            "legs_count": 0,
            "acc_bookish_odds": None,
            "acc_fair_odds": None,
            "acc_probability_pct": None,
            "status": "preview",
        }
    return TemplateResponse(request, "_two_odds_ticket.html", ctx)

# --- Save button uses your existing DB writer, then re-show the saved ticket as JSON (caller can refresh preview) ---
@require_POST
def two_odds_save_api(request):
    data = request.POST
    d = _parse_date_or_today(data.get("date"))
    dt = get_or_create_daily_two_odds_ticket(
        ticket_date=d,
        target_odds=_parse_float(data, "target_odds", 2.0),
        over_tolerance=_parse_float(data, "over_tolerance", 0.10),
        min_legs=_parse_int(data, "min_legs", 2),
        max_legs=_parse_int(data, "max_legs", 6),
        min_p=_parse_float(data, "min_p", 0.60),
        max_fair_odds=_parse_float(data, "max_fair_odds", 1.60),
        attempts=_parse_int(data, "attempts", 500),
        force_regenerate=bool(int(data.get("force", "0"))),
    )
    return JsonResponse({
        "ok": True,
        "date": d.isoformat(),
        "legs": dt.legs,
        "status": dt.status,
        "acc_probability": dt.acc_probability,
        "acc_fair_odds": dt.acc_fair_odds,
        "acc_bookish_odds": dt.acc_bookish_odds,
        "selections": dt.selections,
    }, json_dumps_params={"indent": 2})








from .models import DailyTicket
TWO_ODDS_LEAGUE_ID = -1


def combined_dashboard(request, league_id, days):
    # --- existing JSON calls (unchanged) ---
    match_obj = Match.objects.filter(
        status__in=["FT", "AET", "PEN"]
    ).order_by('-kickoff_utc').first()
    
    venue_ctx = _venue_for_match(match_obj)

    up_resp = upcoming_predictions_json(request, league_id, days)
    upcoming = json.loads(up_resp.content.decode("utf-8"))
    completed_response = completed_today_json(request, 'all')
    completed = json.loads(completed_response.content)
    print(completed)

 
   
    ticket_date = request.GET.get("ticket_date", "")
    if ticket_date:
        rf = RequestFactory()
        req2 = rf.get("/_daily_ticket", {"date": ticket_date})
        ticket_resp = daily_ticket_json(req2)
        selected_ticket_date = ticket_date
    else:
        ticket_resp = daily_ticket_json(request)
        try:
            selected_ticket_date = json.loads(ticket_resp.content.decode("utf-8"))["ticket"]["date"]
        except Exception:
            selected_ticket_date = timezone.now().date().isoformat()

    daily_ticket = json.loads(ticket_resp.content.decode("utf-8"))

    # --- build incoming_matches (server-side) from upcoming JSON ---
    def extract_1x2(markets):
        pH = pD = pA = None
        for m in (markets or []):
            if str(m.get("market", "")).upper() == "1X2":
                s = (m.get("specifier") or "").upper()
                if s == "H": pH = float(m.get("p"))
                elif s == "D": pD = float(m.get("p"))
                elif s == "A": pA = float(m.get("p"))
        if pH is None and pD is None and pA is None:
            for m in (markets or []):
                if str(m.get("market", "")).upper().startswith("1X2"):
                    s = (m.get("specifier") or "").upper()
                    if s == "H" and pH is None: pH = float(m.get("p"))
                    elif s == "D" and pD is None: pD = float(m.get("p"))
                    elif s == "A" and pA is None: pA = float(m.get("p"))
        best = None
        cands = [(k, v) for k, v in (("H", pH), ("D", pD), ("A", pA)) if v is not None]
        if cands:
            best = max(cands, key=lambda kv: kv[1])[0]
        return pH, pD, pA, best

    incoming_matches = []
    for item in upcoming.get("matches", []):
        dt = parse_datetime(item.get("kickoff_utc") or "")
        pH, pD, pA, best = extract_1x2(item.get("markets", []))
        incoming_matches.append({
            "id":         item.get("match_id"),
            "kickoff_dt": dt,
            "home":       item.get("home") or "",
            "away":       item.get("away") or "",
            "home_logo":  item.get("home_logo") or "",
            "away_logo":  item.get("away_logo") or "",
            "status":     item.get("status") or "NS",
            "pH": pH, "pD": pD, "pA": pA, "best": best,
        })

    # --- date filter for incoming (UTC calendar day) ---
    date_qs = request.GET.get("date")
    if date_qs:
        try:
            y, m, d = map(int, date_qs.split("-"))
            target_date = timezone.datetime(y, m, d, tzinfo=timezone.utc).date()
        except Exception:
            target_date = None
        if target_date:
            def same_utc_day(dtobj):
                if not dtobj:
                    return False
                if timezone.is_naive(dtobj):
                    dtobj = timezone.make_aware(dtobj, timezone=timezone.utc)
                return dtobj.astimezone(timezone.utc).date() == target_date
            incoming_matches = [m for m in incoming_matches if same_utc_day(m["kickoff_dt"])]

    # --- make sure logos exist in completed too (safe defaults) ---
    for m in completed.get("matches", []):
        m["home_logo"] = m.get("home_logo") or ""
        m["away_logo"] = m.get("away_logo") or ""

    # --- 7-day calendar rail (links with ?date=YYYY-MM-DD) ---
    today = timezone.now().date()
    selected_incoming_date = date_qs or today.isoformat()
    date_rail = []
    for i in range(7):
        d = today + timedelta(days=i)
        date_rail.append({
            "dow": d.strftime("%a"),
            "day": d.strftime("%d"),
            "url": f"/mega/{league_id}/{days}/?date={d.isoformat()}",
            "active": (d.isoformat() == selected_incoming_date),
        })

    # ================= NEW: 2-odds ticket for the selected date ==============
    def _safe_to_date(iso_s, default_date):
        try:
            y, m, d = map(int, (iso_s or "").split("-"))
            return timezone.datetime(y, m, d, tzinfo=timezone.utc).date()
        except Exception:
            return default_date

    sel_date = _safe_to_date(selected_ticket_date, today)

    base_dt = (
        DailyTicket.objects
        .filter(ticket_date=sel_date, league_id=TWO_ODDS_LEAGUE_ID)
        .order_by("-id")
        .first()
    )

    # Build a compact widget payload for the template
    if base_dt:
        two_odds_widget = {
            "has_ticket": True,
            "date": sel_date.isoformat(),
            "legs": base_dt.legs,
            "status": base_dt.status,  # 'pending' | 'won' | 'lost' | 'void'
            "acc_bookish_odds": base_dt.acc_bookish_odds,
            "acc_fair_odds": base_dt.acc_fair_odds,
            "acc_probability_pct": round((base_dt.acc_probability or 0) * 100, 2) if base_dt.acc_probability is not None else None,
            "view_url": f"/tickets/2odds?date={sel_date.isoformat()}",
            "download_url": f"/tickets/daily-2odds/download?date={sel_date.isoformat()}&format=html",
        }
    else:
        two_odds_widget = {
            "has_ticket": False,
            "date": sel_date.isoformat(),
            "status": "empty",
            "view_url": f"/tickets/2odds?date={sel_date.isoformat()}",
            "download_url": f"/tickets/daily-2odds/download?date={sel_date.isoformat()}&format=html",
        }

    # Short history rail (last 10 days)
    rail_days = 10
    two_odds_rail = []
    for i in range(rail_days):
        d = sel_date - timedelta(days=i)
        dt_i = (
            DailyTicket.objects
            .filter(ticket_date=d, league_id=TWO_ODDS_LEAGUE_ID)
            .order_by("-id")
            .first()
        )
        two_odds_rail.append({
            "dateISO": d.isoformat(),
            "odds": getattr(dt_i, "acc_bookish_odds", None),
            "status": (dt_i.status if dt_i else "empty"),
            "view_url": f"/tickets/2odds?date={d.isoformat()}",
        })
    ctx = {
        "init_date": (_parse_date_or_today(request.GET.get("date"))).isoformat(),
        "init_target_odds": request.GET.get("target_odds", "2.0"),
        "init_over_tol": request.GET.get("over_tolerance", "0.10"),
        "init_min_legs": request.GET.get("min_legs", "2"),
        "init_max_legs": request.GET.get("max_legs", "6"),
        "init_min_p": request.GET.get("min_p", "0.60"),
        "init_max_fair": request.GET.get("max_fair_odds", "1.60"),
        "init_attempts": request.GET.get("attempts", "500"),
    }    
    # ========================================================================

    context = {
        "ctx": ctx,
        "league_id": league_id,
        "days": days,
        "ticket_date": selected_ticket_date,
        "upcoming_json": upcoming,
        "venue":venue_ctx,
        "completed_today_json": completed,
        "daily_ticket_json": daily_ticket,
        "incoming_matches": incoming_matches,
        "date_rail": date_rail,
        "selected_incoming_date": selected_incoming_date,

        # NEW
        "two_odds_widget": two_odds_widget,
        "two_odds_rail": two_odds_rail,
    }
    return render(request, "index.html", context)




def luc(request, league_id):
  
    comp_resp = completed_today_json(request, league_id)
    completed = json.loads(comp_resp.content.decode("utf-8"))
    return render(request, "indexd.html", completed)




def luck(request):
    # Get the JSON data
    completed_response = completed_today_json(request, 'all')
    completed_data = json.loads(completed_response.content.decode('utf-8'))
    print(completed_data)
    context = {
        'completed_today_json': completed_data,  # This key must match your template
    }
    
    return render(request, "indexd.html", context)