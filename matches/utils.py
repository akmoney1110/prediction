# matches/utils.py
from __future__ import annotations
from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from typing import Optional, Tuple
from django.db import transaction
from .models import Match, DailyTicket

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, date as date_cls, timezone

from django.db import transaction

from django.db.models import Prefetch, Q

from .models import Match, PredictedMarket, DailyTicket

# Markets we’re willing to auto-pick from
# --- utils.py: random, no-unders, EXCLUDE TEAM_TOTAL + TEAM_TOTAL_CORNERS ---

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, date as date_cls, timezone as dt_timezone

from django.db import transaction

from .models import PredictedMarket, DailyTicket, Match

# Markets you allow the ticket to pick from (families)
TOP_MARKET_WHITELIST = {
    "1X2", "OU", "BTTS", "OU_CORNERS",
    "CARDS_TOT", "YELLOWS_TOT", "REDS_TOT","1X2_0_15","AWAY_HANDICAP","FIRST_TO_SCORE","GOAL_IN_BOTH_HALVES","HTFT","LAST_TO_SCORE","ODDEVEN",
    # NOTE: TEAM_TOTAL and TEAM_TOTAL_CORNERS intentionally NOT included now
}

# Explicitly excluded families (even if someone adds them back to whitelist)
EXCLUDED_FAMILIES = {"TEAM_TOTAL", "TEAM_TOTAL_CORNERS"}

UPCOMING_STATUSES = {"NS", "TBD", "PST"}

@dataclass
class Pick:
    match_id: int
    league_id: int
    kickoff_utc: datetime
    home: str
    away: str
    market_code: str
    specifier: str
    p_model: float
    fair_odds: float

def _day_bounds_utc(d: date_cls):
    start = datetime(d.year, d.month, d.day, tzinfo=dt_timezone.utc)
    end = start + timedelta(days=1)
    return start, end

def _bookish_odds_for_prob(p: float, overround: float = 0.06) -> float:
    p_eff = max(1e-9, min(1.0 - 1e-9, p)) * (1.0 - overround)
    return float(1.0 / p_eff)

def _market_family(code: str) -> str:
    c = (code or "").upper()
    if c.startswith("1X2"):
        return "1X2"
    return c

def _is_under(code: str, spec: str) -> bool:
    """
    Treat any OU-like market with 'under' in specifier as UNDER.
    (TEAM_TOTAL*_ lines are over-only in your data so they wouldn't match anyway.)
    """
    c = (code or "").upper()
    s = (spec or "").lower()
    if c in {"OU", "OU_CORNERS", "CARDS_TOT", "YELLOWS_TOT", "REDS_TOT"} or c.startswith("OU"):
        return ("_under" in s) or s.endswith("_u") or s.startswith("u")
    return False

def _random_uniform_pick_from_markets(pm_iter, min_p: float, rnd: random.Random) -> PredictedMarket | None:
    """
    Uniform random among candidates:
      - family in whitelist
      - family NOT in excluded
      - NOT an 'under'
      - prefer p >= min_p; if none, fall back to any remaining (still non-under)
    """
    pool = []
    for pm in pm_iter:
        fam = _market_family(pm.market_code)
        if fam not in TOP_MARKET_WHITELIST:
            continue
        if fam in EXCLUDED_FAMILIES:
            continue
        if not pm.p_model:
            continue
        if _is_under(pm.market_code, pm.specifier):
            continue
        pool.append(pm)

    if not pool:
        return None

    qualified = [pm for pm in pool if pm.p_model >= min_p]
    cands = qualified if qualified else pool
    return rnd.choice(cands) if cands else None

def get_or_create_global_daily_ticket(
    ticket_date: date_cls,
    legs: int = 5,
    min_p: float = 0.55,
    force_regenerate: bool = False,
) -> DailyTicket:
    """
    Build one GLOBAL ticket (league_id=0) for the UTC 'ticket_date' by sampling
    across all leagues. Selection is uniformly random among whitelisted, *non-under*
    markets with p >= min_p (falling back to any non-under if none qualify).
    TEAM_TOTAL and TEAM_TOTAL_CORNERS are excluded.
    """

    # Return existing (unless forced)
    dt = DailyTicket.objects.filter(ticket_date=ticket_date, league_id=0).first()
    if dt and not force_regenerate:
        return dt

    start, end = _day_bounds_utc(ticket_date)

    pm_qs = (PredictedMarket.objects
             .filter(kickoff_utc__gte=start, kickoff_utc__lt=end)
             .select_related("match", "match__home", "match__away"))

    markets_by_match: dict[int, list[PredictedMarket]] = {}
    for pm in pm_qs:
        m = pm.match
        if not m:
            continue
        if (m.status or "").upper() not in UPCOMING_STATUSES:
            continue
        markets_by_match.setdefault(m.id, []).append(pm)

    # Deterministic per-day seed (same ticket for the date)
    rnd = random.Random(int(ticket_date.strftime("%Y%m%d")))

    match_ids = list(markets_by_match.keys())
    rnd.shuffle(match_ids)

    picks: list[Pick] = []
    for mid in match_ids:
        if len(picks) >= legs:
            break

        pms = markets_by_match[mid]
        chosen = _random_uniform_pick_from_markets(pms, min_p=min_p, rnd=rnd)
        if not chosen:
            continue

        m = chosen.match
        fair = float(1.0 / max(1e-9, min(1.0 - 1e-9, chosen.p_model)))
        picks.append(Pick(
            match_id=m.id,
            league_id=m.league_id,
            kickoff_utc=m.kickoff_utc,
            home=getattr(m.home, "name", "") or "",
            away=getattr(m.away, "name", "") or "",
            market_code=chosen.market_code,
            specifier=chosen.specifier,
            p_model=float(chosen.p_model),
            fair_odds=fair,
        ))

    # Accumulator metrics
    acc_p = 1.0
    acc_fair = 1.0
    acc_bookish = 1.0
    for pk in picks:
        acc_p *= max(1e-9, min(1.0 - 1e-9, pk.p_model))
        acc_fair *= pk.fair_odds
        acc_bookish *= _bookish_odds_for_prob(pk.p_model, overround=0.06)

    selections = [{
        "match_id": pk.match_id,
        "league_id": pk.league_id,
        "kickoff_utc": pk.kickoff_utc.isoformat(),
        "home": pk.home,
        "away": pk.away,
        "market": pk.market_code,
        "specifier": pk.specifier,
        "p": pk.p_model,
        "fair_odds": pk.fair_odds,
    } for pk in picks]

    with transaction.atomic():
        dt, _created = DailyTicket.objects.update_or_create(
            ticket_date=ticket_date,
            league_id=0,
            defaults={
                "selections": selections,
                "legs": len(selections),
                "acc_probability": float(acc_p if math.isfinite(acc_p) else 0.0),
                "acc_fair_odds": float(acc_fair if math.isfinite(acc_fair) else 0.0),
                "acc_bookish_odds": float(acc_bookish if math.isfinite(acc_bookish) else 0.0),
                "status": "OPEN",
            }
        )

    return dt


# keep your existing TOP_MARKET_WHITELIST and UPCOMING_STATUSES


def _random_unform_pick_from_markets(pm_iter, min_p: float, rnd: random.Random) -> PredictedMarket | None:
    """
    Pure uniform random:
      - filter to whitelist
      - drop all 'unders'
      - pick uniformly among those with p >= min_p
      - if none qualify, pick uniformly among any whitelisted non-under
    """
    pool = [pm for pm in pm_iter
            if _market_family(pm.market_code) in TOP_MARKET_WHITELIST and pm.p_model]

    # exclude unders
    pool = [pm for pm in pool if not _is_under(pm.market_code, pm.specifier)]

    if not pool:
        return None

    qualified = [pm for pm in pool if pm.p_model >= min_p]

    cands = qualified if qualified else pool  # fallback to any allowed non-under
    return rnd.choice(cands) if cands else None




# keepyour existing whitelist
TOP_MAKET_WHITELIST = {
    "1X2", "OU", "BTTS", "TEAM_TOTAL",
    "OU_CORNERS",  # keep
    # "TEAM_TOTAL_CORNERS",   # <- do NOT whitelist this anymore (optional)
    "CARDS_TOT", "YELLOWS_TOT", "REDS_TOT",
}

# add this near the whitelist
EXCLUDED_FAMILIES = {"TEAM_TOTAL_CORNERS","TEAM_TOTAL"}  # hard ban


def _qualify(pm, min_p: float, avoid_under: bool = True) -> bool:
    if not pm or not pm.p_model:
        return False
    fam = _market_family(pm.market_code)

    # hard ban family
    if fam in EXCLUDED_FAMILIES:
        return False

    # must be in whitelist
    if fam not in TOP_MARKET_WHITELIST:
        return False

    # probability floor
    if float(pm.p_model) < float(min_p):
        return False

    # optionally exclude unders
    if avoid_under and _is_under(pm.market_code, pm.specifier):
        return False

    return True

def _random_pick_from_markets(pm_iter, min_p: float, rnd: random.Random,
                              avoid_under: bool = True,
                              family_caps: dict[str, int] | None = None,
                              picked_families: dict[str, int] | None = None):
    pms = list(pm_iter)

    # 1) filter with whitelist + min_p + avoid_under + EXCLUDED_FAMILIES
    cands = [pm for pm in pms if _qualify(pm, min_p, avoid_under=avoid_under)]
    if not cands:
        # fallback: drop min_p but still respect whitelist and exclusions
        cands = [pm for pm in pms if _qualify(pm, 0.0, avoid_under=avoid_under)]
    if not cands:
        # final fallback: any whitelisted (but STILL exclude the banned family)
        any_wl = [
            pm for pm in pms
            if pm.p_model and _market_family(pm.market_code) in TOP_MARKET_WHITELIST
            and _market_family(pm.market_code) not in EXCLUDED_FAMILIES
        ]
        return max(any_wl, key=lambda z: z.p_model, default=None)

    # 2) (optional) apply family caps
    if family_caps and picked_families is not None:
        filtered = []
        for pm in cands:
            fam = _market_family(pm.market_code)
            if picked_families.get(fam, 0) < family_caps.get(fam, 10**9):
                filtered.append(pm)
        if filtered:
            cands = filtered

    # 3) weights (as you already had)
    weights = []
    for pm in cands:
        w = max(1e-6, float(pm.p_model))
        fam = _market_family(pm.market_code)
        if fam == "OU":
            w *= 0.85
            line = _ou_line(pm.specifier) or 0.0
            w *= (1.0 + 0.10 * max(0, int((line - 1.5) // 1)))
        if fam in {"BTTS", "1X2"}:
            w *= 1.15
        if fam.startswith("TEAM_TOTAL"):
            w *= 1.10
        weights.append(w)

    try:
        return rnd.choices(cands, weights=weights, k=1)[0]
    except Exception:
        return rnd.choice(cands)











def _random_pick_frm_markets(pm_iter, min_p: float, rnd: random.Random,
                              avoid_under: bool = True,
                              family_caps: dict[str, int] | None = None,
                              picked_families: dict[str, int] | None = None) -> PredictedMarket | None:
    """
    Weighted-random picker:
      - filters by whitelist, p>=min_p, and (optionally) no-unders
      - diversifies by market family caps (e.g., at most half OU)
      - within OU, upweights higher lines a bit so it’s not always 1.5_over
    """
    pms = [pm for pm in pm_iter]
    # 1) filter
    cands = [pm for pm in pms if _qualify(pm, min_p, avoid_under=avoid_under)]
    if not cands:
        # fallback: ignore min_p but keep whitelist & no-unders
        cands = [pm for pm in pms if _qualify(pm, 0.0, avoid_under=avoid_under)]
    if not cands:
        # final fallback: anything whitelisted (even unders) with largest p
        any_wl = [pm for pm in pms if _market_family(pm.market_code) in TOP_MARKET_WHITELIST and pm.p_model]
        return max(any_wl, key=lambda z: z.p_model, default=None)

    # 2) apply family caps (ticket-level diversity)
    if family_caps is not None and picked_families is not None:
        filtered = []
        for pm in cands:
            fam = _market_family(pm.market_code)
            used = picked_families.get(fam, 0)
            cap = family_caps.get(fam, 10**9)
            if used < cap:
                filtered.append(pm)
        if filtered:
            cands = filtered

    # 3) weights
    weights = []
    for pm in cands:
        w = max(1e-6, float(pm.p_model))  # base weight ~ probability
        fam = _market_family(pm.market_code)

        # So OU doesn’t dominate: slight downweight OU vs others
        if fam == "OU":
            w *= 0.85

            # prefer higher OU lines a bit (2.5/3.5) so we’re not stuck at 1.5
            line = _ou_line(pm.specifier) or 0.0
            # upweight by line: 1.5 → 1.00, 2.5 → 1.10, 3.5 → 1.20, etc.
            w *= (1.0 + 0.10 * max(0, int((line - 1.5) // 1)))  # crude but effective

        # Slight boost to BTTS and 1X2 for variety
        if fam in {"BTTS", "1X2"}:
            w *= 1.15

        # Team totals are good variety too
        if fam.startswith("TEAM_TOTAL"):
            w *= 1.10

        weights.append(w)

    # 4) sample
    try:
        pm = rnd.choices(cands, weights=weights, k=1)[0]
    except Exception:
        pm = rnd.choice(cands)

    return pm

# --- CLEAN TICKET BUILDER (replace your current block) -----------------------
import math, random
from dataclasses import dataclass
from datetime import datetime, timedelta, date as date_cls, timezone as dt_timezone

from django.db import transaction
from .models import PredictedMarket, DailyTicket



def _family(code: str) -> str:
    c = (code or "").upper().strip()
    return "1X2" if c.startswith("1X2") else c




# === Add near your other helpers in matches/utils.py ===
from typing import Optional, Tuple
from .models import Match, MatchStats

FINAL_STATUSES = {"FT", "AET", "PEN"}

def _get_final_int(obj, name: str) -> Optional[int]:
    if not hasattr(obj, name):
        return None
    try:
        v = getattr(obj, name)
        return int(v) if v is not None else None
    except Exception:
        return None

def _ceil_need_for_over(line: float) -> int:
    import math
    # For x.5 lines -> floor+1 (e.g. 2.5 => need 3)
    # For integer lines -> line exactly (we’ll handle push outside)
    return math.floor(float(line)) + 1

def _specifier_has_integer_line(spec: str) -> bool:
    if not spec:
        return False
    s = spec.strip().lower().replace(" ", "")
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return False
    val = float(m.group(1))
    return float(val).is_integer()

def _split_line_side(spec: str) -> Tuple[Optional[float], Optional[str]]:
    # Handles "2.5_over" / "3.0_under" / "home_o1.5" / "away_o2"
    if not spec:
        return None, None
    s = spec.strip().lower()
    # (a) "2.5_over" / "4.5_under"
    if "_" in s and s[0].isdigit():
        try:
            line_s, side = s.split("_", 1)
            return float(line_s), side
        except Exception:
            return None, None
    # (b) "home_o1.5" / "away_o2"
    if "_o" in s:
        try:
            side, line_s = s.split("_o", 1)
            return float(line_s), f"{side}_over"
        except Exception:
            return None, None
    return None, None

def _load_stats_with_fallback(match: Match):
    """
    Returns dict with home/away totals, preferring Match fields; else fall back to MatchStats.
    Keys: goals_home/goals_away, corners_home/_away, cards_home/_away, yellows_home/_away, reds_home/_away
    (yellows/reds often live only in MatchStats)
    """
    out = {
        "goals_home":  _get_final_int(match, "goals_home"),
        "goals_away":  _get_final_int(match, "goals_away"),
        "corners_home": _get_final_int(match, "corners_home"),
        "corners_away": _get_final_int(match, "corners_away"),
        "cards_home":  _get_final_int(match, "cards_home"),
        "cards_away":  _get_final_int(match, "cards_away"),
        "yellows_home": _get_final_int(match, "yellows_home"),
        "yellows_away": _get_final_int(match, "yellows_away"),
        "reds_home":    _get_final_int(match, "reds_home"),
        "reds_away":    _get_final_int(match, "reds_away"),
    }

    # If any of corners/cards/yellows/reds missing, try MatchStats
    need_stats = any(out[k] is None for k in
        ["corners_home","corners_away","cards_home","cards_away","yellows_home","yellows_away","reds_home","reds_away"]
    )
    if need_stats:
        rows = list(MatchStats.objects.filter(match=match).select_related("team"))
        if rows:
            # Map per side
            home_row = next((r for r in rows if getattr(r.team, "id", None) == getattr(match.home, "id", None)), None)
            away_row = next((r for r in rows if getattr(r.team, "id", None) == getattr(match.away, "id", None)), None)

            def put_if_missing(key, v):
                if out[key] is None and v is not None:
                    try: out[key] = int(v)
                    except Exception: pass

            if home_row:
                put_if_missing("corners_home", home_row.corners)
                put_if_missing("cards_home",   home_row.cards)
                put_if_missing("yellows_home", home_row.yellows)
                put_if_missing("reds_home",    home_row.reds)
            if away_row:
                put_if_missing("corners_away", away_row.corners)
                put_if_missing("cards_away",   away_row.cards)
                put_if_missing("yellows_away", away_row.yellows)
                put_if_missing("reds_away",    away_row.reds)

    return out





from datetime import datetime, timezone as dt_timezone
from typing import Optional, Tuple

from django.db import transaction

from .models import Match, DailyTicket

FINAL_STATUSES = {"FT", "AET", "PEN"}

def _get_final_int(obj, name: str) -> Optional[int]:
    """Safely read an integer stat (goals/corners/cards/etc). Return None if absent."""
    if not hasattr(obj, name):
        return None
    try:
        v = getattr(obj, name)
        return int(v) if v is not None else None
    except Exception:
        return None

def _ceil_need_for_over(line: float) -> int:
    """For X.5 lines, over means >= floor(line)+1; generalized ceiling for integer counts."""
    import math
    return math.floor(float(line)) + 1


# utils.py
import re, math

FINAL_STATUSES = {"FT", "AET", "PEN"}

def _get_final_int(obj, name: str):
    if not hasattr(obj, name):
        return None
    try:
        v = getattr(obj, name)
        return int(v) if v is not None else None
    except Exception:
        return None

def _ceil_need_for_over(line: float) -> int:
    # e.g. 1.5 -> 2, 2.5 -> 3
    return math.floor(float(line)) + 1

def _settle_total(total: int, line: float, side: str) -> str:
    """
    Return WIN/LOSE/VOID for a total with possible integer-line push.
    side: "over" or "under"
    """
    if total is None:
        return "VOID"
    side = (side or "").lower().strip()
    # push/VOID on exact integer lines
    if float(line).is_integer() and total == int(line):
        return "VOID"
    need = _ceil_need_for_over(line)
    if side == "over":
        return "WIN" if total >= need else "LOSE"
    if side == "under":
        return "WIN" if total < need else "LOSE"
    return "UNKNOWN"

# -------- specifier parsers (robust to small format drift) --------

def _parse_ou(spec: str):
    """
    Accept: '1.5_over', '2.5_under', 'over_2.5', 'under_3.5'
    Returns (line: float, side: 'over'|'under') or None
    """
    s = (spec or "").strip().lower().replace(" ", "")
    # find over/under anywhere
    side = "over" if "over" in s else ("under" if "under" in s else None)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if side and nums:
        return float(nums[0]), side
    # fallback: exact patterns like '1.5_over'
    try:
        ln, sd = s.split("_", 1)
        if sd in ("over", "under"):
            return float(ln), sd
    except Exception:
        pass
    return None

def _parse_team_total_over(spec: str):
    """
    Accept: 'home_o1.5', 'away_o4.5', 'home_over_1.5', 'over_1.5_home'
    Returns (team: 'home'|'away', line: float) or None
    """
    s = (spec or "").strip().lower().replace(" ", "")

    m = re.match(r"^(home|away)_(?:o|over)_?(\d+(?:\.\d+)?)$", s)
    if m:
        return m.group(1), float(m.group(2))

    m = re.match(r"^(?:o|over)_?(\d+(?:\.\d+)?)[_-]?(home|away)$", s)
    if m:
        return m.group(2), float(m.group(1))

    # very loose catch: look for team and a number
    team = "home" if "home" in s else ("away" if "away" in s else None)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if team and nums:
        return team, float(nums[0])
    return None



# utils.py — robust settlement core (basic markets)

from typing import Optional, Tuple
import math, re

from .models import Match, MatchStats

# Final statuses we treat as "settled"
FINAL_STATUSES = {"FT", "AET", "PEN"}


# -------------------- small helpers --------------------

def _as_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

def _get_final_int(obj, name: str) -> Optional[int]:
    """Safely read an integer stat (goals/corners/cards/etc). Return None if absent/unparsable."""
    if not hasattr(obj, name):
        return None
    try:
        v = getattr(obj, name)
        return int(v) if v is not None else None
    except Exception:
        return None

def _ceil_need_for_over(line: float) -> int:
    """
    For totals, 'over' needs >= floor(line)+1 (works for X.5 and integer lines).
    Example: 1.5 -> 2, 2.5 -> 3, 2.0 -> 2+1 -> 3 (push handled elsewhere).
    """
    return math.floor(float(line)) + 1

def _settle_total(total: Optional[int], line: float, side: str) -> str:
    """
    Return WIN/LOSE/VOID for a total with integer-line push.
    side: "over" or "under"
    """
    if total is None:
        return "VOID"
    side = (side or "").lower().strip()
    # push/VOID on exact integer lines
    if float(line).is_integer() and total == int(line):
        return "VOID"
    need = _ceil_need_for_over(line)
    if side == "over":
        return "WIN" if total >= need else "LOSE"
    if side == "under":
        return "WIN" if total < need else "LOSE"
    return "UNKNOWN"


# -------------------- pull final stats with fallback --------------------

def _pull_final_totals(match: Match):
    """
    Return final stats; fall back to MatchStats when Match.* is missing.
    Keys: gh, ga, ch, ca, card_h, card_a, yel_h, yel_a, red_h, red_a
    """
    # Prefer Match.*
    gh   = _as_int(getattr(match, "goals_home", None))
    ga   = _as_int(getattr(match, "goals_away", None))
    ch   = _as_int(getattr(match, "corners_home", None))
    ca   = _as_int(getattr(match, "corners_away", None))
    card_h = _as_int(getattr(match, "cards_home", None))
    card_a = _as_int(getattr(match, "cards_away", None))
    yel_h  = _as_int(getattr(match, "yellows_home", None))
    yel_a  = _as_int(getattr(match, "yellows_away", None))
    red_h  = _as_int(getattr(match, "reds_home", None))
    red_a  = _as_int(getattr(match, "reds_away", None))

    # Fill missing from MatchStats
    need_stats = any(v is None for v in (ch, ca, card_h, card_a, yel_h, yel_a, red_h, red_a))
    if need_stats:
        rows = list(MatchStats.objects.filter(match=match).select_related("team"))
        home_id = getattr(match.home, "id", None)
        away_id = getattr(match.away, "id", None)
        hrow = next((r for r in rows if getattr(r.team, "id", None) == home_id), None)
        arow = next((r for r in rows if getattr(r.team, "id", None) == away_id), None)

        def _fb_pair(curr_h, curr_a, attr):
            if curr_h is None and hrow is not None:
                curr_h = _as_int(getattr(hrow, attr, None))
            if curr_a is None and arow is not None:
                curr_a = _as_int(getattr(arow, attr, None))
            return curr_h, curr_a

        ch, ca         = _fb_pair(ch, ca, "corners")
        card_h, card_a = _fb_pair(card_h, card_a, "cards")
        yel_h, yel_a   = _fb_pair(yel_h, yel_a, "yellows")
        red_h, red_a   = _fb_pair(red_h, red_a, "reds")

    return dict(gh=gh, ga=ga, ch=ch, ca=ca, card_h=card_h, card_a=card_a,
                yel_h=yel_h, yel_a=yel_a, red_h=red_h, red_a=red_a)


# -------------------- spec parsers (robust) --------------------

def _boolish(s: str) -> Optional[bool]:
    """yes/true/1 -> True, no/false/0 -> False; else None."""
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"yes", "y", "true", "1"}:
        return True
    if t in {"no", "n", "false", "0"}:
        return False
    return None

def _parse_ou(spec: str) -> Optional[Tuple[float, str]]:
    """
    Accept: '1.5_over', '2.5_under', 'over_2.5', 'under_3.5'
    Return (line, 'over'|'under') or None
    """
    s = (spec or "").strip().lower().replace(" ", "")
    if not s:
        return None
    side = "over" if "over" in s else ("under" if "under" in s else None)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if side and nums:
        return float(nums[0]), side
    # fallback: 'X.Y_over'
    try:
        ln, sd = s.split("_", 1)
        if sd in ("over", "under"):
            return float(ln), sd
    except Exception:
        pass
    return None

def _parse_team_total_over(spec: str) -> Optional[Tuple[str, float]]:
    """
    Accept: 'home_o1.5', 'away_o0.5', 'home_over_1.5', 'over_1.5_home'
    Return ('home'|'away', line) or None
    """
    s = (spec or "").strip().lower().replace(" ", "")
    m = re.match(r"^(home|away)_(?:o|over)_?(\d+(?:\.\d+)?)$", s)
    if m:
        return m.group(1), float(m.group(2))
    m = re.match(r"^(?:o|over)_?(\d+(?:\.\d+)?)[_-]?(home|away)$", s)
    if m:
        return m.group(2), float(m.group(1))
    team = "home" if "home" in s else ("away" if "away" in s else None)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if team and nums:
        return team, float(nums[0])
    return None


# -------------------- evaluator --------------------

# matches/utils_settlement.py  (merge into your utils.py as needed)

# -------------------- final stats with robust reds/yellows fallback --------------------

# ===== Settlement core (drop-in) =====
# matches/utils.py
# matches/utils_settlement.py (for example)
import math, re
from typing import Optional, Tuple
from django.db import transaction
from .models import Match, MatchStats, DailyTicket

FINAL_STATUSES = {"FT", "AET", "PEN"}
GOAL_DETAILS_SET = {"Normal Goal", "Penalty", "Own Goal"}  # scoring events only

# ---------------- small helpers ----------------

def _as_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

def _get_final_int(obj, field: str) -> Optional[int]:
    return _as_int(getattr(obj, field, None))

def _ceil_need_for_over(line: float) -> int:
    # Minimum integer total that is a strict "over" for the line
    return math.floor(float(line)) + 1

def _settle_total(total: Optional[int], line: float, side: str) -> str:
    """Side in {'over','under'}; handles integer-line push as VOID."""
    if total is None:
        return "VOID"
    side = (side or "").strip().lower()
    if float(line).is_integer() and total == int(line):
        return "VOID"
    need = _ceil_need_for_over(line)
    if side == "over":
        return "WIN" if total >= need else "LOSE"
    if side == "under":
        return "WIN" if total < need else "LOSE"
    return "UNKNOWN"

def _parse_ou(spec: str) -> Optional[Tuple[float, str]]:
    """
    Accept: '1.5_over', '2.5_under', 'over_2.5', 'under_3.5'
    Return (line, 'over'|'under') or None
    """
    s = (spec or "").strip().lower().replace(" ", "")
    if not s:
        return None
    if "over" in s:
        side = "over"
    elif "under" in s:
        side = "under"
    else:
        side = None
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if side and nums:
        return float(nums[0]), side
    try:
        ln, sd = s.split("_", 1)
        if sd in ("over", "under"):
            return float(ln), sd
    except Exception:
        pass
    return None
def _parse_handicap(spec: str):
    """
    Accepts 'home_0', 'away_-0.5', '+0.25_home', etc.
    Returns (side: 'home'|'away', line: float) or None.
    """
    s = (spec or "").lower().replace(" ", "")
    # common form: side_line
    if "_" in s:
        side, ln = s.split("_", 1)
        if side in {"home","away"}:
            try:
                return side, float(ln.replace("+",""))
            except Exception:
                return None
    # alt: line_side
    parts = s.split("_")
    if len(parts) == 2 and parts[1] in {"home","away"}:
        try:
            return parts[1], float(parts[0].replace("+",""))
        except Exception:
            return None
    return None

def _asian_settle(home_goals: int, away_goals: int, side: str, line: float) -> str:
    """
    Standard Asian handicap settle for whole/half lines.
    (No split lines like +0.25; if you need them, we can add later.)
    side is the side *you backed* with the handicap.
    """
    if home_goals is None or away_goals is None:
        return "UNKNOWN"
    diff = (home_goals - away_goals)
    # Normalize to perspective of backed side
    if side == "away":
        diff = -diff

    # Apply line to the backed side
    diff += line

    # Whole-number push?
    if float(line).is_integer() and diff == 0:
        return "VOID"
    return "WIN" if diff > 0 else "LOSE"

def _parse_team_total_over(spec: str) -> Optional[Tuple[str, float]]:
    """
    Accept 'home_o1.5', 'away_o0.5', 'home_over_1.5', etc.
    Returns ('home'|'away', line) or None
    """
    s = (spec or "").strip().lower().replace(" ", "")
    m = re.search(r"^(home|away).*(?:o|over)_?(\d+(?:\.\d+)?)$", s)
    if m:
        side = m.group(1)
        line = float(m.group(2))
        return side, line
    return None

def _res_symbol(h: int, a: int) -> str:
    if h > a: return "H"
    if h < a: return "A"
    return "D"

def _norm_result_spec(s: str) -> Optional[str]:
    ss = (s or "").strip().upper()
    return ss if ss in {"H","A","D"} else None

def _boolish(s: str) -> Optional[bool]:
    v = (s or "").strip().lower()
    if v in {"yes","y","true","1"}: return True
    if v in {"no","n","false","0"}: return False
    return None

def _who_spec(s: str) -> Optional[str]:
    v = (s or "").strip().lower()
    if v in {"home","h"}: return "home"
    if v in {"away","a"}: return "away"
    if v in {"none","no","no_goal","ng"}: return "none"
    return None

# --- tiny utils ---
def _safe_int(x):
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

def _boolish(s: str) -> bool | None:
    v = (s or "").strip().lower()
    if v in {"yes","y","true","1"}: return True
    if v in {"no","n","false","0"}: return False
    return None

def _norm_result_spec(s: str) -> str | None:
    ss = (s or "").strip().upper()
    return ss if ss in {"H","A","D"} else None

# --- minutes & halftime: RAW FIRST, then DB ---
_BLOCKLIST = ("missed", "cancel", "disallow")

def _minute_from_raw_event(ev: dict) -> int | None:
    t = ev.get("time") or {}
    base  = t.get("elapsed")
    extra = t.get("extra") or t.get("stoppage") or t.get("extra_minute") or 0
    try:
        return (int(base) if base is not None else 0) + int(extra or 0)
    except Exception:
        return None

# Put near the top of utils.py (once), with your other imports
import re

GOAL_DETAILS_SET = {"Normal Goal", "Penalty", "Own Goal"}  # scoring events only
_BLOCKLIST = ("missed", "cancel", "disallow")  # filter out "Missed Penalty", etc.

def _minute_from_raw_event(ev: dict) -> int | None:
    t = ev.get("time") or {}
    base  = t.get("elapsed")
    extra = t.get("extra") or t.get("stoppage") or t.get("extra_minute") or 0
    try:
        return (int(base) if base is not None else 0) + int(extra or 0)
    except Exception:
        return None

def _goal_minutes(match):
    """
    Return (home_minutes, away_minutes) of *scoring* goals.
    1) raw_result_json['events']  (preferred)
    2) DB MatchEvent (type case-insensitive 'Goal' and allowed details)
    """
    # 1) RAW FIRST
    raw = getattr(match, "raw_result_json", None) or {}
    evs = raw.get("events") or []
    if isinstance(evs, list) and evs:
        H, A = [], []
        hid, aid = match.home_id, match.away_id
        for ev in evs:
            if (ev.get("type") or "").strip().lower() != "goal":
                continue
            det = (ev.get("detail") or "").strip()
            # allow only real scoring types; block missed/cancelled/disallowed
            if det and det not in GOAL_DETAILS_SET:
                dlow = det.lower()
                if any(bad in dlow for bad in _BLOCKLIST):
                    continue
            minute = _minute_from_raw_event(ev)
            if minute is None:
                continue
            tid = (ev.get("team") or {}).get("id")
            if tid == hid: H.append(minute)
            elif tid == aid: A.append(minute)
        H.sort(); A.sort()
        if H or A:
            return H, A

    # 2) DB FALLBACK (case-insensitive on type; whitelist details)
    try:
        from .models import MatchEvent
        qs = (MatchEvent.objects
              .filter(match_id=match.id, type__iexact="Goal")
              .exclude(minute=None))
        H, A = [], []
        hid, aid = match.home_id, match.away_id
        for ev in qs:
            det = (getattr(ev, "detail", "") or "").strip()
            if det and det not in GOAL_DETAILS_SET:
                if any(bad in det.lower() for bad in _BLOCKLIST):
                    continue
            base = getattr(ev, "minute", None)
            extra = (getattr(ev, "extra", None)
                     or getattr(ev, "stoppage", None)
                     or getattr(ev, "extra_minute", None) or 0)
            try:
                minute = int(base) + int(extra or 0)
            except Exception:
                continue
            tid = getattr(ev, "team_id", None)
            if tid == hid: H.append(minute)
            elif tid == aid: A.append(minute)
            else:
                # as a last resort, use is_home flag
                (H if getattr(ev, "is_home", False) else A).append(minute)
        H.sort(); A.sort()
        if H or A:
            return H, A
    except Exception:
        pass

    return [], []

def _halftime_scores(match):
    """
    Return (ht_home, ht_away).
    Prefer explicit fields; else raw_result_json['score']['halftime'];
    else derive from goal minutes.
    """
    # explicit model fields first
    for h_name, a_name in [
        ("goals_home_ht","goals_away_ht"),
        ("ht_goals_home","ht_goals_away"),
        ("ht_home","ht_away"),
    ]:
        ht_h = getattr(match, h_name, None)
        ht_a = getattr(match, a_name, None)
        if ht_h is not None and ht_a is not None:
            try:
                return int(ht_h), int(ht_a)
            except Exception:
                break

    # RAW halftime
    raw = getattr(match, "raw_result_json", None) or {}
    sc = (raw.get("score") or {}).get("halftime")
    if isinstance(sc, dict) and "home" in sc and "away" in sc:
        try:
            return int(sc["home"]), int(sc["away"])
        except Exception:
            pass
    if isinstance(sc, str) and "-" in sc:
        try:
            h, a = sc.split("-", 1)
            return int(h), int(a)
        except Exception:
            pass

    # derive via minutes if needed
    H, A = _goal_minutes(match)
    if H or A:
        return sum(1 for m in H if m <= 45), sum(1 for m in A if m <= 45)

    return None, None
# add near your other tiny parsers
import re

def _parse_htft_spec(raw: str):
    """
    Parse HTFT spec into ('H'|'D'|'A', 'H'|'D'|'A').
    Accepts: 'H-D', 'H/D', 'home-draw', '1/X', 'draw/away', 'HD', 'DH', etc.
    """
    if not raw:
        return None, None
    s = str(raw).strip().lower().replace(" ", "")

    # split if we have obvious separator
    if "-" in s or "/" in s:
        p = re.split(r"[-/]", s, maxsplit=1)
    else:
        # compact 2-char like 'hd', 'dh', 'ha', 'ad' ...
        if len(s) == 2 and all(ch in "hda12x" for ch in s):
            p = [s[0], s[1]]
        else:
            # try to extract two tokens like 'home', 'draw', 'away'
            p = re.findall(r"(home|away|draw|h|a|d|1|2|x)", s)
            if len(p) != 2:
                return None, None

    m = {
        "h": "H", "1": "H", "home": "H",
        "d": "D", "x": "D", "draw": "D",
        "a": "A", "2": "A", "away": "A",
    }
    ht = m.get(p[0])
    ft = m.get(p[1])
    return (ht, ft) if ht and ft else (None, None)

# --- totals & helpers reused by markets ---
def _ceil_need_for_over(line: float) -> int:
    import math
    return math.floor(float(line)) + 1

def _parse_ou(spec: str):
    """
    Accept: '1.5_over', '2.5_under', 'over_2.5', 'under 3.5'
    Return (line: float, side: 'over'|'under') or None
    """
    import re
    s = (spec or "").strip().lower().replace(" ", "")
    if not s:
        return None
    side = "over" if "over" in s else ("under" if "under" in s else None)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if side and nums:
        return float(nums[0]), side
    try:
        ln, sd = s.split("_", 1)
        if sd in {"over","under"}:
            return float(ln), sd
    except Exception:
        pass
    return None

def _parse_team_total_over(spec: str):
    """
    'home_o1.5' | 'away_o0.5' -> ('home'|'away', line: float)
    """
    s = (spec or "").strip().lower()
    if "_o" not in s:
        return None
    side, n = s.split("_o", 1)
    side = "home" if side in {"home","h"} else "away" if side in {"away","a"} else None
    try:
        line = float(n)
    except Exception:
        return None
    if not side:
        return None
    return side, line

def _res_symbol(h: int, a: int) -> str:
    if h > a: return "H"
    if h < a: return "A"
    return "D"

def _who_spec(s: str) -> str | None:
    v = (s or "").strip().lower()
    if v in {"home","h"}: return "home"
    if v in {"away","a"}: return "away"
    if v in {"none","no","ng","no_goal"}: return "none"
    return None

def _parse_handicap(spec: str):
    """
    'home_-0.5', 'away_+1', 'home_0' -> (side, line)
    """
    s = (spec or "").strip().lower().replace(" ", "")
    if "_" not in s:
        return None
    side, ln = s.split("_", 1)
    side = "home" if side in {"home","h"} else "away" if side in {"away","a"} else None
    if not side:
        return None
    try:
        line = float(ln)
    except Exception:
        return None
    return side, line

# --- MAIN: evaluate_selection_outcome ---
FINAL_STATUSES = {"FT", "AET", "PEN"}

def _pull_final_totals(match):
    """Return dict of final totals, filling from MatchStats if needed."""
    from .models import MatchStats
    gh   = _safe_int(getattr(match, "goals_home", None))
    ga   = _safe_int(getattr(match, "goals_away", None))
    ch   = _safe_int(getattr(match, "corners_home", None))
    ca   = _safe_int(getattr(match, "corners_away", None))
    card_h = _safe_int(getattr(match, "cards_home", None))
    card_a = _safe_int(getattr(match, "cards_away", None))
    yel_h  = _safe_int(getattr(match, "yellows_home", None))
    yel_a  = _safe_int(getattr(match, "yellows_away", None))
    red_h  = _safe_int(getattr(match, "reds_home", None))
    red_a  = _safe_int(getattr(match, "reds_away", None))

    need_stats = any(v is None for v in (ch, ca, card_h, card_a, yel_h, yel_a, red_h, red_a))
    if need_stats:
        rows = list(MatchStats.objects.filter(match=match).select_related("team"))
        home_id = getattr(match, "home_id", None)
        away_id = getattr(match, "away_id", None)
        hrow = next((r for r in rows if getattr(r, "team_id", None) == home_id), None)
        arow = next((r for r in rows if getattr(r, "team_id", None) == away_id), None)

        def _fb_pair(curr_h, curr_a, attr):
            if curr_h is None and hrow is not None:
                curr_h = _safe_int(getattr(hrow, attr, None))
            if curr_a is None and arow is not None:
                curr_a = _safe_int(getattr(arow, attr, None))
            return curr_h, curr_a

        ch, ca         = _fb_pair(ch, ca, "corners")
        card_h, card_a = _fb_pair(card_h, card_a, "cards")
        yel_h, yel_a   = _fb_pair(yel_h, yel_a, "yellows")
        red_h, red_a   = _fb_pair(red_h, red_a, "reds")

    return dict(gh=gh, ga=ga, ch=ch, ca=ca, card_h=card_h, card_a=card_a,
                yel_h=yel_h, yel_a=yel_a, red_h=red_h, red_a=red_a)

def _settle_total(total: int | None, line: float, side: str) -> str:
    if total is None:
        return "VOID"
    side = (side or "").strip().lower()
    # push on integer lines
    if float(line).is_integer() and total == int(line):
        return "VOID"
    need = _ceil_need_for_over(line)
    if side == "over":
        return "WIN" if total >= need else "LOSE"
    if side == "under":
        return "WIN" if total < need else "LOSE"
    return "UNKNOWN"

def evaluate_selection_outcome(match, market_code: str, specifier: str) -> str:
    """
    Return: WIN | LOSE | VOID | PENDING | UNKNOWN
    """
    status = (getattr(match, "status", "") or "").upper().strip()
    if status not in FINAL_STATUSES:
        return "PENDING"

    S = _pull_final_totals(match)
    gh, ga = S["gh"], S["ga"]
    ch, ca = S["ch"], S["ca"]
    card_h, card_a = S["card_h"], S["card_a"]
    yel_h, yel_a = S["yel_h"], S["yel_a"]
    red_h, red_a = S["red_h"], S["red_a"]

    m = (market_code or "").upper().strip()

    # ---- HANDICAP / DNB style
    if m in {"HANDICAP", "ASIAN_HANDICAP", "HOME_HANDICAP", "AWAY_HANDICAP"}:
        if gh is None or ga is None:
            return "VOID"
        ph = _parse_handicap(specifier)
        if not ph:
            return "UNKNOWN"
        side, line = ph
        home_adj = (gh + line) if side == "home" else gh
        away_adj = (ga + line) if side == "away" else ga
        if home_adj > away_adj: return "WIN" if side == "home" else "LOSE"
        if home_adj < away_adj: return "WIN" if side == "away" else "LOSE"
        return "VOID"

    # ---- 1X2 (FT)
    if m == "1X2":
        if gh is None or ga is None:
            return "VOID"
        want = _norm_result_spec(specifier)
        if not want:
            return "UNKNOWN"
        return "WIN" if _res_symbol(gh, ga) == want else "LOSE"

    # ---- BTTS
    if m == "BTTS":
        if gh is None or ga is None:
            return "VOID"
        want_yes = _boolish(specifier)
        if want_yes is None:
            return "UNKNOWN"
        yes = (gh > 0 and ga > 0)
        return "WIN" if yes == want_yes else "LOSE"

    # ---- OU (GOALS)
    if m == "OU":
        if gh is None or ga is None:
            return "VOID"
        parsed = _parse_ou(specifier)
        if not parsed:
            return "UNKNOWN"
        line, side = parsed
        return _settle_total(gh + ga, line, side)

    # ---- TEAM_TOTAL (GOALS)
    if m == "TEAM_TOTAL":
        if gh is None or ga is None:
            return "VOID"
        parsed = _parse_team_total_over(specifier)
        if not parsed:
            return "UNKNOWN"
        team, line = parsed
        team_total = gh if team == "home" else ga if team == "away" else None
        return _settle_total(team_total, line, "over")

    # ---- OU_CORNERS
    if m == "OU_CORNERS":
        if ch is None or ca is None:
            return "VOID"
        parsed = _parse_ou(specifier)
        if not parsed:
            return "UNKNOWN"
        line, side = parsed
        return _settle_total(ch + ca, line, side)

    # ---- TEAM_TOTAL_CORNERS
    if m == "TEAM_TOTAL_CORNERS":
        if ch is None or ca is None:
            return "VOID"
        parsed = _parse_team_total_over(specifier)
        if not parsed:
            return "UNKNOWN"
        team, line = parsed
        team_total = ch if team == "home" else ca if team == "away" else None
        return _settle_total(team_total, line, "over")

    # ---- CARDS / YELLOWS / REDS totals
    if m in {"CARDS_TOT", "YELLOWS_TOT", "REDS_TOT"}:
        parsed = _parse_ou(specifier)
        if not parsed:
            return "UNKNOWN"
        line, side = parsed
        if m == "CARDS_TOT":
            if card_h is None or card_a is None: return "VOID"
            tot = card_h + card_a
        elif m == "YELLOWS_TOT":
            if yel_h is None or yel_a is None: return "VOID"
            tot = yel_h + yel_a
        else:
            if red_h is None or red_a is None: return "VOID"
            tot = red_h + red_a
        return _settle_total(tot, line, side)

    # ---- Minute/sequence markets (use raw/DB goal minutes)
    if m == "1X2_0_15":
        if gh is None or ga is None:
            return "UNKNOWN"
        want = _norm_result_spec(specifier) or "D"
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            return "UNKNOWN"
        h15 = sum(1 for x in Hm if x <= 15)
        a15 = sum(1 for x in Am if x <= 15)
        return "WIN" if _res_symbol(h15, a15) == want else "LOSE"

    if m == "FIRST_TO_SCORE":
        if gh is None or ga is None:
            return "UNKNOWN"
        who = _who_spec(specifier)
        if who is None:
            return "UNKNOWN"
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            return "WIN" if (gh == 0 and ga == 0 and who == "none") else "UNKNOWN"
        first = ("home", Hm[0]) if Hm else None
        if Am:
            cand = ("away", Am[0])
            if first is None or cand[1] < first[1]:
                first = cand
        if first is None:
            return "WIN" if who == "none" else "LOSE"
        return "WIN" if who == first[0] else "LOSE"
    # ---- HTFT ----
    if m == "HTFT":
        if gh is None or ga is None:
            return "UNKNOWN"
        want_ht, want_ft = _parse_htft_spec(specifier)
        if not want_ht or not want_ft:
            return "UNKNOWN"
        ht_h, ht_a = _halftime_scores(match)
        if ht_h is None or ht_a is None:
            return "UNKNOWN"
        ok = (_res_symbol(ht_h, ht_a) == want_ht) and (_res_symbol(gh, ga) == want_ft)
        return "WIN" if ok else "LOSE"


    if m == "LAST_TO_SCORE":
        if gh is None or ga is None:
            return "UNKNOWN"
        who = _who_spec(specifier)
        if who is None:
            return "UNKNOWN"
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            return "WIN" if (gh == 0 and ga == 0 and who == "none") else "UNKNOWN"
        last = ("home", Hm[-1]) if Hm else None
        if Am:
            cand = ("away", Am[-1])
            if last is None or cand[1] > last[1]:
                last = cand
        if last is None:
            return "WIN" if who == "none" else "LOSE"
        return "WIN" if who == last[0] else "LOSE"

    if m == "GOAL_0_15":
        want_yes = _boolish(specifier)
        if want_yes is None:
            return "UNKNOWN"

        # 1) Primary source: actual goal minutes
        Hm, Am = _goal_minutes(match)
        if Hm or Am:
            happened = (Hm and Hm[0] <= 15) or (Am and Am[0] <= 15)
            return "WIN" if happened == want_yes else "LOSE"

        # 2) Fallback: if we know the game finished 0–0, then "no" must be true
        if gh is not None and ga is not None and (gh + ga) == 0:
            return "WIN" if (want_yes is False) else "LOSE"

        # 3) Otherwise we don't know
        return "UNKNOWN"

    if m == "GOAL_IN_BOTH_HALVES":
        if gh is None or ga is None:
            return "UNKNOWN"
        want_yes = _boolish(specifier)
        if want_yes is None:
            return "UNKNOWN"
        ht_h, ht_a = _halftime_scores(match)
        if ht_h is not None and ht_a is not None:
            first_half_goals  = ht_h + ht_a
            second_half_goals = (gh - ht_h) + (ga - ht_a)
            happened = (first_half_goals > 0) and (second_half_goals > 0)
            return "WIN" if happened == want_yes else "LOSE"
        # minutes fallback
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            return "UNKNOWN"
        first_half  = any(mn <= 45 for mn in (Hm + Am))
        second_half = any(mn >= 46 for mn in (Hm + Am))
        happened = first_half and second_half
        return "WIN" if happened == want_yes else "LOSE"

    # fallthrough
    return "UNKNOWN"


# ---------------- ticket settlement ----------------

from django.db import transaction
from .models import Match, DailyTicket
# import evaluate_selection_outcome and helpers from where you put them

def settle_ticket_for_date(ticket_date, *, league_id: int = 0):
    """
    Atomically settle the latest DailyTicket for (ticket_date, league_id).
    Uses SELECT ... FOR UPDATE inside transaction.atomic().
    """
    with transaction.atomic():
        dt = (DailyTicket.objects
              .filter(ticket_date=ticket_date, league_id=league_id)
              .order_by('-id')
              .select_for_update()
              .first())
        if not dt:
            return None

        changed = False
        any_pending = False
        any_lose = False
        any_win_or_void = False

        new_selections = []
        for sel in (dt.selections or []):
            mid = sel.get("match_id")
            market = sel.get("market") or sel.get("market_code")
            spec   = sel.get("specifier")

            m = Match.objects.filter(id=mid).select_related("home","away").first()
            if not m:
                # cannot locate match → VOID this leg
                if sel.get("result") != "VOID":
                    sel["result"] = "VOID"
                    changed = True
                any_win_or_void = True
                new_selections.append(sel)
                continue

            outcome = evaluate_selection_outcome(m, market, spec)
            # business choice: treat UNKNOWN as VOID
            if outcome == "UNKNOWN":
                outcome = "VOID"

            prev = sel.get("result")
            if prev != outcome:
                sel["result"] = outcome
                changed = True

            any_pending    |= (outcome == "PENDING")
            any_lose       |= (outcome == "LOSE")
            any_win_or_void |= (outcome in {"WIN","VOID"})

            new_selections.append(sel)

        # derive ticket status
        if any_lose:
            new_status = "LOST"
        elif any_pending:
            new_status = "PENDING"
        elif any_win_or_void:
            new_status = "WON"
        else:
            new_status = dt.status or "PENDING"

        if changed or new_status != (dt.status or ""):
            dt.selections = new_selections
            dt.status = new_status
            dt.save(update_fields=["selections", "status", "updated_at"])

        return dt
































from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta, timezone
from typing import List

import math
import random

from django.db import transaction

from .models import DailyTicket, PredictedMarket


# ──────────────────────────────── Config / Constants ───────────────────────────────

# Use this special bucket to store the daily ~2.00 odds ticket.
# Your regular "global" ticket can remain league_id = 0.
TWO_ODDS_LEAGUE_ID = -1

# Markets allowed to be picked (families)
TOP_MARKET_WHITELIST = {
    "1X2",
    
    "BTTS",
    "OU_CORNERS",
    "CARDS_TOT",
    "YELLOWS_TOT",
    "REDS_TOT","1X2_0_15","AWAY_HANDICAP","FIRST_TO_SCORE","GOAL_IN_BOTH_HALVES","HTFT","LAST_TO_SCORE","ODDEVEN",
}

# Hard exclusions even if someone adds them to the whitelist
EXCLUDED_FAMILIES = {"TEAM_TOTAL", "TEAM_TOTAL_CORNERS","OU"}

# Matches we consider upcoming/available
UPCOMING_STATUSES = {"NS", "TBD", "PST"}


# ──────────────────────────────── Helpers ─────────────────────────────────────────

def _day_bounds_utc(d: Date):
    """Return UTC [start, end) datetimes for a UTC date."""
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def _market_family(code: str) -> str:
    """Normalize market code into a family label (e.g., '1X2', 'OU', ...)."""
    c = (code or "").upper()
    return "1X2" if c.startswith("1X2") else c


def _is_under(code: str, spec: str) -> bool:
    """
    Treat any OU-like market with 'under' in specifier as UNDER.
    TEAM_TOTAL* lines are over-only in many datasets and won't match here.
    """
    c = (code or "").upper()
    s = (spec or "").lower()
    if c in {"OU", "OU_CORNERS", "CARDS_TOT", "YELLOWS_TOT", "REDS_TOT"} or c.startswith("OU"):
        return ("_under" in s) or s.endswith("_u") or s.startswith("u")
    return False


def _bookish_odds_for_prob(p: float, overround: float = 0.06) -> float:
    """
    Approximate 'book' odds from model probability by applying an overround.
    """
    p_eff = max(1e-9, min(1.0 - 1e-9, float(p))) * (1.0 - overround)
    return 1.0 / p_eff


# ──────────────────────────────── Candidate Model ─────────────────────────────────

@dataclass
class _Cand:
    pm_id: int
    match_id: int
    league_id: int
    kickoff_utc: datetime
    home: str
    away: str
    market_code: str
    specifier: str
    p: float           # model probability 0..1
    fair_odds: float   # 1/p (margin-free)
    book_odds: float   # book-ish odds from p with overround


# ──────────────────────────────── Candidate Fetch ────────────────────────────────

def _fetch_two_odds_candidates(
    ticket_date: Date,
    *,
    min_p: float,
    max_fair_odds: float,
    max_count: int = 400,
) -> List[_Cand]:
    """
    Pull high-probability, low-fair-odds candidates for the UTC day,
    excluding unders, excluded families, and non-upcoming matches.
    """
    start, end = _day_bounds_utc(ticket_date)

    qs = (
        PredictedMarket.objects
        .filter(kickoff_utc__gte=start, kickoff_utc__lt=end)
        .select_related("match", "match__home", "match__away")
    )[:2000]

    out: List[_Cand] = []
    for pm in qs:
        m = pm.match
        if not m:
            continue
        if (m.status or "").upper() not in UPCOMING_STATUSES:
            continue
        if not pm.p_model:
            continue

        fam = _market_family(pm.market_code)
        if fam not in TOP_MARKET_WHITELIST:
            continue
        if fam in EXCLUDED_FAMILIES:
            continue
        if _is_under(pm.market_code, pm.specifier):
            continue

        p = float(pm.p_model)
        fair = 1.0 / max(1e-9, min(1.0 - 1e-9, p))

        if p < float(min_p):
            continue
        if fair > float(max_fair_odds):
            continue

        out.append(
            _Cand(
                pm_id=pm.id,
                match_id=m.id,
                league_id=getattr(m, "league_id", 0) or 0,
                kickoff_utc=m.kickoff_utc,
                home=getattr(m.home, "name", "") or "",
                away=getattr(m.away, "name", "") or "",
                market_code=pm.market_code,
                specifier=pm.specifier,
                p=p,
                fair_odds=fair,
                book_odds=_bookish_odds_for_prob(p, overround=0.06),
            )
        )

    # Prefer stronger legs by default (we'll still randomize during assembly)
    out.sort(key=lambda c: (-c.p, c.fair_odds))
    return out[:max_count]


# ──────────────────────────────── Ticket Builder ─────────────────────────────────


# --- replace just this function in matches/utils.py ---


from typing import List, Tuple
from datetime import date as Date
import math, random
from django.db import transaction
from .models import DailyTicket



from typing import List
from datetime import date as Date
import math, random
from django.db import transaction
from .models import DailyTicket

TWO_ODDS_LEAGUE_ID = -1  # keep consistent across your app


def _meets_two_odds_rules(dt: DailyTicket, lower: float, upper: float, min_legs: int) -> bool:
    """Check if an existing ticket satisfies constraints."""
    if not dt:
        return False
    if (dt.legs or 0) < int(min_legs):
        return False
    acc = float(dt.acc_bookish_odds or 0.0)
    if not math.isfinite(acc):
        return False
    return (acc >= lower) and (acc <= upper)


@transaction.atomic
def get_or_create_daily_two_odds_ticket(
    ticket_date: Date,
    *,
    target_odds: float = 3.00,
    over_tolerance: float = 0.15,   # allow up to +15% overshoot
    min_p: float = 0.60,
    max_fair_odds: float = 1.40,
    min_legs: int = 2,
    max_legs: int = 6,
    attempts: int = 800,
    force_regenerate: bool = False,
) -> DailyTicket:
    """
    Build (or reuse) a DAILY ticket such that:
      - product(book_odds) >= target_odds  (no undershoot)
      - product(book_odds) <= target_odds*(1+over_tolerance)
      - legs >= min_legs
    If an existing ticket violates these, it will be regenerated automatically.
    """
    # bounds
    lower = float(target_odds)
    upper = float(target_odds) * (1.0 + over_tolerance)

    # try to reuse
    existing = DailyTicket.objects.filter(
        ticket_date=ticket_date,
        league_id=TWO_ODDS_LEAGUE_ID,
    ).first()

    if existing and not force_regenerate:
        if _meets_two_odds_rules(existing, lower, upper, min_legs):
            return existing
        # else: fall through to rebuild

    # ---- build fresh ----
    cands = _fetch_two_odds_candidates(
        ticket_date,
        min_p=min_p,
        max_fair_odds=max_fair_odds,
    )
    if not cands:
        raise ValueError("No eligible candidates for daily two-odds ticket. Relax filters or ingest predictions.")

    rnd = random.Random(int(ticket_date.strftime("%Y%m%d")))  # deterministic per date

    best_inband: List[_Cand] = []
    best_inband_diff = float("inf")
    best_inband_prod = None

    best_over: List[_Cand] = []
    best_over_prod = float("inf")

    def logdiff(prod: float, tgt: float) -> float:
        try:
            return abs(math.log(prod) - math.log(tgt))
        except Exception:
            return abs(prod - tgt)

    base = list(cands)

    for _ in range(int(attempts)):
        rnd.shuffle(base)

        acc: List[_Cand] = []
        prod = 1.0

        # grow greedily
        for c in base:
            if len(acc) >= int(max_legs):
                break
            trial = prod * c.book_odds

            # after we already have min_legs, avoid massive jumps beyond the cap
            if len(acc) >= int(min_legs) and trial > upper * 1.25:
                continue

            acc.append(c)
            prod = trial

            if len(acc) >= int(min_legs) and (lower <= prod <= upper):
                break

        # if we don't have min_legs yet, try next attempt
        if len(acc) < int(min_legs):
            continue

        # if still below lower and we have room, add the best extra leg to approach target
        while prod < lower and len(acc) < int(max_legs):
            best_extra = None
            best_score = float("inf")
            for c in base:
                if c in acc:
                    continue
                trial = prod * c.book_odds
                score = logdiff(trial, target_odds)
                if score < best_score:
                    best_score = score
                    best_extra = c
            if not best_extra:
                break
            acc.append(best_extra)
            prod *= best_extra.book_odds
            if len(acc) >= int(min_legs) and (lower <= prod <= upper):
                break

        # classify attempt
        if len(acc) >= int(min_legs) and (lower <= prod <= upper):
            diff = logdiff(prod, target_odds)
            if diff < best_inband_diff:
                best_inband_diff = diff
                best_inband = acc[:]
                best_inband_prod = prod
        elif prod >= lower:
            if prod < best_over_prod:
                best_over_prod = prod
                best_over = acc[:]
        # else undershoot: discard

    # pick solution to persist
    if best_inband:
        chosen = best_inband
        final_prod = best_inband_prod
    elif best_over:
        if best_over_prod <= upper:
            chosen = best_over
            final_prod = best_over_prod
        else:
            raise ValueError(
                f"Best overshoot {best_over_prod:.2f} exceeds cap {upper:.2f}. "
                "Increase over_tolerance / max_legs or relax filters."
            )
    else:
        raise ValueError(
            f"No combination reached target {target_odds:.2f} with ≥{min_legs} legs. "
            "Relax filters or increase max_legs/attempts."
        )

    # hard guard before saving (never save invalid)
    if not (len(chosen) >= int(min_legs) and lower <= float(final_prod) <= upper):
        raise ValueError("Internal guard: built ticket violates constraints; not saving.")

    # metrics
    acc_p = 1.0
    acc_fair = 1.0
    acc_book = 1.0
    for c in chosen:
        acc_p *= max(1e-9, min(1.0 - 1e-9, c.p))
        acc_fair *= c.fair_odds
        acc_book *= c.book_odds

    selections = [{
        "match_id": c.match_id,
        "league_id": c.league_id,
        "kickoff_utc": c.kickoff_utc.isoformat(),
        "home": c.home,
        "away": c.away,
        "market": c.market_code,
        "specifier": c.specifier,
        "p": c.p,
        "fair_odds": c.fair_odds,
    } for c in chosen]

    # if there is an invalid existing row, we can just update it
    dt, _ = DailyTicket.objects.update_or_create(
        ticket_date=ticket_date,
        league_id=TWO_ODDS_LEAGUE_ID,
        defaults={
            "selections": selections,
            "legs": len(selections),
            "acc_probability": float(acc_p if math.isfinite(acc_p) else 0.0),
            "acc_fair_odds": float(acc_fair if math.isfinite(acc_fair) else 0.0),
            "acc_bookish_odds": float(acc_book if math.isfinite(acc_book) else 0.0),
            "status": "pending",
        },
    )
    return dt




















# matches/utils.py
import math, re
from datetime import date as Date
from typing import Optional, Tuple, List

from django.db import transaction

from .models import Match, MatchStats, DailyTicket

# === keep this constant in one place and re-use everywhere ===
TWO_ODDS_LEAGUE_ID = -1

# A match is "final" if status is one of:
FINAL_STATUSES = {"FT", "AET", "PEN"}

# -------------------- small helpers --------------------
def _norm_result_spec(s: str) -> Optional[str]:
    """
    Normalize many ways of saying the 1X2 result to 'H'/'D'/'A'.
    Accepts: 'home','h','1' -> 'H'; 'draw','d','x' -> 'D'; 'away','a','2' -> 'A'.
    """
    if not s:
        return None
    t = str(s).strip().lower()
    if t in {"h", "home", "1"}:
        return "H"
    if t in {"d", "x", "draw"}:
        return "D"
    if t in {"a", "away", "2"}:
        return "A"
    return None

def _boolish(s: str) -> Optional[bool]:
    """
    Normalize yes/no booleans: yes/true/1 -> True, no/false/0 -> False.
    """
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"yes", "y", "true", "1"}:
        return True
    if t in {"no", "n", "false", "0"}:
        return False
    return None

def _who_spec(s: str) -> Optional[str]:
    """
    Normalize 'home'/'h' and 'away'/'a' and 'none/ng' into 'home'/'away'/'none'.
    """
    if not s:
        return None
    t = str(s).strip().lower()
    if t in {"home", "h"}:
        return "home"
    if t in {"away", "a"}:
        return "away"
    if t in {"none", "no", "no_goal", "ng"}:
        return "none"
    return None

def _as_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None

def _ceil_need_for_over(line: float) -> int:
    """For X.5 lines, over needs >= floor(X)+1; for integers we handle push elsewhere."""
    return math.floor(float(line)) + 1

def _pull_final_totals(match: Match):
    """
    Return final stats; fall back to MatchStats when Match.* is missing.
    Keys: gh, ga, ch, ca, card_h, card_a, yel_h, yel_a, red_h, red_a
    """
    gh   = _as_int(getattr(match, "goals_home", None))
    ga   = _as_int(getattr(match, "goals_away", None))
    ch   = _as_int(getattr(match, "corners_home", None))
    ca   = _as_int(getattr(match, "corners_away", None))
    card_h = _as_int(getattr(match, "cards_home", None))
    card_a = _as_int(getattr(match, "cards_away", None))
    yel_h  = _as_int(getattr(match, "yellows_home", None))
    yel_a  = _as_int(getattr(match, "yellows_away", None))
    red_h  = _as_int(getattr(match, "reds_home", None))
    red_a  = _as_int(getattr(match, "reds_away", None))

    need_stats = any(v is None for v in (ch, ca, card_h, card_a, yel_h, yel_a, red_h, red_a))
    if need_stats:
        rows = list(MatchStats.objects.filter(match=match).select_related("team"))
        home_id = getattr(match.home, "id", None)
        away_id = getattr(match.away, "id", None)
        hrow = next((r for r in rows if getattr(r.team, "id", None) == home_id), None)
        arow = next((r for r in rows if getattr(r.team, "id", None) == away_id), None)

        def _fb_pair(curr_h, curr_a, attr):
            if curr_h is None and hrow is not None:
                curr_h = _as_int(getattr(hrow, attr, None))
            if curr_a is None and arow is not None:
                curr_a = _as_int(getattr(arow, attr, None))
            return curr_h, curr_a

        ch, ca         = _fb_pair(ch, ca, "corners")
        card_h, card_a = _fb_pair(card_h, card_a, "cards")
        yel_h, yel_a   = _fb_pair(yel_h, yel_a, "yellows")
        red_h, red_a   = _fb_pair(red_h, red_a, "reds")

    return dict(gh=gh, ga=ga, ch=ch, ca=ca, card_h=card_h, card_a=card_a,
                yel_h=yel_h, yel_a=yel_a, red_h=red_h, red_a=red_a)

# -------------------- spec parsers --------------------

def _parse_ou(spec: str) -> Optional[Tuple[float, str]]:
    """
    Accept: '1.5_over', '2.5_under', 'over_2.5', 'under_3.5'
    Return (line, 'over'|'under') or None
    """
    s = (spec or "").strip().lower().replace(" ", "")
    if not s:
        return None
    side = "over" if "over" in s else ("under" if "under" in s else None)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if side and nums:
        return float(nums[0]), side
    try:
        ln, sd = s.split("_", 1)
        if sd in ("over", "under"):
            return float(ln), sd
    except Exception:
        pass
    return None

def _parse_team_total_over(spec: str) -> Optional[Tuple[str, float]]:
    """
    Accept: 'home_o1.5', 'away_o0.5', 'home_over_1.5', 'over_1.5_home'
    Return ('home'|'away', line) or None
    """
    s = (spec or "").strip().lower().replace(" ", "")
    m = re.match(r"^(home|away)_(?:o|over)_?(\d+(?:\.\d+)?)$", s)
    if m:
        return m.group(1), float(m.group(2))
    m = re.match(r"^(?:o|over)_?(\d+(?:\.\d+)?)[_-]?(home|away)$", s)
    if m:
        return m.group(2), float(m.group(1))
    team = "home" if "home" in s else ("away" if "away" in s else None)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if team and nums:
        return team, float(nums[0])
    return None

def _settle_total(total: Optional[int], line: float, side: str) -> str:
    """WIN/LOSE/VOID with push on exact integer lines."""
    if total is None:
        return "VOID"
    side = (side or "").lower().strip()
    # push on exact integer line
    if float(line).is_integer() and total == int(line):
        return "VOID"
    need = _ceil_need_for_over(line)
    if side == "over":
        return "WIN" if total >= need else "LOSE"
    if side == "under":
        return "WIN" if total < need else "LOSE"
    return "UNKNOWN"


# --- Add near the top of matches/utils.py (with your other helpers) ---
import re

def _get_attr_any(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def _parse_goal_minutes_blob(v):
    """
    Accepts list[int] or CSV/semicolon strings like '12,45+2,78' or '12;45+1'.
    Returns a sorted list of integer minutes. '45+2' => 47.
    """
    def _to_min(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        # 45+2 etc.
        if "+" in s:
            a, b = s.split("+", 1)
            try:
                return int(a) + int(b)
            except Exception:
                return None
        try:
            return int(float(s))
        except Exception:
            return None

    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        mins = [_to_min(x) for x in v]
        return sorted([m for m in mins if isinstance(m, int)])
    # assume string
    parts = re.split(r"[,; ]+", str(v))
    mins = [_to_min(p) for p in parts]
    return sorted([m for m in mins if isinstance(m, int)])

def _goal_mnutes(match):
    """
    Returns (home_minutes:list[int], away_minutes:list[int]) if we can find any source,
    else ([], []).
    Tries several common field names, then optional MatchEvent model.
    """
    # Try common match attributes first
    pairs = [
        ("goal_minutes_home", "goal_minutes_away"),
        ("goals_time_home", "goals_time_away"),
        ("home_goal_times", "away_goal_times"),
        ("home_goals_minutes", "away_goals_minutes"),
    ]
    for h_attr, a_attr in pairs:
        h = getattr(match, h_attr, None)
        a = getattr(match, a_attr, None)
        if h is not None or a is not None:
            return _parse_goal_minutes_blob(h), _parse_goal_minutes_blob(a)

    # Try pulling from a MatchEvent table if you have one
    try:
        from .models import MatchEvent  # optional model
        evs = list(MatchEvent.objects.filter(
            match=match, type__in=["goal", "penalty_goal", "own_goal"]
        ).select_related("team"))
        if evs:
            home_id = getattr(match.home, "id", None)
            away_id = getattr(match.away, "id", None)
            H, A = [], []
            for e in evs:
                # try multiple field names for elapsed minute
                minute = _get_attr_any(e, ["minute", "elapsed", "time"])
                stoppage = _get_attr_any(e, ["stoppage", "extra", "extra_minute"])
                base = int(minute) if minute is not None else 0
                extra = int(stoppage) if stoppage is not None else 0
                mm = base + extra
                tid = getattr(e.team, "id", None)
                if tid == home_id:
                    H.append(mm)
                elif tid == away_id:
                    A.append(mm)
            return sorted(H), sorted(A)
    except Exception:
        pass

    return [], []

def _halftme_scores(match):
    """
    Returns (ht_home, ht_away) if available or derivable from goal minutes, else (None, None).
    """
    # direct HT fields (try several common names)
    ht_h = _get_attr_any(match, ["goals_home_ht", "ht_goals_home", "home_ht_goals", "ht_home"])
    ht_a = _get_attr_any(match, ["goals_away_ht", "ht_goals_away", "away_ht_goals", "ht_away"])
    if ht_h is not None and ht_a is not None:
        try:
            return int(ht_h), int(ht_a)
        except Exception:
            pass

    # derive from minutes
    H, A = _goal_minutes(match)
    if H or A:
        h1 = sum(1 for m in H if m <= 45)
        a1 = sum(1 for m in A if m <= 45)
        return h1, a1

    return None, None

def _ft_scores(match):
    """Return final score (gh, ga) or (None, None)."""
    gh = _get_final_int(match, "goals_home")
    ga = _get_final_int(match, "goals_away")
    return gh, ga

def _res_symbol(home_goals: int, away_goals: int) -> str:
    """Return 'H' | 'A' | 'D' based on a scoreline (assumes ints)."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


# -------------------- outcome per leg --------------------

def evaluate_selection_outcomes(match: Match, market_code: str, specifier: str) -> str:
    # ---------- local normalizers (robust comparisons) ----------
    def _norm_result_spec(s: str) -> Optional[str]:
        """
        Normalize many ways of saying the 1X2 result to 'H'/'D'/'A'.
        Accepts: 'home','h','1' -> 'H'; 'draw','d','x' -> 'D'; 'away','a','2' -> 'A'.
        """
        if not s:
            return None
        t = str(s).strip().lower()
        if t in {"h", "home", "1"}:
            return "H"
        if t in {"d", "x", "draw"}:
            return "D"
        if t in {"a", "away", "2"}:
            return "A"
        return None

    def _boolish(s: str) -> Optional[bool]:
        """Normalize yes/no booleans: yes/true/1 -> True, no/false/0 -> False."""
        if s is None:
            return None
        t = str(s).strip().lower()
        if t in {"yes", "y", "true", "1"}:
            return True
        if t in {"no", "n", "false", "0"}:
            return False
        return None

    def _who_spec(s: str) -> Optional[str]:
        """Normalize 'home'/'h' and 'away'/'a' and 'none/ng' into 'home'/'away'/'none'."""
        if not s:
            return None
        t = str(s).strip().lower()
        if t in {"home", "h"}:
            return "home"
        if t in {"away", "a"}:
            return "away"
        if t in {"none", "no", "no_goal", "ng"}:
            return "none"
        return None

    # ---------- gate on final status ----------
    status = (getattr(match, "status", "") or "").upper()
    if status not in FINAL_STATUSES:
        return "PENDING"

    # ---------- pull final numbers (with stats fallback) ----------
    S = _pull_final_totals(match)
    gh, ga = S["gh"], S["ga"]
    ch, ca = S["ch"], S["ca"]
    card_h, card_a = S["card_h"], S["card_a"]
    yel_h, yel_a = S["yel_h"], S["yel_a"]
    red_h, red_a = S["red_h"], S["red_a"]

    m = (market_code or "").upper().strip()

    # ---------- EXISTING cases (1X2, BTTS, OU, TEAM_TOTAL, OU_CORNERS, TEAM_TOTAL_CORNERS, CARDS/YELLOWS/REDS) ----------
    # Keep your current code here unchanged. Example:
    # out = evaluate_existing_markets_if_you_have_one(match, m, specifier, S)
    # if out is not None:
    #     return out

    # ---------- NEW cases (hardened) ----------

    # 1) 1X2_0_15 : result over minutes 0-15 (inclusive)
    if m == "1X2_0_15":
        want = _norm_result_spec(specifier) or "D"
        if gh is None or ga is None:
            return "UNKNOWN"
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            return "UNKNOWN"  # no minute data to compute early result
        h15 = sum(1 for x in Hm if x <= 15)
        a15 = sum(1 for x in Am if x <= 15)
        return "WIN" if _res_symbol(h15, a15) == want else "LOSE"

    # 2) AWAY_HANDICAP : spec like '-1.0' or 'away_-1.5' (we use away team always)
    if m == "AWAY_HANDICAP":
        if gh is None or ga is None:
            return "UNKNOWN"
        s = (specifier or "").lower().strip()
        # accept plain numbers or 'away_-1.0'
        line = None
        try:
            line = float(s)
        except Exception:
            if "_" in s:
                try:
                    _, ln = s.split("_", 1)
                    line = float(ln)
                except Exception:
                    pass
        if line is None:
            return "UNKNOWN"
        # apply Asian handicap to away team
        adj_away = ga + line
        # push on integer lines
        if float(line).is_integer() and adj_away == gh:
            return "VOID"
        return "WIN" if adj_away > gh else "LOSE"

    # 3) FIRST_TO_SCORE : spec 'home'|'away'|'none' (no goal)
    if m == "FIRST_TO_SCORE":
        if gh is None or ga is None:
            return "UNKNOWN"
        who = _who_spec(specifier)
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            # if no minute data, fall back to totals: if 0-0 then 'none' wins, else unknown who scored first
            return "WIN" if (gh == 0 and ga == 0 and who == "none") else "UNKNOWN"
        first = None
        if Hm:
            first = ("home", Hm[0])
        if Am:
            cand = ("away", Am[0])
            if first is None or cand[1] < first[1]:
                first = cand
        if first is None:  # no goals
            return "WIN" if who == "none" else "LOSE"
        return "WIN" if who == first[0] else "LOSE"

    # 4) LAST_TO_SCORE : spec 'home'|'away'|'none'
    if m == "LAST_TO_SCORE":
        if gh is None or ga is None:
            return "UNKNOWN"
        who = _who_spec(specifier)
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            # no minute data; if 0-0 => 'none', else unknown who last scored
            return "WIN" if (gh == 0 and ga == 0 and who == "none") else "UNKNOWN"
        last = None
        if Hm:
            last = ("home", Hm[-1])
        if Am:
            cand = ("away", Am[-1])
            if last is None or cand[1] > last[1]:
                last = cand
        if last is None:  # 0-0
            return "WIN" if who == "none" else "LOSE"
        return "WIN" if who == last[0] else "LOSE"

    # 5) GOAL_0_15 : 'yes' if any goal <= 15'
    if m == "GOAL_0_15":
        if gh is None or ga is None:
            return "UNKNOWN"
        want_yes = _boolish(specifier)
        if want_yes is None:
            return "UNKNOWN"
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            return "UNKNOWN"
        happened = (Hm and Hm[0] <= 15) or (Am and Am[0] <= 15)
        return "WIN" if happened == want_yes else "LOSE"

    # 6) GOAL_IN_BOTH_HALVES : 'yes'|'no'
    if m == "GOAL_IN_BOTH_HALVES":
        if gh is None or ga is None:
            return "UNKNOWN"
        want_yes = _boolish(specifier)
        if want_yes is None:
            return "UNKNOWN"
        # try HT fields first
        ht_h, ht_a = _halftime_scores(match)
        if ht_h is not None and ht_a is not None:
            # any goal in 1st half?
            h1 = ht_h + ht_a
            # goals in 2nd half = FT - HT
            h2 = (gh - ht_h) + (ga - ht_a)
            happened = (h1 > 0) and (h2 > 0)
            return "WIN" if happened == want_yes else "LOSE"
        # else use minutes
        Hm, Am = _goal_minutes(match)
        if not Hm and not Am:
            return "UNKNOWN"
        first_half = any(mn <= 45 for mn in Hm + Am)
        second_half = any(mn >= 46 for mn in Hm + Am)
        happened = first_half and second_half
        return "WIN" if happened == want_yes else "LOSE"

    # 7) HTFT : spec 'H-D', 'D-A', and also supports '1/X', 'home-draw'
    if m == "HTFT":
        if gh is None or ga is None:
            return "UNKNOWN"
        raw = (specifier or "").strip()
        if "-" in raw:
            p1, p2 = raw.split("-", 1)
        elif "/" in raw:
            p1, p2 = raw.split("/", 1)
        else:
            return "UNKNOWN"
        want_ht = _norm_result_spec(p1)
        want_ft = _norm_result_spec(p2)
        if not want_ht or not want_ft:
            return "UNKNOWN"
        ht_h, ht_a = _halftime_scores(match)
        if ht_h is None or ht_a is None:
            return "UNKNOWN"
        ok = (_res_symbol(ht_h, ht_a) == want_ht) and (_res_symbol(gh, ga) == want_ft)
        return "WIN" if ok else "LOSE"

    # 8) ODDEVEN : spec 'odd' | 'even'
    if m == "ODDEVEN":
        if gh is None or ga is None:
            return "UNKNOWN"
        want = (specifier or "").strip().lower()
        tot = gh + ga
        is_odd = (tot % 2) == 1
        if want in {"odd", "o"}:
            return "WIN" if is_odd else "LOSE"
        if want in {"even", "e"}:
            return "WIN" if not is_odd else "LOSE"
        return "UNKNOWN"

    # Unknown market
    return "UNKNOWN"



# -------------------- settle a whole ticket and persist --------------------

@transaction.atomic
def settle_two_odds_ticket_for_date(
    ticket_date: Date,
    *,
    league_id: int = TWO_ODDS_LEAGUE_ID,
    treat_unknown_as_void: bool = True,
) -> Optional[DailyTicket]:
    """
    Settle the TWO-ODDS ticket for a given UTC date and league_id.
    - Updates each leg's 'result' in selections.
    - Saves overall DailyTicket.status: 'won' | 'lost' | 'pending' | 'void' (lowercase to match model choices).
    - Returns the updated DailyTicket, or None if not found.
    """
    dt = (DailyTicket.objects
          .filter(ticket_date=ticket_date, league_id=league_id)
          .order_by("-id")
          .first())
    if not dt:
        return None

    changed = False
    any_pending = False
    any_lose = False
    any_resolved = False

    new_selections: List[dict] = []
    for sel in (dt.selections or []):
        mid = sel.get("match_id")
        market = sel.get("market") or sel.get("market_code")
        spec = sel.get("specifier") or ""

        m = Match.objects.select_related("home", "away").filter(id=mid).first()
        if not m:
            outcome = "VOID"
        else:
            outcome = evaluate_selection_outcome(m, market, spec)
            if outcome == "UNKNOWN" and treat_unknown_as_void:
                outcome = "VOID"

        prev = sel.get("result")
        if prev != outcome:
            sel["result"] = outcome
            changed = True
        else:
            # normalize default
            sel["result"] = outcome or prev or "PENDING"

        # flags
        if sel["result"] == "PENDING":
            any_pending = True
        elif sel["result"] == "LOSE":
            any_lose = True
            any_resolved = True
        elif sel["result"] in {"WIN", "VOID"}:
            any_resolved = True

        new_selections.append(sel)

    # derive ticket status (lowercase to match your model choices)
    if any_lose:
        new_status = "lost"
    elif any_pending:
        new_status = "pending"
    elif any_resolved:
        new_status = "won"
    else:
        new_status = "pending"

    if changed or new_status != dt.status:
        dt.selections = new_selections
        dt.status = new_status
        dt.save(update_fields=["selections", "status", "updated_at"])

    return dt












# matches/utils.py
from dataclasses import dataclass
from datetime import date as Date
import math, random

# assumes you already have: _fetch_two_odds_candidates and _bookish_odds_for_prob
# and your TWO_ODDS_LEAGUE_ID constant (if not, set TWO_ODDS_LEAGUE_ID = -1)

@dataclass
class BuiltTicket:
    selections: list[dict]
    legs: int
    acc_probability: float
    acc_fair_odds: float
    acc_bookish_odds: float

def build_two_odds_ticket_preview(
    ticket_date: Date,
    *,
    target_odds: float = 2.0,
    over_tolerance: float = 0.10,   # allow going above target by +10%
    min_legs: int = 2,
    max_legs: int = 6,
    min_p: float = 0.60,
    max_fair_odds: float = 1.60,
    attempts: int = 500,
) -> BuiltTicket:
    cands = _fetch_two_odds_candidates(
        ticket_date, min_p=min_p, max_fair_odds=max_fair_odds
    )
    if not cands:
        raise ValueError("No eligible candidates for this date with the given filters.")

    # Never go below target; allow slight overshoot up to target*(1+over_tolerance)
    target_low  = target_odds
    target_high = target_odds * (1.0 + over_tolerance)

    rnd = random.Random(int(ticket_date.strftime("%Y%m%d")))  # deterministic per day
    base = list(cands)

    best = []
    best_diff = float("inf")

    for _ in range(int(attempts)):
        rnd.shuffle(base)
        acc = []
        prod = 1.0

        # greedily grow to at least min_legs
        for c in base:
            if len(acc) >= int(max_legs):
                break
            new_prod = prod * c.book_odds
            acc.append(c)
            prod = new_prod
            if len(acc) >= int(min_legs) and (target_low <= prod <= target_high):
                break

        if len(acc) < int(min_legs):
            continue

        # nudge closer if still out of band and room remains
        if not (target_low <= prod <= target_high):
            cur_diff = abs(math.log(prod) - math.log(target_odds))
            for c in base:
                if c in acc or len(acc) >= int(max_legs):
                    continue
                trial = prod * c.book_odds
                trial_diff = abs(math.log(trial) - math.log(target_odds))
                if trial_diff < cur_diff:
                    acc.append(c)
                    prod = trial
                    cur_diff = trial_diff
                    if target_low <= prod <= target_high:
                        break

        diff = abs(math.log(prod) - math.log(target_odds))
        if len(acc) >= int(min_legs) and diff < best_diff:
            best_diff = diff
            best = acc[:]
            if target_low <= prod <= target_high:
                break

    if len(best) < int(min_legs):
        raise ValueError(
            f"Could not assemble a ≥{min_legs}-leg ticket around ~{target_odds:.2f}."
        )

    # Accumulate metrics
    acc_p, acc_fair, acc_book = 1.0, 1.0, 1.0
    for c in best:
        acc_p *= max(1e-9, min(1.0 - 1e-9, c.p))
        acc_fair *= c.fair_odds
        acc_book *= c.book_odds

    selections = [{
        "pm_id": getattr(c, "pm_id", None),
        "match_id": c.match_id,
        "league_id": c.league_id,
        "kickoff_utc": c.kickoff_utc.isoformat(),
        "home": c.home,
        "away": c.away,
        "market": c.market_code,
        "specifier": c.specifier,
        "p": c.p,
        "fair_odds": c.fair_odds,
    } for c in best]

    return BuiltTicket(
        selections=selections,
        legs=len(selections),
        acc_probability=float(acc_p),
        acc_fair_odds=float(acc_fair),
        acc_bookish_odds=float(acc_book),
    )
