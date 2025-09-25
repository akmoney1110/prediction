# matches/tickets.py

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import random
from typing import Iterable, Optional, Sequence

from django.db.models import Prefetch

from .models import Match, PredictedMarket

# Which markets are allowed to be "top pick" candidates:
TOP_MARKET_WHITELIST: set[str] = {
    "1X2", "OU", "BTTS", "TEAM_TOTAL",
    "OU_CORNERS", "TEAM_TOTAL_CORNERS",
    "CARDS_TOT", "YELLOWS_TOT", "REDS_TOT",
}

def _approx_book_odds(p: float, margin: float = 0.06) -> float:
    """
    Return a bookmaker-like decimal odds for probability p by embedding a small margin.
    We bump the probability a bit (overround) -> odds go down slightly vs fair.
    """
    if p <= 0.0:
        return 999.0
    p_bk = min(0.999, p * (1.0 + max(0.0, margin)))
    return float(1.0 / p_bk)

def _top_pick(pm_iter: Iterable[PredictedMarket]) -> Optional[PredictedMarket]:
    """
    Highest p_model row from the allowed markets.
    """
    best = None
    best_p = -1.0
    for pm in pm_iter:
        if pm.market_code not in TOP_MARKET_WHITELIST:
            continue
        if pm.p_model is None:
            continue
        p = float(pm.p_model)
        if p > best_p:
            best = pm
            best_p = p
    return best

def build_daily_ticket(
    league_id: int,
    day_offset: int = 0,
    *,
    count: int = 5,
    min_p: float = 0.55,
    rng_seed: Optional[int] = None,
    include_live_status: bool = True,
) -> list[dict]:
    """
    Build a random ticket of `count` selections for `league_id` on day_offset (0=Today, 1=Tomorrow, ...).

    - Picks max 1 selection per match (the highest probability across a whitelist).
    - Randomly samples matches when more than `count` are eligible.
    - If too few eligible picks meet `min_p`, it gradually relaxes the threshold (0.53, 0.50) to fill.

    Returns list of dicts:
    {
      "match_id": int,
      "kickoff_utc": datetime,
      "home": str,
      "away": str,
      "market": str,
      "specifier": str,
      "p": float,
      "fair_odds": float,
      "bookish_odds": float
    }
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    now = datetime.now(timezone.utc)
    start = (now + timedelta(days=max(0, min(day_offset, 7)))).replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)

    # statuses we consider “available” for a ticket
    base_statuses = ["NS", "TBD", "PST"]
    if include_live_status:
        base_statuses += ["1H", "HT", "2H", "LIVE"]

    # pull predicted markets ordered by probability (helps our top-pick scan)
    pm_qs = (PredictedMarket.objects
             .filter(league_id=league_id, kickoff_utc__gte=start, kickoff_utc__lt=end)
             .order_by("-p_model"))

    # fetch matches + markets
    matches = (Match.objects
               .filter(league_id=league_id,
                       kickoff_utc__gte=start,
                       kickoff_utc__lt=end,
                       status__in=base_statuses)
               .select_related("home", "away")
               .prefetch_related(Prefetch("predicted_markets", queryset=pm_qs))
               .order_by("kickoff_utc"))

    # collect candidate picks (one per match)
    candidates: list[dict] = []
    for m in matches:
        pm = _top_pick(m.predicted_markets.all())
        if not pm:
            continue
        p = float(pm.p_model)
        candidates.append({
            "match_id": m.id,
            "kickoff_utc": m.kickoff_utc,
            "home": m.home.name,
            "away": m.away.name,
            "market": pm.market_code,
            "specifier": pm.specifier,
            "p": p,
            "fair_odds": float(1.0 / max(1e-9, p)),
            "bookish_odds": _approx_book_odds(p, margin=0.06),
        })

    if not candidates:
        return []

    # try to enforce min_p; if not enough, relax thresholds
    thresholds: Sequence[float] = (min_p, 0.53, 0.50, 0.0)
    selected: list[dict] = []
    for th in thresholds:
        pool = [c for c in candidates if c["p"] >= th]
        if len(pool) >= count:
            selected = random.sample(pool, count)
            break
        # keep the best we can if we hit the last threshold
        if th == thresholds[-1]:
            # pad with the highest p overall
            pool_sorted = sorted(candidates, key=lambda x: x["p"], reverse=True)
            selected = pool_sorted[:min(count, len(pool_sorted))]
            break

    # optional: shuffle ticket order for variety
    random.shuffle(selected)
    return selected
