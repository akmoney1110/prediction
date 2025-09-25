# matches/utils_preview.py  (new file to keep it tidy)
import math, random
from dataclasses import dataclass
from datetime import date as Date

from .utils import _fetch_two_odds_candidates  # you already have this

@dataclass
class PreviewTicket:
    selections: list[dict]
    legs: int
    acc_probability: float
    acc_fair_odds: float
    acc_bookish_odds: float

def build_two_odds_ticket_preview(
    *,
    ticket_date: Date,
    target_odds: float = 2.0,
    over_tolerance: float = 0.15,   # allow up to +15% over target
    min_legs: int = 2,
    max_legs: int = 6,
    min_p: float = 0.60,
    max_fair_odds: float = 1.60,
    attempts: int = 500,
) -> PreviewTicket:
    cands = _fetch_two_odds_candidates(
        ticket_date,
        min_p=min_p,
        max_fair_odds=max_fair_odds,
        max_count=400,
    )
    if not cands:
        raise ValueError("No eligible candidates for this date & filters.")

    target_low = target_odds  # never go below target
    target_high = target_odds * (1.0 + over_tolerance)

    base = list(cands)
    best: list = []
    best_diff = float("inf")

    rnd = random.Random(int(ticket_date.strftime("%Y%m%d")))

    for _ in range(int(attempts)):
        rnd.shuffle(base)
        acc = []
        prod = 1.0

        for c in base:
            if len(acc) >= int(max_legs):
                break
            new_prod = prod * c.book_odds

            # grow freely until min_legs
            if len(acc) >= int(min_legs):
                # avoid huge overshoots (still permit small)
                if new_prod > target_high * 1.25:
                    continue

            acc.append(c)
            prod = new_prod

            # accept only when we have at least min_legs and prod is in [target, target_high]
            if len(acc) >= int(min_legs) and target_low <= prod <= target_high:
                break

        # if we didn't cross target yet and can still add, try nudging closer
        if len(acc) >= int(min_legs) and not (target_low <= prod <= target_high):
            for c in base:
                if c in acc or len(acc) >= int(max_legs):
                    continue
                trial = prod * c.book_odds
                if trial <= target_high and trial >= prod:  # move up but stay under cap
                    acc.append(c)
                    prod = trial
                    if target_low <= prod <= target_high:
                        break

        if len(acc) < int(min_legs):
            continue

        # score proximity in log-space (closer to target_odds is better; never below target)
        if prod >= target_low:
            diff = abs(math.log(prod) - math.log(target_odds))
            if diff < best_diff:
                best_diff = diff
                best = acc[:]
                if target_low <= prod <= target_high:
                    break

    if not best:
        raise ValueError("Could not assemble a ticket within target range; broaden filters or raise tolerance.")

    # accumulate metrics
    acc_p = 1.0
    acc_fair = 1.0
    acc_book = 1.0
    sels = []
    for c in best:
        acc_p *= max(1e-9, min(1.0 - 1e-9, c.p))
        acc_fair *= c.fair_odds
        acc_book *= c.book_odds
        sels.append({
            "match_id": c.match_id,
            "league_id": c.league_id,
            "kickoff_utc": c.kickoff_utc.isoformat(),
            "home": c.home,
            "away": c.away,
            "market": c.market_code,
            "specifier": c.specifier,
            "p": c.p,
            "fair_odds": c.fair_odds,
        })

    return PreviewTicket(
        selections=sels,
        legs=len(sels),
        acc_probability=acc_p,
        acc_fair_odds=acc_fair,
        acc_bookish_odds=acc_book,
    )
