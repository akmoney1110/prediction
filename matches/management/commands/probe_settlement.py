# matches/management/commands/probe_settlement.py

from django.core.management.base import BaseCommand
from django.db.models import Model
import json
from typing import Any, List, Tuple

from matches.models import Match, MatchStats  # adjust import if your app name differs


HT_ATTRS_HOME = [
    "goals_home_ht","ht_goals_home","home_ht_goals","ht_home",
    "half_time_home","halftime_home","home_goals_ht","score_ht_home",
]
HT_ATTRS_AWAY = [
    "goals_away_ht","ht_goals_away","away_ht_goals","ht_away",
    "half_time_away","halftime_away","away_goals_ht","score_ht_away",
]
HT_COMBINED = ["ht_score","score_ht","halftime","half_time"]

GOAL_MINUTE_PAIRS = [
    ("goal_minutes_home","goal_minutes_away"),
    ("goals_time_home","goals_time_away"),
    ("home_goal_times","away_goal_times"),
    ("home_goals_minutes","away_goals_minutes"),
    ("home_goal_minutes","away_goal_minutes"),
    ("scored_minutes_home","scored_minutes_away"),
    ("goals_home_minutes","goals_away_minutes"),
]

EVENT_FIELDS = ["events","timeline","match_events"]

STATS_ALIAS_MAP = {
    "corners": ["corners","corner","corner_kicks","corners_total"],
    "cards":   ["cards","card_total","cards_total","total_cards"],
    "yellows": ["yellows","yellow_cards","yellowcards","yellows_total","yellow_total"],
    "reds":    ["reds","red_cards","redcards","reds_total","red_total","rcards"],
}

def _get_attr_any(obj: Any, names: List[str]):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None and (not isinstance(v, str) or v.strip() != ""):
                return n, v
    return None, None

def _print_val(label: str, val: Any):
    try:
        print(f"  - {label}: {val!r}")
    except Exception:
        print(f"  - {label}: <unprintable>")

def _peek_events(ev_blob):
    """Return (found, first_two) where found=True if we can detect goal-like entries."""
    if ev_blob is None:
        return False, None
    evs = ev_blob
    if isinstance(evs, str):
        try:
            evs = json.loads(evs)
        except Exception:
            return False, f"<string-not-json len={len(ev_blob)}>"
    if isinstance(evs, dict):
        evs = evs.get("events") or evs.get("timeline") or []
    if not isinstance(evs, list):
        return False, f"<unexpected type {type(evs)}>"
    # detect goal-ish
    goals = []
    for e in evs:
        t = (e.get("type") or e.get("detail") or "").lower()
        if "goal" in t or t in {"goal","penalty_goal","own_goal"}:
            goals.append(e)
    return (len(goals) > 0, goals[:2])

class Command(BaseCommand):
    help = "Probe a match’s DB data relevant to settlement (HT, goal minutes, events, MatchStats aliases)."

    def add_arguments(self, parser):
        parser.add_argument("--match-id", type=int, required=True)

    def handle(self, *args, **opts):
        mid = opts["match_id"]
        m = Match.objects.select_related("home","away").filter(id=mid).first()
        if not m:
            self.stderr.write(self.style.ERROR(f"Match id={mid} not found"))
            return

        print(f"=== MATCH #{mid} ===")
        print(f"status={getattr(m,'status',None)!r}, FT=({getattr(m,'goals_home',None)!r}-{getattr(m,'goals_away',None)!r})")
        print(f"home={getattr(m,'home',None)}, away={getattr(m,'away',None)}")

        # Halftime fields
        print("\n-- Halftime fields --")
        name_h, val_h = _get_attr_any(m, HT_ATTRS_HOME)
        name_a, val_a = _get_attr_any(m, HT_ATTRS_AWAY)
        name_c, val_c = _get_attr_any(m, HT_COMBINED)
        if name_h: _print_val(name_h, val_h)
        if name_a: _print_val(name_a, val_a)
        if name_c: _print_val(name_c, val_c)
        if not (name_h or name_a or name_c):
            print("  (none found)")

        # Goal minutes pairs
        print("\n-- Goal minutes fields --")
        found_any = False
        for h_attr, a_attr in GOAL_MINUTE_PAIRS:
            h = getattr(m, h_attr, None)
            a = getattr(m, a_attr, None)
            if h is not None or a is not None:
                found_any = True
                _print_val(h_attr, h)
                _print_val(a_attr, a)
        if not found_any:
            print("  (none found on Match)")

        # Events-like blobs
        print("\n-- Events/timeline fields --")
        ev_name, ev_val = _get_attr_any(m, EVENT_FIELDS)
        if ev_name:
            print(f"  - {ev_name}: present (type={type(ev_val).__name__})")
            ok, peek = _peek_events(ev_val)
            print(f"    goals_detectable={ok}, sample={peek!r}")
        else:
            print("  (no events/timeline field present)")

        # MatchStats rows
        print("\n-- MatchStats by team --")
        stats = list(MatchStats.objects.filter(match=m).select_related("team"))
        if not stats:
            print("  (no MatchStats rows)")
        else:
            for r in stats:
                tid = getattr(r.team, "id", None)
                tname = getattr(r.team, "name", None)
                print(f"  team_id={tid}, name={tname!r}")
                for key, aliases in STATS_ALIAS_MAP.items():
                    found = []
                    for nm in aliases:
                        if hasattr(r, nm):
                            v = getattr(r, nm)
                            if v is not None:
                                found.append((nm, v))
                    if found:
                        print(f"    {key}: " + ", ".join(f"{nm}={v}" for nm, v in found))
                    else:
                        print(f"    {key}: (none)")

        print("\nDone.")
