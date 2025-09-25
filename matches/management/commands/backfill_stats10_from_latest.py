# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
from django.core.management.base import BaseCommand, CommandParser
from django.db.models import Q

from matches.models import Match, MatchPrediction, MLTrainingMatch


def _safe_json_loads(x) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def _latest_stats10_for_team(team_id: int) -> Optional[Dict[str, Any]]:
    """
    Take the most recent finished MLTrainingMatch that involves this team.
    Return an oriented "slice" of team/allowed/derived suitable for composing
    a nested stats10_json for an upcoming match.
    """
    row = (
        MLTrainingMatch.objects
        .filter(Q(home_team_id=team_id) | Q(away_team_id=team_id))
        .filter(~Q(stats10_json=None))
        .order_by("-kickoff_utc")
        .values("home_team_id", "away_team_id", "stats10_json")
        .first()
    )
    if not row:
        return None

    js = _safe_json_loads(row["stats10_json"]) or {}
    shots   = js.get("shots") or {}
    allowed = js.get("allowed") or {}
    derived = js.get("derived") or {}
    cross   = js.get("cross") or {}
    situ    = js.get("situational") or {}

    elo_home = js.get("elo_home"); elo_away = js.get("elo_away")
    gelo_mu_home = js.get("gelo_mu_home"); gelo_mu_away = js.get("gelo_mu_away")

    was_home = int(row["home_team_id"]) == int(team_id)
    side = "home" if was_home else "away"

    out = {
        "team":    (shots.get(side) or {}) if isinstance(shots, dict) else {},
        "allowed": (allowed.get(side) or {}) if isinstance(allowed, dict) else {},
        "derived": (derived.get(side) or {}) if isinstance(derived, dict) else {},
        "extras": {},
        "cross_ref": cross or {},
        "situ_ref":  situ or {},
    }
    # carry simple rating hints if available
    if elo_home is not None and elo_away is not None:
        out["extras"]["elo"] = float(elo_home if was_home else elo_away)
    if gelo_mu_home is not None and gelo_mu_away is not None:
        try:
            out["extras"]["gelo_mu"] = float(gelo_mu_home if was_home else gelo_mu_away)
        except Exception:
            pass
    return out


def _compose_stats10_for_match(home_team_id: int, away_team_id: int) -> Optional[Dict[str, Any]]:
    """Compose a nested stats10_json from each team’s latest slice."""
    hs = _latest_stats10_for_team(int(home_team_id))
    as_ = _latest_stats10_for_team(int(away_team_id))
    if hs is None or as_ is None:
        return None

    shots = {"home": hs["team"], "away": as_["team"]}
    allowed = {"home": hs["allowed"], "away": as_["allowed"]}
    derived = {"home": hs["derived"], "away": as_["derived"]}

    # minimal cross/situ if we can’t derive sensibly
    cross = {
        "home_xgps_minus_away_sot_allow_rate": 0.0,
        "away_xgps_minus_home_sot_allow_rate": 0.0,
    }
    situ = {
        "h_rest_days": 0.0, "a_rest_days": 0.0,
        "h_matches_14d": 0.0, "a_matches_14d": 0.0,
        "h_matches_7d": 0.0,  "a_matches_7d": 0.0,
    }

    js = {
        "shots": shots,
        "allowed": allowed,
        "derived": derived,
        "cross": cross,
        "situational": situ,
    }

    # pass rating-like hints if both sides have them
    if "extras" in hs and "extras" in as_:
        if "elo" in hs["extras"] and "elo" in as_["extras"]:
            js["elo_home"] = float(hs["extras"]["elo"])
            js["elo_away"] = float(as_["extras"]["elo"])
        if "gelo_mu" in hs["extras"] and "gelo_mu" in as_["extras"]:
            js["gelo_mu_home"] = float(hs["extras"]["gelo_mu"])
            js["gelo_mu_away"] = float(as_["extras"]["gelo_mu"])

    return js


def _match_has_stats10(m: Match) -> bool:
    # 1) dedicated stats10_json on Match (if your schema has it)
    if _safe_json_loads(getattr(m, "stats10_json", None)):
        return True
    # 2) raw_result_json.stats10_json
    raw = getattr(m, "raw_result_json", None)
    if isinstance(raw, dict) and _safe_json_loads(raw.get("stats10_json")):
        return True
    return False


class Command(BaseCommand):
    help = "Backfill Match.stats10_json (or raw_result_json['stats10_json']) for upcoming fixtures from each team’s latest finished MLTrainingMatch."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--write", action="store_true")
        parser.add_argument("--verbose", action="store_true")

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        days = int(opts["days"])
        do_write = bool(opts["write"])
        verbose = bool(opts["verbose"])

        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)

        qs = (
            MatchPrediction.objects
            .filter(
                league_id=league_id,
                kickoff_utc__gte=now,
                kickoff_utc__lte=upto,
                match__status__in=["NS", "PST", "TBD"],
            )
            .select_related("match", "match__home", "match__away")
            .order_by("kickoff_utc")
        )

        n_have = n_make = n_skip = 0

        # detect if Match has a concrete stats10_json field
        match_field_names = {f.name for f in Match._meta.get_fields() if hasattr(f, "attname")}
        has_match_stats10_field = "stats10_json" in match_field_names

        for mp in qs:
            m: Match = mp.match
            if _match_has_stats10(m):
                n_have += 1
                if verbose:
                    self.stdout.write(f"{m.id} | already has stats10_json → keep")
                continue

            js = _compose_stats10_for_match(m.home_id, m.away_id)
            if js is None:
                n_skip += 1
                if verbose:
                    self.stdout.write(f"{m.id} | {getattr(m.home, 'name', m.home_id)} vs {getattr(m.away, 'name', m.away_id)} → cannot compose (missing slices)")
                continue

            if do_write:
                if has_match_stats10_field:
                    # write to Match.stats10_json (JSONField)
                    m.stats10_json = js
                    m.save(update_fields=["stats10_json"])
                else:
                    # write under Match.raw_result_json['stats10_json']
                    raw = getattr(m, "raw_result_json", None)
                    if not isinstance(raw, dict):
                        raw = {}
                    raw = dict(raw)  # copy
                    raw["stats10_json"] = js
                    m.raw_result_json = raw
                    m.save(update_fields=["raw_result_json"])
            n_make += 1

            if verbose:
                hn = getattr(m.home, "name", str(m.home_id))
                an = getattr(m.away, "name", str(m.away_id))
                where = "Match.stats10_json" if has_match_stats10_field else "Match.raw_result_json['stats10_json']"
                self.stdout.write(f"{m.id} | backfilled → {where} ({hn} vs {an})")

        msg = f"Backfill finished. had={n_have}, created={n_make}, skipped={n_skip}."
        if do_write:
            self.stdout.write(self.style.SUCCESS(msg))
        else:
            self.stdout.write(msg + " Dry-run; use --write to persist.")
