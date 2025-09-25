# matches/management/commands/ingest_league_plus_extras.py
import re
import logging
from datetime import datetime, date

from django.core.management.base import BaseCommand
from django.db import transaction
from django.db import IntegrityError

from leagues.models import League, Team
from matches.models import (
    Match, Venue, MatchStats, Player, Lineup,
    StandingsRow, PlayerSeason, Transfer, Injury, Trophy,
    MatchEvent,
)

from services import apifootball as api

logger = logging.getLogger(__name__)

# ---------- Stat normalization (covers common API-Football labels) ----------
def _norm_key(s): return (s or "").strip().lower()

STAT_ALIASES = {
    # shots
    "total shots": "shots", "shots total": "shots",
    "shots on goal": "sot", "shots on target": "sot", "shots on target ": "sot",
    "shots off goal": "shots_off", "shots off target": "shots_off",
    "blocked shots": "shots_blocked",
    "shots insidebox": "shots_in_box", "shots inside box": "shots_in_box",
    "shots outsidebox": "shots_out_box", "shots outside box": "shots_out_box",
    # discipline & misc
    "fouls": "fouls",
    "corner kicks": "corners", "corners": "corners",
    "offsides": "offsides",
    "yellow cards": "yellows", "red cards": "reds",
    # possession / passing / keeper
    "ball possession": "possession_pct", "possession": "possession_pct",
    "passes %": "pass_acc_pct", "pass accuracy": "pass_acc_pct",
    "goalkeeper saves": "saves",
    "total passes": "passes_total",
    "passes accurate": "passes_accurate",
    # models / xG
    "xg": "xg", "expected_goals": "xg",
}

def _as_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("%", "")
    try:
        return float(s)
    except Exception:
        return None

def _as_int(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    try:
        return int(str(v).strip())
    except Exception:
        return None

def _field_names(model_cls):
    names = set()
    for f in model_cls._meta.get_fields():
        # Skip reverse relations
        if getattr(f, "auto_created", False) and not f.concrete:
            continue
        if not getattr(f, "concrete", True):
            continue
        names.add(f.name)
    return names

def _filtered_defaults(model_cls, defaults: dict) -> dict:
    allowed = _field_names(model_cls)
    return {k: v for k, v in defaults.items() if k in allowed}


class Command(BaseCommand):
    help = (
        "Ingest fixtures + per-match stats/lineups/events for one league+season, "
        "and optionally standings, players, transfers, venues, injuries, trophies."
    )

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True, help="API-Football league id")
        parser.add_argument("--season", type=int, required=True, help="Season start year, e.g., 2024")
        parser.add_argument(
    "--players-fixture-only",
    action="store_true",
    help="Ingest players only for teams that actually appear as home/away in this season's fixtures",
)


        # Extras (toggle as needed)
        parser.add_argument("--standings", action="store_true", help="Ingest standings/table")
        parser.add_argument("--players", action="store_true", help="Ingest players & PlayerSeason (paged)")
        parser.add_argument("--transfers", action="store_true", help="Ingest transfers per team")
        parser.add_argument("--venues", action="store_true", help="Ingest full venues referenced by fixtures")
        parser.add_argument("--injuries", action="store_true", help="Ingest injuries for league+season")
        parser.add_argument("--trophies", action="store_true", help="Ingest trophies per player (requires --players)")

    def handle(self, *args, **opts):
        league_id = opts["league_id"]
        season = opts["season"]
        self.league_id = league_id  # available to helpers

        League.objects.get_or_create(
            id=league_id,
            defaults={
                "display_name": f"League {league_id}",
                "country": "Unknown",
                "tier": 1,
                "has_xg": False,
                "has_multi_group": False,
            },
        )

        self.stdout.write(self.style.NOTICE(f"Fetching fixtures league={league_id}, season={season}"))
        fixtures = self._fetch_all_fixtures(league_id, season)
        self.stdout.write(f"  Got {len(fixtures)} fixtures after fallbacks")

        # Upsert fixtures (also collect venue ids)
        venue_ids = set()
        count_inserted = 0
        for fx in fixtures:
            try:
                vid = self._upsert_fixture_block(fx)
                if vid:
                    venue_ids.add(vid)
                count_inserted += 1
            except Exception:
                logger.exception("Failed to upsert fixture %s", fx.get("fixture", {}).get("id"))
        self.stdout.write(self.style.SUCCESS(f"Saved/updated {count_inserted} fixtures"))
        self.stdout.write(self.style.NOTICE("Fetching per-fixture details (stats, lineups, events)..."))

        # Per-fixture details
        fixture_ids = list(
            Match.objects.filter(league_id=league_id, season=season).values_list("id", flat=True)
        )
        for fid in fixture_ids:
            try:
                self._ingest_fixture_details(fid)
            except Exception:
                logger.exception("Failed details for fixture %s", fid)
        self.stdout.write(self.style.SUCCESS("Per-match details complete."))

        # ---- Supervision: ensure FT matches actually have events ----
        # Re-fetch events for FT matches that currently have none.
        missing_evs = list(
            Match.objects.filter(
                league_id=league_id, season=season, status__iexact="FT"
            ).exclude(raw_result_json__has_key="events")
             | Match.objects.filter(
                league_id=league_id, season=season, status__iexact="FT",
                raw_result_json__events__len=0  # works on Postgres JSONB via __len
            ).values_list("id", flat=True)
        )  # noqa

        # Some DBs may not support __events__len; fall back defensively:
        if not missing_evs:
            missing_evs = [
                mid for mid in fixture_ids
                if Match.objects.filter(pk=mid, status__iexact="FT").exists()
                and len((Match.objects.get(pk=mid).raw_result_json or {}).get("events", []) or []) == 0
            ]

        retried, fixed = 0, 0
        for fid in missing_evs:
            retried += 1
            try:
                fixed |= int(self._refetch_events_only(fid))
            except Exception:
                logger.exception("Retry events failed for fixture %s", fid)
        if retried:
            self.stdout.write(self.style.NOTICE(f"Events retry pass: retried={retried}, fixed={fixed}"))

        # ---- Extras ----
        if opts["standings"]:
            self.stdout.write(self.style.NOTICE("Ingesting standings..."))
            try:
                self._ingest_standings(league_id, season)
                self.stdout.write(self.style.SUCCESS("Standings done."))
            except Exception:
                logger.exception("Standings ingest failed")

        team_ids = list(Team.objects.filter(league_id=league_id).values_list("id", flat=True))

        if opts["players"]:
            self.stdout.write(self.style.NOTICE("Ingesting players (paged per team)..."))
            for tid in team_ids:
                try:
                    self._ingest_players_for_team(tid, season, league_id)
                except Exception:
                    logger.exception("Players ingest failed for team %s", tid)
            self.stdout.write(self.style.SUCCESS("Players done."))

        if opts["transfers"]:
            self.stdout.write(self.style.NOTICE("Ingesting transfers per team..."))
            for tid in team_ids:
                try:
                    self._ingest_transfers_for_team(tid)
                except Exception:
                    logger.exception("Transfers ingest failed for team %s", tid)
            self.stdout.write(self.style.SUCCESS("Transfers done."))

        if opts["venues"] and venue_ids:
            self.stdout.write(self.style.NOTICE("Ingesting venues..."))
            for vid in sorted(venue_ids):
                try:
                    self._ingest_venue(vid)
                except Exception:
                    logger.exception("Venue ingest failed for venue %s", vid)
            self.stdout.write(self.style.SUCCESS("Venues done."))

        if opts["injuries"]:
            self.stdout.write(self.style.NOTICE("Ingesting injuries (league+season)..."))
            try:
                self._ingest_injuries(league_id, season)
                self.stdout.write(self.style.SUCCESS("Injuries done."))
            except Exception:
                logger.exception("Injuries ingest failed")

        if opts["trophies"] and opts["players"]:
            self.stdout.write(self.style.NOTICE("Ingesting trophies for known players..."))
            player_ids = list(
                PlayerSeason.objects.filter(team_id__in=team_ids, season=season)
                .values_list("player_id", flat=True)
                .distinct()
            )
            for pid in player_ids:
                try:
                    self._ingest_trophies_for_player(pid)
                except Exception:
                    logger.exception("Trophies ingest failed for player %s", pid)
            self.stdout.write(self.style.SUCCESS("Trophies done."))

        self.stdout.write(self.style.SUCCESS("Ingestion complete."))
        # Collect team_ids from actual fixtures (home + away) for this league+season
        fixture_team_ids = set(
            Match.objects.filter(league_id=league_id, season=season)
            .values_list("home_id", flat=True)
        ) | set(
            Match.objects.filter(league_id=league_id, season=season)
            .values_list("away_id", flat=True)
        )
        fixture_team_ids = sorted(fixture_team_ids)


    # --------------------- helpers ---------------------
    def _parse_score_tuple(self, s):
        if not s:
            return None, None
        m = re.search(r"(\d+)\s*[-:]\s*(\d+)", str(s))
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

    def _set_if_has(self, obj, field, value):
        if hasattr(obj, field):
            setattr(obj, field, value)

    def _minutes_from_event(self, ev):
        if not isinstance(ev, dict):
            return None
        t = ev.get("time") or {}
        minute = t.get("elapsed", ev.get("minute"))
        extra  = t.get("extra", t.get("stoppage", t.get("extra_minute")))
        try:
            base = int(minute) if minute is not None else 0
            ext  = int(extra) if extra is not None else 0
            return base + ext
        except Exception:
            return None

    def _fetch_all_fixtures(self, league_id: int, season: int):
        data = api.fixtures_by_league_season(league_id, season)
        fixtures = data.get("response", []) or []
        if fixtures:
            return fixtures
        # fallback window
        start = f"{season}-07-01"; end = f"{season+1}-07-01"
        fixtures = api.fixtures_by_league_between(league_id, start, end).get("response", []) or []
        if fixtures:
            return fixtures
        # fallback monthly chunks
        chunked = []
        months = [("07","31"),("08","31"),("09","30"),("10","31"),("11","30"),("12","31"),
                  ("01","31"),("02","29"),("03","31"),("04","30"),("05","31"),("06","30")]
        for m,last in months:
            y = season if int(m) >= 7 else season+1
            frm = f"{y}-{m}-01"; to = f"{y}-{m}-{last}"
            resp = api.fixtures_by_league_between(league_id, frm, to).get("response", []) or []
            chunked.extend(resp)
        seen = set(); dedup = []
        for fx in chunked:
            fid = fx.get("fixture", {}).get("id")
            if fid and fid not in seen:
                seen.add(fid); dedup.append(fx)
        return dedup

    @staticmethod
    def _ensure_team_row(team_blob: dict, fallback_league_id: int):
        if not team_blob:
            return None
        tid = team_blob.get("id")
        if not tid:
            return None
        name = team_blob.get("name") or f"Team {tid}"
        Team.objects.get_or_create(
            id=tid,
            defaults={
                "league_id": fallback_league_id,
                "name": name,
                "short_name": name[:32],
                "logo_url": team_blob.get("logo"),
            },
        )
        return tid

    @transaction.atomic
    def _upsert_fixture_block(self, fx: dict):
        fixture = fx["fixture"]
        league = fx["league"]
        teams = fx["teams"]
        goals = fx.get("goals", {})

        fixture_id = fixture["id"]
        status = fixture["status"]["short"]
        kickoff_iso = fixture["date"]
        kickoff_utc = datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))

        # teams
        home_team = teams["home"]; away_team = teams["away"]
        home_id = home_team["id"]; away_id = away_team["id"]

        Team.objects.get_or_create(id=home_id, defaults={
            "league_id": league["id"],
            "name": home_team["name"],
            "short_name": home_team.get("name", "")[:32],
            "logo_url": home_team.get("logo"),
        })
        Team.objects.get_or_create(id=away_id, defaults={
            "league_id": league["id"],
            "name": away_team["name"],
            "short_name": away_team.get("name", "")[:32],
            "logo_url": away_team.get("logo"),
        })

        Match.objects.update_or_create(
            id=fixture_id,
            defaults={
                "league_id": league["id"],
                "season": league.get("season"),
                "home_id": home_id,
                "away_id": away_id,
                "kickoff_utc": kickoff_utc,
                "status": status,
                "goals_home": goals.get("home"),
                "goals_away": goals.get("away"),
                "raw_result_json": fx,
            },
        )

        # return venue id for later
        venue = fixture.get("venue") or {}
        return venue.get("id")

    @staticmethod
    def _coerce_year_from_season(season_val) -> int | None:
        """
        Accepts '2024', 2024, or '2015-2016' and returns the first year as int.
        """
        if season_val is None:
            return None
        s = str(season_val)
        m = re.search(r"\d{4}", s)
        return int(m.group(0)) if m else None

    @staticmethod
    def _parse_transfer_date(date_str: str | None, season_hint: int | None = None) -> date:
        """
        Robustly parse API-Football 'transfers[].date'. Returns a non-null date.
        Fallback to July 1 of season_hint (if provided) or today's date.
        """
        patterns = [
            "%Y-%m-%d", "%Y/%m/%d",
            "%d-%m-%Y", "%m-%d-%Y",
            "%d/%m/%Y", "%m/%d/%Y",
            "%d %b %Y", "%d %B %Y",
            "%Y%m%d",
        ]
        if date_str:
            s = str(date_str).strip()
            # try python's ISO parser first
            try:
                return datetime.fromisoformat(s).date()
            except Exception:
                pass
            for fmt in patterns:
                try:
                    return datetime.strptime(s, fmt).date()
                except Exception:
                    pass
            # handle compact digit-only forms
            digits = re.sub(r"\D+", "", s)
            if len(digits) == 6:
                d, m, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
                y += 2000 if y < 70 else 1900
                for (mm, dd) in ((m, d), (d, m)):
                    try:
                        return date(y, mm, dd)
                    except Exception:
                        pass
            if len(digits) == 8:  # e.g., 20160506
                y, m, d = int(digits[:4]), int(digits[4:6]), int(digits[6:])
                try:
                    return date(y, m, d)
                except Exception:
                    pass

        if season_hint and 1900 <= season_hint <= 2100:
            return date(season_hint, 7, 1)  # middle of off-season
        return date.today()

    def _fetch_players_leaguewide(self, league_id: int, season: int):
        """Fetch ALL players for a league+season across all pages once."""
        all_rows = []
        page = 1
        while True:
            data = api.players_by_league(league_id, season, page=page)
            resp = data.get("response", []) or []
            all_rows.extend(resp)
            paging = data.get("paging") or {}
            cur = int(paging.get("current") or 1)
            total = int(paging.get("total") or 1)
            if cur >= total or not resp:
                break
            page += 1
        return all_rows

    def _upsert_player_and_season(self, item: dict, team_id: int, season: int):
        """Create/Update Player + PlayerSeason from one /players row."""
        p = item.get("player") or {}
        if not p:
            return
        pid = p.get("id")
        # normalize height/weight
        height_cm = None
        if p.get("height"):
            try:
                height_cm = int(str(p["height"]).replace("cm", "").strip())
            except Exception:
                pass
        weight_kg = None
        if p.get("weight"):
            try:
                weight_kg = int(str(p["weight"]).replace("kg", "").strip())
            except Exception:
                pass

        Player.objects.update_or_create(
            id=pid,
            defaults={
                "name": p.get("name") or f"Player {pid}",
                "firstname": p.get("firstname"),
                "lastname": p.get("lastname"),
                "nationality": p.get("nationality"),
                "age": p.get("age"),
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "photo_url": p.get("photo"),
            },
        )

        stats_list = item.get("statistics") or []
        if not stats_list:
            return
        st = stats_list[0]
        games = st.get("games") or {}
        goals = st.get("goals") or {}
        shots = st.get("shots") or {}
        cards = st.get("cards") or {}

        # prefer numeric rating if present
        rating_raw = games.get("rating")
        try:
            rating = float(rating_raw) if rating_raw else None
        except Exception:
            rating = None

        PlayerSeason.objects.update_or_create(
            player_id=pid,
            team_id=team_id,
            season=season,
            defaults={
                "position": games.get("position"),
                "number": games.get("number"),
                "appearances": games.get("appearences"),
                "minutes": games.get("minutes"),
                "rating": rating,
                "goals": goals.get("total"),
                "assists": goals.get("assists"),
                "shots_total": shots.get("total"),
                "shots_on": shots.get("on"),
                "cards_yellow": cards.get("yellow"),
                "cards_red": cards.get("red"),
                "raw_json": st,
            },
        )

    def _ingest_fixture_details(self, fixture_id: int):
        # --- Statistics (team split) ---
        stats_data = api.fixture_statistics(fixture_id)
        stats_rows = 0
        for entry in stats_data.get("response", []) or []:
            try:
                team_id = entry["team"]["id"]
                raw_list = entry.get("statistics", []) or []

                parsed = {}
                for s in raw_list:
                    k = STAT_ALIASES.get(_norm_key(s.get("type")))
                    if not k:
                        continue
                    v = s.get("value")
                    parsed[k] = _as_float(v) if k in ("possession_pct", "pass_acc_pct", "xg") else _as_int(v)

                defaults = {
                    "shots": parsed.get("shots"),
                    "sot": parsed.get("sot"),
                    "shots_off": parsed.get("shots_off"),
                    "shots_blocked": parsed.get("shots_blocked"),
                    "shots_in_box": parsed.get("shots_in_box"),
                    "shots_out_box": parsed.get("shots_out_box"),
                    "fouls": parsed.get("fouls"),
                    "corners": parsed.get("corners"),
                    "offsides": parsed.get("offsides"),
                    "possession_pct": parsed.get("possession_pct"),
                    "pass_acc_pct": parsed.get("pass_acc_pct"),
                    "saves": parsed.get("saves"),
                    "passes_total": parsed.get("passes_total"),
                    "passes_accurate": parsed.get("passes_accurate"),
                    "yellows": parsed.get("yellows"),
                    "reds": parsed.get("reds"),
                    "cards": ((parsed.get("yellows") or 0) + (parsed.get("reds") or 0)
                              if (parsed.get("yellows") is not None and parsed.get("reds") is not None) else None),
                    "xg": parsed.get("xg"),
                    "stats_json": raw_list,
                }
                defaults = _filtered_defaults(MatchStats, defaults)

                MatchStats.objects.update_or_create(
                    match_id=fixture_id,
                    team_id=team_id,
                    defaults=defaults,
                )
                stats_rows += 1
            except Exception:
                continue

        # --- Lineups (optional) ---
        lu_data = api.fixture_lineups(fixture_id)
        for entry in lu_data.get("response", []) or []:
            try:
                defaults = {
                    "formation": entry.get("formation"),
                    "starters_json": entry.get("startXI") or [],
                    "bench_json": entry.get("substitutes") or [],
                }
                defaults = _filtered_defaults(Lineup, defaults)
                Lineup.objects.update_or_create(
                    match_id=fixture_id,
                    team_id=entry["team"]["id"],
                    defaults=defaults,
                )
            except Exception:
                continue

        # --- Events (with storage and row materialization) ---
        ev_data = api.fixture_events(fixture_id)
        match = Match.objects.select_related("home", "away").get(pk=fixture_id)

        # save raw events into raw_result_json
        raw = match.raw_result_json or {}
        events_raw_list = ev_data.get("response", []) or []
        raw["events"] = events_raw_list
        match.raw_result_json = raw
        match.save(update_fields=["raw_result_json"])

        # reset event rows to avoid dupes/stale
        MatchEvent.objects.filter(match_id=fixture_id).delete()

        home_id = getattr(match.home, "id", None)
        away_id = getattr(match.away, "id", None)

        def _mk_minute(ev):
            t = ev.get("time") or {}
            el = t.get("elapsed")
            ex = t.get("extra") or t.get("stoppage") or t.get("extra_minute")
            try:
                base = int(el) if el is not None else 0
            except Exception:
                base = 0
            try:
                ext = int(ex) if ex is not None else 0
            except Exception:
                ext = 0
            minute = base + ext if (el is not None or ex is not None) else None
            return minute, el, ex

        rows = []
        home_goal_minutes = []
        away_goal_minutes = []

        for ev in events_raw_list:
            try:
                team_blob   = ev.get("team") or {}
                player_blob = ev.get("player") or {}
                assist_blob = ev.get("assist") or {}

                team_id   = team_blob.get("id")
                player_id = player_blob.get("id")
                player_nm = player_blob.get("name")
                assist_id = assist_blob.get("id")
                assist_nm = assist_blob.get("name")

                ev_type   = (ev.get("type") or "").strip()     # Goal, Card, Subst, Var
                ev_detail = (ev.get("detail") or "").strip()   # Normal Goal, Own Goal, Penalty, etc.
                ev_comm   = (ev.get("comments") or "").strip() if ev.get("comments") else None

                minute, elapsed, extra = _mk_minute(ev)

                # flags
                is_home  = (team_id == home_id)
                is_var   = (ev_type.lower() == "var")
                dlow     = ev_detail.lower()
                is_own   = ("own goal" in dlow)
                is_pen   = (("penalty" in dlow) and ("missed" not in dlow) and ("confirmed" not in dlow) and ("cancelled" not in dlow))
                is_miss  = ("missed penalty" in dlow)

                if ev_type.lower() == "goal" and minute is not None:
                    if is_home:
                        home_goal_minutes.append(minute)
                    else:
                        away_goal_minutes.append(minute)

                rows.append(MatchEvent(
                    match_id=fixture_id,
                    team_id=team_id,
                    type=ev_type or None,
                    detail=ev_detail or None,
                    comment=ev_comm,
                    minute=minute,
                    elapsed=elapsed if isinstance(elapsed, int) else (int(elapsed) if elapsed is not None else None),
                    extra=extra if isinstance(extra, int) else (int(extra) if extra is not None else None),
                    player_id=player_id,
                    player_name=player_nm,
                    assist_id=assist_id,
                    assist_name=assist_nm,
                    is_home=is_home,
                    is_own_goal=is_own,
                    is_penalty=is_pen,
                    is_missed_penalty=is_miss,
                    is_var=is_var,
                    raw_json=ev,
                ))
            except Exception:
                logger.exception("Bad event row for fixture %s: %s", fixture_id, str(ev)[:300])
                continue

        if rows:
            MatchEvent.objects.bulk_create(rows, batch_size=200)

        # hydrate convenience goal-minute fields if present on Match
        upd = False
        if hasattr(match, "goal_minutes_home") and home_goal_minutes:
            match.goal_minutes_home = sorted(home_goal_minutes)
            upd = True
        if hasattr(match, "goal_minutes_away") and away_goal_minutes:
            match.goal_minutes_away = sorted(away_goal_minutes)
            upd = True
        if upd:
            match.save(update_fields=["goal_minutes_home", "goal_minutes_away"])

        # hydrate quick totals on Match if we have both teams' stats
        if stats_rows >= 2:
            try:
                h_stats = MatchStats.objects.get(match_id=fixture_id, team_id=match.home_id)
                a_stats = MatchStats.objects.get(match_id=fixture_id, team_id=match.away_id)
                changed = False
                if match.corners_home is None and h_stats.corners is not None and a_stats.corners is not None:
                    match.corners_home = h_stats.corners
                    match.corners_away = a_stats.corners
                    changed = True
                if match.cards_home is None and h_stats.cards is not None and a_stats.cards is not None:
                    match.cards_home = h_stats.cards
                    match.cards_away = a_stats.cards
                    changed = True
                if changed:
                    match.save(update_fields=["corners_home", "corners_away", "cards_home", "cards_away"])
            except MatchStats.DoesNotExist:
                pass

    def _refetch_events_only(self, fixture_id: int) -> bool:
        """
        Re-fetch just events for a given fixture, store to raw_result_json + rows.
        Returns True if we ended up with >0 events after the retry.
        """
        match = Match.objects.select_related("home", "away").get(pk=fixture_id)
        ev_data = api.fixture_events(fixture_id)
        events_raw_list = ev_data.get("response", []) or []

        raw = match.raw_result_json or {}
        raw["events"] = events_raw_list
        match.raw_result_json = raw
        match.save(update_fields=["raw_result_json"])

        # reset and re-insert
        MatchEvent.objects.filter(match_id=fixture_id).delete()

        if not events_raw_list:
            logger.warning("No events after retry for fixture %s (%s vs %s)", fixture_id, match.home_id, match.away_id)
            return False

        home_id = getattr(match.home, "id", None)
        rows = []
        for ev in events_raw_list:
            try:
                t = ev.get("time") or {}
                el = t.get("elapsed")
                ex = t.get("extra") or t.get("stoppage") or t.get("extra_minute")
                try:
                    base = int(el) if el is not None else 0
                except Exception:
                    base = 0
                try:
                    ext = int(ex) if ex is not None else 0
                except Exception:
                    ext = 0
                minute = base + ext if (el is not None or ex is not None) else None

                team_id = (ev.get("team") or {}).get("id")
                rows.append(MatchEvent(
                    match_id=fixture_id,
                    team_id=team_id,
                    type=(ev.get("type") or "").strip() or None,
                    detail=(ev.get("detail") or "").strip() or None,
                    minute=minute,
                    raw_json=ev,
                    is_home=True if team_id == home_id else False if team_id else None,
                ))
            except Exception:
                logger.exception("Bad event row (retry) for fixture %s: %s", fixture_id, str(ev)[:300])
        if rows:
            MatchEvent.objects.bulk_create(rows, batch_size=200)
        return True

    # ----------------- extras -----------------
    def _ingest_standings(self, league_id: int, season: int):
        data = api.standings(league_id, season)
        rows = self._dig_standings_list(data)
        StandingsRow.objects.filter(league_id=league_id, season=season).delete()
        for group in rows:
            gname = group.get("group")
            table = group.get("table") or []
            for row in table:
                team_id = row.get("team", {}).get("id") or row.get("team_id")
                if not team_id:
                    continue
                Team.objects.get_or_create(
                    id=team_id,
                    defaults={"league_id": league_id, "name": row.get("team", {}).get("name", f"Team {team_id}")},
                )
                defaults = {
                    "rank": row.get("rank") or row.get("position") or 0,
                    "played": row.get("all", {}).get("played") or row.get("played", 0),
                    "win": row.get("all", {}).get("win") or row.get("win", 0),
                    "draw": row.get("all", {}).get("draw") or row.get("draw", 0),
                    "loss": row.get("all", {}).get("lose") or row.get("loss", 0),
                    "gf": row.get("all", {}).get("goals", {}).get("for") or row.get("gf", 0),
                    "ga": row.get("all", {}).get("goals", {}).get("against") or row.get("ga", 0),
                    "gd": row.get("goalsDiff") or row.get("gd", 0),
                    "points": row.get("points") or 0,
                    "form": row.get("form"),
                    "last5_json": row.get("form_all", {}) or {},
                    "group_name": gname,
                }
                defaults = _filtered_defaults(StandingsRow, defaults)
                StandingsRow.objects.update_or_create(
                    league_id=league_id,
                    season=season,
                    team_id=team_id,
                    defaults=defaults,
                )

    def _dig_standings_list(self, payload: dict):
        if not payload:
            return []
        # response[0].league.standings (API-Football)
        if isinstance(payload, dict) and isinstance(payload.get("response"), list) and payload["response"]:
            first = payload["response"][0]
            if isinstance(first, dict):
                lg = first.get("league")
                if isinstance(lg, dict) and "standings" in lg:
                    st = lg["standings"]
                    if isinstance(st, list):
                        return [{"group": None, "table": grp} for grp in st]
        # already a table shape
        if isinstance(payload, dict) and "table" in payload:
            return [{"group": payload.get("group"), "table": payload.get("table") or []}]
        # best effort
        return [{"group": None, "table": payload.get("standings") or payload.get("table") or []}]

    def _ingest_players_for_team(self, team_id: int, season: int, league_id: int):
        """
        Populate Player and PlayerSeason for a specific team+season using API-Football, with 3 fallbacks:

         1) players_by_team(team, season, league) then players_by_team(team, season)
        2) league-wide players cache (pulled once) filtered by team_id
        3) squad list -> for each player, call players_by_id(player, season) (paginated)

        Notes:
        * Every loop/pagination has a clear stop condition to avoid infinite/long loops.
        * We log what path succeeded so audits are easier.
        * We add small sleeps to be gentle on rate limits.
        """
        import time

        log_prefix = f"[players] team={team_id} season={season}"

        # --- 1) Try team+league, then team-only (pagination) ---
        for include_league in (True, False):
            page = 1
            got_any = False
            while True:
                try:
                    if include_league:
                        data = api.players_by_team(team_id, season, page=page, league_id=league_id)
                    else:
                        data = api.players_by_team(team_id, season, page=page)
                except Exception as e:
                    logger.exception("%s team-by-team fetch failed (include_league=%s, page=%s)", log_prefix, include_league, page)
                    break  # break this include_league branch; try the next fallback

                resp = data.get("response", []) or []
                if resp:
                    for item in resp:
                        try:
                            self._upsert_player_and_season(item, team_id=team_id, season=season)
                            got_any = True
                        except Exception:
                            logger.exception("%s upsert failed from players_by_team", log_prefix)

                # paging stop condition
                paging = data.get("paging") or {}
                try:
                    cur = int(paging.get("current") or 1)
                    total = int(paging.get("total") or 1)
                except Exception:
                    cur, total = 1, 1

                # If no response rows OR last page reached -> stop paging
                if not resp or cur >= total:
                    break

                page += 1
                time.sleep(0.15)  # be nice to the API

            if got_any:
                logger.info("%s loaded via players_by_team (include_league=%s)", log_prefix, include_league)
                return  # success via path #1

        # --- 2) League-wide cache once, filter by team ---
        try:
            if not hasattr(self, "_league_players_cache"):
                logger.info("%s building league-wide player cache...", log_prefix)
                self._league_players_cache = self._fetch_players_leaguewide(league_id, season)
        except Exception:
            logger.exception("%s league-wide cache fetch failed", log_prefix)
            self._league_players_cache = []

        found = False
        for item in self._league_players_cache:
            stats = item.get("statistics") or []
            if not stats:
                continue
            # API-Football shape: statistics[0].team.id points to the team row for that season
            tinfo = stats[0].get("team") or {}
            if tinfo.get("id") == team_id:
                try:
                    self._upsert_player_and_season(item, team_id=team_id, season=season)
                    found = True
                except Exception:
                    logger.exception("%s upsert failed from league cache", log_prefix)

        if found:
            logger.info("%s loaded via league-wide cache", log_prefix)
            return  # success via path #2

        # --- 3) Final fallback: squad -> per-player stats ---
        try:
            squad_payload = api.players_squads(team_id)
        except Exception:
            logger.exception("%s squad fetch failed", log_prefix)
            squad_payload = {}

        squad = squad_payload.get("response", []) or []
        # API-Football usually returns: [{"team": {...}, "players": [ {...}, ... ]}]
        players = []
        if squad and isinstance(squad[0], dict) and "players" in squad[0]:
            players = squad[0].get("players") or []
        elif isinstance(squad, list):
            # Rare edge-case: already a flat list of players
            players = squad

        if not players:
            logger.warning("%s squad empty; no players to try via players_by_id()", log_prefix)
            return  # nothing else we can do

        # To avoid a long stream of "[API DEBUG] Empty response ...", we stop per-player
        # immediately if page=1 is empty; and we also cap pages just in case.
        MAX_PAGES_PER_PLAYER = 10

        processed = 0
        for p in players:
            pid = (p or {}).get("id")
            if not pid:
                continue

            # Ensure Player exists from the squad payload (lightweight upsert)
            try:
                Player.objects.get_or_create(
                    id=pid,
                    defaults={
                    "name": p.get("name") or f"Player {pid}",
                    "firstname": p.get("firstname"),
                    "lastname": p.get("lastname"),
                    "nationality": p.get("nationality"),
                    "photo_url": p.get("photo"),
                    },
                )
            except Exception:
                logger.exception("%s failed to ensure Player row (pid=%s) from squad payload", log_prefix, pid)

            # Now fetch their season stats and upsert PlayerSeason (paginated)
            page = 1
            while page <= MAX_PAGES_PER_PLAYER:
                try:
                    pdata = api.players_by_id(pid, season, page=page)
                except Exception:
                    logger.exception("%s players_by_id failed (pid=%s, page=%s)", log_prefix, pid, page)
                    break

                presp = pdata.get("response", []) or []

                if not presp:
                    # Nothing for this page. If it's the first page, stop early for this player.
                    if page == 1:
                        # This is expected for many players; just move on.
                        pass
                    break

                for item in presp:
                    try:
                        self._upsert_player_and_season(item, team_id=team_id, season=season)
                    except Exception:
                        logger.exception("%s upsert failed from players_by_id (pid=%s)", log_prefix, pid)

                # handle paging
                paging = pdata.get("paging") or {}
                try:
                    cur = int(paging.get("current") or 1)
                    total = int(paging.get("total") or 1)
                except Exception:
                    cur, total = 1, 1

                if cur >= total:
                    break

                page += 1
                time.sleep(0.12)

            processed += 1
            # small pause between players to be nice to API
            time.sleep(0.05)

        logger.info("%s finished via squad->players_by_id fallback (processed %d players)", log_prefix, processed)

    def _ingest_transfers_for_team(self, team_id: int):
        data = api.transfers_by_team(team_id)
        for row in data.get("response", []) or []:
            pinfo = row.get("player") or {}
            pid = pinfo.get("id")
            if not pid:
                continue

            # Ensure FK Player exists
            Player.objects.get_or_create(
                id=pid,
                defaults={
                    "name": pinfo.get("name") or f"Player {pid}",
                    "firstname": pinfo.get("firstname"),
                    "lastname": pinfo.get("lastname"),
                    "nationality": pinfo.get("nationality"),
                    "photo_url": pinfo.get("photo"),
                    "raw_json": pinfo,
                },
            )

            for tr in row.get("transfers", []) or []:
                teams = tr.get("teams") or {}
                to_tid   = self._ensure_team_row(teams.get("in"),  fallback_league_id=self.league_id)
                from_tid = self._ensure_team_row(teams.get("out"), fallback_league_id=self.league_id)

                # Derive season year + robust, never-null date
                season_hint = self._coerce_year_from_season(tr.get("season"))
                date_obj = self._parse_transfer_date(tr.get("date"), season_hint=season_hint)

                t_defaults = _filtered_defaults(Transfer, {
                    "type": tr.get("type"),
                    "season": tr.get("season"),
                    "fee_text": tr.get("type") or tr.get("reason"),
                    "raw_json": tr,
                    "to_team_id": to_tid,
                    "from_team_id": from_tid,
                })

                # Build a lookup that avoids collisions when to_team is NULL
                if to_tid is not None:
                    lookup = dict(player_id=pid, date=date_obj, to_team_id=to_tid)
                elif from_tid is not None:
                    lookup = dict(player_id=pid, date=date_obj, to_team_id=None, from_team_id=from_tid)
                else:
                    # Last resort: player+date with NULL to_team — may have dupes; handled below.
                    lookup = dict(player_id=pid, date=date_obj, to_team_id=None)

                try:
                    Transfer.objects.update_or_create(
                        **lookup,
                        defaults=t_defaults,
                    )

                except Transfer.MultipleObjectsReturned:
                    # Heal duplicates in-place: keep one canonical row, update it, delete the extras
                    with transaction.atomic():
                        qs = (Transfer.objects
                            .select_for_update()
                            .filter(**lookup)
                            .order_by('id'))

                        primary = qs.filter(from_team_id=from_tid).first() or qs.first()
                        # Update canonical row
                        for k, v in t_defaults.items():
                            setattr(primary, k, v)
                        primary.save(update_fields=list(t_defaults.keys()))
                        # Remove duplicates
                        qs.exclude(pk=primary.pk).delete()
                        logger.warning(
                            "Deduped duplicate transfers (player=%s, date=%s, to=%s, from=%s)",
                            pid, date_obj, to_tid, from_tid
                        )

                except IntegrityError:
                    # Unique race (e.g., another process inserted first). Retry as an update.
                    Transfer.objects.filter(**lookup).update(**t_defaults)

    def _ingest_venue(self, venue_id: int):
        data = api.venues_by_id(venue_id)
        resp = data.get("response", []) or []
        if not resp:
            self.stdout.write(self.style.WARNING(f"Venue {venue_id}: no data from API; skipping"))
            return
        for v in resp:
            if isinstance(v, dict) and "venue" in v and isinstance(v["venue"], dict):
                v = v["venue"]
            v_defaults = _filtered_defaults(Venue, {
                "name": v.get("name"),
                "city": v.get("city"),
                "country": v.get("country"),
                "capacity": v.get("capacity"),
                "surface": v.get("surface"),
                "address": v.get("address"),
                "image_url": v.get("image"),
                "built": v.get("built"),
                "raw_json": v,
            })
            Venue.objects.update_or_create(
                id=v.get("id"),
                defaults=v_defaults,
            )

    def _ingest_injuries(self, league_id: int, season: int):
        """
        Robust injuries ingest:
          - Requires fixture_id, player_id, team_id (skip if team_id missing to satisfy NOT NULL).
          - Stores raw JSON.
          - Lookup uses (fixture_id, player_id, team_id) to avoid collisions.
        """
        data = api.injuries_by_league_season(league_id, season)
        resp = data.get("response", []) or []

        created, updated, skipped = 0, 0, 0

        for item in resp:
            try:
                p = item.get("player") or {}
                t = item.get("team") or {}
                f = item.get("fixture") or {}

                pid = p.get("id")
                fid = f.get("id")

                # Be defensive about team id — some payloads are oddly shaped / nested
                tid = t.get("id")
                if tid is None:
                    tid = (item.get("teams") or {}).get("id") \
                          or (item.get("team_info") or {}).get("id")

                # Must have fixture, player, and team due to NOT NULL on team_id
                if not (pid and fid and tid):
                    skipped += 1
                    logger.warning("Skipping injury row (missing ids): fixture=%s player=%s team=%s raw=%s",
                                   fid, pid, tid, str(item)[:300])
                    continue

                # Ensure FK rows exist first
                Player.objects.get_or_create(
                    id=pid,
                    defaults=_filtered_defaults(Player, {
                        "name": p.get("name") or f"Player {pid}",
                        "photo_url": p.get("photo"),
                        "nationality": p.get("nationality"),
                    })
                )
                Team.objects.get_or_create(
                    id=tid,
                    defaults={"league_id": league_id, "name": (t.get("name") or f"Team {tid}")[:128]}
                )

                # Parse date (ISO first, then YYYY-MM-DD)
                date_d = None
                date_raw = item.get("date")
                if date_raw:
                    try:
                        date_d = datetime.fromisoformat(date_raw).date()
                    except Exception:
                        try:
                            date_d = datetime.strptime(date_raw, "%Y-%m-%d").date()
                        except Exception:
                            date_d = None

                # Normalize 'type' which can be string or {"name": "..."}
                itype = item.get("type")
                if isinstance(itype, dict):
                    itype = itype.get("name")

                i_defaults = _filtered_defaults(Injury, {
                    "team_id": tid,
                    "type": itype,
                    "reason": item.get("reason"),
                    "date": date_d,
                    "raw_json": item,
                })

                obj, was_created = Injury.objects.update_or_create(
                    fixture_id=fid,
                    player_id=pid,
                    team_id=tid,
                    defaults=i_defaults,
                )
                created += int(was_created)
                updated += int(not was_created)

            except Exception:
                skipped += 1
                logger.exception("Failed to ingest injury row: %s", str(item)[:300])

        self.stdout.write(self.style.SUCCESS(
            f"Injuries: created={created}, updated={updated}, skipped={skipped}"
        ))

    def _ingest_trophies_for_player(self, player_id: int):
        data = api.trophies_by_player(player_id)
        for tr in data.get("response", []) or []:
            defaults = _filtered_defaults(Trophy, {
                "country": tr.get("country"),
                "place": tr.get("place"),
                "raw_json": tr,
            })
            Trophy.objects.update_or_create(
                player_id=player_id,
                league=tr.get("league") or tr.get("title") or "Unknown",
                season=tr.get("season") or tr.get("year"),
                defaults=defaults,
            )
