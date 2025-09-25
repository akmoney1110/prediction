# matches/management/commands/ingest_transfers_only.py

import logging
import re
from datetime import datetime, date, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import IntegrityError
from django.db.models import Q, Max

from leagues.models import Team, League
from matches.models import Player, Transfer

from services import apifootball as api

logger = logging.getLogger(__name__)


def _norm_yyyy_mm_dd(s):
    """Return date from 'YYYY-MM-DD' or None."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _coerce_year_from_season(season_val):
    """Accept '2024', 2024, or '2015-2016' and return the first year as int or None."""
    if season_val is None:
        return None
    m = re.search(r"\d{4}", str(season_val))
    return int(m.group(0)) if m else None


def _parse_transfer_date(date_str, season_hint=None):
    """
    Robustly parse API-Football transfers[].date into a non-null date.
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
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            pass
        for fmt in patterns:
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                pass
        digits = re.sub(r"\D+", "", s)
        if len(digits) == 6:
            d, m, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
            y += 2000 if y < 70 else 1900
            for (yy, mm, dd) in ((y, m, d), (y, d, m)):
                try:
                    return date(yy, mm, dd)
                except Exception:
                    pass
        if len(digits) == 8:
            y, m, d = int(digits[:4]), int(digits[4:6]), int(digits[6:])
            try:
                return date(y, m, d)
            except Exception:
                pass

    if season_hint and 1900 <= season_hint <= 2100:
        return date(season_hint, 7, 1)
    return date.today()


def _field_names(model_cls):
    names = set()
    for f in model_cls._meta.get_fields():
        if getattr(f, "auto_created", False) and not getattr(f, "concrete", True):
            continue
        if not getattr(f, "concrete", True):
            continue
        names.add(f.name)
    return names


def _filtered_defaults(model_cls, defaults: dict) -> dict:
    allowed = _field_names(model_cls)
    return {k: v for k, v in defaults.items() if k in allowed}


def _ensure_team_row(team_blob, fallback_league_id=None):
    """
    Ensure a Team row exists; return team id (or None).
    Requires a league_id (directly or via fallback) if the team does not already exist.
    """
    if not team_blob:
        return None
    tid = team_blob.get("id")
    if not tid:
        return None

    # If it already exists, return the id (don't touch league)
    if Team.objects.filter(id=tid).exists():
        return tid

    # If we need to create, we must have a league_id
    if fallback_league_id is None:
        # As a last resort, use a generic placeholder league if you keep one.
        # Otherwise, refuse creation to avoid NOT NULL violation.
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


class Command(BaseCommand):
    help = (
        "Ingest transfers for today (default) or a custom date window, "
        "for a team (--team-id) or all teams in a league (--league-id). "
        "Use --since-last to auto-start from the latest transfer date already in your DB."
    )

    def add_arguments(self, parser):
        tgrp = parser.add_mutually_exclusive_group(required=True)
        tgrp.add_argument("--team-id", type=int, help="Single team id to ingest")
        tgrp.add_argument("--league-id", type=int, help="League id to ingest all its teams")

        # Time window (default: today only)
        wgrp = parser.add_mutually_exclusive_group()
        wgrp.add_argument("--today", action="store_true", help="Only transfers dated today (default)")
        wgrp.add_argument("--since", type=str, help="YYYY-MM-DD (inclusive)")
        parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (inclusive)")
        parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (inclusive)")

        # Smart windowing
        parser.add_argument("--since-last", action="store_true",
                            help="For each team, start at that team's latest transfer date in DB (inbound or outbound).")
        parser.add_argument("--lookback-days", type=int, default=30,
                            help="Fallback window length if team has no transfers yet (default: 30).")

        # Parsing hint for odd dates
        parser.add_argument("--season", type=int, help="Season start year, used as a hint for odd dates")

    # ---------- core ----------
    def handle(self, *args, **opts):
        team_id = opts.get("team_id")
        league_id = opts.get("league_id")
        season = opts.get("season")
        since_last = bool(opts.get("since_last"))
        lookback_days = int(opts.get("lookback_days") or 30)

        # If running for one team without league, try inferring league from that team, for safe counterparty creation.
        if team_id and league_id is None:
            league_id = Team.objects.filter(id=team_id).values_list("league_id", flat=True).first()

        # Build base window (may be overridden per-team if --since-last)
        base_start, base_end = self._resolve_window(opts)
        self.stdout.write(self.style.NOTICE(
            f"Ingesting transfers window: {base_start} → {base_end}"
        ))

        # Determine target teams
        if team_id:
            team_ids = [team_id]
        else:
            team_ids = list(Team.objects.filter(league_id=league_id).values_list("id", flat=True))
            if not team_ids:
                raise CommandError(f"No teams found for league {league_id}")

        total_created = total_updated = total_skipped = 0

        for tid in team_ids:
            # Per-team window override if requested
            if since_last:
                latest = self._latest_transfer_date_for_team(tid)
                if latest:
                    win_start = latest  # inclusive to catch same-day edits
                else:
                    win_start = date.today() - timedelta(days=lookback_days)
                win_end = date.today()
            else:
                win_start, win_end = base_start, base_end

            c, u, s = self._ingest_transfers_for_team(
                team_id=tid,
                win_start=win_start,
                win_end=win_end,
                season_hint_year=(season or win_start.year),
                # league fallback for counterparty creation
                counterparty_league_id=league_id,
            )
            total_created += c
            total_updated += u
            total_skipped += s

        self.stdout.write(self.style.SUCCESS(
            f"Transfers complete: +{total_created} created / ~{total_updated} updated / {total_skipped} skipped"
        ))

    def _resolve_window(self, opts):
        """Resolve the desired date window (without --since-last)."""
        today = date.today()
        if opts.get("start") or opts.get("end"):
            start = _norm_yyyy_mm_dd(opts.get("start")) or date(1900, 1, 1)
            end = _norm_yyyy_mm_dd(opts.get("end")) or date(2100, 12, 31)
            return start, end

        if opts.get("since"):
            start = _norm_yyyy_mm_dd(opts["since"])
            if not start:
                raise CommandError("--since must be YYYY-MM-DD")
            return start, date(2100, 12, 31)

        # default: today only (also covers --today)
        return today, today

    def _latest_transfer_date_for_team(self, team_id: int) -> date | None:
        """
        Latest transfer date in DB that involves this team (as destination or source).
        """
        agg = (Transfer.objects
               .filter(Q(to_team_id=team_id) | Q(from_team_id=team_id))
               .aggregate(latest=Max("date")))
        return agg.get("latest")

    # ---------- ingestion for one team ----------
    def _ingest_transfers_for_team(self, team_id, win_start, win_end, season_hint_year, counterparty_league_id):
        import time
        t0 = time.time()
        created = updated = skipped = 0

        data = api.transfers_by_team(team_id)
        resp = data.get("response", []) or []
        if not resp:
            self.stdout.write(f"Transfers team {team_id} [{win_start}→{win_end}]: none ({time.time()-t0:.1f}s)")
            return created, updated, skipped

        seen = set()

        for row in resp:
            p = row.get("player") or {}
            pid = p.get("id")
            if not pid:
                continue

            # ensure Player FK
            Player.objects.get_or_create(
                id=pid,
                defaults=_filtered_defaults(Player, {
                    "name": p.get("name") or f"Player {pid}",
                    "firstname": p.get("firstname"),
                    "lastname": p.get("lastname"),
                    "nationality": p.get("nationality"),
                    "photo_url": p.get("photo"),
                    "raw_json": p,
                }),
            )

            for tr in row.get("transfers", []) or []:
                teams = tr.get("teams") or {}
                to_tid   = _ensure_team_row(teams.get("in"),  fallback_league_id=counterparty_league_id)
                from_tid = _ensure_team_row(teams.get("out"), fallback_league_id=counterparty_league_id)

                # robust date (never-null)
                sh = _coerce_year_from_season(tr.get("season")) or season_hint_year
                tdate = _parse_transfer_date(tr.get("date"), season_hint=sh)

                # window filter
                if not (win_start <= tdate <= win_end):
                    skipped += 1
                    continue

                # intra-run de-dupe
                key = (pid, tdate, to_tid, from_tid)
                if key in seen:
                    continue
                seen.add(key)

                # If destination unknown and we couldn't create it safely, best we can do is update a null-destination row
                if to_tid is None:
                    updates = _filtered_defaults(Transfer, {
                        "type": tr.get("type"),
                        "fee_text": tr.get("type") or tr.get("reason"),
                        "raw_json": tr,
                    })
                    cnt = Transfer.objects.filter(
                        player_id=pid, date=tdate, to_team_id__isnull=True, from_team_id=from_tid
                    ).update(**updates)
                    if cnt:
                        updated += cnt
                    else:
                        skipped += 1
                    continue

                # UPDATE first
                updates = _filtered_defaults(Transfer, {
                    "type": tr.get("type"),
                    "fee_text": tr.get("type") or tr.get("reason"),
                    "raw_json": tr,
                    "from_team_id": from_tid,
                })
                cnt = Transfer.objects.filter(
                    player_id=pid, date=tdate, to_team_id=to_tid
                ).update(**updates)
                if cnt:
                    updated += cnt
                    continue

                # CREATE if not found
                try:
                    create_vals = _filtered_defaults(Transfer, {
                        "player_id": pid,
                        "date": tdate,
                        "to_team_id": to_tid,
                        "from_team_id": from_tid,
                        "type": tr.get("type"),
                        "fee_text": tr.get("type") or tr.get("reason"),
                        "raw_json": tr,
                    })
                    Transfer.objects.create(**create_vals)
                    created += 1
                except IntegrityError:
                    # race/duplicate: ensure row is up to date
                    Transfer.objects.filter(
                        player_id=pid, date=tdate, to_team_id=to_tid
                    ).update(**updates)
                    updated += 1

        self.stdout.write(
            f"Transfers team {team_id}: +{created} / ~{updated} updated, {skipped} skipped in {time.time()-t0:.1f}s"
        )
        return created, updated, skipped
