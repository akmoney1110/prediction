# matches/services/standings.py
from datetime import date
from typing import Any, Dict, List, Tuple, Optional

from django.db.models import QuerySet

from matches.models import StandingsSnapshot  # adjust import path

def _standings_exist_info(league_id: int, season: int) -> Dict[str, Any]:
    qs: QuerySet[StandingsSnapshot] = (
        StandingsSnapshot.objects.filter(league_id=league_id, season=season)
        .order_by("-as_of_date")
    )
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

def _dig_standings_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tolerantly extract a flat list of row dicts from various provider shapes.
    """
    if not payload or not isinstance(payload, dict):
        return []

    # Common API-Football shape: response[0].league.standings = [ [ rows... ], [ group2 rows... ] ]
    resp = payload.get("response")
    if isinstance(resp, list) and resp:
        node = resp[0]
        if isinstance(node, dict):
            league = node.get("league") or {}
            st = league.get("standings")
            if isinstance(st, list) and st:
                rows: List[Dict[str, Any]] = []
                for grp in st:  # st may be list of groups (each a list of rows)
                    if isinstance(grp, list):
                        rows.extend(grp)
                return rows

    # Some payloads: {"standings":[ [rows], [rows] ]}  or {"standings":[rows]}
    st2 = payload.get("standings")
    if isinstance(st2, list):
        if st2 and isinstance(st2[0], list):
            rows = []
            for grp in st2:
                rows.extend(grp)
            return rows
        return st2  # already flat

    # Grouped formats: {"groups":[{"name": "...", "table":[...]}, ...]}
    groups = payload.get("groups")
    if isinstance(groups, list) and groups:
        rows: List[Dict[str, Any]] = []
        for g in groups:
            gname = (g.get("name") or g.get("group") or "") if isinstance(g, dict) else ""
            table = (g.get("table") or g.get("rows") or []) if isinstance(g, dict) else []
            for r in table:
                if isinstance(r, dict):
                    r.setdefault("group", gname)
                    rows.append(r)
        return rows

    # Simple "table" lists
    if isinstance(payload.get("table"), list):
        return payload["table"]

    return []

def _norm_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a provider's row into a single shape.
    Returns None if we can't even get team id or rank.
    """
    try:
        team_id = (
            r.get("team_id")
            or (r.get("team") or {}).get("id")
            or r.get("tid")
            or r.get("id")
        )
        rank = r.get("rank") or r.get("position")
        if team_id is None or rank is None:
            return None

        group_name = r.get("group") or r.get("group_name") or r.get("name") or ""

        all_stats = r.get("all") or r.get("overall") or {}
        def i(x, default=0):
            try: return int(x)
            except (TypeError, ValueError): return default

        played = r.get("played") or r.get("gamesPlayed") or all_stats.get("played")
        win    = r.get("win")    or r.get("wins")        or all_stats.get("win")
        draw   = r.get("draw")   or r.get("draws")       or all_stats.get("draw")
        loss   = r.get("loss")   or r.get("loses") or r.get("losses") or all_stats.get("lose") or all_stats.get("loss")

        goals  = r.get("goals") or {}
        gf     = r.get("gf") or r.get("goalsFor")     or goals.get("for")
        ga     = r.get("ga") or r.get("goalsAgainst") or goals.get("against")
        gd     = r.get("gd") or r.get("goalsDiff")
        if gd is None and (gf is not None and ga is not None):
            gd = i(gf) - i(ga)

        points = r.get("points") or r.get("pts")
        form   = r.get("form") or r.get("last5") or ""
        last5  = r.get("last5_json") if isinstance(r.get("last5_json"), dict) else {}

        return {
            "team_id":  i(team_id, None),
            "rank":     i(rank, 0),
            "group_name": str(group_name or ""),
            "played":   i(played),
            "win":      i(win),
            "draw":     i(draw),
            "loss":     i(loss),
            "gf":       i(gf),
            "ga":       i(ga),
            "gd":       i(gd),
            "points":   i(points),
            "form":     str(form or ""),
            "last5_json": last5,
        }
    except Exception:
        return None

def _load_standings_rows(
    league_id: int,
    season: int,
    as_of_date: date,
) -> Tuple[List[Dict[str, Any]], Optional[date], Dict[str, Any]]:
    """
    Fallback order:
      1) season, on/before as_of_date
      2) season, latest
      3) league (any season), latest
    Returns: (rows, as_of_date, meta)
    """
    base = StandingsSnapshot.objects.filter(league_id=league_id)
    tried: List[int] = []

    def try_qs(qs, source_label: str):
        snap = qs.first()
        if not snap or snap.id in tried:
            return None
        tried.append(snap.id)
        rows_raw = _dig_standings_list(snap.json)
        rows: List[Dict[str, Any]] = []
        for r in rows_raw or []:
            nr = _norm_row(r)
            if nr:
                rows.append(nr)
        if rows:
            return rows, snap.as_of_date, {"reason": "ok", "source": source_label, "rows": len(rows)}
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
    best_snap = base.order_by("-as_of_date").first()
    best_date = best_snap.as_of_date if best_snap else None
    return [], best_date, meta

def group_rows(rows: List[Dict[str, Any]]):
    """
    Turn flat rows into grouped {name, rows[]} ready for templates.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("group_name") or "Table"].append(r)
    out = []
    for name, lst in groups.items():
        lst.sort(key=lambda x: (x.get("rank") or 0))
        out.append({"name": name, "rows": lst})
    out.sort(key=lambda g: (g["name"] or "Table"))
    return out
