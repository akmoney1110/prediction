import os
import time
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
API_KEY = os.getenv("APIFOOTBALL_KEY")
HEADERS = {"x-apisports-key": API_KEY}

class ApiError(Exception):
    pass

def _get(url: str, params: Dict[str, Any], max_retries: int = 3, backoff: float = 0.8) -> Dict[str, Any]:
    if not API_KEY:
        raise ApiError("APIFOOTBALL_KEY not set in environment.")
    attempt = 0
    while True:
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            # DEBUG:
            # print("REQ", url, params, "STATUS", resp.status_code)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "2"))
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            data = resp.json()
            # DEBUG: surface useful fields if empty
            if not data.get("response"):
                print("[API DEBUG] Empty response",
                      "| params:", params,
                      "| results:", data.get("results"),
                      "| errors:", data.get("errors"),
                      "| paging:", data.get("paging"))
            return data
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise ApiError(f"API request failed after {max_retries} retries: {e}")
            time.sleep(backoff * attempt)





# services/apifootball.py

from typing import Dict, Any, Optional

# ... keep your existing imports, BASE_URL, HEADERS, _get(...)

# ---------- PLAYERS (statistics) ----------
def players_by_team(team_id: int, season: int, page: int = 1,
                    league_id: Optional[int] = None,
                    search: Optional[str] = None) -> Dict[str, Any]:
    """
    /players?season=YYYY&team=ID[&league=ID][&search=str][&page=N]
    """
    params = {"season": season, "team": team_id, "page": page}
    if league_id is not None:
        params["league"] = league_id
    if search:
        params["search"] = search
    return _get(f"{BASE_URL}/players", params)
def players_squads(team_id: int):
    return _get(f"{BASE_URL}/players/squads", {"team": team_id})

def players_by_id(player_id: int, season: int, page: int = 1):
    return _get(f"{BASE_URL}/players", {"id": player_id, "season": season, "page": page})


def players_by_league(league_id: int, season: int, page: int = 1) -> Dict[str, Any]:
    """ /players?season=YYYY&league=ID[&page=N] """
    return _get(f"{BASE_URL}/players", {"season": season, "league": league_id, "page": page})

def player_stats_by_id(player_id: int, season: int, page: int = 1) -> Dict[str, Any]:
    """ /players?id=PID&season=YYYY[&page=N] """
    return _get(f"{BASE_URL}/players", {"id": player_id, "season": season, "page": page})

def players_search(search: str,
                   season: Optional[int] = None,
                   team_id: Optional[int] = None,
                   league_id: Optional[int] = None,
                   page: int = 1) -> Dict[str, Any]:
    """
    /players?search=name[&team=ID][&league=ID][&season=YYYY][&page=N]
    """
    params: Dict[str, Any] = {"search": search, "page": page}
    if season is not None:
        params["season"] = season
    if team_id is not None:
        params["team"] = team_id
    if league_id is not None:
        params["league"] = league_id
    return _get(f"{BASE_URL}/players", params)

# ---------- PLAYERS PROFILES / TEAMS (different endpoints) ----------
def players_profiles(player_id: Optional[int] = None,
                     search: Optional[str] = None,
                     page: int = 1) -> Dict[str, Any]:
    """
    /players/profiles[?player=PID][&search=lastname][&page=N]
    Returns player bio/profile data (not per-season stats).
    """
    params: Dict[str, Any] = {"page": page}
    if player_id is not None:
        params["player"] = player_id
    if search:
        params["search"] = search
    return _get(f"{BASE_URL}/players/profiles", params)

def players_teams(player_id: int) -> Dict[str, Any]:
    """ /players/teams?player=PID """
    return _get(f"{BASE_URL}/players/teams", {"player": player_id})

# ---------- LEADERBOARDS ----------
def players_topscorers(league_id: int, season: int) -> Dict[str, Any]:
    """ /players/topscorers?season=YYYY&league=ID """
    return _get(f"{BASE_URL}/players/topscorers", {"season": season, "league": league_id})

def players_topassists(league_id: int, season: int) -> Dict[str, Any]:
    """ /players/topassists?season=YYYY&league=ID """
    return _get(f"{BASE_URL}/players/topassists", {"season": season, "league": league_id})





def fixtures_by_league_season(league_id: int, season: int) -> Dict[str, Any]:
    """Fetch all fixtures for a league+season (no pagination param)."""
    url = f"{BASE_URL}/fixtures"
    return _get(url, {"league": league_id, "season": season})

def fixtures_by_league_between(league_id: int, date_from: str, date_to: str) -> Dict[str, Any]:
    """Fetch fixtures in a date window (no page param)."""
    url = f"{BASE_URL}/fixtures"
    return _get(url, {"league": league_id, "from": date_from, "to": date_to})


def fixture_statistics(fixture_id: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/fixtures/statistics"
    return _get(url, {"fixture": fixture_id})

def fixture_events(fixture_id: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/fixtures/events"
    return _get(url, {"fixture": fixture_id})

def fixture_lineups(fixture_id: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/fixtures/lineups"
    return _get(url, {"fixture": fixture_id})

def standings(league_id: int, season: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/standings"
    return _get(url, {"league": league_id, "season": season})




# --- NEW endpoint helpers (paste under your existing functions) ---


def transfers_by_team(team_id: int) -> Dict[str, Any]:
    return _get(f"{BASE_URL}/transfers", {"team": team_id})

def venues_by_id(venue_id: int) -> Dict[str, Any]:
    return _get(f"{BASE_URL}/venues", {"id": venue_id})

def trophies_by_player(player_id: int) -> Dict[str, Any]:
    return _get(f"{BASE_URL}/trophies", {"player": player_id})


def injuries_by_league_season(league_id: int, season: int) -> Dict[str, Any]:
    url = f"{BASE_URL}/injuries"
    # ✅ no page parameter
    return _get(url, {"league": league_id, "season": season})




