# predictions/api_football.py
import requests
from datetime import datetime, timedelta
from dateutil.parser import parse

""
API_KEY = '6c5bcbd4223dd1407b92d768d451af75'  
BASE_URL = 'https://v3.football.api-sports.io'
HEADERS = {
    'x-apisports-key': API_KEY
}


def get_upcoming_fixtures(date_str):
    """
    Get all fixtures for a specific date with team logos included
    :param date_str: 'YYYY-MM-DD'
    """
    url = f"{BASE_URL}/fixtures?date={date_str}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        fixtures = response.json().get('response', [])
        
        # Add team logos to the response
        for fixture in fixtures:
            if 'teams' in fixture:
                fixture['teams']['home']['logo'] = fixture['teams']['home'].get('logo')
                fixture['teams']['away']['logo'] = fixture['teams']['away'].get('logo')
        
        return fixtures
    except Exception as e:
        print(f"[ERROR] Failed to fetch fixtures for {date_str}: {e}")
        return []

def get_fixture_info(fixture_id):
    """
    Get detailed fixture data using fixture ID.
    """
    url = f"{BASE_URL}/fixtures?id={fixture_id}"
    try:
        res = requests.get(url, headers=HEADERS)
        data = res.json()
        if not data.get('response'):
            print(f"[ERROR] No fixture found for ID {fixture_id}")
            return None
        return data['response'][0]
    except Exception as e:
        print(f"[ERROR] Fetching fixture failed: {e}")
        return None


def get_last_team_matches(team_id, side, opponent_id=None):
    """
    Fetch last 10 completed matches for a team filtered by home/away side,
    excluding any match against the given opponent_id.
    
    :param team_id: int
    :param side: 'home' or 'away'
    :param opponent_id: int (optional) – team to exclude (e.g. current opponent)
    """
    url = f"{BASE_URL}/fixtures?team={team_id}&last=50"
    try:
        res = requests.get(url, headers=HEADERS)
        fixtures = res.json().get('response', [])

        filtered = []
        for f in fixtures:
            if f['fixture']['status']['short'] != 'FT':
                continue
            if f['teams'][side]['id'] != team_id:
                continue

            # Skip matches vs current opponent
            other_side = 'away' if side == 'home' else 'home'
            opponent_in_match = f['teams'][other_side]['id']
            if opponent_id and opponent_in_match == opponent_id:
                continue

            filtered.append(f)

        print(f"[DEBUG] {side.upper()} matches for team {team_id} (excluding {opponent_id}): {len(filtered)} found.")
        return filtered[:10]

    except Exception as e:
        print(f"[ERROR] Fetching matches for team {team_id}: {e}")
        return []




def score_team(matches, side):
    wins = 0
    draws = 0

    for match in matches:
        goals_home = match['goals']['home']
        goals_away = match['goals']['away']
        is_draw = goals_home == goals_away
        is_winner = match['teams'][side]['winner']

        if is_draw:
            draws += 1
        elif is_winner:
            wins += 1

    # Calculate win score
    if wins >= 7:
        win_score = 4
    elif wins >= 5:
        win_score = 3
    elif wins >= 3:
        win_score = 2
    else:
        win_score = 0

    # Calculate draw score
    if draws >= 7:
        draw_score = 2
    elif draws >= 5:
        draw_score = 1
    elif draws >= 3:
        draw_score = 0.5
    else:
        draw_score = 0

    return win_score + draw_score




def score_team_h2h(matches, side):
    wins = 0
    draws = 0

    for match in matches:
        goals_home = match['goals']['home']
        goals_away = match['goals']['away']
        is_draw = goals_home == goals_away
        is_winner = match['teams'][side]['winner']

        if is_draw:
            draws += 1
        elif is_winner:
            wins += 1

    # Calculate win score
    if wins >= 4:
        win_score = 4
    elif wins >= 2:
        win_score = 2
  
    else:
        win_score = 0

    # Calculate draw score
    if draws >= 4:
        draw_score = 2
    elif draws >= 2:
        draw_score = 1
    
    else:
        draw_score = 0

    return win_score + draw_score




def get_last_h2h_matches(home_id, away_id, max_matches=5):
    """
    Fetch up to `max_matches` completed same-side H2H matches:
    - home_id was home
    - away_id was away
    """
    url = f"{BASE_URL}/fixtures/headtohead?h2h={home_id}-{away_id}&last=20"
    try:
        res = requests.get(url, headers=HEADERS)
        fixtures = res.json().get('response', [])

        filtered = [
            f for f in fixtures
            if (
                f['fixture']['status']['short'] == 'FT' and
                f['teams']['home']['id'] == home_id and
                f['teams']['away']['id'] == away_id
            )
        ]

        print(f"[DEBUG] H2H same-side matches for {home_id} vs {away_id}: {len(filtered)} found.")
        return filtered[:max_matches]

    except Exception as e:
        print(f"[ERROR] Fetching H2H matches for {home_id} vs {away_id}: {e}")
        return []












def get_team_injuries_and_impact(team_id):
    """
    Fetch injured players and apply penalties based on role and rating:
    - Goalkeeper with rating ≥ 7.9: +1
    - Striker with rating ≥ 7.9: +2
    - Midfielder with rating ≥ 7.5: +1
    - Defender with rating ≥ 7.5: +1
    """
    url = f"{BASE_URL}/injuries?team={team_id}"
    try:
        res = requests.get(url, headers=HEADERS)
        players = res.json().get('response', [])
        penalty = 0

        for player_info in players:
            player = player_info.get("player", {})
            statistics = player_info.get("statistics", [{}])[0]

            position = player.get("position", "").lower()
            rating_str = statistics.get("games", {}).get("rating")

            if rating_str:
                try:
                    rating = float(rating_str)

                    # Goalkeeper
                    if position == "goalkeeper":
                        started_matches = statistics.get("games", {}).get("lineups", 0)
                        if started_matches > 0:
                            penalty += 0.5


                    # Striker / Attacker
                    elif position == "attacker" and rating >= 7.9:
                        penalty += 1

                    # Midfielder
                    elif position == "midfielder" and rating >= 7.5:
                        penalty += 0.5

                    # Defender
                    elif position == "defender" and rating >= 7.5:
                        penalty += 0.5

                except ValueError:
                    continue  # Skip invalid ratings

        print(f"[DEBUG] Injury penalty for team {team_id}: -{penalty}")
        return penalty

    except Exception as e:
        print(f"[ERROR] Fetching injuries for team {team_id}: {e}")
        return 0




def get_team_rating_awards(team_id, season=2024):
    url = f"{BASE_URL}/players?team={team_id}&season={season}"
    try:
        res = requests.get(url, headers=HEADERS)
        players_data = res.json().get('response', [])
        award = 0

        for player_info in players_data:
            player = player_info.get("player", {})
            statistics = player_info.get("statistics", [{}])[0]
            position = statistics.get("games", {}).get("position", "").lower()
            rating_str = statistics.get("games", {}).get("rating")

            if not rating_str:
                continue  # skip if no rating

            try:
                rating = float(rating_str)
            except ValueError:
                continue

            if position == "attacker" and rating >= 7.9:
                award += 0.5
            elif position == "midfielder" and rating >= 7.5:
                award += 0.5
            elif position == "defender" and rating >= 7.5:
                award += 0.5
            elif position == "goalkeeper":
                started_matches = statistics.get("games", {}).get("lineups", 0)
                if started_matches > 0:
                    penalty += 0.5

        print(f"[DEBUG] Award score for team {team_id}: +{award}")
        return award

    except Exception as e:
        print(f"[ERROR] Fetching team player ratings for team {team_id}: {e}")
        return 0



def get_league_standings(league_id, season):
    """
    Returns a dictionary of team_id → position from the standings.
    """
    url = f"{BASE_URL}/standings?league={league_id}&season={season}"
    try:
        res = requests.get(url, headers=HEADERS)
        standings = res.json().get("response", [])
        if not standings:
            return {}

        table = standings[0]["league"]["standings"][0]  # First group
        return {team["team"]["id"]: team["rank"] for team in table}

    except Exception as e:
        print(f"[ERROR] Failed to fetch standings: {e}")
        return {}















def bulk_predict(fixtures, filter_for=('Win', '1X')):
    results = []
    for fixture in fixtures:
        prediction_data = predict_by_last_matches(fixture['fixture']['id'])

        if any(tag in prediction_data['prediction'] for tag in filter_for):
            results.append({
                'fixture_id': fixture['fixture']['id'],
                'home_team': prediction_data['home_team'],
                'away_team': prediction_data['away_team'],
                'prediction': prediction_data['prediction'],
                'score_diff': prediction_data.get('score_difference', 0)
            })

    return results






from datetime import datetime

from dateutil.parser import parse
from datetime import datetime

from dateutil.parser import parse
from datetime import datetime, timezone

def get_team_recent_matches(team_id, days=10):
    url = f"{BASE_URL}/fixtures?team={team_id}&last=30"
    try:
        res = requests.get(url, headers=HEADERS)
        matches = res.json().get('response', [])
        now = datetime.now(timezone.utc)  # Make it timezone-aware
        recent = []
        for match in matches:
            date_str = match['fixture']['date']
            match_date = parse(date_str)
            if (now - match_date).days <= days:
                recent.append(match)
        return recent
    except Exception as e:
        print(f"[ERROR] Fetching recent matches for fatigue: {e}")
        return []






def fatigue_penalty(team_id):
    recent_matches = get_team_recent_matches(team_id, days=5)
    match_count = len(recent_matches)

    if match_count >= 3:
        return -1
    elif match_count == 2:
        return -0.5
    elif match_count == 1:
        return 0
   
    return 0






def check_cup_matches(team_id, fixture_date, window=5):
    """
    Checks if a team has any cup matches ±window days around fixture_date.
    Returns (has_recent_cup, has_upcoming_cup)
    """
    url = f"{BASE_URL}/fixtures?team={team_id}&from={fixture_date - timedelta(days=window)}&to={fixture_date + timedelta(days=window)}"
    try:
        res = requests.get(url, headers=HEADERS)
        fixtures = res.json().get('response', [])

        has_recent_cup = False
        has_upcoming_cup = False

        for match in fixtures:
            match_date = parse(match['fixture']['date']).date()
            league_type = match['league']['type'].lower()
            is_cup = league_type == "cup"

            if not is_cup:
                continue

            if team_id not in [match['teams']['home']['id'], match['teams']['away']['id']]:
                continue

            if match_date < fixture_date:
                has_recent_cup = True
            elif match_date > fixture_date:
                has_upcoming_cup = True

        return has_recent_cup, has_upcoming_cup

    except Exception as e:
        print(f"[ERROR] Checking cup matches for team {team_id}: {e}")
        return False, False










def predict_by_last_matches(fixture_id):
    fixture = get_fixture_info(fixture_id)
    if not fixture:
        return {
            "prediction": "Invalid fixture ID or no data available.",
            "home_score": 0,
            "away_score": 0,
            "home_team": "Unknown",
            "away_team": "Unknown"
        }

    home_team = fixture['teams']['home']
    away_team = fixture['teams']['away']
    league_id = fixture['league']['id']
    season = fixture['league']['season']

    # --- Team Form ---
    home_matches = get_last_team_matches(home_team['id'], 'home')
    away_matches = get_last_team_matches(away_team['id'], 'away')

    if not home_matches or not away_matches:
        return {
            "prediction": "Not enough match data to make a prediction.",
            "home_score": 0,
            "away_score": 0,
            "home_team": home_team['name'],
            "away_team": away_team['name']
        }

    
    home_score = score_team(home_matches, 'home')
    away_score = score_team(away_matches, 'away')
    print(f"Form score: {away_score}")
    print(f"Form score: {home_score}")


    # --- H2H Matches ---
    h2h_matches = get_last_h2h_matches(home_team['id'], away_team['id'])
    if h2h_matches:
        home_score += score_team_h2h(h2h_matches, 'home')  # Now adds to existing score
        away_score += score_team_h2h(h2h_matches, 'away')
    print(f"After H2H - Home: {home_score}, Away: {away_score}")
     # --- Injury Penalties ---
    home_score -= get_team_injuries_and_impact(home_team['id'])
    away_score -= get_team_injuries_and_impact(away_team['id'])
    print(f"with injury score: {away_score}")
    print(f"with injury score: {home_score}")
    # --- player rating ---
    home_score += get_team_rating_awards(home_team['id'])
    away_score += get_team_rating_awards(away_team['id'])
    print(f"with ratin score: {away_score}")
    print(f"with rating score: {home_score}")
    
    # --- Clean Sheet Analysis ---
    home_clean_sheets = sum(1 for match in home_matches if match['goals']['away'] == 0)
    away_clean_sheets = sum(1 for match in away_matches if match['goals']['home'] == 0)    
    if home_clean_sheets >= 6:
        home_score += 1
    if away_clean_sheets >= 6:
        away_score += 1  
    print(f"Home clean sheets: {home_clean_sheets}")
    print(f"Away clean sheets: {away_clean_sheets}")      
    print(f"with clean score: {away_score}")    
    print(f"with clean score: {home_score}")    
    

    

   # --- League Ranking Boost ---
    # --- League Ranking Boost ---
    standings = get_league_standings(league_id, season)

    standings_dict = {}

    if isinstance(standings, list):
        for entry in standings:
            try:
                team_id = entry['team']['id']
                rank = entry['rank']
                standings_dict[team_id] = rank
            except (KeyError, TypeError) as e:
                print(f"[WARNING] Skipping entry due to missing data: ")
    else:
        print("[ERROR] Standings not in expected list format")

    # Now use the dict for your logic
    home_rank = standings_dict.get(home_team['id'])
    away_rank = standings_dict.get(away_team['id'])

    if home_rank and away_rank:
        if home_rank <= 3 and away_rank >= 18:
            home_score += 2
        elif away_rank <= 3 and home_rank >= 18:
            away_score += 2
        print(f"with rank score: {away_score}")


    print(f"with table score: {away_score}")    
    print(f"with table score: {home_score}")    

    # --- player fatigue ---
    home_penalty = fatigue_penalty(home_team['id'])
    away_penalty = fatigue_penalty(away_team['id'])

    home_score += home_penalty
    away_score += away_penalty
    print(f"with fatigue score: {away_score}")    


   
    # --- CUP Match Distraction Check ---
    fixture_date = parse(fixture['fixture']['date']).date()

    home_recent_cup, home_upcoming_cup = check_cup_matches(home_team['id'], fixture_date)
    away_recent_cup, away_upcoming_cup = check_cup_matches(away_team['id'], fixture_date)

    if home_recent_cup:
        print(f"[CUP] Home team {home_team['name']} played a cup match recently (-1)")
        home_score -= 0.5
    if home_upcoming_cup:
        print(f"[CUP] Home team {home_team['name']} has an upcoming cup match (-0.5)")
        home_score -= 1

    if away_recent_cup:
        print(f"[CUP] Away team {away_team['name']} played a cup match recently (-1)")
        away_score -= 0.5
    if away_upcoming_cup:
        print(f"[CUP] Away team {away_team['name']} has an upcoming cup match (-0.5)")
        away_score -= 1
    print(f"with cup score: {away_score}")    

    
    # --- Final Prediction ---
    diff = abs(home_score - away_score)
    print(f"with score: {away_score}") 
    print(f"with score: {home_score}") 
    print(f"with score: {diff}") 
    prediction = ""
    details = ""
    draw_predictions = []

    # Clean Sheet Analysis for Draw Predictions
    if diff < 2 and home_clean_sheets >= 6 and away_clean_sheets >= 6:
       
        draw_predictions.append("Strong defense but close match - High chance of draw")
                    
    else:
        if diff < 1 and home_clean_sheets >= 5 and away_clean_sheets >= 5:
            draw_predictions.append("Very close match - High chance of draw")
        elif diff <= 2 and home_clean_sheets >= 4 and away_clean_sheets >= 4:
            draw_predictions.append("Close match - low chance of draw")

    # Add specific minute predictions if applicable
    if diff <= 3 and home_clean_sheets >= 6 and away_clean_sheets >= 6:
        draw_predictions.extend(["15-minute draw likely", "10-minute draw possible"])
    elif diff <= 2 and home_clean_sheets >= 6 and away_clean_sheets >= 6:
        draw_predictions.append("10-minute draw possible")

    if home_score > away_score:
        details = f"Home Advantage: {home_team['name']}"
        if diff > 5 and away_clean_sheets <= 3 and home_score > 5:
            prediction = f"{home_team['name']} to Win"
        elif diff <=1 and home_clean_sheets >= 6 and away_clean_sheets >= 6:
            prediction = f"15-minute draw - {home_team['name']}" 
            
        elif diff <=2 and home_clean_sheets > 5 and away_clean_sheets > 5:
            prediction = f"10-minute draw - {home_team['name']}"   
            
        elif diff >= 4 and home_clean_sheets >= 4 and away_clean_sheets < 4 and home_score > 4:
            prediction = f"Handicap 1 first half - {home_team['name']}" 
        elif diff >= 2 and home_clean_sheets >= 4 and away_clean_sheets < 4 and home_score > 2:
            prediction = f"Handicap 2 first half - {home_team['name']}"     
        elif diff >= 4 and home_clean_sheets >= 4 and away_clean_sheets < 4 and home_score > 4:
            prediction = f"1x first half - {home_team['name']}"                 
        elif diff >= 2 and   away_clean_sheets < 4 and home_score > 2:
            prediction = f"Handicap 2 - {home_team['name']}"         
        elif diff >= 4  and away_clean_sheets < 4 and home_score > 4:
            prediction = f"Handicap 1 - {home_team['name']}"
        elif  diff > 4 and away_clean_sheets <= 3 and home_score > 4:
            prediction = f"1X (Draw)"
        elif diff >=3  and away_clean_sheets < 5 and home_score < 5:
            prediction = f"12 (Either to winner)"
        elif diff  > 5 and away_clean_sheets <= 5 and home_score > 5:
            prediction = f"Draw No Bet - {home_team['name']}" 
        else: 
            prediction = f"No Bet "

    elif away_score > home_score:
        details = f"Away Advantage: {away_team['name']}"
        if diff > 5 and home_clean_sheets <= 3 and away_score > 5:
            prediction = f"{away_team['name']} to Win"
        elif diff < 0 and away_clean_sheets >= 6 and home_clean_sheets >= 6:
            prediction = f"15-minute draw - {away_team['name']}"  
            
        elif diff < 2 and away_clean_sheets > 5 and home_clean_sheets > 5:
            prediction = f"10-minute draw - {away_team['name']}"  
            
        elif diff >= 4 and away_clean_sheets >= 4 and home_clean_sheets < 4 and away_score > 4:
            prediction = f"Handicap 1 first half - {away_team['name']}"
        elif diff >= 2 and away_clean_sheets >= 4 and home_clean_sheets < 4 and away_score > 2:
            prediction = f"Handicap 2 first half - {away_team['name']}"      
        elif diff >= 4 and away_clean_sheets >= 4 and home_clean_sheets < 4 and away_score > 4:
            prediction = f"2x first half - {away_team['name']}"      
        elif diff >= 2 and away_score > 2 and home_clean_sheets < 4:
            prediction = f"Handicap 2 - {away_team['name']}"            
        elif diff >=4 and away_score > 3 and home_clean_sheets < 4:
            prediction =  f"Handicap 1 - {away_team['name']}"
        elif diff > 4 and away_score > 4 and home_clean_sheets < 4:
            prediction = f"2X (Away or Draw) - {away_team['name']}"
        elif diff >= 3 and away_score < 5 and home_clean_sheets < 5:
            prediction = f"12 (Either to win)"
        elif diff > 5 and away_score > 5 and home_clean_sheets < 4:
            prediction = f"Draw No Bet - {away_team['name']}"    
        else:  # diff <= 30
            prediction = f" no bet"
    else:
        if away_clean_sheets > 5 and home_clean_sheets > 5:
            prediction = "Draw"
            details = "Equal strength teams"
        if home_clean_sheets > 7 and away_clean_sheets > 7:
            prediction += " (Strong defense)"
        draw_predictions.append("Full-time draw likely")

    return {
        "home_team": home_team['name'],
        "away_team": away_team['name'],
        "home_score": home_score,
        "away_score": away_score,
        "score_difference": diff,
        "prediction": prediction,
        "details": details,
        "clean_sheets": {
            "home": home_clean_sheets,
            "away": away_clean_sheets
        },
        "draw_analysis": draw_predictions if draw_predictions else ["No strong draw indications"],
        "match_analysis": {
            "home_clean_sheets": home_clean_sheets,
            "away_clean_sheets": away_clean_sheets,
            "home_recent_form": len([m for m in home_matches if m['teams']['home']['winner']]),
            "away_recent_form": len([m for m in away_matches if m['teams']['away']['winner']])
        }
    }




 



















def get_last_10_goals_boost(team_id, side):
   
    matches = get_last_team_matches(team_id, side)
    total_goals = 0

    for match in matches:
        goals = match['goals']['home'] if side == 'home' else match['goals']['away']
        total_goals += goals

    # Unified boost rules
    if total_goals >= 35:
        boost = 3.5 
    elif total_goals >= 25:
        boost = 2.5 
    elif total_goals >= 20:
        boost = 1.5 
    elif total_goals >= 15:
        boost = 1 
    else:
        boost = 0

    return {
        'total_goals': total_goals,
        'boost': boost
    }




def get_goal_frequency_boost(team_id, side):
    """
    Calculate frequency-based goal scoring boost based on 4+, 3+, and 2+ goals
    scored in the last 10 matches, adjusted by home/away side.
    Also penalizes for 0 goals scored, and rewards clean sheets.
    """
    matches = get_last_team_matches(team_id, side)
    count_4_goals = 0
    count_3_goals = 0
    count_2_goals = 0
    count_0_goals = 0
    

    for match in matches:
        goals = match['goals']['home'] if side == 'home' else match['goals']['away']
        conceded = match['goals']['away'] if side == 'home' else match['goals']['home']

        # Count goals scored
        if goals >= 4: count_4_goals += 1
        if goals == 3: count_3_goals += 1
        if goals == 2: count_2_goals += 1
        if goals == 0: count_0_goals += 1

        

    freq_boost = 0

    # --- 4+ Goals ---
    if count_4_goals >= 9: freq_boost += 5
    elif count_4_goals >= 6: freq_boost += 4
    elif count_4_goals >= 3: freq_boost += 3
    elif count_4_goals >= 1: freq_boost += 1
    
    # --- 3+ Goals ---
    if count_3_goals >= 9: freq_boost += 4
    elif count_3_goals >= 6: freq_boost += 3 
    elif count_3_goals >= 3: freq_boost += 2
    elif count_3_goals >= 1: freq_boost += 0.5
    

    # --- 2+ Goals ---
    if count_2_goals >= 9: freq_boost += 3
    elif count_2_goals >= 6: freq_boost += 2
    elif count_2_goals >= 3: freq_boost += 0.5
    

    # --- 0 Goal Penalty ---
    if count_0_goals >= 9: freq_boost -= 3
    elif count_0_goals == 6: freq_boost -= 2
    elif count_0_goals == 3: freq_boost -= 0.5
   
    elif count_0_goals == 0: freq_boost += 2

    

    return {
        'freq_boost': freq_boost,
        'count_4_goals': count_4_goals,
        'count_3_goals': count_3_goals,
        'count_2_goals': count_2_goals,
        'count_0_goals': count_0_goals,
        
    }







def get_h2h_last_10_goals_boost(team_id, opponent_id, side):
   
    matches = get_last_h2h_matches(team_id, opponent_id, side)
    total_goals = 0

    for match in matches:
        goals = match['goals']['home'] if side == 'home' else match['goals']['away']
        total_goals += goals

    # Apply boost based on total goals and side
    if total_goals >= 16:
        boost = 3 
    elif total_goals >= 11:
        boost = 2 
    elif total_goals >= 6:
        boost = 1 
    elif total_goals >= 3:
        boost = 0.5 
    else:
        boost = 0

    return {
        'total_goals': total_goals,
        'boost': boost
    }







def get_h2h_goal_frequency_boost(team_id, opponent_id, side):
    """
    Calculates frequency-based boost from last 10 H2H matches.
    Considers how often the team scored 2+, 3+, 4+ goals against this specific opponent.
    """
    matches = get_last_h2h_matches(team_id, opponent_id, side)
    count_4_goals = 0
    count_3_goals = 0
    count_2_goals = 0
    count_0_goals = 0

    for match in matches:
        goals = match['goals']['home'] if side == 'home' else match['goals']['away']
        if goals >= 4: count_4_goals += 1
        if goals == 3: count_3_goals += 1
        if goals == 2: count_2_goals += 1
        if goals == 0: count_0_goals += 1

    freq_boost = 0
    is_away = side == 'away'

    # --- 4+ Goals ---
    if count_4_goals >= 4:
        freq_boost += 4 
    elif count_4_goals == 3:
        freq_boost += 3 
    elif count_4_goals >= 2:
        freq_boost += 2 
    
    elif count_4_goals == 1:
        freq_boost += 0.5 

    # --- 3+ Goals ---
    if count_3_goals >= 4:
        freq_boost += 3 
    elif count_3_goals == 3:
        freq_boost += 2 
    elif count_3_goals == 2:
        freq_boost += 0.5 
    

    # --- 2+ Goals ---
    if count_2_goals >= 4:
        freq_boost += 2 
    elif count_2_goals == 3:
        freq_boost += 1 
    elif count_2_goals == 2:
        freq_boost += 0.5 
    

    # --- 0 Goal Penalty ---
    if count_0_goals >= 4:
        freq_boost -= 3 
    elif count_0_goals == 3:
        freq_boost -= 2 
    elif count_0_goals == 2:
        freq_boost -= 0.5 
    
    elif count_0_goals == 0:
        freq_boost += 2 

    return {
        'freq_boost': freq_boost,
        'count_4_goals': count_4_goals,
        'count_3_goals': count_3_goals,
        'count_2_goals': count_2_goals,
        'count_0_goals': count_0_goals
    }


import requests
def get_team_players(team_id, season=2024):
    import requests

    url = "https://v3.football.api-sports.io/players"
    headers = {
        "x-apisports-key": "6c5bcbd4223dd1407b92d768d451af75"
    }
    params = {
        "team": team_id,
        "season": season
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return []

    data = response.json()
    players = []

    for p in data['response']:
        # Skip if no statistics available
        if not p.get('statistics') or len(p['statistics']) == 0:
            continue

        games_data = p['statistics'][0].get('games', {})
        position = games_data.get('position')
        rating = games_data.get('rating')

        # Optional: skip players with missing position or rating
        if position is None and rating is None:
            continue

        player = p['player'] | {
            'position': position,
            'rating': rating
        }
        players.append(player)

    return players



def get_player_rating_boost(team_id):
    """
    Checks player ratings and applies:
    - Positive boost for attackers/midfielders with high ratings
      +1 if rating > 7.5
      +2 if rating > 8.5
      +3 if rating > 9.5
    - Negative boost for defenders with low ratings
      -1 if rating < 5.5
      -2 if rating < 4.5
      -3 if rating < 3.5
    """
    players = get_team_players(team_id)
    
    top_attacking_rating = 0
    worst_defender_rating = 10  # Start with highest possible rating

    for player in players:
        position = player.get("position", "").lower()
        rating = player.get("rating")
        
        if not rating:
            continue
            
        try:
            rating = float(rating)
        except:
            continue

        # Check attacking players (positive boost)
        if position in ["attacker", "striker", "forward", "midfielder"]:
            if rating > top_attacking_rating:
                top_attacking_rating = rating
                
        # Check defenders (negative boost)
        elif position in ["defender", "full-back", "center-back", "left-back", "right-back"]:
            if rating < worst_defender_rating:
                worst_defender_rating = rating

    # Calculate positive boost from attackers
    positive_boost = 0
    if top_attacking_rating > 9.5:
        positive_boost = 3
    elif top_attacking_rating > 8.5:
        positive_boost = 2
    elif top_attacking_rating > 7.5:
        positive_boost = 1

    # Calculate negative boost from defenders
    negative_boost = 0
    if worst_defender_rating < 3.5:
        negative_boost = -3
    elif worst_defender_rating < 4.5:
        negative_boost = -2
    elif worst_defender_rating < 5.5:
        negative_boost = -1

    total_boost = positive_boost + negative_boost

    return {
        "total_boost": total_boost,
        "attacking_boost": positive_boost,
        "defending_penalty": negative_boost,
        "top_attacking_rating": top_attacking_rating,
        "worst_defender_rating": worst_defender_rating
    }














def get_clean_sheet_boost(team_id, side):
    """
    Calculate clean sheet reward boost from the last 10 matches.
    """
    matches = get_last_team_matches(team_id, side)
    clean_sheets = 0

    for match in matches:
        conceded = match['goals']['away'] if side == 'home' else match['goals']['home']
        if conceded == 0:
            clean_sheets += 1


    return {
        'clean_sheets': clean_sheets,
   
    }








def get_h2h_clean_sheet_boost(team_id, opponent_id, side):
    """
    Calculates clean sheet boost from the last 10 head-to-head matches.
    A clean sheet is when the team did not concede any goals against the opponent.
    """
    matches = get_last_h2h_matches(team_id, opponent_id, side)
    clean_sheets = 0

    for match in matches:
        conceded = match['goals']['away'] if side == 'home' else match['goals']['home']
        if conceded == 0:
            clean_sheets += 1

   

    return {
        "clean_sheets": clean_sheets,
       
    }



def get_team_goal_prediction(boost, opponent_total_clean):
    if boost >= 10 and opponent_total_clean < 5:
        return "Over 2.5 Goals"
    elif boost >= 8 and opponent_total_clean < 5:
        return "Over 2 Goals"
    elif boost >= 6 and opponent_total_clean < 5:
        return "Over 1.5 Goals"
    elif boost >= 4 and opponent_total_clean < 5:
        return "Over 1 Goal"
    elif boost >= 2 and opponent_total_clean < 4:
        return "Over 0.5"
    elif boost <= 1 and opponent_total_clean < 4:
        return "Over 0.5 (risky)"
    else:
        return "Under 1.5 Goals"










def predict_goals(fixture_id):
    fixture = get_fixture_info(fixture_id)
    if not fixture:
        return {
            "prediction": "Invalid fixture ID or no data available.",
            "home_team": "Unknown",
            "away_team": "Unknown"
        }

    home = fixture['teams']['home']
    away = fixture['teams']['away']

    # --- FORM-BASED BOOSTS ---
    form_home_goals = get_last_10_goals_boost(home['id'], 'home')
    form_away_goals = get_last_10_goals_boost(away['id'], 'away')
    print(form_home_goals)
    print(form_away_goals)

    form_home_freq = get_goal_frequency_boost(home['id'], 'home')
    form_away_freq = get_goal_frequency_boost(away['id'], 'away')
    print(form_home_freq)
    print(form_away_freq)
    # --- H2H BOOSTS ---
    h2h_home_goals = get_h2h_last_10_goals_boost(home['id'], away['id'], 'home')
    h2h_away_goals = get_h2h_last_10_goals_boost(away['id'], home['id'], 'away')
    print(h2h_home_goals)
    print(h2h_away_goals)

    h2h_home_freq = get_h2h_goal_frequency_boost(home['id'], away['id'], 'home')
    h2h_away_freq = get_h2h_goal_frequency_boost(away['id'], home['id'], 'away')
    print(h2h_home_freq)
    print(h2h_away_freq)
    # --- PLAYER RATING BOOSTS ---
    rating_home = get_player_rating_boost(home['id'])
    rating_away = get_player_rating_boost(away['id'])

    
    print(f"with ratin score: {rating_away}")
    print(f"with rating score: {rating_home}")

    # --- CLEAN SHEETS (FORM + H2H) ---
    clean_home = get_clean_sheet_boost(home['id'], 'home')
    clean_away = get_clean_sheet_boost(away['id'], 'away')

    h2h_clean_home = get_h2h_clean_sheet_boost(home['id'], away['id'], 'home')
    h2h_clean_away = get_h2h_clean_sheet_boost(away['id'], home['id'], 'away')

    # --- TOTAL BOOSTS ---
    home_total = (
        form_home_goals.get('boost', 0) + form_home_freq.get('freq_boost', 0) +
        h2h_home_goals.get('boost', 0) + h2h_home_freq.get('freq_boost', 0) +
        rating_home.get('total_boost', 0)  # Changed from ['boost'] to ['total_boost']
    )
    print(f"with ratin score: {home_total}")
    
    away_total = (
        form_away_goals.get('boost', 0) + form_away_freq.get('freq_boost', 0) +
        h2h_away_goals.get('boost', 0) + h2h_away_freq.get('freq_boost', 0) +
        rating_away.get('total_boost', 0)  # Changed from ['boost'] to ['total_boost']
    )
    print(f"with ratin score: {away_total}")
    # Apply clean sheet adjustments
    home_total_clean = clean_home.get('clean_sheets', 0) + h2h_clean_home.get('clean_sheets', 0)
    away_total_clean = clean_away.get('clean_sheets', 0) + h2h_clean_away.get('clean_sheets', 0)
    
    print(f"with clean score: {away_total_clean}")
    print(f"with clean score: {home_total_clean}")
    if home_total_clean >= 10:
        away_total -= 3
    if home_total_clean >= 5:
        away_total -= 2
    if away_total_clean >= 5:    
        home_total -= 2

    # Get goal predictions
    home_team_total_prediction = get_team_goal_prediction(home_total,away_total_clean)
    away_team_total_prediction = get_team_goal_prediction(away_total,home_total_clean)
    print(f"with  score: {away_total}")
    print(f"with  score: {home_total}")
    total = home_total + away_total

    # --- GOAL LINE PREDICTION ---
    if total >= 17 and home_total_clean < 5 and away_total_clean < 5:
        prediction = "Over 3.5 Goals"
    elif total >= 17 and home_total_clean < 7 and away_total_clean < 7:
        prediction = "Over 3 Goals"    
    elif total >= 15 and home_total_clean < 7 and away_total_clean < 7:
        prediction = "Over 2.5 Goals"
    elif total >= 12 and home_total_clean < 7 and away_total_clean < 7:
        prediction = "Over 2 Goals"
    elif total >= 8 and home_total_clean < 7 and away_total_clean < 7:
        prediction = "Over 1.5 Goals"
    elif total >= 5 and home_total_clean < 5 and away_total_clean < 5:
        prediction = "Over 1 Goals"
    elif total >= 3 and home_total_clean < 5 and away_total_clean < 5:
        prediction = "Over 0.5 Goals"
    elif total >= 1 and home_total_clean < 5 and away_total_clean < 5:
        prediction = "Over 0.5 Goals with risk"
    elif total <= 1 and home_total_clean > 10 and away_total_clean > 10:
        prediction = "Under 4.5 Goals"        
    else:
        prediction = "No bet"

    # --- BTTS PREDICTION LOGIC ---
  
    diff = abs(home_total - away_total)
    combined = home_total + away_total
  
    # Further analysis when defense is weak
    btts = None  # Initialize once

    if home_total_clean >= 12 and away_total_clean >= 12:
        btts = "NO (Strong defense)"
    elif home_total_clean >= 10 and away_total_clean >= 10:
        btts = "NO (Strong defense with little risk)"
    elif home_total_clean <= 3 and away_total_clean <= 3:
        btts = "YES (No defense with little risk)"
  
    elif home_total <= 2 and away_total <= 2:
        btts = "NO (Both weak attacks)"
    elif combined < 3 and home_total_clean > 10 and away_total_clean > 10:
        btts = "NO (Low scoring match)"
    
    elif combined > 15 and home_total_clean < 5 and away_total_clean < 5:
        btts = "YES (High Probability)"
    elif combined > 13 and home_total_clean < 5 and away_total_clean < 5:
        btts = "YES (Moderate Probability)"
    elif combined < 8 and home_total_clean < 3 and away_total_clean < 3:
        btts = "YES (Low Probability)"
    elif home_total < 5 and away_total > 5 and (home_total_clean >= 10 or away_total_clean >= 10):
        btts = "NO (Unbalanced attack)"
    elif away_total < 5 and home_total > 5 and (home_total_clean >= 10 or away_total_clean >= 10):
        btts = "NO (Unbalanced attack)"
    elif combined > 6 and diff > 8 and home_total_clean > 10 and away_total_clean > 10:
        btts = "NO (Low Probability)"
    elif combined > 13 and (home_total_clean < 5 or away_total_clean < 5):
        btts = "YES (Low Probability)"

    
    

    # Calculate difference and make decision
    result = predict_by_last_matches(fixture_id)

    home_score = result.get('home_score', 0)
    away_score = result.get('away_score', 0)

    first_home_prediction = None
    first_away_prediction = None
    

    print(f"Predicted scores: Home {home_score}, Away {away_score}")

    if home_score > away_score and home_total > away_total and home_total_clean > 8 and away_total_clean < 5:
        first_home_prediction = f"{result['home_team']} first team to score"
    elif away_score > home_score and away_total > home_total and away_total_clean > 8 and home_total_clean < 5:
        first_away_prediction = f"{result['away_team']} first team to score"
    
    # Adjust attacking totals based on weak attack and strong defense
    if home_total <= 1:
        home_total -= 2
    elif home_total < 3:
        home_total -= 1

    if away_total <= 1:
        away_total -= 2
    elif away_total < 3:
        away_total -= 1
   
    return {
        "home_team": home['name'],
        "away_team": away['name'],
        "form_home_boosts": {**form_home_goals, **form_home_freq},
        "form_away_boosts": {**form_away_goals, **form_away_freq},
        "h2h_home_boosts": {**h2h_home_goals, **h2h_home_freq},
        "h2h_away_boosts": {**h2h_away_goals, **h2h_away_freq},
        "rating_home": rating_home,
        "rating_away": rating_away,
        "total_boost": total,
        "prediction": prediction,
        "btts": btts,
        "first_away_prediction": first_away_prediction,
        "first_home_prediction": first_home_prediction,
        "home_team_total_prediction": home_team_total_prediction,
        "away_team_total_prediction": away_team_total_prediction,
        "source": "Combined (Form + H2H + Player Ratings)",
        "home_total": home_total,
        "away_total": away_total,
        "total": total,
        "clean_sheet_home": clean_home,
        "clean_sheet_away": clean_away,
        "h2h_clean_sheet_home": h2h_clean_home,
        "h2h_clean_sheet_away": h2h_clean_away,
        "home_total_clean": home_total_clean,
        "away_total_clean": away_total_clean
    }






import requests
from datetime import datetime

def get_live_matches():
    """Fetch all currently live matches."""
    url = f"{BASE_URL}/fixtures?live=all"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json().get('response', [])
    except Exception as e:
        print(f"[ERROR] Failed to fetch live matches: {e}")
        return []


def get_today_matches_with_scores():
    """Fetch all matches for today (regardless of status)."""
    today = datetime.utcnow().strftime('%Y-%m-%d')  # ✅ Use UTC for consistency
    url = f"{BASE_URL}/fixtures?date={today}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json().get('response', [])
    except Exception as e:
        print(f"[ERROR] Failed to fetch today's matches: {e}")
        return []


def get_todays_finished_matches():
    """
    Fetch matches from today that have finished (FT, AET, PEN).
    """
    today = datetime.utcnow().strftime('%Y-%m-%d')
    url = f"{BASE_URL}/fixtures?date={today}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        matches = response.json().get('response', [])

        # Manual filtering to avoid relying only on status query
        finished_statuses = {'FT', 'AET', 'PEN'}
        return [
            match for match in matches
            if match.get('fixture', {}).get('status', {}).get('short') in finished_statuses
        ]
    except Exception as e:
        print(f"[ERROR] Failed to fetch today's finished matches: {e}")
        return []

















import random
from datetime import datetime

def predict_todays_matches():
    # Get today's date in YYYY-MM-DD format
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get all fixtures for today
    all_fixtures = get_upcoming_fixtures(today)
    
    # If there are more than 50 fixtures, select 50 random ones
    if len(all_fixtures) > 50:
        selected_fixtures = random.sample(all_fixtures, 50)
    else:
        selected_fixtures = all_fixtures
    
    # Predict each match and format the results
    predictions = []
    for fixture in selected_fixtures:
        fixture_id = fixture['fixture']['id']
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        
        # Get both match outcome and goals prediction
        match_prediction = predict_by_last_matches(fixture_id)
        goals_prediction = predict_goals(fixture_id)
        
        predictions.append({
            'Match': f"{home_team} vs {away_team}",
            'Predicted Outcome': match_prediction['prediction'],
            'Score Difference': match_prediction['score_difference'],
            'Goal Prediction': goals_prediction['prediction'],
            'BTTS': goals_prediction['btts'],
            'Home Team Prediction': goals_prediction['home_team_total_prediction'],
            'Away Team Prediction': goals_prediction['away_team_total_prediction'],
            'First to Score': goals_prediction.get('first_home_prediction') or goals_prediction.get('first_away_prediction') or "Uncertain"
        })
    
    return predictions










def predict_all_today_matches():
    fixtures = get_todays_fixtures()
    predictions = []

    for fixture in fixtures:
        fixture_id = fixture['fixture']['id']
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']

        result = predict_goals(fixture_id)

        predictions.append({
            'home': home_team,
            'away': away_team,
            'prediction': result['prediction']
        })

    return predictions




















def get_fixture_by_id(fixture_id):
    url = f"{BASE_URL}/fixtures?id={fixture_id}"
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        print(f"[ERROR] get_fixture_by_id: {e}")
        return []






def get_todays_fixtures():
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"{BASE_URL}/fixtures?date={today}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    
    fixtures = data['response']

    # Filter only fixtures that haven't started or are in-play
    upcoming_fixtures = [
        fixture for fixture in fixtures
        if fixture['fixture']['status']['short'] in ['NS', '1H', '2H', 'ET', 'P', 'BT', 'INT']
    ]

    return upcoming_fixtures






def get_fixture_statistics(fixture_id):
    url = f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}"
    
    print(f"[DEBUG] Fetching stats from: {url}")  # Debug the exact URL being called
    
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        print(f"[DEBUG] Response status: {res.status_code}")  # Debug status code
        
        res.raise_for_status()
        data = res.json()
        
        print(f"[DEBUG] Raw API response: {data}")  # Debug raw response
        
        # Check if the response has the expected structure
        if not isinstance(data.get("response"), list):
            print(f"[WARNING] Unexpected response format: {data}")
            return []
            
        return data["response"]
        
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP Error for fixture {fixture_id}: {e}")
        print(f"Response content: {e.response.text if hasattr(e, 'response') else 'None'}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed for fixture {fixture_id}: {e}")
        return []
    except ValueError as e:
        print(f"[ERROR] JSON decode failed for fixture {fixture_id}: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error for fixture {fixture_id}: {e}")
        return []



def get_fixture_lineups(fixture_id):
    url = f"{BASE_URL}/fixtures/lineups?fixture={fixture_id}"
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        print(f"[ERROR] get_fixture_lineups: {e}")
        return []







def get_head_to_head(team1_id, team2_id, limit=5):
    url = f"{BASE_URL}/fixtures/headtohead?h2h={team1_id}-{team2_id}&last={limit}"
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        print(f"[ERROR] get_head_to_head: {e}")
        return []


def get_league_standings(league_id, season):
    url = f"{BASE_URL}/standings?league={league_id}&season={season}"
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        print(f"[ERROR] get_league_standings: {e}")
        return []

def get_last_fixtures(team_id, count=5):
    url = f"{BASE_URL}/fixtures?team={team_id}&last={count}"
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        print(f"[ERROR] get_last_fixtures: {e}")
        return []




def get_fixture_events(fixture_id):
    response = requests.get(
        f"{BASE_URL}/fixtures/events?fixture={fixture_id}",
        headers=HEADERS
    )
    return response.json().get('response', [])

def get_fixture_player_stats(fixture_id):
    response = requests.get(
        f"{BASE_URL}/fixtures/players?fixture={fixture_id}",
        headers=HEADERS
    )
    return response.json().get('response', [])


def get_team_injuries(team_id, league_id, season):
    response = requests.get(
        f"{BASE_URL}/injuries?team={team_id}&league={league_id}&season={season}",
        headers=HEADERS
    )
    return response.json().get('response', [])