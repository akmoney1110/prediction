from django.shortcuts import render

# predictions/views.py

from django.shortcuts import render
from django.http import HttpResponse

from .rules_engine import predict_winner

def predict_fixture(request, fixture_id, home_id, away_id):
    season = 2023
    league = 39  # EPL

    # Fetch stats
    home_data = get_team_stats(home_id, league, season)
    away_data = get_team_stats(away_id, league, season)
    h2h_data = get_h2h_matches(home_id, away_id)

    # Extract the real data from the API response
    home_stats = home_data.get("response", {})
    away_stats = away_data.get("response", {})
    h2h_matches = h2h_data.get("response", [])

    if not home_stats or not away_stats:
        return HttpResponse("Could not fetch team stats", status=500)

    # Predict using the separate rule engine
    prediction = predict_winner(home_stats, away_stats, h2h_matches)

    return render(request, 'prediction_result.html', {
        'prediction': prediction,
        'home': home_stats['team']['name'],
        'away': away_stats['team']['name'],
    })















from django.shortcuts import render
from .api_football import predict_by_last_matches,get_today_matches_with_scores

def predict_fixtures(request, fixture_id):
    season = 2023  # or dynamically determine
    result = predict_by_last_matches(fixture_id, season)

    # Ensure the context is always a dictionary
    if isinstance(result, str):
        return render(request, 'result.html', {"error": result})

    return render(request, 'result.html', result)









from django.shortcuts import render
from .api_football import get_upcoming_fixtures, predict_by_last_matches

def fixtureslist(request):
    league_id = 39  # EPL
    season = 2023
    fixtures = get_upcoming_fixtures(league_id, season)
    print("[DEBUG] Fixtures fetched:", len(fixtures))
    return render(request, 'fixtures_list.html', {'fixtures': fixtures})

def predict_fixres(request, fixture_id):
    result = predict_by_last_matches(fixture_id, season=2025)
    return render(request, 'fixtures_list.html', {"result": result})

# views.py
from datetime import datetime, timedelta
from django.shortcuts import render
from .api_football import get_upcoming_fixtures

from django.shortcuts import render
from datetime import datetime, timedelta


def fixtures_ist(request):
    today = datetime.utcnow().date()
    tomorrow = today + timedelta(days=1)

    today_str = today.strftime('%Y-%m-%d')
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')

    today_fixtures = get_upcoming_fixtures(today_str)
    tomorrow_fixtures = get_upcoming_fixtures(tomorrow_str)

    context = {
        "today": today_str,
        "tomorrow": tomorrow_str,
        "today_fixtures": today_fixtures,
        "tomorrow_fixtures": tomorrow_fixtures
    }

    return render(request, 'fixtures_list.html', context)



def predict_fixturs(request, fixture_id):
    season = 2023
    result = predict_by_last_matches(fixture_id, season)
    return render(request, 'prediction_results.html', result)

from .api_football import predict_by_last_matches

from .api_football import predict_by_last_matches

# views.py
# views.py
def predict_fixturr(request, fixture_id, season):
    result = predict_by_last_matches(fixture_id)
    return render(request, "result.html", {"result": result})


from .api_football import predict_by_last_matches, predict_goals

def predict_fixturer(request, fixture_id, season):
    result_1x2 = predict_by_last_matches(fixture_id)
    result_goals = predict_goals(fixture_id)
    
    return render(request, "result.html", {
        "result": result_1x2,
        "goals": result_goals
    })




from datetime import datetime
from django.shortcuts import render
from .api_football import (predict_by_last_matches, predict_goals, 
                          get_fixture_info, get_last_team_matches,
                          get_last_h2h_matches, get_team_injuries_and_impact,fatigue_penalty)

def predict_fixtur(request, fixture_id, season):
    # Get prediction results
    result_1x2 = predict_by_last_matches(fixture_id)
    result_goals = predict_goals(fixture_id)
    
    # Get additional fixture data
    fixture = get_fixture_info(fixture_id)
    
    # Calculate win probabilities
    total_score = result_1x2['home_score'] + result_1x2['away_score']
    home_win_prob = round((result_1x2['home_score'] / total_score * 100), 1) if total_score > 0 else 0
    away_win_prob = round((result_1x2['away_score'] / total_score * 100), 1) if total_score > 0 else 0
    draw_prob = round((20 / total_score * 100), 1) if total_score > 0 else 0  # Your draw calculation logic
    
    # Get team recent matches
    home_matches = get_last_team_matches(fixture['teams']['home']['id'], 'home')
    away_matches = get_last_team_matches(fixture['teams']['away']['id'], 'away')
    
    # Get head-to-head matches
    h2h_matches = get_last_h2h_matches(
        fixture['teams']['home']['id'], 
        fixture['teams']['away']['id']
    )
    
    # Get injury data
    home_injury_impact = get_team_injuries_and_impact(fixture['teams']['home']['id'])
    away_injury_impact = get_team_injuries_and_impact(fixture['teams']['away']['id'])
    
    # Format fixture date
    try:
        fixture_date = datetime.strptime(fixture['fixture']['date'], "%Y-%m-%dT%H:%M:%S%z")
        formatted_date = fixture_date.strftime("%b %d, %Y %H:%M")
    except:
        formatted_date = fixture['fixture']['date']
    
    context = {
        "result": result_1x2,
        "goals": result_goals,
        "fixture": fixture,
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "draw_prob": draw_prob,
        "home_matches": home_matches,
        "away_matches": away_matches,
        "h2h_matches": h2h_matches,
        "home_injury_impact": home_injury_impact,
        "away_injury_impact": away_injury_impact,
        "formatted_date": formatted_date,
        "home_penalty": fatigue_penalty(fixture['teams']['home']['id']),
        "away_penalty": fatigue_penalty(fixture['teams']['away']['id']),
        "current_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    return render(request, "result.html", context)






from django.shortcuts import render
from datetime import datetime, timedelta
from .api_football import get_upcoming_fixtures,get_live_matches
from dateutil.parser import parse
import pytz

def fixture_list(request):
    today = datetime.utcnow().date()
    week_dates = [today + timedelta(days=i) for i in range(7)]  # Next 7 days
    
    # Fetch fixtures for each day
    fixtures_by_day = {}
    local_tz = pytz.timezone('Africa/Lagos')  # Adjust timezone as needed
    
    for date in week_dates:
        date_str = date.strftime('%Y-%m-%d')
        fixtures = get_upcoming_fixtures(date_str)
        
        # Convert UTC time to local timezone
        for match in fixtures:
            utc_dt = parse(match['fixture']['date'])
            match['fixture']['datetime_obj'] = utc_dt.astimezone(local_tz)
        
        fixtures_by_day[date_str] = fixtures
    
    context = {
        "week_dates": week_dates,
        "fixtures_by_day": fixtures_by_day,
        "today": today.strftime('%Y-%m-%d'),
    }
    
    return render(request, 'fixtures_list.html', context)



def fixturs_list(request):
    today = datetime.utcnow().date()
    week_dates = [today + timedelta(days=i) for i in range(7)]
    
    fixtures_by_day = {}
    local_tz = pytz.timezone('Africa/Lagos')
    
    for date in week_dates:
        date_str = date.strftime('%Y-%m-%d')
        fixtures = get_upcoming_fixtures(date_str)
        
        for match in fixtures:
            utc_dt = parse(match['fixture']['date'])
            match['fixture']['datetime_obj'] = utc_dt.astimezone(local_tz)
            
            # Use country from API if available
            if 'country' in match['teams']['home']:
                match['teams']['home']['country_code'] = match['teams']['home']['country'].lower()
            else:
                match['teams']['home']['country_code'] = get_country_code(match['teams']['home']['name'])
            
            if 'country' in match['teams']['away']:
                match['teams']['away']['country_code'] = match['teams']['away']['country'].lower()
            else:
                match['teams']['away']['country_code'] = get_country_code(match['teams']['away']['name'])
            
            # Logo handling remains the same
            if 'id' in match['teams']['home']:
                match['teams']['home']['logo'] = get_team_logo(match['teams']['home']['id'])
            if 'id' in match['teams']['away']:
                match['teams']['away']['logo'] = get_team_logo(match['teams']['away']['id'])
        
        fixtures_by_day[date_str] = fixtures
    
    context = {
        "week_dates": week_dates,
        "fixtures_by_day": fixtures_by_day,
        "today": today.strftime('%Y-%m-%d'),
    }
    return render(request, 'fixtures_list.html', context)



def is_upcoming(match):
    """Return True if match hasn't started yet."""
    return match.get('fixture', {}).get('status', {}).get('short') == 'NS'


from datetime import datetime, timedelta
import pytz
from dateutil.parser import parse
from django.shortcuts import render

def fixtures_list(request):
    today = datetime.utcnow().date()
    week_dates = [today + timedelta(days=i) for i in range(7)]
    fixtures_by_day = {}

    # Use UTC for filtering (consistent for all users)
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)

    # Timezone for display (you can also make this dynamic later)
    display_tz = pytz.timezone('Africa/Lagos')

    for date in week_dates:
        date_str = date.strftime('%Y-%m-%d')

        all_fixtures = get_upcoming_fixtures(date_str)
        valid_fixtures = []

        for match in all_fixtures:
            try:
                kickoff_utc = parse(match['fixture']['date']).astimezone(pytz.UTC)
                if kickoff_utc > now_utc:
                    # Add datetime object for display purposes
                    match['fixture']['datetime_obj'] = kickoff_utc.astimezone(display_tz)

                    # Add country code for flags
                    home = match['teams']['home']
                    away = match['teams']['away']

                    home['country_code'] = (
                        home.get('country', '') or get_country_code(home['name'])
                    ).lower()
                    away['country_code'] = (
                        away.get('country', '') or get_country_code(away['name'])
                    ).lower()

                    # Add logos
                    if 'id' in home:
                        home['logo'] = get_team_logo(home['id'])
                    if 'id' in away:
                        away['logo'] = get_team_logo(away['id'])

                    valid_fixtures.append(match)
            except Exception as e:
                print(f"⚠️ Error processing match: {e}")

        if valid_fixtures:
            fixtures_by_day[date_str] = valid_fixtures

    context = {
        "week_dates": week_dates,
        "fixtures_by_day": fixtures_by_day,
        "today": today.strftime('%Y-%m-%d'),
    }
    return render(request, 'fixtures_list.html', context)















# Add this at the top of your views.py
from django.conf import settings
import os
import json

# Country code mapping - you'll need to customize this
COUNTRY_CODE_MAP = {
    'england': 'gb-eng',
    'scotland': 'gb-sct',
    'wales': 'gb-wls',
    'france': 'fr',
    'germany': 'de',
    'spain': 'es',
    'italy': 'it',
    'brazil': 'br',
    'argentina': 'ar',
    # Add more mappings as needed
}

def get_country_code(team_name):
    """
    Extract country code from team name or league
    You'll need to customize this based on your data structure
    """
    team_name_lower = team_name.lower()
    
    # First try to find direct matches
    for country, code in COUNTRY_CODE_MAP.items():
        if country in team_name_lower:
            return code
    
    # Special cases for clubs
    if 'manchester' in team_name_lower:
        return 'gb-eng'
    if 'liverpool' in team_name_lower:
        return 'gb-eng'
    if 'celtic' in team_name_lower or 'rangers' in team_name_lower:
        return 'gb-sct'
    
    # Default to the most common in your data
    return 'gb-eng'  # Default to England if unknown

def get_team_logo(team_data):
    """
    Get team logo from API-Football response
    team_data should be the team dictionary from the API response
    """
    if not team_data:
        return None
    
    # Check for logo in different possible keys (API-Football has changed these over time)
    for key in ['logo', 'image_path', 'img', 'crest']:
        if key in team_data and team_data[key]:
            return team_data[key]
    
    return None






def fixtures_lt(request):
    today = datetime.utcnow().date()
    week_dates = [today + timedelta(days=i) for i in range(7)]
    
    fixtures_by_day = {}
    local_tz = pytz.timezone('Africa/Lagos')
    
    for date in week_dates:
        date_str = date.strftime('%Y-%m-%d')
        fixtures = get_upcoming_fixtures(date_str)
        
        for match in fixtures:
            utc_dt = parse(match['fixture']['date'])
            match['fixture']['datetime_obj'] = utc_dt.astimezone(local_tz)
            
            # Logos are already included from the API
            # Just ensure we have country codes
            match['teams']['home']['country_code'] = get_country_code(match['teams']['home']['name'])
            match['teams']['away']['country_code'] = get_country_code(match['teams']['away']['name'])
        
        fixtures_by_day[date_str] = fixtures
    
    context = {
        "week_dates": week_dates,
        "fixtures_by_day": fixtures_by_day,
        "today": today.strftime('%Y-%m-%d'),
    }
    return render(request, 'fixtures_list.html', context)


from django.shortcuts import render
from .api_football import get_upcoming_fixtures, predict_by_last_matches
from datetime import datetime

def predict_10_wins_view(request):
    today = datetime.today().strftime('%Y-%m-%d')
    fixtures = get_upcoming_fixtures(today)

    results = []
    for fixture in fixtures[:10]:
        prediction = predict_by_last_matches(fixture['fixture']['id'])

        if 'Draw' in prediction['prediction'] or prediction['prediction'] == '1X':
            results.append(prediction)
        
        if len(results) == 10:
            break

    return render(request, 'predict_10_wins.html', {'predictions': results})












from django.views.decorators.cache import cache_page

@cache_page(60)  # Cache for 1 minute
def live_scores(request):
    live_matches = get_live_matches()
    today_matches = get_today_matches_with_scores()
    
    # Process matches to add status information
    for match in live_matches + today_matches:
        match['fixture']['status_display'] = get_status_display(match['fixture']['status'])
        match['fixture']['current_time'] = get_current_match_time(match['fixture']['status'])
    
    context = {
        'live_matches': live_matches,
        'today_matches': today_matches,
        'last_updated': datetime.now().strftime('%H:%M:%S')
    }
    return render(request, 'live_scores.html', context)

def get_status_display(status):
    """Convert API status to user-friendly text"""
    status_map = {
        'NS': 'Not Started',
        '1H': '1st Half',
        'HT': 'Half Time',
        '2H': '2nd Half',
        'ET': 'Extra Time',
        'P': 'Penalty',
        'FT': 'Full Time',
        'AET': 'After Extra Time',
        'PEN': 'Penalties',
        'BT': 'Break Time',
        'SUSP': 'Suspended',
        'INT': 'Interrupted',
        'PST': 'Postponed',
        'CANC': 'Cancelled',
        'ABD': 'Abandoned',
        'AWD': 'Awarded',
        'WO': 'Walkover'
    }
    return status_map.get(status['short'], status['long'])

def get_current_match_time(status):
    """Get current match time if available"""
    if status['elapsed']:
        return f"{status['elapsed']}'"
    return ""




def get_finished_matches(matches):
    """Filter matches to only include finished ones"""
    return [
        match for match in matches
        if match.get('fixture', {}).get('status', {}).get('short') in {'FT', 'AET', 'PEN'}
    ]


from django.views.decorators.cache import cache_page
from datetime import datetime, timedelta
from dateutil.parser import parse
import pytz

@cache_page(60)  # Cache for 1 minute
def fixtures_list(request):
    # Get live and today's matches
    live_matches = get_live_matches()
    today_matches = get_today_matches_with_scores()
    
    # Get finished matches from today
    finished_matches = get_finished_matches(today_matches)
    
    # Process all matches
    for match in live_matches + today_matches:
        match['fixture']['status_display'] = get_status_display(match['fixture']['status'])
        match['fixture']['current_time'] = get_current_match_time(match['fixture']['status'])
        match['fixture']['datetime_obj'] = parse(match['fixture']['date']).astimezone(pytz.timezone('Africa/Lagos'))
        
        # Add country codes
        match['teams']['home']['country_code'] = get_country_code(match['teams']['home']['name'])
        match['teams']['away']['country_code'] = get_country_code(match['teams']['away']['name'])
    
    # Get upcoming fixtures for the week
    today = datetime.utcnow().date()
    week_dates = [today + timedelta(days=i) for i in range(7)]
    fixtures_by_day = {}
    
    for date in week_dates:
        date_str = date.strftime('%Y-%m-%d')
        fixtures = get_upcoming_fixtures(date_str)
        
        for match in fixtures:
            utc_dt = parse(match['fixture']['date'])
            match['fixture']['datetime_obj'] = utc_dt.astimezone(pytz.timezone('Africa/Lagos'))
            match['fixture']['status_display'] = get_status_display(match['fixture']['status'])
            match['teams']['home']['country_code'] = get_country_code(match['teams']['home']['name'])
            match['teams']['away']['country_code'] = get_country_code(match['teams']['away']['name'])
        
        fixtures_by_day[date_str] = fixtures
    
    context = {
        'live_matches': live_matches,
        'today_matches': today_matches,
        'finished_matches': finished_matches,  # Add this line
        'last_updated': datetime.now().strftime('%H:%M:%S'),
        'week_dates': week_dates,
        'fixtures_by_day': fixtures_by_day,
        'today': today.strftime('%Y-%m-%d'),
    }
    return render(request, 'fixtures_list.html', context)






def get_status_display(status):
    """Convert API status to user-friendly text"""
    status_map = {
        'NS': 'Not Started',
        '1H': '1st Half',
        'HT': 'Half Time',
        '2H': '2nd Half',
        'ET': 'Extra Time',
        'P': 'Penalty',
        'FT': 'Full Time',
        'AET': 'After Extra Time',
        'PEN': 'Penalties',
        'BT': 'Break Time',
        'SUSP': 'Suspended',
        'INT': 'Interrupted',
        'PST': 'Postponed',
        'CANC': 'Cancelled',
        'ABD': 'Abandoned',
        'AWD': 'Awarded',
        'WO': 'Walkover'
    }
    return status_map.get(status['short'], status['long'])

def get_current_match_time(status):
    """Get current match time if available"""
    if status['elapsed']:
        return f"{status['elapsed']}'"
    return ""











from django.shortcuts import render
from datetime import datetime
from .api_football import (
    get_fixture_by_id, get_fixture_statistics,predict_all_today_matches,
    get_fixture_lineups, get_head_to_head,
    get_last_fixtures, get_league_standings,get_fixture_events, get_fixture_player_stats,get_team_injuries,
)
def live_match_analysis(request, fixture_id):
    fixture = get_fixture_by_id(fixture_id)
    if not fixture:
        return render(request, "error.html", {"message": "Fixture not found."})

    fixture = fixture[0]
    home_id = fixture['teams']['home']['id']
    away_id = fixture['teams']['away']['id']
    league_id = fixture['league']['id']
    season = fixture['league']['season']
    
    # Initialize counters
    home_wins = 0
    away_wins = 0
    draws = 0
    
    # Get head-to-head data (remove the trailing comma)
    head_to_head = get_head_to_head(home_id, away_id)
    
    # Calculate head-to-head stats with proper error handling
    if head_to_head and isinstance(head_to_head, list):
        for match in head_to_head:
            try:
                # Check if match has the expected structure
                if not all(key in match for key in ['teams', 'goals']):
                    continue
                    
                if match['teams']['home']['id'] == home_id and match['goals']['home'] > match['goals']['away']:
                    home_wins += 1
                elif match['teams']['away']['id'] == home_id and match['goals']['away'] > match['goals']['home']:
                    home_wins += 1
                elif match['teams']['home']['id'] == away_id and match['goals']['home'] > match['goals']['away']:
                    away_wins += 1
                elif match['teams']['away']['id'] == away_id and match['goals']['away'] > match['goals']['home']:
                    away_wins += 1
                elif match['goals']['home'] == match['goals']['away']:
                    draws += 1
            except (KeyError, TypeError):
                # Skip any malformed match data
                continue


    # In your view
    raw_stats = get_fixture_statistics(fixture_id)
    statistics = []
    stats_available = False

    if raw_stats and isinstance(raw_stats, list):
        try:
            # New structure processing
            for team_stats in raw_stats:
                if 'statistics' in team_stats:
                    team_id = team_stats['team']['id']
                    for stat in team_stats['statistics']:
                        # Find or create the stat type entry
                        existing = next((s for s in statistics if s['type'] == stat['type']), None)
                        if not existing:
                            existing = {'type': stat['type'], 'home': None, 'away': None}
                            statistics.append(existing)
                        
                        # Assign to home or away
                        if team_id == home_id:
                            existing['home'] = stat['value']
                        elif team_id == away_id:
                            existing['away'] = stat['value']
            
            stats_available = len(statistics) > 0
        except Exception as e:
            print(f"Error processing stats: {e}")



    context = {
        "fixture": fixture,
        "live_matches":  get_live_matches(),
        "statistics": get_fixture_statistics(fixture_id),
        "lineups": get_fixture_lineups(fixture_id),
        "head_to_head": head_to_head,  # Use the variable we already fetched
        "home_recent": get_last_fixtures(home_id),
        "away_recent": get_last_fixtures(away_id),
        "standings": get_league_standings(league_id, season),
        "events": get_fixture_events(fixture_id),
        "player_stats": get_fixture_player_stats(fixture_id),
        "home_injuries": get_team_injuries(home_id, league_id, season),
        "away_injuries": get_team_injuries(away_id, league_id, season),
        'h2h_home_wins': home_wins,
        'h2h_away_wins': away_wins,
        'h2h_draws': draws,
        "statistics": statistics,
        "stats_available": stats_available,
    }
    print(statistics)
    print("Raw statistics data:", raw_stats)
    return render(request, "live_match_analysis.html", context)







def live_matc_analysis(request, fixture_id):
    fixture = get_fixture_by_id(fixture_id)
    if not fixture:
        return render(request, "error.html", {"message": "Fixture not found."})

    fixture = fixture[0]
    home_id = fixture['teams']['home']['id']
    away_id = fixture['teams']['away']['id']
    league_id = fixture['league']['id']
    season = fixture['league']['season']
    fixture_status = fixture.get('status', {}).get('short', 'NS')

    # Initialize statistics variables
    statistics = None
    stats_available = False
    stats_message = "Statistics not available"

    # Try to get statistics with error handling
    try:
        raw_stats = get_fixture_statistics(fixture_id)
       
        
        if raw_stats and isinstance(raw_stats, list):
            if len(raw_stats) >= 2:  # Expected format with home and away stats
                home_stats = raw_stats[0].get('statistics', [])
                away_stats = raw_stats[1].get('statistics', [])
                
                if home_stats or away_stats:
                    statistics = []
                    away_stats_dict = {stat['type']: stat['value'] for stat in away_stats}
                    
                    for home_stat in home_stats:
                        stat_type = home_stat.get('type')
                        if stat_type:
                            statistics.append({
                                'type': stat_type,
                                'home': home_stat.get('value'),
                                'away': away_stats_dict.get(stat_type)
                            })
                    stats_available = bool(statistics)
        
        # Determine appropriate message
        if fixture_status == 'NS':
            stats_message = "Statistics will appear after match starts"
        elif not stats_available:
            stats_message = "Statistics not available for this match"
        else:
            stats_message = ""

    except Exception as e:
        print(f"Error fetching statistics: {e}")
        stats_message = "Error loading statistics"

    context = {
        "fixture": fixture,
        "statistics": statistics,
        "stats_available": stats_available,
        "stats_message": stats_message,
        "fixture_status": fixture_status,
        "lineups": get_fixture_lineups(fixture_id),
        "head_to_head": get_head_to_head(home_id, away_id),
        "home_recent": get_last_fixtures(home_id),
        "away_recent": get_last_fixtures(away_id),
        "standings": get_league_standings(league_id, season),
        "events": get_fixture_events(fixture_id),
        "player_stats": get_fixture_player_stats(fixture_id),
        "home_injuries": get_team_injuries(home_id, league_id, season),
        "away_injuries": get_team_injuries(away_id, league_id, season),
    }
    print(f"Statistics API response for fixture {fixture_id}:", raw_stats)  # Debug
    return render(request, "live_match_analysis.html", context)












from django.shortcuts import render

def all_predictions_view(request):
    predictions = predict_all_today_matches()
    return render(request, 'today_predictions.html', {'predictions': predictions})






















from django.shortcuts import render
from datetime import datetime
import random
from .api_football import get_upcoming_fixtures, predict_by_last_matches, predict_goals

def todays_predicions(request):
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get all fixtures for today
    all_fixtures = get_upcoming_fixtures(today)
    
    # Select 50 random fixtures or all if less than 50
    selected_fixtures = random.sample(all_fixtures, min(2, len(all_fixtures))) if all_fixtures else []
    
    predictions = []
    for fixture in selected_fixtures:
        fixture_id = fixture['fixture']['id']
        
        # Get predictions
        match_pred = predict_by_last_matches(fixture_id)
        goals_pred = predict_goals(fixture_id)
    
        
        predictions.append({
            'home_team': fixture['teams']['home']['name'],
            'away_team': fixture['teams']['away']['name'],
            'home_logo': fixture['teams']['home']['logo'],
            'away_logo': fixture['teams']['away']['logo'],
            'outcome': match_pred['prediction'],
            'confidence': match_pred.get('score_difference', 0),
            'total_goals': goals_pred['prediction'],
            'btts': goals_pred['btts'],
            'home_goals_pred': goals_pred['home_team_total_prediction'],
            'away_goals_pred': goals_pred['away_team_total_prediction'],
            'first_to_score': goals_pred.get('first_home_prediction') or 
                             goals_pred.get('first_away_prediction') or "Uncertain"
        })
    
    for pred in predictions:
        pred.confidence_times_10 = pred.confidence * 10    
    context = {
        'predictions': predictions,
        'prediction_date': datetime.now().strftime('%A, %B %d, %Y'),
        'total_matches': len(predictions)
    }
    
    return render(request, 'todays_predictions.html', context)


tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

import logging
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)
def todays_predictions(request):
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Get all fixtures for today
        all_fixtures = get_upcoming_fixtures(today)
        
        # Select 50 random fixtures or all if less than 50
        selected_fixtures = random.sample(all_fixtures, min(1, len(all_fixtures))) if all_fixtures else []
        
        predictions = []
        for fixture in selected_fixtures:
            if not fixture or 'fixture' not in fixture or 'teams' not in fixture:
                continue
                
            fixture_id = fixture['fixture']['id']
            
            # Get predictions
            match_pred = predict_by_last_matches(fixture_id) or {}
            goals_pred = predict_goals(fixture_id) or {}
            
            # Calculate confidence percentage (0-100)
            confidence = match_pred.get('score_difference', 0)
            confidence_percentage = min(max(confidence * 10, 0), 100)  # Ensure between 0-100
            
            predictions.append({
                'home_team': fixture['teams']['home'].get('name', 'Unknown'),
                'away_team': fixture['teams']['away'].get('name', 'Unknown'),
                'home_logo': fixture['teams']['home'].get('logo', ''),
                'away_logo': fixture['teams']['away'].get('logo', ''),
                'league': fixture.get('league', {}).get('name', 'Unknown League'),
                'time': fixture['fixture'].get('date', '').split('T')[1][:5] if 'date' in fixture['fixture'] else 'TBD',
                'outcome': match_pred.get('prediction', 'No prediction'),
                'confidence': confidence,
                'confidence_percentage': confidence_percentage,
                'total_goals': goals_pred.get('prediction', 'Unknown'),
                'btts': goals_pred.get('btts', 'Unknown'),
                'home_goals_pred': goals_pred.get('home_team_total_prediction', 'Unknown'),
                'away_goals_pred': goals_pred.get('away_team_total_prediction', 'Unknown'),
                'first_to_score': (goals_pred.get('first_home_prediction') or 
                                 goals_pred.get('first_away_prediction') or "Uncertain"),
                'home_total': goals_pred.get('home_total', 0),
                'away_total': goals_pred.get('away_total', 0),
                'home_clean_sheets': goals_pred.get('home_total_clean', 0),
                'away_clean_sheets': goals_pred.get('away_total_clean', 0),
                'country': fixture.get('league', {}).get('country', 'Unknown Country'),
            })
        
        context = {
            'predictions': predictions,
            'prediction_date': datetime.now().strftime('%A, %B %d, %Y'),
            'total_matches': len(predictions)
        }
        
        return render(request, 'todays_predictions.html', context)
        
    except Exception as e:
        logger.error(f"Error in todays_predictions: {str(e)}")
        return render(request, 'error.html', {'error': str(e)})