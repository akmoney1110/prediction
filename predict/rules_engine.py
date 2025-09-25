# predictions/rules_engine.py

def predict_winner(home_stats, away_stats, h2h_data):
    score = 0

    # Rule 1: Higher rank
    home_rank = home_stats['league'].get('rank')
    away_rank = away_stats['league'].get('rank')
    if home_rank and away_rank:
        if home_rank < away_rank:
            score += 1

    # Rule 2: Better form
    home_form = home_stats.get('form', '').count('W')
    away_form = away_stats.get('form', '').count('W')
    if home_form > away_form:
        score += 1

    # Rule 3: Home advantage
    score += 1

    # Rule 4: Goals per game
    try:
        home_goals = float(home_stats['goals']['for']['average']['home'])
        away_goals = float(away_stats['goals']['for']['average']['away'])
        if home_goals > away_goals:
            score += 1
    except:
        pass

    # Rule 5: H2H wins
    home_wins = sum(1 for match in h2h_data if match['teams']['home']['winner'])
    away_wins = sum(1 for match in h2h_data if match['teams']['away']['winner'])
    if home_wins > away_wins:
        score += 1

    if score >= 4:
        return "Home Win"
    elif score >= 2:
        return "Draw"
    else:
        return "Away Win"
