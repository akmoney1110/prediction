from matches.models import Match, Lineup

def check_lineup_exists(match_id):
    """Check if lineup data exists for a specific match"""
    try:
        match = Match.objects.get(id=match_id)
        
        # Check if any lineup exists for this match
        lineup_exists = Lineup.objects.filter(match=match).exists()
        
        # If you want to check for both home and away lineups
        home_lineup_exists = Lineup.objects.filter(match=match, team=match.home).exists()
        away_lineup_exists = Lineup.objects.filter(match=match, team=match.away).exists()
        
        return {
            'any_lineup': lineup_exists,
            'home_lineup': home_lineup_exists,
            'away_lineup': away_lineup_exists,
            'both_lineups': home_lineup_exists and away_lineup_exists
        }
        
    except Match.DoesNotExist:
        return {'error': 'Match not found'}

# Usage
match_id = 123456  # Your fixture ID
lineup_status = check_lineup_exists(match_id)
print(f"Lineup status: {lineup_status}")