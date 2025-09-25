from matches.models import Match, Lineup, StandingsSnapshot
from django.db.models import Q, Count, Case, When, IntegerField, Max
from django.db import models
from datetime import datetime, timedelta
import json

def check_lineup_exists(match_id):
    """
    Check if lineup data exists for a specific match
    """
    try:
        match = Match.objects.get(id=match_id)
        
        lineups = Lineup.objects.filter(match=match)
        home_lineup = lineups.filter(team=match.home).first()
        away_lineup = lineups.filter(team=match.away).first()
        
        return {
            'match_id': match_id,
            'home_team': match.home.name,
            'away_team': match.away.name,
            'any_lineup_exists': lineups.exists(),
            'home_lineup_exists': home_lineup is not None,
            'away_lineup_exists': away_lineup is not None,
            'both_lineups_exist': home_lineup is not None and away_lineup is not None,
            'total_lineups': lineups.count()
        }
        
    except Match.DoesNotExist:
        return {'error': f'Match with id {match_id} not found'}

def check_standings_for_match(match_id):
    """
    Check standings for the league and season of a specific match
    """
    try:
        match = Match.objects.get(id=match_id)
        return check_standings_exist(match.league_id, match.season)
    except Match.DoesNotExist:
        return {'error': f'Match with id {match_id} not found'}

def check_standings_exist(league_id, season):
    """
    Check if standings exist for a league/season
    """
    try:
        standings = StandingsSnapshot.objects.filter(
            league_id=league_id,
            season=season
        ).order_by('-as_of_date')
        
        latest_standings = standings.first()
        
        return {
            'league_id': league_id,
            'season': season,
            'any_standings_exist': standings.exists(),
            'latest_standings_date': latest_standings.as_of_date if latest_standings else None,
            'latest_standings_exists': latest_standings is not None,
            'days_since_last_standings': (datetime.now().date() - latest_standings.as_of_date).days if latest_standings else None,
            'is_recent_standings': (datetime.now().date() - latest_standings.as_of_date).days <= 7 if latest_standings else False
        }
        
    except Exception as e:
        return {'error': f'Error checking standings: {str(e)}'}

def check_match_data_completeness(match_id):
    """
    Comprehensive check for both lineup and standings data for a match
    Uses the match's own league and season for standings check
    """
    try:
        match = Match.objects.get(id=match_id)
        
        lineup_status = check_lineup_exists(match_id)
        standings_status = check_standings_exist(match.league_id, match.season)
        
        both_lineups_exist = (
            lineup_status.get('home_lineup_exists', False) and 
            lineup_status.get('away_lineup_exists', False)
        )
        
        return {
            'match_id': match_id,
            'league_id': match.league_id,
            'league_name': match.league.name,
            'season': match.season,
            'home_team': match.home.name,
            'away_team': match.away.name,
            'kickoff': match.kickoff_utc,
            'lineup_status': lineup_status,
            'standings_status': standings_status,
            'all_data_available': both_lineups_exist and standings_status.get('latest_standings_exists', False),
            'data_quality_score': {
                'lineups_score': 1.0 if both_lineups_exist else 0.5 if lineup_status.get('any_lineup_exists', False) else 0.0,
                'standings_score': 1.0 if standings_status.get('latest_standings_exists', False) else 0.0,
                'overall_score': (1.0 if both_lineups_exist else 0.5 if lineup_status.get('any_lineup_exists', False) else 0.0) + 
                                (1.0 if standings_status.get('latest_standings_exists', False) else 0.0)
            }
        }
        
    except Match.DoesNotExist:
        return {'error': f'Match with id {match_id} not found'}

def get_latest_match_for_league(league_id):
    """
    Get the most recent match for a league
    """
    try:
        latest_match = Match.objects.filter(
            league_id=league_id
        ).order_by('-kickoff_utc').first()
        
        if latest_match:
            return {
                'match_id': latest_match.id,
                'home_team': latest_match.home.name,
                'away_team': latest_match.away.name,
                'kickoff_utc': latest_match.kickoff_utc,
                'season': latest_match.season,
                'status': latest_match.status
            }
        return None
    except Exception as e:
        return {'error': f'Error getting latest match: {str(e)}'}

def check_recent_matches_data(league_id=None, days_back=7):
    """
    Check data completeness for recent matches
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    query = Q(kickoff_utc__gte=cutoff_date)
    if league_id:
        query &= Q(league_id=league_id)
    
    recent_matches = Match.objects.filter(query).order_by('-kickoff_utc')
    
    results = []
    for match in recent_matches:
        completeness = check_match_data_completeness(match.id)
        results.append(completeness)
    
    return results

def get_current_season_for_league(league_id):
    """
    Get the current season for a league based on most recent match
    """
    latest_match = get_latest_match_for_league(league_id)
    if latest_match and not isinstance(latest_match, dict):
        return latest_match.season
    return None

def bulk_check_recent_lineups(league_id=None, days_back=30):
    """
    Check lineup existence for recent matches
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    query = Q(kickoff_utc__gte=cutoff_date)
    if league_id:
        query &= Q(league_id=league_id)
    
    matches = Match.objects.filter(query).prefetch_related('lineups')
    
    results = []
    for match in matches:
        lineups = match.lineups.all()
        home_lineup = lineups.filter(team=match.home).first()
        away_lineup = lineups.filter(team=match.away).first()
        
        results.append({
            'match_id': match.id,
            'date': match.kickoff_utc.date(),
            'home_team': match.home.name,
            'away_team': match.away.name,
            'league_id': match.league_id,
            'season': match.season,
            'has_lineups': lineups.exists(),
            'has_both_lineups': home_lineup is not None and away_lineup is not None,
            'lineup_count': lineups.count()
        })
    
    return results

def check_standings_for_recent_matches(league_id, days_back=60):
    """
    Check standings for all seasons that have recent matches
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    # Get all seasons with recent matches
    recent_seasons = Match.objects.filter(
        league_id=league_id,
        kickoff_utc__gte=cutoff_date
    ).values('season').distinct()
    
    results = []
    for season_data in recent_seasons:
        season = season_data['season']
        standings_status = check_standings_exist(league_id, season)
        results.append(standings_status)
    
    return results

def analyze_recent_data_coverage(league_id=None, days_back=90):
    """
    Analyze data coverage for recent matches
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    query = Q(kickoff_utc__gte=cutoff_date)
    if league_id:
        query &= Q(league_id=league_id)
    
    matches = Match.objects.filter(query).annotate(
        lineup_count=Count('lineups'),
        has_home_lineup=Count(Case(
            When(lineups__team_id=models.F('home_id'), then=1),
            output_field=IntegerField()
        )),
        has_away_lineup=Count(Case(
            When(lineups__team_id=models.F('away_id'), then=1),
            output_field=IntegerField()
        ))
    )
    
    total_matches = matches.count()
    matches_with_lineups = matches.filter(lineup_count__gt=0).count()
    matches_with_both_lineups = matches.filter(has_home_lineup__gt=0, has_away_lineup__gt=0).count()
    
    # Check standings for the seasons in recent matches
    recent_seasons = matches.values('season').distinct()
    standings_coverage = []
    
    for season_data in recent_seasons:
        season = season_data['season']
        standings_exist = StandingsSnapshot.objects.filter(
            league_id=league_id,
            season=season
        ).exists()
        standings_coverage.append({
            'season': season,
            'standings_exist': standings_exist
        })
    
    return {
        'period_days': days_back,
        'total_matches': total_matches,
        'matches_with_any_lineup': matches_with_lineups,
        'matches_with_both_lineups': matches_with_both_lineups,
        'lineup_coverage_percentage': (matches_with_lineups / total_matches * 100) if total_matches > 0 else 0,
        'both_lineups_coverage_percentage': (matches_with_both_lineups / total_matches * 100) if total_matches > 0 else 0,
        'standings_coverage': standings_coverage,
        'league_id': league_id
    }




# matches/management/commands/check_data.py
from django.core.management.base import BaseCommand
from django.db.models import Q
from datetime import datetime, timedelta
import json

# Import the functions (make sure they're in the same app or adjust the import path)


class Command(BaseCommand):
    help = 'Check availability of lineup and standings data for recent matches'
    
    def add_arguments(self, parser):
        parser.add_argument('--league', type=int, required=True, help='League ID to check')
        parser.add_argument('--days', type=int, default=30, help='Number of days back to check')
        parser.add_argument('--output', type=str, help='Output file for results')
    
    def handle(self, *args, **options):
        league_id = options.get('league')
        days_back = options.get('days')
        output_file = options.get('output')
        
        results = {
            'league_id': league_id,
            'days_back': days_back,
            'check_date': datetime.now().isoformat(),
            'recent_matches': check_recent_matches_data(league_id, days_back),
            'standings_coverage': check_standings_for_recent_matches(league_id, days_back),
            'coverage_analysis': analyze_recent_data_coverage(league_id, days_back)
        }
        
        # Output results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.stdout.write(self.style.SUCCESS(f'Results saved to {output_file}'))
        else:
            self.stdout.write(json.dumps(results, indent=2, default=str))



# matches/management/commands/check_simple.py
from django.core.management.base import BaseCommand
from django.db.models import Q
from matches.models import Match, Lineup, StandingsSnapshot
from datetime import datetime, timedelta
import json

class Command(BaseCommand):
    help = 'Simple check of lineup and standings data'
    
    def add_arguments(self, parser):
        parser.add_argument('--league', type=int, help='League ID to check')
        parser.add_argument('--days', type=int, default=7, help='Days back to check')
    
    def handle(self, *args, **options):
        league_id = options.get('league')
        days_back = options.get('days')
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Build query
        query = Q(kickoff_utc__gte=cutoff_date)
        if league_id:
            query &= Q(league_id=league_id)
        
        matches = Match.objects.filter(query).order_by('-kickoff_utc')[:10]
        
        results = []
        for match in matches:
            # Check lineups
            lineups = Lineup.objects.filter(match=match)
            home_lineup = lineups.filter(team=match.home).exists()
            away_lineup = lineups.filter(team=match.away).exists()
            
            # Check standings
            standings = StandingsSnapshot.objects.filter(
                league_id=match.league_id,
                season=match.season
            ).exists()
            
            results.append({
                'match_id': match.id,
                'date': match.kickoff_utc.date(),
                'home_team': match.home.name,
                'away_team': match.away.name,
                'league_id': match.league_id,
                'season': match.season,
                'has_home_lineup': home_lineup,
                'has_away_lineup': away_lineup,
                'has_both_lineups': home_lineup and away_lineup,
                'has_standings': standings
            })
        
        self.stdout.write(json.dumps(results, indent=2, default=str))
