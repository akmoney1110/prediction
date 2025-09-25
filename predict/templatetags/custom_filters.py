# yourapp/templatetags/custom_filters.py

from django import template
register = template.Library()

@register.filter
def dict_get(stat_list, key):
    """
    Retrieves a value from a list of dictionaries where 'type' == key.
    Used for comparing stats between two teams.
    """
    if not isinstance(stat_list, list):
        return None
    for stat in stat_list:
        if stat.get('type') == key:
            return stat.get('value')
    return None
# myapp/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def filter_teams_wins(matches, team_id):
    return len([m for m in matches if (
        (m.home_team_id == team_id and m.home_score > m.away_score) or
        (m.away_team_id == team_id and m.away_score > m.home_score)
    )])

@register.filter
def filter_draws(matches):
    return len([m for m in matches if m.home_score == m.away_score])