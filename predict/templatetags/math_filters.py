from django import template

register = template.Library()

@register.filter
def filter_teams_wins(matches, team_id):
    count = 0
    for match in matches:
        if (match['teams']['home']['id'] == team_id and match['goals']['home'] > match['goals']['away']) or \
           (match['teams']['away']['id'] == team_id and match['goals']['away'] > match['goals']['home']):
            count += 1
    return count

@register.filter
def filter_draws(matches):
    count = 0
    for match in matches:
        if match['goals']['home'] == match['goals']['away']:
            count += 1
    return count