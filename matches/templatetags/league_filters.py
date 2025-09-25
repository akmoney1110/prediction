from django import template

register = template.Library()

LEAGUE_NAMES = {
    39: "Premier League",
    61: "Ligue 1",
    78: "Bundesliga", 
    140: "La Liga",
    135: "Serie A",
    128: "Liga Profesional",
    94: "Primeira Liga",
    88: "Eredivisie",
    144: "Jupiler Pro League",
    283: "MLS",
    # Add more leagues as needed
}

@register.filter
def league_name(league_id):
    return LEAGUE_NAMES.get(league_id, f"League {league_id}")