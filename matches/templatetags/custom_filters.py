
# predict/templatetags/league_extras.py
from django import template

register = template.Library()

LEAGUE_NAMES = {
    39: "Premier League",
    61: "Ligue 1",
    78: "Bundliga",
    140: "La Liga",
    135: "Serie A",
    128: "Liga Profesional",
    # ...add as needed...
}

@register.filter
def league_name(league_id):
    try:
        lid = int(league_id)
    except (TypeError, ValueError):
        return league_id  # fallback
    return LEAGUE_NAMES.get(lid, league_id)



