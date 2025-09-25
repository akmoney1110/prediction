from django import template

register = template.Library()

@register.filter
def sub(a, b):
    """Return a - b (tolerant of None/blank)."""
    try:
        a = float(a) if a is not None and a != "" else 0.0
        b = float(b) if b is not None and b != "" else 0.0
        # if both were actually missing, return empty so your template can show '-'
        if (a == 0.0 and (a is None or a == "")) and (b == 0.0 and (b is None or b == "")):
            return ""
        return a - b
    except Exception:
        return ""


# matches/templatetags/math_filters.py
from django import template

register = template.Library()

@register.filter
def sub(a, b):
    """Return a - b, tolerating None/blank."""
    try:
        if a in (None, "") or b in (None, ""):
            return ""
        return float(a) - float(b)
    except Exception:
        return ""




from django import template
register = template.Library()

@register.filter
def mul(value, arg):
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''
