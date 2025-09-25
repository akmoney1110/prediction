from django.db import models

# Create your models here.
from django.db import models

class League(models.Model):
    """
    API-Football league/competition.
    id matches API-Football league id for simplicity.
    """
    id = models.IntegerField(primary_key=True)  # API-Football league id
    display_name = models.CharField(max_length=128)
    country = models.CharField(max_length=64)
    tier = models.SmallIntegerField(default=1)
    has_xg = models.BooleanField(default=False)
    has_multi_group = models.BooleanField(default=False)
    notes = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.display_name} ({self.country})"


class Team(models.Model):
    """
    API-Football team.
    """
    id = models.IntegerField(primary_key=True)  # API-Football team id
    league = models.ForeignKey(League, on_delete=models.PROTECT, related_name="teams")
    name = models.CharField(max_length=128)
    short_name = models.CharField(max_length=32, blank=True, null=True)
    logo_url = models.URLField(blank=True, null=True)
    venue_lat = models.FloatField(blank=True, null=True)
    venue_lon = models.FloatField(blank=True, null=True)

    def __str__(self):
        return self.name
