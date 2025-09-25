from django.db import models

# Create your models here.
class Fixture(models.Model):
    league_code = models.CharField(max_length=10)  # e.g. E0, E1, etc
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    match_time = models.DateTimeField()
    # other fields...
