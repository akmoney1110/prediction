# admin.py
from django.contrib import admin
from .models import StandingsSnapshot, StandingsRow

@admin.register(StandingsSnapshot)
class StandingsSnapshotAdmin(admin.ModelAdmin):
    list_display  = ("league_id", "season", "as_of_date", "created_at")
    list_filter   = ("season",)
    search_fields = ("league_id", "season")

@admin.register(StandingsRow)
class StandingsRowAdmin(admin.ModelAdmin):
    list_display  = ("league_id","season","group_name","rank","team","points","played","gd")
    list_filter   = ("league_id","season","group_name")
    search_fields = ("team__name",)
# prediction/matches/admin.py
from django.contrib import admin
from .models import MatchEvent

@admin.register(MatchEvent)
class MatchEventAdmin(admin.ModelAdmin):
    list_display = ("match", "minute", "type", "detail", "player_name", "is_home")
    list_filter = ("type", "detail", "is_home")
    search_fields = ("player_name", "assist_name", "detail", "comments")
