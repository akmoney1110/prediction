from django.urls import path
from .views import upcoming_predictions_json,luck,two_odds_builder_page,two_odds_save_api,two_odds_preview_fragment,rollover_rail_json,daily_2_odds_ticket_json,download_daily_2odds_ticket,match_detail,match_detail,combined_dashboard,completed_today_json,dashboard,daily_ticket_json
from . import views_user_tickets as uv


urlpatterns = [
    path("predictions/<int:league_id>/<int:days>/", upcoming_predictions_json, name="upcoming_predictions_json"),
    path("dashboard/<int:league_id>/", dashboard, name="dashboard"),
    path("match/<int:match_id>/", dashboard, name="match_detail"),  # placeholder if you haven’t built detail yet
    path("api/daily-ticket/", daily_ticket_json, name="daily_ticket_json"),
    path("mega/<int:league_id>/<int:days>/", combined_dashboard, name="combined_dashboard"),  # 👈 new
    path("completed-today/<int:league_id>/", completed_today_json),  # 👈
    
    path("mega/match/<int:match_id>/", match_detail, name="match-detail"),
    path("mega/match/<int:pk>/", match_detail, name="match_detail"),
    path("mega/match/<int:fixture_id>/", match_detail, name="match-detail"),
    path("api/daily-2odds-ticket", daily_2_odds_ticket_json, name="daily-2odds-ticket-json"),
    path("tickets/daily-2odds/download", download_daily_2odds_ticket, name="download-daily-2odds-ticket"),
    path("api/rollover/rail", rollover_rail_json, name="rollover-rail-json"),
    path("tickets/2odds/builder", two_odds_builder_page, name="two_odds_builder"),
    path("tickets/2odds/preview-frag", two_odds_preview_fragment, name="two_odds_preview_frag"),
    path("api/tickets/2odds/save", two_odds_save_api, name="two_odds_save"),
    path("tickets/2odds/builder-range", uv.two_odds_builder_page, name="two_odds_builder_range"),
    path("tickets/2odds/preview-range-frag", uv.two_odds_preview_range_fragment, name="two_odds_preview_range_frag"),
    path("api/tickets/2odds/save-range", uv.save_two_odds_range_api, name="two_odds_save_range"),
    path("me/tickets/2odds", uv.my_two_odds_dashboard, name="my_two_odds_dashboard"),
    path("me/tickets/2odds/", uv.my_two_odds_list, name="my_two_odds_list"),
    path("me/tickets/2odds/<int:pk>/", uv.my_two_odds_detail, name="my_two_odds_detail"),
    path("ak", luck)



    
]

