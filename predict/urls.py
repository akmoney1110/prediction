from predict.views import predict_fixturer,todays_predictions,all_predictions_view,live_scores,live_match_analysis, predict_fixtures,fixtures_list, predict_fixture,predict_10_wins_view
from django.urls import path





urlpatterns = [

    path("me",predict_fixtures, name="task_status"),
    path('', live_scores, name='live_scores'),
    path('fix', fixtures_list, name='fixtures_list'),
    path('predict/<int:fixture_id>/<int:home_id>/<int:away_id>/', predict_fixture, name='predict_fixture'),
    path('predict/<int:fixture_id>/', predict_fixture, name='predict_fixtures'),
    path('predict/<int:fixture_id>/<int:season>/', predict_fixturer, name='predict_fixturer'),
    path('predict-10-wins/', predict_10_wins_view, name='predict_10_wins'),
    path('live/', live_scores, name='live_scores'),
    path('ready/', all_predictions_view, name='all_predictions_view'),
    path("match/<int:fixture_id>/", live_match_analysis, name="live_match_analysis"),
    path('todays-predictions/', todays_predictions, name='todays_predictions'),

]

