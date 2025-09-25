from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone
from leagues.models import League, Team

STATUS_CHOICES = [
    ("NS", "Not started"), ("1H", "1st Half"), ("HT", "Half-time"), ("2H", "2nd Half"),
    ("ET", "Extra time"), ("P", "Penalties"), ("FT", "Full-time"),
    ("AET", "After extra time"), ("PEN", "Penalties (final)"),
    ("PST", "Postponed"), ("ABD", "Abandoned"), ("SUSP", "Suspended"), ("INT", "Interrupted"),
]

TS_MODE_CHOICES = [("T24", "T minus 24h"), ("T60", "T minus 60m")]


class Match(models.Model):
    """
    One fixture. Use API-Football fixture id as PK.
    Finals are 90' only (your policy).
    """
    id = models.BigIntegerField(primary_key=True)  # fixture id
    league = models.ForeignKey(League, on_delete=models.PROTECT, related_name="matches")
    season = models.IntegerField()  # e.g. 2023 means 2023/24
    home = models.ForeignKey(Team, on_delete=models.PROTECT, related_name="home_matches")
    away = models.ForeignKey(Team, on_delete=models.PROTECT, related_name="away_matches")
    kickoff_utc = models.DateTimeField()
    status = models.CharField(max_length=8, choices=STATUS_CHOICES, default="NS")

    # 90' finals
    goals_home = models.SmallIntegerField(blank=True, null=True)
    goals_away = models.SmallIntegerField(blank=True, null=True)
    corners_home = models.SmallIntegerField(blank=True, null=True)
    corners_away = models.SmallIntegerField(blank=True, null=True)
    cards_home = models.SmallIntegerField(blank=True, null=True)  # Red=1 convention
    cards_away = models.SmallIntegerField(blank=True, null=True)

    raw_result_json = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["league", "season"]),
            models.Index(fields=["kickoff_utc"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self):
        return f"{self.home.name} vs {self.away.name} ({self.kickoff_utc.isoformat()})"





# matches/models.py (add this if you don't already have it)
from django.db import models

class MatchEvent(models.Model):
    match = models.ForeignKey("Match", on_delete=models.CASCADE)
    team  = models.ForeignKey("leagues.Team", null=True, blank=True, on_delete=models.SET_NULL)

    # Core attributes from API-Football
    type    = models.CharField(max_length=32, db_index=True)   # "Goal", "Card", "Subst", "Var"
    detail  = models.CharField(max_length=64, null=True, blank=True)  # e.g. "Normal Goal", "Own Goal", "Penalty", "Yellow Card", "Red Card", "Penalty confirmed", "Goal cancelled"
    comment = models.CharField(max_length=128, null=True, blank=True)

    # Timing
    minute  = models.IntegerField(null=True, blank=True)  # elapsed + extra if present
    elapsed = models.IntegerField(null=True, blank=True)
    extra   = models.IntegerField(null=True, blank=True)

    # Actors
    player_id   = models.IntegerField(null=True, blank=True)
    player_name = models.CharField(max_length=128, null=True, blank=True)
    assist_id   = models.IntegerField(null=True, blank=True)
    assist_name = models.CharField(max_length=128, null=True, blank=True)

    # Convenience flags
    is_home    = models.BooleanField(default=False)
    is_own_goal = models.BooleanField(default=False)
    is_penalty  = models.BooleanField(default=False)
    is_missed_penalty = models.BooleanField(default=False)
    is_var      = models.BooleanField(default=False)

    # Raw for audit/debug
    raw_json = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["match", "minute", "type"]),
        ]

    def __str__(self):
        return f"{self.match_id} {self.type}/{self.detail} {self.player_name or ''} @ {self.minute}"









# matches/models.py
class MatchStats(models.Model):
    """
    Team-split statistics for a match (API-Football).
    Percent fields are stored 0–100 (not 0–1).
    """
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="stats")
    team  = models.ForeignKey(Team,  on_delete=models.PROTECT,  related_name="match_stats")

    # Existing
    shots = models.IntegerField(blank=True, null=True)          # Total Shots
    sot   = models.IntegerField(blank=True, null=True)          # Shots on Target
    possession_pct = models.FloatField(blank=True, null=True)   # 0–100
    pass_acc_pct   = models.FloatField(blank=True, null=True)   # 0–100
    corners = models.IntegerField(blank=True, null=True)
    cards   = models.IntegerField(blank=True, null=True)        # (yellows + reds) if both present
    xg = models.FloatField(blank=True, null=True)
    pens_won = models.IntegerField(blank=True, null=True)
    pens_conceded = models.IntegerField(blank=True, null=True)
    reds = models.IntegerField(blank=True, null=True)
    yellows = models.IntegerField(blank=True, null=True)

    # NEW — to match the full API bar set
    shots_off      = models.IntegerField(blank=True, null=True)   # Shots off Goal / off target
    shots_blocked  = models.IntegerField(blank=True, null=True)   # Blocked Shots
    shots_in_box   = models.IntegerField(blank=True, null=True)   # Shots Insidebox
    shots_out_box  = models.IntegerField(blank=True, null=True)   # Shots Outsidebox
    fouls          = models.IntegerField(blank=True, null=True)
    offsides       = models.IntegerField(blank=True, null=True)
    saves          = models.IntegerField(blank=True, null=True)   # Goalkeeper Saves
    passes_total   = models.IntegerField(blank=True, null=True)
    passes_accurate= models.IntegerField(blank=True, null=True)

    # Keep the raw API list so you don’t lose anything the API adds later
    stats_json = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = ("match", "team")
        indexes = [models.Index(fields=["team"])]

    def __str__(self):
        return f"Stats {self.team.name} @ {self.match_id}"





















class Lineup(models.Model):
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="lineups")
    team = models.ForeignKey(Team, on_delete=models.PROTECT, related_name="lineups")
    formation = models.CharField(max_length=32, blank=True, null=True)
    starters_json = models.JSONField()  # store full payload
    bench_json = models.JSONField(blank=True, null=True)
    xi_strength = models.FloatField(blank=True, null=True)
    xi_changes = models.SmallIntegerField(blank=True, null=True)

    class Meta:
        unique_together = ("match", "team")



class FeaturesSnapshot(models.Model):
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="feature_snapshots")
    ts_mode = models.CharField(max_length=8, choices=TS_MODE_CHOICES, default="T24")
    features_json = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ("match", "ts_mode")




from django.db import models

class ModelVersion(models.Model):
    name = models.CharField(max_length=128, null=True, blank=True)
    kind = models.CharField(max_length=32, null=True, blank=True)  # "goals"
    league_id = models.IntegerField(null=True, blank=True)
    trained_until = models.DateTimeField(null=True, blank=True)
    metrics_json = models.TextField(null=True, blank=True)
    file_home = models.CharField(max_length=256, null=True, blank=True)
    file_away = models.CharField(max_length=256, null=True, blank=True)
    calibration_json = models.JSONField(null=True, blank=True)   # holds {"file": ".../cal
    created_at = models.DateTimeField(auto_now_add=True)











class LeagueSeasonParams(models.Model):
    league_id   = models.IntegerField(db_index=True)
    season      = models.IntegerField(db_index=True)
    intercept   = models.FloatField(default=0.0)
    hfa         = models.FloatField(default=0.0)
    avg_goals   = models.FloatField(default=0.0)
    n_matches   = models.IntegerField(default=0)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("league_id", "season")

    def __str__(self) -> str:
        return f"L{self.league_id} S{self.season} (μ0={self.intercept:.3f}, HFA={self.hfa:.3f})"




class Prediction(models.Model):
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="predictions")
    ts_mode = models.CharField(max_length=8, choices=TS_MODE_CHOICES, default="T24")
    model_ver = models.ForeignKey(ModelVersion, on_delete=models.PROTECT, related_name="predictions")
    goals_base_json = models.JSONField()     # {"lambda_home": .., "lambda_away": .., "rho": ..}
    corners_base_json = models.JSONField(blank=True, null=True)
    cards_base_json = models.JSONField(blank=True, null=True)
    markets_json = models.JSONField()        # all calibrated markets for UI/API
    explain_json = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ("match", "ts_mode", "model_ver")


class PredictedMarket(models.Model):
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name="predicted_markets")
    league = models.ForeignKey(League, on_delete=models.PROTECT, related_name="predicted_markets")
    market_code = models.CharField(max_length=64)    # e.g., "1X2.H", "BTTS.GG", "OU.GOALS.2_5.OVER"
    specifier = models.CharField(max_length=32, blank=True, null=True)  # line like "2_5"
    p_model = models.FloatField()                    # calibrated probability
    fair_odds = models.FloatField()
    book_odds = models.FloatField(blank=True, null=True)
    edge = models.FloatField(blank=True, null=True)
    kickoff_utc = models.DateTimeField()
    created_at = models.DateTimeField(default=timezone.now)
    lambda_home = models.FloatField(null=True, blank=True)
    lambda_away = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["kickoff_utc"]),
            models.Index(fields=["league", "market_code"]),
            models.Index(fields=["p_model"]),
        ]















from django.db import models

class MarketProb(models.Model):
    PERIOD_CHOICES = [
        ("FT", "Full Time"),
        ("1H", "First Half"),
        ("2H", "Second Half"),
    ]

    fixture = models.ForeignKey("matches.Match", on_delete=models.CASCADE)
    model_version = models.ForeignKey("matches.ModelVersion", on_delete=models.PROTECT)
    league_id = models.IntegerField()
    season = models.IntegerField()
    kickoff_utc = models.DateTimeField()

    # market key & outcome key keep it flexible:
    # examples:
    # market="1X2", outcome in {"1","X","2"}
    # market="DC", outcome in {"1X","12","X2"}
    # market="DNB", outcome in {"1","2"}
    # market="OU", outcome in {">","<"}, line=2.5
    # market="TT_H", outcome in {">","<"}, line=1.5
    # market="AH_H", outcome in {"+","-"}, line=-0.5  (home -0.5)
    # market="BTTS", outcome in {"YES","NO"}
    # market="ODD_EVEN", outcome in {"ODD","EVEN"}
    # market="FTTS", outcome in {"HOME","AWAY","NO_GOAL"}
    market = models.CharField(max_length=32)
    outcome = models.CharField(max_length=32)
    period = models.CharField(max_length=2, choices=PERIOD_CHOICES, default="FT")
    line = models.FloatField(null=True, blank=True)

    prob_raw = models.FloatField()
    prob_calibrated = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("fixture", "model_version", "market", "outcome", "period", "line")
















# prediction/matches/models.py
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class MatchPrediction(models.Model):
    # Identity / linkage
    match = models.OneToOneField(
        "Match", on_delete=models.CASCADE, related_name="prediction"
    )
    league_id = models.IntegerField(db_index=True)
    season = models.IntegerField(db_index=True)
    kickoff_utc = models.DateTimeField(db_index=True)

    # (Optional) tie prediction to the exact model used
    
    # (Optional) record the blend toward priors used at inference
    alpha = models.FloatField(
        default=0.70, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )

    # Goals λ
    lambda_home = models.FloatField(validators=[MinValueValidator(0.0)])
    lambda_away = models.FloatField(validators=[MinValueValidator(0.0)])

    # Corners λ
    lambda_corners_home = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.0)]
    )
    lambda_corners_away = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.0)]
    )

    # Cards λ (total or team—kept per-team as you had)
    lambda_cards_home = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.0)]
    )
    lambda_cards_away = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.0)]
    )

    # ✅ Yellow cards λ (per-team)
    lambda_yellows_home = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.0)]
    )
    lambda_yellows_away = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.0)]
    )

    # Core markets (0..1). Keep floats for speed; validated into range.
    prob_home = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    prob_draw = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    prob_away = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    prob_over25 = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    prob_btts = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )

    # Rich dump for everything else (per-threshold overs, raw vs calibrated, debug)
    markets_json = models.JSONField(default=dict, blank=True)

    # Bookkeeping
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        # OneToOne already enforces uniqueness on `match`, but we still add useful indexes
        indexes = [
            models.Index(fields=["league_id", "kickoff_utc"]),
            models.Index(fields=["season", "league_id"]),
        ]
        constraints = [
            # Basic sanity for λ and probabilities
            models.CheckConstraint(
                check=models.Q(lambda_home__gte=0.0) & models.Q(lambda_away__gte=0.0),
                name="mp_lambda_goals_nonneg",
            ),
            models.CheckConstraint(
                check=(
                    models.Q(prob_home__gte=0.0) & models.Q(prob_home__lte=1.0) &
                    models.Q(prob_draw__gte=0.0) & models.Q(prob_draw__lte=1.0) &
                    models.Q(prob_away__gte=0.0) & models.Q(prob_away__lte=1.0) &
                    models.Q(prob_over25__gte=0.0) & models.Q(prob_over25__lte=1.0) &
                    models.Q(prob_btts__gte=0.0) & models.Q(prob_btts__lte=1.0)
                ),
                name="mp_probs_in_0_1",
            ),
        ]
        ordering = ["kickoff_utc"]
        verbose_name = "Match Prediction"
        verbose_name_plural = "Match Predictions"
















from django.db import models

class MLTrainingMatch(models.Model):
    """
    One row per fixture with labels + all features (as-of kickoff).
    This is the table you export to train scikit-learn models.
    """
    fixture_id = models.BigIntegerField(primary_key=True)
    league_id = models.IntegerField()
    season = models.IntegerField()
    kickoff_utc = models.DateTimeField()
    home_team_id = models.IntegerField()
    away_team_id = models.IntegerField()
    ts_cutoff = models.DateTimeField()  # when features were computed

    data_quality_score = models.FloatField(blank=True, null=True)
    league_cluster = models.CharField(max_length=32, blank=True, null=True)
    has_xg = models.BooleanField(default=False)
    notes = models.TextField(blank=True, null=True)
    minute_labels_json = models.JSONField(default=dict, blank=True)
    y_ht_home = models.SmallIntegerField(blank=True, null=True)
    y_ht_away = models.SmallIntegerField(blank=True, null=True)

    # Labels (90')
    y_home_goals_90 = models.SmallIntegerField(blank=True, null=True)
    y_away_goals_90 = models.SmallIntegerField(blank=True, null=True)
    y_home_corners_90 = models.SmallIntegerField(blank=True, null=True)
    y_away_corners_90 = models.SmallIntegerField(blank=True, null=True)
    y_home_cards_90 = models.SmallIntegerField(blank=True, null=True)
    y_away_cards_90 = models.SmallIntegerField(blank=True, null=True)

    # --- A compact set of last-10/5 features (extend as needed) ---
    # Home overall last-10
    h_gf10 = models.FloatField(blank=True, null=True)
    h_ga10 = models.FloatField(blank=True, null=True)
    h_gd10 = models.FloatField(blank=True, null=True)
    h_sot10 = models.FloatField(blank=True, null=True)
    h_conv10 = models.FloatField(blank=True, null=True)
    h_sot_pct10 = models.FloatField(blank=True, null=True)
    h_poss10 = models.FloatField(blank=True, null=True)
    h_corners_for10 = models.FloatField(blank=True, null=True)
    h_cards_for10 = models.FloatField(blank=True, null=True)
    h_clean_sheets10 = models.FloatField(blank=True, null=True)

    # Away overall last-10
    a_gf10 = models.FloatField(blank=True, null=True)
    a_ga10 = models.FloatField(blank=True, null=True)
    a_gd10 = models.FloatField(blank=True, null=True)
    a_sot10 = models.FloatField(blank=True, null=True)
    a_conv10 = models.FloatField(blank=True, null=True)
    a_sot_pct10 = models.FloatField(blank=True, null=True)
    a_poss10 = models.FloatField(blank=True, null=True)
    a_corners_for10 = models.FloatField(blank=True, null=True)
    a_cards_for10 = models.FloatField(blank=True, null=True)
    a_clean_sheets10 = models.FloatField(blank=True, null=True)

    # Venue split
    h_home_gf10 = models.FloatField(blank=True, null=True)
    a_away_gf10 = models.FloatField(blank=True, null=True)

    # Situational
    h_rest_days = models.FloatField(blank=True, null=True)
    a_rest_days = models.FloatField(blank=True, null=True)
    h_matches_14d = models.SmallIntegerField(blank=True, null=True)
    a_matches_14d = models.SmallIntegerField(blank=True, null=True)

    # Deltas (home - away)
    d_gf10 = models.FloatField(blank=True, null=True)
    d_sot10 = models.FloatField(blank=True, null=True)
    d_rest_days = models.FloatField(blank=True, null=True)
 

    # Missingness flags
    h_stats_missing = models.BooleanField(default=False)
    a_stats_missing = models.BooleanField(default=False)

    stats10_json = models.JSONField(default=dict, blank=True)
    stats5_json  = models.JSONField(default=dict, blank=True)

    # (nice to have)
    built_at = models.DateTimeField(default=timezone.now)
    feature_version = models.CharField(max_length=16, blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["league_id", "season"]),
            models.Index(fields=["kickoff_utc"]),
            models.Index(fields=["home_team_id"]),
            models.Index(fields=["away_team_id"]),
        ]










# matches/models_ratings.py (or add to models.py)
from django.db import models

class TeamRating(models.Model):
    league_id = models.IntegerField(db_index=True)
    season = models.IntegerField(db_index=True)
    team_id = models.IntegerField(db_index=True)
    attack = models.FloatField(default=0.0)   # log-attack strength
    defense = models.FloatField(default=0.0)  # log-defense strength (negative is better)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("league_id", "season", "team_id")



















class CornerPrediction(models.Model):
    match = models.OneToOneField("Match", on_delete=models.CASCADE)
    league_id = models.IntegerField()
    season = models.IntegerField()
    kickoff_utc = models.DateTimeField()
    lambda_home = models.FloatField()
    lambda_away = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class CardPrediction(models.Model):
    match = models.OneToOneField("Match", on_delete=models.CASCADE)
    league_id = models.IntegerField()
    season = models.IntegerField()
    kickoff_utc = models.DateTimeField()
    lambda_home = models.FloatField()
    lambda_away = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)























# matches/models.py  (append if missing)
from django.db import models

class DailyTicket(models.Model):
    TICKET_STATUS = [
        ("pending", "Pending"),
        ("won", "Won"),
        ("lost", "Lost"),
        ("void", "Void"),
    ]

    # NOTE: we'll use league_id=0 to mean “global ticket (all leagues)”
    league_id = models.IntegerField(db_index=True)

    ticket_date = models.DateField(db_index=True)
    selections = models.JSONField(default=list)

    legs = models.IntegerField(default=0)
    acc_fair_odds = models.FloatField(null=True, blank=True)
    acc_bookish_odds = models.FloatField(null=True, blank=True)
    acc_probability = models.FloatField(null=True, blank=True)

    status = models.CharField(max_length=12, choices=TICKET_STATUS, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("league_id", "ticket_date")]
        ordering = ["-ticket_date", "-created_at"]

    def __str__(self):
        return f"DailyTicket L{self.league_id} {self.ticket_date} ({self.legs} legs)"
















# matches/models.py
from django.db import models

class Venue(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=128)
    city = models.CharField(max_length=128, blank=True, null=True)
    country = models.CharField(max_length=64, blank=True, null=True)
    address = models.CharField(max_length=256, blank=True, null=True)
    capacity = models.IntegerField(blank=True, null=True)
    surface = models.CharField(max_length=64, blank=True, null=True)
    image_url = models.URLField(max_length=500, blank=True, null=True)

    built = models.IntegerField(blank=True, null=True)         # you referenced this
    raw_json = models.JSONField(default=dict, blank=True, null=True)

    def __str__(self):
        return self.name or f"Venue {self.pk}"

# models.py
from django.db import models

class StandingsSnapshot(models.Model):
    league_id  = models.IntegerField(db_index=True)
    season     = models.IntegerField(db_index=True)
    # date or datetime works; DateField is usually enough
    as_of_date = models.DateField(db_index=True)
    json       = models.JSONField()  # raw provider payload

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("league_id", "season", "as_of_date")
        indexes = [
            models.Index(fields=["league_id", "season", "-as_of_date"]),
        ]

    def __str__(self):
        return f"{self.league_id} {self.season} @ {self.as_of_date}"


class StandingsRow(models.Model):
    league_id = models.IntegerField()
    season = models.IntegerField()
    group_name = models.CharField(max_length=80, blank=True, null=True)
    rank = models.IntegerField()
    team = models.ForeignKey(Team, on_delete=models.CASCADE)


    played = models.IntegerField(default=0)
    win = models.IntegerField(default=0)
    draw = models.IntegerField(default=0)
    loss = models.IntegerField(default=0)
    gf = models.IntegerField(default=0)
    ga = models.IntegerField(default=0)
    gd = models.IntegerField(default=0)
    points = models.IntegerField(default=0)
    form = models.CharField(max_length=20, blank=True, null=True)
    last5_json = models.JSONField(default=dict) # if API returns last5


    class Meta:
        unique_together = ("league_id", "season", "group_name", "team")
        indexes = [
        models.Index(fields=["league_id", "season"]),
            ]




# matches/models.py
class Player(models.Model):
    id = models.BigIntegerField(primary_key=True)  # API ids can be big
    name = models.CharField(max_length=128)
    firstname = models.CharField(max_length=64, blank=True, null=True)
    lastname = models.CharField(max_length=64, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    nationality = models.CharField(max_length=64, blank=True, null=True)
    height_cm = models.IntegerField(blank=True, null=True)
    weight_kg = models.FloatField(blank=True, null=True)
    photo_url = models.URLField(max_length=500, blank=True, null=True)
    raw_json = models.JSONField(default=dict, blank=True, null=True)

    def __str__(self):
        return self.name

class PlayerSeason(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="seasons")
    team   = models.ForeignKey('leagues.Team', on_delete=models.PROTECT)
    season = models.IntegerField()

    number = models.IntegerField(blank=True, null=True)
    position = models.CharField(max_length=32, blank=True, null=True)
    injured = models.BooleanField(default=False)

    appearances = models.IntegerField(blank=True, null=True)
    minutes     = models.IntegerField(blank=True, null=True)
    rating      = models.FloatField(blank=True, null=True)

    goals   = models.IntegerField(blank=True, null=True)
    assists = models.IntegerField(blank=True, null=True)
    yellows = models.IntegerField(blank=True, null=True)
    reds    = models.IntegerField(blank=True, null=True)

    raw_json = models.JSONField(default=dict, blank=True, null=True)
    # in matches/models.py -> PlayerSeason
    shots_total   = models.IntegerField(null=True, blank=True)
    shots_on      = models.IntegerField(null=True, blank=True)
    cards_yellow  = models.IntegerField(null=True, blank=True)
    cards_red     = models.IntegerField(null=True, blank=True)


    class Meta:
        unique_together = ("player", "team", "season")
        indexes = [models.Index(fields=["team", "season"])]








class Transfer(models.Model):
    # API-Football returns a history list per player. We'll key by (player, date, to_team)
    player = models.ForeignKey(Player, on_delete=models.CASCADE)
    date = models.DateField(null=True,blank=True)
    type = models.CharField(max_length=40, blank=True, null=True)
    from_team = models.ForeignKey(Team, on_delete=models.SET_NULL, related_name="out_transfers", null=True)
    to_team = models.ForeignKey(Team, on_delete=models.SET_NULL, related_name="in_transfers", null=True)
    fee = models.CharField(max_length=80, blank=True, null=True)
    raw_json = models.JSONField(default=dict)


    class Meta:
        unique_together = ("player", "date", "to_team")
        indexes = [models.Index(fields=["to_team", "date"])]


class Injury(models.Model):
    fixture_id = models.IntegerField()
    player = models.ForeignKey(Player, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    type = models.CharField(max_length=120, blank=True, null=True)
    reason = models.CharField(max_length=200, blank=True, null=True)
    date = models.DateField(blank=True, null=True)
    raw_json = models.JSONField(default=dict)


    class Meta:
        unique_together = ("fixture_id", "player")
        indexes = [models.Index(fields=["team"])]




class Trophy(models.Model):
    # Player/Coach trophies: here we store for players; add a Coach model similarly if needed
    player = models.ForeignKey(Player, on_delete=models.CASCADE)
    league = models.CharField(max_length=120)
    country = models.CharField(max_length=80, blank=True, null=True)
    place = models.CharField(max_length=80, blank=True, null=True) # e.g., Winner, Runner-up
    season = models.CharField(max_length=40, blank=True, null=True)
    raw_json = models.JSONField(default=dict)


    class Meta:
        indexes = [models.Index(fields=["player", "league"])]








# matches/models.py
# matches/models.py
from django.conf import settings
from django.db import models

class UserDailyTicket(models.Model):
    STATUS = [
        ("pending", "Pending"),
        ("won", "Won"),
        ("lost", "Lost"),
        ("void", "Void"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="daily_tickets")
    base_ticket = models.ForeignKey("matches.DailyTicket", null=True, blank=True,
                                    on_delete=models.SET_NULL, related_name="user_variants")

    name = models.CharField(max_length=120, blank=True, default="My ticket")
    filters = models.JSONField(default=dict, blank=True)
    ticket_date = models.DateField(db_index=True)

    selections = models.JSONField(default=list)
    legs = models.IntegerField(default=0)
    acc_probability = models.FloatField(null=True, blank=True)
    acc_fair_odds = models.FloatField(null=True, blank=True)
    acc_bookish_odds = models.FloatField(null=True, blank=True)

    status = models.CharField(max_length=12, choices=STATUS, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["user", "ticket_date"])]
        ordering = ["-ticket_date", "-created_at"]

    def __str__(self):
        return f"{self.user} — {self.ticket_date} ({self.legs} legs)"
