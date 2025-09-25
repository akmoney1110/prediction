from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List

import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Avg, Count, Q
from django.utils import timezone as djtz

from matches.models import Match, MatchPrediction
# Import TeamRating if available (you pasted it inside matches.models)
try:
    from matches.models import TeamRating  # ✅ your model is in matches.models
except Exception:
    TeamRating = None  # graceful fallback


EPS = 1e-9

# ---------- small helpers ----------

import math
from datetime import timedelta
import numpy as np
from django.db.models import Avg, Count

HALF_LIFE_DAYS = 180.0          # recency weighting: ~6 months half-life
SHRINK_K       = 4.0            # lighter shrinkage than before
EPS = 1e-9

def _decay_weight(dt, cutoff):
    # exponential decay by age in days (half-life = 180d by default)
    age_days = max(0.0, (cutoff - dt).total_seconds() / 86400.0)
    lam = math.log(2.0) / HALF_LIFE_DAYS
    return math.exp(-lam * age_days)

@dataclass
class LeaguePriors:
    base_home: float
    base_away: float
    home_for: float
    home_against: float
    away_for: float
    away_against: float

def _league_priors(league_id: int, cutoff: datetime, lookback_days: int = 720) -> LeaguePriors:
    since = cutoff - timedelta(days=lookback_days)
    done = (Match.objects
            .filter(league_id=league_id,
                    kickoff_utc__lt=cutoff,
                    kickoff_utc__gte=since,
                    goals_home__isnull=False,
                    goals_away__isnull=False))

    agg = done.aggregate(h_for=Avg("goals_home"), a_for=Avg("goals_away"))
    # fallback if sparse
    h_for = float(agg["h_for"] or 1.45)
    a_for = float(agg["a_for"] or 1.15)

    return LeaguePriors(
        base_home=h_for, base_away=a_for,
        home_for=h_for, home_against=a_for,
        away_for=a_for, away_against=h_for,
    )
def _clip(x: float, lo: float, hi: float) -> float:
    try:
        return float(np.clip(float(x), lo, hi))
    except Exception:
        return float(lo)



def _shrunk(mean: float, wsum: float, prior: float, k: float = SHRINK_K) -> float:
    # shrink a weighted mean toward a prior with pseudo-weight k
    if not np.isfinite(mean): mean = prior
    wsum = float(max(wsum, 0.0))
    k = float(max(k, 0.0))
    denom = wsum + k
    return float((mean * wsum + prior * k) / (denom if denom > 0 else 1.0))

def _team_splits_weighted(team_id: int, league_id: int, cutoff: datetime, lookback_days: int = 720):
    since = cutoff - timedelta(days=lookback_days)
    # fetch home & away games with results
    qs_home = (Match.objects
               .filter(league_id=league_id,
                       kickoff_utc__lt=cutoff, kickoff_utc__gte=since,
                       home_id=team_id,
                       goals_home__isnull=False, goals_away__isnull=False)
               .only("kickoff_utc", "goals_home", "goals_away"))
    qs_away = (Match.objects
               .filter(league_id=league_id,
                       kickoff_utc__lt=cutoff, kickoff_utc__gte=since,
                       away_id=team_id,
                       goals_home__isnull=False, goals_away__isnull=False)
               .only("kickoff_utc", "goals_home", "goals_away"))

    # weighted means with exponential decay
    def wmean_for(qs, getter):
        num = 0.0; den = 0.0
        for m in qs:
            w = _decay_weight(m.kickoff_utc, cutoff)
            num += w * getter(m)
            den += w
        return (num / den if den > 0 else float("nan")), den

    # home context
    h_for, w_hf = wmean_for(qs_home, lambda m: float(m.goals_home))
    h_again, w_ha = wmean_for(qs_home, lambda m: float(m.goals_away))
    # away context
    a_for, w_af = wmean_for(qs_away, lambda m: float(m.goals_away))
    a_again, w_aa = wmean_for(qs_away, lambda m: float(m.goals_home))

    return {
        "home_for":     (h_for, w_hf),
        "home_against": (h_again, w_ha),
        "away_for":     (a_for, w_af),
        "away_against": (a_again, w_aa),
    }

def _rating_mu_using_teamrating(m: Match, pri: LeaguePriors):
    # unchanged from your last version (log link via TeamRating if present)
    try:
        from matches.models import TeamRating
    except Exception:
        TeamRating = None
    if TeamRating is None:
        return None
    th = TeamRating.objects.filter(league_id=m.league_id, season=m.season, team_id=m.home_id).first()
    ta = TeamRating.objects.filter(league_id=m.league_id, season=m.season, team_id=m.away_id).first()
    if not th or not ta:
        return None
    mu_h = pri.base_home * math.exp(float(th.attack) - float(ta.defense))
    mu_a = pri.base_away * math.exp(float(ta.attack) - float(th.defense))
    return float(np.clip(mu_h, 0.2, 6.0)), float(np.clip(mu_a, 0.2, 6.0))

def _rating_mu_from_recent_results(m: Match, pri: LeaguePriors) -> Tuple[float, float]:
    T_h = _team_splits_weighted(m.home_id, m.league_id, m.kickoff_utc)
    T_a = _team_splits_weighted(m.away_id, m.league_id, m.kickoff_utc)

    # shrink toward league priors with lighter K
    h_for   = _shrunk(T_h["home_for"][0],     T_h["home_for"][1],     pri.home_for)
    h_again = _shrunk(T_h["home_against"][0], T_h["home_against"][1], pri.away_for)
    a_for   = _shrunk(T_a["away_for"][0],     T_a["away_for"][1],     pri.away_for)
    a_again = _shrunk(T_a["away_against"][0], T_a["away_against"][1], pri.home_for)

    # multiplicative structure in ratio space (home/away specific priors)
    mu_h = pri.base_home * (h_for / (pri.home_for + EPS)) * (a_again / (pri.away_against + EPS))
    mu_a = pri.base_away * (a_for / (pri.away_for + EPS)) * (h_again / (pri.home_against + EPS))
    return float(np.clip(mu_h, 0.2, 6.0)), float(np.clip(mu_a, 0.2, 6.0))


def _team_splits(team_id: int, league_id: int, cutoff: datetime, lookback_days: int = 720) -> Dict[str, Tuple[float, int]]:
    """
    Returns per-team home/away scoring & conceding means with sample sizes:
      {
        "home_for":   (mean, n),
        "home_against": (mean, n),
        "away_for":   (mean, n),
        "away_against": (mean, n),
      }
    """
    since = cutoff - timedelta(days=lookback_days)

    qs_home = (Match.objects
               .filter(league_id=league_id,
                       kickoff_utc__lt=cutoff, kickoff_utc__gte=since,
                       home_id=team_id,
                       goals_home__isnull=False, goals_away__isnull=False))
    qs_away = (Match.objects
               .filter(league_id=league_id,
                       kickoff_utc__lt=cutoff, kickoff_utc__gte=since,
                       away_id=team_id,
                       goals_home__isnull=False, goals_away__isnull=False))

    agg_home = qs_home.aggregate(for_=Avg("goals_home"), ag=Avg("goals_away"), n=Count("id"))
    agg_away = qs_away.aggregate(for_=Avg("goals_away"), ag=Avg("goals_home"), n=Count("id"))

    return {
        "home_for":     (float(agg_home["for_"] or 0.0), int(agg_home["n"] or 0)),
        "home_against": (float(agg_home["ag"]  or 0.0), int(agg_home["n"] or 0)),
        "away_for":     (float(agg_away["for_"] or 0.0), int(agg_away["n"] or 0)),
        "away_against": (float(agg_away["ag"]  or 0.0), int(agg_away["n"] or 0)),
    }


def _rating_mu_using_teamrating(m: Match, pri: LeaguePriors) -> Optional[Tuple[float, float]]:
    """
    If TeamRating exists for both teams (same season), use a log-link mapping:
      μ_home = base_home * exp(att_home - def_away)
      μ_away = base_away * exp(att_away - def_home)
    """
    if TeamRating is None:  # model not loaded
        return None
    try:
        tr_h = TeamRating.objects.filter(league_id=m.league_id, season=m.season, team_id=m.home_id).first()
        tr_a = TeamRating.objects.filter(league_id=m.league_id, season=m.season, team_id=m.away_id).first()
    except Exception:
        return None
    if not tr_h or not tr_a:
        return None

    mu_h = pri.base_home * math.exp(float(tr_h.attack) - float(tr_a.defense))
    mu_a = pri.base_away * math.exp(float(tr_a.attack) - float(tr_h.defense))
    return _clip(mu_h, 0.2, 6.0), _clip(mu_a, 0.2, 6.0)


def _rating_mu_from_recent_results(m: Match, pri: LeaguePriors) -> Tuple[float, float]:
    """
    Data-driven fallback with shrinkage.
    """
    H = _team_splits(m.home_id, m.league_id, m.kickoff_utc)
    A = _team_splits(m.away_id, m.league_id, m.kickoff_utc)

    # shrink each team stat toward league priors
    h_for   = _shrunk(H["home_for"][0],     H["home_for"][1],     pri.home_for,  k=10)
    h_again = _shrunk(H["home_against"][0], H["home_against"][1], pri.away_for,  k=10)
    a_for   = _shrunk(A["away_for"][0],     A["away_for"][1],     pri.away_for,  k=10)
    a_again = _shrunk(A["away_against"][0], A["away_against"][1], pri.home_for,  k=10)

    # multiplicative structure: base * (attack strength) * (opponent defensive weakness)
    mu_h = pri.base_home * (h_for / (pri.home_for + EPS)) * (a_again / (pri.home_against + EPS))
    mu_a = pri.base_away * (a_for / (pri.away_for + EPS)) * (h_again / (pri.away_against + EPS))

    # safety clamps
    return _clip(mu_h, 0.2, 6.0), _clip(mu_a, 0.2, 6.0)


# ---------- command ----------

class Command(BaseCommand):
    help = "Write/refresh MatchPrediction λ_home/λ_away using ratings or recent results (no feature JSON needed)."

    def add_arguments(self, parser):
        parser.add_argument("--leagues", type=str, required=True, help="e.g. '39' or '39,61'")
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--delete-first", action="store_true")

    def handle(self, *args, **opts):
        leagues = [int(x) for x in str(opts["leagues"]).split(",") if x.strip()]
        days = int(opts["days"])
        delete_first = bool(opts["delete_first"])

        now = djtz.now()
        upto = now + timedelta(days=days)

        total_wrote = 0
        used_tr = 0
        used_recent = 0

        for L in leagues:
            pri = _league_priors(L, upto)

            qs = (Match.objects
                  .filter(league_id=L,
                          kickoff_utc__gte=now,
                          kickoff_utc__lte=upto,
                          status__in=["NS", "PST", "TBD"])
                  .select_related("league", "home", "away")
                  .order_by("kickoff_utc"))

            if delete_first:
                MatchPrediction.objects.filter(
                    league_id=L, kickoff_utc__gte=now, kickoff_utc__lte=upto
                ).delete()

            wrote = 0
            for m in qs:
                # Try TeamRating first
                mu = _rating_mu_using_teamrating(m, pri)
                if mu is not None:
                    mh, ma = mu
                    used_tr += 1
                else:
                    mh, ma = _rating_mu_from_recent_results(m, pri)
                    used_recent += 1

                MatchPrediction.objects.update_or_create(
                    match=m,
                    defaults={
                        "league_id": m.league_id,
                        "season": m.season,
                        "kickoff_utc": m.kickoff_utc,
                        "lambda_home": float(mh),
                        "lambda_away": float(ma),
                        # leave calibration fields for predict_full_markets to fill
                    },
                )
                wrote += 1

            total_wrote += wrote
            self.stdout.write(self.style.SUCCESS(
                f"League {L}: wrote {wrote} MatchPrediction rows "
                f"(TeamRating={used_tr}, recent-results={used_recent})"
            ))

        self.stdout.write(self.style.SUCCESS(f"TOTAL wrote: {total_wrote}"))
