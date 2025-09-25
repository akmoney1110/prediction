# prediction/matches/management/commands/compare_lambdas.py

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.db.models import Q

from matches.models import (
    Match,
    MatchStats,
    ModelVersion,
    TeamRating,
)

# =========================
# Feature lists (numeric)
# =========================
# Base numeric features you trained with (expand if you added more).
# NOTE: Even if you forget something here, we align by name against the model's
#       feature_names_in_ and auto-fill missing columns with 0.0 at inference.
BASE_FEATS = [
    "h_gf10","a_gf10","d_gf10",
    "h_ga10","a_ga10",
    "h_gd10","a_gd10",
    "h_sot10","a_sot10","d_sot10",
    "h_sot_pct10","a_sot_pct10",
    "h_conv10","a_conv10",
    "h_poss10","a_poss10",
    "h_clean_sheets10","a_clean_sheets10",
    "h_corners_for10","a_corners_for10",
    "h_cards_for10","a_cards_for10",
    "h_rest_days","a_rest_days","d_rest_days",
    "h_matches_14d","a_matches_14d",
    "h_stats_missing","a_stats_missing",
    # If your training included venue splits, these should be present too:
    "h_home_gf10","a_away_gf10",
]

# Priors appended during training
PRIOR_FEATS = [
    "attack_home", "defense_home",
    "attack_away", "defense_away",
    "rating_delta_attack", "rating_delta_defense",
    "league_home_adv",
]

# Many pipelines also left raw IDs & simple calendricals in:
AUX_FEATS = [
    "home_team_id", "away_team_id",
    "league_id", "season",
    "kickoff_month", "kickoff_weekday",
]

# =========================
# Helpers
# =========================

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _avg(a): return float(np.mean(a)) if a else 0.0

def _matches_in_14d(team_id, upto_dt):
    since = upto_dt - timedelta(days=14)
    return (Match.objects
            .filter(Q(home_id=team_id) | Q(away_id=team_id),
                    kickoff_utc__lt=upto_dt,
                    kickoff_utc__gte=since,
                    status__in=["FT","AET","PEN"])
            .count())

def _venue_form_gf(team_id: int, upto_dt, lookback=10, require_home=False, require_away=False) -> float:
    qs = (Match.objects
          .filter(Q(home_id=team_id) | Q(away_id=team_id),
                  kickoff_utc__lt=upto_dt,
                  status__in=["FT","AET","PEN"])
          .order_by("-kickoff_utc"))
    if require_home:
        qs = qs.filter(home_id=team_id)
    if require_away:
        qs = qs.filter(away_id=team_id)
    qs = qs[:lookback]
    gf = []
    for m in qs:
        if m.home_id == team_id:
            gf.append(m.goals_home or 0)
        else:
            gf.append(m.goals_away or 0)
    return _avg(gf)

def _form_for_team(team_id, upto_dt, lookback=10):
    """
    Compute rolling form features from past finished matches, per team.
    Uses Match + MatchStats (per-team row).
    """
    qs = (Match.objects
          .filter(Q(home_id=team_id) | Q(away_id=team_id),
                  kickoff_utc__lt=upto_dt,
                  status__in=["FT","AET","PEN"])
          .order_by("-kickoff_utc")[:lookback])

    gf, ga, sot, shots, poss, corners, cards, cs = [], [], [], [], [], [], [], []
    for m in qs:
        is_home = (m.home_id == team_id)
        if is_home:
            gf.append(m.goals_home or 0)
            ga.append(m.goals_away or 0)
            cs.append(1.0 if (m.goals_away or 0) == 0 else 0.0)
        else:
            gf.append(m.goals_away or 0)
            ga.append(m.goals_home or 0)
            cs.append(1.0 if (m.goals_home or 0) == 0 else 0.0)

        st = MatchStats.objects.filter(match=m, team_id=team_id).first()
        if st:
            sot.append(st.sot or 0)
            shots.append(st.shots or 0)
            poss.append(st.possession_pct or 0)
            corners.append(st.corners or 0)
            cards.append((st.yellows or 0) + (st.reds or 0))

    shots_avg = _avg(shots)
    return {
        "gf": _avg(gf),
        "ga": _avg(ga),
        "sot": _avg(sot),
        "shots": shots_avg,
        "sot_pct": (_avg(sot) / shots_avg) if shots_avg > 0 else 0.0,
        "conv": (_avg(gf) / shots_avg) if shots_avg > 0 else 0.0,
        "poss": _avg(poss),
        "corners": _avg(corners),
        "cards": _avg(cards),
        "cs_rate": _avg(cs),
        "count": len(qs),
    }

def _rest_days(team_id, upto_dt):
    last = (Match.objects
            .filter(Q(home_id=team_id) | Q(away_id=team_id),
                    kickoff_utc__lt=upto_dt,
                    status__in=["FT","AET","PEN"])
            .order_by("-kickoff_utc").first())
    if not last:
        return 7.0
    d = (upto_dt - last.kickoff_utc)
    return max(0.0, d.days + d.seconds/86400.0)

def _estimate_league_hfa(league_id: int, seasons: Optional[List[int]]):
    """
    Estimate league home-field advantage using average (home_goals - away_goals)
    over finished matches.
    """
    qs = (Match.objects
          .filter(league_id=league_id, status__in=["FT","AET","PEN"])
          .exclude(Q(goals_home__isnull=True) | Q(goals_away__isnull=True)))
    if seasons:
        qs = qs.filter(season__in=seasons)

    diffs = []
    for m in qs.only("goals_home", "goals_away"):
        diffs.append(_safe_float(m.goals_home, 0.0) - _safe_float(m.goals_away, 0.0))
    if not diffs:
        return 0.15
    return float(np.mean(diffs))

def _team_prior(league_id: int, season: int, team_id: int):
    """
    TeamRating -> (attack, defense). Default (0,0) if not found.
    """
    r = TeamRating.objects.filter(league_id=league_id, season=season, team_id=team_id).first()
    if not r:
        return 0.0, 0.0
    return _safe_float(r.attack, 0.0), _safe_float(r.defense, 0.0)

def _build_numeric_blocks(m: Match, hfa_val: float):
    """
    Build base numeric and prior dicts from the DB for a single match.
    """
    dt = m.kickoff_utc
    H = _form_for_team(m.home_id, dt)
    A = _form_for_team(m.away_id, dt)

    h_rest = _rest_days(m.home_id, dt)
    a_rest = _rest_days(m.away_id, dt)

    base = {
        "h_gf10": H["gf"], "a_gf10": A["gf"], "d_gf10": H["gf"] - A["gf"],
        "h_ga10": H["ga"], "a_ga10": A["ga"],
        "h_gd10": H["gf"] - H["ga"], "a_gd10": A["gf"] - A["ga"],
        "h_sot10": H["sot"], "a_sot10": A["sot"], "d_sot10": H["sot"] - A["sot"],
        "h_sot_pct10": H["sot_pct"], "a_sot_pct10": A["sot_pct"],
        "h_conv10": H["conv"], "a_conv10": A["conv"],
        "h_poss10": H["poss"], "a_poss10": A["poss"],
        "h_clean_sheets10": H["cs_rate"], "a_clean_sheets10": A["cs_rate"],
        "h_corners_for10": H["corners"], "a_corners_for10": A["corners"],
        "h_cards_for10": H["cards"], "a_cards_for10": A["cards"],
        "h_rest_days": h_rest,
        "a_rest_days": a_rest,
        "d_rest_days": h_rest - a_rest,
        "h_matches_14d": float(_matches_in_14d(m.home_id, dt)),
        "a_matches_14d": float(_matches_in_14d(m.away_id, dt)),
        "h_stats_missing": 1.0 if H.get("count", 0) < 3 else 0.0,
        "a_stats_missing": 1.0 if A.get("count", 0) < 3 else 0.0,
        # venue splits (if your model expects them)
        "h_home_gf10": _venue_form_gf(m.home_id, dt, require_home=True),
        "a_away_gf10": _venue_form_gf(m.away_id, dt, require_away=True),
    }

    # priors
    atk_h, dfn_h = _team_prior(m.league_id, m.season, m.home_id)
    atk_a, dfn_a = _team_prior(m.league_id, m.season, m.away_id)
    prior = {
        "attack_home": atk_h, "defense_home": dfn_h,
        "attack_away": atk_a, "defense_away": dfn_a,
        "rating_delta_attack": atk_h - atk_a,
        "rating_delta_defense": dfn_h - dfn_a,
        "league_home_adv": _safe_float(hfa_val, 0.0),
    }

    # aux/raw cols that sometimes are in the training set
    aux = {
        "home_team_id": int(m.home_id),
        "away_team_id": int(m.away_id),
        "league_id": int(m.league_id),
        "season": int(m.season),
        "kickoff_month": int(m.kickoff_utc.month),
        "kickoff_weekday": int(m.kickoff_utc.weekday()),  # Mon=0..Sun=6
    }

    return base, prior, aux

def _build_X_for_model(m: Match, hfa_val: float, model):
    """
    Return an input aligned to the model's expected columns.
    If the model has feature_names_in_, we create a single-row DataFrame with
    those exact names, filling missing with 0.0.
    Otherwise, we fall back to ndarray (order-sensitive).
    """
    base, prior, aux = _build_numeric_blocks(m, hfa_val)

    # the universe of features we know how to compute
    feat: Dict[str, float] = {}
    for k in BASE_FEATS:   feat[k] = base.get(k, 0.0)
    for k in PRIOR_FEATS:  feat[k] = prior.get(k, 0.0)
    for k in AUX_FEATS:    feat[k] = _safe_float(aux.get(k, 0.0))

    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        row = {col: _safe_float(feat.get(col, 0.0)) for col in expected_cols}
        X = pd.DataFrame([row], columns=expected_cols)
        return X, base, prior

    # Fallback: no column names available; build in our canonical order
    arr = np.array(
        [feat.get(k, 0.0) for k in BASE_FEATS] +
        [feat.get(k, 0.0) for k in PRIOR_FEATS] +
        [feat.get(k, 0.0) for k in AUX_FEATS],
        dtype=np.float64
    ).reshape(1, -1)

    exp = getattr(model, "n_features_in_", None)
    if exp is not None and arr.shape[1] != exp:
        raise ValueError(
            f"Inference feature count {arr.shape[1]} != model expects {exp}. "
            "Model lacks feature_names_in_, so we can't align by name. "
            "Retrain with a pandas DataFrame so feature_names_in_ is stored, "
            "or modify BASE_FEATS/PRIOR_FEATS/AUX_FEATS to match training exactly."
        )
    return arr, base, prior

def _blend(lam_prior, lam_ml, alpha):
    return float(alpha * lam_prior + (1.0 - alpha) * lam_ml)

# ----- Poisson helpers -----
def _log_poiss_pmf(k, lam):
    if lam <= 0:
        return -np.inf if k > 0 else 0.0
    return k * math.log(lam) - lam - math.lgamma(k + 1)

def _joint_grid_logspace(lh, la, K=10):
    logPH = np.array([_log_poiss_pmf(h, lh) for h in range(K+1)])
    logPA = np.array([_log_poiss_pmf(a, la) for a in range(K+1)])
    logP = logPH[:, None] + logPA[None, :]
    m = np.max(logP); P = np.exp(logP - m); s = P.sum()
    if s <= 0 or not np.isfinite(s): return None
    return P / s

def _oneXtwo(lh, la):
    P = _joint_grid_logspace(lh, la, K=10)
    if P is None: return (1/3, 1/3, 1/3)
    idx = np.arange(P.shape[0])
    pH = float(P[idx[:, None] > idx[None, :]].sum())
    pD = float(np.trace(P))
    pA = float(P[idx[:, None] < idx[None, :]].sum())
    s = pH + pD + pA
    return (pH/s, pD/s, pA/s)

def _outcome_triple(yh: int, ya: int) -> Tuple[int, int, int]:
    if yh > ya:   return (1, 0, 0)
    if yh == ya:  return (0, 1, 0)
    return (0, 0, 1)

# =========================
# Evaluation
# =========================

@dataclass
class EvalItem:
    match_id: int
    yh: int
    ya: int
    lam_h: float
    lam_a: float
    pH: float
    pD: float
    pA: float
    nll: float
    mae_h: float
    mae_a: float
    brier: float
    logloss: float

def _score_match(yh: int, ya: int, lam_h: float, lam_a: float) -> EvalItem:
    yh = int(yh); ya = int(ya)
    # Poisson NLL on goals
    nll = -(_log_poiss_pmf(yh, lam_h) + _log_poiss_pmf(ya, lam_a))
    # MAE on goals
    mae_h = abs(yh - lam_h)
    mae_a = abs(ya - lam_a)
    # 1X2 from grid
    pH, pD, pA = _oneXtwo(lam_h, lam_a)
    oh, od, oa = _outcome_triple(yh, ya)
    # Brier & Log loss for 1X2
    brier = (pH - oh) ** 2 + (pD - od) ** 2 + (pA - oa) ** 2
    p_true = pH if oh else (pD if od else pA)
    logloss = -math.log(max(p_true, 1e-12))
    return EvalItem(-1, yh, ya, lam_h, lam_a, pH, pD, pA, nll, mae_h, mae_a, brier, logloss)

def _parse_alphas(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

# =========================
# Command
# =========================

class Command(BaseCommand):
    help = "Compare ML vs prior lambdas and blended variants, optionally evaluate over a window."

    def add_arguments(self, parser):
        parser.add_argument("--match-id", type=int, help="Specific match id")
        parser.add_argument("--league-id", type=int, help="League to scan if match-id not given")
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--alpha", type=float, default=0.70, help="Blend weight toward priors [0..1]")
        # evaluation options
        parser.add_argument("--evaluate", action="store_true", help="Run evaluation on finished matches")
        parser.add_argument("--from-date", type=str, help="YYYY-MM-DD (inclusive)")
        parser.add_argument("--to-date", type=str, help="YYYY-MM-DD (inclusive)")
        parser.add_argument("--alphas", type=str, help='Comma list like "0,0.3,0.6,0.9"')
        parser.add_argument("--topn", type=int, default=10, help="Print first N examples")

    def handle(self, *args, **opts):
        match_id = opts.get("match_id")
        league_id = opts.get("league_id")
        days = int(opts.get("days", 7))
        alpha = float(opts.get("alpha", 0.70))
        do_eval = bool(opts.get("evaluate", False))
        topn = int(opts.get("topn", 10))

        # Load latest trained models (with ColumnTransformer)
        # If league_id not given for a single match, we’ll pull from that match
        model_league_id = league_id if league_id else 39
        mv = (ModelVersion.objects
              .filter(kind="goals", league_id=model_league_id)
              .order_by("-trained_until", "-id").first())
        if not mv:
            self.stderr.write("No ModelVersion found. Train first.")
            return
        home_model = joblib.load(mv.file_home)
        away_model = joblib.load(mv.file_away)

        if do_eval:
            if not league_id:
                self.stderr.write("Provide --league-id for evaluation.")
                return
            if not opts.get("from_date") or not opts.get("to_date"):
                self.stderr.write("Provide --from-date and --to-date (YYYY-MM-DD).")
                return
            d0 = datetime.strptime(opts["from_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            d1 = datetime.strptime(opts["to_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

            matches = (Match.objects
                       .filter(league_id=league_id,
                               kickoff_utc__gte=d0,
                               kickoff_utc__lt=d1,
                               status__in=["FT","AET","PEN"])
                       .exclude(Q(goals_home__isnull=True) | Q(goals_away__isnull=True))
                       .order_by("kickoff_utc"))
            if not matches.exists():
                self.stdout.write("No finished matches in that window.")
                return

            # HFA from finished matches (all seasons OK)
            hfa_val = _estimate_league_hfa(league_id, seasons=None)
            alphas = _parse_alphas(opts.get("alphas", str(alpha))) if opts.get("alphas") else [alpha]

            out = {
                "mode": "evaluation",
                "league_id": league_id,
                "from_date": opts["from_date"],
                "to_date": opts["to_date"],
                "count": matches.count(),
                "alphas": alphas,
                "metrics": {},
                "sample": [],
            }

            # precompute ML lambdas to avoid double transforms
            pre_ml: Dict[int, Tuple[float, float]] = {}
            for m in matches:
                # if models trained for another league_id, reload by this league
                if mv.league_id != m.league_id:
                    mv2 = (ModelVersion.objects
                           .filter(kind="goals", league_id=m.league_id)
                           .order_by("-trained_until", "-id").first())
                    if not mv2:
                        continue
                    home_model = joblib.load(mv2.file_home)
                    away_model = joblib.load(mv2.file_away)

                X_h, base, prior = _build_X_for_model(m, hfa_val, home_model)
                X_a, _, _ = X_h, base, prior  # same X; identical features used by both models
                lam_ml_h = float(np.clip(home_model.predict(X_h)[0], 0.05, 10.0))
                lam_ml_a = float(np.clip(away_model.predict(X_a)[0], 0.05, 10.0))
                pre_ml[m.id] = (lam_ml_h, lam_ml_a)

            # now evaluate across alphas
            for a in alphas:
                items: List[EvalItem] = []
                for m in matches:
                    if m.id not in pre_ml:
                        continue
                    lam_ml_h, lam_ml_a = pre_ml[m.id]

                    # priors from team ratings
                    base, prior, _ = _build_numeric_blocks(m, hfa_val)
                    atk_h = prior["attack_home"]; dfn_a = prior["defense_away"]
                    atk_a = prior["attack_away"]; dfn_h = prior["defense_home"]
                    hfa   = prior["league_home_adv"]

                    lam_pr_h = float(np.exp(atk_h - dfn_a + hfa))
                    lam_pr_a = float(np.exp(atk_a - dfn_h))

                    lam_h = _blend(lam_pr_h, lam_ml_h, a)
                    lam_a = _blend(lam_pr_a, lam_ml_a, a)

                    it = _score_match(int(m.goals_home), int(m.goals_away), lam_h, lam_a)
                    it.match_id = m.id
                    items.append(it)

                if not items:
                    out["metrics"][str(a)] = {"count": 0}
                    continue

                n = len(items)
                out["metrics"][str(a)] = {
                    "count": n,
                    "nll_per_match": float(np.mean([t.nll for t in items])),
                    "mae_home": float(np.mean([t.mae_h for t in items])),
                    "mae_away": float(np.mean([t.mae_a for t in items])),
                    "brier_1x2": float(np.mean([t.brier for t in items])),
                    "logloss_1x2": float(np.mean([t.logloss for t in items])),
                    "avg_lambda_home": float(np.mean([t.lam_h for t in items])),
                    "avg_lambda_away": float(np.mean([t.lam_a for t in items])),
                }

            # put a small human sample (first N)
            for m in matches[:topn]:
                lam_ml_h, lam_ml_a = pre_ml.get(m.id, (None, None))
                if lam_ml_h is None:
                    continue
                base, prior, _ = _build_numeric_blocks(m, hfa_val)
                atk_h = prior["attack_home"]; dfn_a = prior["defense_away"]
                atk_a = prior["attack_away"]; dfn_h = prior["defense_home"]
                hfa   = prior["league_home_adv"]
                lam_pr_h = float(np.exp(atk_h - dfn_a + hfa))
                lam_pr_a = float(np.exp(atk_a - dfn_h))

                row = {
                    "match_id": m.id,
                    "kickoff_utc": m.kickoff_utc.isoformat(),
                    "home": str(m.home),
                    "away": str(m.away),
                    "score": [int(m.goals_home), int(m.goals_away)],
                    "lambda": {
                        "ml": {"home": lam_ml_h, "away": lam_ml_a},
                        "prior": {"home": lam_pr_h, "away": lam_pr_a},
                    },
                    "alphas_preview": {},
                }
                for a in alphas:
                    lam_h = _blend(lam_pr_h, lam_ml_h, a)
                    lam_a = _blend(lam_pr_a, lam_ml_a, a)
                    pH, pD, pA = _oneXtwo(lam_h, lam_a)
                    row["alphas_preview"][str(a)] = {
                        "final": {"home": lam_h, "away": lam_a},
                        "oneXtwo": {"H": pH, "D": pD, "A": pA},
                    }
                out["sample"].append(row)

            self.stdout.write(json.dumps(out, indent=2))
            return

        # -------- non-evaluate: preview upcoming or a specific match --------
        # matches
        if match_id:
            matches = Match.objects.filter(pk=match_id)
            if not matches.exists():
                self.stdout.write("No match found.")
                return
            league_id = matches.first().league_id  # for HFA below
        else:
            if not league_id:
                self.stderr.write("Provide --match-id OR --league-id")
                return
            now = datetime.now(timezone.utc)
            upto = now + timedelta(days=days)
            matches = (Match.objects
                       .filter(league_id=league_id,
                               kickoff_utc__gte=now,
                               kickoff_utc__lte=upto)
                       .exclude(status__in=["FT","AET","PEN"])
                       .order_by("kickoff_utc"))

        if not matches.exists():
            self.stdout.write("No matches found.")
            return

        # estimate HFA from finished matches in this league
        hfa_val = _estimate_league_hfa(matches.first().league_id, seasons=None)
        out = {"league_id": matches.first().league_id, "alpha": alpha, "count": matches.count(), "items": []}

        for m in matches:
            # If model was trained per-league, consider reloading per match (optional).
            if mv.league_id != m.league_id:
                mv2 = (ModelVersion.objects
                       .filter(kind="goals", league_id=m.league_id)
                       .order_by("-trained_until", "-id").first())
                if not mv2:
                    continue
                home_model = joblib.load(mv2.file_home)
                away_model = joblib.load(mv2.file_away)

            # Build aligned input
            X, base_feats, prior_feats = _build_X_for_model(m, hfa_val, home_model)

            # ML lambdas via pipeline
            lam_ml_h = float(np.clip(home_model.predict(X)[0], 0.05, 10.0))
            lam_ml_a = float(np.clip(away_model.predict(X)[0], 0.05, 10.0))

            # priors -> Poisson means
            atk_h = prior_feats["attack_home"]; dfn_a = prior_feats["defense_away"]
            atk_a = prior_feats["attack_away"]; dfn_h = prior_feats["defense_home"]
            hfa   = prior_feats["league_home_adv"]

            lam_pr_h = float(np.exp(atk_h - dfn_a + hfa))
            lam_pr_a = float(np.exp(atk_a - dfn_h))

            # blend
            lam_h = _blend(lam_pr_h, lam_ml_h, alpha)
            lam_a = _blend(lam_pr_a, lam_ml_a, alpha)

            # quick 1X2 for sanity
            pH, pD, pA = _oneXtwo(lam_h, lam_a)

            out["items"].append({
                "match_id": m.id,
                "kickoff_utc": m.kickoff_utc.isoformat(),
                "home": str(m.home),
                "away": str(m.away),
                "lambdas": {
                    "ml": {"home": lam_ml_h, "away": lam_ml_a},
                    "prior": {"home": lam_pr_h, "away": lam_pr_a},
                    "final": {"home": lam_h, "away": lam_a},
                },
                "blend_alpha": alpha,
                "hfa_used": hfa_val,
                "priors_used": prior_feats,
                "base_feats_sample": {k: base_feats[k] for k in [
                    "h_gf10","a_gf10","h_ga10","a_ga10","h_sot10","a_sot10",
                    "h_sot_pct10","a_sot_pct10","h_conv10","a_conv10",
                    "h_poss10","a_poss10","h_clean_sheets10","a_clean_sheets10",
                    "h_rest_days","a_rest_days","h_home_gf10","a_away_gf10"
                ] if k in base_feats},
                "oneXtwo": {"H": pH, "D": pD, "A": pA},
            })

        self.stdout.write(json.dumps(out, indent=2))
