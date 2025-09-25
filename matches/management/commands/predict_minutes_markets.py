# prediction/matches/management/commands/predict_minutes_markets.py
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction

from matches.models import MatchPrediction, PredictedMarket

EPS = 1e-9

def _iso_apply_scalar(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve:
        return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve.get("x", []), float); y = np.array(curve.get("y", []), float)
    if x.size == 0:
        return float(np.clip(p, 0.0, 1.0))
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))

def _bp_grid_pmf(l1: float, l2: float, l12: float, G: int) -> np.ndarray:
    H = int(G) + 1; A = int(G) + 1
    P = np.zeros((H, A), float); e = math.exp(-(l1 + l2 + l12))
    from math import factorial
    for i in range(H):
        for j in range(A):
            s = 0.0
            for k in range(0, min(i, j) + 1):
                s += (l1 ** (i - k)) / factorial(i - k) * \
                     (l2 ** (j - k)) / factorial(j - k) * \
                     (l12 ** k) / factorial(k)
            P[i, j] = e * s
    s = P.sum()
    if s > 0:
        P /= s
    else:
        P[:] = 0.0; P[0,0] = 1.0
    return P

def _bp_no_goal_prob(mu_h: float, mu_a: float, c_bp: float, G: int) -> float:
    lam12 = float(c_bp) * min(mu_h, mu_a)
    l1 = max(1e-12, mu_h - lam12)
    l2 = max(1e-12, mu_a - lam12)
    return float(_bp_grid_pmf(l1, l2, lam12, G)[0, 0])

def _survival_from_list(S_list: List[float]) -> np.ndarray:
    # artifacts store survival; expect len = tmax+1, with S[0]=1.0
    S = np.array(S_list, float)
    S = np.clip(S, 1e-9, 1.0)
    return S

def _cdf_from_base_and_gamma(S_base: np.ndarray, gamma: float) -> np.ndarray:
    return 1.0 - np.power(S_base, float(gamma))

def _hazard_from_cdf(F: np.ndarray) -> np.ndarray:
    tmax = F.shape[0] - 1
    h = np.zeros(tmax + 1, float)
    for t in range(1, tmax + 1):
        num = max(float(F[t] - F[t - 1]), 0.0)
        den = max(1.0 - float(F[t - 1]), 1e-9)
        h[t] = num / den
    return h

def _last_scorer_from_h(h: np.ndarray, pH1: float, pH2: float, second_half_start: int) -> Dict[str, float]:
    T = h.shape[0] - 1
    one_minus = 1.0 - h
    R = np.ones(T + 2, float)
    for t in range(T, 0, -1):
        R[t] = R[t+1] * one_minus[t]
    p_none = float(R[1])
    pH = 0.0; pA = 0.0
    for t in range(1, T+1):
        pH_t = pH1 if t < second_half_start else pH2
        inc = float(h[t] * R[t+1])
        pH += pH_t * inc
        pA += (1.0 - pH_t) * inc
    s = pH + pA + p_none
    if s > 0:
        pH /= s; pA /= s; p_none /= s
    return {"home": pH, "away": pA, "none": p_none}

class Command(BaseCommand):
    help = "Write TFG bands + FTS/LTS markets from minutes artifact using MatchPrediction λs."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--artifacts-minutes", type=str, required=True,
                            help="Path to artifacts.minutes.json from matches.train_minutes.")
        parser.add_argument("--delete-first", action="store_true")
        parser.add_argument("--verbose", action="store_true")

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        days = int(opts["days"])
        verbose = bool(opts["verbose"])
        delete_first = bool(opts["delete_first"])

        # Load artifact
        with open(opts["artifacts_minutes"], "r") as f:
            art = json.load(f)

        tmax = int(art.get("tmax", 90))
        bands = [(int(b["lo"]), int(b["hi"])) for b in art["bands"]]
        S_base = _survival_from_list(art["baseline"]["survival_0_T"])
        second_half_start = int(art["baseline"].get("second_half_start", 46))
        theta_2h = float(art["baseline"].get("theta_2h", 0.0))  # already baked into S_base in trainer
        beta_mu = float(art["shape_mapping"]["beta_mu"])
        mu_bar = float(art["shape_mapping"]["mu_bar"])
        c_bp = float(art["bp"]["c_bp"])
        Gmax = int(art["bp"]["max_goals"])

        # share model
        sm = art["share_model"]
        coef = np.array(sm.get("coef", [0.0,0.0,0.0]), float).reshape(1, -1)
        intercept = float(sm.get("intercept", 0.0))

        def p_home_halves(log_mu_ratio: float, mu_tot: float) -> Tuple[float,float]:
            X1 = np.array([[log_mu_ratio, mu_tot, 0.0]], float)
            X2 = np.array([[log_mu_ratio, mu_tot, 1.0]], float)
            z1 = float(intercept + X1.dot(coef.T)[0,0]); p1 = 1.0 / (1.0 + math.exp(-z1))
            z2 = float(intercept + X2.dot(coef.T)[0,0]); p2 = 1.0 / (1.0 + math.exp(-z2))
            return p1, p2

        # calibrators
        cal = art.get("calibration", {}) or {}
        cal_cdf = cal.get("cdf")
        cal_fts_home = cal.get("fts_home")

        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)

        qs = (MatchPrediction.objects
              .filter(
                  league_id=league_id,
                  kickoff_utc__gte=now,
                  kickoff_utc__lte=upto,
                  match__status__in=["NS", "PST", "TBD"],
              )
              .select_related("match")
              .order_by("kickoff_utc"))

        if not qs.exists():
            self.stdout.write("No MatchPrediction rows in window.")
            return

        if delete_first:
            with transaction.atomic():
                PredictedMarket.objects.filter(
                    match__prediction__league_id=league_id,
                    kickoff_utc__gte=now,
                    kickoff_utc__lte=upto,
                    market_code__in=["TFG_BAND","FTS","LTS"],
                ).delete()

        wrote = 0
        for mp in qs:
            mu_h = float(np.clip(getattr(mp, "lambda_home", 0.0), 0.01, 8.0))
            mu_a = float(np.clip(getattr(mp, "lambda_away", 0.0), 0.01, 8.0))
            mu_tot = float(mu_h + mu_a)
            log_ratio = float(np.log(max(mu_h, 1e-9) / max(mu_a, 1e-9)))

            # match CDF
            S_target = _bp_no_goal_prob(mu_h, mu_a, c_bp, G=Gmax)
            S_T = float(S_base[-1])
            gamma0 = max(1e-6, math.log(max(S_target, 1e-12)) / math.log(max(S_T, 1e-12)))
            gamma = gamma0 * ((mu_tot / max(mu_bar, 1e-9)) ** beta_mu)
            F = 1.0 - np.power(S_base, float(gamma))
            if cal_cdf:
                for t in range(1, len(F)):
                    F[t] = _iso_apply_scalar(float(F[t]), cal_cdf)

            # share halves
            pH1, pH2 = p_home_halves(log_ratio, mu_tot)

            # FTS
            dF = (F[1:tmax+1] - F[0:tmax])
            pH_t = np.array([pH1]*(second_half_start-1) + [pH2]*(tmax-(second_half_start-1)), float)
            p_fts_home = float(max(0.0, (dF * pH_t).sum()))
            if cal_fts_home:
                p_fts_home = _iso_apply_scalar(p_fts_home, cal_fts_home)
            p_any = float(F[-1])
            p_fts_none = float(np.clip(1.0 - p_any, 0.0, 1.0))
            p_fts_away = float(np.clip(p_any - p_fts_home, 0.0, 1.0))
            s = p_fts_home + p_fts_away + p_fts_none
            if s > 0:
                p_fts_home /= s; p_fts_away /= s; p_fts_none /= s

            # LTS
            h_match = _hazard_from_cdf(F)
            lts = _last_scorer_from_h(h_match, pH1, pH2, second_half_start)

            # Bands
            tfg = {}
            for (L, R) in bands:
                key = f"{L:02d}-{R:02d}"
                tfg[key] = float(np.clip(F[R] - F[L-1], EPS, 1.0 - EPS))
            tfg["no_goal"] = float(np.clip(1.0 - F[-1], EPS, 1.0 - EPS))
            # normalize numerically
            s_tfg = sum(tfg.values())
            if s_tfg > 0:
                for k in tfg:
                    tfg[k] /= s_tfg

            # persist
            rows = []
            # FTS
            rows += [("FTS", "home", p_fts_home), ("FTS", "away", p_fts_away), ("FTS", "none", p_fts_none)]
            # LTS
            rows += [("LTS", "home", lts["home"]), ("LTS", "away", lts["away"]), ("LTS", "none", lts["none"])]
            # TFG_BAND
            for k, p in tfg.items():
                rows.append(("TFG_BAND", k, p))

            for market_code, specifier, p in rows:
                if not np.isfinite(p):
                    continue
                p = float(np.clip(p, EPS, 1 - EPS))
                PredictedMarket.objects.update_or_create(
                    match=mp.match,
                    market_code=market_code,
                    specifier=str(specifier),
                    defaults={
                        "league_id": mp.league_id,
                        "kickoff_utc": mp.kickoff_utc,
                        "p_model": p,
                        "fair_odds": float(1.0 / p),
                        "lambda_home": float(mu_h),
                        "lambda_away": float(mu_a),
                    },
                )
                wrote += 1

            if verbose:
                m = mp.match
                hn = getattr(m.home, "name", str(m.home_id))
                an = getattr(m.away, "name", str(m.away_id))
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an} | FTS(H/A/N)=({p_fts_home:.3f},{p_fts_away:.3f},{p_fts_none:.3f})"
                )

        self.stdout.write(self.style.SUCCESS(
            f"Wrote/updated {wrote} minutes-based PredictedMarket rows for league {league_id}"
        ))
