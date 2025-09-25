# prediction/matches/management/commands/predict_corner_market.py
from __future__ import annotations

import json, math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction

from matches.models import MatchPrediction, PredictedMarket

DEFAULT_EPS = 1e-9

# -------- NB helpers --------
def nb_pmf_vec(mu: float, k: float, nmax: int) -> np.ndarray:
    mu = max(1e-9, float(mu)); k = max(1e-9, float(k))
    p = mu / (k + mu)
    q = 1.0 - p
    pmf = np.zeros(nmax+1, dtype=float)
    pmf[0] = q**k
    coef = 1.0
    for y in range(1, nmax+1):
        coef *= (k + y - 1) / y
        pmf[y] = coef * (q**k) * (p**y)
    s = pmf.sum()
    if s > 0: pmf /= s
    return pmf

def conv_sum(pmf_a: np.ndarray, pmf_b: np.ndarray, nmax: int) -> np.ndarray:
    out = np.zeros(nmax+1, dtype=float)
    for i, pa in enumerate(pmf_a):
        if pa == 0: continue
        jmax = min(nmax - i, len(pmf_b) - 1)
        out[i:i+jmax+1] += pa * pmf_b[:jmax+1]
    s = out.sum()
    if s > 0: out /= s
    return out

def _phi_arr(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def totals_pmf_copula(pmfH: np.ndarray, pmfA: np.ndarray, rho: float, sims: int, seed: int, nmax: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    Z = rng.multivariate_normal(mean=[0.0, 0.0],
                                cov=[[1.0, rho], [rho, 1.0]],
                                size=int(sims))
    U = _phi_arr(Z)
    cdfH = np.cumsum(pmfH)
    cdfA = np.cumsum(pmfA)
    h = np.searchsorted(cdfH, U[:, 0], side="left")
    a = np.searchsorted(cdfA, U[:, 1], side="left")
    h = np.clip(h, 0, nmax)
    a = np.clip(a, 0, nmax)
    tot = h + a
    tot = np.clip(tot, 0, nmax)
    pmfT = np.bincount(tot, minlength=nmax+1).astype(float)
    pmfT /= pmfT.sum()
    return pmfT

# -------- calibration helpers --------
def apply_iso_scalar(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve: return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve["x"], float)
    y = np.array(curve["y"], float)
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))

def blend(p_raw: float, p_cal: float, w: float) -> float:
    w = float(np.clip(w, 0.0, 1.0))
    return float(w * p_cal + (1.0 - w) * p_raw)

def enforce_nonincreasing(lines: List[float], probs: List[float]) -> List[float]:
    """Ensure Over probs are non-increasing as line increases."""
    if not lines: return probs
    idx = np.argsort(lines)  # ascending lines
    p = np.array(probs, float)[idx]
    # cumulative minimum going forward (ensures p[i] >= p[i+1] after reordering back)
    for i in range(1, len(p)):
        if p[i] > p[i-1]:
            p[i] = p[i-1]
    # restore original order
    out = np.zeros_like(p)
    out[idx] = p
    return out.tolist()

def fmt_line(L: float) -> str:
    s = f"{L:.2f}"
    s = s.rstrip("0").rstrip(".")
    return s


class Command(BaseCommand):
    help = "Predict corners OU & team totals markets for upcoming matches (writes PredictedMarket)."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--artifacts", type=str, required=True, help="Path to artifacts.corners.json")
        parser.add_argument("--delete-first", action="store_true")
        parser.add_argument("--sims", type=int, default=20000, help="Copula simulation count (if rho != 0)")
        parser.add_argument("--verbose", action="store_true")

        # New stability knobs
        parser.add_argument("--no-calibration", action="store_true", help="Ignore any isotonic calibration curves.")
        parser.add_argument("--calib-weight", type=float, default=0.6,
                            help="Blend weight for calibration (0=off, 1=full).")
        parser.add_argument("--prob-floor", type=float, default=0.01,
                            help="Clip probabilities into [prob_floor, 1 - prob_floor].")
        parser.add_argument("--enforce-monotone", action="store_true",
                            help="Force Over(L) to be non-increasing as L grows (totals & team totals).")

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        days      = int(opts["days"])
        art_path  = str(opts["artifacts"])
        sims      = int(opts["sims"])
        verbose   = bool(opts["verbose"])

        use_cal   = not bool(opts["no_calibration"])
        w_cal     = float(np.clip(opts["calib_weight"], 0.0, 1.0))
        p_floor   = float(np.clip(opts["prob_floor"], 0.0, 0.49))
        do_monot  = bool(opts["enforce_monotone"])

        # Load artifact
        with open(art_path, "r") as f:
            art = json.load(f)

        lines = art.get("lines", {})
        totals_lines = [float(x) for x in (lines.get("totals", [8.5, 9.5, 10.5]))]
        team_lines   = [float(x) for x in (lines.get("team",   [3.5, 4.5]))]
        nmax_art     = int(art.get("nmax", 30))
        # Make sure nmax is safely above the largest line
        nmax_need    = int(max([0.0] + totals_lines + team_lines)) + 12
        nmax         = max(nmax_art, nmax_need)

        rho          = float(art.get("rho", 0.0))
        mean_floor   = float(art.get("mean_floor", 0.8))

        cal_totals = art.get("totals_calibration", {}) or {}
        cal_team_h = (art.get("team_calibration", {}) or {}).get("home", {}) or {}
        cal_team_a = (art.get("team_calibration", {}) or {}).get("away", {}) or {}

        mapping = art.get("mapping", {}) or {}
        mapping_kind = str(art.get("mapping_kind", mapping.get("kind", "heuristic")))

        # NB dispersion maps
        disp = art.get("dispersion", {}) or {}
        kH_global = float(disp.get("k_home_global", 150.0))
        kA_global = float(disp.get("k_away_global", 150.0))

        def _parse_map(m: Dict[str, float]) -> Dict[Tuple[int,int], float]:
            out: Dict[Tuple[int,int], float] = {}
            for k, v in (m or {}).items():
                try:
                    lg, ss = k.split("-", 1)
                    out[(int(lg), int(ss))] = float(v)
                except Exception:
                    continue
            return out

        kH_map = _parse_map(disp.get("k_home_map", {}))
        kA_map = _parse_map(disp.get("k_away_map", {}))

        # μ mapping
        def predict_means_from_goals(muH: float, muA: float) -> Tuple[float, float]:
            eps = 1e-6
            def _clip(x): return float(np.clip(x, mean_floor, 15.0))
            if mapping_kind == "glm":
                ch = mapping["home"]; sh = float(mapping["scale"]["home"])
                ca = mapping["away"]; sa = float(mapping["scale"]["away"])
                zH = ch["intercept"] + ch["b1"]*math.log(max(eps, muH)) + ch["b2"]*math.log(max(eps, muA))
                zA = ca["intercept"] + ca["b1"]*math.log(max(eps, muA)) + ca["b2"]*math.log(max(eps, muH))
                return _clip(math.exp(zH)*sh), _clip(math.exp(zA)*sa)
            if mapping_kind == "heuristic":
                p = mapping["params"]
                alpha = float(p["alpha"]); beta = float(p["beta"]); mu_bar = float(p["mu_bar"])
                baseH = float(p["prior_home"]); baseA = float(p["prior_away"])
                denom = max(eps, mu_bar*(1.0+beta))
                mH = _clip(baseH * ((muH + beta*muA)/denom)**alpha)
                mA = _clip(baseA * ((muA + beta*muH)/denom)**alpha)
                return mH, mA
            baseH = float(mapping.get("fallback_means", {}).get("home", 5.2))
            baseA = float(mapping.get("fallback_means", {}).get("away", 4.8))
            return _clip(baseH), _clip(baseA)

        # time window
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

        # Delete existing markets first if asked
        if bool(opts["delete_first"]):
            with transaction.atomic():
                PredictedMarket.objects.filter(
                    match__prediction__league_id=league_id,
                    kickoff_utc__gte=now,
                    kickoff_utc__lte=upto,
                    market_code__in=["OU_CORNERS","TEAM_TOTAL_CORNERS"],
                ).delete()

        wrote = 0
        for mp in qs:
            # goals μ from MatchPrediction
            muH_goals = float(np.clip(getattr(mp, "lambda_home", 0.0), 0.05, 6.0))
            muA_goals = float(np.clip(getattr(mp, "lambda_away", 0.0), 0.05, 6.0))

            # corners μ via mapping
            mH, mA = predict_means_from_goals(muH_goals, muA_goals)

            # NB dispersion: league-season if present
            season = getattr(mp.match, "season", None)
            kH = kH_map.get((int(mp.league_id), int(season))) if season is not None else None
            kA = kA_map.get((int(mp.league_id), int(season))) if season is not None else None
            if kH is None: kH = kH_global
            if kA is None: kA = kA_global

            pmfH = nb_pmf_vec(mH, kH, nmax)
            pmfA = nb_pmf_vec(mA, kA, nmax)
            pmfT = conv_sum(pmfH, pmfA, nmax) if abs(rho) < 1e-9 \
                else totals_pmf_copula(pmfH, pmfA, rho=float(rho), sims=int(sims), seed=int(mp.id), nmax=nmax)

            # -------- Totals (with calibration blend + monotone + floor) --------
            totals_over_raw: List[float] = []
            totals_over_out: List[float] = []
            for L in totals_lines:
                thr = math.floor(L + 1e-9) + 1  # P(T >= ceil(L+epsilon))
                p_over_raw = float(pmfT[thr:].sum())
                if use_cal:
                    curve = cal_totals.get(fmt_line(L))
                    p_cal = apply_iso_scalar(p_over_raw, curve)
                    p_over = blend(p_over_raw, p_cal, w_cal)
                else:
                    p_over = p_over_raw
                totals_over_raw.append(p_over_raw)
                totals_over_out.append(p_over)

            # Enforce non-increasing across lines if requested
            if do_monot:
                totals_over_out = enforce_nonincreasing(totals_lines, totals_over_out)

            # Clip to practical range
            totals_over_out = [float(np.clip(p, p_floor, 1.0 - p_floor)) for p in totals_over_out]

            # Persist totals
            for L, p_over in zip(totals_lines, totals_over_out):
                p_under = float(1.0 - p_over)
                line_str = fmt_line(L)
                for spec, p in ((f"{line_str}_over", p_over), (f"{line_str}_under", p_under)):
                    PredictedMarket.objects.update_or_create(
                        match=mp.match,
                        market_code="OU_CORNERS",
                        specifier=spec,
                        defaults={
                            "league_id": mp.league_id,
                            "kickoff_utc": mp.kickoff_utc,
                            "p_model": float(p),
                            "fair_odds": float(1.0 / p),
                            # store corners μ in lambda_* for audit
                            "lambda_home": float(mH),
                            "lambda_away": float(mA),
                        },
                    )
                    wrote += 1

            # -------- Team totals (same treatment, per side) --------
            cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)

            # Home
            home_over = []
            for L in team_lines:
                thr = math.floor(L + 1e-9) + 1
                idxH = min(thr - 1, nmax)
                p_raw = float(1.0 - cdfH[idxH])
                if use_cal:
                    p_cal = apply_iso_scalar(p_raw, cal_team_h.get(fmt_line(L)))
                    p = blend(p_raw, p_cal, w_cal)
                else:
                    p = p_raw
                home_over.append(p)
            if do_monot:
                home_over = enforce_nonincreasing(team_lines, home_over)
            home_over = [float(np.clip(p, p_floor, 1.0 - p_floor)) for p in home_over]

            # Away
            away_over = []
            for L in team_lines:
                thr = math.floor(L + 1e-9) + 1
                idxA = min(thr - 1, nmax)
                p_raw = float(1.0 - cdfA[idxA])
                if use_cal:
                    p_cal = apply_iso_scalar(p_raw, cal_team_a.get(fmt_line(L)))
                    p = blend(p_raw, p_cal, w_cal)
                else:
                    p = p_raw
                away_over.append(p)
            if do_monot:
                away_over = enforce_nonincreasing(team_lines, away_over)
            away_over = [float(np.clip(p, p_floor, 1.0 - p_floor)) for p in away_over]

            for L, pH, pA in zip(team_lines, home_over, away_over):
                for spec, p in ((f"home_o{fmt_line(L)}", pH), (f"away_o{fmt_line(L)}", pA)):
                    PredictedMarket.objects.update_or_create(
                        match=mp.match,
                        market_code="TEAM_TOTAL_CORNERS",
                        specifier=spec,
                        defaults={
                            "league_id": mp.league_id,
                            "kickoff_utc": mp.kickoff_utc,
                            "p_model": float(p),
                            "fair_odds": float(1.0 / p),
                            "lambda_home": float(mH),
                            "lambda_away": float(mA),
                        },
                    )
                    wrote += 1

            if verbose:
                self.stdout.write(
                    f"{mp.id} | μ_corners≈({mH:.2f},{mA:.2f}) "
                    f"| OU over@{[fmt_line(L) for L in totals_lines]} → "
                    f"{[f'{p*100:.1f}%' for p in totals_over_out]}"
                )

        self.stdout.write(self.style.SUCCESS(
            f"Wrote/updated {wrote} PredictedMarket rows (OU_CORNERS / TEAM_TOTAL_CORNERS) for league {league_id}"
        ))
