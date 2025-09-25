# prediction/matches/management/commands/predict_cards_markets.py
from __future__ import annotations

import json
import math
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from matches.models import MatchPrediction, PredictedMarket

# Numerical floors for market output (avoid 0 / 1 and silly 1e9 odds)
EPS_CAL = 1e-4      # floor/ceiling after calibration
EPS_RAW = 1e-9      # internal floor for intermediate calcs

# ---------- math helpers ----------
def _phi_arr(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF (vectorized) without relying on numpy.erf."""
    z = np.asarray(z, dtype=float)
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))

def nb_pmf_vec(mu: float, k: float, nmax: int) -> np.ndarray:
    mu = max(1e-9, float(mu))
    k = max(1e-9, float(k))
    p = mu / (k + mu)
    q = 1.0 - p
    pmf = np.zeros(nmax + 1, dtype=float)
    # numerically stable: q**k ≈ exp(k * log(q))
    pmf[0] = float(np.exp(k * np.log(max(1e-15, q))))
    coef = 1.0
    for y in range(1, nmax + 1):
        coef *= (k + y - 1) / y
        pmf[y] = coef * pmf[0] * (p**y)
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf

def conv_sum(pmf_a: np.ndarray, pmf_b: np.ndarray, nmax: int) -> np.ndarray:
    out = np.zeros(nmax + 1, dtype=float)
    for i, pa in enumerate(pmf_a):
        if pa == 0.0:
            continue
        jmax = min(nmax - i, len(pmf_b) - 1)
        out[i : i + jmax + 1] += pa * pmf_b[: jmax + 1]
    s = out.sum()
    if s > 0.0:
        out /= s
    return out

def totals_pmf_copula(
    pmfH: np.ndarray, pmfA: np.ndarray, rho: float, sims: int, seed: int, nmax: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    Z = rng.multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]], size=int(sims)
    )
    U = _phi_arr(Z)
    cdfH = np.cumsum(pmfH)
    cdfA = np.cumsum(pmfA)
    h = np.searchsorted(cdfH, U[:, 0], side="left")
    a = np.searchsorted(cdfA, U[:, 1], side="left")
    h = np.clip(h, 0, nmax)
    a = np.clip(a, 0, nmax)
    tot = np.clip(h + a, 0, nmax)
    pmfT = np.bincount(tot, minlength=nmax + 1).astype(float)
    pmfT /= pmfT.sum()
    return pmfT, h, a

# ---------- calibration helpers ----------
def apply_iso_scalar(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve:
        return float(np.clip(p, 0.0, 1.0))
    x = np.asarray(curve["x"], float)
    y = np.asarray(curve["y"], float)
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))

def apply_iso_scalar_safe(p_raw: float,
                          curve: Optional[Dict[str, List[float]]],
                          clip: float = EPS_CAL,
                          blend: float = 0.25) -> float:
    """
    Safe calibration:
      - interpolate with isotonic curve (if present)
      - clamp away from 0/1 using `clip`
      - if the curve maps to extreme (near 0/1) but raw is moderate (0.05..0.95),
        blend back toward raw by `blend`.
    """
    p_raw = float(np.clip(p_raw, EPS_RAW, 1.0 - EPS_RAW))
    if not curve:
        return float(np.clip(p_raw, clip, 1.0 - clip))

    pc = apply_iso_scalar(p_raw, curve)
    pc = float(np.clip(pc, clip, 1.0 - clip))

    # Overcorrection guard: if isotonic pushes moderate raw p to an extreme, blend back.
    if 0.05 <= p_raw <= 0.95 and (pc <= 2 * clip or pc >= 1.0 - 2 * clip):
        pc = (1.0 - blend) * pc + blend * p_raw
        pc = float(np.clip(pc, clip, 1.0 - clip))
    return pc

# ---------- artifact I/O ----------
def _load_artifact(path: str) -> Dict:
    with open(path, "r") as f:
        art = json.load(f)
    for k in ["mapping", "dispersion", "lines", "nmax"]:
        if k not in art:
            raise ValueError(f"artifact missing key '{k}'")
    return art

# ---------- goals λ -> cards μ ----------
def _predict_means_from_mapping(
    lam_home: float, lam_away: float, mapping: Dict, mean_floor: float
) -> Tuple[float, float]:
    lam_home = float(max(EPS_RAW, lam_home))
    lam_away = float(max(EPS_RAW, lam_away))

    def _clip(x: float) -> float:
        # cards are small counts; a mild floor helps avoid degenerate tails
        return float(np.clip(x, max(0.15, mean_floor), 8.0))

    kind = mapping.get("kind") or mapping.get("mapping_kind") or mapping.get("type", "")
    if kind == "glm" or (mapping.get("home") and mapping.get("away")):
        ch = mapping["home"]
        ca = mapping["away"]
        sh = float(mapping["scale"]["home"])
        sa = float(mapping["scale"]["away"])
        zH = ch["intercept"] + ch["b1"] * math.log(lam_home) + ch["b2"] * math.log(lam_away)
        zA = ca["intercept"] + ca["b1"] * math.log(lam_away) + ca["b2"] * math.log(lam_home)
        mH = _clip(math.exp(zH) * sh)
        mA = _clip(math.exp(zA) * sa)
        return mH, mA

    if kind == "constant" or "means" in mapping:
        baseH = float(mapping["means"]["home"])
        baseA = float(mapping["means"]["away"])
        return _clip(baseH), _clip(baseA)

    fb = mapping.get("fallback_means", {"home": 2.0, "away": 1.8})
    return _clip(float(fb["home"])), _clip(float(fb["away"]))

# ---------- management command ----------
class Command(BaseCommand):
    help = "Write card markets (CARDS_TOT, CARDS_TEAM) from a trained artifacts.cards.<type>.json."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--artifact", type=str, required=True)
        parser.add_argument("--delete-first", action="store_true")
        parser.add_argument("--sims", type=int, default=20000)
        parser.add_argument("--no-calibration", action="store_true",
                            help="Skip isotonic calibration application (use raw model probabilities).")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--debug", action="store_true")

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        days = int(opts["days"])
        artifact_path = str(opts["artifact"])
        sims = int(opts["sims"])
        use_cal = not bool(opts["no_calibration"])
        verbose = bool(opts["verbose"])

        art = _load_artifact(artifact_path)
        mapping = art["mapping"]
        nmax = int(art.get("nmax", 15))
        mean_floor = float(art.get("mean_floor", 0.05))
        rho = float(art.get("rho", 0.0))

        lines_totals: List[float] = [float(x) for x in art["lines"].get("totals", [3.5, 4.5, 5.5])]
        lines_team: List[float] = [float(x) for x in art["lines"].get("team", [1.5, 2.5])]

        cal_totals: Dict[str, Dict[str, List[float]]] = art.get("totals_calibration", {}) or {}
        team_cal = art.get("team_calibration", {}) or {}
        cal_team_home: Dict[str, Dict[str, List[float]]] = team_cal.get("home", {}) or {}
        cal_team_away: Dict[str, Dict[str, List[float]]] = team_cal.get("away", {}) or {}

        disp = art["dispersion"]
        kH_global = float(disp["k_home_global"])
        kA_global = float(disp["k_away_global"])
        kH_map = {tuple(map(int, k.split("-"))): float(v) for k, v in disp.get("k_home_map", {}).items()}
        kA_map = {tuple(map(int, k.split("-"))): float(v) for k, v in disp.get("k_away_map", {}).items()}

        now = timezone.now()
        upto = now + timedelta(days=days)

        qs = (
            MatchPrediction.objects.filter(
                league_id=league_id,
                kickoff_utc__gte=now,
                kickoff_utc__lte=upto,
                match__status__in=["NS", "PST", "TBD"],
            )
            .select_related("match")
            .order_by("kickoff_utc")
        )

        if not qs.exists():
            self.stdout.write("No MatchPrediction rows in window.")
            return

        if bool(opts["delete_first"]):
            with transaction.atomic():
                PredictedMarket.objects.filter(
                    match__prediction__league_id=league_id,
                    kickoff_utc__gte=now,
                    kickoff_utc__lte=upto,
                    market_code__in=["CARDS_TOT", "CARDS_TEAM"],
                ).delete()

        def _season_for(mp: MatchPrediction) -> int:
            try:
                return int(mp.match.season)
            except Exception:
                return int(mp.kickoff_utc.year)

        def _k_for(lg: int, ssn: int, side: str) -> float:
            key = (int(lg), int(ssn))
            return kH_map.get(key, kH_global) if side == "H" else kA_map.get(key, kA_global)

        wrote = 0
        warn_count = 0

        for mp in qs:
            lam_home = float(getattr(mp, "lambda_home", 0.0) or 0.0)
            lam_away = float(getattr(mp, "lambda_away", 0.0) or 0.0)
            if lam_home <= 0 or lam_away <= 0:
                if verbose:
                    self.stdout.write(f"{mp.id} | missing λ goals; skipping")
                continue

            ssn = _season_for(mp)
            mH, mA = _predict_means_from_mapping(lam_home, lam_away, mapping, mean_floor)
            kH = _k_for(league_id, ssn, "H")
            kA = _k_for(league_id, ssn, "A")

            pmfH = nb_pmf_vec(mH, kH, nmax)
            pmfA = nb_pmf_vec(mA, kA, nmax)

            if abs(rho) < 1e-9:
                pmfT = conv_sum(pmfH, pmfA, nmax)
            else:
                pmfT, _, _ = totals_pmf_copula(pmfH, pmfA, rho=rho, sims=sims, seed=int(mp.id), nmax=nmax)

            if verbose:
                hn = getattr(mp.match.home, "name", str(mp.match.home_id))
                an = getattr(mp.match.away, "name", str(mp.match.away_id))
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an} | μ_cards≈({mH:.2f},{mA:.2f}) | k≈({kH:.1f},{kA:.1f}) | ρ={rho:.3f}"
                )

            # ---------- totals ----------
            for L in lines_totals:
                thr = math.floor(L + 1e-9) + 1
                p_over_raw = float(pmfT[thr:].sum())
                p_over = (apply_iso_scalar_safe(p_over_raw, cal_totals.get(str(L)))
                          if use_cal else float(np.clip(p_over_raw, EPS_CAL, 1.0 - EPS_CAL)))
                p_under = 1.0 - p_over

                if p_over <= EPS_CAL or p_over >= 1.0 - EPS_CAL:
                    warn_count += 1

                PredictedMarket.objects.update_or_create(
                    match=mp.match,
                    market_code="CARDS_TOT",
                    specifier=f"L={L}_OVER",
                    defaults={
                        "league_id": mp.league_id,
                        "kickoff_utc": mp.kickoff_utc,
                        "p_model": p_over,
                        "fair_odds": float(1.0 / p_over),
                        "lambda_home": float(mH),
                        "lambda_away": float(mA),
                    },
                )
                wrote += 1

                PredictedMarket.objects.update_or_create(
                    match=mp.match,
                    market_code="CARDS_TOT",
                    specifier=f"L={L}_UNDER",
                    defaults={
                        "league_id": mp.league_id,
                        "kickoff_utc": mp.kickoff_utc,
                        "p_model": p_under,
                        "fair_odds": float(1.0 / p_under),
                        "lambda_home": float(mH),
                        "lambda_away": float(mA),
                    },
                )
                wrote += 1

            # ---------- team totals ----------
            cdfH = np.cumsum(pmfH)
            cdfA = np.cumsum(pmfA)
            for L in lines_team:
                thr = math.floor(L + 1e-9) + 1
                idx = max(0, min(nmax, thr - 1))

                pH_over_raw = float(1.0 - cdfH[idx])
                pA_over_raw = float(1.0 - cdfA[idx])

                pH_over = (apply_iso_scalar_safe(pH_over_raw, cal_team_home.get(str(L)))
                           if use_cal else float(np.clip(pH_over_raw, EPS_CAL, 1.0 - EPS_CAL)))
                pA_over = (apply_iso_scalar_safe(pA_over_raw, cal_team_away.get(str(L)))
                           if use_cal else float(np.clip(pA_over_raw, EPS_CAL, 1.0 - EPS_CAL)))

                pH_under = 1.0 - pH_over
                pA_under = 1.0 - pA_over

                if pA_over <= EPS_CAL or pA_over >= 1.0 - EPS_CAL or pH_over <= EPS_CAL or pH_over >= 1.0 - EPS_CAL:
                    warn_count += 1

                for side, p_over_side, p_under_side in [
                    ("HOME", pH_over, pH_under),
                    ("AWAY", pA_over, pA_under),
                ]:
                    PredictedMarket.objects.update_or_create(
                        match=mp.match,
                        market_code="CARDS_TEAM",
                        specifier=f"{side}_OVER_{L}",
                        defaults={
                            "league_id": mp.league_id,
                            "kickoff_utc": mp.kickoff_utc,
                            "p_model": p_over_side,
                            "fair_odds": float(1.0 / p_over_side),
                            "lambda_home": float(mH),
                            "lambda_away": float(mA),
                        },
                    )
                    wrote += 1

                    PredictedMarket.objects.update_or_create(
                        match=mp.match,
                        market_code="CARDS_TEAM",
                        specifier=f"{side}_UNDER_{L}",
                        defaults={
                            "league_id": mp.league_id,
                            "kickoff_utc": mp.kickoff_utc,
                            "p_model": p_under_side,
                            "fair_odds": float(1.0 / p_under_side),
                            "lambda_home": float(mH),
                            "lambda_away": float(mA),
                        },
                    )
                    wrote += 1

        suffix = "" if use_cal else " (no calibration)"
        msg = f"Wrote/updated {wrote} PredictedMarket rows (CARDS_TOT / CARDS_TEAM){suffix} for league {league_id}"
        if warn_count and verbose:
            msg += f" | clipped/extreme probs encountered: {warn_count}"
        self.stdout.write(self.style.SUCCESS(msg))
