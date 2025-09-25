import math, numpy as np
from django.core.management.base import BaseCommand
from matches.models import MatchPrediction, PredictedMarket

def _pois_pmf(k, lam):
    k = int(k); lam = max(1e-9, float(lam))
    return math.exp(-lam) * lam**k / math.factorial(k)

def _one_by_two(lh, la, max_goals=12):
    lh = float(np.clip(lh, 1e-6, 10.0))
    la = float(np.clip(la, 1e-6, 10.0))
    pmf_h = [_pois_pmf(i, lh) for i in range(max_goals + 1)]
    pmf_a = [_pois_pmf(j, la) for j in range(max_goals + 1)]
    pH = pD = pA = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pmf_h[i] * pmf_a[j]
            if i>j: pH += p
            elif i==j: pD += p
            else: pA += p
    s = pH + pD + pA
    if s>0: pH, pD, pA = pH/s, pD/s, pA/s
    return pH, pD, pA

def _p_btts(lh, la):
    e_h = math.exp(-lh); e_a = math.exp(-la)
    return float(np.clip(1.0 - e_h - e_a + e_h*e_a, 0.0, 1.0))

def _p_total_over_k(lh, la, k):
    lam = lh + la
    term, s = 1.0, 1.0
    for n in range(1, k+1):
        term *= lam / n
        s += term
    return float(np.clip(1.0 - math.exp(-lam)*s, 0.0, 1.0))

class Command(BaseCommand):
    help = "Verify that markets priced from stored lambdas match PredictedMarket (1X2, BTTS, OU)."

    def add_arguments(self, parser):
        parser.add_argument("--match-id", type=int, required=True)
        parser.add_argument("--max-goals", type=int, default=12)
        parser.add_argument("--eps", type=float, default=1e-6)

    def handle(self, *args, **opts):
        mid = opts["match_id"]; MG = int(opts["max_goals"]); EPS = float(opts["eps"])

        mp = MatchPrediction.objects.select_related("match").get(match_id=mid)
        lh, la = float(mp.lambda_home), float(mp.lambda_away)

        calc = {
            ("1X2","H"): _one_by_two(lh, la, MG)[0],
            ("1X2","D"): _one_by_two(lh, la, MG)[1],
            ("1X2","A"): _one_by_two(lh, la, MG)[2],
            ("BTTS","yes"): _p_btts(lh, la),
            ("BTTS","no"): 1.0 - _p_btts(lh, la),
            ("OU","1.5_over"): _p_total_over_k(lh, la, 1),
            ("OU","1.5_under"): 1.0 - _p_total_over_k(lh, la, 1),
            ("OU","2.5_over"): _p_total_over_k(lh, la, 2),
            ("OU","2.5_under"): 1.0 - _p_total_over_k(lh, la, 2),
        }

        db_rows = PredictedMarket.objects.filter(match_id=mid, market_code__in=["1X2","BTTS","OU"])
        db = {(r.market_code, r.specifier): float(r.p_model) for r in db_rows}

        self.stdout.write(f"λ_home={lh:.6f} λ_away={la:.6f}")
        bad = 0
        for k,v in calc.items():
            got = db.get(k)
            if got is None:
                self.stdout.write(self.style.WARNING(f"{k}: calc={v:.6f} db=NA"))
                continue
            diff = abs(v - got)
            line = f"{k}: calc={v:.6f} db={got:.6f} diff={diff:.6f}"
            if diff > EPS and k[0] == "1X2":
                bad += 1
                self.stdout.write(self.style.ERROR(line))
            else:
                self.stdout.write(line)

        if bad == 0:
            self.stdout.write(self.style.SUCCESS("1X2 matches. BTTS/OU may differ if calibration is applied (expected)."))
        else:
            self.stdout.write(self.style.ERROR("Differences detected — check lambdas, max_goals grid, or calibration."))













# prediction/matches/management/commands/verify_pricing.py

import math
import numpy as np
from datetime import datetime, timedelta, timezone

from django.core.management.base import BaseCommand

from matches.models import Match, MatchPrediction, PredictedMarket


# ----------------------- shared helpers (mirror pricing) -----------------------

def _finite(x, d=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(d)

def _apply_cal(calibrators: dict | None, key: str, p: float) -> float:
    """
    Apply isotonic calibration if available; otherwise return p unchanged.
    """
    if p is None or not np.isfinite(p):
        return 0.5
    p = float(np.clip(p, 0.0, 1.0))
    if not calibrators:
        return p
    model = calibrators.get(key)
    if model is None:
        return p
    try:
        out = model.predict(np.array([p], dtype=np.float64))[0]
        if np.isfinite(out):
            return float(np.clip(out, 0.0, 1.0))
        return p
    except Exception:
        return p

def _calibrators_for_league(league_id: int) -> dict:
    """
    Load the latest saved isotonic calibrators for this league, if present.
    Returns dict like {"over25": iso_model, "btts": iso_model} or {}.
    """
    import json, joblib
    from matches.models import ModelVersion

    mv = (ModelVersion.objects
          .filter(kind="goals", league_id=league_id)
          .order_by("-trained_until", "-id")
          .first())
    if not mv:
        return {}

    calinfo = mv.calibration_json
    try:
        if isinstance(calinfo, str):
            calinfo = json.loads(calinfo or "{}")
        elif calinfo is None:
            calinfo = {}
    except Exception:
        calinfo = {}

    cal_file = (calinfo or {}).get("file")
    if not cal_file:
        return {}
    try:
        cals = joblib.load(cal_file)
        return cals if isinstance(cals, dict) else {}
    except Exception:
        return {}

def _prob_btts(lh: float, la: float) -> float:
    # 1 - e^{-lh} - e^{-la} + e^{-(lh+la)}
    e_h = math.exp(-lh)
    e_a = math.exp(-la)
    return float(np.clip(1.0 - e_h - e_a + math.exp(-(lh + la)), 0.0, 1.0))

def _prob_over_k_total(lh: float, la: float, k: int) -> float:
    """
    P(Total > k) = 1 - P(Total <= k) for Poisson(lh+la), via stable series sum.
    """
    lam = float(lh + la)
    if lam <= 0:
        return 0.0
    term, s = 1.0, 1.0  # n=0
    for n in range(1, int(k) + 1):
        term *= lam / n
        s += term
    p = 1.0 - math.exp(-lam) * s
    return float(np.clip(p, 0.0, 1.0))

def _make_grid_probs(lh: float, la: float, max_goals: int = 10, dc_rho: float | None = None):
    """
    Build independent Poisson score grid (0..max_goals) x (0..max_goals),
    optional tiny diagonal-correlation tweak (dc_rho ~ 0..0.2).
    Returns aggregates identical in spirit to predict_full_markets.
    """
    lh = float(np.clip(lh, 1e-6, 10.0))
    la = float(np.clip(la, 1e-6, 10.0))

    ks = np.arange(0, max_goals + 1, dtype=float)
    # use lgamma for stable factorial in log-space
    lgfact = np.vectorize(math.lgamma)(ks + 1.0)

    log_ph = ks * math.log(lh) - lh - lgfact
    log_pa = ks * math.log(la) - la - lgfact

    # stabilize by subtracting max
    ph = np.exp(log_ph - np.max(log_ph))
    ph = ph / ph.sum() if ph.sum() > 0 else np.zeros_like(ph)

    pa = np.exp(log_pa - np.max(log_pa))
    pa = pa / pa.sum() if pa.sum() > 0 else np.zeros_like(pa)

    # outer product => independent grid
    P = np.outer(ph, pa)
    S = P.sum()
    P = P / S if S > 0 else np.eye(1)  # renormalize in case of truncation

    # Diagonal correlation nudges (small 0..0.2)
    if dc_rho is not None and 0.0 <= dc_rho <= 0.2:
        adj = np.ones_like(P)
        # simplest local tweak near 0-0 / 1-0 / 0-1 / 1-1 cells
        adj[0, 0] *= (1 + dc_rho)
        if P.shape[0] > 1: adj[1, 0] *= (1 - dc_rho)
        if P.shape[1] > 1: adj[0, 1] *= (1 - dc_rho)
        if P.shape[0] > 1 and P.shape[1] > 1: adj[1, 1] *= (1 + dc_rho / 2.0)
        P *= adj
        s = P.sum()
        if s > 0 and np.isfinite(s):
            P /= s

    # aggregates
    idx = np.arange(P.shape[0])
    p_home = float(P[idx[:, None] > idx[None, :]].sum())
    p_draw = float(np.trace(P))
    p_away = float(P[idx[:, None] < idx[None, :]].sum())

    # team totals using row/col sums
    home_ge1 = float(P[1:, :].sum())
    home_ge2 = float(P[2:, :].sum())
    away_ge1 = float(P[:, 1:].sum())
    away_ge2 = float(P[:, 2:].sum())

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "home_ge1": home_ge1,
        "home_ge2": home_ge2,
        "away_ge1": away_ge1,
        "away_ge2": away_ge2,
    }


# ------------------------------------ command ------------------------------------

class Command(BaseCommand):
    help = "Verify PredictedMarket for a match by recomputing RAW and CALIBRATED probs from stored lambdas."

    def add_arguments(self, parser):
        parser.add_argument("--match-id", type=int, required=True)
        parser.add_argument("--max-goals", type=int, default=10, help="Score grid size for 1X2.")
        parser.add_argument("--dc-rho", type=float, default=0.04, help="Diagonal-correlation tweak (0..0.2).")
        parser.add_argument("--no-cal", action="store_true", help="Ignore calibrators (show raw only).")

    def handle(self, *args, **opts):
        match_id = int(opts["match_id"])
        max_goals = int(opts["max_goals"])
        dc_rho = float(opts["dc_rho"])
        use_cal = not bool(opts["no_cal"])

        # Load lambdas from MatchPrediction
        mp = (MatchPrediction.objects
              .select_related("match")
              .filter(match_id=match_id)
              .first())
        if not mp:
            self.stderr.write(self.style.ERROR(f"No MatchPrediction found for match {match_id}. Run predict_markets first."))
            return

        m = mp.match
        lh = float(np.clip(_finite(mp.lambda_home, 1.2), 1e-6, 10.0))
        la = float(np.clip(_finite(mp.lambda_away, 1.0), 1e-6, 10.0))

        self.stdout.write(f"λ_home={lh:.6f} λ_away={la:.6f}")

        # Load DB markets for comparison
        db_rows = PredictedMarket.objects.filter(match_id=match_id)
        db = {(r.market_code, r.specifier): float(r.p_model) for r in db_rows}

        # Load calibrators (same source as predict_full_markets)
        cals = _calibrators_for_league(int(mp.league_id)) if use_cal else {}
        if cals:
            self.stdout.write(f"Loaded calibrators: {sorted(cals.keys())}")
        else:
            self.stdout.write("No calibrators loaded (raw-only) — use what predict_full_markets would if available.")

        # Recompute using identical math to pricing
        agg = _make_grid_probs(lh, la, max_goals=max_goals, dc_rho=dc_rho)

        # 1X2 from grid (renormalize just in case)
        pH_raw = float(agg["p_home"])
        pD_raw = float(agg["p_draw"])
        pA_raw = float(agg["p_away"])
        s = pH_raw + pD_raw + pA_raw
        if s > 0:
            pH_raw, pD_raw, pA_raw = pH_raw / s, pD_raw / s, pA_raw / s
        pH_cal, pD_cal, pA_cal = pH_raw, pD_raw, pA_raw  # no calibration for 1X2

        # Totals & BTTS (mirror pricing behavior)
        p_o15_raw = _prob_over_k_total(lh, la, 1)
        p_o25_raw = _prob_over_k_total(lh, la, 2)
        p_o35_raw = _prob_over_k_total(lh, la, 3)
        p_btts_raw = _prob_btts(lh, la)

        p_o15_cal = p_o15_raw  # no calibration at 1.5
        p_o25_cal = _apply_cal(cals, "over25", p_o25_raw)
        p_o35_cal = p_o35_raw   # no calibration at 3.5
        p_btts_cal = _apply_cal(cals, "btts", p_btts_raw)

        # Team totals from grid aggregates (same as pricing)
        home_o05_raw = float(agg["home_ge1"]); home_o15_raw = float(agg["home_ge2"])
        away_o05_raw = float(agg["away_ge1"]); away_o15_raw = float(agg["away_ge2"])
        # no calibration for team totals
        home_o05_cal = home_o05_raw; home_o15_cal = home_o15_raw
        away_o05_cal = away_o05_raw; away_o15_cal = away_o15_raw

        # Compare (we clamp like pricing before writing)
        def clamp(p): return float(np.clip(p, 1e-9, 1 - 1e-9))

        comparisons = [
            ("1X2", "H", pH_raw, pH_cal),
            ("1X2", "D", pD_raw, pD_cal),
            ("1X2", "A", pA_raw, pA_cal),
            ("BTTS", "yes", p_btts_raw, p_btts_cal),
            ("BTTS", "no", 1.0 - p_btts_raw, 1.0 - p_btts_cal),
            ("OU", "1.5_over", p_o15_raw, p_o15_cal),
            ("OU", "1.5_under", 1.0 - p_o15_raw, 1.0 - p_o15_cal),
            ("OU", "2.5_over", p_o25_raw, p_o25_cal),
            ("OU", "2.5_under", 1.0 - p_o25_raw, 1.0 - p_o25_cal),
            ("OU", "3.5_over", p_o35_raw, p_o35_cal),
            ("OU", "3.5_under", 1.0 - p_o35_raw, 1.0 - p_o35_cal),
            ("TEAM_TOTAL", "home_o0.5", home_o05_raw, home_o05_cal),
            ("TEAM_TOTAL", "home_o1.5", home_o15_raw, home_o15_cal),
            ("TEAM_TOTAL", "away_o0.5", away_o05_raw, away_o05_cal),
            ("TEAM_TOTAL", "away_o1.5", away_o15_raw, away_o15_cal),
            ("AH", "home_0", pH_raw + 0.5 * pD_raw, pH_cal + 0.5 * pD_cal),
            ("AH", "away_0", pA_raw + 0.5 * pD_raw, pA_cal + 0.5 * pD_cal),
            ("AH", "home_-0.5", pH_raw, pH_cal),
            ("AH", "away_+0.5", pA_raw, pA_cal),
        ]

        any_diff = False
        for market, spec, p_raw, p_cal in comparisons:
            p_raw = clamp(p_raw)
            p_cal = clamp(p_cal)
            p_db = db.get((market, spec))
            if p_db is None:
                self.stdout.write(f"[missing in DB] ({market!r}, {spec!r}) calc_raw={p_raw:.6f} calc_cal={p_cal:.6f}")
                any_diff = True
                continue
            diff_raw = abs(p_raw - p_db)
            diff_cal = abs(p_cal - p_db)
            # show both, and highlight which should match pricing (calibrated)
            self.stdout.write(
                f"({market!r}, {spec!r}): raw={p_raw:.6f} cal={p_cal:.6f} db={p_db:.6f} "
                f"| diff(raw-db)={diff_raw:.6f} diff(cal-db)={diff_cal:.6f}"
            )
            # if calibrated markets mismatch, flag
            if diff_cal > 1e-6 and (market, spec) in {
                ("BTTS", "yes"), ("BTTS", "no"),
                ("OU", "2.5_over"), ("OU", "2.5_under")
            }:
                any_diff = True
            # for others, raw == cal so either diff indicates a grid/parity issue
            if (market, spec) not in {("BTTS", "yes"), ("BTTS", "no"),
                                      ("OU", "2.5_over"), ("OU", "2.5_under")}:
                if diff_raw > 1e-6:
                    any_diff = True

        if any_diff:
            self.stdout.write(self.style.WARNING("Differences detected — check lambdas, max_goals, dc_rho, or calibration presence."))
        else:
            self.stdout.write(self.style.SUCCESS("All markets match (using calibrated values where applicable)."))
