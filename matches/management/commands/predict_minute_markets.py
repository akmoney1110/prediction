# prediction/matches/management/commands/predict_minutes_markets.py
from __future__ import annotations

"""
Write rolling 1X2-by-minute markets, e.g. T=5,10,...,85.

Core ideas
- Build a bivariate Poisson grid for a time fraction f = T/90 by scaling λ's.
- Apply a small diagonal (draw) fudge that reduces 0-0 / 1-1 mass and
  softly compensates near the diagonal. This curbs draw inflation.
- (Optional) Apply a minute-aware heuristic draw shrink that is strongest
  at early minutes and fades to 0 by 90'. Preserves H/A ratio.

Outputs
- PredictedMarket rows with market_code="1X2_T" and specifier like "T=15_H/D/A".

Usage examples
  export DJANGO_SETTINGS_MODULE=prediction.settings
  python manage.py predict_minutes_markets \
      --league-id 39 \
      --days 7 \
      --artifacts artifacts/goals/artifacts.goals.json \
      --use-artifacts-c \
      --minute-grid 5:85:5 \
      --minute-draw-shrink 0.08 \
      --tune-verbose \
      --delete-first \
      --verbose
"""

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction

from matches.models import MatchPrediction, PredictedMarket

# ----------------------------- Tunables / defaults -----------------------------
DEFAULT_MAX_GOALS_GRID = 10
DEFAULT_RHO = 0.10                 # shared component scale for bp via sqrt(lh*la)
DEFAULT_RHO_MAX = 0.35
DEFAULT_EPS = 1e-9

DEFAULT_MINUTE_GRID = [5 * i for i in range(1, 18)]  # 5,10,...,85
DEFAULT_C_TAPER = 0.60
DEFAULT_SWAP_MARGIN = 0.25

# ----------------------------- numeric helpers -----------------------------
def _bp_grid_from_components(lam1: float, lam2: float, lam12: float, max_goals: int) -> np.ndarray:
    """Exact bivariate Poisson PMF grid up to max_goals per side."""
    H = int(max_goals) + 1
    A = int(max_goals) + 1
    P = np.zeros((H, A), dtype=np.float64)
    e = math.exp(-(lam1 + lam2 + lam12))
    from math import factorial
    for i in range(H):
        for j in range(A):
            s = 0.0
            m = min(i, j)
            for k in range(m + 1):
                s += (lam1 ** (i - k)) / factorial(i - k) * \
                     (lam2 ** (j - k)) / factorial(j - k) * \
                     (lam12 ** k) / factorial(k)
            P[i, j] = e * s
    S = P.sum()
    if not np.isfinite(S) or S <= 0:
        P[:] = 0.0
        P[0, 0] = 1.0
    else:
        P /= S
    return P

def _bp_grid_rho(lh: float, la: float, rho: float, max_goals: int) -> np.ndarray:
    """Grid with lam12 = rho * sqrt(lh*la)."""
    lh = max(1e-7, float(lh))
    la = max(1e-7, float(la))
    lam12 = float(np.clip(rho, 0.0, DEFAULT_RHO_MAX)) * float(np.sqrt(lh * la))
    lam1 = max(1e-7, lh - lam12)
    lam2 = max(1e-7, la - lam12)
    return _bp_grid_from_components(lam1, lam2, lam12, max_goals)

def _bp_grid_cmin(lh: float, la: float, c: float, max_goals: int) -> np.ndarray:
    """Grid with lam12 = c * min(lh, la)."""
    lh = max(1e-7, float(lh))
    la = max(1e-7, float(la))
    lam12 = max(0.0, float(c)) * float(min(lh, la))
    lam1 = max(1e-7, lh - lam12)
    lam2 = max(1e-7, la - lam12)
    return _bp_grid_from_components(lam1, lam2, lam12, max_goals)

def _one_x_two_from_grid(P: np.ndarray) -> Tuple[float, float, float]:
    """Aggregate a score grid into (pH, pD, pA)."""
    if P.size == 0 or not np.isfinite(P).all():
        return (1/3, 1/3, 1/3)
    H, A = np.indices(P.shape)
    pH = float(P[(H > A)].sum())
    pD = float(np.trace(P))
    pA = float(P[(H < A)].sum())
    s = pH + pD + pA
    if s > 0:
        pH, pD, pA = pH / s, pD / s, pA / s
    return pH, pD, pA

# ----------------------------- draw controls (minute markets) -----------------------------
def _kappa_for(lh: float, la: float) -> float:
    """
    Adaptive shrink factor for the low-score diagonal (0-0, 1-1).
    More shrink when expected goals are small (worst draw bias).
    Returns kappa in [0, 0.10].
    """
    lam = max(1e-6, float(lh + la))
    # Around lam≈2 -> ~0.08, larger for lower totals, smaller for higher totals.
    k = 0.08 * math.exp(-0.35 * (lam - 2.0))
    return float(np.clip(k, 0.0, 0.10))

def _dc_diagonal_fudge(P: np.ndarray, kappa: float) -> np.ndarray:
    """
    Shrink low-score diagonal mass (0-0, 1-1) and softly compensate into (0-1)/(1-0).
    Renormalizes. This curbs draw inflation in bivariate Poisson.
    """
    if P.size == 0 or not np.isfinite(P).all():
        return P
    Q = P.copy()
    if Q.shape[0] > 0 and Q.shape[1] > 0:
        Q[0, 0] *= (1.0 - kappa)
    if Q.shape[0] > 1 and Q.shape[1] > 1:
        Q[1, 1] *= (1.0 - 0.5 * kappa)
        Q[0, 1] *= (1.0 + 0.5 * kappa)
        Q[1, 0] *= (1.0 + 0.5 * kappa)
    s = Q.sum()
    return (Q / s) if s > 0 else P

def _shrink_draw_minute(pH: float, pD: float, pA: float, f: float, alpha0: float) -> Tuple[float, float, float]:
    """
    Heuristic minute-aware draw shrink (no calibrator needed).
    alpha0 is the maximum proportional shrink at T->0 (e.g., 0.06–0.10).
    Tapers linearly to 0 at 90', and preserves the H:A ratio.
    """
    alpha0 = float(np.clip(alpha0, 0.0, 0.20))
    if alpha0 <= 0.0:
        return pH, pD, pA
    shrink = alpha0 * (1.0 - float(np.clip(f, 0.0, 1.0)))
    pD_new = float(np.clip(pD * (1.0 - shrink), 1e-9, 1.0 - 1e-9))

    rem_raw = pH + pA
    if rem_raw <= 0.0:
        x = (1.0 - pD_new) * 0.5
        return x, pD_new, x

    rem_new = 1.0 - pD_new
    scale = rem_new / rem_raw
    pH_new = pH * scale
    pA_new = pA * scale
    s = pH_new + pD_new + pA_new
    if s > 0:
        pH_new, pD_new, pA_new = pH_new / s, pD_new / s, pA_new / s
    return pH_new, pD_new, pA_new

# ----------------------------- artifact helpers -----------------------------
def _load_minute_grid(artifact_path: Optional[str]) -> List[int]:
    """
    Accept both:
      - minutes_1x2 artifacts with 'minute_grid'
      - older band artifacts with 'bands' [{lo,hi}, ...] -> use high edge
    """
    if not artifact_path:
        return list(DEFAULT_MINUTE_GRID)
    try:
        with open(artifact_path, "r") as f:
            art = json.load(f)
    except Exception:
        return list(DEFAULT_MINUTE_GRID)

    if isinstance(art, dict):
        if "minute_grid" in art and isinstance(art["minute_grid"], list):
            try:
                return [int(round(float(x))) for x in art["minute_grid"]]
            except Exception:
                pass
        if "bands" in art and isinstance(art["bands"], list) and len(art["bands"]) > 0:
            grid = []
            for b in art["bands"]:
                hi = b.get("hi")
                if hi is not None:
                    grid.append(int(hi))
            if grid:
                return sorted(set(grid))
    return list(DEFAULT_MINUTE_GRID)

# ----------------------------- command ------------------------------
class Command(BaseCommand):
    help = "Write 1X2-by-minute markets using time-scaled bivariate Poisson (with draw controls)."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--max-goals", type=int, default=DEFAULT_MAX_GOALS_GRID)

        # coupling / artifacts
        parser.add_argument("--rho", type=float, default=DEFAULT_RHO,
                            help="Coupling scale for bp via sqrt(lh*la).")
        parser.add_argument("--artifacts", type=str, default=None,
                            help="Optional goals artifacts.goals.json (to read bp_c and/or minute grid).")
        parser.add_argument("--use-artifacts-c", action="store_true",
                            help="Use lam12 = c * min(lh, la) if artifacts has bp_c.")
        parser.add_argument("--c-taper", type=float, default=DEFAULT_C_TAPER,
                            help="Taper coupling as exp(-c_taper*|lh-la|).")

        # minute grid
        parser.add_argument("--minute-grid", type=str, default=None,
                            help="Cutoffs '5:85:5' or '5,10,15,...'. If omitted, taken from artifact or defaults.")

        # swap safety
        parser.add_argument("--suspect-swap-margin", type=float, default=DEFAULT_SWAP_MARGIN,
                            help="Swap lambdas if p_home(after swap) - p_home(raw) >= this (full-time diagnostic).")
        parser.add_argument("--no-swap-override", action="store_true",
                            help="Disable runtime swap override.")

        # draw controls
        parser.add_argument("--minute-draw-shrink", type=float, default=0.0,
                            help="Max proportional shrink to draw at T→0 (0..0.20). "
                                 "Tapers to 0 by 90' and preserves H/A ratio.")

        # housekeeping / verbosity
        parser.add_argument("--delete-first", action="store_true")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--tune-verbose", action="store_true",
                            help="Print per-minute diagnostics for the first few matches.")

    def handle(self, *args, **opts):
        league_id: int = int(opts["league_id"])
        days: int = int(opts["days"])
        max_goals: int = int(opts["max_goals"])
        rho: float = float(np.clip(opts["rho"], 0.0, DEFAULT_RHO_MAX))
        c_taper: float = float(opts["c_taper"])

        artifact_path = opts.get("artifacts")
        use_artifacts_c = bool(opts.get("use_artifacts_c"))
        suspect_swap_margin: float = float(opts["suspect_swap_margin"])
        swap_override_disabled: bool = bool(opts["no_swap_override"])
        verbose: bool = bool(opts["verbose"])
        tune_verbose: bool = bool(opts["tune_verbose"])
        minute_draw_shrink: float = float(opts.get("minute_draw_shrink", 0.0))

        # minute grid: CLI > artifact > default
        cli_grid = opts.get("minute_grid")
        if cli_grid:
            minute_grid = self._parse_minute_grid(cli_grid)
        else:
            minute_grid = _load_minute_grid(artifact_path)
        minute_grid = sorted({int(x) for x in minute_grid if 1 <= int(x) <= 90})
        if not minute_grid:
            minute_grid = list(DEFAULT_MINUTE_GRID)

        # optionally read bp_c from goals artifacts
        art_c = None
        if artifact_path and use_artifacts_c:
            try:
                with open(artifact_path, "r") as f:
                    art = json.load(f)
                if isinstance(art, dict) and "bp" in art and "c_bp" in art["bp"]:
                    art_c = float(art["bp"]["c_bp"])
                elif isinstance(art, dict) and "bp_c" in art:
                    art_c = float(art["bp_c"])
            except Exception:
                art_c = None

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

        if bool(opts["delete_first"]):
            with transaction.atomic():
                PredictedMarket.objects.filter(
                    league_id=league_id,
                    kickoff_utc__gte=now,
                    kickoff_utc__lte=upto,
                    market_code="1X2_T",
                ).delete()

        def _bp_for(lh: float, la: float, f: float) -> np.ndarray:
            """
            Build a BP grid at time fraction f = T/90 by scaling the full-match
            intensities and the shared component by f (homogeneous rate).
            Taper coupling in mismatches to avoid draw inflation.
            """
            d = abs(lh - la)
            taper = math.exp(-c_taper * d) if c_taper > 0 else 1.0

            if art_c is not None and use_artifacts_c:
                # lam12_full = c * min(lh, la); scale lam12 and marginals by f
                c_eff = art_c * taper
                lam12_full = c_eff * min(lh, la)
                lam12_t = lam12_full * f
                lam1_t = max(1e-7, lh * f - lam12_t)
                lam2_t = max(1e-7, la * f - lam12_t)
                return _bp_grid_from_components(lam1_t, lam2_t, lam12_t, max_goals)
            else:
                # rho model: lam12_t = rho_eff * sqrt(lh*la) * f
                rho_eff = rho * taper
                return _bp_grid_rho(lh * f, la * f, rho_eff, max_goals)

        wrote = 0
        printed = 0

        for mp in qs:
            # base lambdas
            lh_raw = float(np.clip(getattr(mp, "lambda_home", 0.0), 0.05, 6.0))
            la_raw = float(np.clip(getattr(mp, "lambda_away", 0.0), 0.05, 6.0))

            # swap-safety diagnostic at FULL TIME (f=1)
            P_raw_full = _bp_for(lh_raw, la_raw, 1.0)
            pH_raw, _, _ = _one_x_two_from_grid(P_raw_full)

            P_sw_full = _bp_for(la_raw, lh_raw, 1.0)
            pH_swap, _, _ = _one_x_two_from_grid(P_sw_full)
            delta = float(pH_swap - pH_raw)
            swapped = (delta >= suspect_swap_margin)

            use_lh, use_la = lh_raw, la_raw
            if swapped and not swap_override_disabled:
                use_lh, use_la = la_raw, lh_raw

            if verbose:
                m = mp.match
                hn = getattr(m.home, "name", str(m.home_id))
                an = getattr(m.away, "name", str(m.away_id))
                tag = " [SWAPPED]" if swapped and not swap_override_disabled else ""
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an}{tag} | "
                    f"λH={use_lh:.2f} λA={use_la:.2f} | Δ_full={delta:+.3f}"
                )

            # Per-minute loop
            for T in minute_grid:
                f = float(np.clip(T / 90.0, 1e-6, 1.0))
                P_t = _bp_for(use_lh, use_la, f)

                # DC diagonal fudge on the minute grid (use scaled lambdas to size shrink)
                kappa = _kappa_for(use_lh * f, use_la * f)
                P_t = _dc_diagonal_fudge(P_t, kappa)

                # Aggregate to 1X2
                pH, pD, pA = _one_x_two_from_grid(P_t)

                # Optional: heuristic minute draw shrink (keeps H/A ratio), if enabled
                if minute_draw_shrink > 0.0:
                    pH, pD, pA = _shrink_draw_minute(pH, pD, pA, f, alpha0=minute_draw_shrink)

                # Persist H/D/A
                for spec, p in (("H", pH), ("D", pD), ("A", pA)):
                    p = float(np.clip(p, DEFAULT_EPS, 1.0 - DEFAULT_EPS))
                    PredictedMarket.objects.update_or_create(
                        match=mp.match,
                        market_code="1X2_T",
                        specifier=f"T={int(T)}_{spec}",
                        defaults={
                            "league_id": mp.league_id,
                            "kickoff_utc": mp.kickoff_utc,
                            "p_model": p,
                            "fair_odds": float(1.0 / p),
                            "lambda_home": float(use_lh),
                            "lambda_away": float(use_la),
                        },
                    )
                    wrote += 1

                if tune_verbose and printed < 6:
                    printed += 1
                    self.stdout.write(
                        f"  T={int(T):2d} → (H={pH:.3f}, D={pD:.3f}, A={pA:.3f})  "
                        f"[f={f:.3f}, kappa={kappa:.3f}]"
                    )

        self.stdout.write(self.style.SUCCESS(
            f"Wrote/updated {wrote} PredictedMarket rows (1X2_T) for league {league_id}"
        ))

    @staticmethod
    def _parse_minute_grid(arg: str) -> List[int]:
        """Allow 'start:end:step' or comma list."""
        s = str(arg).strip()
        if ":" in s:
            a, b, step = s.split(":")
            a, b, step = float(a), float(b), float(step)
            n = int(round((b - a) / step)) + 1
            return [int(round(a + i * step)) for i in range(max(1, n))]
        return [int(round(float(x))) for x in s.split(",") if x.strip()]
