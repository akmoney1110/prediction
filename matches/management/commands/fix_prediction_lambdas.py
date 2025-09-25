# matches/management/commands/fix_swapped_lambdas.py
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction

from matches.models import MatchPrediction, PredictedMarket


EPS = 1e-9
DEFAULT_RHO = 0.10       # for swap detection grid
DEFAULT_RHO_MAX = 0.35


def _bp_grid_from_components(lam1: float, lam2: float, lam12: float, max_goals: int) -> np.ndarray:
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
    lh = max(1e-7, float(lh))
    la = max(1e-7, float(la))
    lam12 = float(np.clip(rho, 0.0, DEFAULT_RHO_MAX)) * float(np.sqrt(lh * la))
    lam1 = max(1e-7, lh - lam12)
    lam2 = max(1e-7, la - lam12)
    return _bp_grid_from_components(lam1, lam2, lam12, max_goals)


def _bp_grid_cmin(lh: float, la: float, c: float, max_goals: int) -> np.ndarray:
    lh = max(1e-7, float(lh))
    la = max(1e-7, float(la))
    lam12 = max(0.0, float(c)) * float(min(lh, la))
    lam1 = max(1e-7, lh - lam12)
    lam2 = max(1e-7, la - lam12)
    return _bp_grid_from_components(lam1, lam2, lam12, max_goals)


def _one_x_two_from_grid(P: np.ndarray):
    H, A = np.indices(P.shape)
    pH = float(P[(H > A)].sum())
    pD = float(np.trace(P))
    pA = float(P[(H < A)].sum())
    s = pH + pD + pA
    if s > 0:
        pH, pD, pA = pH / s, pD / s, pA / s
    return pH, pD, pA


class Command(BaseCommand):
    help = "Detect and FIX MatchPrediction rows where home/away lambdas look swapped, by writing corrected values."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=14)
        parser.add_argument("--margin", type=float, default=0.25,
                            help="Swap if p_home(after swap) - p_home(raw) >= margin.")
        parser.add_argument("--max-goals", type=int, default=10)
        parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
        parser.add_argument("--artifacts", type=str, default=None,
                            help="Optional JSON with {'bp_c': float, 'max_goals': int}.")
        parser.add_argument("--use-artifacts-c", action="store_true",
                            help="Use c*min(lh,la) coupling instead of rho*sqrt(lh*la).")
        parser.add_argument("--c-taper", type=float, default=0.60,
                            help="exp(-c_taper * |lh-la|) taper on coupling.")
        parser.add_argument("--ids", nargs="+", type=int, default=None,
                            help="Limit to specific MatchPrediction IDs.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Only print what would change; do not write.")
        parser.add_argument("--wipe-markets", action="store_true",
                            help="Delete PredictedMarket for affected matches (forces clean reprice).")

    def handle(self, *args, **opts):
        league_id: int = int(opts["league_id"])
        days: int = int(opts["days"])
        max_goals: int = int(opts["max_goals"])
        rho: float = float(np.clip(opts["rho"], 0.0, DEFAULT_RHO_MAX))
        margin: float = float(opts["margin"])
        ids = opts.get("ids")
        dry = bool(opts["dry_run"])
        wipe = bool(opts["wipe_markets"])
        c_taper: float = float(opts["c_taper"])
        use_artifacts_c: bool = bool(opts["use_artifacts_c"])
        art_path: Optional[str] = opts.get("artifacts")

        art_c = None
        if use_artifacts_c and art_path:
            try:
                with open(art_path, "r") as f:
                    art = json.load(f)
                if "bp_c" in art:
                    art_c = float(art["bp_c"])
                if "max_goals" in art and isinstance(art["max_goals"], (int, float)):
                    max_goals = int(art["max_goals"])
            except Exception:
                art_c = None

        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)

        qs = (MatchPrediction.objects
              .filter(league_id=league_id,
                      kickoff_utc__gte=now,
                      kickoff_utc__lte=upto,
                      match__status__in=["NS", "PST", "TBD"])
              .select_related("match")
              .order_by("kickoff_utc"))

        if ids:
            qs = qs.filter(id__in=ids)

        def _grid(lh: float, la: float) -> np.ndarray:
            d = abs(lh - la)
            taper = math.exp(-c_taper * d) if c_taper > 0 else 1.0
            if art_c is not None:
                return _bp_grid_cmin(lh, la, art_c * taper, max_goals)
            return _bp_grid_rho(lh, la, rho * taper, max_goals)

        to_fix = []
        for mp in qs:
            lh = float(np.clip(getattr(mp, "lambda_home", 0.0), 0.05, 6.0))
            la = float(np.clip(getattr(mp, "lambda_away", 0.0), 0.05, 6.0))

            P_raw  = _grid(lh, la)
            P_swap = _grid(la, lh)
            pH_raw, _, _ = _one_x_two_from_grid(P_raw)
            pH_sw,  _, _ = _one_x_two_from_grid(P_swap)
            delta = float(pH_sw - pH_raw)

            if delta >= margin:
                to_fix.append((mp, lh, la, pH_raw, pH_sw, delta))

        if not to_fix:
            self.stdout.write("No suspicious rows found.")
            return

        self.stdout.write(f"Found {len(to_fix)} suspicious rows (margin ≥ {margin:.2f}).")
        for mp, lh, la, pH_raw, pH_sw, delta in to_fix:
            m = mp.match
            hn = getattr(m.home, "name", str(m.home_id))
            an = getattr(m.away, "name", str(m.away_id))
            self.stdout.write(
                f"{mp.id} | {hn} vs {an} | λH={lh:.2f} λA={la:.2f} | "
                f"pH_raw={pH_raw:.3f} pH_swap={pH_sw:.3f} Δ={delta:+.3f}"
            )

        if dry:
            self.stdout.write("Dry-run: no changes written.")
            return

        with transaction.atomic():
            for mp, lh, la, *_ in to_fix:
                # swap main goal lambdas
                mp.lambda_home, mp.lambda_away = mp.lambda_away, mp.lambda_home

                # swap ancillary lambdas if present
                if mp.lambda_corners_home is not None and mp.lambda_corners_away is not None:
                    mp.lambda_corners_home, mp.lambda_corners_away = mp.lambda_corners_away, mp.lambda_corners_home
                if mp.lambda_yellows_home is not None and mp.lambda_yellows_away is not None:
                    mp.lambda_yellows_home, mp.lambda_yellows_away = mp.lambda_yellows_away, mp.lambda_yellows_home
                if mp.lambda_cards_home is not None and mp.lambda_cards_away is not None:
                    mp.lambda_reds_home, mp.lambda_reds_away = mp.lambda_reds_away, mp.lambda_reds_home

                mp.save()

                if wipe:
                    PredictedMarket.objects.filter(match=mp.match).delete()

        self.stdout.write(self.style.SUCCESS(
            f"Fixed {len(to_fix)} MatchPrediction rows"
            + (" and wiped related PredictedMarket rows" if wipe else "")
        ))
