# prediction/matches/management/commands/predict_full_markets.py
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction

from matches.models import (
    MatchPrediction,
    PredictedMarket,
    ModelVersion,
)

# ----------------------------- Tunables / defaults -----------------------------

DEFAULT_MAX_GOALS_GRID = 10
DEFAULT_RHO = 0.10                  # bivariate Poisson shared component scale (sqrt(lh*la))
DEFAULT_RHO_MAX = 0.35
DEFAULT_TOP_CS = 12
DEFAULT_MIN_CS_P = 0.005            # drop ultra-tiny CS probs
DEFAULT_DC_THRESHOLD = 0.55         # DC markets only if no big fav
EPS = 1e-9

# Calibrator sanity thresholds (BTTS)
BTTS_GLOBAL_SPREAD_MIN = 0.10       # est(0.95) - est(0.05) must be at least this, else bypass
BTTS_LOCAL_SPREAD_MIN_SMALL = 0.01  # |f(p+0.05)-f(p-0.05)| must be ≥ this for full weight
BTTS_LOCAL_SPREAD_MIN_BIG   = 0.03  # |f(p+0.15)-f(p-0.15)| must be ≥ this for full weight
BTTS_LOCAL_DELTA_SMALL = 0.05
BTTS_LOCAL_DELTA_BIG   = 0.15

# ----------------------------- numeric helpers -----------------------------

def _pois_cdf_le(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    lam = max(1e-12, float(lam))
    term, s = 1.0, 1.0  # n=0
    for n in range(1, int(k) + 1):
        term *= lam / n
        s += term
    return float(math.exp(-lam) * s)

def _prob_total_over_half(lam_total: float, line: float) -> float:
    need = math.floor(line) + 1
    return float(np.clip(1.0 - _pois_cdf_le(need - 1, lam_total), 0.0, 1.0))

def _prob_team_over_half(lam_side: float, line: float) -> float:
    need = math.floor(line) + 1
    return float(np.clip(1.0 - _pois_cdf_le(need - 1, lam_side), 0.0, 1.0))

# --------------------------- bivariate Poisson grid ---------------------------

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

# ------------------------------- market maths ------------------------------

def _one_x_two_from_grid(P: np.ndarray) -> Tuple[float, float, float]:
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

def _prob_btts_from_grid(P: np.ndarray) -> float:
    H, A = np.indices(P.shape)
    return float(np.clip(P[(H > 0) & (A > 0)].sum(), 0.0, 1.0))

def _apply_calibrator(calibrators: Dict[str, object], key: str, p: float) -> float:
    if p is None or not np.isfinite(p):
        return 0.5
    p = float(np.clip(p, 0.0, 1.0))
    est = (calibrators or {}).get(key)
    if est is None:
        return p
    try:
        out = est.predict(np.array([p], dtype=np.float64))[0]
        return float(np.clip(out, 0.0, 1.0)) if np.isfinite(out) else p
    except Exception:
        return p

# ----- BTTS calibrator: aliasing + safety + blending -----

def _get_calibrator(calibrators: Dict[str, object], key: str) -> Optional[object]:
    if not calibrators:
        return None
    if key in calibrators:
        return calibrators[key]
    aliases = []
    if key.lower() == "btts":
        aliases = ["BTTS", "btts_yes", "both_teams_to_score"]
    for a in aliases:
        if a in calibrators:
            return calibrators[a]
    return None

def _safe_blend_btts_calibrator(
    calibrators: Dict[str, object],
    p_raw: float,
    max_weight: float = 1.0,
    disabled: bool = False,
) -> Tuple[float, float, dict]:
    """
    Returns (p_final, p_cal, meta)
      p_raw -> raw BTTS from grid
      p_cal -> calibrator(p_raw) if available
      p_final -> blended result
      meta -> debug dict: {'w':..., 'uniq':..., 'gspread':..., 'lsmall':..., 'lbig':...}
    """
    p = float(np.clip(p_raw, EPS, 1 - EPS))
    if disabled:
        return p, p, {"w": 0.0, "reason": "disabled"}

    est = _get_calibrator(calibrators, "btts")
    if est is None:
        return p, p, {"w": 0.0, "reason": "no_calibrator"}

    def _pred(x):
        try:
            y = float(est.predict(np.array([x], dtype=np.float64))[0])
            return float(np.clip(y, EPS, 1 - EPS)) if np.isfinite(y) else p
        except Exception:
            return p

    # Global probe across domain – count unique plateaus & total spread
    xs = np.array([0.02, 0.10, 0.20, 0.33, 0.40, 0.50, 0.60, 0.67, 0.80, 0.90, 0.98], dtype=float)
    ys = np.array([_pred(x) for x in xs], dtype=float)
    gspread = float(ys.max() - ys.min())
    uniq = int(np.unique(np.round(ys, 3)).size)  # treat near-equal as same bin

    # If globally flat or too few steps, bypass
    if (gspread < BTTS_GLOBAL_SPREAD_MIN) or (uniq <= 3):
        return p, p, {"w": 0.0, "uniq": uniq, "gspread": gspread, "reason": "global_flat"}

    # Local probes around p – small & big windows
    x1s = float(np.clip(p - BTTS_LOCAL_DELTA_SMALL, EPS, 1 - EPS))
    x2s = float(np.clip(p + BTTS_LOCAL_DELTA_SMALL, EPS, 1 - EPS))
    y1s, y2s = _pred(x1s), _pred(x2s)
    lsmall = float(abs(y2s - y1s))

    x1b = float(np.clip(p - BTTS_LOCAL_DELTA_BIG, EPS, 1 - EPS))
    x2b = float(np.clip(p + BTTS_LOCAL_DELTA_BIG, EPS, 1 - EPS))
    y1b, y2b = _pred(x1b), _pred(x2b)
    lbig = float(abs(y2b - y1b))

    # Translate sensitivities into a weight [0..1]
    # - small window guards the local plateau issue
    # - big window guards coarse but directional calibrators
    def _scale(x, lo, hi):
        # maps x in [lo, hi] to [0,1], clipped
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    w_small = _scale(lsmall, BTTS_LOCAL_SPREAD_MIN_SMALL, 0.06)  # ≥0.06 -> full local confidence
    w_big   = _scale(lbig,   BTTS_LOCAL_SPREAD_MIN_BIG,   0.15)  # ≥0.15 -> full big-window confidence
    w_global= _scale(gspread, BTTS_GLOBAL_SPREAD_MIN,     0.30)  # ≥0.30 -> very informative globally

    # Combine — conservative: take the minimum of the three
    w = min(w_small, w_big, w_global)
    w = float(np.clip(w, 0.0, float(max_weight)))

    p_cal = _pred(p)
    p_out = float(np.clip((1.0 - w) * p + w * p_cal, EPS, 1 - EPS))
    return p_out, p_cal, {"w": w, "uniq": uniq, "gspread": gspread, "lsmall": lsmall, "lbig": lbig}

def _double_chance_from_1x2(pH: float, pD: float, pA: float, threshold: float = DEFAULT_DC_THRESHOLD) -> List[Tuple[str, str, float]]:
    s = pH + pD + pA
    if s <= 0 or not np.isfinite(s):
        return []
    pH, pD, pA = pH / s, pD / s, pA / s
    if max(pH, pD, pA) >= float(threshold):
        return []
    return [
        ("DC", "1X", float(np.clip(pH + pD, EPS, 1 - EPS))),
        ("DC", "12", float(np.clip(pH + pA, EPS, 1 - EPS))),
        ("DC", "X2", float(np.clip(pD + pA, EPS, 1 - EPS))),
    ]

def _correct_score_from_grid(P: np.ndarray, top_n: int = DEFAULT_TOP_CS, min_p: float = DEFAULT_MIN_CS_P) -> List[Tuple[str, str, float]]:
    if P.size == 0 or not np.isfinite(P).all():
        return []
    H, A = P.shape
    flat = [(float(P[i, j]), i, j) for i in range(H) for j in range(A) if P[i, j] >= float(min_p)]
    if not flat:
        ij = int(np.argmax(P))
        i, j = ij // P.shape[1], ij % P.shape[1]
        return [("CS", f"{i}-{j}", float(P[i, j]))]
    flat.sort(reverse=True, key=lambda t: t[0])
    out = []
    for (p, i, j) in flat[: max(1, int(top_n))]:
        out.append(("CS", f"{i}-{j}", float(np.clip(p, EPS, 1.0 - EPS))))
    return out

def _odd_even_from_total(lh: float, la: float) -> Dict[str, float]:
    L = max(1e-9, float(lh + la))
    p_even = 0.5 * (1.0 + math.exp(-2.0 * L))
    p_odd = 1.0 - p_even
    return {"EVEN": float(np.clip(p_even, EPS, 1 - EPS)), "ODD": float(np.clip(p_odd, EPS, 1 - EPS))}

def _goal_in_both_halves(lh: float, la: float, split: float) -> float:
    split = float(np.clip(split, 0.05, 0.95))
    L1 = (lh + la) * split
    L2 = (lh + la) * (1 - split)
    return float(np.clip((1.0 - math.exp(-L1)) * (1.0 - math.exp(-L2)), EPS, 1.0 - EPS))

def _first_last_to_score(lh: float, la: float) -> Dict[str, float]:
    L = max(1e-9, float(lh + la))
    p_ng = math.exp(-L)
    p_h = (lh / L) * (1.0 - p_ng)
    p_a = (la / L) * (1.0 - p_ng)
    return {
        "FIRST_HOME": float(np.clip(p_h, EPS, 1 - EPS)),
        "FIRST_AWAY": float(np.clip(p_a, EPS, 1 - EPS)),
        "FIRST_NONE": float(np.clip(p_ng, EPS, 1 - EPS)),
        "LAST_HOME":  float(np.clip(p_h, EPS, 1 - EPS)),
        "LAST_AWAY":  float(np.clip(p_a, EPS, 1 - EPS)),
        "LAST_NONE":  float(np.clip(p_ng, EPS, 1 - EPS)),
    }

# ----------------------------- monotonicity guards -----------------------------

def _enforce_monotone_ou(rows: List[Tuple[str, str, float]]) -> None:
    ou = [(i, spec, p) for i, (m, spec, p) in enumerate(rows) if m == "OU" and "_over" in spec]
    items = []
    for idx, spec, p in ou:
        try:
            line = float(spec.split("_")[0])
            items.append((line, idx, spec, p))
        except Exception:
            continue
    items.sort(key=lambda t: t[0])  # ascending lines
    max_allowed = 1.0
    for (_, idx, spec, p) in items:
        p_new = min(p, max_allowed)
        p_new = float(np.clip(p_new, EPS, 1 - EPS))
        rows[idx] = ("OU", spec, p_new)
        under_spec = spec.replace("_over", "_under")
        for j, (m2, sp2, p2) in enumerate(rows):
            if m2 == "OU" and sp2 == under_spec:
                rows[j] = ("OU", sp2, float(np.clip(1.0 - p_new, EPS, 1 - EPS)))
                break
        max_allowed = p_new

def _enforce_monotone_team_totals(rows: List[Tuple[str, str, float]]) -> None:
    def _mono(side: str):
        items = []
        for i, (m, spec, p) in enumerate(rows):
            if m == "TEAM_TOTAL" and spec.startswith(f"{side}_o"):
                try:
                    line = float(spec.split("o")[1])
                    items.append((line, i))
                except Exception:
                    pass
        items.sort(key=lambda t: t[0])
        max_allowed = 1.0
        for _, idx in items:
            m, spec, p = rows[idx]
            p_new = float(np.clip(min(p, max_allowed), EPS, 1 - EPS))
            rows[idx] = (m, spec, p_new)
            max_allowed = p_new
    _mono("home")
    _mono("away")

# ----------------------------- calibration loader -----------------------------

def _latest_goals_mv(league_id: int) -> Optional[ModelVersion]:
    return (ModelVersion.objects
            .filter(kind="goals", league_id=league_id)
            .order_by("-trained_until", "-id")
            .first())

def _load_calibrators_for_league(league_id: int) -> Dict[str, object]:
    mv = _latest_goals_mv(league_id)
    if not mv:
        return {}
    calinfo = mv.calibration_json
    try:
        if isinstance(calinfo, str):
            calinfo = json.loads(calinfo or "{}")
    except Exception:
        calinfo = {}
    cal_file = (calinfo or {}).get("file")
    if not cal_file:
        return {}
    try:
        cal = joblib.load(cal_file)
        return cal if isinstance(cal, dict) else {}
    except Exception:
        return {}

# ----------------------------- command class ------------------------------

class Command(BaseCommand):
    help = "Convert MatchPrediction (λ_home, λ_away, etc.) into PredictedMarket rows."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--max-goals", type=int, default=DEFAULT_MAX_GOALS_GRID)

        # grid coupling
        parser.add_argument("--rho", type=float, default=DEFAULT_RHO,
                            help="Coupling for main grid (sqrt(lh*la) scaling).")
        parser.add_argument("--artifacts", type=str, default=None,
                            help="Optional artifacts.goals.json path to read bp_c and max_goals.")
        parser.add_argument("--use-artifacts-c", action="store_true",
                            help="If set, use λ12 = c * min(lh, la) from artifacts instead of rho*sqrt(lh*la).")
        parser.add_argument("--c-taper", type=float, default=0.60,
                            help="Tapers coupling as exp(-c_taper*|lh-la|) to reduce mismatched draw inflation.")

        # markets
        parser.add_argument("--totals-lines", nargs="+", type=float, default=[0.5, 1.5, 2.5, 3.5, 4.5])
        parser.add_argument("--team-totals-lines", nargs="+", type=float, default=[0.5, 1.5, 2.5])
        parser.add_argument("--ah-lines", nargs="+", type=float, default=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])

        # swap safety
        parser.add_argument("--suspect-swap-margin", type=float, default=0.25,
                            help="Swap home/away lambdas for pricing if p_home(after swap) - p_home(raw) >= this.")
        parser.add_argument("--no-swap-override", action="store_true",
                            help="Disable runtime swap override (diagnostics still printed).")

        # misc
        parser.add_argument("--half-split", type=float, default=0.45)
        parser.add_argument("--dc-threshold", type=float, default=DEFAULT_DC_THRESHOLD)
        parser.add_argument("--delete-first", action="store_true")
        parser.add_argument("--verbose-matches", action="store_true")

        # BTTS calibrator control
        parser.add_argument("--disable-btts-calibration", action="store_true",
                            help="Always use raw BTTS (ignore calibrator).")
        parser.add_argument("--btts-calibration-weight", type=float, default=1.0,
                            help="Cap the maximum blending weight (0..1) for the BTTS calibrator.")

    def handle(self, *args, **opts):
        league_id: int = int(opts["league_id"])
        days: int = int(opts["days"])
        max_goals: int = int(opts["max_goals"])
        rho: float = float(np.clip(opts["rho"], 0.0, DEFAULT_RHO_MAX))
        totals_lines = list(opts["totals_lines"])
        team_totals_lines = list(opts["team_totals_lines"])
        ah_lines = list(opts["ah_lines"])
        half_split: float = float(opts["half_split"])
        dc_threshold: float = float(opts["dc_threshold"])
        delete_first: bool = bool(opts["delete_first"])
        verbose_matches: bool = bool(opts["verbose_matches"])
        disable_btts_cal: bool = bool(opts["disable_btts_calibration"])
        btts_max_w: float = float(np.clip(opts["btts_calibration_weight"], 0.0, 1.0))

        artifacts_path = opts.get("artifacts")
        use_artifacts_c = bool(opts.get("use_artifacts_c"))
        suspect_swap_margin: float = float(opts["suspect_swap_margin"])
        swap_override_disabled: bool = bool(opts["no_swap_override"])
        c_taper: float = float(opts["c_taper"])

        now = datetime.now(timezone.utc)
        upto = now + timedelta(days=days)

        # Load goals calibrators
        calibrators = _load_calibrators_for_league(league_id)
        if not calibrators:
            self.stdout.write("Calibration not loaded; continuing with raw probabilities.")

        # Optional: c and max_goals from artifacts
        art_c = None
        if use_artifacts_c and artifacts_path:
            try:
                with open(artifacts_path, "r") as f:
                    art = json.load(f)
                if "bp_c" in art:
                    art_c = float(art["bp_c"])
                if "max_goals" in art and isinstance(art["max_goals"], (int, float)):
                    max_goals = int(art["max_goals"])
            except Exception:
                art_c = None

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
                ).delete()

        def _grid_for(lh: float, la: float) -> np.ndarray:
            # taper coupling in mismatches to avoid draw inflation and extreme tails
            d = abs(lh - la)
            taper = math.exp(-c_taper * d) if c_taper > 0 else 1.0
            if art_c is not None:
                return _bp_grid_cmin(lh, la, art_c * taper, max_goals)
            return _bp_grid_rho(lh, la, rho * taper, max_goals)

        wrote = 0
        for mp in qs:
            # --- Base λs (clamped)
            lh_raw = float(np.clip(getattr(mp, "lambda_home", 0.0), 0.05, 6.0))
            la_raw = float(np.clip(getattr(mp, "lambda_away", 0.0), 0.05, 6.0))

            # --- Swap detection (diagnostic + optional runtime override)
            P_raw = _grid_for(lh_raw, la_raw)
            pH_raw, pD_raw, pA_raw = _one_x_two_from_grid(P_raw)

            P_sw = _grid_for(la_raw, lh_raw)
            pH_swap, _, _ = _one_x_two_from_grid(P_sw)
            delta = float(pH_swap - pH_raw)
            swapped = (delta >= suspect_swap_margin)

            use_lh, use_la = lh_raw, la_raw
            if swapped and not swap_override_disabled:
                use_lh, use_la = la_raw, lh_raw
                P = P_sw
            else:
                P = P_raw

            if verbose_matches:
                m = mp.match
                hn = getattr(m.home, "name", str(m.home_id))
                an = getattr(m.away, "name", str(m.away_id))
                tag = " [SWAPPED]" if swapped and not swap_override_disabled else ""
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an}{tag} | "
                    f"λH={use_lh:.2f} λA={use_la:.2f} | "
                    f"pH_raw={pH_raw:.3f} pH_swap={pH_swap:.3f} Δ={delta:+.3f}"
                )

            # --- Core markets --------------------------------------------
            Hidx, Aidx = np.indices(P.shape)
            total_idx = Hidx + Aidx

            pH, pD, pA = _one_x_two_from_grid(P)

            rows: List[Tuple[str, str, float]] = []
            rows.extend([("1X2", "H", pH), ("1X2", "D", pD), ("1X2", "A", pA)])
            rows.extend(_double_chance_from_1x2(pH, pD, pA, threshold=dc_threshold))

            # ----- BTTS (grid) with SAFE calibration + blending -----
            p_btts_raw = _prob_btts_from_grid(P)
            p_btts_final, p_btts_cal, meta = _safe_blend_btts_calibrator(
                calibrators,
                p_btts_raw,
                max_weight=btts_max_w,
                disabled=disable_btts_cal,
            )
            rows.extend([("BTTS", "yes", p_btts_final), ("BTTS", "no", 1.0 - p_btts_final)])

            # OU totals (grid-based), optional calibrators keyed like over15/25/35...
            for L in sorted(set(totals_lines)):
                k = int(round(L * 10))
                p_over_raw = float(P[total_idx > L].sum())
                p_over = _apply_calibrator(calibrators, f"over{k}", p_over_raw)
                p_over = float(np.clip(p_over, EPS, 1 - EPS))
                rows.append(("OU", f"{L:g}_over", p_over))
                rows.append(("OU", f"{L:g}_under", 1.0 - p_over))

            # Team totals via grid
            for L in sorted(set(team_totals_lines)):
                home_over = float(np.clip(P[Hidx > L].sum(), EPS, 1 - EPS))
                away_over = float(np.clip(P[Aidx > L].sum(), EPS, 1 - EPS))
                rows.append(("TEAM_TOTAL", f"home_o{L:g}", home_over))
                rows.append(("TEAM_TOTAL", f"away_o{L:g}", away_over))

            # Asian Handicap two-way (home/away) for requested lines.
            diff = np.subtract(*np.indices(P.shape))  # H - A

            def _ah_half_or_int(line: float) -> Tuple[float, float]:
                # returns (p_home_line_wins, p_away_line_wins)
                if abs(line - round(line)) < 1e-9:  # integer
                    phh = float(P[diff > -line].sum())
                    paa = float(P[diff < -line].sum())
                else:  # half
                    phh = float(P[diff > -line].sum())
                    paa = 1.0 - phh
                return phh, paa

            def _ah_quarter(line: float) -> Tuple[float, float]:
                lo = math.floor(line * 2.0) / 2.0
                hi = math.ceil(line * 2.0) / 2.0
                ph1, pa1 = _ah_half_or_int(lo)
                ph2, pa2 = _ah_half_or_int(hi)
                return 0.5 * (ph1 + ph2), 0.5 * (pa1 + pa2)

            for L in sorted(set(ah_lines)):
                frac = abs(L - math.floor(L))
                if abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9:
                    phh, paa = _ah_quarter(L)
                else:
                    phh, paa = _ah_half_or_int(L)
                rows.append(("AH", f"home_{L:+g}".replace("+", "+").replace("--", "-"),
                             float(np.clip(phh, EPS, 1 - EPS))))
                rows.append(("AH", f"away_{(-L):+g}".replace("+", "+").replace("--", "-"),
                             float(np.clip(paa, EPS, 1 - EPS))))

            # HT/FT (split λs; independent halves for speed)
            split = float(np.clip(half_split, 0.05, 0.95))
            lh1, la1 = use_lh * split, use_la * split
            lh2, la2 = use_lh * (1 - split), use_la * (1 - split)
            P1 = _bp_grid_rho(lh1, la1, DEFAULT_RHO, 6)
            P2 = _bp_grid_rho(lh2, la2, DEFAULT_RHO, 6)
            htft = {k: 0.0 for k in ["HH","HD","HA","DH","DD","DA","AH","AD","AA"]}
            for i1 in range(P1.shape[0]):
                for j1 in range(P1.shape[1]):
                    r1 = "H" if i1 > j1 else ("D" if i1 == j1 else "A")
                    for i2 in range(P2.shape[0]):
                        for j2 in range(P2.shape[1]):
                            r2 = "H" if (i1 + i2) > (j1 + j2) else ("D" if (i1 + i2) == (j1 + j2) else "A")
                            htft[r1 + r2] += float(P1[i1, j1] * P2[i2, j2])
            s_htft = sum(htft.values())
            if s_htft > 0:
                for k in htft:
                    rows.append(("HTFT", k, float(np.clip(htft[k] / s_htft, EPS, 1 - EPS))))

            # ODD/EVEN
            oe = _odd_even_from_total(use_lh, use_la)
            rows.append(("ODDEVEN", "EVEN", oe["EVEN"]))
            rows.append(("ODDEVEN", "ODD",  oe["ODD"]))

            # Goal in both halves
            p_both_halves = _goal_in_both_halves(use_lh, use_la, split=half_split)
            rows.append(("GOAL_IN_BOTH_HALVES", "yes", p_both_halves))
            rows.append(("GOAL_IN_BOTH_HALVES", "no",  1.0 - p_both_halves))

            # First/Last to score
            fl = _first_last_to_score(use_lh, use_la)
            rows.append(("FIRST_TO_SCORE", "home", fl["FIRST_HOME"]))
            rows.append(("FIRST_TO_SCORE", "away", fl["FIRST_AWAY"]))
            rows.append(("FIRST_TO_SCORE", "none", fl["FIRST_NONE"]))
            rows.append(("LAST_TO_SCORE",  "home", fl["LAST_HOME"]))
            rows.append(("LAST_TO_SCORE",  "away", fl["LAST_AWAY"]))
            rows.append(("LAST_TO_SCORE",  "none", fl["LAST_NONE"]))

            # Correct Score: grid top-N
            rows.extend(_correct_score_from_grid(P, top_n=DEFAULT_TOP_CS, min_p=DEFAULT_MIN_CS_P))

            # Coherence: OU & team totals monotonicity
            _enforce_monotone_ou(rows)
            _enforce_monotone_team_totals(rows)

            # Secondary markets (if λ present) — swap too if we overrode
            lam_ch = getattr(mp, "lambda_corners_home", None)
            lam_ca = getattr(mp, "lambda_corners_away", None)
            if lam_ch is not None and lam_ca is not None:
                lch = float(np.clip(lam_ch, 0.05, 20.0))
                lca = float(np.clip(lam_ca, 0.05, 20.0))
                if swapped and not swap_override_disabled:
                    lch, lca = lca, lch
                lam_ct = lch + lca
                for line in (8.5, 9.5, 10.5):
                    p_over = _prob_total_over_half(lam_ct, line)
                    rows.append(("OU_CORNERS", f"{line}_over", p_over))
                    rows.append(("OU_CORNERS", f"{line}_under", 1.0 - p_over))
                for line in (3.5, 4.5):
                    rows.append(("TEAM_TOTAL_CORNERS", f"home_o{line}", _prob_team_over_half(lch, line)))
                    rows.append(("TEAM_TOTAL_CORNERS", f"away_o{line}", _prob_team_over_half(lca, line)))

            lam_yh = getattr(mp, "lambda_yellows_home", None)
            lam_ya = getattr(mp, "lambda_yellows_away", None)
            if lam_yh is not None and lam_ya is not None:
                lyh = float(np.clip(lam_yh, 0.01, 20.0))
                lya = float(np.clip(lam_ya, 0.01, 20.0))
                if swapped and not swap_override_disabled:
                    lyh, lya = lya, lyh
                lyt = lyh + lya
                for line in (1.5, 2.5, 3.5, 4.5):
                    p_over = _prob_total_over_half(lyt, line)
                    rows.append(("OU_YELLOWS", f"{line}_over", p_over))
                    rows.append(("OU_YELLOWS", f"{line}_under", 1.0 - p_over))
                for line in (0.5, 1.5, 2.5):
                    rows.append(("TEAM_TOTAL_YELLOWS", f"home_o{line}", _prob_team_over_half(lyh, line)))
                    rows.append(("TEAM_TOTAL_YELLOWS", f"away_o{line}", _prob_team_over_half(lya, line)))

            lam_rh = getattr(mp, "lambda_reds_home", None)
            lam_ra = getattr(mp, "lambda_reds_away", None)
            if lam_rh is not None and lam_ra is not None:
                lrh = float(np.clip(lam_rh, 1e-6, 5.0))
                lra = float(np.clip(lam_ra, 1e-6, 5.0))
                if swapped and not swap_override_disabled:
                    lrh, lra = lra, lrh
                lrt = lrh + lra
                for line in (0.5, 1.5):
                    p_over = _prob_total_over_half(lrt, line)
                    rows.append(("OU_REDS", f"{line}_over", p_over))
                    rows.append(("OU_REDS", f"{line}_under", 1.0 - p_over))
                rows.append(("TEAM_TOTAL_REDS", "home_o0.5", _prob_team_over_half(lrh, 0.5)))
                rows.append(("TEAM_TOTAL_REDS", "away_o0.5", _prob_team_over_half(lra, 0.5)))

            # Persist
            for market_code, specifier, p in rows:
                if p is None or not np.isfinite(p):
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
                        "lambda_home": float(use_lh),
                        "lambda_away": float(use_la),
                    },
                )
                wrote += 1

            # Optional verbose per match tail
            if verbose_matches:
                self.stdout.write(
                    f"    1X2=({pH:.3f},{pD:.3f},{pA:.3f}) "
                    f"BTTS(raw={p_btts_raw:.3f}, cal={p_btts_cal:.3f}, w={meta.get('w',0):.2f})"
                )

        self.stdout.write(self.style.SUCCESS(
            f"Wrote/updated {wrote} PredictedMarket rows for league {league_id}"
        ))
