# -*- coding: utf-8 -*-
from __future__ import annotations

import csv, json, math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from django.core.management.base import BaseCommand, CommandParser
from matches.models import MatchPrediction

EPS = 1e-9
DEFAULT_MAX_GOALS = 10
DEFAULT_RHO = 0.10
DEFAULT_RHO_MAX = 0.35

# ---------- BP helpers ----------
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

def _grid_cmin(lh: float, la: float, c: float, max_goals: int, taper: float) -> np.ndarray:
    lh = max(1e-7, float(lh)); la = max(1e-7, float(la))
    lam12 = max(0.0, float(c*taper)) * float(min(lh, la))
    lam1 = max(1e-7, lh - lam12); lam2 = max(1e-7, la - lam12)
    return _bp_grid_from_components(lam1, lam2, lam12, max_goals)

def _one_x_two(P: np.ndarray) -> Tuple[float,float,float]:
    H, A = np.indices(P.shape)
    pH = float(P[(H > A)].sum())
    pD = float(np.trace(P))
    pA = float(P[(H < A)].sum())
    s = pH + pD + pA
    if s > 0: pH, pD, pA = pH/s, pD/s, pA/s
    return pH, pD, pA

# ---------- artifact & features ----------
@dataclass
class GoalsArtifact:
    feature_names: Optional[List[str]]
    kept_feature_idx: Optional[List[int]]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    bp_c: float
    max_goals: int
    cal_1x2: Optional[Dict[str, Dict[str, list]]]

def _load_art(path: str) -> GoalsArtifact:
    with open(path, "r") as f:
        art = json.load(f)
    cal = art.get("onextwo_cal") or (art.get("calibration", {}) or {}).get("onextwo")
    return GoalsArtifact(
        feature_names=art.get("feature_names"),
        kept_feature_idx=art.get("kept_feature_idx"),
        scaler_mean=np.array(art["scaler_mean"], float),
        scaler_scale=np.array(art["scaler_scale"], float),
        coef=np.array(art["poisson_coef"], float),
        intercept=float(art["poisson_intercept"]),
        bp_c=float(art.get("bp_c", 0.0)),
        max_goals=int(art.get("max_goals", DEFAULT_MAX_GOALS)),
        cal_1x2=cal if isinstance(cal, dict) else None,
    )

try:
    from prediction.train_goals import build_oriented_features  # type: ignore
except Exception:
    try:
        from train_goals import build_oriented_features  # type: ignore
    except Exception:
        build_oriented_features = None  # type: ignore

def _get_stats10(m) -> Any:
    js = getattr(m, "raw_result_json", None)
    if isinstance(js, dict) and "stats10_json" in js:
        return js["stats10_json"]
    return getattr(m, "stats10_json", None)

def _interp_iso(p: float, curve: Optional[Dict[str, list]]) -> float:
    if not curve: return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve.get("x") or curve.get("X_thresholds_") or curve.get("X_"), float)
    y = np.array(curve.get("y") or curve.get("y_thresholds_") or curve.get("y_"), float)
    if x.size == 0 or y.size == 0: return float(np.clip(p, 0.0, 1.0))
    return float(np.interp(float(np.clip(p, 0, 1)), x, y))

def _vector_from_stats_by_names(stats: Any, target_names: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    if build_oriented_features is None: return None
    try:
        xh, xa, names = build_oriented_features({"stats10_json": stats})
        name2idx = {n:i for i,n in enumerate(names)}
        def make_vec(xside):
            vals = []
            missing = []
            for n in target_names:
                j = name2idx.get(n, None)
                if j is None:
                    vals.append(0.0); missing.append(n)
                else:
                    vals.append(float(xside[j]))
            return np.array(vals, float), missing
        vh, miss_h = make_vec(xh)
        va, miss_a = make_vec(xa)
        missing = sorted(set(miss_h + miss_a))
        return vh, va, missing
    except Exception:
        return None

def _vector_from_stats_by_idx(stats: Any, idx: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if build_oriented_features is None: return None
    try:
        xh, xa, _ = build_oriented_features({"stats10_json": stats})
        return np.array(xh, float)[idx], np.array(xa, float)[idx]
    except Exception:
        return None

def _mus_from_art(stats: Any, art: GoalsArtifact) -> Tuple[Optional[float], Optional[float], str, str]:
    """
    returns: (mu_h, mu_a, source, detail)
    """
    # Attempt alignment by names (most robust)
    if art.feature_names:
        out = _vector_from_stats_by_names(stats, art.feature_names)
        if out is not None:
            vh, va, missing = out
            xs_h = (vh - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
            xs_a = (va - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
            mu_h = float(np.clip(math.exp(art.intercept + xs_h.dot(art.coef)), 1e-6, 8.0))
            mu_a = float(np.clip(math.exp(art.intercept + xs_a.dot(art.coef)), 1e-6, 8.0))
            detail = "names_align:ok" if not missing else f"names_align:missing={len(missing)}"
            return mu_h, mu_a, "artifact_names", detail

    # Fallback: kept_feature_idx
    if art.kept_feature_idx:
        out = _vector_from_stats_by_idx(stats, art.kept_feature_idx)
        if out is not None:
            vh, va = out
            if vh.shape[0] == art.scaler_mean.shape[0]:
                xs_h = (vh - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
                xs_a = (va - art.scaler_mean) / np.where(art.scaler_scale==0, 1.0, art.scaler_scale)
                mu_h = float(np.clip(math.exp(art.intercept + xs_h.dot(art.coef)), 1e-6, 8.0))
                mu_a = float(np.clip(math.exp(art.intercept + xs_a.dot(art.coef)), 1e-6, 8.0))
                return mu_h, mu_a, "artifact_idx", "idx_align:ok"

    return None, None, "artifact_failed", "builder_or_align_failed"

# ---------- Command ----------
class Command(BaseCommand):
    help = "Audit pipeline inputs & outputs for 1X2/DC generation (DB lambdas vs artifact μ, feature alignment, fallbacks, swaps)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--league-id", type=int, required=True)
        parser.add_argument("--days", type=int, default=10)
        parser.add_argument("--artifact", type=str, required=True)
        parser.add_argument("--c-taper", type=float, default=0.60)
        parser.add_argument("--out", type=str, required=True, help="Path to write CSV audit.")
        parser.add_argument("--verbose", action="store_true")

    def handle(self, *args, **opts):
        lg = int(opts["league_id"]); days = int(opts["days"])
        art = _load_art(str(opts["artifact"]))
        c_taper = float(opts["c_taper"])
        out_path = str(opts["out"])
        verbose = bool(opts["verbose"])

        now = datetime.now(timezone.utc); upto = now + timedelta(days=days)
        qs = (MatchPrediction.objects
              .filter(league_id=lg,
                      kickoff_utc__gte=now,
                      kickoff_utc__lte=upto,
                      match__status__in=["NS","PST","TBD"])
              .select_related("match")
              .order_by("kickoff_utc"))

        rows = []
        if not qs.exists():
            self.stdout.write("No upcoming MatchPrediction rows.")
        for mp in qs:
            m = mp.match
            hn = getattr(m.home, "name", str(getattr(m, "home_id", "?")))
            an = getattr(m.away, "name", str(getattr(m, "away_id", "?")))
            s10 = _get_stats10(m)

            # DB lambdas
            lH_db = mp.lambda_home; lA_db = mp.lambda_away
            have_db = lH_db is not None and lA_db is not None
            lH_db = float(lH_db) if have_db else None
            lA_db = float(lA_db) if have_db else None

            # Artifact μ
            muH_art = muA_art = None
            art_src = art_det = ""
            if s10 is not None:
                muH_art, muA_art, art_src, art_det = _mus_from_art(s10, art)

            # Chosen μ (prefer DB if present; else artifact; else intercept)
            reason = ""
            if have_db:
                muH, muA = lH_db, lA_db; reason = "DB"
            elif (muH_art is not None and muA_art is not None):
                muH, muA = muH_art, muA_art; reason = art_src
            else:
                mu0 = float(np.clip(math.exp(art.intercept), 0.6, 2.4))
                muH, muA = mu0, mu0; reason = "intercept_fallback"

            # Coupling & grid
            d = abs(muH - muA)
            taper = math.exp(-c_taper * d) if c_taper > 0 else 1.0
            P = _grid_cmin(muH, muA, c=art.bp_c, max_goals=art.max_goals, taper=taper)
            pH, pD, pA = _one_x_two(P)

            # Swap diagnostic
            Psw = _grid_cmin(muA, muH, c=art.bp_c, max_goals=art.max_goals, taper=taper)
            pH_sw, _, _ = _one_x_two(Psw)
            delta = float(pH_sw - pH)

            rows.append({
                "mp_id": mp.id,
                "kickoff_utc": mp.kickoff_utc.isoformat(),
                "home": hn, "away": an,
                "db_lh": lH_db, "db_la": lA_db,
                "stats10_present": s10 is not None,
                "mu_art_h": muH_art, "mu_art_a": muA_art,
                "art_source": art_src, "art_detail": art_det,
                "mu_used_h": muH, "mu_used_a": muA,
                "mu_reason": reason,
                "c_taper": taper, "bp_c": art.bp_c, "max_goals": art.max_goals,
                "p_home": pH, "p_draw": pD, "p_away": pA,
                "dc_1x": pH + pD, "dc_12": pH + pA, "dc_x2": pD + pA,
                "swap_delta_home": delta
            })

            if verbose:
                self.stdout.write(
                    f"{mp.id} | {hn} vs {an} | reason={reason}"
                    + (f" | art={art_src}({art_det})" if art_src else "")
                    + f" | μ=({muH:.2f},{muA:.2f}) | 1X2=({pH:.3f},{pD:.3f},{pA:.3f})"
                )

        # Write CSV
        fieldnames = list(rows[0].keys()) if rows else []
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # Quick summary
        n = len(rows)
        by_reason = {}
        for r in rows:
            by_reason[r["mu_reason"]] = by_reason.get(r["mu_reason"], 0) + 1
        self.stdout.write(f"Wrote {n} rows → {out_path}")
        for k,v in sorted(by_reason.items(), key=lambda t: (-t[1], t[0])):
            self.stdout.write(f"  {k}: {v}")
