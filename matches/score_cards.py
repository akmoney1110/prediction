#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score YELLOW or RED cards for upcoming (or any) fixtures using a saved artifact.
Reads artifact JSON (mapping, k, rho, calibration), builds goal μ from your goals artifact,
then outputs probs for totals, team totals, and home handicap.

Usage examples:
  export DJANGO_SETTINGS_MODULE=prediction.settings
  python -m matches.score_cards --which yellow --leagues 61 --season 2025 \
    --artifact artifacts/cards_yellow/artifacts.cards.yellow.json \
    --goals-artifact artifacts/goals/artifacts.goals.json \
    --out preds_cards_yellow_live.csv

  # specific fixtures
  python -m matches.score_cards --which red \
    --artifact artifacts/cards_red/artifacts.cards.red.json \
    --goals-artifact artifacts/goals/artifacts.goals.json \
    --fixture-ids 123456,123457 --out preds_cards_red_live.csv
"""
import os, json, argparse, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------- Django bootstrap ----------------
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "prediction.settings"
import django
django.setup()
# --------------------------------------------------

from matches.models import Match, MatchStats  # noqa: F401
try:
    from matches.models import MLTrainingMatch  # may not exist in every project
    HAVE_MLTRAIN = True
except Exception:
    MLTrainingMatch = None  # type: ignore
    HAVE_MLTRAIN = False

# --- goals μ feature builder (import path tolerant)
try:
    from prediction.train_goals import build_oriented_features  # type: ignore
except Exception:
    try:
        from matches.train_goals import build_oriented_features  # type: ignore
    except Exception:
        try:
            from train_goals import build_oriented_features  # type: ignore
        except Exception:
            build_oriented_features = None  # type: ignore

# ===== helpers reused from training (trimmed) =====
def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]

def _to_py(obj):
    import numpy as _np
    if isinstance(obj, (_np.floating, _np.integer)): return obj.item()
    if isinstance(obj, dict):  return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_to_py(v) for v in obj]
    return obj

class GoalsArtifact:
    def __init__(self, art: Dict[str, Any]):
        self.mean = np.array(art["scaler_mean"], float)
        self.scale = np.array(art["scaler_scale"], float)
        self.coef = np.array(art["poisson_coef"], float)
        self.intercept = float(art["poisson_intercept"])

def load_goals_artifact(path: Optional[str]) -> Optional[GoalsArtifact]:
    if not path: return None
    with open(path, "r") as f:
        return GoalsArtifact(json.load(f))

def mu_from_features(x: np.ndarray, art: GoalsArtifact) -> float:
    xs = (x - art.mean) / art.scale
    return float(np.exp(art.intercept + xs.dot(art.coef)))

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

def totals_pmf_copula(pmfH: np.ndarray, pmfA: np.ndarray, rho: float,
                      sims: int, seed: int, nmax: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) & 0x7fffffff)
    Z = rng.multivariate_normal(mean=[0.0, 0.0],
                                cov=[[1.0, rho], [rho, 1.0]],
                                size=int(sims))
    from math import sqrt
    U = 0.5 * (1.0 + (2/np.sqrt(np.pi)) * np.vectorize(np.math.erf)(Z / sqrt(2)))  # Φ via erf
    cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)
    h = np.searchsorted(cdfH, U[:, 0], side="left")
    a = np.searchsorted(cdfA, U[:, 1], side="left")
    h = np.clip(h, 0, nmax); a = np.clip(a, 0, nmax)
    tot = np.clip(h + a, 0, nmax)
    pmfT = np.bincount(tot, minlength=nmax+1).astype(float)
    pmfT /= pmfT.sum()
    return pmfT

def apply_iso_scalar(p: float, curve: Optional[Dict[str, List[float]]]) -> float:
    if not curve: return float(np.clip(p, 0.0, 1.0))
    x = np.array(curve["x"], float); y = np.array(curve["y"], float)
    return float(np.interp(np.clip(p, 0.0, 1.0), x, y))

def predict_means(muH_goals: float, muA_goals: float, mapping: Dict[str, Any],
                  mean_floor: float = 0.5) -> Tuple[float, float]:
    eps = 1e-6
    def _clip(x): return float(np.clip(x, mean_floor, 15.0))
    kind = mapping.get("kind", "heuristic")

    if kind == "glm":
        ch = mapping["home"]; sh = float(mapping["scale"]["home"])
        ca = mapping["away"]; sa = float(mapping["scale"]["away"])
        zH = ch["intercept"] + ch["b1"] * math.log(max(eps, muH_goals)) + ch["b2"] * math.log(max(eps, muA_goals))
        zA = ca["intercept"] + ca["b1"] * math.log(max(eps, muA_goals)) + ca["b2"] * math.log(max(eps, muH_goals))
        return _clip(math.exp(zH) * sh), _clip(math.exp(zA) * sa)

    # heuristic
    p = mapping["params"]
    alpha = float(p["alpha"]); beta = float(p["beta"]); mu_bar = float(p["mu_bar"])
    baseH = float(p["prior_home"]); baseA = float(p["prior_away"])
    denom = max(eps, mu_bar * (1.0 + beta))
    mH = _clip(baseH * ((muH_goals + beta * muA_goals) / denom) ** alpha)
    mA = _clip(baseA * ((muA_goals + beta * muH_goals) / denom) ** alpha)
    return mH, mA

# ===== data pull (no labels needed) =====
def load_fixtures(leagues: List[int], season: int,
                  fixture_ids: Optional[List[int]] = None) -> pd.DataFrame:
    recs = []
    if fixture_ids:
        ids = [int(x) for x in fixture_ids]
        qs = (MLTrainingMatch.objects.filter(fixture_id__in=ids)
              if HAVE_MLTRAIN else
              Match.objects.filter(pk__in=ids))
    else:
        if HAVE_MLTRAIN:
            qs = MLTrainingMatch.objects.filter(league_id__in=leagues, season=season)
        else:
            qs = Match.objects.filter(league_id__in=leagues, season=season)

    qs = qs.only("fixture_id","league_id","season","kickoff_utc","stats10_json") if HAVE_MLTRAIN else \
         qs.only("id","league_id","season","kickoff_utc","raw_result_json")

    for r in qs.iterator():
        if HAVE_MLTRAIN:
            recs.append({
                "fixture_id": int(r.fixture_id),
                "league_id": int(r.league_id),
                "season": int(r.season),
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "stats10_json": getattr(r, "stats10_json", None),
            })
        else:
            recs.append({
                "fixture_id": int(r.id),
                "league_id": int(r.league_id),
                "season": int(r.season),
                "kickoff_utc": r.kickoff_utc.isoformat(),
                "stats10_json": (getattr(r, "raw_result_json", None) or {}).get("stats10_json"),
            })
    return pd.DataFrame(recs)

# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["yellow","red"], required=True, help="which card model to use")
    ap.add_argument("--artifact", required=True, help="path to artifacts.cards.<which>.json")
    ap.add_argument("--goals-artifact", required=True, help="path to artifacts.goals.json")
    ap.add_argument("--leagues", type=int, nargs="+")
    ap.add_argument("--season", type=int)
    ap.add_argument("--fixture-ids", type=str, default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=50000)
    args = ap.parse_args()

    # read artifact
    with open(args.artifact, "r") as f:
        art = json.load(f)

    totals_lines = art["lines"]["totals"]
    team_lines   = art["lines"]["team"]
    hcp_lines    = art["lines"]["handicap"]
    nmax         = int(art["nmax"])
    rho          = float(art.get("rho", 0.0))
    mapping      = art["mapping"]
    mean_floor   = float(art.get("mean_floor", 0.5))
    kH_global    = float(art["dispersion"]["k_home_global"])
    kA_global    = float(art["dispersion"]["k_away_global"])
    kH_map = {tuple(map(int, k.split("-"))): float(v) for k, v in art["dispersion"]["k_home_map"].items()}
    kA_map = {tuple(map(int, k.split("-"))): float(v) for k, v in art["dispersion"]["k_away_map"].items()}
    cal_totals   = art.get("totals_calibration", {})
    cal_team     = art.get("team_calibration", {"home": {}, "away": {}})

    # pull fixtures
    fix_ids = [int(x) for x in args.fixture_ids.split(",") if x.strip().isdigit()]
    if fix_ids:
        df = load_fixtures([], season=0, fixture_ids=fix_ids)
    else:
        if not args.leagues or not args.season:
            raise SystemExit("Provide --fixture-ids or both --leagues and --season")
        df = load_fixtures(args.leagues, args.season)

    if df.empty:
        print("[INFO] No fixtures found.")
        pd.DataFrame().to_csv(args.out, index=False)
        return

    # attach goals μ
    gart = load_goals_artifact(args.goals_artifact)
    muH, muA = [], []
    fails = 0
    for _, r in df.iterrows():
        if gart is None or build_oriented_features is None:
            muH.append(np.nan); muA.append(np.nan); continue
        try:
            xh, xa, _ = build_oriented_features({"stats10_json": r.get("stats10_json")})
            muH.append(mu_from_features(xh, gart))
            muA.append(mu_from_features(xa, gart))
        except Exception:
            muH.append(np.nan); muA.append(np.nan); fails += 1
    if fails: print(f"[WARN] goals μ features failed on {fails} rows (set NaNs).")
    df["mu_goals_home"] = muH; df["mu_goals_away"] = muA

    # predict
    rows = []
    for _, r in df.iterrows():
        key = (int(r["league_id"]), int(r["season"])) if "season" in r else None
        kH = kH_map.get(key, kH_global) if key else kH_global
        kA = kA_map.get(key, kA_global) if key else kA_global

        mH, mA = predict_means(float(r["mu_goals_home"]), float(r["mu_goals_away"]), mapping, mean_floor=mean_floor)
        pmfH = nb_pmf_vec(mH, kH, nmax)
        pmfA = nb_pmf_vec(mA, kA, nmax)
        tot = conv_sum(pmfH, pmfA, nmax) if abs(rho) < 1e-9 else totals_pmf_copula(pmfH, pmfA, rho=rho, sims=args.sims, seed=int(r["fixture_id"]), nmax=nmax)

        out = {
            "fixture_id": int(r["fixture_id"]),
            "league_id": int(r["league_id"]),
            "season": int(r.get("season", 0)),
            "kickoff_utc": r.get("kickoff_utc"),
            "m_cards_home": float(mH),
            "m_cards_away": float(mA),
        }

        # totals
        for L in totals_lines:
            thr = math.floor(float(L) + 1e-9) + 1
            p_over = float(tot[thr:].sum())
            p_over = apply_iso_scalar(p_over, cal_totals.get(str(L)))
            out[f"totals_over_{L}"] = p_over
            out[f"totals_under_{L}"] = float(1.0 - p_over)

        # team lines
        cdfH = np.cumsum(pmfH); cdfA = np.cumsum(pmfA)
        for L in team_lines:
            thr = math.floor(float(L) + 1e-9) + 1
            iH = min(thr-1, nmax); iA = min(thr-1, nmax)
            pH_over = float(1.0 - cdfH[iH]); pA_over = float(1.0 - cdfA[iA])
            pH_over = apply_iso_scalar(pH_over, cal_team.get("home", {}).get(str(L)))
            pA_over = apply_iso_scalar(pA_over, cal_team.get("away", {}).get(str(L)))
            out[f"home_over_{L}"] = pH_over; out[f"home_under_{L}"] = float(1.0 - pH_over)
            out[f"away_over_{L}"] = pA_over; out[f"away_under_{L}"] = float(1.0 - pA_over)

        # simple handicaps only for integer/half-integer around zero (match training)
        for h in hcp_lines:
            # simulate diff by convolution if rho=0; exact push handling for integers
            dmin, dmax = -nmax, nmax
            # build diff distribution by brute force:
            D = np.zeros(dmax - dmin + 1, float)
            for i, ph in enumerate(pmfH):
                if ph == 0: continue
                for j, pa in enumerate(pmfA):
                    if pa == 0: continue
                    d = i - j
                    if dmin <= d <= dmax:
                        D[d - dmin] += ph * pa
            D /= D.sum()
            if float(h).is_integer():
                h_int = int(h)
                win  = float(D[(h_int+1 - dmin):].sum()) if (h_int+1 - dmin) < len(D) else 0.0
                push = float(D[(h_int - dmin)]) if (dmin <= h_int <= dmax) else 0.0
                lose = float(1.0 - win - push)
            else:
                thr = math.floor(float(h) + 1e-9)
                start = (thr+1 - dmin)
                win  = float(D[start:].sum()) if (start < len(D)) else 0.0
                push = 0.0
                lose = float(1.0 - win)
            out[f"hcp_home_{h}_win"]  = win
            out[f"hcp_home_{h}_push"] = push
            out[f"hcp_home_{h}_lose"] = lose

        rows.append(out)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[SCORED] wrote {len(rows)} rows → {args.out}")

if __name__ == "__main__":
    main()
