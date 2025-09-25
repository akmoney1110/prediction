# prediction/matches/management/commands/train_minutes.py
from __future__ import annotations
import json, math, argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.db.models import Q

from matches.models import (
    MLTrainingMatch, Match,
    TeamRating, LeagueSeasonParams,
)

np.seterr(all="ignore")

# ---------- small utils ----------
def _expand_seasons(expr: str) -> List[int]:
    out: List[int] = []
    for seg in str(expr).split(","):
        seg = seg.strip()
        if not seg: continue
        if "-" in seg:
            a,b = seg.split("-",1); a=int(a); b=int(b)
            out.extend(range(min(a,b), max(a,b)+1))
        else:
            out.append(int(seg))
    return sorted(set(out))

def _parse_grid(arg: Optional[str], default: List[float]) -> List[float]:
    if not arg: return default
    s = str(arg).strip()
    if ":" in s:
        a,b,step = s.split(":"); a=float(a); b=float(b); step=float(step)
        n = int(round((b-a)/step))+1
        return [a + i*step for i in range(max(1,n))]
    return [float(x) for x in s.split(",") if x.strip()]

# ---------- (optional) goals artifact ----------
@dataclass
class GoalsArtifact:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    max_goals: int
    c_bp: float

def _load_goals_artifact(path: str) -> GoalsArtifact:
    with open(path, "r") as f:
        art = json.load(f)
    return GoalsArtifact(
        mean=np.array(art["scaler_mean"], float),
        scale=np.array(art["scaler_scale"], float),
        coef=np.array(art["poisson_coef"], float),
        intercept=float(art["poisson_intercept"]),
        max_goals=int(art.get("max_goals", 10)),
        c_bp=float(art.get("bp_c", 0.0)),
    )

# if you have this available; otherwise we’ll skip in ratings mode
try:
    from prediction.train_goals import build_oriented_features  # type: ignore
except Exception:
    build_oriented_features = None

# ---------- data ----------
def _load_rows(leagues: List[int], seasons: List[int]) -> pd.DataFrame:
    qs = (MLTrainingMatch.objects
          .filter(league_id__in=leagues, season__in=seasons)
          .order_by("kickoff_utc")
          .only("fixture_id","league_id","season","kickoff_utc",
                "stats10_json","minute_labels_json"))
    recs = []
    for r in qs.iterator():
        recs.append({
            "fixture_id": int(r.fixture_id),
            "league_id": int(r.league_id),
            "season": int(r.season),
            "kickoff_utc": r.kickoff_utc,
            "stats10_json": r.stats10_json,
            "minute_labels_json": r.minute_labels_json or {},
        })
    df = pd.DataFrame(recs)

    # bring home/away IDs (needed for ratings μ)
    if not df.empty:
        mids = list(df["fixture_id"].values)
        mrows = Match.objects.filter(pk__in=mids).values("id","home_id","away_id")
        mmap = { int(x["id"]): (int(x["home_id"]), int(x["away_id"])) for x in mrows }
        df["home_id"] = df["fixture_id"].map(lambda k: mmap.get(int(k),(None,None))[0])
        df["away_id"] = df["fixture_id"].map(lambda k: mmap.get(int(k),(None,None))[1])
    return df

# ---------- minutes labels helpers ----------
def _first_goal(dfrow: pd.Series) -> Tuple[Optional[int], Optional[str]]:
    mj = dfrow["minute_labels_json"] or {}
    H = [int(x) for x in (mj.get("goal_minutes_home") or []) if 1 <= int(x) <= 120]
    A = [int(x) for x in (mj.get("goal_minutes_away") or []) if 1 <= int(x) <= 120]
    if not H and not A: return None, None
    H = [min(x,90) for x in H]; A=[min(x,90) for x in A]
    ev = [(m,"H") for m in H] + [(m,"A") for m in A]
    ev.sort(key=lambda t: (t[0], 0 if t[1]=="H" else 1))
    return ev[0][0], ev[0][1]

# ---------- hazards (first-goal) ----------
def _estimate_baseline_hazard_first(df: pd.DataFrame, min_events: int = 200) -> np.ndarray:
    TMAX=90; h = np.zeros(TMAX+1, float)
    fmins = df["first_min"].dropna().astype(int).values
    if len(fmins) < min_events:
        # fallback ~ 92% goal by 90’
        lam = -math.log(1-0.92)/TMAX
        h[1:] = lam
        return h
    # discrete Nelson–Aalen-ish with 5-min MA + clipping
    survivors = len(df)
    counts = pd.Series(fmins).value_counts().to_dict()
    raw = np.zeros(TMAX+1, float)
    for t in range(1,TMAX+1):
        at_risk = max(survivors, 1e-9)
        e = float(counts.get(t,0.0))
        raw[t] = e/at_risk
        survivors -= e
    hs = raw.copy()
    for t in range(1,TMAX+1):
        lo,hi = max(1,t-2), min(TMAX,t+2)
        hs[t] = float(raw[lo:hi+1].mean())
    hs = np.clip(hs, 1e-5, 0.25)
    return hs

def _survival_from_h(h: np.ndarray) -> np.ndarray:
    TMAX=90; S = np.ones(TMAX+1,float)
    for t in range(1,TMAX+1):
        S[t] = S[t-1]*(1.0-h[t])
    return S

# ---------- μ sources ----------
def _mu_from_artifact_row(stats10_json, art: GoalsArtifact) -> Tuple[float,float]:
    if build_oriented_features is None:
        raise RuntimeError("build_oriented_features not importable; use --mu-source ratings")
    xh, xa, _ = build_oriented_features({"stats10_json": stats10_json})
    xh = np.asarray(xh, float); xa = np.asarray(xa, float)
    if xh.shape[0] != art.mean.shape[0]:
        raise RuntimeError(
            f"Artifact feature length {art.mean.shape[0]} != current builder {xh.shape[0]} "
            f"(retrain goals artifact or use --mu-source ratings)."
        )
    xs_h = (xh - art.mean)/art.scale
    xs_a = (xa - art.mean)/art.scale
    mu_h = float(np.exp(art.intercept + xs_h.dot(art.coef)))
    mu_a = float(np.exp(art.intercept + xs_a.dot(art.coef)))
    return mu_h, mu_a

def _mu_from_ratings_row(league_id: int, season: int, home_id: int, away_id: int) -> Tuple[float,float]:
    # fetch params (fallbacks to zeros if missing)
    p = (LeagueSeasonParams.objects
         .filter(league_id=league_id, season=season)
         .only("intercept","hfa").first())
    intercept = float(getattr(p,"intercept", 0.0))
    hfa       = float(getattr(p,"hfa",       0.0))

    trh = (TeamRating.objects
           .filter(league_id=league_id, season=season, team_id=home_id)
           .only("attack","defense").first())
    tra = (TeamRating.objects
           .filter(league_id=league_id, season=season, team_id=away_id)
           .only("attack","defense").first())
    ah = float(getattr(trh,"attack", 0.0)); dh = float(getattr(trh,"defense", 0.0))
    aa = float(getattr(tra,"attack", 0.0)); da = float(getattr(tra,"defense", 0.0))

    # µ model: log µ_home = intercept + hfa + a_home - d_away
    mu_h = math.exp(intercept + hfa + ah - da)
    mu_a = math.exp(intercept      + aa - dh)
    # clamp sanity
    mu_h = float(np.clip(mu_h, 0.05, 5.0))
    mu_a = float(np.clip(mu_a, 0.05, 5.0))
    return mu_h, mu_a

# ---------- share model (home vs away scorer) ----------
from sklearn.linear_model import LogisticRegression

def _goal_events(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _,r in df.iterrows():
        mj = r["minute_labels_json"] or {}
        H = [int(x) for x in (mj.get("goal_minutes_home") or []) if 1<=int(x)<=90]
        A = [int(x) for x in (mj.get("goal_minutes_away") or []) if 1<=int(x)<=90]
        for m in H: rows.append({"log_mu_ratio":r["log_mu_ratio"], "mu_tot":r["mu_tot"], "is_2h":1 if m>45 else 0, "y":1})
        for m in A: rows.append({"log_mu_ratio":r["log_mu_ratio"], "mu_tot":r["mu_tot"], "is_2h":1 if m>45 else 0, "y":0})
    return pd.DataFrame(rows)

def _fit_share_logit(df: pd.DataFrame):
    ev = _goal_events(df)
    if ev.empty:
        class Tiny:
            def predict_proba(self, X): 
                p = 0.5*np.ones((len(X),1)); return np.hstack([1-p,p])
        return Tiny()
    X = ev[["log_mu_ratio","mu_tot","is_2h"]].values.astype(float)
    y = ev["y"].astype(int).values
    clf = LogisticRegression(C=2.0, solver="lbfgs", class_weight="balanced", max_iter=300)
    clf.fit(X,y); return clf

def _p_home_half(clf, log_mu_ratio: float, mu_tot: float) -> Tuple[float,float]:
    X1 = np.array([[log_mu_ratio, mu_tot, 0.0]], float)
    X2 = np.array([[log_mu_ratio, mu_tot, 1.0]], float)
    ph1 = float(clf.predict_proba(X1)[:,1][0])
    ph2 = float(clf.predict_proba(X2)[:,1][0])
    return ph1, ph2

# ---------- 2H bump tuning ----------
def _apply_2h_bump(h: np.ndarray, theta: float) -> np.ndarray:
    z = h.copy(); z[46:91] = np.clip(z[46:91]*math.exp(theta), 1e-6, 0.8); return z

def _tune_theta_2h(df: pd.DataFrame, h_base: np.ndarray, thetas: List[float]) -> float:
    # simple likelihood on first-goal times
    best = (1e99, 0.0)
    for th in thetas:
        h = _apply_2h_bump(h_base, th); S = _survival_from_h(h)
        nll=0.0
        for _,r in df.iterrows():
            fm = r["first_min"]
            if pd.isna(fm):
                p = max(1.0 - float(S[90]), 1e-12)
            else:
                t = int(fm); p = max(float((1.0-S[t-1])*h[t]), 1e-12)
            nll -= math.log(p)
        if nll<best[0]: best = (nll, float(th))
    return best[1]

# ---------- command ----------
class Command(BaseCommand):
    help = "Train minute 1X2 primitives using either goals artifact or team ratings for µ."

    def add_arguments(self, parser):
        parser.add_argument("--leagues", type=int, nargs="+", required=True)
        parser.add_argument("--train-seasons", required=True)
        parser.add_argument("--val-seasons",   required=True)
        parser.add_argument("--test-seasons",  required=True)
        parser.add_argument("--outdir",        required=True)

        parser.add_argument("--minute-grid", default="5:85:5",
                            help="Cutoffs e.g. '5:85:5' or comma list '5,10,15,...'")
        parser.add_argument("--max-goals", type=int, default=10)

        # µ source
        parser.add_argument("--mu-source", choices=["artifact","ratings"], default="ratings",
                            help="Where to get µ (expected goals) per side.")
        parser.add_argument("--goals-artifact", default=None,
                            help="Path to goals artifact (required if --mu-source=artifact).")

        # 2H bump
        parser.add_argument("--tune-2h", action="store_true")
        parser.add_argument("--theta-2h", type=float, default=0.0)
        parser.add_argument("--theta-grid", default="-0.35:0.35:0.05",
                            help="Only used with --tune-2h. Example: -0.35:0.35:0.05")

        parser.add_argument("--write-test-csv", action="store_true")
        parser.add_argument("--quiet", action="store_true")

    def handle(self, *args, **opts):
        leagues = list(opts["leagues"])
        train_seasons = _expand_seasons(opts["train_seasons"])
        val_seasons   = _expand_seasons(opts["val_seasons"])
        test_seasons  = _expand_seasons(opts["test_seasons"])
        outdir = opts["outdir"]

        minute_grid = _parse_grid(opts["minute_grid"], [5*i for i in range(1,18)])
        mu_source   = opts["mu_source"]
        goals_artifact_path = opts.get("goals_artifact")

        tune_2h    = bool(opts["tune_2h"])
        theta_2h   = float(opts["theta_2h"])
        theta_grid = _parse_grid(opts.get("theta_grid"), [-0.35 + 0.05*i for i in range(15)])

        if mu_source == "artifact":
            if not goals_artifact_path:
                raise ValueError("--goals-artifact is required when --mu-source=artifact")
            art = _load_goals_artifact(goals_artifact_path)
        else:
            art = None  # ratings mode

        if not opts["quiet"]:
            self.stdout.write("────────────────────────────────────────────────────────────────")
            self.stdout.write("Minute 1X2 Trainer: config summary")
            self.stdout.write(f"• Leagues: {leagues}   Train: {train_seasons}  Val: {val_seasons}  Test: {test_seasons}")
            self.stdout.write(f"• Grid (minutes): {minute_grid}")
            self.stdout.write(f"• µ source: {mu_source}")
            self.stdout.write(f"• 2H bump: {'TUNE' if tune_2h else f'FIX({theta_2h:+.2f})'}")
            self.stdout.write("────────────────────────────────────────────────────────────────")

        # load data
        df_tr_raw = _load_rows(leagues, train_seasons)
        df_va_raw = _load_rows(leagues, val_seasons)
        df_te_raw = _load_rows(leagues, test_seasons)

        # attach first-goal labels
        for df in (df_tr_raw, df_va_raw, df_te_raw):
            if df.empty: continue
            fm, fs = [], []
            for _,r in df.iterrows():
                t, side = _first_goal(r); fm.append(t); fs.append(side)
            df["first_min"] = fm
            df["first_side"] = fs

        # attach µ
        def _attach_mu(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty: return df
            muH=[]; muA=[]
            for _,r in df.iterrows():
                if mu_source == "artifact":
                    mu_h, mu_a = _mu_from_artifact_row(r["stats10_json"], art)  # may raise if dim mismatch
                else:
                    mu_h, mu_a = _mu_from_ratings_row(int(r["league_id"]), int(r["season"]),
                                                      int(r["home_id"]),   int(r["away_id"]))
                muH.append(mu_h); muA.append(mu_a)
            out = df.copy()
            out["mu_home"] = muH
            out["mu_away"] = muA
            out["mu_tot"]  = out["mu_home"] + out["mu_away"]
            eps=1e-9
            out["log_mu_ratio"] = np.log((out["mu_home"]+eps)/(out["mu_away"]+eps))
            return out

        df_tr = _attach_mu(df_tr_raw)
        df_va = _attach_mu(df_va_raw)
        df_te = _attach_mu(df_te_raw)

        # baseline hazard on train
        h0 = _estimate_baseline_hazard_first(df_tr, min_events=200)

        # optional 2H bump tune
        if tune_2h:
            theta_2h = _tune_theta_2h(df_tr, h0, theta_grid)
        h = _apply_2h_bump(h0, theta_2h) if abs(theta_2h)>1e-12 else h0.copy()

        # share model (home scorer prob in 1H/2H as function of log_mu_ratio, mu_tot)
        share_clf = _fit_share_logit(df_tr)

        # save artifact
        art_out = {
            "type": "minutes_1x2_primitives_v1",
            "mu_source": mu_source,
            "goals_artifact_used": bool(mu_source=="artifact"),
            "minute_grid": minute_grid,
            "baseline_hazard": [None] + [float(x) for x in h[1:].tolist()],
            "theta_2h": float(theta_2h),
            "share_model": {
                "coef": [float(x) for x in getattr(share_clf, "coef_", np.array([[0,0,0]])).ravel().tolist()],
                "intercept": float(getattr(share_clf, "intercept_", np.array([0.0]))[0]),
                "features": ["log_mu_ratio","mu_tot","is_2h"],
                "note": "p_H(t) = sigmoid(b0 + b1*log_mu_ratio + b2*mu_tot + b3*1[t>45])",
            },
        }
        import os
        os.makedirs(opts["outdir"], exist_ok=True)
        art_path = os.path.join(opts["outdir"], "artifacts.minutes_1x2.json")
        with open(art_path, "w") as f:
            json.dump(art_out, f, indent=2)
        if not opts["quiet"]:
            self.stdout.write(self.style.SUCCESS(f"Saved minutes artifact -> {art_path}"))

        # optional: write a simple preview CSV (first-goal labels + mu for test)
        if bool(opts["write_test_csv"]) and not df_te.empty:
            prev = df_te[["fixture_id","league_id","season","kickoff_utc",
                          "home_id","away_id","mu_home","mu_away","first_min","first_side"]].copy()
            csv_path = os.path.join(opts["outdir"], "preview_test_minutes_mu.csv")
            prev.to_csv(csv_path, index=False)
            if not opts["quiet"]:
                self.stdout.write(self.style.SUCCESS(f"Wrote test preview -> {csv_path}"))
