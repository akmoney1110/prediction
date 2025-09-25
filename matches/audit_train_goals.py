# matches/audit_train_goals.py
"""
Audit the training dataset used by train_goals:
- Split sizes and goal distributions (H, A, total, 0-0 share)
- Schema presence audit for nested blocks in stats10_json
- Oriented feature extraction via train_goals.build_oriented_features
- Zero-variance columns and high-zero(>95%) columns
- Top-N features by zero rate
- Cosine similarity between oriented H and A vectors (lower is better; << 0.9 ideal)
- Optional JSON report dump

USAGE:
  PYTHONPATH=$(pwd) DJANGO_SETTINGS_MODULE=prediction.settings \
  python matches/audit_train_goals.py \
    --leagues 61 \
    --train-seasons 2020-2023 \
    --val-seasons 2024 \
    --test-seasons 2025 \
    --max-rows 5000 \
    --out audit_report.json
"""

from __future__ import annotations
import os
import json
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Django bootstrap
import django  # noqa: E402
if "DJANGO_SETTINGS_MODULE" not in os.environ:
    raise RuntimeError("DJANGO_SETTINGS_MODULE not set. e.g. export DJANGO_SETTINGS_MODULE=prediction.settings")
django.setup()

from django.db.models import Q  # noqa: E402
from matches.models import MLTrainingMatch  # noqa: E402

# Pull the exact feature builder the trainer uses
try:
    from train_goals import build_oriented_features  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Could not import build_oriented_features from train_goals.py. "
        "Make sure PYTHONPATH includes the project root."
    ) from e

# --------------------------- logging ---------------------------
logger = logging.getLogger("audit_train_goals")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------------- utils ---------------------------
def parse_year_list(s: str) -> List[int]:
    s = str(s).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]

def _as_json(v) -> Dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}

def _safe_float(x, default=0.0) -> float:
    try:
        f = float(x)
        return f if np.isfinite(f) else float(default)
    except Exception:
        return float(default)

@dataclass
class Split:
    name: str
    leagues: List[int]
    seasons: List[int]
    rows: List[Dict[str, Any]]

def _load_split(name: str, leagues: List[int], seasons: List[int], limit: Optional[int]) -> Split:
    qs = (
        MLTrainingMatch.objects
        .filter(league_id__in=leagues, season__in=seasons)
        .filter(~Q(y_home_goals_90=None), ~Q(y_away_goals_90=None))
        .order_by("kickoff_utc")
        .values("league_id","season","kickoff_utc","home_team_id","away_team_id",
                "y_home_goals_90","y_away_goals_90","stats10_json")
    )
    if limit:
        qs = qs[:limit]
    rows = list(qs)
    for r in rows:
        r["stats10_json"] = _as_json(r.get("stats10_json"))
    return Split(name=name, leagues=leagues, seasons=seasons, rows=rows)

def _basic_goal_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"rows": 0, "mean_home": 0.0, "mean_away": 0.0, "mean_total": 0.0, "share_0_0": 0.0}
    H = np.array([_safe_float(r["y_home_goals_90"]) for r in rows], float)
    A = np.array([_safe_float(r["y_away_goals_90"]) for r in rows], float)
    tot = H + A
    share_0_0 = float(np.mean((H == 0) & (A == 0))) if len(H) else 0.0
    return {
        "rows": len(rows),
        "mean_home": float(H.mean()),
        "mean_away": float(A.mean()),
        "mean_total": float(tot.mean()),
        "share_0_0": share_0_0,
    }

def _schema_presence(rows: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    if not rows:
        return []
    keys = [
        ("shots", "shots"),
        ("shots_opp", "shots_opp"),
        ("allowed", "allowed"),
        ("derived", "derived"),
        ("situational", "situational"),
        ("elo", "elo"),
        ("gelo", "gelo"),
    ]
    pres = []
    for label, k in keys:
        cnt = 0
        for r in rows:
            js = _as_json(r.get("stats10_json"))
            v = js.get(k)
            # Treat non-empty dicts as present
            if isinstance(v, dict) and len(v) > 0:
                cnt += 1
        pres.append((label, cnt / float(len(rows))))
    return pres

def _collect_oriented_matrix(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    XH = []
    XA = []
    names_ref: Optional[List[str]] = None
    for r in rows:
        xh, xa, names = build_oriented_features({"stats10_json": _as_json(r.get("stats10_json"))})
        if names_ref is None:
            names_ref = names
        XH.append(xh.astype(float, copy=False))
        XA.append(xa.astype(float, copy=False))
    if not XH:
        return np.zeros((0,)), np.zeros((0,)), []
    XH = np.vstack(XH)
    XA = np.vstack(XA)
    return XH, XA, names_ref or []

def _variance_and_zeroes(X: np.ndarray, names: List[str]) -> Dict[str, Any]:
    if X.size == 0:
        return {"n_features": 0, "zero_var_cols": [], "high_zero_cols": [], "table": []}
    var = X.var(axis=0)
    zero_var_idx = np.where(var < 1e-12)[0].tolist()
    zero_rate = np.mean(np.isclose(X, 0.0), axis=0)
    high_zero_idx = np.where(zero_rate > 0.95)[0].tolist()
    table = []
    for i, n in enumerate(names):
        table.append({"name": n, "zero_rate": float(zero_rate[i]), "variance": float(var[i])})
    # sort by zero_rate desc, variance asc
    table.sort(key=lambda d: (-d["zero_rate"], d["variance"]))
    return {
        "n_features": X.shape[1],
        "zero_var_cols": [names[i] for i in zero_var_idx],
        "high_zero_cols": [names[i] for i in high_zero_idx],
        "table": table,
    }

def _cosine_stats(XH: np.ndarray, XA: np.ndarray) -> Dict[str, Any]:
    if XH.size == 0 or XA.size == 0:
        return {"n": 0, "median": None, "mean": None}
    # compute row-wise cosine between H and A oriented vectors
    num = np.sum(XH * XA, axis=1)
    den = np.linalg.norm(XH, axis=1) * np.linalg.norm(XA, axis=1)
    good = den > 0
    cos = np.clip(num[good] / den[good], -1.0, 1.0)
    return {
        "n": int(good.sum()),
        "median": float(np.median(cos)) if good.any() else None,
        "mean": float(np.mean(cos)) if good.any() else None,
    }

def _print_presence_table(pres: List[Tuple[str, float]]):
    logger.info("Schema audit (nested presence rates on train sample):")
    logger.info("")
    # pretty two-column
    hdr = f"{'block':>10}  {'present_rate':>12}"
    logger.info(hdr)
    for label, rate in pres:
        logger.info(f"{label:>10}  {rate:12.4f}")

def _print_top_zero_rate(table: List[Dict[str, Any]], topn: int = 20):
    logger.info("Top %d by zero_rate:", topn)
    logger.info(f"{'name':>30}  {'zero_rate':>9}  {'variance':>8}")
    for row in table[:topn]:
        logger.info(f"{row['name']:>30}  {row['zero_rate']:9.3f}  {row['variance']:8.6f}")

def audit(leagues: List[int], train_seasons: List[int], val_seasons: List[int], test_seasons: List[int],
          max_rows: Optional[int], out_path: Optional[str]) -> Dict[str, Any]:

    logger.info("Loading splits...")
    sp_train = _load_split("train", leagues, train_seasons, max_rows)
    sp_val   = _load_split("val",   leagues, val_seasons,   max_rows)
    sp_test  = _load_split("test",  leagues, test_seasons,  max_rows)

    # Basic label stats
    for sp in (sp_train, sp_val, sp_test):
        s = _basic_goal_stats(sp.rows)
        logger.info(f"[{sp.name}] rows={s['rows']} | mean goals H={s['mean_home']:.3f} "
                    f"A={s['mean_away']:.3f} TOT={s['mean_total']:.3f} | 0-0 share={s['share_0_0']*100:.1f}%")

    # Schema presence on TRAIN
    pres = _schema_presence(sp_train.rows)
    _print_presence_table(pres)

    # Oriented features on TRAIN (name reference), then VAL+TEST reuse the same names ordering
    XH_tr, XA_tr, names = _collect_oriented_matrix(sp_train.rows)
    if XH_tr.size == 0:
        logger.info("No oriented features could be built on train. Aborting audit.")
        return {}

    X_tr = np.hstack([XH_tr, XA_tr])  # purely to compute zero-rate/variance on combined set if desired
    # Variance & zero rate per feature *using the team_* (home POV) columns*
    var_zero = _variance_and_zeroes(XH_tr, names)
    nfeat = var_zero["n_features"]
    n_zero_var = len(var_zero["zero_var_cols"])
    n_high_zero = len(var_zero["high_zero_cols"])
    logger.info(f"Feature matrix: shape=({XH_tr.shape[0]*2},{nfeat}) | zero-variance-cols={n_zero_var} | high-zero-cols(>95%)={n_high_zero}")
    _print_top_zero_rate(var_zero["table"], topn=20)

    # Cosine(H vs A) on TRAIN
    cos_stats = _cosine_stats(XH_tr, XA_tr)
    logger.info(f"Cosine similarity(H vs A) median={cos_stats['median']:.3f} mean={cos_stats['mean']:.3f} "
                f"n={cos_stats['n']} (lower is better; << 0.9 indicates good orientation)")

    # Also evaluate VAL/TEST quickly for drift
    XH_val, XA_val, _ = _collect_oriented_matrix(sp_val.rows)
    XH_te,  XA_te,  _ = _collect_oriented_matrix(sp_test.rows)
    cos_val = _cosine_stats(XH_val, XA_val) if XH_val.size else {"median": None, "mean": None, "n": 0}
    cos_te  = _cosine_stats(XH_te,  XA_te)  if XH_te.size  else {"median": None, "mean": None, "n": 0}

    # Package report
    report = {
        "splits": {
            "train": _basic_goal_stats(sp_train.rows),
            "val":   _basic_goal_stats(sp_val.rows),
            "test":  _basic_goal_stats(sp_test.rows),
        },
        "schema_presence_train": {k: v for (k, v) in pres},
        "features": {
            "count": nfeat,
            "zero_variance_cols": var_zero["zero_var_cols"],
            "high_zero_cols": var_zero["high_zero_cols"],
            "top_zero_rate": var_zero["table"][:50],
        },
        "cosine": {
            "train": cos_stats,
            "val": cos_val,
            "test": cos_te,
        },
    }

    if out_path:
        try:
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Wrote audit report to {out_path}")
        except Exception as e:
            logger.warning(f"Failed to write report JSON: {e}")

    logger.info("Audit done.")
    return report

def main():
    ap = argparse.ArgumentParser(description="Audit dataset for train_goals features & schema")
    ap.add_argument("--leagues", type=str, required=True, help="e.g. '61' or '39,61,140'")
    ap.add_argument("--train-seasons", type=str, required=True, help="e.g. '2016-2023'")
    ap.add_argument("--val-seasons", type=str, required=True, help="e.g. '2024'")
    ap.add_argument("--test-seasons", type=str, required=True, help="e.g. '2025'")
    ap.add_argument("--max-rows", type=int, default=None, help="optional cap per split")
    ap.add_argument("--out", type=str, default=None, help="optional path to dump JSON report")
    args = ap.parse_args()

    leagues = [int(x) for x in str(args.leagues).split(",") if x.strip()]
    train_seasons = parse_year_list(args.train_seasons)
    val_seasons = parse_year_list(args.val_seasons)
    test_seasons = parse_year_list(args.test_seasons)

    audit(leagues, train_seasons, val_seasons, test_seasons, args.max_rows, args.out)

if __name__ == "__main__":
    main()
