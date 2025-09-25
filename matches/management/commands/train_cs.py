from __future__ import annotations

"""
Train a Correct Score (CS) multinomial model for a league.

Key features:
- Time-ordered tail holdout for validation (val_frac)
- Optional time-ordered K-fold CV metrics (--folds)
- Stable softmax predictions (log-sum-exp) to avoid overflow NaNs
- Gentle ridge fallback on MNLogit to keep coefficients sane
- Winsorize (+) z-scale features with clipping
- Compact class mapping: only classes seen in TRAIN are modeled
- Optional per-class isotonic calibration
- Registers a ModelVersion(kind="market:CS") with paths + metrics
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logsumexp
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone as dj_tz

from matches.models import MLTrainingMatch, ModelVersion


# -------------------- config --------------------

@dataclass
class TrainConfig:
    league_id: int
    seasons: Optional[List[int]]
    dt_start: Optional[datetime]
    dt_end: datetime
    cap: int = 5                 # cap scorelines at [0..cap] per side
    val_frac: float = 0.15       # tail holdout fraction (time-ordered)
    outdir: Path = Path("models/cs")
    calibrate: bool = True
    folds: int = 0               # time-ordered K-fold CV for metrics (0/1 = off)


# -------------------- labels --------------------

def label_grid(cap: int) -> List[str]:
    return [f"{i}-{j}" for i in range(cap + 1) for j in range(cap + 1)]

def make_label(h: Optional[int], a: Optional[int], cap: int) -> Optional[str]:
    if h is None or a is None:
        return None
    try:
        hh = int(h); aa = int(a)
    except Exception:
        return None
    hh = max(0, min(cap, hh))
    aa = max(0, min(cap, aa))
    return f"{hh}-{aa}"


# -------------------- features --------------------

CS_FEATURES: List[str] = [
    # home last-10
    "h_gf10","h_ga10","h_gd10","h_sot10","h_conv10","h_sot_pct10","h_poss10",
    "h_corners_for10","h_cards_for10","h_clean_sheets10",
    # away last-10
    "a_gf10","a_ga10","a_gd10","a_sot10","a_conv10","a_sot_pct10","a_poss10",
    "a_corners_for10","a_cards_for10","a_clean_sheets10",
    # venue & situational
    "h_home_gf10","a_away_gf10",
    "h_rest_days","a_rest_days","h_matches_14d","a_matches_14d",
    # deltas + robustness flags
    "d_gf10","d_sot10","d_rest_days","h_stats_missing","a_stats_missing",
]

FETCH_FIELDS = ["fixture_id", "kickoff_utc", "season", "y_home_goals_90", "y_away_goals_90"] + CS_FEATURES


def winsorize_df(df: pd.DataFrame, cols: List[str], lo: float = 0.005, hi: float = 0.995) -> pd.DataFrame:
    """Clip extreme tails to reduce the impact of outliers before scaling."""
    if not cols:
        return df
    loq = df[cols].quantile(lo)
    hiq = df[cols].quantile(hi)
    out = df.copy()
    for c in cols:
        out[c] = out[c].clip(loq[c], hiq[c])
    return out


def fetch_df(cfg: TrainConfig) -> pd.DataFrame:
    qs = (MLTrainingMatch.objects
          .filter(league_id=cfg.league_id)
          .exclude(y_home_goals_90__isnull=True)
          .exclude(y_away_goals_90__isnull=True))

    if cfg.seasons:
        qs = qs.filter(season__in=cfg.seasons)
    else:
        if cfg.dt_start:
            qs = qs.filter(kickoff_utc__gte=cfg.dt_start)
        if cfg.dt_end:
            qs = qs.filter(kickoff_utc__lte=cfg.dt_end)

    rows = list(qs.values(*FETCH_FIELDS))
    if not rows:
        return pd.DataFrame(columns=FETCH_FIELDS)

    df = pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)

    # Label
    df["label"] = [
        make_label(h, a, cfg.cap)
        for h, a in zip(df["y_home_goals_90"], df["y_away_goals_90"])
    ]
    df = df.dropna(subset=["label", "kickoff_utc"]).reset_index(drop=True)

    # Numeric cleanup
    for c in CS_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median(skipna=True))

    # booleans -> float
    for c in ["h_stats_missing", "a_stats_missing"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    # Winsorize features to reduce extreme tails
    df = winsorize_df(df, CS_FEATURES, lo=0.005, hi=0.995)

    return df


# -------------------- scaling --------------------

@dataclass
class ZScaler:
    mean_: np.ndarray
    std_: np.ndarray
    clip_: float = 6.0  # winsorize in z-space

    def transform(self, X: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ > 1e-12, self.std_, 1.0)
        Z = (X - self.mean_) / std
        if self.clip_ is not None:
            Z = np.clip(Z, -self.clip_, self.clip_)
        return Z

def fit_zscaler(X: np.ndarray) -> ZScaler:
    return ZScaler(mean_=np.nanmean(X, axis=0), std_=np.nanstd(X, axis=0))


# -------------------- probability utilities --------------------

def sanitize_probs(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Ensure probabilities are finite, clipped to [eps, 1-eps], and rows re-normalized.
    Replace any bad rows with a uniform distribution.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.size == 0:
        return P
    N, C = P.shape
    bad = ~np.isfinite(P).all(axis=1) | (P.sum(axis=1) <= 0)
    if bad.any():
        P[bad] = 1.0 / C
    P = np.clip(P, eps, 1.0 - eps)
    P = P / P.sum(axis=1, keepdims=True)
    return P


# -------------------- metrics --------------------

def topk_acc(probs: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    probs = sanitize_probs(probs)
    if probs.size == 0:
        return 0.0
    k = min(k, probs.shape[1])
    top = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
    return float(((top == y[:, None]).any(axis=1)).mean())

def nll_mc(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    probs = sanitize_probs(probs, eps)
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.log(p).mean())

def brier_mc(probs: np.ndarray, y: np.ndarray, C: int) -> float:
    probs = sanitize_probs(probs)
    Y = np.eye(C, dtype=float)[y]
    diff = probs - Y
    return float(np.mean(np.sum(diff * diff, axis=1)))


# -------------------- MNLogit helpers --------------------

def fit_mnlogit(X_df: pd.DataFrame, y: np.ndarray, *, ridge: float | None = None, maxiter: int = 200):
    exog = sm.add_constant(X_df, has_constant="add")
    model = sm.MNLogit(endog=y, exog=exog)
    if ridge is None or ridge <= 0:
        # try plain Newton; fall back to ridge if it fails
        try:
            return model.fit(method="newton", maxiter=maxiter, disp=False)
        except Exception:
            pass
        # mild ridge fallback
        return model.fit_regularized(alpha=1.0, L1_wt=0.0, maxiter=maxiter)
    else:
        # explicit ridge always
        return model.fit_regularized(alpha=float(ridge), L1_wt=0.0, maxiter=maxiter, disp=False)

def predict_mnlogit(res, X_df: pd.DataFrame) -> np.ndarray:
    """
    Stable softmax for statsmodels MNLogit.
    Params are for J-1 classes; baseline is the last class (J-1).
    Returns [N, J] in the standard 0..J-1 class order.
    """
    exog = sm.add_constant(X_df, has_constant="add").to_numpy(dtype=float)
    B = np.asarray(res.params, dtype=float)           # [k_exog, J-1]
    scores = exog @ B                                 # [N, J-1]
    scores_full = np.concatenate([scores, np.zeros((scores.shape[0], 1))], axis=1)  # add baseline
    lse = logsumexp(scores_full, axis=1, keepdims=True)
    P = np.exp(scores_full - lse)
    return sanitize_probs(P)


# -------------------- class compaction --------------------

@dataclass
class ClassMap:
    full_labels: List[str]         # e.g., 36 labels for cap=5
    present_codes: List[int]       # codes seen in TRAIN set
    code2compact: Dict[int, int]   # old -> 0..J-1
    compact2code: List[int]        # 0..J-1 -> old

def build_class_map(y_codes: np.ndarray, full_labels: List[str]) -> ClassMap:
    present = sorted(int(v) for v in np.unique(y_codes))
    code2compact = {c: i for i, c in enumerate(present)}
    compact2code = list(present)
    return ClassMap(
        full_labels=full_labels,
        present_codes=present,
        code2compact=code2compact,
        compact2code=compact2code,
    )

def remap_to_compact(y_codes: np.ndarray, cmap: ClassMap) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_compact, keep_mask). Rows with classes absent in TRAIN are dropped for metrics."""
    keep = np.array([c in cmap.code2compact for c in y_codes], dtype=bool)
    y_comp = np.array([cmap.code2compact[c] for c in y_codes[keep]], dtype=int)
    return y_comp, keep


# -------------------- training core --------------------

@dataclass
class TrainedCS:
    res: object
    scaler: ZScaler
    feat_order: List[str]
    class_map: ClassMap
    train_n: int
    val_n: int
    metrics: Dict[str, float]

def train_once(cfg: TrainConfig, df: pd.DataFrame) -> TrainedCS:
    # encode labels to full codes 0..(cap+1)^2-1 by index in label_grid
    full_labels = label_grid(cfg.cap)
    lab2code = {lbl: i for i, lbl in enumerate(full_labels)}
    y_full = np.array([lab2code[lbl] for lbl in df["label"].astype(str)], dtype=int)

    # time-ordered tail holdout
    n = len(df)
    n_val = max(50, int(math.ceil(cfg.val_frac * n)))
    n_trn = n - n_val
    if n_trn <= 0:
        raise CommandError("Validation split leaves no training data.")

    X_trn_raw = df.loc[:n_trn-1, CS_FEATURES].to_numpy(dtype=float)
    X_val_raw = df.loc[n_trn:, CS_FEATURES].to_numpy(dtype=float)
    y_trn_full = y_full[:n_trn]
    y_val_full = y_full[n_trn:]

    scaler = fit_zscaler(X_trn_raw)
    X_trn = scaler.transform(X_trn_raw)
    X_val = scaler.transform(X_val_raw)

    # compact classes to those present in TRAIN
    cmap = build_class_map(y_trn_full, full_labels)
    y_trn, keep_trn = remap_to_compact(y_trn_full, cmap)
    X_trn = X_trn[keep_trn]
    # VAL: keep only rows whose class exists in TRAIN for metrics/calibration
    y_val_comp, keep_val = remap_to_compact(y_val_full, cmap)
    X_val_keep = X_val[keep_val]

    if len(np.unique(y_trn)) < 2:
        raise CommandError("Not enough distinct CS classes in training fold.")

    Xtrn_df = pd.DataFrame(X_trn, columns=CS_FEATURES)
    Xval_df = pd.DataFrame(X_val_keep, columns=CS_FEATURES)

    # Fit: try Newton; fallback to gentle ridge
    res = fit_mnlogit(Xtrn_df, y_trn)

    # predictions on val (only kept rows)
    P_val = predict_mnlogit(res, Xval_df)  # [Nv_keep, C_present]
    C = P_val.shape[1]
    metrics = {
        "val_nll": nll_mc(P_val, y_val_comp, eps=1e-12),
        "val_brier": brier_mc(P_val, y_val_comp, C),
        "val_top1": topk_acc(P_val, y_val_comp, k=1),
        "val_top3": topk_acc(P_val, y_val_comp, k=3),
        "val_top5": topk_acc(P_val, y_val_comp, k=5),
        "val_coverage": float(keep_val.mean()),  # fraction of val rows whose class existed in train
        "train_rows": float(len(y_trn)),
        "val_rows_used": float(len(y_val_comp)),
    }

    return TrainedCS(
        res=res,
        scaler=scaler,
        feat_order=list(CS_FEATURES),
        class_map=cmap,
        train_n=int(len(y_trn)),
        val_n=int(len(y_val_comp)),
        metrics=metrics,
    )


# -------------------- calibration (optional) --------------------

def fit_isotonic_per_class(P: np.ndarray, y: np.ndarray, labels_present: List[str]):
    """Return dict[label_str] -> IsotonicRegression model (if sklearn available)."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return {}
    cal: Dict[str, object] = {}
    for k, lbl in enumerate(labels_present):
        yk = (y == k).astype(int)
        pk = P[:, k].astype(float)
        # skip degenerate
        s = int(yk.sum())
        if s == 0 or s == len(yk):
            continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(pk, yk)
        cal[lbl] = ir
    return cal


# -------------------- CV (optional) --------------------

def time_kfold_metrics(cfg: TrainConfig, df: pd.DataFrame, folds: int) -> Dict[str, float]:
    folds = int(folds)
    if folds <= 1 or len(df) < 200:
        return {}
    n = len(df)
    fold_sizes = [n // folds + (1 if i < n % folds else 0) for i in range(folds)]
    idx = np.arange(n)
    start = 0
    vals = []
    for _, fs in enumerate(fold_sizes):
        val_idx = idx[start:start+fs]
        trn_idx = np.concatenate([idx[:start], idx[start+fs:]])
        start += fs

        df_trn = df.iloc[trn_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        # Train on df_trn; evaluate on df_val using same compaction rule
        full_labels = label_grid(cfg.cap)
        lab2code = {lbl: i for i, lbl in enumerate(full_labels)}
        y_trn_full = np.array([lab2code[l] for l in df_trn["label"].astype(str)], dtype=int)
        y_val_full = np.array([lab2code[l] for l in df_val["label"].astype(str)], dtype=int)

        X_trn_raw = df_trn[CS_FEATURES].to_numpy(dtype=float)
        X_val_raw = df_val[CS_FEATURES].to_numpy(dtype=float)

        scaler = fit_zscaler(X_trn_raw)
        X_trn = scaler.transform(X_trn_raw)
        X_val = scaler.transform(X_val_raw)

        cmap = build_class_map(y_trn_full, full_labels)
        y_trn, keep_trn = remap_to_compact(y_trn_full, cmap)
        if len(np.unique(y_trn)) < 2:
            continue
        X_trn = X_trn[keep_trn]
        y_val_comp, keep_val = remap_to_compact(y_val_full, cmap)
        X_val_keep = X_val[keep_val]

        res = fit_mnlogit(pd.DataFrame(X_trn, columns=CS_FEATURES), y_trn)
        P = predict_mnlogit(res, pd.DataFrame(X_val_keep, columns=CS_FEATURES))
        if P.size == 0 or len(y_val_comp) == 0:
            continue

        vals.append({
            "nll": nll_mc(P, y_val_comp),
            "brier": brier_mc(P, y_val_comp, P.shape[1]),
            "top1": topk_acc(P, y_val_comp, 1),
            "top3": topk_acc(P, y_val_comp, 3),
            "top5": topk_acc(P, y_val_comp, 5),
            "coverage": float(keep_val.mean()),
        })
    if not vals:
        return {}
    out = {k: float(np.mean([d[k] for d in vals])) for k in vals[0].keys()}
    out["folds"] = float(len(vals))
    return out


# -------------------- persistence --------------------

def save_bundle(cfg: TrainConfig, t: TrainedCS) -> Tuple[str, Optional[str], Dict]:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    ts = dj_tz.now().strftime("%Y%m%d_%H%M%S")
    league_dir = cfg.outdir / f"L{cfg.league_id}"
    league_dir.mkdir(parents=True, exist_ok=True)

    present_labels = [t.class_map.full_labels[c] for c in t.class_map.present_codes]

    model_path = str(league_dir / f"cs_model_{ts}.joblib")
    bundle = {
        "type": "statsmodels_mnlogit_cs",
        "feature_order": t.feat_order,
        "zscaler_mean": t.scaler.mean_,
        "zscaler_std": t.scaler.std_,
        "z_clip": t.scaler.clip_,
        "full_labels": t.class_map.full_labels,      # full grid (e.g., 36 labels for cap=5)
        "present_codes": t.class_map.present_codes,  # codes used in training
        "present_labels": present_labels,            # strings for convenience
        "code2compact": t.class_map.code2compact,    # mapping for inference
        "compact2code": t.class_map.compact2code,
        "model": t.res,
    }
    joblib.dump(bundle, model_path, compress=3)

    # Calibration on the tail holdout rows
    cal_path = None
    try:
        # Build a fresh val set for calibration using same split and compaction
        df_all = fetch_df(cfg)
        n = len(df_all)
        n_val = max(50, int(math.ceil(cfg.val_frac * n)))
        n_trn = n - n_val
        full_labels = label_grid(cfg.cap)
        lab2code = {lbl: i for i, lbl in enumerate(full_labels)}
        y_full = np.array([lab2code[l] for l in df_all["label"].astype(str)], dtype=int)
        X_raw = df_all[CS_FEATURES].to_numpy(dtype=float)

        X = t.scaler.transform(X_raw)

        cmap = t.class_map
        y_val_full = y_full[n_trn:]
        y_val_comp, keep_val = remap_to_compact(y_val_full, cmap)
        X_val_keep = X[n_trn:][keep_val]

        if len(y_val_comp) > 0:
            P_val = predict_mnlogit(t.res, pd.DataFrame(X_val_keep, columns=CS_FEATURES))
            calibrators = fit_isotonic_per_class(P_val, y_val_comp, present_labels)
            if calibrators:
                cal_path = str(league_dir / f"cs_cal_{ts}.joblib")
                joblib.dump(calibrators, cal_path, compress=3)
    except Exception:
        cal_path = None  # calibration is optional; never fail training because of it

    metrics_blob = dict(t.metrics)
    if cal_path:
        metrics_blob["calibration_file"] = cal_path

    return model_path, cal_path, metrics_blob


def register_model_version(
    league_id: int,
    trained_until: datetime,
    model_path: str,
    metrics_blob: Dict,
    cal_path: Optional[str],
) -> None:
    mv = ModelVersion(
        name=f"CS L{league_id} {trained_until:%Y-%m-%d}",
        kind="market:CS",
        league_id=league_id,
        trained_until=trained_until,
        metrics_json=json.dumps(metrics_blob, default=float),
        file_home=model_path,     # reuse this field to store the bundle
        file_away=None,
        calibration_json={"file": cal_path} if cal_path else {},
    )
    mv.save()


# -------------------- CLI --------------------

class Command(BaseCommand):
    help = "Train a Correct Score (CS) multinomial model and register ModelVersion(kind='market:CS')."

    def add_arguments(self, parser):
        parser.add_argument("--league-id", type=int, required=True)
        # Either seasons or date window
        parser.add_argument("--seasons", type=int, nargs="*", default=None,
                            help="Filter MLTrainingMatch by season values (e.g. 2020 2021 2022).")
        parser.add_argument("--start", type=str, default=None,
                            help="ISO date YYYY-MM-DD (naive OK). Ignored if --seasons is used.")
        parser.add_argument("--end", type=str, default=None,
                            help="ISO date YYYY-MM-DD (naive OK). Default: now. Ignored if --seasons is used.")
        # Model args
        parser.add_argument("--cap", type=int, default=5)
        parser.add_argument("--val-frac", type=float, default=0.15)
        parser.add_argument("--no-cal", action="store_true")
        # Paths / misc
        parser.add_argument("--outdir", type=str, default="models/cs",
                            help="Directory for artifacts.")
        parser.add_argument("--models-dir", type=str, default=None,
                            help="Alias for --outdir.")
        parser.add_argument("--folds", type=int, default=0,
                            help="Optional time-ordered K-fold CV for metrics (0/1 disables).")

    def _aware(self, s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            raise CommandError(f"Invalid date: {s!r}. Use YYYY-MM-DD.")
        # Make timezone-aware
        if dj_tz.is_naive(dt):
            dt = dj_tz.make_aware(dt, dj_tz.get_default_timezone())
        return dt

    def handle(self, *args, **opts):
        league_id = int(opts["league_id"])
        seasons = opts.get("seasons") or None

        # Resolve dates only if seasons not provided
        dt_start = self._aware(opts.get("start")) if not seasons else None
        dt_end = self._aware(opts.get("end")) if not seasons else dj_tz.now()

        outdir = Path(opts.get("models_dir") or opts.get("outdir") or "models/cs")

        cfg = TrainConfig(
            league_id=league_id,
            seasons=seasons,
            dt_start=dt_start,
            dt_end=dt_end or dj_tz.now(),
            cap=int(opts["cap"]),
            val_frac=float(opts["val_frac"]),
            outdir=outdir,
            calibrate=(not bool(opts["no_cal"])),
            folds=int(opts.get("folds") or 0),
        )

        # Info banner
        if cfg.seasons:
            span = f"seasons={cfg.seasons}"
        else:
            span = f"window=({cfg.dt_start or 'MIN'} .. {cfg.dt_end:%Y-%m-%d})"

        self.stdout.write(
            f"Training CS model for league {league_id} {span} "
            f"cap={cfg.cap} val_frac={cfg.val_frac} calibrate={cfg.calibrate} folds={cfg.folds}"
        )

        df = fetch_df(cfg)
        if df.empty:
            raise CommandError("No rows found for the specified filters.")

        # Optional time-ordered K-fold CV (metrics only)
        if cfg.folds and cfg.folds > 1:
            cvm = time_kfold_metrics(cfg, df, cfg.folds)
            if cvm:
                self.stdout.write("CV metrics (time-ordered):")
                for k, v in cvm.items():
                    self.stdout.write(f"  {k}: {v:.6f}")

        # Final train (with tail holdout used for metrics & calibration)
        trained = train_once(cfg, df)

        model_path, cal_path, metrics_blob = save_bundle(cfg, trained)
        # Merge CV metrics into metrics_blob if present
        if cfg.folds and cfg.folds > 1:
            cvm = time_kfold_metrics(cfg, df, cfg.folds)
            if cvm:
                metrics_blob["cv"] = cvm

        register_model_version(
            league_id=league_id,
            trained_until=(cfg.dt_end if cfg.seasons is None else dj_tz.now()),
            model_path=model_path,
            metrics_blob=metrics_blob,
            cal_path=cal_path,
        )

        self.stdout.write(self.style.SUCCESS("Training complete."))
        self.stdout.write(f"Model: {model_path}")
        if cal_path:
            self.stdout.write(f"Calibration: {cal_path}")
        self.stdout.write("Metrics:")
        for k, v in trained.metrics.items():
            self.stdout.write(f"  {k}: {v:.6f}")
