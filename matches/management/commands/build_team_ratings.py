from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Q

from matches.models import Match, TeamRating, LeagueSeasonParams

# Treat these as “finished” labels; extend if your provider uses more
FINISHED = ("FT", "AET", "PEN")


# ------------------------- small utils -------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _team_name(team_id: int) -> str:
    """Best-effort human name for pretty printing; falls back to the ID."""
    try:
        from matches.models import Team
        t = Team.objects.filter(pk=team_id).only("name").first()
        if t and getattr(t, "name", None):
            return str(t.name)
    except Exception:
        pass
    return str(team_id)


# ------------------------- data shapes -------------------------

@dataclass
class FitResult:
    league_id: int
    season: int
    team_ids: List[int]
    attack: Dict[int, float]   # mean-zero
    defense: Dict[int, float]  # mean-zero, SUBTRACT in link: exp(a_home - d_away + ...)
    hfa: float                 # home-field advantage coefficient
    intercept: float           # baseline log-mean so averages are preserved
    avg_goals: float           # empirical average goals per TEAM (per row)
    n_matches: int


# ------------------------- core math -------------------------

def _collect_matches(league_id: int, season: int, min_matches: int) -> List[Match]:
    """
    Gather finished & labeled matches for one (league, season).
    Skips rows with null team IDs or goal labels.
    """
    qs = (
        Match.objects
        .filter(league_id=league_id, season=season, status__in=FINISHED)
        .exclude(Q(goals_home__isnull=True) | Q(goals_away__isnull=True))
        .exclude(Q(home_id__isnull=True) | Q(away_id__isnull=True))
        .only("home_id", "away_id", "goals_home", "goals_away", "kickoff_utc")
        .order_by("kickoff_utc")
    )
    matches = list(qs)
    return matches if len(matches) >= int(min_matches) else []


def _build_design(matches: List[Match]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Two rows per match:
      row1: y = goals_home; X = one-hot attack(home) + one-hot (+defense)(away) + [home_ind=1]
      row2: y = goals_away; X = one-hot attack(away) + one-hot (+defense)(home) + [home_ind=0]
    We fit +defense in the linear predictor, and later convert to stored `defense` = -coef.
    """
    # Collect the team ID universe
    team_ids = sorted({int(m.home_id) for m in matches} | {int(m.away_id) for m in matches})
    T = len(team_ids)
    if T == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float), []

    idx = {tid: i for i, tid in enumerate(team_ids)}
    rows: List[np.ndarray] = []
    yy: List[float] = []

    for m in matches:
        # Home row
        x1 = np.zeros(2 * T + 1, dtype=float)
        x1[idx[int(m.home_id)]] = 1.0          # attack(home)
        x1[T + idx[int(m.away_id)]] = 1.0      # +defense(away)
        x1[-1] = 1.0                            # home indicator
        rows.append(x1)
        yy.append(_safe_float(m.goals_home, 0.0))

        # Away row
        x2 = np.zeros(2 * T + 1, dtype=float)
        x2[idx[int(m.away_id)]] = 1.0          # attack(away)
        x2[T + idx[int(m.home_id)]] = 1.0      # +defense(home)
        x2[-1] = 0.0                            # not home
        rows.append(x2)
        yy.append(_safe_float(m.goals_away, 0.0))

    X = np.vstack(rows)
    y = np.asarray(yy, dtype=float)
    return X, y, team_ids


def _fit_poisson(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float
) -> Tuple[np.ndarray, float]:
    """
    Poisson GLM with log link + L2 regularization.
    Returns (coef, intercept), where coef has length 2T+1 in the layout from _build_design.
    """
    from sklearn.linear_model import PoissonRegressor

    model = PoissonRegressor(
        alpha=float(alpha),
        fit_intercept=True,
        max_iter=int(max_iter),
        tol=float(tol),
        warm_start=False,
    )
    model.fit(X, y)
    return model.coef_.astype(float), float(model.intercept_)


def _center_and_convert(
    coef: np.ndarray,
    intercept: float,
    T: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Convert (+defense) to stored defense (negative of that block), and center
    attack/defense to mean 0. Adjust intercept so predictions are unchanged.
    """
    atk = coef[:T].copy()
    d_plus = coef[T:2 * T].copy()  # fitted with a + sign in the linear predictor
    hfa = float(coef[2 * T])

    a_mean = float(atk.mean()) if T else 0.0
    d_mean = float(d_plus.mean()) if T else 0.0

    atk -= a_mean
    d_plus -= d_mean

    intercept_adj = float(intercept + a_mean + d_mean)
    defense = -d_plus  # store as "defense" (subtract it during scoring)

    return atk, defense, hfa, intercept_adj


def _fit_one(
    league_id: int,
    season: int,
    alpha: float,
    max_iter: int,
    tol: float,
    min_matches: int
) -> Optional[FitResult]:
    matches = _collect_matches(league_id, season, min_matches=min_matches)
    if not matches:
        return None

    X, y, team_ids = _build_design(matches)
    if X.size == 0 or y.size == 0:
        return None

    coef, intercept = _fit_poisson(X, y, alpha=alpha, max_iter=max_iter, tol=tol)
    T = len(team_ids)
    atk, defense, hfa, intercept_adj = _center_and_convert(coef, intercept, T)

    # Empirical average goals per TEAM (per “row”); useful diagnostics
    avg_goals = float(np.mean(y)) if y.size else 0.0

    return FitResult(
        league_id=league_id,
        season=season,
        team_ids=team_ids,
        attack={tid: float(atk[i]) for i, tid in enumerate(team_ids)},
        defense={tid: float(defense[i]) for i, tid in enumerate(team_ids)},
        hfa=float(hfa),
        intercept=float(intercept_adj),
        avg_goals=avg_goals,
        n_matches=len(matches),
    )


# ------------------------- discover helpers -------------------------

def _discover_leagues() -> List[int]:
    qs = (
        Match.objects
        .filter(status__in=FINISHED)
        .values_list("league_id", flat=True)
        .distinct()
    )
    return sorted(int(x) for x in qs)


def _discover_seasons(league_id: int) -> List[int]:
    qs = (
        Match.objects
        .filter(status__in=FINISHED, league_id=league_id)
        .values_list("season", flat=True)
        .distinct()
    )
    return sorted(int(x) for x in qs)


# ------------------------- persistence -------------------------

def _save_fit(fr: FitResult) -> None:
    """
    Persist ratings for one (league, season) atomically: either all teams + params
    are written, or none are (if an exception occurs).
    """
    now = datetime.now(timezone.utc)
    with transaction.atomic():
        # per-team ratings
        for tid in fr.team_ids:
            TeamRating.objects.update_or_create(
                league_id=fr.league_id,
                season=fr.season,
                team_id=int(tid),
                defaults={
                    "attack": fr.attack[tid],
                    "defense": fr.defense[tid],
                    "last_updated": now,
                },
            )

        # league-season global params for scoring
        LeagueSeasonParams.objects.update_or_create(
            league_id=fr.league_id,
            season=fr.season,
            defaults={
                "intercept": fr.intercept,
                "hfa": fr.hfa,
                "avg_goals": fr.avg_goals,
                "n_matches": fr.n_matches,
            },
        )


# ------------------------- command -------------------------

class Command(BaseCommand):
    help = "Build team attack/defense ratings + league-season intercept & HFA using a Poisson GLM."

    def add_arguments(self, parser):
        parser.add_argument(
            "--league-ids", type=int, nargs="*",
            help="Leagues to fit (omit to auto-discover)."
        )
        parser.add_argument(
            "--seasons", type=int, nargs="*",
            help="Seasons to fit (omit to auto-discover per league)."
        )
        parser.add_argument("--alpha", type=float, default=1.0, help="L2 regularization strength.")
        parser.add_argument("--max-iter", type=int, default=2000, help="Max iterations for optimizer.")
        parser.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance.")
        parser.add_argument("--min-matches", type=int, default=10, help="Minimum finished matches required.")
        parser.add_argument("--quiet", action="store_true", help="Less verbose output.")

    def handle(self, *args, **opts):
        league_ids: Optional[Sequence[int]] = opts.get("league_ids")
        seasons_cli: Optional[Sequence[int]] = opts.get("seasons")
        alpha = float(opts["alpha"])
        max_iter = int(opts["max_iter"])
        tol = float(opts["tol"])
        min_matches = int(opts["min_matches"])
        quiet = bool(opts["quiet"])

        leagues = list(league_ids or []) or _discover_leagues()
        if not leagues:
            self.stderr.write(self.style.ERROR("No leagues with finished matches found."))
            return

        total_pairs = 0
        fitted_pairs = 0

        for lg in leagues:
            seasons = list(seasons_cli or []) or _discover_seasons(lg)
            if not seasons:
                if not quiet:
                    self.stdout.write(self.style.WARNING(f"League {lg}: no seasons with finished matches."))
                continue

            for ssn in seasons:
                total_pairs += 1
                fr = _fit_one(
                    league_id=int(lg),
                    season=int(ssn),
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    min_matches=min_matches,
                )
                if fr is None:
                    if not quiet:
                        self.stdout.write(self.style.WARNING(
                            f"League {lg} season {ssn}: skipped (not enough finished & labeled matches)."
                        ))
                    continue

                _save_fit(fr)
                fitted_pairs += 1

                if not quiet:
                    self.stdout.write("")
                    self.stdout.write(self.style.SUCCESS(
                        f"L={lg} S={ssn}  matches={fr.n_matches}  "
                        f"avg_goals={fr.avg_goals:.3f}  HFA={fr.hfa:.3f}  intercept={fr.intercept:.3f}"
                    ))
                    self.stdout.write(" Team ID  Team                            Attack   Defense")
                    self.stdout.write("---------------------------------------------------------")
                    for tid in fr.team_ids:
                        self.stdout.write(
                            f"{tid:8d}  {(_team_name(tid)[:30]).ljust(30)}  "
                            f"{fr.attack[tid]:7.3f}   {fr.defense[tid]:7.3f}"
                        )

        if fitted_pairs == 0:
            self.stderr.write(self.style.ERROR("No (league, season) pairs were fitted."))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Done. Fitted {fitted_pairs}/{total_pairs} (league, season) pairs."
            ))
