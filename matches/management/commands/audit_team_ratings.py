# prediction/matches/management/commands/audit_team_ratings.py
from __future__ import annotations
import csv
import math
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q, Count

from matches.models import Match, TeamRating

# ---------- helpers ----------
def _safe_float(x, d=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except Exception:
        return d

def _poisson_logpmf(k: int, lam: float) -> float:
    if lam <= 0:
        return -1e9 if k > 0 else 0.0
    # Stirling-ish safe log pmf: k*log(lam) - lam - log(k!)
    # use scipy if you have it; otherwise quick approx
    # here we do a tiny table for k<=12 and stirling for larger
    from math import lgamma, log
    return k * log(lam) - lam - lgamma(k + 1.0)

@dataclass
class PairStats:
    n_matches: int = 0
    mae_home: float = 0.0
    mae_away: float = 0.0
    rmse_home: float = 0.0
    rmse_away: float = 0.0
    ll_total: float = 0.0
    goals_avg: float = 0.0
    xg_avg: float = 0.0
    coverage_ok: bool = True
    missing_teams: List[int] = None

# ---------- command ----------
class Command(BaseCommand):
    help = "Audit saved TeamRating rows by comparing predicted vs actual goals on finished matches."

    def add_arguments(self, p):
        scope = p.add_mutually_exclusive_group(required=False)
        scope.add_argument("--season", type=int, help="Single season.")
        scope.add_argument("--season-min", type=int, help="Lower bound (inclusive). Use with --season-max.")
        p.add_argument("--season-max", type=int, help="Upper bound (inclusive). Use with --season-min.")
        p.add_argument("--league-ids", type=int, nargs="*", help="Optional league filter.")
        p.add_argument("--min-matches", type=int, default=30, help="Require at least this many finished matches to audit.")
        p.add_argument("--sample", type=int, default=0, help="Sample size per pair (0 = use all).")
        p.add_argument("--out", type=str, default="audit_team_ratings.csv", help="CSV of per-match residuals (optional).")
        p.add_argument("--summary-out", type=str, default="audit_team_ratings_summary.csv", help="CSV of per-pair summary.")
        p.add_argument("--calib-bins", type=int, default=10, help="Calibration bins for expected goals.")
        p.add_argument("--verbose", action="store_true", help="Print extra info per pair.")

    def handle(self, *args, **opt):
        season      = opt.get("season")
        smin        = opt.get("season_min")
        smax        = opt.get("season_max")
        league_ids  = opt.get("league_ids") or []
        min_matches = int(opt.get("min_matches"))
        sample      = int(opt.get("sample"))
        out_path    = opt.get("out")
        sum_path    = opt.get("summary_out")
        nbins       = max(2, int(opt.get("calib_bins")))
        verbose     = bool(opt.get("verbose"))

        if (smin is None) ^ (smax is None):
            raise CommandError("Use --season-min together with --season-max (or pass --season).")

        seasons = [int(season)] if season is not None else list(range(int(smin), int(smax) + 1))

        base = Match.objects.filter(
            status__in=["FT", "AET", "PEN"]
        ).exclude(Q(goals_home__isnull=True) | Q(goals_away__isnull=True))
        if league_ids:
            base = base.filter(league_id__in=league_ids)
        base = base.filter(season__in=seasons)

        pairs = list(
            base.values("league_id", "season")
                .annotate(n=Count("id"))
                .order_by("league_id", "season")
        )

        # prepare CSV writers
        outf = open(out_path, "w", newline="")
        w = csv.writer(outf)
        w.writerow([
            "league_id","season","fixture_id","home_id","away_id",
            "g_home","g_away","mu_home","mu_away",
            "res_home","res_away"
        ])

        sumf = open(sum_path, "w", newline="")
        ws = csv.writer(sumf)
        ws.writerow([
            "league_id","season","n_matches",
            "mae_home","mae_away","rmse_home","rmse_away",
            "avg_goals","avg_mu","loglik_per_match",
            "coverage_ok","missing_teams_count",
            "hfa","intercept"
        ])

        total_pairs = 0
        for p in pairs:
            L, S, N = int(p["league_id"]), int(p["season"]), int(p["n"])
            if N < min_matches:
                continue

            # load ratings for this pair
            trs = list(TeamRating.objects.filter(league_id=L, season=S))
            atk = {r.team_id: _safe_float(getattr(r, "attack", 0.0), 0.0) for r in trs}
            dfn = {r.team_id: _safe_float(getattr(r, "defense", 0.0), 0.0) for r in trs}

            # pull intercept/HFA if you stored them (optional fields).
            # If you didn’t persist, we back off to 0.0 and the fit still audits OK.
            # You can add fields to TeamRating or a LeagueSeason table; here we try attributes.
            hfa = _safe_float(getattr(TeamRating, "HFA_DEFAULT", 0.0), 0.0)  # fallback
            intercept = 0.0
            # If you saved a special row (team_id=0) or another store, try to fetch it:
            # (comment this out if not used)
            special = next((r for r in trs if getattr(r, "team_id", None) in (0, -1)), None)
            if special:
                hfa = _safe_float(getattr(special, "hfa", hfa), hfa)
                intercept = _safe_float(getattr(special, "intercept", intercept), intercept)

            # matches to score
            mqs = base.filter(league_id=L, season=S).order_by("kickoff_utc")
            rows = list(mqs)
            if sample and sample < len(rows):
                # simple head sample (or random if you prefer)
                rows = rows[:sample]

            # coverage check (do we have ratings for all teams that played?)
            teams_in_matches = set()
            for m in rows:
                teams_in_matches.add(int(m.home_id)); teams_in_matches.add(int(m.away_id))
            missing = sorted([t for t in teams_in_matches if t not in atk or t not in dfn])
            coverage_ok = (len(missing) == 0)

            # compute metrics
            n = 0
            abs_home = []; abs_away = []
            sq_home = []; sq_away = []
            ll = 0.0
            goals = 0.0; mus = 0.0

            # calibration collectors
            # bins on mu for home/away separately (stack them together for simplicity)
            mu_vals = []
            g_vals  = []

            for m in rows:
                a_h = atk.get(int(m.home_id), 0.0)
                d_a = dfn.get(int(m.away_id), 0.0)
                a_a = atk.get(int(m.away_id), 0.0)
                d_h = dfn.get(int(m.home_id), 0.0)

                mu_home = math.exp(intercept + a_h - d_a + hfa)
                mu_away = math.exp(intercept + a_a - d_h + 0.0)

                g_home = _safe_float(m.goals_home, 0.0)
                g_away = _safe_float(m.goals_away, 0.0)

                abs_home.append(abs(g_home - mu_home))
                abs_away.append(abs(g_away - mu_away))
                sq_home.append((g_home - mu_home)**2)
                sq_away.append((g_away - mu_away)**2)

                ll += _poisson_logpmf(int(g_home), mu_home)
                ll += _poisson_logpmf(int(g_away), mu_away)

                goals += g_home + g_away
                mus   += mu_home + mu_away
                mu_vals.extend([mu_home, mu_away])
                g_vals.extend([g_home, g_away])

                w.writerow([L, S, m.id, m.home_id, m.away_id,
                            f"{g_home:.6f}", f"{g_away:.6f}",
                            f"{mu_home:.6f}", f"{mu_away:.6f}",
                            f"{(g_home-mu_home):.6f}", f"{(g_away-mu_away):.6f}"])
                n += 1

            if n == 0:
                continue

            mae_h = float(np.mean(abs_home))
            mae_a = float(np.mean(abs_away))
            rmse_h = float(np.sqrt(np.mean(sq_home)))
            rmse_a = float(np.sqrt(np.mean(sq_away)))
            avg_goals = goals / (n * 2.0)
            avg_mu    = mus   / (n * 2.0)
            ll_per_match = ll / n  # (home+away) both included

            # simple calibration table (optional print)
            calib = []
            try:
                q = np.quantile(mu_vals, np.linspace(0,1,nbins+1))
                for i in range(nbins):
                    lo, hi = q[i], q[i+1]
                    idx = [j for j, mu in enumerate(mu_vals) if (mu >= lo and mu <= hi if i==nbins-1 else mu >= lo and mu < hi)]
                    if not idx:
                        calib.append((i+1, lo, hi, 0, float('nan'), float('nan')))
                        continue
                    obs = float(np.mean([g_vals[j] for j in idx]))
                    exp = float(np.mean([mu_vals[j] for j in idx]))
                    calib.append((i+1, lo, hi, len(idx), obs, exp))
            except Exception:
                pass

            if verbose:
                self.stdout.write(
                    f"league={L} season={S} n={n} "
                    f"MAE(H/A)={mae_h:.3f}/{mae_a:.3f} RMSE(H/A)={rmse_h:.3f}/{rmse_a:.3f} "
                    f"avg_goals≈{avg_goals:.3f} avg_mu≈{avg_mu:.3f} ll/match={ll_per_match:.3f} "
                    f"coverage_ok={coverage_ok} missing={len(missing)}"
                )
                if calib:
                    self.stdout.write("  calibration (bin, mu_lo-hi, n, obs, exp):")
                    for b, lo, hi, nn, obs, exp in calib:
                        self.stdout.write(f"    {b:2d} [{lo:.2f},{hi:.2f}] n={nn:4d}  obs={obs:.3f}  exp={exp:.3f}")

            ws.writerow([
                L, S, n, f"{mae_h:.6f}", f"{mae_a:.6f}", f"{rmse_h:.6f}", f"{rmse_a:.6f}",
                f"{avg_goals:.6f}", f"{avg_mu:.6f}", f"{ll_per_match:.6f}",
                int(coverage_ok), len(missing),
                f"{hfa:.6f}", f"{intercept:.6f}",
            ])
            total_pairs += 1

        outf.close(); sumf.close()
        self.stdout.write(self.style.SUCCESS(f"Audited {total_pairs} (league, season) pairs."))
        self.stdout.write(self.style.SUCCESS(f"Per-match residuals → {out_path}"))
        self.stdout.write(self.style.SUCCESS(f"Per-pair summary    → {sum_path}"))
