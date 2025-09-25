# debug_match.py
import json, math, numpy as np, joblib
from datetime import datetime, timezone, timedelta
from django.core.management.base import BaseCommand
from django.db.models import Q
from matches.models import Match, ModelVersion
# add near the top of debug_match.py (below imports)

def team_name(team_obj):
    # Works whether team is a FK to a Team model or already a string/int
    if team_obj is None:
        return None
    # Try common attribute names
    for attr in ("name", "team_name", "fullname", "short_name"):
        if hasattr(team_obj, attr):
            val = getattr(team_obj, attr)
            if isinstance(val, str):
                return val
    # Fallback to str()
    return str(team_obj)


FEATS = [
    "h_gf10","a_gf10","d_gf10",
    "h_ga10","a_ga10","d_ga10",
    "h_sot10","a_sot10","d_sot10",
    "h_shots10","a_shots10","d_shots10",
    "h_sot_pct10","a_sot_pct10","d_sot_pct10",
    "h_conv10","a_conv10","d_conv10",
    "h_poss10","a_poss10","d_poss10",
    "h_cs_rate10","a_cs_rate10","d_cs_rate10",
    "h_corners10","a_corners10","d_corners10",
    "h_cards10","a_cards10","d_cards10",
    "h_rest_days","a_rest_days","matches_14d",
    "table_rank_home","table_rank_away","table_rank_delta",
    "h_stats_missing","a_stats_missing",
]

def _avg(a): return float(np.mean(a)) if a else 0.0

def _form_goals_only(team_id, upto_dt, lookback=10):
    qs = (Match.objects
          .filter(Q(home_id=team_id)|Q(away_id=team_id),
                  kickoff_utc__lt=upto_dt,
                  status__in=["FT","AET","PEN"])
          .order_by("-kickoff_utc")[:lookback])
    gf, ga, cs = [], [], []
    for m in qs:
        if m.home_id == team_id:
            gf.append(m.goals_home or 0); ga.append(m.goals_away or 0)
            cs.append(1.0 if (m.goals_away or 0)==0 else 0.0)
        else:
            gf.append(m.goals_away or 0); ga.append(m.goals_home or 0)
            cs.append(1.0 if (m.goals_home or 0)==0 else 0.0)
    return {"gf": _avg(gf), "ga": _avg(ga), "cs_rate": _avg(cs), "count": len(qs)}

def _rest_days(team_id, upto_dt):
    last = (Match.objects
            .filter(Q(home_id=team_id)|Q(away_id=team_id),
                    kickoff_utc__lt=upto_dt,
                    status__in=["FT","AET","PEN"])
            .order_by("-kickoff_utc").first())
    if not last: return 7.0
    d = upto_dt - last.kickoff_utc
    return max(0.0, d.days + d.seconds/86400.0)

def _build_minimal_feats(m: Match):
    H = _form_goals_only(m.home_id, m.kickoff_utc)
    A = _form_goals_only(m.away_id, m.kickoff_utc)
    feats = {k:0.0 for k in FEATS}
    feats.update({
        "h_gf10": H["gf"], "a_gf10": A["gf"], "d_gf10": H["gf"]-A["gf"],
        "h_ga10": H["ga"], "a_ga10": A["ga"], "d_ga10": H["ga"]-A["ga"],
        "h_cs_rate10": H["cs_rate"], "a_cs_rate10": A["cs_rate"], "d_cs_rate10": H["cs_rate"]-A["cs_rate"],
        "h_rest_days": _rest_days(m.home_id, m.kickoff_utc),
        "a_rest_days": _rest_days(m.away_id, m.kickoff_utc),
        "table_rank_home": 0.0, "table_rank_away": 0.0, "table_rank_delta": 0.0,
        "h_stats_missing": 1.0 if H["count"]<3 else 0.0,
        "a_stats_missing": 1.0 if A["count"]<3 else 0.0,
    })
    X = np.array([feats.get(f, 0.0) for f in FEATS], dtype=float)
    return feats, X

def _poiss_pmf(k, lam): return math.exp(-lam) * (lam**k) / math.factorial(k)

def _joint_grid(lh, la, K=10):
    P = np.zeros((K+1, K+1))
    for h in range(K+1):
        ph = _poiss_pmf(h, lh)
        for a in range(K+1):
            P[h,a] = ph * _poiss_pmf(a, la)
    s = P.sum()
    if s>0: P /= s
    return P

def _market_probs(lh, la, K=10):
    P = _joint_grid(lh, la, K)
    pH = P[np.arange(K+1)[:,None] > np.arange(K+1)[None,:]].sum()
    pD = np.diag(P).sum()
    pA = P[np.arange(K+1)[:,None] < np.arange(K+1)[None,:]].sum()
    pBTTS = P[1:,1:].sum()
    pO15 = sum(P[h,a] for h in range(K+1) for a in range(K+1) if h+a >= 2)
    pO25 = sum(P[h,a] for h in range(K+1) for a in range(K+1) if h+a >= 3)
    pO35 = sum(P[h,a] for h in range(K+1) for a in range(K+1) if h+a >= 4)
    return {
        "1X2": {"H": float(pH), "D": float(pD), "A": float(pA)},
        "BTTS": {"yes": float(pBTTS), "no": float(1-pBTTS)},
        "OU": {"1.5_over": float(pO15), "1.5_under": float(1-pO15),
               "2.5_over": float(pO25), "2.5_under": float(1-pO25),
               "3.5_over": float(pO35), "3.5_under": float(1-pO35)}
    }

def _odds(p): return float('inf') if p<=0 else 1.0/p

class Command(BaseCommand):
    help = "Debug: show features, lambdas, and market odds for one match (no shell needed)."

    def add_arguments(self, parser):
        parser.add_argument("--match-id", type=int, help="Specific match ID")
        parser.add_argument("--league-id", type=int, help="Pick next upcoming in this league")
        parser.add_argument("--days", type=int, default=14, help="Search window for upcoming")
        parser.add_argument("--json-out", type=str, default="", help="If set, write JSON to this path")

    def handle(self, *args, **o):
        m = None
        if o.get("match_id"):
            m = Match.objects.filter(pk=o["match_id"]).first()
        else:
            if not o.get("league_id"):
                self.stderr.write("Provide --match-id OR --league-id")
                return
            now = datetime.now(timezone.utc)
            upto = now + timedelta(days=o["days"])
            m = (Match.objects
                 .filter(league_id=o["league_id"], kickoff_utc__gte=now, kickoff_utc__lte=upto)
                 .order_by("kickoff_utc").first())
        if not m:
            self.stderr.write("No match found.")
            return

        mv = (ModelVersion.objects
              .filter(kind="goals", league_id=m.league_id)
              .order_by("-trained_until","-id").first())
        if not mv:
            self.stderr.write("No ModelVersion. Run train_goals first.")
            return

        feats, X = _build_minimal_feats(m)
        home_model = joblib.load(mv.file_home)
        away_model = joblib.load(mv.file_away)
        lh = float(np.clip(home_model.predict([X])[0], 0.05, 6.0))
        la = float(np.clip(away_model.predict([X])[0], 0.05, 6.0))
        markets = _market_probs(lh, la)

        payload = {
            "match_id": m.id,
 "home": m.home,
   "away": m.away,
   "home": team_name(m.home),
   "away": team_name(m.away),
    "kickoff_utc": m.kickoff_utc.isoformat(),
            "lambdas": {"home": lh, "away": la},
            "markets": {
                "1X2": {k: {"p": v, "fair_odds": _odds(v)} for k,v in markets["1X2"].items()},
                "BTTS": {k: {"p": v, "fair_odds": _odds(v)} for k,v in markets["BTTS"].items()},
                "OU":   {k: {"p": v, "fair_odds": _odds(v)} for k,v in markets["OU"].items()},
            },
            "key_feats": {k: feats[k] for k in ["h_gf10","a_gf10","h_ga10","a_ga10","h_cs_rate10","a_cs_rate10","h_rest_days","a_rest_days"]},
        }

        if o.get("json_out"):
            with open(o["json_out"], "w") as f:
                json.dump(payload, f, indent=2)
            self.stdout.write(self.style.SUCCESS(f"Wrote {o['json_out']}"))
        else:
            self.stdout.write(json.dumps(payload, indent=2))
