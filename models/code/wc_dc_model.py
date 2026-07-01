#!/usr/bin/env python3
"""wc_dc_model.py — score-based (Poisson / Dixon-Coles) 1X2 head for the WC sleeve.

DROP-IN replacement for the Gaussian draw curve (base.elo_predict_3way) used by the
live wc_scorer. It maps national-team Elo -> expected goals -> a bivariate-Poisson
(Dixon-Coles low-score corrected) score grid -> (p_home, p_draw, p_away), so draws
are modelled BY CONSTRUCTION instead of bolted on with a Gaussian + a hard Elo-gap
band-aid.

WHY (OOS backtest, research/wc_model/dc_vs_elo3way_backtest.py, 7,529 competitive
internationals 2015-2026, leak-free, params fit on a disjoint pre-2015 window):
    metric            elo3way(LIVE)   dixoncoles    delta
    log-loss          0.8856          0.8594        -2.96%
    3-way Brier       0.5116          0.5047        -1.35%
    RPS               0.3449          0.3385        -1.86%
    favourite ECE     0.0525          0.0081        6.5x better
Direction + magnitude stable across 2012/2015/2018/2021 splits (delta LL -0.026..-0.031).

INTEGRATION (NOT done here — leaves the running scorer untouched):
    in wc_scorer.model_probs_for_match, swap
        p_a, p_draw, p_b = base.elo_predict_3way(ra_eff, rb_eff)
    for
        import wc_dc_model
        p_a, p_draw, p_b = wc_dc_model.predict_3way(ra, rb, neutral=neutral)
    (pass the RAW ratings + neutral flag; this module applies the home bump itself.)

Params are fit on ALL competitive internationals through the fit date and cached to
data/wc_dc_params.json. Refitting on history that is strictly in the past of every
WC2026 fixture is leak-free for forward scoring. Refit weekly (see report cadence).

CLI:
  python3 harness/wc_dc_model.py fit            # (re)fit params -> data/wc_dc_params.json
  python3 harness/wc_dc_model.py compare [date] # live-curve vs DC on a wc_scores file
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "harness"))
import soccer_squad_model as base  # noqa: E402

PARAMS_PATH = ROOT / "data" / "wc_dc_params.json"
MAX_GOALS = 10
# Default params (fit 2026-06-16 on all competitive internationals through 2026-06-13;
# overwritten by `fit`). a0/beta/gamma for log-lambda, rho = DC low-score correction.
_DEFAULT = {"a0": 0.0738, "beta": 0.1604, "gamma": 0.2973, "rho": -0.03,
            "fit_through": "2026-06-13", "n_train": 14624}


def _load_params() -> dict:
    if PARAMS_PATH.exists():
        try:
            return json.loads(PARAMS_PATH.read_text())
        except Exception:  # noqa: BLE001
            pass
    return dict(_DEFAULT)


_P = _load_params()


def _pois(lam: float, kmax: int = MAX_GOALS):
    out = []
    for k in range(kmax + 1):
        out.append(math.exp(-lam + k * math.log(max(lam, 1e-9)) -
                            math.lgamma(k + 1)))
    return out


def _lambdas(elo_h: float, elo_a: float, neutral: bool, p=None):
    p = p or _P
    d = (elo_h - elo_a) / 100.0
    hadv = p["gamma"] if not neutral else 0.0
    lam_h = math.exp(p["a0"] + p["beta"] * d + hadv)
    lam_a = math.exp(p["a0"] - p["beta"] * d)
    return lam_h, lam_a


def predict_3way(elo_h: float, elo_a: float, neutral: bool = True,
                 p: dict | None = None) -> tuple:
    """Elo -> (p_home, p_draw, p_away) via Dixon-Coles bivariate Poisson.

    Pass RAW ratings; the home bump is applied internally via the fitted `gamma`
    (NOT an Elo offset), so do not pre-add HOME_ADV to elo_h."""
    p = p or _P
    lam_h, lam_a = _lambdas(elo_h, elo_a, neutral, p)
    ph = _pois(lam_h)
    pa = _pois(lam_a)
    rho = p.get("rho", 0.0)
    pH = pD = pA = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            m = ph[i] * pa[j]
            if rho:
                if i == 0 and j == 0:
                    m *= 1 - lam_h * lam_a * rho
                elif i == 0 and j == 1:
                    m *= 1 + lam_h * rho
                elif i == 1 and j == 0:
                    m *= 1 + lam_a * rho
                elif i == 1 and j == 1:
                    m *= 1 - rho
                m = max(m, 1e-12)
            if i > j:
                pH += m
            elif i == j:
                pD += m
            else:
                pA += m
    s = pH + pD + pA
    return pH / s, pD / s, pA / s


# --------------------------------------------------------------------------- #
def cmd_fit(_args):
    """Refit on all competitive internationals through today; cache params."""
    import numpy as np
    from scipy.optimize import minimize

    intl = base.load_intl()
    comp = sorted((m for m in intl if "friendly" not in m["tournament"].lower()),
                  key=lambda x: x["date"])
    R, games, rows = {}, {}, []
    K, HA = 30.0, 85.0

    def _gd(g):
        g = abs(g)
        return 1.0 if g <= 1 else (1.5 if g == 2 else (11 + g) / 8.0)

    def _tw(t):
        t = t.lower()
        if "friendly" in t: return 1.0
        if "qualif" in t: return 2.0
        if "world cup" in t or "nations" in t: return 3.0
        return 2.5

    last = comp[-1]["date"] if comp else ""
    for m in comp:
        h, a = m["home_team"], m["away_team"]
        rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
        neu = m.get("neutral", False)
        ha = 0.0 if neu else HA
        hg, ag = m["home_goals"], m["away_goals"]
        if games.get(h, 0) >= 25 and games.get(a, 0) >= 25:
            rows.append((rh, ra, neu, hg, ag))
        dr = (rh + ha) - ra
        E = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        W = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = K * _gd(hg - ag) * _tw(m.get("tournament", ""))
        R[h] = rh + k * (W - E)
        R[a] = ra + k * ((1 - W) - (1 - E))
        games[h] = games.get(h, 0) + 1
        games[a] = games.get(a, 0) + 1

    H = np.array([r[3] for r in rows], float)
    A = np.array([r[4] for r in rows], float)
    D = np.array([(r[0] - r[1]) / 100.0 for r in rows])
    NEU = np.array([0.0 if r[2] else 1.0 for r in rows])

    def nll(pp):
        a0, beta, gamma = pp
        lh = np.clip(np.exp(a0 + beta * D + gamma * NEU), 1e-6, 20)
        la = np.clip(np.exp(a0 - beta * D), 1e-6, 20)
        return -(H * np.log(lh) - lh + A * np.log(la) - la).sum()

    res = minimize(nll, [0.0, 0.3, 0.2], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 5000})
    a0, beta, gamma = (float(x) for x in res.x)
    p0 = {"a0": a0, "beta": beta, "gamma": gamma, "rho": 0.0}

    best_rho, best_ll = 0.0, -1e18
    for ri in range(-20, 13):
        rho = ri * 0.01
        ll = 0.0
        for rh, ra, neu, hg, ag in rows:
            lh, la = _lambdas(rh, ra, neu, {**p0})
            ph = _pois(lh, 6); pa = _pois(la, 6)
            x, y = min(hg, 6), min(ag, 6)
            prob = ph[x] * pa[y]
            if x == 0 and y == 0: prob *= 1 - lh * la * rho
            elif x == 0 and y == 1: prob *= 1 + lh * rho
            elif x == 1 and y == 0: prob *= 1 + la * rho
            elif x == 1 and y == 1: prob *= 1 - rho
            ll += math.log(max(prob, 1e-12))
        if ll > best_ll:
            best_ll, best_rho = ll, rho

    params = {"a0": a0, "beta": beta, "gamma": gamma, "rho": round(best_rho, 3),
              "fit_through": last, "n_train": len(rows)}
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    print(f"fit on {len(rows)} matches through {last}: {params}")
    print(f"  implied neutral equal-strength lambda = {math.exp(a0):.3f}/side, "
          f"home bump x{math.exp(gamma):.3f}, DC rho={best_rho:+.3f}")


def cmd_compare(args):
    """Side-by-side: live Gaussian curve vs DC on an existing wc_scores file."""
    d = args.date or __import__("datetime").date.today().isoformat()
    f = ROOT / "data" / "wc_scores" / f"{d}.json"
    if not f.exists():
        print(f"no scores file {f}")
        return
    payload = json.loads(f.read_text())
    print(f"params: {_P}\n")
    print(f"{'match':<30}{'gap':>6}  {'LIVE A/TIE/B':>20}   {'DC A/TIE/B':>20}  dTIE")
    print("-" * 92)
    for g in payload["games"]:
        ea, eb = g["elo_a"], g["elo_b"]
        neutral = g.get("neutral", True)
        # live curve replicate (host bump baked into rating, as the live scorer does)
        ha = 0.0 if neutral else 85.0
        la_, ld_, lb_ = base.elo_predict_3way(ea + ha, eb)
        da_, dd_, db_ = predict_3way(ea, eb, neutral=neutral)
        title = g["title"][:28]
        print(f"{title:<30}{abs(ea-eb):>6.0f}  "
              f"{la_:>5.2f}/{ld_:.2f}/{lb_:.2f}        "
              f"{da_:>5.2f}/{dd_:.2f}/{db_:.2f}      {dd_-ld_:+.2f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="WC Dixon-Coles 1X2 head")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit").set_defaults(func=cmd_fit)
    c = sub.add_parser("compare"); c.add_argument("date", nargs="?")
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
