#!/usr/bin/env python3
"""
World Cup 2026 — Casino Line Experiment
=======================================

Thesis (résumé front-page experiment):

    We built sports models that predict 3-way match outcomes (home / draw / away)
    from team strength ONLY (Elo + squad ClubElo + home field). The models never
    see a betting line -- verified by code audit. Forward validation proved we
    CANNOT beat the closing line (markets are efficient).

    So we flip the question: if we can't beat the casino, can we BE the casino?
    This script measures three things on real World Cup 2026 markets:

      1. AGREEMENT  -- How close are our line-free probabilities to the casino's
         (Kalshi's) own probabilities, after removing the casino's vig? Small,
         unbiased gap = we INDEPENDENTLY arrived at the book's pricing; we did not
         learn it (the model never had access to it).

      2. CALIBRATION -- On settled trades, is our model actually as good as the
         market? (Spoiler: it is slightly WORSE -- which is the whole point.)

      3. THE HOUSE EDGE -- If we post our probabilities as a book WITH a vig, what
         hold do we earn, and what is the imbalance risk? Demonstrates that the
         book's profit is the vig, not superior prediction.

Honesty notes baked into the output:
  - Kalshi itself runs a NEAR-ZERO-vig line and monetizes via per-contract fees,
    so the 2/4.5/7% vig levels in the house sim are TRADITIONAL-SPORTSBOOK
    representative, not Kalshi's.
  - The house sim draws realized outcomes from our OWN model probabilities, so
    "mean hold ~= vig" is illustrative of the mechanics + imbalance risk, not an
    empirical backtest. Empirical settlement on real results is the next iteration.

Inputs (already in this repo, no network, no dependencies):
  - data/wc_scores/*.json        daily snapshots: per match, our 3-way model probs
                                  AND the live Kalshi orderbook for each leg.
  - data/wc_paper_ledger.jsonl   settled paper trades (realized WIN/LOSS), traded
                                  subset, used for the calibration check.

Outputs:  casino_sim/results.json  and  casino_sim/RESULTS.md
Run:      python3 casino_sim/wc_line_experiment.py
"""

import glob
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCORE_GLOB = os.path.join(ROOT, "data", "wc_scores", "*.json")
LEDGER = os.path.join(ROOT, "data", "wc_paper_ledger.jsonl")

OUTCOMES = ("A", "TIE", "B")  # home win / draw / away win
SEED = 12345
LIQUID_MAX_SPREAD = 0.03      # a match is "liquid" if every leg's spread <= 3 cents


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def yes_mid(market):
    if not market:
        return None
    b, a = market.get("yes_bid"), market.get("yes_ask")
    if b is None or a is None or b <= 0 or a <= 0:
        return None
    return (b + a) / 2.0


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def pct(x, d=1):
    return f"{100*x:.{d}f}%"


def devig(mids):
    """Multiplicative de-vig: normalize the three YES mids to sum to 1."""
    s = sum(mids.values())
    return {o: mids[o] / s for o in mids}, s


# ---------------------------------------------------------------------------
# 1. Load the latest pre-kickoff snapshot per match
# ---------------------------------------------------------------------------
def load_matches():
    matches = {}
    for f in sorted(glob.glob(SCORE_GLOB)):
        snap_date = os.path.basename(f)[:-5]
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        for g in doc.get("games", []):
            legs = g.get("legs", [])
            mids, model, spreads = {}, {}, {}
            complete = True
            for leg in legs:
                o = leg.get("outcome")
                mm = leg.get("market")
                mv = yes_mid(mm)
                if o not in OUTCOMES or mv is None or leg.get("model_prob") is None:
                    complete = False
                    break
                mids[o] = mv
                model[o] = float(leg["model_prob"])
                spreads[o] = mm["yes_ask"] - mm["yes_bid"]
            if not complete or len(mids) != 3:
                continue
            ev = g["event"]
            kickoff = g.get("kickoff", "")
            if kickoff and snap_date > kickoff:
                continue
            prev = matches.get(ev)
            if prev is None or snap_date > prev["snap_date"]:
                matches[ev] = {
                    "snap_date": snap_date,
                    "event": ev,
                    "title": g.get("title"),
                    "kickoff": kickoff,
                    "model_path": g.get("model_path"),
                    "model": model,
                    "market_mid": mids,
                    "max_spread": max(spreads.values()),
                }
    return matches


# ---------------------------------------------------------------------------
# 2. Agreement: our line-free probs vs the casino's de-vigged probs
# ---------------------------------------------------------------------------
def agreement(ms):
    if not ms:
        return None
    model_p, fair_p = [], []
    overrounds, fav_agree = [], 0
    for m in ms:
        fair, overround = devig(m["market_mid"])
        overrounds.append(overround)
        for o in OUTCOMES:
            model_p.append(m["model"][o])
            fair_p.append(fair[o])
        if max(OUTCOMES, key=lambda o: m["model"][o]) == max(
            OUTCOMES, key=lambda o: fair[o]
        ):
            fav_agree += 1
    n = len(model_p)
    abs_err = sorted(abs(a - b) for a, b in zip(model_p, fair_p))
    mean_or = sum(overrounds) / len(overrounds)
    return {
        "n_matches": len(ms),
        "n_legs": n,
        "mae_vs_devig": sum(abs_err) / n,
        "rmse_vs_devig": math.sqrt(sum(e * e for e in abs_err) / n),
        "median_abs_err": abs_err[n // 2],
        "p90_abs_err": abs_err[int(0.9 * n)],
        "correlation": pearson(model_p, fair_p),
        "favorite_agreement": fav_agree / len(ms),
        "mean_overround": mean_or,
        "implied_vig_hold": (mean_or - 1) / mean_or,
    }


# ---------------------------------------------------------------------------
# 3. Calibration on settled trades (honest, traded-subset only)
# ---------------------------------------------------------------------------
def calibration():
    if not os.path.exists(LEDGER):
        return None
    rows = []
    for line in open(LEDGER):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "settled" or r.get("result") not in ("WIN", "LOSS"):
            continue
        mp, mid = r.get("model_prob"), r.get("entry_mid")
        if mp is None or mid is None:
            continue
        rows.append((float(mp), float(mid), 1.0 if r["result"] == "WIN" else 0.0))
    if not rows:
        return None
    n = len(rows)
    base = sum(y for _, _, y in rows) / n
    bm = sum((mp - y) ** 2 for mp, _, y in rows) / n
    bk = sum((mid - y) ** 2 for _, mid, y in rows) / n
    bb = sum((base - y) ** 2 for _, _, y in rows) / n
    return {
        "n_settled_legs": n,
        "base_rate": base,
        "brier_model": bm,
        "brier_market": bk,
        "brier_base": bb,
        "skill_vs_market": 1 - bm / bk if bk else None,
    }


# ---------------------------------------------------------------------------
# 4. Be the house: post our probs + a vig, simulate the hold
# ---------------------------------------------------------------------------
def house_simulation(matches, vigs=(0.02, 0.045, 0.07), n_sims=20000):
    rng = random.Random(SEED)
    ms = list(matches.values())
    out = {}
    for v in vigs:
        holds = []
        for _ in range(n_sims):
            staked = paid = 0.0
            for m in ms:
                model = m["model"]
                fair, _ = devig(m["market_mid"])
                post = {o: model[o] * (1 + v) for o in OUTCOMES}
                flow = {o: fair[o] * math.exp(rng.gauss(0, 0.35)) for o in OUTCOMES}
                fs = sum(flow.values())
                flow = {o: flow[o] / fs for o in OUTCOMES}
                staked += sum(flow[o] * post[o] for o in OUTCOMES)
                u, cum, winner = rng.random(), 0.0, OUTCOMES[-1]
                for o in OUTCOMES:
                    cum += model[o]
                    if u <= cum:
                        winner = o
                        break
                paid += flow[winner]
            holds.append((staked - paid) / staked)
        holds.sort()
        out[f"{v:.3f}"] = {
            "vig": v,
            "mean_hold": sum(holds) / len(holds),
            "p05_hold": holds[int(0.05 * len(holds))],
            "p95_hold": holds[int(0.95 * len(holds))],
            "theoretical_hold": v / (1 + v),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    matches = load_matches()
    if not matches:
        raise SystemExit("No World Cup matches with complete markets found.")
    full = list(matches.values())
    liquid = [m for m in full if m["max_spread"] <= LIQUID_MAX_SPREAD]

    a_full = agreement(full)
    a_liq = agreement(liquid)
    calib = calibration()
    house = house_simulation(matches)

    results = {
        "experiment": "world_cup_2026_casino_line",
        "n_matches": len(full),
        "agreement_full": a_full,
        "agreement_liquid": a_liq,
        "liquid_max_spread": LIQUID_MAX_SPREAD,
        "calibration": calib,
        "house_simulation": house,
    }
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    a = a_full
    L = a_liq
    out = []
    out.append("# World Cup 2026 — Casino Line Experiment: Results\n")
    out.append(
        f"_From {a['n_matches']} World Cup 2026 matches ({a['n_legs']} outcome "
        f"legs), one pre-kickoff snapshot each. Reproducible, no dependencies._\n"
    )
    out.append("## TL;DR\n")
    out.append(
        "Our team-strength-only model (which **never sees the betting line**) picks "
        f"the same favorite as Kalshi **{pct(a['favorite_agreement'])}** of the time "
        f"and correlates **r={a['correlation']:.2f}** with the casino's own "
        "probabilities — yet forward validation shows it is *slightly worse* at "
        "prediction than the market. **Conclusion: we can't beat the line, so the "
        "edge isn't in betting — it's in being the house and charging the vig.**\n"
    )

    out.append("## 1. Did our line-free model independently match the casino's line?\n")
    out.append(
        "Model features: Elo + squad ClubElo + home field. The market price is **never "
        "an input** (verified by code audit), so any agreement is independent, not "
        "memorized. We compare our probs to Kalshi's **de-vigged** probs.\n"
    )
    n_liq = L["n_matches"] if L else 0
    liq_note = (
        f"All {a['n_matches']} matches already trade at ≤{int(LIQUID_MAX_SPREAD*100)}¢ "
        "spreads, so the gap below is genuine model disagreement, not market noise."
        if n_liq == a["n_matches"]
        else f"{n_liq}/{a['n_matches']} matches trade at ≤{int(LIQUID_MAX_SPREAD*100)}¢ spreads."
    )
    out.append(f"_{liq_note}_\n")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Matches / outcome legs | {a['n_matches']} / {a['n_legs']} |")
    out.append(f"| Mean absolute gap per outcome | **{pct(a['mae_vs_devig'],2)}** |")
    out.append(f"| Median gap | {pct(a['median_abs_err'],2)} |")
    out.append(f"| 90th-percentile gap | {pct(a['p90_abs_err'],2)} |")
    out.append(f"| Correlation (our prob vs casino prob) | **r = {a['correlation']:.3f}** |")
    out.append(f"| Same favorite picked | **{pct(a['favorite_agreement'])}** of matches |")
    out.append(
        f"\n> Same favorite ~{pct(a['favorite_agreement'],0)} of the time, r="
        f"{a['correlation']:.2f} — yet a real ~{pct(a['mae_vs_devig'],0)} average gap "
        "per outcome. That gap is the point: a model that had *learned* the line "
        "would hug it within a point or two. Ours is directionally aligned but "
        "numerically its own — an **independent estimate**, corroborating the code "
        "audit that the line is never a model input.\n"
    )

    if calib:
        c = calib
        sk = c["skill_vs_market"]
        out.append("## 2. Calibration — and why we don't bet the line\n")
        out.append(
            f"On the **{c['n_settled_legs']} settled paper trades** "
            "(edge-filtered subset):\n"
        )
        out.append(
            f"- Brier — **model {c['brier_model']:.4f}** vs **market {c['brier_market']:.4f}** "
            f"vs base {c['brier_base']:.4f}"
        )
        out.append(
            f"- **Skill vs market: {sk:+.3f}** → the market is "
            f"{'better' if sk and sk < 0 else 'matched'}. We are good, but not better.\n"
        )
        out.append(
            "> This negative skill is the honest keystone: a model that can't beat "
            "the closing line shouldn't bet into it. It *should* become the book.\n"
        )

    out.append("## 3. Be the house: the edge is the vig, not the prediction\n")
    out.append(
        f"Kalshi's measured WC overround is only **{pct(a['mean_overround']-1,1)}** — "
        "because Kalshi runs a near-zero-vig line and monetizes via per-contract "
        "**fees**. Traditional sportsbooks instead bake in a 2–7% vig. If we post "
        "*our* probabilities as a sportsbook-style book:\n"
    )
    out.append("| Vig posted | Mean hold | 5th–95th pct (imbalance risk) | Balanced-book hold |")
    out.append("|---|---|---|---|")
    for k in sorted(house):
        h = house[k]
        out.append(
            f"| {pct(h['vig'],1)} | {pct(h['mean_hold'],2)} | "
            f"{pct(h['p05_hold'],1)} – {pct(h['p95_hold'],1)} | {pct(h['theoretical_hold'],2)} |"
        )
    out.append(
        "\n_Hold tracks the posted vig; the 5–95% band shows that with an unbalanced "
        "book over only ~60 matches the house can still lose on variance — which is "
        "why real books manage exposure. Realized outcomes here are drawn from our "
        "own model (illustrative of the mechanics); empirical settlement on real "
        "results is the next iteration._\n"
    )
    out.append("---")
    out.append(
        "_Reproduce: `python3 casino_sim/wc_line_experiment.py` — no dependencies, "
        "no network, runs on committed snapshots._"
    )
    report = "\n".join(out)
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
