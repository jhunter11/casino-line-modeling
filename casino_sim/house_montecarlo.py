#!/usr/bin/env python3
"""
World Cup 2026 — House Economics Monte Carlo
============================================

Question (the one that makes this interesting):

    A book vigs its odds to make money. But is the vig the ONLY source of profit, or
    can the house earn MORE by *shading* the line — deliberately mispricing toward the
    side the public over-bets, so it pays out less when the crowd is wrong?

We answer it with a real Monte Carlo: $50M of house money, randomized betting flow
driven by public bias + per-match sentiment, settled on realized outcomes. We compare
an HONEST line (true probs + a flat vig) against the PROFIT-MAXIMIZING shaded line, and
decompose how much profit comes from the vig vs from shading.

Mechanics (Kalshi-style binary contracts, $1 settlement):
  - For outcome o the house posts a YES price q(o) in (0,1); posted prices sum to 1+vig.
  - Bettors wager handle H on the match, split by flow f(o); money on o = H·f(o), which
    buys H·f(o)/q(o) contracts paying $1 if o occurs.
  - House profit on the match = H·(1 − f(w)/q(w)) for realized winner w.
      * balanced flow f = true prob, honest prices  →  profit = H·vig/(1+vig)  (pure vig)
      * if the crowd piles onto a side and it wins, f(w)/q(w) > 1 → the house LOSES that
        match. That imbalance risk is why shading is a trade-off, not free money.

The profit-maximizing allocation of a fixed overround across outcomes (Lagrange on
maximize −Σ p_true·f/q s.t. Σq = 1+vig) is q*(o) ∝ sqrt(p_true(o)·f(o)). When the public
is unbiased (f = p_true) this collapses to q ∝ p_true — i.e. shading reduces to the flat
vig and adds nothing. The benefit appears ONLY to the extent the public is biased.

Honest caveats:
  - "True" outcome probabilities are OUR model's probs (the house's best estimate). A
    robustness pass also runs with the market's de-vigged probs as truth.
  - The public-bias parameter gamma and the sentiment noise are ASSUMPTIONS; results are
    conditional on them, which is the whole point — we show the answer flips with bias.
  - Stylized model, not a production sportsbook risk engine.

Run:  python3 casino_sim/house_montecarlo.py
"""

import json
import math
import os
import random

from wc_line_experiment import load_matches, devig, OUTCOMES

HERE = os.path.dirname(os.path.abspath(__file__))

BANKROLL = 50_000_000          # house money at risk
TOTAL_HANDLE = 50_000_000      # total wagered across the slate (turnover ~1x bankroll)
N_SIMS = 20_000
SEED = 7
VIG = 0.045                    # posted overround (traditional-sportsbook representative)
GAMMAS = (1.0, 1.3, 1.6)       # public favorite-bias: 1.0 = unbiased, >1 over-bets favorites
SENT_SIGMA = 0.30              # per-match idiosyncratic sentiment (book imbalance) noise


def scale_to_overround(base, total):
    """Scale a positive vector so it sums to `total` (=1+vig); clamp prices < 1."""
    s = sum(base.values())
    q = {o: base[o] / s * total for o in base}
    # a YES price can't reach 1; clamp and renormalize the tiny residual
    for o in q:
        q[o] = min(q[o], 0.985)
    return q


def expected_flow(consensus, gamma):
    """Public's systematic appetite: over-weight by prob^gamma, normalized."""
    raw = {o: consensus[o] ** gamma for o in OUTCOMES}
    s = sum(raw.values())
    return {o: raw[o] / s for o in OUTCOMES}


def honest_prices(p_true):
    return scale_to_overround({o: p_true[o] for o in OUTCOMES}, 1 + VIG)


def shaded_prices(p_true, exp_flow):
    """Profit-max prices vs the EXPECTED (known, systematic) bias: q ∝ sqrt(p_true·flow)."""
    a = {o: math.sqrt(max(p_true[o] * exp_flow[o], 1e-12)) for o in OUTCOMES}
    return scale_to_overround(a, 1 + VIG)


def run(matches, truth_key, gamma):
    """Monte Carlo for one (truth source, gamma). Returns honest vs shaded stats."""
    rng = random.Random(SEED)
    ms = list(matches.values())
    h_per = TOTAL_HANDLE / len(ms)

    # Precompute per-match: truth probs, consensus, expected flow, both price sets.
    plan = []
    avg_fav_shade = []
    for m in ms:
        fair, _ = devig(m["market_mid"])
        p_true = m["model"] if truth_key == "model" else fair
        # The public's UNBIASED anchor is the truth itself, so gamma=1 is a balanced
        # book = pure vig (no smuggled model-vs-public information edge). gamma>1 then
        # isolates the favorite-shading effect, which is the question being asked.
        consensus = p_true
        ef = expected_flow(consensus, gamma)
        qh = honest_prices(p_true)
        qs = shaded_prices(p_true, ef)
        fav = max(OUTCOMES, key=lambda o: p_true[o])
        avg_fav_shade.append(qs[fav] - qh[fav])  # how much the favorite's price is moved
        plan.append((p_true, ef, qh, qs))

    def simulate(price_key):
        profits = []
        for _ in range(N_SIMS):
            total = 0.0
            for p_true, ef, qh, qs in plan:
                q = qh if price_key == "honest" else qs
                # realized flow = expected flow * idiosyncratic sentiment, renormalized
                rf = {o: ef[o] * math.exp(rng.gauss(0, SENT_SIGMA)) for o in OUTCOMES}
                fs = sum(rf.values())
                rf = {o: rf[o] / fs for o in OUTCOMES}
                # realized outcome ~ true probs
                u, cum, w = rng.random(), 0.0, OUTCOMES[-1]
                for o in OUTCOMES:
                    cum += p_true[o]
                    if u <= cum:
                        w = o
                        break
                total += h_per * (1 - rf[w] / q[w])
            profits.append(total)
        profits.sort()
        n = len(profits)
        mean = sum(profits) / n
        return {
            "mean_profit": mean,
            "mean_hold_pct": mean / TOTAL_HANDLE,
            "p05_profit": profits[int(0.05 * n)],
            "p95_profit": profits[int(0.95 * n)],
            "worst_profit": profits[0],
            "prob_loss": sum(1 for p in profits if p < 0) / n,
            "max_drawdown_vs_bankroll": -profits[0] / BANKROLL if profits[0] < 0 else 0.0,
        }

    return {
        "honest": simulate("honest"),
        "shaded": simulate("shaded"),
        "avg_favorite_shade_cents": 100 * (sum(avg_fav_shade) / len(avg_fav_shade)),
    }


def main():
    matches = load_matches()
    if not matches:
        raise SystemExit("No matches found.")
    results = {
        "params": {
            "bankroll": BANKROLL, "total_handle": TOTAL_HANDLE, "n_sims": N_SIMS,
            "vig": VIG, "gammas": list(GAMMAS), "sentiment_sigma": SENT_SIGMA,
            "n_matches": len(matches),
        },
        "by_truth": {},
    }
    for truth_key in ("model", "market"):
        results["by_truth"][truth_key] = {f"{g:.1f}": run(matches, truth_key, g) for g in GAMMAS}

    with open(os.path.join(HERE, "house_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    def money(x):
        return f"${x/1e6:+.2f}M"

    out = []
    out.append("# World Cup 2026 — House Economics: Vig vs Line-Shading\n")
    out.append(
        f"_$\\,{BANKROLL/1e6:.0f}M house money · ${TOTAL_HANDLE/1e6:.0f}M handle across "
        f"{results['params']['n_matches']} matches · {N_SIMS:,} Monte Carlo seasons · "
        f"{int(VIG*100*10)/10}% posted vig. True outcomes drawn from our model._\n"
    )
    out.append(
        "**The question:** does the house profit only from the vig, or can it earn more by "
        "shading the line toward what the public over-bets?\n"
    )
    out.append("## Answer, by how biased the public is\n")
    out.append(
        "`gamma` = public favorite-bias. **gamma 1.0 = unbiased crowd; higher = the public "
        "over-bets favorites.** We compare an honest line (true probs + flat vig) to the "
        "profit-maximizing shaded line.\n"
    )
    M = results["by_truth"]["model"]
    out.append("| Public bias gamma | Honest line hold | Shaded line hold | Extra from shading | Fav price moved | Shaded P(house loss) |")
    out.append("|---|---|---|---|---|---|")
    for g in GAMMAS:
        r = M[f"{g:.1f}"]
        h, s = r["honest"], r["shaded"]
        extra = s["mean_profit"] - h["mean_profit"]
        out.append(
            f"| {g:.1f} | {h['mean_hold_pct']*100:.2f}% ({money(h['mean_profit'])}) | "
            f"{s['mean_hold_pct']*100:.2f}% ({money(s['mean_profit'])}) | "
            f"**{money(extra)}** | {r['avg_favorite_shade_cents']:+.2f}¢ | "
            f"{s['prob_loss']*100:.1f}% |"
        )
    out.append("")
    g0 = M[f"{GAMMAS[0]:.1f}"]
    extra0 = g0["shaded"]["mean_profit"] - g0["honest"]["mean_profit"]
    gN = M[f"{GAMMAS[-1]:.1f}"]
    extraN = gN["shaded"]["mean_profit"] - gN["honest"]["mean_profit"]
    out.append("## What it says\n")
    out.append(
        f"- **Unbiased crowd (gamma 1.0): shading adds essentially nothing** "
        f"({money(extra0)}). The profit-max line collapses to the flat vig — *the vig is "
        f"the entire edge.* This is the mathematically forced result when bettors bet the "
        f"true probabilities.\n"
    )
    out.append(
        f"- **Biased crowd (gamma {GAMMAS[-1]:.1f}): shading adds real money** "
        f"({money(extraN)} on ${TOTAL_HANDLE/1e6:.0f}M handle, on top of the vig) by moving "
        f"the favorite's price ~{gN['avg_favorite_shade_cents']:+.1f}¢ — charging the crowd "
        f"more for the side it loves.\n"
    )
    rr = M["1.3"]
    plh, pls = rr["honest"]["prob_loss"] * 100, rr["shaded"]["prob_loss"] * 100
    out.append(
        f"- **The vig still dominates:** it earns ~{money(M['1.3']['honest']['mean_profit'])} "
        f"vs shading's +{money(rr['shaded']['mean_profit']-rr['honest']['mean_profit'])[2:]} "
        "increment — so your instinct is right, the vig is the bigger lever; shading is a "
        "real but secondary boost.\n"
    )
    out.append(
        f"- **Risk (the surprising part):** at the profit-max shade, downside risk did NOT "
        f"rise — P(house loss) moved {plh:.1f}% → {pls:.1f}% (gamma 1.3), because the extra "
        f"margin sits on the favorite, the modal winner. Shading only becomes risk-"
        f"*increasing* if pushed PAST the profit-max point to take a directional position; "
        f"real books cap it at their risk appetite.\n"
    )
    out.append("## Risk view (gamma 1.3, our model as truth)\n")
    r = M["1.3"]
    for name, lab in (("honest", "Honest line"), ("shaded", "Shaded line")):
        d = r[name]
        out.append(
            f"- **{lab}:** mean {money(d['mean_profit'])} · 5th–95th pct "
            f"{money(d['p05_profit'])} to {money(d['p95_profit'])} · worst season "
            f"{money(d['worst_profit'])} ({d['max_drawdown_vs_bankroll']*100:.1f}% of bankroll) · "
            f"P(loss) {d['prob_loss']*100:.1f}%"
        )
    out.append("")
    out.append("## Robustness: same conclusion if the *market* is right instead of us\n")
    Mk = results["by_truth"]["market"]
    rb = Mk[f"{GAMMAS[-1]:.1f}"]
    extra_mk = rb["shaded"]["mean_profit"] - rb["honest"]["mean_profit"]
    out.append(
        f"Re-running with the market's de-vigged probs as truth (gamma {GAMMAS[-1]:.1f}): "
        f"shading still adds {money(extra_mk)} — the vig-vs-shading conclusion does not "
        f"depend on whose probabilities are correct.\n"
    )
    out.append("---")
    out.append(
        "_Reproduce: `python3 casino_sim/house_montecarlo.py` — no dependencies. "
        "Bias and sentiment parameters are explicit assumptions; the result is the "
        "*dependence on them*, not a single number._"
    )
    report = "\n".join(out)
    with open(os.path.join(HERE, "HOUSE_RESULTS.md"), "w") as fh:
        fh.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
