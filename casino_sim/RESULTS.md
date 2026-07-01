# World Cup 2026 — Casino Line Experiment: Results

_From 62 World Cup 2026 matches (186 outcome legs), one pre-kickoff snapshot each. Reproducible, no dependencies._

## TL;DR

Our team-strength-only model (which **never sees the betting line**) picks the same favorite as Kalshi **80.6%** of the time and correlates **r=0.77** with the casino's own probabilities — yet forward validation shows it is *slightly worse* at prediction than the market. **Conclusion: we can't beat the line, so the edge isn't in betting — it's in being the house and charging the vig.**

## 1. Did our line-free model independently match the casino's line?

Model features: Elo + squad ClubElo + home field. The market price is **never an input** (verified by code audit), so any agreement is independent, not memorized. We compare our probs to Kalshi's **de-vigged** probs.

_All 62 matches already trade at ≤3¢ spreads, so the gap below is genuine model disagreement, not market noise._

| Metric | Value |
|---|---|
| Matches / outcome legs | 62 / 186 |
| Mean absolute gap per outcome | **11.55%** |
| Median gap | 8.31% |
| 90th-percentile gap | 27.42% |
| Correlation (our prob vs casino prob) | **r = 0.765** |
| Same favorite picked | **80.6%** of matches |

> Same favorite ~81% of the time, r=0.77 — yet a real ~12% average gap per outcome. That gap is the point: a model that had *learned* the line would hug it within a point or two. Ours is directionally aligned but numerically its own — an **independent estimate**, corroborating the code audit that the line is never a model input.

## 2. Calibration — and why we don't bet the line

On the **43 settled paper trades** (edge-filtered subset):

- Brier — **model 0.1572** vs **market 0.1428** vs base 0.2012
- **Skill vs market: -0.101** → the market is better. We are good, but not better.

> This negative skill is the honest keystone: a model that can't beat the closing line shouldn't bet into it. It *should* become the book.

## 3. Be the house: the edge is the vig, not the prediction

Kalshi's measured WC overround is only **0.7%** — because Kalshi runs a near-zero-vig line and monetizes via per-contract **fees**. Traditional sportsbooks instead bake in a 2–7% vig. If we post *our* probabilities as a sportsbook-style book:

| Vig posted | Mean hold | 5th–95th pct (imbalance risk) | Balanced-book hold |
|---|---|---|---|
| 2.0% | 2.04% | -9.3% – 13.7% | 1.96% |
| 4.5% | 4.24% | -6.9% – 15.6% | 4.31% |
| 7.0% | 6.51% | -4.4% – 17.6% | 6.54% |

_Hold tracks the posted vig; the 5–95% band shows that with an unbalanced book over only ~60 matches the house can still lose on variance — which is why real books manage exposure. Realized outcomes here are drawn from our own model (illustrative of the mechanics); empirical settlement on real results is the next iteration._

---
_Reproduce: `python3 casino_sim/wc_line_experiment.py` — no dependencies, no network, runs on committed snapshots._
