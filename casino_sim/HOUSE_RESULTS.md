# World Cup 2026 — House Economics: Vig vs Line-Shading

_$\,50M house money · $50M handle across 62 matches · 20,000 Monte Carlo seasons · 4.5% posted vig. True outcomes drawn from our model._

**The question:** does the house profit only from the vig, or can it earn more by shading the line toward what the public over-bets?

## Answer, by how biased the public is

`gamma` = public favorite-bias. **gamma 1.0 = unbiased crowd; higher = the public over-bets favorites.** We compare an honest line (true probs + flat vig) to the profit-maximizing shaded line.

| Public bias gamma | Honest line hold | Shaded line hold | Extra from shading | Fav price moved | Shaded P(house loss) |
|---|---|---|---|---|---|
| 1.0 | 4.29% ($+2.15M) | 4.29% ($+2.14M) | **$-0.00M** | -0.00¢ | 5.9% |
| 1.3 | 4.29% ($+2.15M) | 4.65% ($+2.32M) | **$+0.18M** | +3.02¢ | 4.0% |
| 1.6 | 4.30% ($+2.15M) | 5.74% ($+2.87M) | **$+0.72M** | +5.81¢ | 2.0% |

## What it says

- **Unbiased crowd (gamma 1.0): shading adds essentially nothing** ($-0.00M). The profit-max line collapses to the flat vig — *the vig is the entire edge.* This is the mathematically forced result when bettors bet the true probabilities.

- **Biased crowd (gamma 1.6): shading adds real money** ($+0.72M on $50M handle, on top of the vig) by moving the favorite's price ~+5.8¢ — charging the crowd more for the side it loves.

- **The vig still dominates:** it earns ~$+2.15M vs shading's +0.18M increment — so your instinct is right, the vig is the bigger lever; shading is a real but secondary boost.

- **Risk (the surprising part):** at the profit-max shade, downside risk did NOT rise — P(house loss) moved 6.8% → 4.0% (gamma 1.3), because the extra margin sits on the favorite, the modal winner. Shading only becomes risk-*increasing* if pushed PAST the profit-max point to take a directional position; real books cap it at their risk appetite.

## Risk view (gamma 1.3, our model as truth)

- **Honest line:** mean $+2.15M · 5th–95th pct $-0.22M to $+4.54M · worst season $-3.48M (7.0% of bankroll) · P(loss) 6.8%
- **Shaded line:** mean $+2.32M · 5th–95th pct $+0.12M to $+4.50M · worst season $-3.05M (6.1% of bankroll) · P(loss) 4.0%

## Robustness: same conclusion if the *market* is right instead of us

Re-running with the market's de-vigged probs as truth (gamma 1.6): shading still adds $+1.04M — the vig-vs-shading conclusion does not depend on whose probabilities are correct.

---
_Reproduce: `python3 casino_sim/house_montecarlo.py` — no dependencies. Bias and sentiment parameters are explicit assumptions; the result is the *dependence on them*, not a single number._
