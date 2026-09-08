# World Cup house simulation: vig and line shading

This recorded simulation uses 62 matches, $50 million of handle, a $50 million bankroll, and 20,000 simulated seasons. It assumes a 4.5 percent posted vig and draws outcomes from the model probabilities.

The demand parameter `gamma` controls favorite bias. A value of 1.0 represents the model's unbiased-demand setting. Higher values increase the simulated preference for favorites.

| Gamma | Flat-vig hold | Shaded-line hold | Additional simulated profit | Average favorite-price change | Shaded probability of loss |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 4.29% ($2.15M) | 4.29% ($2.14M) | Approximately $0 | Approximately 0 cents | 5.9% |
| 1.3 | 4.29% ($2.15M) | 4.65% ($2.32M) | $0.18M | +3.02 cents | 4.0% |
| 1.6 | 4.30% ($2.15M) | 5.74% ($2.87M) | $0.72M | +5.81 cents | 2.0% |

Under this demand function, shading adds little at gamma 1.0 and more as favorite bias increases. This is a result of the assumed demand and outcome models. It does not measure actual customer behavior or establish the profit of a real sportsbook.

## Recorded risk at gamma 1.3

The flat-vig simulation has a mean profit of $2.15 million and a 6.8 percent probability of loss. Its fifth-to-ninety-fifth percentile range is -$0.22 million to $4.54 million. The worst simulated season loses $3.48 million.

The shaded simulation has a mean profit of $2.32 million and a 4.0 percent probability of loss. Its corresponding range is $0.12 million to $4.50 million. The worst simulated season loses $3.05 million.

These finite simulated tails are not loss limits. Other demand patterns, correlated outcomes, or inaccurate probabilities can change both margin and risk.

## Alternative outcome assumption

The recorded run using normalized market probabilities as truth adds $1.04 million from shading at gamma 1.6. That comparison tests a second probability assumption. It does not establish that the conclusion holds for arbitrary pricing errors or demand responses.

Run `python -X utf8 casino_sim/house_montecarlo.py` from the repository root to inspect the simulation. Review its bias, sentiment, pricing, and outcome assumptions with the result.
