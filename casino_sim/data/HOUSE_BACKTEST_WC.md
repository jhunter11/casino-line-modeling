# World Cup — calibration & blind house backtest (real outcomes)

_43 settled contracts, model probability set pre-game, settled on the actual result. Edge-filtered subset — selection-biased, small N; stated honestly._

## Are our probabilities actually right? (vs TRUTH, not the casino)

- Brier — **model 0.1572** vs market 0.1428 (skill vs market -0.101) · ECE 0.1271 · base rate 0.2791
- **Favorites (model ≥50%):** predicted 71% → actually won 67%  (n=12)
- **Underdogs (model <50%):** predicted 22% → actually won 13%  (n=31)

> Favorites won 4pp LESS than the model predicted — model not under-confident here.

## Would a book on our line make money? (blind, settled on real results)

House hold per unit of action — **crowd** = bettors follow the market; **sharp** = bettors exploit wherever we price below the market (worst case):

| Posted vig | Crowd flow | Sharp flow |
|---|---|---|
| 0.0% | -6.6% | -20.6% |
| 4.5% | -2.0% | -15.4% |
| 7.0% | +0.4% | -12.8% |

> At a 4.5% vig the book LOSES 2.0% against crowd flow, but LOSES 15.4% against sharp bettors — the honest verdict on whether our line is sharp enough to be the book.

