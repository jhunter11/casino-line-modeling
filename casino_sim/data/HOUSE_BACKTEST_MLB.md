# MLB — calibration & blind house backtest (real outcomes)

_Primary spine: **1364** settled KXMLBKS pitcher-strikeout prop contracts (binary). Model probability set pre-game, settled on the actual strikeout result. Edge-observed prop subset — selection-biased, deduped by ticker; stated honestly._

## Are our probabilities actually right? (vs TRUTH, not the casino)

- Brier — **model 0.1602** vs market 0.1576 (skill vs market -0.017) · ECE 0.0818 · base rate 0.5132
- **Favorites (model >=50%):** predicted 77% -> actually won 83%  (n=577)
- **Underdogs (model <50%):** predicted 18% -> actually won 28%  (n=787)

> Favorites won 6pp MORE than the model predicted — model under-prices favorites.

## Would a book on our line make money? (blind, settled on real results)

House hold per unit of action — **crowd** = flow proportional to the market price; **sharp** = bettors exploit wherever we price below the market (adversarial worst case):

| Posted vig | Crowd flow | Sharp flow |
|---|---|---|
| 0.0% | -2.7% | -33.8% |
| 4.5% | +1.5% | -21.7% |
| 7.0% | +3.5% | -19.3% |

> At a 4.5% vig the book KEEPS 1.5% against crowd flow, and LOSES 21.7% against sharp bettors — the honest verdict on whether our line is sharp enough to be the book.

## Secondary: clean game-winner calibration (KXMLBGAME, no market price)

_361 home-team game-winner contracts, real outcomes. These training-matrix rows carry **no captured market price**, so Brier-vs-market and the house sim are N/A here; model-vs-truth only._

- Brier model **0.2488** · ECE 0.0301 · base rate 0.518
- Favorites (model >=50%): predicted 56% -> won 54% (n=289)
- Underdogs (model <50%): predicted 47% -> won 44% (n=72)

> The game-winner model is **flat**: all predictions fall in ~[0.4,0.8] (the known ~52% calibration collapse — feature-starved, squashed). A book on this line has no edge to sell.

---
_Offline committed data only; no network. Mirrors casino_sim/house_backtest.py (World Cup)._
