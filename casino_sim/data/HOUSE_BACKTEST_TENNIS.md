# Tennis — calibration & blind house backtest (real outcomes)

_26 settled matches (one binary row per match, player-1-centric), model probability set pre-match, settled on the actual result. Subset of the full slate — only matches that later settled get an outcome, so N is small and skews toward traded events; stated honestly._

## Are our probabilities actually right? (vs TRUTH, not the casino)

- Brier — **model 0.2128** vs market 0.1502 (skill vs market -0.417) · ECE 0.1875 · base rate 0.4615
- **Favorites (model ≥50%):** predicted 78% → actually won 60%  (n=15)
- **Underdogs (model <50%):** predicted 33% → actually won 27%  (n=11)

> Favorites won 18pp LESS than the model predicted — model is OVER-confident on favorites (over-prices them).

## Would a book on our line make money? (blind, settled on real results)

House hold per unit of action — **crowd** = bettors follow the market; **sharp** = bettors exploit wherever we price below the market (worst case):

| Posted vig | Crowd flow | Sharp flow |
|---|---|---|
| 0.0% | -63.6% | -99.6% |
| 4.5% | -56.6% | -91.0% |
| 7.0% | -52.9% | -86.5% |

> At a 4.5% vig the book LOSES 56.6% against crowd flow, and LOSES 91.0% against sharp bettors — the honest verdict on whether our tennis line is sharp enough to be the book.

