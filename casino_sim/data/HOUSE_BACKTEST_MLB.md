# MLB strikeout props: recorded calibration and house simulation

This report uses 1,364 selected settled pitcher-strikeout contracts. Selection and shared outcomes limit how broadly the results can be interpreted. The public export alone does not independently authenticate when each source forecast was first recorded.

## Forecast measurements

The model Brier score is 0.1602, compared with 0.1576 for the market baseline. Skill relative to the market is -0.017. The recorded calibration error is 0.0818, and the outcome base rate is 0.5132.

For the 577 observations with model probability at least 50 percent, the mean prediction is 77% and the outcome rate is 83%. For the other 787 observations, those values are 18% and 28%. These grouped summaries omit uncertainty intervals and do not establish performance on a new sample.

## Assumed house demand

The crowd rule allocates demand in proportion to market prices. The sharp rule directs demand toward favorable differences between the model and market prices. Neither rule measures actual customer flow or proves a worst-case exposure.

| Posted vig | Crowd hold | Sharp hold |
| --- | ---: | ---: |
| 0.0% | -2.7% | -33.8% |
| 4.5% | +1.5% | -21.7% |
| 7.0% | +3.5% | -19.3% |

The table reports simulated hold under those demand rules using recorded outcomes. It excludes many costs and operating constraints of a real book. The negative sharp-demand result supplies a reason to reject this pricing configuration for that scenario.

## Separate game-winner sample

A second sample contains 361 home-team game-winner rows without captured market prices. Its model Brier score is 0.2488, calibration error is 0.0301, and base rate is 0.518. The 289 model favorites average 56 percent predicted probability and 54 percent wins. The other 72 observations average 47 percent predicted probability and 44 percent wins.

That sample supports a model-versus-outcome comparison. It cannot support a market-price comparison or the same house-demand simulation without the missing quotes.
