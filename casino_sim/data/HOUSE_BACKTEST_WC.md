# World Cup: recorded calibration and house simulation

This report uses 43 selected settled contracts. Selection and shared outcomes limit how broadly the results can be interpreted. The public export alone does not independently authenticate when each source forecast was first recorded.

## Forecast measurements

The model Brier score is 0.1572, compared with 0.1428 for the market baseline. Skill relative to the market is -0.101. The recorded calibration error is 0.1271, and the outcome base rate is 0.2791.

For the 12 observations with model probability at least 50 percent, the mean prediction is 71% and the outcome rate is 67%. For the other 31 observations, those values are 22% and 13%. These grouped summaries omit uncertainty intervals and do not establish performance on a new sample.

## Assumed house demand

The crowd rule allocates demand in proportion to market prices. The sharp rule directs demand toward favorable differences between the model and market prices. Neither rule measures actual customer flow or proves a worst-case exposure.

| Posted vig | Crowd hold | Sharp hold |
| --- | ---: | ---: |
| 0.0% | -6.6% | -20.6% |
| 4.5% | -2.0% | -15.4% |
| 7.0% | +0.4% | -12.8% |

The table reports simulated hold under those demand rules using recorded outcomes. It excludes many costs and operating constraints of a real book. The negative sharp-demand result supplies a reason to reject this pricing configuration for that scenario.
