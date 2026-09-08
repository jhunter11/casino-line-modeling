# Tennis: recorded calibration and house simulation

This report uses 26 selected settled matches. Selection and shared outcomes limit how broadly the results can be interpreted. The public export alone does not independently authenticate when each source forecast was first recorded.

## Forecast measurements

The model Brier score is 0.2128, compared with 0.1502 for the market baseline. Skill relative to the market is -0.417. The recorded calibration error is 0.1875, and the outcome base rate is 0.4615.

For the 15 observations with model probability at least 50 percent, the mean prediction is 78% and the outcome rate is 60%. For the other 11 observations, those values are 33% and 27%. These grouped summaries omit uncertainty intervals and do not establish performance on a new sample.

## Assumed house demand

The crowd rule allocates demand in proportion to market prices. The sharp rule directs demand toward favorable differences between the model and market prices. Neither rule measures actual customer flow or proves a worst-case exposure.

| Posted vig | Crowd hold | Sharp hold |
| --- | ---: | ---: |
| 0.0% | -63.6% | -99.6% |
| 4.5% | -56.6% | -91.0% |
| 7.0% | -52.9% | -86.5% |

The table reports simulated hold under those demand rules using recorded outcomes. It excludes many costs and operating constraints of a real book. The negative sharp-demand result supplies a reason to reject this pricing configuration for that scenario.
