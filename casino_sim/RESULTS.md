# World Cup line comparison and house simulation

The recorded comparison uses 62 World Cup 2026 matches and 186 outcome legs, with one pre-kickoff snapshot per match. The model uses Elo, squad ClubElo, and a home-field term. The historical feature review is described in [the audit note](LEAKAGE_AUDIT.md).

## Price comparison

The comparison normalizes the market probabilities. A narrow quoted spread does not by itself prove freshness, executable size, or an absence of market noise.

| Measurement | Recorded value |
| --- | ---: |
| Matches / outcome legs | 62 / 186 |
| Mean absolute probability gap | 11.55 percentage points |
| Median absolute gap | 8.31 percentage points |
| Ninetieth-percentile gap | 27.42 percentage points |
| Probability correlation | 0.765 |
| Same selected favorite | 80.6% of matches |

Agreement on favorites coexists with substantial probability differences in this sample. Neither agreement nor disagreement establishes whether a training pipeline had access to prices.

## Settled subset

The selected set of 43 settled paper decisions has a model Brier score of 0.1572 and a market score of 0.1428. Its base score is 0.2012 and its skill relative to the market is -0.101.

Lower Brier scores are better, so the model trails the market baseline on this subset. These selected, dependent observations do not establish a general result about market efficiency or the viability of running a sportsbook.

## Illustrative house simulation

This simulation draws outcomes from the model probabilities and varies the posted vig. The recorded hold statistics are:

| Posted vig | Mean hold | Fifth-to-ninety-fifth percentile range | Balanced-book hold |
| --- | --- | --- | --- |
| 2.0% | 2.04% | -9.3% to 13.7% | 1.96% |
| 4.5% | 4.24% | -6.9% to 15.6% | 4.31% |
| 7.0% | 6.51% | -4.4% to 17.6% | 6.54% |

The simulation illustrates margin and exposure under assumed probabilities and demand. It does not measure realized revenue or incorporate every operating cost. A positive mean in that setting cannot establish that the model can price a profitable book against informed customers.

Run `python -X utf8 casino_sim/wc_line_experiment.py` from the repository root. It uses committed snapshots without network access.
