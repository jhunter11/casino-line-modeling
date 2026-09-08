# Historical review of market inputs

The earlier project review examined whether model features or calibration targets used betting prices. Its notes cover three review passes: MLB models, World Cup models, and price references across the wider workspace.

This public note reports that historical review and its limits. It does not certify the full source-data lineage or independently reproduce every private-workspace check.

## Recorded findings

The review reported no market-price features in the selected World Cup Elo and squad model. It reported the same finding for the MLB game-winner ensemble, MLB strikeout model, and tennis feature list. It described calibration against outcomes and ensembles that combined model outputs.

The reported time checks included training on matches before kickoff, a pre-tournament squad snapshot, and MLB walk-forward splits by year. The underlying data producer schemas did not receive a separate comparison in that review.

A wider-workspace research model, `research/soccer_xgb_model.py`, used bookmaker-implied probabilities in its default feature pool. The notes describe an option to exclude those features. That research model is outside this curated public release and was not the model used for the casino simulation.

Using a market price as a declared feature is not inherently invalid for forecasting. It changes the claim being tested. Such a model cannot serve as evidence for a prediction built independently of market inputs.

## What output comparisons cannot prove

The World Cup model differs from market probabilities by about 11.5 percentage points on average. The correlation is about 0.77. The settled subset has negative Brier skill relative to the market baseline.

Those results do not prove absence of leakage. A model that sees prices can transform them, combine them with other features, or generalize poorly. It can therefore disagree with or perform worse than its price inputs.

Confidence in an independence claim requires inspection of selected features, targets, calibration, joins, timestamps, and data provenance. A behavioral comparison can help identify questions for that inspection but cannot replace it.

## Remaining limits

The historical notes mention World Cup 2022 squads as a proxy for 2026 rosters. They also mention proxy labels in some MLB settlement work. Those choices require accuracy and lineage checks before reuse. The public repository does not supply evidence sufficient to rule out every temporal or upstream-data error.

The September 8, 2026 documentation review removed the earlier blanket clean verdict. It also removed the claim that divergence from market prices proves independence. It did not change model code, stored probabilities, or outcome data.
