# Three models against market-price baselines

These recorded results compare model forecasts with outcomes and simulate a book using two assumed demand rules. The samples are selected settled subsets. They are small or dependent enough that the reported point estimates need qualification.

| Sport | Rows | Brier model / market | Skill relative to market | Favorite prediction / outcome rate | Hold at 4.5% vig, crowd | Hold at 4.5% vig, sharp |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| World Cup | 43 | 0.157 / 0.143 | -0.101 | 71% / 67% | -2.0% | -15.4% |
| MLB strikeout props | 1,364 | 0.160 / 0.158 | -0.017 | 77% / 83% | +1.5% | -21.7% |
| Tennis | 26 | 0.213 / 0.150 | -0.417 | 78% / 60% | -56.6% | -91.0% |

All three model Brier scores trail their market baselines in these samples. Under the specified sharp-demand simulation, all three books lose. MLB shows a positive hold only under the crowd-demand assumption at this vig.

The simulated demand rules do not represent measured customer flow or a proven worst case. These comparisons establish no trading edge, real operating profit, or universal claim about market efficiency.
