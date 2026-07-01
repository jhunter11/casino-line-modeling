# Three line-free models, tested as bettor AND book (blind, real outcomes)

_Each model sets its probabilities pre-game, with no access to any betting line. We then check calibration against actual results and simulate running a book on the line. Samples are settled, selection-biased traded subsets — Ns stated._

| Sport | N | Brier (model / market) | Skill vs market | Favorites pred→actual | House @4.5% (crowd) | House @4.5% (sharp) |
|---|---|---|---|---|---|---|
| World Cup | 43 | 0.157 / 0.143 | -0.101 | 71% → 67% | -2.0% | -15.4% |
| MLB (k-prop) | 1364 | 0.160 / 0.158 | -0.017 | 77% → 83% | +1.5% | -21.7% |
| Tennis | 26 | 0.213 / 0.150 | -0.417 | 78% → 60% | -56.6% | -91.0% |

**Takeaway:** none of the three beats the market on calibration (all skill ≤ 0), and none is sharp enough to profitably *be* the book against informed money (every sharp-flow column is deeply negative). MLB props come closest — near-market calibration, and the only line that skims the casual crowd at vig (+1.5%). World Cup and tennis lose. The honest verdict: **the market is hard to beat from either side, and we can show exactly how each model falls short.**

