# Casino Line Modeling

A sports-model calibration study using committed predictions, outcomes, and market-price baselines.
The project covers World Cup match outcomes, MLB strikeout props, and tennis matches.
None of the evaluated models beats the market baseline on Brier score in its committed sample.

## Results

Lower Brier scores are better. Negative skill means the model trails the market-price baseline.

| Sample | Settled decisions | Model / market Brier | Brier skill |
| --- | ---: | --- | ---: |
| World Cup | 43 | 0.157 / 0.143 | -0.10 |
| MLB strikeout props | 1,364 | 0.160 / 0.158 | -0.02 |
| Tennis | 26 | 0.213 / 0.150 | -0.42 |

The book simulations also test a quoted line against casual and informed flow.
Their results depend on the stated flow and margin assumptions. They do not report realized sportsbook returns.
An earlier profitable simulation treated model probabilities as truth. That circular assumption does not establish an edge.

## Reproduce the analysis

```bash
python -m pip install -r requirements.txt
python explore.py 7
```

The analysis runs offline on committed data. It regenerates summary tables and figures.
On Windows PowerShell, set `$env:PYTHONUTF8='1'` before running the menu so its child processes can print the report symbols.
Individual scripts are under [casino_sim](casino_sim/):

```bash
python casino_sim/house_backtest.py
python casino_sim/house_backtest_mlb.py
python casino_sim/house_backtest_tennis.py
python casino_sim/book_compare.py
python casino_sim/three_model_summary.py
```

## Model demonstration

```bash
python demo.py
```

The demonstration loads committed model artifacts and runs sample inputs.
It includes XGBoost boosters for MLB game winners and tennis, plus an Elo and Dixon-Coles model for World Cup outcomes.
The MLB game-winner demonstration is separate from the strikeout-prop evaluation summarized above.
Running sample inference does not submit orders or establish current predictive performance.

See [models/code](models/code/) for inference code and [models](models/) for the saved artifacts.

## Leakage and provenance

The [leakage audit](casino_sim/LEAKAGE_AUDIT.md) records a review of the model inputs and evaluation path.
It distinguishes models built without bookmaker prices from a separate research model that used odds as a feature.
That research model is excluded from the shipped demonstration.

Prediction differences from market prices do not prove independence. Feature provenance and the actual scoring path require inspection.
The committed audit provides that inspection record for the examined versions.

The data came from public results and market sources, including ClubElo, Kalshi, Sporttery, and The Odds API.
This repository records the inputs used for the study. It does not promise current access, pricing, or redistribution rights for those sources.

## Limits

- The evaluated samples are selected traded subsets.
- World Cup and tennis samples are small.
- Some calibration bins contain only a few observations.
- The input set omits richer player, injury, and lineup information.
- Results for these versions and samples do not establish market efficiency or the limits of other models.

Historical reports retain their dated results and assumptions.

The related [agentic-quant-operator](https://github.com/jhunter11/agentic-quant-operator) repository contains the research workflow and promotion checks.
