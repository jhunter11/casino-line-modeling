# Leakage Audit — Does the model just re-learn the betting line?

**Question this answers:** Are the win probabilities produced by these models *independent
predictions from team strength*, or did the model secretly fit to the betting line (in which
case it would be "predicting" a number it was already handed)? For a model whose whole claim
is "we reproduce the casino's odds without ever seeing them," this is the only question that
matters.

**Method.** Three independent, *adversarial* code auditors were run — each instructed to
assume the line **is** leaking and to try to prove it, reading the actual code (not comments):
1. MLB model family — features, training, calibration, temporal hygiene.
2. World Cup model family (the casino-sim model) — features, Poisson/Dixon-Coles fits, blend.
3. Codebase-wide grep sweep — every site that touches a market price, classified as
   *feature/fit* (leakage) vs *comparison-only* (legitimate post-prediction edge/CLV/PnL).

---

## Verdict

| Model | Role | Verdict |
|---|---|---|
| **World Cup 3-way (Elo + squad-DC)** | **powers the casino simulation** | ✅ **CLEAN** |
| MLB game-winner (Elo/XGBoost ensemble + Platt) | deployed | ✅ CLEAN |
| MLB strikeout props (per-PA Poisson) | deployed | ✅ CLEAN |
| Tennis structural model | deployed | ✅ CLEAN (17 odds-free features) |
| `research/soccer_xgb_model.py` | **research-only, never deployed** | ⚠️ uses market features by default — **excluded / quarantined** |

### The four leakage vectors, checked

1. **Feature leakage** — No deployed model's feature vector contains a line, Kalshi price,
   bid/ask, mid, implied probability, or odds. WC features = Elo (from match results only),
   squad ClubElo (pre-tournament snapshot), home-field constant, goals-only Poisson/DC params.
2. **Calibration leakage (the likeliest hiding spot)** — Every probability calibrator
   (Platt / logistic / isotonic) is fit against the **realized outcome** (`home_win`, match
   result, goals), *never* against the market price. Confirmed at every fit site.
3. **Model + market blending** — No final output probability is a `w·model + (1−w)·market`
   mix. The two ensembles that exist are **model+model** (XGB+MLP; and v5/v7/v8/v9 averaging),
   not model+market.
4. **Temporal leakage** — Elo trains only on matches strictly *before* kickoff; the ClubElo
   snapshot predates the tournament; MLB walk-forward trains on years `< Y`, tests on `Y`.

### Behavioral corroboration (black-box test)

The code audit is confirmed by the model's *behavior* on real markets — a leaked model is
mathematically forced to track the line:

| If the model had learned the line | What our model actually does |
|---|---|
| Mean gap vs line ≈ 0–2pp | **11.5pp** mean gap per outcome |
| Brier skill vs market ≈ large **positive** | **−0.10** (slightly *worse* than the market) |
| Correlation ≈ 0.99 | **0.77** |

A model that copies the line cannot be *worse* than the line. Ours is. Code audit and
behavior agree: the probabilities are derived independently.

---

## The one finding — disclosed in full

The codebase-wide sweep found exactly one model that ingests the line:
`research/soccer_xgb_model.py` places de-vigged bookmaker-implied probabilities
(`mkt_ph / mkt_pd / mkt_pa`) into its **default** training pool (opt-out only via a
`--no-market` flag). This is genuine feature leakage **and we are disclosing it rather than
hiding it**, because:

- It is **research-only**: wired to no cron, no registry, no trader; it sits in the backlog
  and was never promoted to live use.
- The project's own notes already flagged it: the market features "heavily dominate SHAP …
  risks the model merely re-discovering the market rather than finding independent edge."
- It is **not** the model used in the casino simulation or any deployed scorer.

It is therefore **excluded from the curated modeling repository** (or, if included, shipped
only as a labeled *negative control* demonstrating exactly the failure mode the deployed
models avoid).

## Scope & honest caveats

- This audit verifies **code paths**. It confirms feature lists are explicit and price-free,
  so a stray price column in an upstream cache would not be *selected* — but the raw parquet
  schemas were not separately diffed against an external producer.
- Unrelated, non-leakage limitations disclosed elsewhere: WC2026 rosters use WC2022 squads as
  a proxy; some labeling proxies in MLB settlement. These affect accuracy, not independence.

**Bottom line:** the casino-simulation model predicts win probability *independently of the
betting line* — verified by adversarial code audit and confirmed by its measurable divergence
from the line.
