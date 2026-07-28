#!/usr/bin/env python3
"""Consolidate the three models (World Cup, MLB, tennis) into one comparison
table + summary figure. Reads the per-sport house_backtest_*.json.
Run: python3 casino_sim/three_model_summary.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")

SPORTS = [("World Cup", "house_backtest_wc.json"),
          ("MLB (k-prop)", "house_backtest_mlb.json"),
          ("Tennis", "house_backtest_tennis.json")]


def load():
    out = []
    for name, fn in SPORTS:
        d = json.load(open(os.path.join(DATA, fn)))
        c = d["calibration"]
        h = d["house_hold"]["0.045"]
        out.append({
            "sport": name, "n": c["n"],
            "brier_model": c["brier_model"], "brier_market": c["brier_market"],
            "skill": c["skill_vs_market"], "ece": c["ece"],
            "fav_pred": c["favorites"]["mean_pred"], "fav_act": c["favorites"]["mean_actual"],
            "crowd": h["crowd"], "sharp": h["sharp"],
        })
    return out


def main():
    rows = load()
    L = ["# Three line-free models, tested as bettor AND book (blind, real outcomes)\n",
         "_Each model sets its probabilities pre-game, with no access to any betting line. "
         "We then check calibration against actual results and simulate running a book on the "
         "line. Samples are settled, selection-biased traded subsets — Ns stated._\n",
         "| Sport | N | Brier (model / market) | Skill vs market | Favorites pred→actual | "
         "House @4.5% (crowd) | House @4.5% (sharp) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['sport']} | {r['n']} | {r['brier_model']:.3f} / {r['brier_market']:.3f} | "
                 f"{r['skill']:+.3f} | {r['fav_pred']*100:.0f}% → {r['fav_act']*100:.0f}% | "
                 f"{r['crowd']*100:+.1f}% | {r['sharp']*100:+.1f}% |")
    L.append("\n**Takeaway:** none of the three beats the market on calibration (all skill ≤ 0), "
             "and none is sharp enough to profitably *be* the book against informed money "
             "(every sharp-flow column is deeply negative). MLB props come closest — near-market "
             "calibration, and the only line that skims the casual crowd at vig (+1.5%). World "
             "Cup and tennis lose. The honest verdict: **the market is hard to beat from either "
             "side, and we can show exactly how each model falls short.**\n")
    open(os.path.join(DATA, "THREE_MODEL_SUMMARY.md"), "w").write("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import vizstyle as vs
    vs.use()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    names = [r["sport"] for r in rows]
    x = list(range(len(names)))

    # Left — can this line profitably BE the book, against each kind of flow?
    w = 0.34
    b1 = a1.bar([i - w / 2 for i in x], [r["crowd"] * 100 for r in rows], w,
                label="casual flow", color=vs.SERIES[0])
    b2 = a1.bar([i + w / 2 for i in x], [r["sharp"] * 100 for r in rows], w,
                label="informed flow", color=vs.SERIES[1])
    vs.zero_line(a1, "break-even")
    vs.label_bars(a1, list(b1) + list(b2))
    a1.set_xticks(x)
    a1.set_xticklabels(names)
    a1.set_ylabel("House hold at 4.5% vig  (%)")
    a1.set_title("As the book: does the vig survive?")
    a1.legend(loc="lower left")
    a1.set_ylim(min(r["sharp"] for r in rows) * 100 - 18, 22)
    vs.horizontal_grid_only(a1)

    # Right — is the line better calibrated than the market's own price?
    b3 = a2.bar(x, [r["skill"] for r in rows], 0.5,
                color=[vs.GOOD if r["skill"] >= 0 else vs.BAD for r in rows])
    vs.zero_line(a2, "the market")
    vs.label_bars(a2, b3, fmt="{:+.3f}")
    a2.set_xticks(x)
    a2.set_xticklabels(names)
    a2.set_ylabel("Brier skill vs the market")
    a2.set_title("As the bettor: is the model sharper than the line?")
    a2.set_ylim(min(r["skill"] for r in rows) - 0.09, 0.06)
    vs.horizontal_grid_only(a2)

    fig.suptitle("Three from-scratch sports lines, blind-tested from both sides",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout(w_pad=4)
    vs.caption(fig, "Every bar that matters is negative: the models lose to the market "
                    "as bettors, and lose to informed money as the book.")
    fig.savefig(os.path.join(FIG, "three_model_summary.png"))
    plt.close(fig)

    print("\n".join(L))
    print("\nfigure -> casino_sim/figures/three_model_summary.png")


if __name__ == "__main__":
    main()
