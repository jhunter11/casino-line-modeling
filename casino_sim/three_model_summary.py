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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    names = [r["sport"] for r in rows]
    x = range(len(names))

    # left: house hold @4.5% crowd vs sharp
    w = 0.36
    a1.bar([i - w/2 for i in x], [r["crowd"]*100 for r in rows], w, label="crowd flow", color="#3182ce")
    a1.bar([i + w/2 for i in x], [r["sharp"]*100 for r in rows], w, label="sharp flow", color="#e53e3e")
    a1.axhline(0, color="#333", lw=1)
    a1.set_xticks(list(x)); a1.set_xticklabels(names)
    a1.set_ylabel("House hold (%) at 4.5% vig")
    a1.set_title("Run a book on our line: does it profit?")
    a1.legend(fontsize=8)

    # right: Brier skill vs market (0 = market; negative = worse)
    a2.bar(list(x), [r["skill"] for r in rows],
           color=["#38a169" if r["skill"] >= 0 else "#dd6b20" for r in rows])
    a2.axhline(0, color="#333", lw=1)
    a2.set_xticks(list(x)); a2.set_xticklabels(names)
    a2.set_ylabel("Brier skill vs market")
    a2.set_title("Calibration vs the market (0 = market, <0 = worse)")
    fig.suptitle("Three from-scratch sports lines — honest blind evaluation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "three_model_summary.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(L))
    print("\nfigure -> casino_sim/figures/three_model_summary.png")


if __name__ == "__main__":
    main()
