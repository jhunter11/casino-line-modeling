#!/usr/bin/env python3
"""
Blind house backtest + calibration — WORLD CUP (settled on REAL outcomes)
========================================================================

The honest test of "be the book": our model set each probability PRE-game (blind to
the result); we then (a) check how well-calibrated those probabilities were against
what actually happened, and (b) simulate running a book on our line and settling on
the REAL outcome — does it make money, or does miscalibration sink it?

This directly tests the observation that the model may under-price favorites: if so,
a book on our line bleeds when favorites win.

Data: data/wc_paper_ledger.jsonl settled rows — model_prob (pre-game), entry_mid
(market YES mid), result (WIN/LOSS = the contract resolved yes/no).
CAVEAT: this is the edge-filtered traded subset (selection-biased, small N). Stated
plainly; full-slate + tennis + MLB are the next extensions.

Run: python3 casino_sim/house_backtest.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEDGER = os.path.join(ROOT, "data", "wc_paper_ledger.jsonl")
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
VIGS = (0.0, 0.045, 0.07)


def load():
    rows = []
    for line in open(LEDGER):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "settled" or r.get("result") not in ("WIN", "LOSS"):
            continue
        p, mkt = r.get("model_prob"), r.get("entry_mid")
        if p is None or mkt is None:
            continue
        rows.append({"p": float(p), "mkt": float(mkt),
                     "y": 1.0 if r["result"] == "WIN" else 0.0})
    return rows


def calibration(rows):
    n = len(rows)
    base = sum(r["y"] for r in rows) / n
    bm = sum((r["p"] - r["y"]) ** 2 for r in rows) / n
    bk = sum((r["mkt"] - r["y"]) ** 2 for r in rows) / n
    # favorite/underdog skew vs TRUTH
    favs = [r for r in rows if r["p"] >= 0.5]
    dogs = [r for r in rows if r["p"] < 0.5]
    def mp(g): return sum(r["p"] for r in g) / len(g) if g else float("nan")
    def my(g): return sum(r["y"] for r in g) / len(g) if g else float("nan")
    # reliability deciles
    bins = []
    for lo in [i / 5 for i in range(5)]:
        hi = lo + 0.2
        g = [r for r in rows if lo <= r["p"] < hi or (hi == 1.0 and r["p"] == 1.0)]
        if g:
            bins.append({"lo": lo, "hi": hi, "n": len(g), "pred": mp(g), "obs": my(g)})
    ece = sum(b["n"] * abs(b["pred"] - b["obs"]) for b in bins) / n
    return {
        "n": n, "base_rate": round(base, 4),
        "brier_model": round(bm, 4), "brier_market": round(bk, 4),
        "skill_vs_market": round(1 - bm / bk, 3) if bk else None,
        "ece": round(ece, 4),
        "favorites": {"n": len(favs), "mean_pred": round(mp(favs), 4), "mean_actual": round(my(favs), 4)},
        "underdogs": {"n": len(dogs), "mean_pred": round(mp(dogs), 4), "mean_actual": round(my(dogs), 4)},
        "bins": bins,
    }


def house(rows, vig, flow):
    """Per contract: house posts q_yes=p*(1+vig), q_no=(1-p)*(1+vig). 1 unit of
    action split f_yes/f_no; settle on real y. flow='crowd' (∝ market) or 'sharp'
    (bettors take whichever side we price below the market = adverse selection)."""
    tot = 0.0
    for r in rows:
        p, mkt, y = r["p"], r["mkt"], r["y"]
        q_yes = min(max(p * (1 + vig), 1e-3), 0.999)
        q_no = min(max((1 - p) * (1 + vig), 1e-3), 0.999)
        if flow == "crowd":
            f_yes = mkt
        else:  # sharp: buy the cheap side relative to the market's fair price
            f_yes = 1.0 if q_yes < mkt else 0.0
        f_no = 1.0 - f_yes
        payout = (f_yes / q_yes) if y == 1 else (f_no / q_no)
        tot += 1.0 - payout            # collected 1 unit, paid `payout`
    return tot / len(rows)             # house hold per unit of action


def main():
    rows = load()
    if not rows:
        raise SystemExit("no settled WC rows")
    os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    cal = calibration(rows)
    bt = {f"{v:.3f}": {"crowd": round(house(rows, v, "crowd"), 4),
                       "sharp": round(house(rows, v, "sharp"), 4)} for v in VIGS}
    out = {"n": cal["n"], "calibration": cal, "house_hold": bt}
    json.dump(out, open(os.path.join(DATA, "house_backtest_wc.json"), "w"), indent=2)

    f, fa = cal["favorites"], cal["underdogs"]
    L = []
    L.append("# World Cup — calibration & blind house backtest (real outcomes)\n")
    L.append(f"_{cal['n']} settled contracts, model probability set pre-game, settled on the "
             "actual result. Edge-filtered subset — selection-biased, small N; stated honestly._\n")
    L.append("## Are our probabilities actually right? (vs TRUTH, not the casino)\n")
    L.append(f"- Brier — **model {cal['brier_model']}** vs market {cal['brier_market']} "
             f"(skill vs market {cal['skill_vs_market']:+}) · ECE {cal['ece']} · base rate {cal['base_rate']}")
    L.append(f"- **Favorites (model ≥50%):** predicted {f['mean_pred']*100:.0f}% → actually won "
             f"{f['mean_actual']*100:.0f}%  (n={f['n']})")
    L.append(f"- **Underdogs (model <50%):** predicted {fa['mean_pred']*100:.0f}% → actually won "
             f"{fa['mean_actual']*100:.0f}%  (n={fa['n']})")
    gap_fav = f["mean_actual"] - f["mean_pred"]
    L.append(f"\n> Favorites won {abs(gap_fav)*100:.0f}pp "
             f"{'MORE' if gap_fav>0 else 'LESS'} than the model predicted — "
             f"{'confirms under-confidence (model under-prices favorites)' if gap_fav>0 else 'model not under-confident here'}.\n")
    L.append("## Would a book on our line make money? (blind, settled on real results)\n")
    L.append("House hold per unit of action — **crowd** = bettors follow the market; "
             "**sharp** = bettors exploit wherever we price below the market (worst case):\n")
    L.append("| Posted vig | Crowd flow | Sharp flow |")
    L.append("|---|---|---|")
    for v in VIGS:
        b = bt[f"{v:.3f}"]
        L.append(f"| {v*100:.1f}% | {b['crowd']*100:+.1f}% | {b['sharp']*100:+.1f}% |")
    L.append("")
    crowd45 = bt["0.045"]["crowd"]; sharp45 = bt["0.045"]["sharp"]
    L.append(f"> At a 4.5% vig the book {'KEEPS' if crowd45>0 else 'LOSES'} "
             f"{abs(crowd45)*100:.1f}% against crowd flow, but {'still profits' if sharp45>0 else 'LOSES '+f'{abs(sharp45)*100:.1f}%'} "
             "against sharp bettors — the honest verdict on whether our line is sharp enough to be the book.\n")
    report = "\n".join(L)
    open(os.path.join(DATA, "HOUSE_BACKTEST_WC.md"), "w").write(report + "\n")

    _plot(cal)
    print(report)


def _plot(cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = cal["bins"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    ax.plot([b["pred"] for b in bins], [b["obs"] for b in bins], "o-", color="#2b6cb0",
            label="our model")
    for b in bins:
        ax.annotate(f"n={b['n']}", (b["pred"], b["obs"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel("Predicted probability (our model)")
    ax.set_ylabel("Actual win frequency")
    ax.set_title(f"World Cup reliability — Brier {cal['brier_model']} vs market {cal['brier_market']}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "wc_reliability.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
