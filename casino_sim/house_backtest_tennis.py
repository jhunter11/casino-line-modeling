#!/usr/bin/env python3
"""
Blind house backtest + calibration — TENNIS (settled on REAL outcomes)
======================================================================

The honest test of "be the book": our tennis model set each probability PRE-match
(blind to the result); we then (a) check how well-calibrated those probabilities were
against what actually happened, and (b) simulate running a book on our line and
settling on the REAL outcome — does it make money, or does miscalibration sink it?

Tennis is a 2-OUTCOME market (player 1 wins / player 2 wins) — simpler than the World
Cup 3-way, so there is no draw to handle. We model everything from player 1's point of
view: model_p1_win is P(player 1 wins), the market mid is the price of player-1-YES, and
the realized outcome y=1 iff player 1 actually won.

Data (all offline, committed; no network):
  - data/tennis_paper_signals.jsonl : full-slate daily scorer, one row per match, with
    model_p1_win (pre-match) and kalshi_p1_mid (market YES mid for player 1). No outcome.
  - data/tennis_paper_ledger.jsonl  : settled paper rows, settled_yes per player ticker.
  - data/tennis_trader_settled.jsonl: settled traded rows, outcome per signal-side ticker.
We JOIN signals -> outcomes by event_ticker (matching the player-1 ticker suffix), giving
ONE clean binary row per match: (model_p1, market_mid, y). This avoids the both-sides
double-counting and the stale duplicate rows present in the raw ledger.

CAVEAT: only matches that later appear in a settled file get an outcome, so this is a
subset of the full slate (N is small and the matches that settled skew toward traded
events). Stated plainly; numbers are reported as-is even when miscalibrated.

Run: python3 casino_sim/house_backtest_tennis.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIGNALS = os.path.join(ROOT, "data", "tennis_paper_signals.jsonl")
LEDGER = os.path.join(ROOT, "data", "tennis_paper_ledger.jsonl")
TRADER = os.path.join(ROOT, "data", "tennis_trader_settled.jsonl")
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
VIGS = (0.0, 0.045, 0.07)


def _suffix(ticker):
    return ticker.split("-")[-1] if ticker else None


def _outcome_maps():
    """Build event_ticker -> {ticker_suffix: y} from both settled sources.
    y=1.0 means that player's YES contract resolved yes (they won the match)."""
    ledger, trader = {}, {}
    for line in open(LEDGER):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("settlement_status") != "settled" or r.get("settled_yes") not in (0.0, 1.0):
            continue
        ev = r.get("event_ticker")
        suf = _suffix(r.get("market_ticker"))
        if ev and suf:
            ledger.setdefault(ev, {})[suf] = float(r["settled_yes"])
    for line in open(TRADER):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ev = r.get("event_ticker")
        suf = _suffix(r.get("ticker"))
        if ev is not None and suf and r.get("outcome") is not None:
            trader.setdefault(ev, {})[suf] = 1.0 if r["outcome"] else 0.0
    return ledger, trader


def load():
    """One clean binary row per settled match, player-1-centric:
    p = model_p1_win, mkt = kalshi_p1_mid, y = 1 iff player 1 won."""
    ledger, trader = _outcome_maps()
    # Collapse signals to one row per match (keep the most recent capture).
    sig_by_ev = {}
    for line in open(SIGNALS):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("event_ticker"):
            sig_by_ev[r["event_ticker"]] = r

    rows = []
    for ev, r in sig_by_ev.items():
        p, mkt = r.get("model_p1_win"), r.get("kalshi_p1_mid")
        if p is None or mkt is None:
            continue
        p1suf, p2suf = _suffix(r.get("p1_ticker")), _suffix(r.get("p2_ticker"))
        y = None
        for src in (ledger, trader):
            if ev in src:
                if p1suf in src[ev]:
                    y = src[ev][p1suf]; break
                if p2suf in src[ev]:           # outcome logged on the opponent's side
                    y = 1.0 - src[ev][p2suf]; break
        if y is None:                          # no settled outcome -> can't calibrate
            continue
        rows.append({"ev": ev, "p": float(p), "mkt": float(mkt), "y": float(y)})
    return rows


def calibration(rows):
    n = len(rows)
    base = sum(r["y"] for r in rows) / n
    bm = sum((r["p"] - r["y"]) ** 2 for r in rows) / n
    bk = sum((r["mkt"] - r["y"]) ** 2 for r in rows) / n
    # favorite/underdog skew vs TRUTH (model probability, not the casino)
    favs = [r for r in rows if r["p"] >= 0.5]
    dogs = [r for r in rows if r["p"] < 0.5]
    def mp(g): return sum(r["p"] for r in g) / len(g) if g else float("nan")
    def my(g): return sum(r["y"] for r in g) / len(g) if g else float("nan")
    # reliability bins (5 buckets of width 0.2)
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
    """Per match (2-outcome): house posts q_yes=p*(1+vig), q_no=(1-p)*(1+vig). 1 unit
    of action is split f_yes/f_no; settle on real y. flow='crowd' (bettors follow the
    market, f_yes = market mid) or 'sharp' (bettors take whichever side we price below
    the market's fair price = adverse selection)."""
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
        raise SystemExit("no settled tennis matches after join")
    os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    cal = calibration(rows)
    bt = {f"{v:.3f}": {"crowd": round(house(rows, v, "crowd"), 4),
                       "sharp": round(house(rows, v, "sharp"), 4)} for v in VIGS}
    out = {"n": cal["n"], "calibration": cal, "house_hold": bt}
    json.dump(out, open(os.path.join(DATA, "house_backtest_tennis.json"), "w"), indent=2)

    f, fa = cal["favorites"], cal["underdogs"]
    L = []
    L.append("# Tennis — calibration & blind house backtest (real outcomes)\n")
    L.append(f"_{cal['n']} settled matches (one binary row per match, player-1-centric), "
             "model probability set pre-match, settled on the actual result. Subset of the "
             "full slate — only matches that later settled get an outcome, so N is small and "
             "skews toward traded events; stated honestly._\n")
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
             f"{'model is under-confident on favorites (under-prices them)' if gap_fav>0 else 'model is OVER-confident on favorites (over-prices them)'}.\n")
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
             f"{abs(crowd45)*100:.1f}% against crowd flow, and "
             f"{'still profits '+f'{sharp45*100:.1f}%' if sharp45>0 else 'LOSES '+f'{abs(sharp45)*100:.1f}%'} "
             "against sharp bettors — the honest verdict on whether our tennis line is sharp "
             "enough to be the book.\n")
    report = "\n".join(L)
    open(os.path.join(DATA, "HOUSE_BACKTEST_TENNIS.md"), "w").write(report + "\n")

    _plot(cal)
    print(report)


def _plot(cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = cal["bins"]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    ax.plot([b["pred"] for b in bins], [b["obs"] for b in bins], "o-", color="#2f855a",
            label="our model")
    for b in bins:
        ax.annotate(f"n={b['n']}", (b["pred"], b["obs"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel("Predicted probability (our model, player 1)")
    ax.set_ylabel("Actual win frequency")
    ax.set_title(f"Tennis reliability — Brier {cal['brier_model']} vs market {cal['brier_market']}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tennis_reliability.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
