#!/usr/bin/env python3
"""
Blind house backtest + calibration — MLB (settled on REAL outcomes)
===================================================================

Mirror of casino_sim/house_backtest.py (World Cup), ported to MLB. The honest
test of "be the book": our model set each probability PRE-game (blind to the
result); we then (a) check how well-calibrated those probabilities were against
what actually happened, and (b) simulate running a book on our line and settling
on the REAL outcome — does it make money, or does miscalibration sink it?

MLB contracts are BINARY (the event resolves yes/no), simpler than WC's 3-way.

DATA — two committed, offline MLB paper ledgers, used for different jobs because
no single MLB ledger carries all three of {model prob, market price, real outcome}
cleanly:

  PRIMARY (house backtest + market comparison):
    data/mlb_kprop_paper_ledger.jsonl  — pitcher-strikeout props (KXMLBKS).
    Each settled row has model.prob_yes (pre-game), kalshi.mid (the market YES
    mid we captured), and settlement.outcome_yes (the contract really resolved
    yes/no). This is the only MLB ledger with model + market + truth together,
    so it carries the full WC-style pipeline: Brier(model) vs Brier(market),
    ECE, reliability, fav/dog skew, AND the blind house simulation.

  SECONDARY (clean game-winner calibration, the closest analog to WC's match
  winner):
    data/mlb_gamewinner_paper_ledger.jsonl — home-team game-winner (KXMLBGAME),
    361 rows, model.prob_yes + settlement.outcome_yes, all real outcomes. These
    rows were settled from a training matrix and carry NO captured market price
    (kalshi.mid is null on every row, and our pregame Kalshi price capture only
    begins 2026-06-12, after these games), so the market column and the house
    simulation are N/A for this spine. We still report its model calibration vs
    truth because it is the cleanest binary game-winner check we have.

HONESTY CAVEATS (stated plainly, not buried):
  - kprop is the EDGE-OBSERVED prop subset, not a full slate → selection bias.
  - kprop model probs are mostly low (strikeout overs); favorites/underdogs are
    defined by model_prob>=0.5 vs <0.5 exactly as in WC, but the >=0.5 bucket is
    thin for props — N stated per bucket.
  - The MLB game-winner model is known to have had a calibration collapse toward
    ~52% (flat, feature-starved); the gamewinner panel here shows that directly.
  - The "sharp" house flow is an adversarial worst case (bettors always take the
    side we price below the market), not a realistic mix.

Run: python3 casino_sim/house_backtest_mlb.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KPROP = os.path.join(ROOT, "data", "mlb_kprop_paper_ledger.jsonl")
GAMEWINNER = os.path.join(ROOT, "data", "mlb_gamewinner_paper_ledger.jsonl")
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
VIGS = (0.0, 0.045, 0.07)


def _truthy_outcome(v):
    """settlement.outcome_yes is bool in kprop, int 0/1 in gamewinner."""
    if v is True or v == 1:
        return 1.0
    if v is False or v == 0:
        return 0.0
    return None


def load_kprop():
    """model prob + market mid + real outcome. Dedup by ticker (keep first)."""
    rows, seen = [], set()
    for line in open(KPROP):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = r.get("settlement", {})
        if s.get("join_status") != "settled":
            continue
        y = _truthy_outcome(s.get("outcome_yes"))
        p = r.get("model", {}).get("prob_yes")
        mkt = r.get("kalshi", {}).get("mid")
        if y is None or p is None or mkt is None:
            continue
        t = r.get("ticker")
        if t in seen:
            continue
        seen.add(t)
        rows.append({"p": float(p), "mkt": float(mkt), "y": y})
    return rows


def load_gamewinner():
    """model prob + real outcome only (no market price on these rows)."""
    rows = []
    for line in open(GAMEWINNER):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = r.get("settlement", {})
        if s.get("join_status") != "settled":
            continue
        y = _truthy_outcome(s.get("outcome_yes"))
        p = r.get("model", {}).get("prob_yes")
        if y is None or p is None:
            continue
        rows.append({"p": float(p), "mkt": None, "y": y})
    return rows


def calibration(rows, with_market=True):
    n = len(rows)
    base = sum(r["y"] for r in rows) / n
    bm = sum((r["p"] - r["y"]) ** 2 for r in rows) / n
    bk = None
    skill = None
    if with_market and all(r["mkt"] is not None for r in rows):
        bk = sum((r["mkt"] - r["y"]) ** 2 for r in rows) / n
        skill = round(1 - bm / bk, 3) if bk else None
    # favorite/underdog skew vs TRUTH (same split as WC)
    favs = [r for r in rows if r["p"] >= 0.5]
    dogs = [r for r in rows if r["p"] < 0.5]

    def mp(g):
        return sum(r["p"] for r in g) / len(g) if g else float("nan")

    def my(g):
        return sum(r["y"] for r in g) / len(g) if g else float("nan")

    # reliability quintiles (0-.2, .2-.4, ... ) matching WC's 5 bins
    bins = []
    for lo in [i / 5 for i in range(5)]:
        hi = lo + 0.2
        g = [r for r in rows if lo <= r["p"] < hi or (hi == 1.0 and r["p"] == 1.0)]
        if g:
            bins.append({"lo": lo, "hi": hi, "n": len(g),
                         "pred": mp(g), "obs": my(g)})
    ece = sum(b["n"] * abs(b["pred"] - b["obs"]) for b in bins) / n
    return {
        "n": n, "base_rate": round(base, 4),
        "brier_model": round(bm, 4),
        "brier_market": round(bk, 4) if bk is not None else None,
        "skill_vs_market": skill,
        "ece": round(ece, 4),
        "favorites": {"n": len(favs), "mean_pred": round(mp(favs), 4) if favs else None,
                      "mean_actual": round(my(favs), 4) if favs else None},
        "underdogs": {"n": len(dogs), "mean_pred": round(mp(dogs), 4) if dogs else None,
                      "mean_actual": round(my(dogs), 4) if dogs else None},
        "bins": bins,
    }


def house(rows, vig, flow):
    """Per contract: house posts q_yes=p*(1+vig), q_no=(1-p)*(1+vig). 1 unit of
    action split f_yes/f_no; settle on real y. flow='crowd' (proportional to the
    market price) or 'sharp' (bettors take whichever side we price below the
    market = adverse selection). Identical mechanics to the WC template."""
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
    kp = load_kprop()
    gw = load_gamewinner()
    if not kp:
        raise SystemExit("no settled MLB kprop rows")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    cal = calibration(kp, with_market=True)            # primary spine
    gw_cal = calibration(gw, with_market=False) if gw else None
    bt = {f"{v:.3f}": {"crowd": round(house(kp, v, "crowd"), 4),
                       "sharp": round(house(kp, v, "sharp"), 4)} for v in VIGS}

    out = {
        "primary_market": "KXMLBKS pitcher-strikeout props (binary)",
        "n": cal["n"],
        "calibration": cal,
        "house_hold": bt,
        "secondary_gamewinner_calibration": gw_cal,
        "notes": {
            "kprop_caveat": "edge-observed prop subset; selection-biased; "
                            "deduped by ticker; 2026-06-14..2026-06-25.",
            "gamewinner_caveat": "361 home-team game-winner rows, real outcomes, "
                                 "NO captured market price (training-matrix settle, "
                                 "pre-dates our pregame price capture) -> market "
                                 "column & house sim N/A for this spine.",
        },
    }
    json.dump(out, open(os.path.join(DATA, "house_backtest_mlb.json"), "w"), indent=2)

    report = _report(cal, gw_cal, bt)
    open(os.path.join(DATA, "HOUSE_BACKTEST_MLB.md"), "w").write(report + "\n")
    _plot(cal, gw_cal)
    print(report)


def _report(cal, gw_cal, bt):
    f, fa = cal["favorites"], cal["underdogs"]
    L = []
    L.append("# MLB — calibration & blind house backtest (real outcomes)\n")
    L.append(f"_Primary spine: **{cal['n']}** settled KXMLBKS pitcher-strikeout "
             "prop contracts (binary). Model probability set pre-game, settled on "
             "the actual strikeout result. Edge-observed prop subset — "
             "selection-biased, deduped by ticker; stated honestly._\n")

    L.append("## Are our probabilities actually right? (vs TRUTH, not the casino)\n")
    bk = cal["brier_market"]
    sk = cal["skill_vs_market"]
    sk_str = f"{sk:+.3f}" if sk is not None else "n/a"
    L.append(f"- Brier — **model {cal['brier_model']}** vs market {bk} "
             f"(skill vs market {sk_str}) · ECE {cal['ece']} · base rate {cal['base_rate']}")
    if f["n"]:
        L.append(f"- **Favorites (model >=50%):** predicted {f['mean_pred']*100:.0f}% -> actually won "
                 f"{f['mean_actual']*100:.0f}%  (n={f['n']})")
    if fa["n"]:
        L.append(f"- **Underdogs (model <50%):** predicted {fa['mean_pred']*100:.0f}% -> actually won "
                 f"{fa['mean_actual']*100:.0f}%  (n={fa['n']})")
    if f["n"]:
        gap_fav = f["mean_actual"] - f["mean_pred"]
        L.append(f"\n> Favorites won {abs(gap_fav)*100:.0f}pp "
                 f"{'MORE' if gap_fav > 0 else 'LESS'} than the model predicted — "
                 f"{'model under-prices favorites' if gap_fav > 0 else 'model over-prices favorites here'}.\n")

    L.append("## Would a book on our line make money? (blind, settled on real results)\n")
    L.append("House hold per unit of action — **crowd** = flow proportional to the "
             "market price; **sharp** = bettors exploit wherever we price below the "
             "market (adversarial worst case):\n")
    L.append("| Posted vig | Crowd flow | Sharp flow |")
    L.append("|---|---|---|")
    for v in VIGS:
        b = bt[f"{v:.3f}"]
        L.append(f"| {v*100:.1f}% | {b['crowd']*100:+.1f}% | {b['sharp']*100:+.1f}% |")
    L.append("")
    crowd45 = bt["0.045"]["crowd"]
    sharp45 = bt["0.045"]["sharp"]
    L.append(f"> At a 4.5% vig the book {'KEEPS' if crowd45 > 0 else 'LOSES'} "
             f"{abs(crowd45)*100:.1f}% against crowd flow, and "
             f"{'still profits ' + f'{sharp45*100:.1f}%' if sharp45 > 0 else 'LOSES ' + f'{abs(sharp45)*100:.1f}%'} "
             "against sharp bettors — the honest verdict on whether our line is sharp "
             "enough to be the book.\n")

    L.append("## Secondary: clean game-winner calibration (KXMLBGAME, no market price)\n")
    if gw_cal:
        gf, gd = gw_cal["favorites"], gw_cal["underdogs"]
        L.append(f"_{gw_cal['n']} home-team game-winner contracts, real outcomes. "
                 "These training-matrix rows carry **no captured market price**, so "
                 "Brier-vs-market and the house sim are N/A here; model-vs-truth only._\n")
        L.append(f"- Brier model **{gw_cal['brier_model']}** · ECE {gw_cal['ece']} · "
                 f"base rate {gw_cal['base_rate']}")
        if gf["n"]:
            L.append(f"- Favorites (model >=50%): predicted {gf['mean_pred']*100:.0f}% -> won "
                     f"{gf['mean_actual']*100:.0f}% (n={gf['n']})")
        if gd["n"]:
            L.append(f"- Underdogs (model <50%): predicted {gd['mean_pred']*100:.0f}% -> won "
                     f"{gd['mean_actual']*100:.0f}% (n={gd['n']})")
        pmin = min(b["lo"] for b in gw_cal["bins"]) if gw_cal["bins"] else 0
        pmax = max(b["hi"] for b in gw_cal["bins"]) if gw_cal["bins"] else 1
        L.append(f"\n> The game-winner model is **flat**: all predictions fall in "
                 f"~[{pmin:.1f},{pmax:.1f}] (the known ~52% calibration collapse — "
                 "feature-starved, squashed). A book on this line has no edge to sell.\n")
    else:
        L.append("_No usable game-winner rows found._\n")

    L.append("---\n_Offline committed data only; no network. Mirrors "
             "casino_sim/house_backtest.py (World Cup)._")
    return "\n".join(L)


def _plot(cal, gw_cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have_gw = bool(gw_cal and gw_cal["bins"])
    ncol = 2 if have_gw else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol, 5.2), squeeze=False)

    def draw(ax, c, title, color):
        bins = c["bins"]
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
        ax.plot([b["pred"] for b in bins], [b["obs"] for b in bins], "o-",
                color=color, label="our model")
        for b in bins:
            ax.annotate(f"n={b['n']}", (b["pred"], b["obs"]), fontsize=7,
                        textcoords="offset points", xytext=(4, -8))
        ax.set_xlabel("Predicted probability (our model)")
        ax.set_ylabel("Actual win frequency")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)

    bkt = f"vs market {cal['brier_market']}" if cal["brier_market"] is not None else ""
    draw(axes[0][0], cal,
         f"MLB k-prop reliability — Brier {cal['brier_model']} {bkt}".strip(),
         "#2b6cb0")
    if have_gw:
        draw(axes[0][1], gw_cal,
             f"MLB game-winner reliability — Brier {gw_cal['brier_model']}",
             "#c05621")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mlb_reliability.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
