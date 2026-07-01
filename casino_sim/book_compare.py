#!/usr/bin/env python3
"""
World Cup 2026 — OUR line (model + our vig) vs REAL sportsbooks, per game
========================================================================

For every match this builds the table the project is really about:

    | Match | OUR line (H/D/A) | Pinnacle | DraftKings | Bet365 | Consensus | Avg deviance |

"OUR line" = our model's probabilities (computed without ever seeing a betting line)
with OUR vig added, expressed both as de-vigged probabilities (for an apples-to-apples
"chance" comparison) and as posted decimal odds (the real line we'd offer). Each book's
odds are de-vigged the same way. The deviance column is the average |our prob − consensus
prob| across the three outcomes — how far our independent line sits from the market's.

Outputs:
  - casino_sim/data/book_comparison.csv   full spreadsheet (probs + our posted odds + every book)
  - casino_sim/data/book_comparison.md    readable per-game table (for the README)
  - casino_sim/data/book_summary.json     headline stats
  - casino_sim/figures/*.png              graphs

INPUT (source-agnostic) — casino_sim/book_odds.csv:
    home_team, away_team, book, home_dec, draw_dec, away_dec [, commence_time]
  one row per (match, book). Fill from `harness/books_consensus.py --sport
  soccer_fifa_world_cup` (needs THE_ODDS_API_KEY) or a manual/historical export.

Run:  python3 casino_sim/book_compare.py [path/to/book_odds.csv]
"""

import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_OUT = os.path.join(HERE, "data")
FIG_OUT = os.path.join(HERE, "figures")
SCORE_GLOB = os.path.join(ROOT, "data", "wc_scores", "*.json")
DEFAULT_ODDS = os.path.join(HERE, "book_odds.csv")

OUR_VIG = 0.045                          # the vig WE add to post a line
HEADLINE = ["sporttery", "kalshi"]       # real-market sources available for free
SHARP = "sporttery"


def norm(s):
    return "".join(c for c in (s or "").lower() if c.isalnum())


def devig_decimal(dh, dd, da):
    inv = [1.0 / dh, 1.0 / dd, 1.0 / da]
    s = sum(inv)
    return (inv[0] / s, inv[1] / s, inv[2] / s), (s - 1.0)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def dec_to_american(dec):
    if not dec or dec <= 1:
        return "—"
    return f"+{round((dec - 1) * 100)}" if dec >= 2 else f"-{round(100 / (dec - 1))}"


def prob_to_american(p):
    return dec_to_american(1 / p) if p and p > 0 else "—"


def _am3(decs):
    return "/".join(dec_to_american(d) for d in decs)


def _amp3(probs):
    return "/".join(prob_to_american(p) for p in probs)


def load_model_probs():
    by_pair = {}
    for f in sorted(glob.glob(SCORE_GLOB)):
        snap = os.path.basename(f)[:-5]
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        for g in doc.get("games", []):
            ko = g.get("kickoff", "")
            if ko and snap > ko:
                continue
            legs = {l["outcome"]: l.get("model_prob") for l in g.get("legs", [])}
            if not all(o in legs and legs[o] is not None for o in ("A", "TIE", "B")):
                continue
            key = frozenset((norm(g["team_a"]), norm(g["team_b"])))
            rec = {"snap": snap, "team_a": g["team_a"], "team_b": g["team_b"], "kickoff": ko,
                   "pH": legs["A"], "pD": legs["TIE"], "pB": legs["B"]}
            if key not in by_pair or snap > by_pair[key]["snap"]:
                by_pair[key] = rec
    return by_pair


def load_book_odds(path):
    games = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                dh, dd, da = float(r["home_dec"]), float(r["draw_dec"]), float(r["away_dec"])
            except (KeyError, ValueError):
                continue
            key = frozenset((norm(r["home_team"]), norm(r["away_team"])))
            g = games.setdefault(key, {"home": r["home_team"], "away": r["away_team"], "rows": []})
            g["rows"].append({"book": r["book"].strip().lower(), "dh": dh, "dd": dd, "da": da})
    return games


def main():
    odds_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ODDS
    if not os.path.exists(odds_path):
        raise SystemExit(
            f"No odds file at {odds_path}. Provide real sportsbook odds — e.g.\n"
            "  python3 harness/books_consensus.py --sport soccer_fifa_world_cup --regions us,uk,eu\n"
            "then map its output into casino_sim/book_odds.csv (see docstring)."
        )
    os.makedirs(DATA_OUT, exist_ok=True); os.makedirs(FIG_OUT, exist_ok=True)
    model, books = load_model_probs(), load_book_odds(odds_path)

    rows, per_book_absgap = [], {}
    mx, cy, gaps = [], [], []
    for key, g in books.items():
        m = model.get(key)
        if not m:
            continue
        if norm(g["home"]) == norm(m["team_a"]):
            mH, mD, mA = m["pH"], m["pD"], m["pB"]
        else:
            mH, mD, mA = m["pB"], m["pD"], m["pH"]
        per_book = {}
        for b in g["rows"]:
            (pH, pD, pA), vig = devig_decimal(b["dh"], b["dd"], b["da"])
            per_book[b["book"]] = {"pH": pH, "pD": pD, "pA": pA, "vig": vig,
                                   "dh": b["dh"], "dd": b["dd"], "da": b["da"]}
            for mv, bv in ((mH, pH), (mD, pD), (mA, pA)):
                per_book_absgap.setdefault(b["book"], []).append(abs(mv - bv))
        nb = len(per_book)
        cH = sum(v["pH"] for v in per_book.values()) / nb
        cD = sum(v["pD"] for v in per_book.values()) / nb
        cA = sum(v["pA"] for v in per_book.values()) / nb
        # OUR posted line: model prob + our vig
        our_odds = {o: round((1 + OUR_VIG) / p, 2) if p > 0 else ""
                    for o, p in (("H", mH), ("D", mD), ("A", mA))}
        our_dec = [(1 + OUR_VIG) / p if p > 0 else 0 for p in (mH, mD, mA)]
        our_american = _am3(our_dec)
        consensus_american = _amp3((cH, cD, cA))
        for mv, cv in ((mH, cH), (mD, cD), (mA, cA)):
            mx.append(mv); cy.append(cv); gaps.append(mv - cv)
        outc = ("H", "D", "A")
        same_fav = outc[[mH, mD, mA].index(max(mH, mD, mA))] == \
            outc[[cH, cD, cA].index(max(cH, cD, cA))]
        row = {
            "match": f'{g["home"]} v {g["away"]}', "kickoff": m["kickoff"], "n_books": nb,
            "our_pH": round(mH, 4), "our_pD": round(mD, 4), "our_pA": round(mA, 4),
            "our_odds_H": our_odds["H"], "our_odds_D": our_odds["D"], "our_odds_A": our_odds["A"],
            "our_american": our_american,
            "consensus_pH": round(cH, 4), "consensus_pD": round(cD, 4), "consensus_pA": round(cA, 4),
            "consensus_american": consensus_american,
            "avg_chance_deviance_pp": round(100 * (abs(mH-cH)+abs(mD-cD)+abs(mA-cA)) / 3, 2),
            "mean_book_vig_pct": round(100 * sum(v["vig"] for v in per_book.values()) / nb, 2),
            "same_favorite": same_fav,
        }
        for bk in HEADLINE:
            v = per_book.get(bk)
            row[f"{bk}_pH"] = round(v["pH"], 4) if v else ""
            row[f"{bk}_pD"] = round(v["pD"], 4) if v else ""
            row[f"{bk}_pA"] = round(v["pA"], 4) if v else ""
            row[f"{bk}_am"] = _am3((v["dh"], v["dd"], v["da"])) if v else "—"
        rows.append(row)

    if not rows:
        raise SystemExit("No matches joined — check team-name spelling between odds CSV and wc_scores.")
    rows.sort(key=lambda r: r["match"])

    # full spreadsheet
    with open(os.path.join(DATA_OUT, "book_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # readable per-game markdown tables: probabilities + American odds
    _write_markdown(rows)
    _write_markdown_american(rows)

    n = len(mx)
    summary = {
        "n_matches": len(rows), "n_legs": n, "our_vig_pct": OUR_VIG * 100,
        "books_seen": sorted(per_book_absgap.keys()),
        "avg_chance_deviance_pp": round(sum(r["avg_chance_deviance_pp"] for r in rows) / len(rows), 2),
        "correlation_model_vs_consensus": round(pearson(mx, cy), 3),
        "favorite_agreement": round(sum(r["same_favorite"] for r in rows) / len(rows), 3),
        "mean_book_vig_pct": round(sum(r["mean_book_vig_pct"] for r in rows) / len(rows), 2),
        "per_book_mean_abs_gap_pp": {b: round(100 * sum(v) / len(v), 2)
                                     for b, v in sorted(per_book_absgap.items())},
    }
    json.dump(summary, open(os.path.join(DATA_OUT, "book_summary.json"), "w"), indent=2)
    _make_graphs(mx, cy, rows, summary)

    print(f"{summary['n_matches']} matches · books: {', '.join(summary['books_seen'])}")
    print(f"avg chance deviance {summary['avg_chance_deviance_pp']}pp · "
          f"r={summary['correlation_model_vs_consensus']} · "
          f"same-fav {summary['favorite_agreement']*100:.0f}% · "
          f"mean book vig {summary['mean_book_vig_pct']}%")
    print("table -> casino_sim/data/book_comparison.{csv,md}")


def _pct3(h, d, a):
    return f"{round(h*100)}/{round(d*100)}/{round(a*100)}"


def _write_markdown(rows):
    cols = HEADLINE
    head = " | ".join(c.capitalize() for c in cols)
    L = ["# Our line vs the books — per game\n",
         "_Cells are **Home win % / Draw % / Away win %** (de-vigged). "
         "“Our line” is our line-free model; the book columns are real de-vigged odds. "
         "“Deviance” = average |our % − consensus %| across the three outcomes._\n",
         f"| Match | Our line | {head} | Consensus | Avg deviance |",
         "|---|---|" + "---|" * len(cols) + "---|---|"]
    for r in rows:
        def cell(bk):
            h, d, a = r.get(f"{bk}_pH"), r.get(f"{bk}_pD"), r.get(f"{bk}_pA")
            return _pct3(h, d, a) if h != "" and h is not None else "—"
        cells = " | ".join(cell(bk) for bk in cols)
        L.append(f"| {r['match']} | {_pct3(r['our_pH'], r['our_pD'], r['our_pA'])} | "
                 f"{cells} | "
                 f"{_pct3(r['consensus_pH'], r['consensus_pD'], r['consensus_pA'])} | "
                 f"{r['avg_chance_deviance_pp']:.1f}pp |")
    open(os.path.join(DATA_OUT, "book_comparison.md"), "w").write("\n".join(L) + "\n")


def _write_markdown_american(rows):
    cols = HEADLINE
    head = " | ".join(c.capitalize() for c in cols)
    L = ["# Our line vs the books — American odds (moneyline)\n",
         "_Cells are **Home / Draw / Away** in American odds. “Our line” includes our 4.5% "
         "vig; book columns are their actual posted odds; consensus is fair (de-vigged)._\n",
         f"| Match | Our line | {head} | Consensus (fair) |",
         "|---|---|" + "---|" * len(cols) + "---|"]
    for r in rows:
        cells = " | ".join(r.get(f"{bk}_am", "—") for bk in cols)
        L.append(f"| {r['match']} | {r['our_american']} | {cells} | {r['consensus_american']} |")
    open(os.path.join(DATA_OUT, "book_comparison_american.md"), "w").write("\n".join(L) + "\n")


def _make_graphs(mx, cy, rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(mx, cy, s=18, alpha=0.6, color="#2b6cb0")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.set_xlabel("Our model probability (line-free)")
    ax.set_ylabel("Sportsbook consensus (de-vigged)")
    ax.set_title(f"Our independent line vs the books\nr={summary['correlation_model_vs_consensus']}  "
                 f"avg deviance={summary['avg_chance_deviance_pp']}pp")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_OUT, "model_vs_consensus.png"), dpi=130); plt.close(fig)

    pb = summary["per_book_mean_abs_gap_pp"]
    if pb:
        items = sorted(pb.items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(6.2, max(2.5, 0.4 * len(items))))
        ax.barh([k for k, _ in items], [v for _, v in items], color="#38a169")
        ax.set_xlabel("Mean |gap| vs our model (pp)")
        ax.set_title("Which books our model agrees with most")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_OUT, "per_book_gap.png"), dpi=130); plt.close(fig)

    # deviance by game (sorted) — shows where we agree / disagree with the market
    rr = sorted(rows, key=lambda r: r["avg_chance_deviance_pp"])
    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.22 * len(rr))))
    ax.barh([r["match"] for r in rr], [r["avg_chance_deviance_pp"] for r in rr],
            color="#dd6b20")
    ax.set_xlabel("Avg chance deviance vs consensus (pp)")
    ax.set_title("Per-game: how far our line sits from the books")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_OUT, "deviance_by_game.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
