#!/usr/bin/env python3
"""
Build casino_sim/book_odds.csv from FREE, real market sources:
  - Sporttery (China Sports Lottery) HAD market home/draw/away odds
    (external_odds/sporttery_odds_history.json — sourced from the public
     worldcup-ev-site dataset; underlying odds are Sporttery's).
  - Kalshi (real-money exchange) mids from our own data/wc_scores/*.json.

Team names are normalized to our model's spellings so both sources join cleanly.
Run: python3 casino_sim/adapt_sporttery.py
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPORTTERY = os.path.join(HERE, "external_odds", "sporttery_odds_history.json")
OUT = os.path.join(HERE, "book_odds.csv")


def norm(s):
    return "".join(c for c in (s or "").lower() if c.isalnum())


def yes_mid(m):
    b, a = (m or {}).get("yes_bid"), (m or {}).get("yes_ask")
    return (b + a) / 2.0 if b and a and b > 0 and a > 0 else None


# canonical names + latest pre-kickoff Kalshi mids per match
canon, kalshi = {}, {}
for f in sorted(glob.glob(os.path.join(ROOT, "data", "wc_scores", "*.json"))):
    snap = os.path.basename(f)[:-5]
    for g in json.load(open(f)).get("games", []):
        canon[norm(g["team_a"])] = g["team_a"]
        canon[norm(g["team_b"])] = g["team_b"]
        ko = g.get("kickoff", "")
        if ko and snap > ko:
            continue
        legs = {l["outcome"]: l for l in g.get("legs", [])}
        mids = {o: yes_mid(legs[o].get("market")) for o in ("A", "TIE", "B") if o in legs}
        if len(mids) == 3 and all(mids.values()):
            key = frozenset((norm(g["team_a"]), norm(g["team_b"])))
            rec = {"snap": snap, "home": g["team_a"], "away": g["team_b"],
                   "dh": 1 / mids["A"], "dd": 1 / mids["TIE"], "da": 1 / mids["B"]}
            if key not in kalshi or snap > kalshi[key]["snap"]:
                kalshi[key] = rec

rows = [["home_team", "away_team", "book", "home_dec", "draw_dec", "away_dec"]]

# Sporttery rows (real bookmaker)
spo = json.load(open(SPORTTERY)).get("records", {})
n_spo = 0
for r in spo.values():
    o = r.get("had_odds") or {}
    if not all(k in o for k in ("home", "draw", "away")):
        continue
    h = canon.get(norm(r.get("home_en", "")), r.get("home_en"))
    a = canon.get(norm(r.get("away_en", "")), r.get("away_en"))
    rows.append([h, a, "sporttery", o["home"], o["draw"], o["away"]])
    n_spo += 1

# Kalshi rows (real-money exchange)
for rec in kalshi.values():
    rows.append([rec["home"], rec["away"], "kalshi",
                 round(rec["dh"], 3), round(rec["dd"], 3), round(rec["da"], 3)])

csv.writer(open(OUT, "w", newline="")).writerows(rows)
print(f"wrote {OUT}: {n_spo} sporttery rows + {len(kalshi)} kalshi rows")
