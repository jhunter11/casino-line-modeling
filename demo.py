#!/usr/bin/env python3
"""
Live model demo — loads the REAL trained models from models/ and makes predictions.

  Tennis & MLB : the actual trained XGBoost boosters.
  World Cup    : the Elo + Dixon-Coles model.

These are the same models used throughout this repo's analysis — not descriptions of
them. Run:  pip install -r requirements.txt  &&  python3 demo.py
"""
import json
import os
import sys

import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "models")
sys.path.insert(0, os.path.join(M, "code"))


def hdr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


TENNIS_FEATURES = [
    "elo_diff", "surface_elo_diff", "rank_log_advantage", "seed_advantage",
    "age_diff", "height_cm_diff", "h2h_win_pct_diff", "recent_win_pct_diff",
    "recent_match_count_diff", "days_rest_diff", "best_of_5", "is_grand_slam",
    "surface_hard", "surface_clay", "surface_grass", "surface_carpet", "surface_unknown",
]


def tennis_demo():
    hdr("TENNIS — real trained XGBoost booster (reproduces the recorded predictions)")
    b = xgb.Booster()
    b.load_model(os.path.join(M, "tennis", "tennis_xgboost_noodds_booster.bin"))
    shown = 0
    for line in open(os.path.join(HERE, "data", "tennis_paper_signals.jsonl")):
        r = json.loads(line)
        f = r.get("features")
        if not isinstance(f, dict):
            continue
        X = [[float(f.get(k, 0.0)) for k in TENNIS_FEATURES]]
        dm = xgb.DMatrix(X, feature_names=TENNIS_FEATURES)
        p = float(b.predict(dm)[0])
        rec = r.get("model_p1_win")
        match = (r.get("match") or "")[:42]
        ok = "✓" if rec is not None and abs(p - rec) < 0.02 else " "
        print(f"  {match:42s}  model P(P1 win) = {p:.3f}   recorded {rec}  {ok}")
        shown += 1
        if shown >= 6:
            break


def mlb_demo():
    hdr("MLB — real trained XGBoost booster (v7 game-winner)")
    b = xgb.Booster()
    b.load_model(os.path.join(M, "mlb", "mlb_elo_v7_xgb_booster.bin"))
    feats = json.load(open(os.path.join(M, "mlb", "mlb_elo_v7_xgb.json")))["features"]
    print(f"  features (home-team differentials): {feats}\n")
    scenarios = [
        ("Strong home favorite", [150.0, 0.9, -0.8, 3.0, -0.2]),
        ("Even matchup",         [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Home underdog",        [-120.0, -0.6, 0.7, -2.0, 0.3]),
    ]
    for name, vals in scenarios:
        dm = xgb.DMatrix([vals])
        p = float(b.predict(dm)[0])
        print(f"  {name:22s} {vals}  ->  P(home win) = {p:.3f}")


def wc_demo():
    hdr("WORLD CUP — Elo + Dixon-Coles 3-way model")
    import wc_dc_model
    games = [("Argentina", "Algeria", 2050, 1650),
             ("USA", "Bosnia & Herzegovina", 1990, 1710),
             ("Evenly matched", "side", 1800, 1800)]
    for h, a, eh, ea in games:
        pa, pd, pb = wc_dc_model.predict_3way(eh, ea, neutral=True)
        print(f"  {h} (Elo {eh}) vs {a} (Elo {ea}):  "
              f"home {pa:.1%} / draw {pd:.1%} / away {pb:.1%}")


if __name__ == "__main__":
    for fn in (tennis_demo, mlb_demo, wc_demo):
        try:
            fn()
        except Exception as e:
            print(f"  [section error: {type(e).__name__}: {e}]")
    print("\nDone. Every number above came from the actual committed models in ./models/.")
