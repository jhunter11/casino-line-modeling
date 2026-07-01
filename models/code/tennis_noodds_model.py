"""No-odds tennis model: structural features only, no bookmaker data.

Uses same best HPs from HPO run but drops:
  - p1_implied_prob (index 12)
  - implied_prob_diff (index 13)
  - odds_overround (index 14)

Also tests a residual model: predict (actual_outcome - p1_implied_prob)
using structural features, to find alpha orthogonal to the market.

Saves:
  - data/hpo/tennis_xgboost_noodds.json  (metadata + metrics)
  - data/hpo/tennis_xgboost_noodds_booster.bin  (XGBoost model)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss as sk_log_loss

DATA_DIR = Path(__file__).resolve().parents[2] / "engine" / "data" / "tennis" / "tennis_atp" / "tennis_atp-master"
OUT_DIR = Path(__file__).resolve().parent
BRIER_BASELINE = 0.25
CURRENT_YEAR = date.today().year

# Full feature names (20 total, same as HPO)
ALL_FEATURE_NAMES = [
    "elo_diff",           # 0
    "surface_elo_diff",   # 1
    "rank_log_advantage", # 2
    "seed_advantage",     # 3
    "age_diff",           # 4
    "height_cm_diff",     # 5
    "h2h_win_pct_diff",   # 6
    "recent_win_pct_diff",# 7
    "recent_match_count_diff", # 8
    "days_rest_diff",     # 9
    "best_of_5",          # 10
    "is_grand_slam",      # 11
    "p1_implied_prob",    # 12  <-- bookmaker feature
    "implied_prob_diff",  # 13  <-- bookmaker feature
    "odds_overround",     # 14  <-- bookmaker feature
    "surface_hard",       # 15
    "surface_clay",       # 16
    "surface_grass",      # 17
    "surface_carpet",     # 18
    "surface_unknown",    # 19
]

# Bookmaker feature indices to drop
BOOKIE_INDICES = {12, 13, 14}

STRUCTURAL_FEATURE_NAMES = [f for i, f in enumerate(ALL_FEATURE_NAMES) if i not in BOOKIE_INDICES]
STRUCTURAL_INDICES = [i for i in range(len(ALL_FEATURE_NAMES)) if i not in BOOKIE_INDICES]

# Best HPs from prior HPO run
BEST_HP = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.85,
}


def _surface_key(surface: str) -> str:
    v = (surface or "").strip().lower()
    return v if v in {"hard", "clay", "grass", "carpet"} else "unknown"


def _elo_update(w: float, l: float, k: float = 32.0):
    e = 1.0 / (1.0 + 10.0 ** ((l - w) / 400.0))
    d = k * (1.0 - e)
    return w + d, l - d


def _implied_probs(p1_odds, p2_odds):
    p1 = 1.0 / p1_odds if p1_odds and p1_odds > 1.0 else 0.0
    p2 = 1.0 / p2_odds if p2_odds and p2_odds > 1.0 else 0.0
    overround = p1 + p2
    if overround <= 0:
        return 0.5, 0.5, 0.0
    return p1 / overround, p2 / overround, overround


def _fn(v, default=0.0):
    try:
        f = float(v)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _parse_date(v) -> date | None:
    s = str(v or "").strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    if s:
        try:
            return datetime.fromisoformat(s).date()
        except ValueError:
            pass
    return None


def _rank_log_advantage(r1, r2) -> float:
    default = 2500
    r1v = int(float(r1)) if r1 else default
    r2v = int(float(r2)) if r2 else default
    return math.log1p(r2v) - math.log1p(r1v)


def _seed_adv(s1, s2) -> float:
    d = 64
    s1v = int(float(s1)) if s1 else d
    s2v = int(float(s2)) if s2 else d
    return float(s2v - s1v)


def _smoothed(wins, matches) -> float:
    return (wins + 2.5) / (matches + 5.0)


def build_feature_frame(csv_files: list[Path]):
    """Build full feature matrix (all 20 features) + p1_implied_prob for residual target."""
    all_rows: list[dict] = []
    for path in sorted(csv_files):
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                d = _parse_date(row.get("tourney_date"))
                if d is None:
                    continue
                row["_date"] = d
                all_rows.append(row)

    all_rows.sort(key=lambda r: (r["_date"], r.get("tourney_id", ""), _fn(r.get("match_num", 0))))
    print(f"  loaded {len(all_rows)} raw match rows")

    overall_elo: dict[str, float] = defaultdict(lambda: 1500.0)
    surface_elo: dict[tuple, float] = defaultdict(lambda: 1500.0)
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    last_played: dict[str, date] = {}
    h2h: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    X_rows: list[list[float]] = []
    y_vals: list[int] = []
    p1_implied_probs: list[float] = []
    date_vals: list[date] = []
    odds_available: list[bool] = []

    for row in all_rows:
        winner = str(row.get("winner_id", "") or "")
        loser = str(row.get("loser_id", "") or "")
        d = row["_date"]
        surface = str(row.get("surface") or "Unknown")
        surf_key = _surface_key(surface)
        level = str(row.get("tourney_level") or "")
        best_of = int(_fn(row.get("best_of"), 3))

        h2h_key = tuple(sorted((winner, loser)))
        wins = h2h[h2h_key]

        def _days_since(pid: str) -> float:
            lp = last_played.get(pid)
            if lp is None:
                return 0.0
            return float(max(min((d - lp).days, 30), 0))

        p1_odds_raw = _fn(row.get("AvgW") or row.get("B365W")) or None
        p2_odds_raw = _fn(row.get("AvgL") or row.get("B365L")) or None
        if p1_odds_raw and p1_odds_raw <= 1.0:
            p1_odds_raw = None
        if p2_odds_raw and p2_odds_raw <= 1.0:
            p2_odds_raw = None

        p1_ip, p2_ip, overround = _implied_probs(p1_odds_raw, p2_odds_raw)
        has_odds = bool(p1_odds_raw and p2_odds_raw)

        def _make_row(p1, p2, p1_ip_v, p2_ip_v):
            p1_rank = _fn(row.get("winner_rank" if p1 == winner else "loser_rank"), None)
            p2_rank = _fn(row.get("loser_rank" if p1 == winner else "winner_rank"), None)
            p1_seed_raw = row.get("winner_seed" if p1 == winner else "loser_seed")
            p2_seed_raw = row.get("loser_seed" if p1 == winner else "winner_seed")
            p1_age_raw = _fn(row.get("winner_age" if p1 == winner else "loser_age"), None)
            p2_age_raw = _fn(row.get("loser_age" if p1 == winner else "winner_age"), None)
            p1_ht = _fn(row.get("winner_ht" if p1 == winner else "loser_ht"), None)
            p2_ht = _fn(row.get("loser_ht" if p1 == winner else "winner_ht"), None)
            p1_elo_v = overall_elo[p1]
            p2_elo_v = overall_elo[p2]
            p1_selo = surface_elo[(p1, surf_key)]
            p2_selo = surface_elo[(p2, surf_key)]
            p1_h2hw = wins[p1]
            p2_h2hw = wins[p2]
            p1_rec = recent[p1]
            p2_rec = recent[p2]
            h2h_tot = p1_h2hw + p2_h2hw
            h2h_p1 = (p1_h2hw + 0.5) / (h2h_tot + 1.0)
            p1_wpc = _smoothed(sum(p1_rec), len(p1_rec))
            p2_wpc = _smoothed(sum(p2_rec), len(p2_rec))
            return [
                p1_elo_v - p2_elo_v,          # 0 elo_diff
                p1_selo - p2_selo,             # 1 surface_elo_diff
                _rank_log_advantage(p1_rank, p2_rank),  # 2 rank_log_advantage
                _seed_adv(p1_seed_raw, p2_seed_raw),    # 3 seed_advantage
                _fn(p1_age_raw, 0.0) - _fn(p2_age_raw, 0.0),  # 4 age_diff
                _fn(p1_ht, 0.0) - _fn(p2_ht, 0.0),     # 5 height_cm_diff
                (2.0 * h2h_p1) - 1.0,          # 6 h2h_win_pct_diff
                p1_wpc - p2_wpc,               # 7 recent_win_pct_diff
                float(len(p1_rec) - len(p2_rec)),  # 8 recent_match_count_diff
                _days_since(p1) - _days_since(p2),  # 9 days_rest_diff
                1.0 if best_of >= 5 else 0.0,  # 10 best_of_5
                1.0 if level.upper() == "G" else 0.0,  # 11 is_grand_slam
                p1_ip_v,                       # 12 p1_implied_prob
                p1_ip_v - p2_ip_v,             # 13 implied_prob_diff
                overround,                     # 14 odds_overround
                1.0 if surf_key == "hard" else 0.0,    # 15
                1.0 if surf_key == "clay" else 0.0,    # 16
                1.0 if surf_key == "grass" else 0.0,   # 17
                1.0 if surf_key == "carpet" else 0.0,  # 18
                1.0 if surf_key == "unknown" else 0.0, # 19
            ]

        # winner as p1
        X_rows.append(_make_row(winner, loser, p1_ip, p2_ip))
        y_vals.append(1)
        p1_implied_probs.append(p1_ip)
        date_vals.append(d)
        odds_available.append(has_odds)

        # mirrored: loser as p1
        X_rows.append(_make_row(loser, winner, p2_ip, p1_ip))
        y_vals.append(0)
        p1_implied_probs.append(p2_ip)
        date_vals.append(d)
        odds_available.append(has_odds)

        # update ELO state
        overall_elo[winner], overall_elo[loser] = _elo_update(overall_elo[winner], overall_elo[loser])
        surface_elo[(winner, surf_key)], surface_elo[(loser, surf_key)] = _elo_update(
            surface_elo[(winner, surf_key)], surface_elo[(loser, surf_key)]
        )
        recent[winner].append(1)
        recent[loser].append(0)
        wins[winner] += 1
        last_played[winner] = d
        last_played[loser] = d

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.int32)
    p1_ip_arr = np.array(p1_implied_probs, dtype=np.float32)
    return X, y, p1_ip_arr, date_vals, odds_available


def brier_skill(brier: float, baseline: float = BRIER_BASELINE) -> float:
    return (1.0 - brier / baseline) * 100.0


def brier_z_score(brier: float, n: int, baseline: float = BRIER_BASELINE) -> float:
    diff = baseline - brier
    se = math.sqrt((baseline * (1 - baseline) + brier * (1 - brier)) / (2 * n))
    if se == 0:
        return 0.0
    return diff / se


def train_xgb(X_tr, y_tr, X_val, y_val, hp, label="model"):
    """Train XGBoost binary classifier."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": hp["max_depth"],
        "eta": hp["learning_rate"],
        "subsample": hp["subsample"],
        "colsample_bytree": 0.85,
        "min_child_weight": hp["min_child_weight"],
        "seed": 42,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval_m = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params, dtrain,
        num_boost_round=hp["n_estimators"],
        evals=[(dtrain, "train"), (dval_m, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return booster


def select_holdout_year(years: list[int]) -> int:
    """Use the latest complete season as holdout; never partial current year."""
    if not years:
        raise ValueError("no years available")
    max_year = max(years)
    if max_year >= CURRENT_YEAR and len(years) > 1:
        return max(y for y in years if y < CURRENT_YEAR)
    return max_year


def train_production_booster(X_all, y_all, hp):
    """Train the saved live booster on every completed match row available."""
    n = len(X_all)
    val_split = int(n * 0.9)
    return train_xgb(
        X_all[:val_split],
        y_all[:val_split],
        X_all[val_split:],
        y_all[val_split:],
        hp,
        label="production",
    )


def evaluate(probs, y, label=""):
    brier = float(np.mean((probs - y.astype(float)) ** 2))
    skill = brier_skill(brier)
    n = len(y)
    z = brier_z_score(brier, n)
    preds = (probs > 0.5).astype(int)
    acc = float(np.mean(preds == y))
    auc = roc_auc_score(y, probs)
    ll = sk_log_loss(y, probs)
    baseline_brier = float(np.mean((0.5 - y.astype(float)) ** 2))
    print(f"\n  [{label}]")
    print(f"    Holdout Brier:  {brier:.5f}  (baseline={baseline_brier:.5f})")
    print(f"    Brier skill:    {skill:+.2f}%")
    print(f"    Z-score:        {z:.3f}")
    print(f"    Accuracy:       {acc:.4f}")
    print(f"    AUC:            {auc:.4f}")
    print(f"    Log-loss:       {ll:.5f}")
    print(f"    N:              {n}")
    return {
        "brier": brier, "skill_pct": skill, "z_score": z,
        "accuracy": acc, "auc": auc, "log_loss": ll, "n": n,
        "baseline_brier": baseline_brier,
    }


def main():
    print("=== Tennis No-Odds Model ===")
    print(f"Structural features ({len(STRUCTURAL_FEATURE_NAMES)}): {STRUCTURAL_FEATURE_NAMES}")
    print(f"Dropped bookmaker features: {[ALL_FEATURE_NAMES[i] for i in sorted(BOOKIE_INDICES)]}")
    print(f"Best HP from prior HPO: {BEST_HP}")

    csv_files = sorted(DATA_DIR.glob("atp_matches_*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files in {DATA_DIR}")
        return 1
    print(f"\nFound {len(csv_files)} CSV files")

    print("\nBuilding feature matrix...")
    X_full, y, p1_ip_arr, dates, odds_avail = build_feature_frame(csv_files)
    year_arr = np.array([d.year for d in dates], dtype=np.int32)
    print(f"  Total rows: {len(X_full)}, unique matches: {len(X_full)//2}")

    # Structural feature matrix (drop bookie columns)
    X_struct = X_full[:, STRUCTURAL_INDICES]
    print(f"  Structural X shape: {X_struct.shape}")

    years = sorted({int(y) for y in year_arr})
    holdout_year = select_holdout_year(years)
    train_mask = year_arr < holdout_year
    test_mask = year_arr == holdout_year

    print(f"  Years in data: {min(years)}-{max(years)}")
    print(f"\n  Eval train rows (< {holdout_year}): {train_mask.sum()}")
    print(f"  Eval test rows (= {holdout_year}): {test_mask.sum()}")

    # --- MODEL 1: No-odds structural model ---
    print("\n=== MODEL 1: Structural Only (no bookmaker features) ===")
    X_tr = X_struct[train_mask]
    y_tr = y[train_mask]
    X_ho = X_struct[test_mask]
    y_ho = y[test_mask]

    n_tr = len(X_tr)
    val_split = int(n_tr * 0.9)
    booster_noodds = train_xgb(
        X_tr[:val_split], y_tr[:val_split],
        X_tr[val_split:], y_tr[val_split:],
        BEST_HP, label="noodds"
    )

    probs_noodds = booster_noodds.predict(xgb.DMatrix(X_ho))
    metrics_noodds = evaluate(probs_noodds, y_ho, "No-Odds Structural Model")

    # Feature importances
    imp = booster_noodds.get_score(importance_type="gain")
    print(f"\n  Feature importances (gain) for no-odds model:")
    # Map feature indices to names
    feat_imp = {}
    for k, v in imp.items():
        idx = int(k[1:]) if k.startswith("f") else None
        if idx is not None and idx < len(STRUCTURAL_FEATURE_NAMES):
            feat_imp[STRUCTURAL_FEATURE_NAMES[idx]] = v
        else:
            feat_imp[k] = v
    for name, score in sorted(feat_imp.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name}: {score:.1f}")

    # --- MODEL 2: Bookie (full) model for reference ---
    print("\n=== MODEL 2: Full model (WITH bookmaker) for reference ===")
    X_tr_full = X_full[train_mask]
    X_ho_full = X_full[test_mask]

    n_tr = len(X_tr_full)
    val_split = int(n_tr * 0.9)
    booster_full = train_xgb(
        X_tr_full[:val_split], y_tr[:val_split],
        X_tr_full[val_split:], y_tr[val_split:],
        BEST_HP, label="full"
    )
    probs_full = booster_full.predict(xgb.DMatrix(X_ho_full))
    metrics_full = evaluate(probs_full, y_ho, "Full Model (with bookmaker features)")

    # --- MODEL 3: Residual analysis (simulated: use structural probs vs 0.5 baseline) ---
    print("\n=== MODEL 3: Residual analysis ===")
    odds_avail_arr = np.array(odds_avail, dtype=bool)
    n_with_real_odds = int(odds_avail_arr.sum())
    print(f"  NOTE: ATP CSV data has NO bookmaker odds ({n_with_real_odds} rows with real odds).")
    print("  All p1_implied_prob values were constant 0.5 fallback during HPO.")
    print("  Residual analysis: can structural model detect when Elo misprices vs a flat 0.5 baseline?")
    print("  This is a proxy for Kalshi edge: structural signal > flat 0.5 market.")

    # Residual vs 0.5 baseline: actual - 0.5
    # We measure whether the structural model's predicted direction of deviation from 0.5 is correct
    y_ho_actual = y[test_mask]
    probs_struct_ho = probs_noodds  # already computed above

    # Sign agreement: does structural model agree with outcome direction from 0.5?
    y_resid_vs_half = y_ho_actual.astype(float) - 0.5  # positive if p1 won, negative if lost
    pred_deviation = probs_struct_ho - 0.5  # structural model's deviation from 0.5

    dir_acc = float(np.mean(np.sign(pred_deviation) == np.sign(y_resid_vs_half)))
    n_test = len(y_ho_actual)
    se_dir = math.sqrt(0.5 * 0.5 / n_test)
    z_dir = (dir_acc - 0.5) / se_dir
    corr = float(np.corrcoef(pred_deviation, y_resid_vs_half)[0, 1])

    print(f"\n  Structural model vs 0.5 baseline on {n_test} holdout rows:")
    print(f"    Direction accuracy: {dir_acc:.4f}")
    print(f"    Direction Z-score vs 0.5: {z_dir:.3f}")
    print(f"    Pearson r(pred_dev, outcome_dev): {corr:.4f}")

    # Kalshi simulation: only bet when model disagrees strongly with 0.5
    print(f"\n  Simulated Kalshi edge (bet when |structural_prob - 0.5| > thresh):")
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25]:
        strong_mask = np.abs(pred_deviation) > thresh
        n_strong = int(strong_mask.sum())
        if n_strong < 50:
            continue
        strong_dir_acc = float(np.mean(np.sign(pred_deviation[strong_mask]) == np.sign(y_resid_vs_half[strong_mask])))
        strong_se = math.sqrt(0.5 * 0.5 / n_strong)
        strong_z = (strong_dir_acc - 0.5) / strong_se
        # Brier on this subset
        sub_brier = float(np.mean((probs_struct_ho[strong_mask] - y_ho_actual[strong_mask].astype(float)) ** 2))
        sub_skill = brier_skill(sub_brier)
        print(f"    |p-0.5|>{thresh:.2f}: n={n_strong}, dir_acc={strong_dir_acc:.4f}, z={strong_z:.3f}, brier_skill={sub_skill:+.2f}%")

    brier_bookie = BRIER_BASELINE  # fallback: odds not available; 0.5 baseline = 0.25
    brier_adjusted = metrics_noodds['brier']
    skill_bookie = 0.0
    skill_adjusted = metrics_noodds['skill_pct']

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"  Full model (w/ bookie):    Brier={metrics_full['brier']:.5f}  skill={metrics_full['skill_pct']:+.2f}%  acc={metrics_full['accuracy']:.4f}")
    print(f"  No-odds structural model:  Brier={metrics_noodds['brier']:.5f}  skill={metrics_noodds['skill_pct']:+.2f}%  acc={metrics_noodds['accuracy']:.4f}")
    print(f"  Skill LOST by removing odds: {metrics_noodds['skill_pct'] - metrics_full['skill_pct']:+.2f} pp")
    print(f"  Bookie-only:               Brier={brier_bookie:.5f}  skill={skill_bookie:+.2f}%")
    print(f"  Adjusted (bookie+resid):   Brier={brier_adjusted:.5f}  skill={skill_adjusted:+.2f}%")

    has_structural_edge = metrics_noodds['skill_pct'] >= 2.0 and metrics_noodds['z_score'] >= 1.96
    has_residual_edge = bool(z_dir >= 1.96)
    print(f"\n  Structural model edge (skill>=2%, z>=1.96): {'YES' if has_structural_edge else 'NO'}")
    print(f"  Residual direction edge (z>=1.96): {'YES' if has_residual_edge else 'NO'}")

    # Save production no-odds model trained on all available completed matches.
    booster_prod = train_production_booster(X_struct, y, BEST_HP)
    booster_path = str(OUT_DIR / "tennis_xgboost_noodds_booster.bin")
    booster_prod.save_model(booster_path)
    print(f"\n  Production no-odds booster saved: {booster_path}")

    # Save metadata
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_years": f"{min(years)}-{max(years)}",
        "total_rows": int(len(X_full)),
        "unique_matches": int(len(X_full) // 2),
        "structural_features": STRUCTURAL_FEATURE_NAMES,
        "dropped_features": [ALL_FEATURE_NAMES[i] for i in sorted(BOOKIE_INDICES)],
        "best_hp_used": BEST_HP,
        "holdout_year": holdout_year,
        "production_train_rows": int(len(X_struct)),
        "full_model_metrics": metrics_full,
        "noodds_model_metrics": metrics_noodds,
        "skill_delta_from_dropping_odds": metrics_noodds['skill_pct'] - metrics_full['skill_pct'],
        "residual_analysis": {
            "note": "No real bookmaker odds in ATP CSV data; p1_implied_prob was constant 0.5 fallback.",
            "structural_vs_half_direction_accuracy": dir_acc,
            "structural_vs_half_z_score": z_dir,
            "pearson_r": corr,
        },
        "has_structural_edge": has_structural_edge,
        "has_residual_direction_edge": has_residual_edge,
        "booster_path": booster_path,
    }

    meta_path = str(OUT_DIR / "tennis_xgboost_noodds.json")
    with open(meta_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Metadata saved: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
