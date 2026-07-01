#!/usr/bin/env python3
"""
MLB Win-Probability Model v7 — XGBoost head replacing LogReg.

Primary hypothesis: LogReg linearly applies home field advantage regardless
of Elo gap. XGBoost can capture non-linear interactions (e.g., elite away
teams vs. weak home teams).

Comparison:
  - LogReg baseline   (known: +2.08% Brier skill on 2023 holdout)
  - XGBoost all-features  (16 features from v5)
  - XGBoost core-only     (5 features)

Gate: if XGBoost 2023 skill >= 2.0%, save model as data/mlb_elo_v7_xgb.json

Usage:
    python3.11 harness/mlb_elo_v7_xgb.py 2>&1 | tee data/mlb_elo_v7_xgb.log
"""
import sys
import json
from pathlib import Path
from collections import defaultdict, deque
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'engine/python/src'))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ── Elo hyperparameters (K_PITCHER=16 optimal from HPO sweep) ─────────────────
INIT_ELO = 1500.0
K_TEAM = 20.0
K_PITCHER = 16.0    # optimal from hyperparameter sweep
HOME_FIELD = 35.0
MIN_TEAM_GAMES = 15
MIN_SP_STARTS = 5
ELO_PATH = ROOT / "models/baseball/elo/game_elo_snapshots.parquet"

TZ = {
    'BOS':0,'NYY':0,'NYM':0,'BAL':0,'TOR':0,'TB':0,'MIA':0,'ATL':0,'WSH':0,'PHI':0,'PIT':0,'CIN':0,'CLE':0,
    'CHW':1,'CHC':1,'MIN':1,'MIL':1,'STL':1,'KC':1,'HOU':1,'TEX':1,
    'COL':2,'ARI':2,
    'LAD':3,'LAA':3,'SF':3,'OAK':3,'SEA':3,'SD':3,
}
DIVISIONS = {
    'AL_East':{'BOS','NYY','BAL','TOR','TB'}, 'AL_Central':{'CHW','CLE','DET','KC','MIN'},
    'AL_West':{'HOU','LAA','OAK','SEA','TEX'}, 'NL_East':{'ATL','MIA','NYM','PHI','WSH'},
    'NL_Central':{'CHC','CIN','MIL','PIT','STL'}, 'NL_West':{'ARI','COL','LAD','SD','SF'},
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}


def expected(a, b): return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))
def brier_skill(y, p): return (1 - np.mean((np.array(p)-np.array(y))**2) / 0.25) * 100


def build_features(df, park_factor, lineup_elo_by_game,
                   K_TEAM=K_TEAM, K_PITCHER=K_PITCHER,
                   HOME_FIELD=HOME_FIELD, start_year=2010):
    """Single pass through game log — returns feature DataFrame.
    Copied verbatim from harness/mlb_sensitivity.py."""
    team_elo = defaultdict(lambda: INIT_ELO)
    pitcher_elo = defaultdict(lambda: INIT_ELO)
    team_games = defaultdict(int)
    pitcher_starts = defaultdict(int)
    team_hist = defaultdict(lambda: deque(maxlen=10))
    last_game_date = {}
    last_sp_app = {}
    team_game_dates = defaultdict(lambda: deque(maxlen=7))
    current_streak = defaultdict(int)
    dh_days = defaultdict(set)
    games_per_day = {}
    series_tracker = {}
    h2h_record = defaultdict(lambda: [0, 0])   # (ht,at) -> [wins, games]
    # bullpen usage: team -> deque of (date, relief_innings_proxy)
    bullpen_usage = defaultdict(lambda: deque(maxlen=3))
    road_trip = defaultdict(int)   # consecutive road games
    home_run = defaultdict(int)    # consecutive home games

    rows = []
    prev_year = None

    for _, r in df.iterrows():
        year = int(r['year'])
        ht, at = str(r['home_team']), str(r['away_team'])
        hsp = str(r.get('home_pitcher') or 'unknown')
        asp = str(r.get('away_pitcher') or 'unknown')
        hs = float(r['home_score']) if pd.notna(r['home_score']) else None
        as_ = float(r['away_score']) if pd.notna(r['away_score']) else None
        hw = bool(r['home_win']) if pd.notna(r['home_win']) else None
        gdate = r['date']
        gpk = r.get('game_pk')
        if hs is None or hw is None:
            continue

        if prev_year is not None and year != prev_year:
            for t in list(team_elo): team_elo[t] = team_elo[t]*2/3 + INIT_ELO/3
            for p in list(pitcher_elo): pitcher_elo[p] = pitcher_elo[p]*0.8 + INIT_ELO*0.2
            team_games.clear(); pitcher_starts.clear(); team_hist.clear()
            last_game_date.clear(); last_sp_app.clear(); team_game_dates.clear()
            current_streak.clear(); dh_days.clear(); games_per_day.clear()
            series_tracker.clear(); h2h_record.clear(); bullpen_usage.clear()
            road_trip.clear(); home_run.clear()
        prev_year = year

        h_te, a_te = team_elo[ht], team_elo[at]
        h_spe, a_spe = pitcher_elo[hsp], pitcher_elo[asp]

        # run diff form
        def form_rd(hist):
            if not hist: return 0.5, 0.0
            return (np.mean([w*(oe/1500) for w,rd,oe in hist]),
                    np.mean([rd for w,rd,oe in hist]))
        h_form, h_rd = form_rd(team_hist[ht])
        a_form, a_rd = form_rd(team_hist[at])

        # park factor
        pf = park_factor.get(ht, 1.0)

        # SP runs-allowed diff (from SP Elo vs league avg — proxy for ERA)
        sp_ra_diff = (a_spe - h_spe) / 100.0   # higher = home SP is better vs avg

        # SP rest
        h_sp_rest = min((gdate - last_sp_app[hsp]).days if hsp in last_sp_app else 5, 6)
        a_sp_rest = min((gdate - last_sp_app[asp]).days if asp in last_sp_app else 5, 6)
        sp_rest_diff = h_sp_rest - a_sp_rest
        h_short_rest = int(h_sp_rest < 4)
        a_short_rest = int(a_sp_rest < 4)
        sp_short_rest_diff = float(a_short_rest - h_short_rest)

        # Schedule density
        h_dens = sum(1 for d in team_game_dates[ht] if (gdate-d).days <= 7)
        a_dens = sum(1 for d in team_game_dates[at] if (gdate-d).days <= 7)

        # Series game number
        sk = (ht, at)
        ps = series_tracker.get(sk)
        sgn = 1
        if ps and (gdate - ps['last']).days <= 1:
            sgn = min(ps['num'] + 1, 3)

        # Streaks
        win_streak_diff = float(current_streak[ht] - current_streak[at])

        # H2H this season
        h2h = h2h_record[(ht,at)]
        h2h_wr = (h2h[0]/h2h[1] - 0.5) if h2h[1] >= 3 else 0.0

        # Timezone
        tz_ch = float(TZ.get(ht,1) - TZ.get(at,1))

        # Doubleheader yesterday
        yesterday = gdate - timedelta(1)
        h_dh = int(yesterday in dh_days.get(ht, set()))
        a_dh = int(yesterday in dh_days.get(at, set()))

        # Road trip fatigue (consecutive away games for the visitor)
        rt_away = road_trip.get(at, 0)
        rt_home = home_run.get(ht, 0)
        road_trip_diff = float(rt_away - rt_home) / 7.0

        # Bullpen drain proxy: total runs in last 3 games (more runs = longer game = more bullpen)
        def bullpen_drain(team):
            usage = bullpen_usage[team]
            if not usage: return 0.0
            return np.mean([r for d,r in usage if (gdate-d).days <= 3])
        h_bp = bullpen_drain(ht)
        a_bp = bullpen_drain(at)
        bullpen_drain_diff = float(a_bp - h_bp)

        # Tape study: games this season vs this specific opponent (familiarity)
        opp_games_h = sum(1 for x in team_hist[ht] if True)  # proxy: total games seen
        tape_diff = float(min(team_games[ht], 50) - min(team_games[at], 50)) / 50.0

        # Lineup Elo
        le = lineup_elo_by_game.get(gpk, {})
        lineup_diff = le.get('home', 1500.0) - le.get('away', 1500.0)

        # Division
        is_div = float(TEAM_DIV.get(ht) == TEAM_DIV.get(at) and ht in TEAM_DIV)

        past_warmup = (
            team_games[ht] >= MIN_TEAM_GAMES and team_games[at] >= MIN_TEAM_GAMES
            and (pitcher_starts[hsp] >= MIN_SP_STARTS or hsp=='unknown')
            and (pitcher_starts[asp] >= MIN_SP_STARTS or asp=='unknown')
        )

        if year >= start_year and past_warmup:
            rows.append({
                'year': year, 'date': str(gdate)[:10],
                'home_team': ht, 'away_team': at,
                # CORE features
                'team_elo_diff': h_te - a_te,
                'rd_diff': h_rd - a_rd,
                'sp_ra_diff': sp_ra_diff,
                'bullpen_drain_diff': bullpen_drain_diff,
                # SCHEDULE features
                'sp_short_rest_diff': sp_short_rest_diff,
                'road_trip_fatigue_diff': road_trip_diff,
                'schedule_density_diff': float(h_dens - a_dens),
                'series_game_num': float(sgn),
                'dh_diff': float(a_dh - h_dh),
                'tz_change': tz_ch,
                # MATCHUP features
                'win_streak_diff': win_streak_diff,
                'h2h_season_diff': h2h_wr,
                'tape_study_diff': tape_diff,
                'park_factor': pf,
                'lineup_elo_diff': lineup_diff,
                'is_division': is_div,
                'home_field': 1.0,
                # v3 features (for comparison)
                'sp_elo_diff': float(h_spe - a_spe),
                'form_diff': float(h_form - a_form),
                'rest_diff': float(
                    min((gdate-last_game_date[ht]).days if ht in last_game_date else 3, 5) -
                    min((gdate-last_game_date[at]).days if at in last_game_date else 3, 5)
                ),
                'home_win': int(hw),
            })

        # Updates
        rd = hs - as_
        mk = K_TEAM * np.log1p(abs(rd))
        h_exp = expected(h_te, a_te)
        actual = 1.0 if hw else 0.0
        team_elo[ht] = h_te + mk*(actual - h_exp)
        team_elo[at] = a_te + mk*((1-actual) - (1-h_exp))
        def perf(ra): return 1.0/(1.0+np.exp((ra-4.3)/2.5))
        pitcher_elo[hsp] = h_spe + K_PITCHER*(perf(as_) - expected(h_spe, a_te))
        pitcher_elo[asp] = a_spe + K_PITCHER*(perf(hs) - expected(a_spe, h_te))

        team_hist[ht].append((actual, rd, a_te)); team_hist[at].append((1-actual,-rd,h_te))
        last_game_date[ht] = gdate; last_game_date[at] = gdate
        team_games[ht] += 1; team_games[at] += 1
        pitcher_starts[hsp] += 1; pitcher_starts[asp] += 1
        last_sp_app[hsp] = gdate; last_sp_app[asp] = gdate
        team_game_dates[ht].append(gdate); team_game_dates[at].append(gdate)
        bullpen_usage[ht].append((gdate, hs + as_))
        bullpen_usage[at].append((gdate, hs + as_))

        if hw:
            current_streak[ht] = max(current_streak[ht],0)+1
            current_streak[at] = min(current_streak[at],0)-1
        else:
            current_streak[ht] = min(current_streak[ht],0)-1
            current_streak[at] = max(current_streak[at],0)+1

        series_tracker[sk] = {'num': sgn, 'last': gdate}
        h2h_record[(ht,at)][0] += int(hw); h2h_record[(ht,at)][1] += 1

        dpk = (ht, str(gdate)[:10])
        dpk_a = (at, str(gdate)[:10])
        games_per_day[dpk] = games_per_day.get(dpk,0)+1
        games_per_day[dpk_a] = games_per_day.get(dpk_a,0)+1
        if games_per_day[dpk] >= 2: dh_days[ht].add(gdate)
        if games_per_day[dpk_a] >= 2: dh_days[at].add(gdate)

        # Road trip tracking
        road_trip[at] = road_trip.get(at,0)+1; home_run[at] = 0
        home_run[ht] = home_run.get(ht,0)+1; road_trip[ht] = 0

    return pd.DataFrame(rows), team_elo, pitcher_elo


# ── Feature sets ───────────────────────────────────────────────────────────────
V5_FEATURES = ['team_elo_diff','rd_diff','sp_ra_diff','bullpen_drain_diff',
               'sp_short_rest_diff','road_trip_fatigue_diff','schedule_density_diff',
               'series_game_num','win_streak_diff','h2h_season_diff',
               'tape_study_diff','park_factor','tz_change','dh_diff',
               'lineup_elo_diff','home_field']

CORE_ONLY = ['team_elo_diff', 'rd_diff', 'sp_ra_diff', 'win_streak_diff', 'bullpen_drain_diff']


def eval_model(model, X_test, y_test, _is_xgb=False):
    """Return (skill, accuracy, z, p) for a fitted model."""
    p = model.predict_proba(X_test)[:,1]
    y = np.array(y_test)
    skill = brier_skill(y, p)
    acc = float((p >= 0.5) == y).mean() if hasattr(y, '__len__') else float(int(p >= 0.5) == int(y))
    acc = ((p >= 0.5) == y).mean()
    n = len(y)
    z = (acc - 0.5) * np.sqrt(n) / np.sqrt(0.25)
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return skill, acc, z, p_val


def train_logreg(train_df, feats):
    """Fit a logistic regression on train_df, return fitted model + scaler params."""
    tr = train_df[feats].fillna(0)
    mu = tr.mean()
    sd = tr.std().replace(0, 1)
    Xtr = (tr - mu) / sd
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(Xtr, train_df['home_win'])
    return clf, mu, sd


def train_xgb(train_df, feats, n_estimators=300, max_depth=4,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              random_state=42, early_stopping_rounds=20):
    """Fit XGBClassifier with last-20%-of-training as validation set."""
    tr = train_df[feats].fillna(0).values
    y = train_df['home_win'].values.astype(int)

    # Split: last 20% of training as val (temporal order preserved)
    n_val = max(1, int(len(tr) * 0.20))
    X_tr, X_val = tr[:-n_val], tr[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def main():
    print("=== MLB v7 XGBoost Comparison ===\n", flush=True)

    # Load game log
    df = pd.read_parquet(ELO_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'game_pk']).reset_index(drop=True)
    print(f"Loaded {len(df)} games", flush=True)

    # Park factors (pre-2023, holdout-safe)
    hist = df[df['year'] < 2023].copy()
    hist['tr'] = hist['home_score'] + hist['away_score']
    lg_avg = hist['tr'].mean()
    park_factor = (hist.groupby('home_team')['tr'].mean() / lg_avg).to_dict()

    # Lineup Elo (optional — skip if file missing)
    lineup_elo_by_game = {}
    lp = ROOT / "data/mlb_lineups_2026_backfill.jsonl"
    if lp.exists():
        bwar_path = ROOT / "models/baseball/raw/bref/bwar_batting_all.parquet"
        if bwar_path.exists():
            bwar = pd.read_parquet(bwar_path)
            bwar_2025 = bwar[bwar['year_ID'] == 2025]
            batter_init = {
                row['name_common']: float(np.clip(row.get('WAR_off', 0)*5+1500, 1300, 1700))
                for _, row in bwar_2025.iterrows() if pd.notna(row.get('WAR_off'))
            }
            for line in lp.read_text().strip().split('\n'):
                try:
                    rec = json.loads(line)
                    gpk = rec.get('game_pk')
                    h = np.mean([batter_init.get(p['name'], 1500) for p in rec.get('home_lineup', [])]) if rec.get('home_lineup') else 1500
                    a = np.mean([batter_init.get(p['name'], 1500) for p in rec.get('away_lineup', [])]) if rec.get('away_lineup') else 1500
                    lineup_elo_by_game[gpk] = {'home': h, 'away': a}
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            print(f"Lineup Elo loaded: {len(lineup_elo_by_game)} games", flush=True)

    # Build features (2010–2026) with K_PITCHER=16, K_TEAM=20, HOME_FIELD=35
    print(f"\nBuilding features (K_PITCHER={K_PITCHER}, K_TEAM={K_TEAM}, HOME_FIELD={HOME_FIELD})...", flush=True)
    data, final_team_elo, final_pitcher_elo = build_features(
        df, park_factor, lineup_elo_by_game,
        K_TEAM=K_TEAM, K_PITCHER=K_PITCHER, HOME_FIELD=HOME_FIELD, start_year=2010
    )
    print(f"Feature rows: {len(data)}", flush=True)

    # Train/test splits
    train_df = data[data['year'] <= 2022]
    test_2023 = data[data['year'] == 2023]
    test_2026 = data[data['year'] == 2026]
    print(f"Train (2010-2022): {len(train_df)} | Test 2023: {len(test_2023)} | Test 2026: {len(test_2026)}\n", flush=True)

    # ── 1. LogReg baseline ────────────────────────────────────────────────────
    print("Fitting LogReg baseline...", flush=True)
    lr_clf, lr_mu, lr_sd = train_logreg(train_df, V5_FEATURES)

    # LogReg predict
    def lr_predict(test_df, feats, mu, sd, clf):
        X = (test_df[feats].fillna(0) - mu) / sd
        return clf.predict_proba(X)[:, 1]

    lr_p23 = lr_predict(test_2023, V5_FEATURES, lr_mu, lr_sd, lr_clf)
    lr_skill23 = brier_skill(test_2023['home_win'].values, lr_p23)
    lr_acc23 = ((lr_p23 >= 0.5) == test_2023['home_win'].values).mean()
    lr_n23 = len(test_2023)
    lr_z23 = (lr_acc23 - 0.5) * np.sqrt(lr_n23) / np.sqrt(0.25)
    lr_p23_val = 2 * (1 - stats.norm.cdf(abs(lr_z23)))

    lr_p26 = lr_predict(test_2026, V5_FEATURES, lr_mu, lr_sd, lr_clf)
    lr_skill26 = brier_skill(test_2026['home_win'].values, lr_p26)
    lr_acc26 = ((lr_p26 >= 0.5) == test_2026['home_win'].values).mean()
    lr_n26 = len(test_2026)
    lr_z26 = (lr_acc26 - 0.5) * np.sqrt(lr_n26) / np.sqrt(0.25)
    lr_p26_val = 2 * (1 - stats.norm.cdf(abs(lr_z26)))

    # ── 2. XGBoost all-features ───────────────────────────────────────────────
    print("Fitting XGBoost (all features)...", flush=True)
    xgb_all = train_xgb(train_df, V5_FEATURES)
    xgb_all_p23 = xgb_all.predict_proba(test_2023[V5_FEATURES].fillna(0).values)[:, 1]
    xgb_all_skill23 = brier_skill(test_2023['home_win'].values, xgb_all_p23)
    xgb_all_acc23 = ((xgb_all_p23 >= 0.5) == test_2023['home_win'].values).mean()
    xgb_all_n23 = len(test_2023)
    xgb_all_z23 = (xgb_all_acc23 - 0.5) * np.sqrt(xgb_all_n23) / np.sqrt(0.25)
    xgb_all_p23_val = 2 * (1 - stats.norm.cdf(abs(xgb_all_z23)))

    xgb_all_p26 = xgb_all.predict_proba(test_2026[V5_FEATURES].fillna(0).values)[:, 1]
    xgb_all_skill26 = brier_skill(test_2026['home_win'].values, xgb_all_p26)
    xgb_all_acc26 = ((xgb_all_p26 >= 0.5) == test_2026['home_win'].values).mean()
    xgb_all_n26 = len(test_2026)
    xgb_all_z26 = (xgb_all_acc26 - 0.5) * np.sqrt(xgb_all_n26) / np.sqrt(0.25)
    xgb_all_p26_val = 2 * (1 - stats.norm.cdf(abs(xgb_all_z26)))

    # ── 3. XGBoost core-only ──────────────────────────────────────────────────
    print("Fitting XGBoost (core-only features)...", flush=True)
    xgb_core = train_xgb(train_df, CORE_ONLY)
    xgb_core_p23 = xgb_core.predict_proba(test_2023[CORE_ONLY].fillna(0).values)[:, 1]
    xgb_core_skill23 = brier_skill(test_2023['home_win'].values, xgb_core_p23)
    xgb_core_acc23 = ((xgb_core_p23 >= 0.5) == test_2023['home_win'].values).mean()
    xgb_core_n23 = len(test_2023)
    xgb_core_z23 = (xgb_core_acc23 - 0.5) * np.sqrt(xgb_core_n23) / np.sqrt(0.25)
    xgb_core_p23_val = 2 * (1 - stats.norm.cdf(abs(xgb_core_z23)))

    xgb_core_p26 = xgb_core.predict_proba(test_2026[CORE_ONLY].fillna(0).values)[:, 1]
    xgb_core_skill26 = brier_skill(test_2026['home_win'].values, xgb_core_p26)
    xgb_core_acc26 = ((xgb_core_p26 >= 0.5) == test_2026['home_win'].values).mean()
    xgb_core_n26 = len(test_2026)
    xgb_core_z26 = (xgb_core_acc26 - 0.5) * np.sqrt(xgb_core_n26) / np.sqrt(0.25)
    xgb_core_p26_val = 2 * (1 - stats.norm.cdf(abs(xgb_core_z26)))

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*80, flush=True)
    print("COMPARISON TABLE — 2023 PRE-REGISTERED HOLDOUT", flush=True)
    print("="*80, flush=True)
    hdr = f"{'Model':35s} {'2023 skill':>12} {'2023 acc':>10} {'z':>7} {'p':>8}"
    print(hdr, flush=True)
    print("-"*80, flush=True)

    rows_table = [
        ("LogReg v5 (baseline)",    lr_skill23,        lr_acc23,        lr_z23,        lr_p23_val),
        ("XGBoost v7 all-features", xgb_all_skill23,   xgb_all_acc23,   xgb_all_z23,   xgb_all_p23_val),
        ("XGBoost v7 core-only",    xgb_core_skill23,  xgb_core_acc23,  xgb_core_z23,  xgb_core_p23_val),
    ]
    for name, skill, acc, z, pv in rows_table:
        print(f"{name:35s} {skill:+11.2f}% {acc:9.1%} {z:7.2f} {pv:8.4f}", flush=True)

    print("\n" + "="*80, flush=True)
    print("COMPARISON TABLE — 2026 LIVE TEST", flush=True)
    print("="*80, flush=True)
    print(hdr, flush=True)
    print("-"*80, flush=True)

    rows_table26 = [
        ("LogReg v5 (baseline)",    lr_skill26,        lr_acc26,        lr_z26,        lr_p26_val),
        ("XGBoost v7 all-features", xgb_all_skill26,   xgb_all_acc26,   xgb_all_z26,   xgb_all_p26_val),
        ("XGBoost v7 core-only",    xgb_core_skill26,  xgb_core_acc26,  xgb_core_z26,  xgb_core_p26_val),
    ]
    for name, skill, acc, z, pv in rows_table26:
        print(f"{name:35s} {skill:+11.2f}% {acc:9.1%} {z:7.2f} {pv:8.4f}", flush=True)

    # ── Feature importances ───────────────────────────────────────────────────
    print("\n" + "="*80, flush=True)
    print("XGBoost Feature Importances (all-features model, by gain)", flush=True)
    print("="*80, flush=True)
    importances = xgb_all.get_booster().get_score(importance_type='gain')
    imp_sorted = sorted(importances.items(), key=lambda x: -x[1])
    # Map f0, f1... back to feature names
    feat_map = {f'f{i}': name for i, name in enumerate(V5_FEATURES)}
    for k, v in imp_sorted:
        fname = feat_map.get(k, k)
        print(f"  {fname:35s} {v:.4f}", flush=True)

    # ── Gate verdict ──────────────────────────────────────────────────────────
    print("\n" + "="*80, flush=True)
    best_xgb_skill = max(xgb_all_skill23, xgb_core_skill23)
    best_xgb_name = "XGBoost all-features" if xgb_all_skill23 >= xgb_core_skill23 else "XGBoost core-only"
    best_xgb_model = xgb_all if xgb_all_skill23 >= xgb_core_skill23 else xgb_core
    best_xgb_feats = V5_FEATURES if xgb_all_skill23 >= xgb_core_skill23 else CORE_ONLY

    gate_cleared = best_xgb_skill >= 2.0
    print(f"Best XGBoost model: {best_xgb_name}", flush=True)
    print(f"Best XGBoost 2023 skill: {best_xgb_skill:+.2f}%", flush=True)
    print(f"Gate threshold: 2.0%", flush=True)
    print(f"Gate cleared: {gate_cleared}", flush=True)

    if gate_cleared:
        print(f"\nGATE CLEARED — saving model as data/mlb_elo_v7_xgb.json", flush=True)

        # Build importances dict with feature names
        imp_named = {}
        for k, v in importances.items():
            fname = feat_map.get(k, k)
            imp_named[fname] = float(v)

        # Extract XGBoost model config
        booster = best_xgb_model.get_booster()
        model_json = {
            "version": "v7_xgb",
            "model_type": "XGBClassifier",
            "gate_cleared": True,
            "features": best_xgb_feats,
            "hyperparameters": {
                "n_estimators": int(best_xgb_model.n_estimators),
                "best_iteration": int(best_xgb_model.best_iteration) if hasattr(best_xgb_model, 'best_iteration') and best_xgb_model.best_iteration is not None else None,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "early_stopping_rounds": 20,
            },
            "elo_params": {
                "K_TEAM": K_TEAM,
                "K_PITCHER": K_PITCHER,
                "HOME_FIELD": HOME_FIELD,
            },
            "performance": {
                "holdout_2023_skill": float(best_xgb_skill),
                "holdout_2023_acc": float(xgb_all_acc23 if xgb_all_skill23 >= xgb_core_skill23 else xgb_core_acc23),
                "live_2026_skill": float(xgb_all_skill26 if xgb_all_skill23 >= xgb_core_skill23 else xgb_core_skill26),
                "logreg_baseline_skill": 2.08,
            },
            "feature_importances_gain": imp_named,
            "booster_dump": booster.save_model(str(ROOT / "data/mlb_elo_v7_xgb_booster.bin")) or "saved",
        }

        (ROOT / "data/mlb_elo_v7_xgb.json").write_text(json.dumps(model_json, indent=2))
        print(f"Model saved to data/mlb_elo_v7_xgb.json", flush=True)
        print(f"Booster saved to data/mlb_elo_v7_xgb_booster.bin", flush=True)
    else:
        print(f"\nGATE NOT CLEARED (best XGBoost skill {best_xgb_skill:+.2f}% < 2.0%)", flush=True)
        print(f"LogReg v5 remains the production model at +2.08%", flush=True)

    print("\n=== Done ===\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
