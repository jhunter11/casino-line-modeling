#!/usr/bin/env python3
"""
KXMLBKS pitcher-strikeout engine v2 — principled per-PA structural rebuild.

This is the LEAD-sleeve rebuild for task t179. It replaces the curve-fit-prone
single-lambda Poisson baseline (`pitcher-k-baseline-v0`, model_id) with a
per-plate-appearance structural model:

  For a starter expected to face E[BF] batters, we walk the opposing lineup
  in batting order (slot 1..9, wrapping for times-through-order) and assign
  each expected PA to the batter in that slot. For PA i:

      p_K(i) = log5(K_pitcher, K_batter_platoon, K_league)
               * stuff_mult                     (CSW%-based, open-data Stuff proxy)
               * fatigue(cum_pa_i)               (SMOOTH logistic decay, NOT a TTO dummy)

  The pitcher's strikeout total is then the sum of independent Bernoulli(p_K(i)),
  so P(K >= t) is the upper tail of a **Poisson-binomial** distribution (exact
  DFT-free convolution). This captures per-matchup heterogeneity that a single
  Poisson lambda smears out, and makes the model structural rather than a fit to
  one day's outcomes.

Stuff signal:
  tjStuff+ (tnestico) is a *model* that requires per-pitch release physics
  (release_speed, pfx_x/z, spin, release pos) which our cached Statcast does NOT
  carry. Running it is a Tier-2 data fetch, out of scope for t179 today. Per the
  task's explicit fallback, we use **CSW%** (Called Strikes + Whiffs / pitches),
  the single best open-data one-number K-stabilizing proxy (matrix M06), computed
  from the pitch-level `description` column we already cache. Documented gap.

Leakage doctrine (identical to v0):
  - All pitcher/batter/CSW rates are as-of strictly < game_date (prior season +
    2026 trailing). Market mid is comparison baseline only, never a feature.
  - 2023 burned (not in cache anyway). Lineups filed pre-first-pitch.

Backtest evidence is vs the MARKET-implied line (Brier skill_vs_market + CLV),
with bootstrap CI and a single-pitcher concentration check (the lesson from the
bf_2026 RETRY: 92% of edge from 5 contracts / one George Kirby game).

Usage:
  python3 harness/mlb_kprop_engine_v2.py backtest          # offline, settled days
  python3 harness/mlb_kprop_engine_v2.py wire-forward       # append v2 preds to paper ledger
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse proven helpers from the v0 module (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlb_kprop_model import (  # noqa: E402
    DATA, SC_CACHE, LINEUP_DIR,
    LEAGUE_K_RATE_FALLBACK, KALSHI_FEE_RATE,
    MONTHS,
    parse_ticker, player_token_from_name, norm,
    brier_score, brier_skill, expected_calibration_error,
)

MODEL_ID = "pitcher-k-perpa-v2"
PROP_QUOTES = DATA / "mlb_prop_quotes.jsonl"
PAPER_LEDGER = DATA / "mlb_kprop_paper_ledger.jsonl"

# Engine hyperparameters (set from priors / pitching literature, NOT fit to the
# backtest day — that would re-introduce the curve-fit failure mode).
EB_PRIOR_PA_PITCH = 200      # shrinkage prior weight, pitcher K rate
EB_PRIOR_PA_BAT = 150        # shrinkage prior weight, batter K rate
BLEND_W_2026 = 0.4           # weight on 2026 trailing vs 2025 prior season
PLATOON_PRIOR_PA = 120       # shrinkage prior for a batter's platoon split
CSW_BETA = 0.6               # stuff elasticity: stuff_mult = (csw/csw_lg)^beta, shrunk
CSW_PRIOR_PITCHES = 300      # shrinkage prior weight on CSW toward league CSW
CSW_LEAGUE = 0.285           # ~ MLB CSW% (called+whiff / pitches)
# Smooth fatigue: K-effectiveness decays late as cumulative PA grows. Modeled as a
# gentle logistic on cumulative PA centered past the 2nd time through the order.
# This is a SMOOTH decay, deliberately NOT a 3rd-time-through-order step dummy
# (which the matrix flags as an artifact, M03).
FATIGUE_MAX_DROP = 0.12      # asymptotic multiplicative K-rate drop when gassed
FATIGUE_CENTER_PA = 20.0     # center of the decay (~ start of 3rd time through)
FATIGUE_SCALE_PA = 7.0       # softness of the decay (larger = smoother)


# ─── feature computation (leakage-safe, as-of < game_date) ────────────────────

def _load_cache(years: list[int]) -> pd.DataFrame:
    files = []
    for y in years:
        files += sorted(glob.glob(str(SC_CACHE / f"statcast_{y}_*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["is_k"] = df["events"].isin(["strikeout", "strikeout_double_play"])
    df["is_pa"] = df["events"].notna() & (df["events"].astype(str) != "")
    return df


_CSW_DESCS = {
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "bunt_foul_tip",
}


def compute_features(asof: pd.Timestamp) -> dict:
    """All pitcher/batter/CSW/platoon features strictly as-of < asof."""
    pitch = _load_cache([2024, 2025, 2026])
    pitch = pitch[pitch["game_date"] < asof].copy()

    # Prior-season (2025) vs trailing-2026 split for blending.
    is25 = pitch["game_date"].dt.year == 2025
    is26 = pitch["game_date"].dt.year == 2026

    league_k = pitch.loc[is25, "is_k"].sum() / max(pitch.loc[is25, "is_pa"].sum(), 1)
    league_k = float(league_k) if league_k > 0 else LEAGUE_K_RATE_FALLBACK

    pa = pitch[pitch["is_pa"]]

    def k_rate_eb(sub: pd.DataFrame, group: str, prior_pa: int) -> dict:
        g = sub.groupby(group)["is_k"].agg(["sum", "count"])
        out = {}
        for idx, row in g.iterrows():
            n, k = row["count"], row["sum"]
            out[int(idx)] = (k + prior_pa * league_k) / (n + prior_pa)
        return out

    # Pitcher K/PA: blend 2025 prior + 2026 trailing (EB-shrunk each).
    p25 = k_rate_eb(pa[is25.loc[pa.index]], "pitcher", EB_PRIOR_PA_PITCH)
    p26 = k_rate_eb(pa[is26.loc[pa.index]], "pitcher", EB_PRIOR_PA_PITCH)
    pitcher_k = {}
    for pid in set(p25) | set(p26):
        r25 = p25.get(pid, league_k)
        r26 = p26.get(pid, league_k)
        if pid in p26:
            pitcher_k[pid] = (1 - BLEND_W_2026) * r25 + BLEND_W_2026 * r26
        else:
            pitcher_k[pid] = r25

    # Batter overall K/PA (2025 + 2026 trailing combined, EB-shrunk).
    batter_k = k_rate_eb(pa, "batter", EB_PRIOR_PA_BAT)

    # Batter platoon K/PA: vs LHP and vs RHP, EB-shrunk toward the batter's own
    # overall rate (so split is a perturbation of true talent, not noise).
    plat = {}  # batter -> {'L': rate_vs_LHP, 'R': rate_vs_RHP}
    grp = pa.groupby(["batter", "p_throws"])["is_k"].agg(["sum", "count"])
    for (bid, hand), row in grp.iterrows():
        bid = int(bid)
        base = batter_k.get(bid, league_k)
        n, k = row["count"], row["sum"]
        rate = (k + PLATOON_PRIOR_PA * base) / (n + PLATOON_PRIOR_PA)
        plat.setdefault(bid, {})[hand] = rate

    # Pitcher handedness (most common p_throws).
    phand = pitch.dropna(subset=["p_throws"]).groupby("pitcher")["p_throws"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else "R"
    ).to_dict()
    phand = {int(k): v for k, v in phand.items()}

    # Pitcher CSW% (open-data Stuff proxy), EB-shrunk toward league CSW.
    pitch["is_csw"] = pitch["description"].isin(_CSW_DESCS)
    cg = pitch.groupby("pitcher")["is_csw"].agg(["sum", "count"])
    csw = {}
    for pid, row in cg.iterrows():
        n, c = row["count"], row["sum"]
        csw[int(pid)] = (c + CSW_PRIOR_PITCHES * CSW_LEAGUE) / (n + CSW_PRIOR_PITCHES)

    # Starter expected batters-faced: mean PA in games where pitcher saw >= 15 PA
    # (a "start"), 2025+2026 trailing; default 22.
    pa_per_game = pa.groupby(["pitcher", "game_pk"]).size()
    bf = {}
    for pid, n in pa_per_game.groupby(level=0):
        starts = n[n >= 15]
        if len(starts):
            bf[int(pid)] = float(starts.mean())

    # Pitcher name -> id (statcast: player_name is the pitcher in pitch rows).
    nm = pitch.dropna(subset=["player_name"]).groupby("pitcher")["player_name"].first()
    name_to_id = {}
    for pid, nameraw in nm.items():
        pid = int(pid)
        # statcast format "Last, First"
        name_to_id[norm(nameraw.replace(",", " "))] = pid
        tok = player_token_from_name(nameraw)
        if tok:
            name_to_id.setdefault(tok, pid)

    return {
        "league_k": league_k,
        "pitcher_k": pitcher_k,
        "batter_k": batter_k,
        "platoon": plat,
        "phand": phand,
        "csw": csw,
        "bf": bf,
        "name_to_id": name_to_id,
    }


# ─── per-PA structural probability ────────────────────────────────────────────

def _log5(pp: float, pb: float, pl: float) -> float:
    denom = pp * pb / pl + (1 - pp) * (1 - pb) / (1 - pl)
    if denom < 1e-9:
        return pl
    return (pp * pb / pl) / denom


def _fatigue_mult(cum_pa: float) -> float:
    """Smooth multiplicative K-rate decay vs cumulative PA (logistic).

    1.0 when fresh, declining toward (1 - FATIGUE_MAX_DROP) deep into the start.
    Deliberately continuous — NOT a times-through-order step dummy.
    """
    z = (cum_pa - FATIGUE_CENTER_PA) / FATIGUE_SCALE_PA
    s = 1.0 / (1.0 + np.exp(-z))   # 0..1
    return float(1.0 - FATIGUE_MAX_DROP * s)


def per_pa_probs(feat: dict, pitcher_id: int, lineup_ids: list[int],
                 bf_expected: float) -> tuple[list[float], dict]:
    """Return the list of per-PA K probabilities for the start, plus diagnostics."""
    league_k = feat["league_k"]
    pk = feat["pitcher_k"].get(pitcher_id, league_k)
    p_hand = feat["phand"].get(pitcher_id, "R")
    csw = feat["csw"].get(pitcher_id, CSW_LEAGUE)
    stuff_mult = (csw / CSW_LEAGUE) ** CSW_BETA

    lineup = [b for b in lineup_ids if b] or list(range(9))  # fallback
    n_pa = int(round(bf_expected))
    n_pa = max(9, min(n_pa, 30))

    probs = []
    for i in range(n_pa):
        bid = lineup[i % len(lineup)]
        # platoon-split batter K rate vs this pitcher's hand, fall back to overall
        pb = feat["platoon"].get(bid, {}).get(p_hand)
        if pb is None:
            pb = feat["batter_k"].get(bid, league_k)
        base = _log5(pk, pb, league_k)
        p = base * stuff_mult * _fatigue_mult(i + 1)
        probs.append(float(min(max(p, 1e-4), 0.95)))

    diag = {
        "pitcher_k": pk, "csw": csw, "stuff_mult": stuff_mult,
        "p_hand": p_hand, "n_pa": n_pa, "exp_k": float(sum(probs)),
        "mean_pa_p": float(np.mean(probs)),
    }
    return probs, diag


def poisson_binomial_tail(probs: list[float], threshold: int) -> float:
    """P(sum of Bernoulli(probs) >= threshold) via exact convolution."""
    # pmf via iterative convolution
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    if threshold <= 0:
        return 1.0
    if threshold > len(pmf) - 1:
        return 0.0
    return float(pmf[threshold:].sum())


# ─── pitcher / lineup resolution per ticker ───────────────────────────────────

def _resolve_pitcher_id(token: str, team: str, feat: dict) -> int | None:
    """Resolve a Kalshi player token (e.g. 'GKIRBY') to MLBAM id via name map."""
    n2i = feat["name_to_id"]
    if token in n2i:
        return n2i[token]
    # token is FIRSTINITIAL+LASTNAME-ish; match against tokenized names
    for k, v in n2i.items():
        if k == token:
            return v
    # last-name substring fallback (token minus first char)
    last = token[1:]
    cands = [v for k, v in n2i.items() if len(k) > 1 and k[1:] == last]
    if len(set(cands)) == 1:
        return cands[0]
    # looser: lastname contained
    cands = [v for k, v in n2i.items() if last and last in k]
    if len(set(cands)) == 1:
        return cands[0]
    return None


def _load_lineups(date_str: str) -> dict[int, dict]:
    """game_pk -> {home/away lineup id lists, teams}.

    The capture file holds many snapshots per game across the day; early ones have
    empty lineups ("Scheduled"). Keep, per game, the snapshot with the MOST posted
    batters so we get the final pregame lineup rather than an early empty one.
    """
    path = LINEUP_DIR / f"{date_str}.jsonl"
    out = {}
    best_fill = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                g = json.loads(line)
            except Exception:
                continue
            gpk = g.get("game_pk")
            if not gpk:
                continue
            fill = len(g.get("home_lineup") or []) + len(g.get("away_lineup") or [])
            if gpk not in out or fill > best_fill.get(gpk, -1):
                out[gpk] = g
                best_fill[gpk] = fill
    return out


def _team_lineup_index(lineups: dict) -> dict:
    """Index lineups by team abbrev -> ordered opposing-batter id list.

    Returns {team_abbrev: {'own': [ids], 'opp': [ids]}} for the day, matching on
    the 3-letter team code embedded in tickers via the lineup team names.
    """
    # Build abbrev from team full name's last word isn't reliable; instead map via
    # the lineup record's own short codes if present, else fall back to scanning
    # all games and matching pitcher team to home/away lineup.
    return lineups  # callers iterate games directly


def settle_actual_ks(date_str: str) -> dict[int, int]:
    """pitcher_id -> actual strikeouts on date_str from the statcast cache."""
    yr = int(date_str[:4]); mo = date_str[5:7]
    fp = SC_CACHE / f"statcast_{yr}_{mo}.parquet"
    if not fp.exists():
        return {}
    df = pd.read_parquet(fp)
    df = df[df["game_date"] == pd.Timestamp(date_str)]
    df["is_k"] = df["events"].isin(["strikeout", "strikeout_double_play"])
    return df.groupby("pitcher")["is_k"].sum().astype(int).to_dict()


def build_start_index(window_dates: list[str]) -> dict:
    """pitcher_id -> {date: {game_pk, team, k, n_pa}} for each day in window.

    Used to settle a quote against the day the pitcher ACTUALLY started, which is
    not always the quote's nominal label date (the prop logger captures pregame
    quotes for next-day games; the embedded ticker date can be a calendar day off
    from the pitcher's real start in the Statcast cache). We resolve each quote's
    pitcher to their nearest real SP appearance in the window. Only counts a day
    as a "start" when the pitcher saw >= 30 pitch rows (filters relief cameos).
    """
    idx: dict[int, dict] = defaultdict(dict)
    months = sorted({d[:7] for d in window_dates})
    frames = []
    for ym in months:
        yr, mo = ym[:4], ym[5:7]
        fp = SC_CACHE / f"statcast_{yr}_{mo}.parquet"
        if fp.exists():
            frames.append(pd.read_parquet(fp))
    if not frames:
        return idx
    df = pd.concat(frames, ignore_index=True)
    df = df[df["game_date"].isin([pd.Timestamp(d) for d in window_dates])].copy()
    df["is_k"] = df["events"].isin(["strikeout", "strikeout_double_play"])
    df["pitch_team"] = np.where(df["inning_topbot"] == "Top",
                                df["home_team"], df["away_team"])
    for (pid, gd), sub in df.groupby(["pitcher", "game_date"]):
        if len(sub) < 30:   # not a start
            continue
        ds = gd.strftime("%Y-%m-%d")
        # Opposing lineup approximation: the distinct batters this pitcher faced,
        # in order of first appearance (~ batting order). NOTE: this uses the
        # realized set of batters as a proxy for the pregame projected lineup,
        # because the captured lineup files for these dates hold only probable
        # SPs (empty batter lists). The opposing K-RATES are still as-of priors
        # (no outcome leakage); only the 9-batter identity is realized. Documented
        # limitation in the results writeup.
        seen, opp = set(), []
        for b in sub["batter"].tolist():
            b = int(b)
            if b not in seen:
                seen.add(b); opp.append(b)
            if len(opp) >= 9:
                break
        idx[int(pid)][ds] = {
            "game_pk": int(sub["game_pk"].iloc[0]),
            "team": str(sub["pitch_team"].mode().iloc[0]),
            "k": int(sub["is_k"].sum()),
            "n_pa": int(len(sub)),
            "opp_lineup": opp,
        }
    return idx


def _nearest_start(start_idx: dict, pid: int, target_date: str):
    """Return (date, info) for the pitcher's start nearest target_date, or None."""
    starts = start_idx.get(pid)
    if not starts:
        return None
    td = pd.Timestamp(target_date)
    best = min(starts.items(), key=lambda kv: abs((pd.Timestamp(kv[0]) - td).days))
    # only accept within +/- 1 day of the ticker's nominal date
    if abs((pd.Timestamp(best[0]) - td).days) <= 1:
        return best
    return None


# ─── prop-quote loading ───────────────────────────────────────────────────────

def load_ks_quotes(date_str: str) -> list[dict]:
    """KXMLBKS quotes for a date from the prop-quote logger (one row per strike)."""
    rows = []
    if not PROP_QUOTES.exists():
        return rows
    with open(PROP_QUOTES) as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("series") != "KXMLBKS":
                continue
            if o.get("as_of_utc", "")[:10] != date_str:
                continue
            rows.append(o)
    return rows


# ─── backtest ─────────────────────────────────────────────────────────────────

def _pitcher_game_map(date_str: str) -> dict:
    """pitcher_id -> {'game_pk', 'team'} from same-day statcast (which side pitched).

    Reliable: a pitcher's team in a game = home_team when inning_topbot=='Top'
    (home pitches to the visiting top-half), else away_team. game_pk ties the
    pitcher to a specific game so we can pull the correct opposing lineup.
    """
    yr = int(date_str[:4]); mo = date_str[5:7]
    fp = SC_CACHE / f"statcast_{yr}_{mo}.parquet"
    if not fp.exists():
        return {}
    df = pd.read_parquet(fp)
    df = df[df["game_date"] == pd.Timestamp(date_str)].copy()
    if df.empty:
        return {}
    df["pitch_team"] = np.where(df["inning_topbot"] == "Top",
                                df["home_team"], df["away_team"])
    out = {}
    for pid, sub in df.groupby("pitcher"):
        gpk = sub["game_pk"].iloc[0]
        team = sub["pitch_team"].mode().iloc[0]
        # only treat as a startable SP candidate if they saw a real workload
        out[int(pid)] = {"game_pk": int(gpk), "team": str(team), "n_pa": int(len(sub))}
    return out


def _match_game_for_token(lineups: dict, token: str, team3: str, feat: dict,
                          pgmap: dict):
    """(pitcher_id, opposing_lineup_ids, game_pk) for a ticker's pitcher token.

    Resolves pitcher id by name token, then uses same-day statcast (pgmap) to get
    the pitcher's game_pk, then pulls the OTHER side's lineup from the lineup file.
    Falls back to None if the pitcher did not actually start that day.
    """
    pid = _resolve_pitcher_id(token, team3, feat)
    if pid is None:
        return None, None, None
    info = pgmap.get(pid)
    if not info:
        return pid, None, None
    gpk = info["game_pk"]
    g = lineups.get(gpk)
    if not g:
        return pid, None, gpk
    pteam = info["team"]
    # pick the opposing lineup: the side whose team abbrev != the pitcher's team
    home_ab = _name_to_abbr(g.get("home", ""))
    away_ab = _name_to_abbr(g.get("away", ""))
    if pteam == home_ab:
        opp = [b["id"] for b in g.get("away_lineup", []) if b.get("id")]
    elif pteam == away_ab:
        opp = [b["id"] for b in g.get("home_lineup", []) if b.get("id")]
    else:
        # team-abbrev mismatch (rare alias) — default to away then home
        opp = [b["id"] for b in g.get("away_lineup", []) if b.get("id")] \
              or [b["id"] for b in g.get("home_lineup", []) if b.get("id")]
    return pid, (opp or None), gpk


def _name_to_abbr(full_name: str) -> str:
    up = (full_name or "").upper()
    for nm, ab in _TEAM_NAME_TO_ABBR.items():
        if nm in up:
            return ab
    return ""


_TEAM_NAME_TO_ABBR = {
    "ANGELS": "LAA", "ASTROS": "HOU", "ATHLETICS": "ATH", "BLUE JAYS": "TOR",
    "BRAVES": "ATL", "BREWERS": "MIL", "CARDINALS": "STL", "CUBS": "CHC",
    "DIAMONDBACKS": "AZ", "DODGERS": "LAD", "GIANTS": "SF", "GUARDIANS": "CLE",
    "MARINERS": "SEA", "MARLINS": "MIA", "METS": "NYM", "NATIONALS": "WSH",
    "ORIOLES": "BAL", "PADRES": "SD", "PHILLIES": "PHI", "PIRATES": "PIT",
    "RANGERS": "TEX", "RAYS": "TB", "RED SOX": "BOS", "REDS": "CIN",
    "ROCKIES": "COL", "ROYALS": "KC", "TIGERS": "DET", "TWINS": "MIN",
    "WHITE SOX": "CWS", "YANKEES": "NYY",
}


def _team_abbrev_match(full_name: str, team3: str) -> bool:
    for nm, ab in _TEAM_NAME_TO_ABBR.items():
        if nm in full_name:
            return ab == team3
    return False


def run_backtest(dates: list[str]) -> dict:
    print(f"\n{'='*64}\n  KXMLBKS per-PA engine v2 — offline backtest\n{'='*64}")
    # Settlement window: the quote-label dates plus +/-1 day (the prop logger's
    # captured-date can be a calendar day off from the pitcher's real start date).
    window = sorted({d for base in dates for d in (
        (pd.Timestamp(base) + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        for k in (-1, 0, 1))})
    start_idx = build_start_index(window)
    # lineups for all window days, indexed by game_pk
    lineups_all = {}
    for d in window:
        lineups_all.update(_load_lineups(d))

    all_rows = []
    for date_str in dates:
        print(f"\n[{date_str}] computing as-of features (< {date_str}) ...")
        # Features must be as-of < the pitcher's REAL start date. Real starts can
        # land on date_str-1; compute features as-of the earliest possible start
        # (date_str-1) to stay strictly leakage-free for those.
        feat = compute_features(pd.Timestamp(date_str) - pd.Timedelta(days=1))
        quotes = load_ks_quotes(date_str)
        # Dedup: keep one quote per market_ticker (latest pregame snapshot).
        # Without this, multi-snapshot days (hourly logger = 8-9 snapshots/ticker) inflate N
        # with non-independent identical predictions → fake statistical power. Mirror of
        # the wire_forward() dedup added 2026-06-17 (t148/kprop N-inflation fix).
        _by_tkr: dict[str, dict] = {}
        _skipped_post = 0
        for _q in quotes:
            _t = _q.get("market_ticker", "")
            if not _t:
                continue
            _gs = _parse_game_start_utc(_q.get("event_ticker", ""))
            _qt_str = _q.get("as_of_utc", "")
            if _gs is not None and _qt_str:
                try:
                    _qt_dt = datetime.fromisoformat(_qt_str.replace("Z", "+00:00"))
                    if _qt_dt >= _gs:
                        _skipped_post += 1
                        continue
                except (ValueError, AttributeError):
                    pass
            _prev = _by_tkr.get(_t)
            if _prev is None or (_q.get("as_of_utc") or "") >= (_prev.get("as_of_utc") or ""):
                _by_tkr[_t] = _q
        if len(_by_tkr) < len(quotes):
            print(f"  deduped {len(quotes)} quotes -> {len(_by_tkr)} unique tickers "
                  f"({_skipped_post} post-game-start dropped)")
        quotes = list(_by_tkr.values())
        print(f"  features: {len(feat['pitcher_k'])} pitchers, {len(feat['batter_k'])} batters, "
              f"league_k={feat['league_k']:.4f}; {len(quotes)} KS quotes (deduped)")

        cache = {}
        n_resolved = 0
        for q in quotes:
            tkr = q.get("market_ticker", "")
            parsed = parse_ticker(tkr)
            if not parsed:
                continue
            token, team3, thr = parsed["player_token"], parsed["pitcher_team"], parsed["threshold"]
            mid = q.get("mid")
            if mid is None:
                continue
            key = token
            if key not in cache:
                pid = _resolve_pitcher_id(token, team3, feat)
                start = _nearest_start(start_idx, pid, date_str) if pid else None
                if pid is None or start is None:
                    cache[key] = None
                else:
                    sdate, info = start
                    # Prefer the captured pregame lineup if it has batters; else
                    # fall back to the realized opposing batters (see build_start_index).
                    opp = None
                    g = lineups_all.get(info["game_pk"])
                    if g:
                        home_ab = _name_to_abbr(g.get("home", ""))
                        away_ab = _name_to_abbr(g.get("away", ""))
                        if info["team"] == home_ab:
                            opp = [b["id"] for b in g.get("away_lineup", []) if b.get("id")]
                        elif info["team"] == away_ab:
                            opp = [b["id"] for b in g.get("home_lineup", []) if b.get("id")]
                    if not opp:
                        opp = info.get("opp_lineup")
                    if not opp:
                        cache[key] = None
                    else:
                        bf = feat["bf"].get(pid, 22.0)
                        probs, diag = per_pa_probs(feat, pid, opp, bf)
                        cache[key] = (pid, probs, diag, info["k"], sdate)
            c = cache[key]
            if c is None:
                continue
            pid, probs, diag, actual_k, sdate = c
            n_resolved += 1
            model_p = poisson_binomial_tail(probs, thr)
            outcome = 1 if actual_k >= thr else 0
            all_rows.append({
                "date": date_str, "settle_date": sdate, "ticker": tkr,
                "pitcher_id": pid, "token": token, "threshold": thr,
                "model_p": model_p, "market_p": float(mid),
                "outcome": outcome, "actual_k": int(actual_k),
                "exp_k": diag["exp_k"], "stuff_mult": diag["stuff_mult"],
            })
        print(f"  resolved+settled quote-outcomes: {n_resolved}")

    return _score(all_rows)


def _score(rows: list[dict]) -> dict:
    if not rows:
        print("\n  NO settled rows — cannot score.")
        return {"n": 0, "rows": []}
    df = pd.DataFrame(rows)
    # spread/quality filter: drop degenerate 0/1 market mids (no real two-sided book)
    df = df[(df["market_p"] > 0.02) & (df["market_p"] < 0.98)].copy()
    n = len(df)
    y = df["outcome"].values.astype(float)
    pm = df["model_p"].values
    pk = df["market_p"].values

    base = float(y.mean())
    bs_base = brier_score(np.full(n, base), y)
    bs_model = brier_score(pm, y)
    bs_market = brier_score(pk, y)
    skill_vs_base = brier_skill(bs_model, bs_base)
    skill_vs_market = brier_skill(bs_model, bs_market)
    ece = expected_calibration_error(pm, y)

    # CLV proxy: model takes the side it prefers vs market mid; "edge" = model_p - market_p
    # on YES side (or market_p - model_p on NO). Realized "value" = (outcome - entry)
    # in probability terms for the side taken. This is an offline CLV-style realized
    # value, NOT settled forward CLV (which is time-gated).
    side_yes = pm > pk
    entry = np.where(side_yes, pk, 1 - pk)               # price paid for chosen side
    realized = np.where(side_yes, y, 1 - y)              # 1 if chosen side won
    edge_signed = np.where(side_yes, pm - pk, pk - pm)   # model's claimed edge
    # only "act" where claimed edge clears fee
    act = edge_signed > KALSHI_FEE_RATE
    realized_value = realized[act] - entry[act]          # per-contract realized $ (paper)
    n_act = int(act.sum())
    mean_realized = float(realized_value.mean()) if n_act else 0.0

    # bootstrap CI on skill_vs_market AND on mean realized value
    rng = np.random.default_rng(7)
    B = 5000
    skills = np.empty(B); rvs = np.empty(B)
    idx_all = np.arange(n)
    act_idx = np.where(act)[0]
    for b in range(B):
        s = rng.choice(idx_all, n, replace=True)
        ys, pms, pks = y[s], pm[s], pk[s]
        bm = brier_score(pms, ys); bk = brier_score(pks, ys)
        skills[b] = brier_skill(bm, bk)
        if len(act_idx):
            sa = rng.choice(act_idx, len(act_idx), replace=True)
            sy = np.where(pm[sa] > pk[sa], y[sa], 1 - y[sa])
            se = np.where(pm[sa] > pk[sa], pk[sa], 1 - pk[sa])
            rvs[b] = (sy - se).mean()
        else:
            rvs[b] = 0.0
    skill_ci = (float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5)))
    rv_ci = (float(np.percentile(rvs, 2.5)), float(np.percentile(rvs, 97.5)))
    p_skill_pos = float((skills > 0).mean())     # bootstrap one-sided support

    # concentration: share of total |edge mass| (acted) from the top single pitcher
    conc = {}
    if n_act:
        dfa = df.iloc[act_idx].copy()
        dfa["abs_edge"] = np.abs(edge_signed[act_idx])
        by_p = dfa.groupby("token")["abs_edge"].sum().sort_values(ascending=False)
        total = by_p.sum()
        conc = {
            "top_pitcher": str(by_p.index[0]),
            "top_share": float(by_p.iloc[0] / total) if total > 0 else 0.0,
            "top3_share": float(by_p.iloc[:3].sum() / total) if total > 0 else 0.0,
            "n_pitchers_acted": int(dfa["token"].nunique()),
        }

    res = {
        "n": n, "n_acted": n_act, "base_rate": base,
        "bs_base": bs_base, "bs_model": bs_model, "bs_market": bs_market,
        "skill_vs_base": skill_vs_base, "skill_vs_market": skill_vs_market,
        "skill_vs_market_ci95": skill_ci, "skill_vs_market_p_pos": p_skill_pos,
        "ece": ece,
        "mean_realized_value": mean_realized, "realized_value_ci95": rv_ci,
        "concentration": conc,
        "rows": rows,
    }
    _print_score(res)
    return res


def _print_score(r: dict) -> None:
    print(f"\n{'─'*64}\n  SCORE (settled offline backtest vs MARKET)\n{'─'*64}")
    print(f"  N (filtered)        : {r['n']}   (acted: {r['n_acted']})")
    print(f"  base rate (YES)     : {r['base_rate']:.3f}")
    print(f"  Brier model         : {r['bs_model']:.5f}")
    print(f"  Brier market mid    : {r['bs_market']:.5f}")
    print(f"  Brier base-rate     : {r['bs_base']:.5f}")
    print(f"  skill vs base       : {r['skill_vs_base']:+.4f}")
    print(f"  skill vs MARKET     : {r['skill_vs_market']:+.4f}   <-- binding gate")
    lo, hi = r["skill_vs_market_ci95"]
    print(f"  skill vs market CI95: [{lo:+.4f}, {hi:+.4f}]   P(skill>0)={r['skill_vs_market_p_pos']:.3f}")
    print(f"  calibration ECE     : {r['ece']:.4f}")
    rlo, rhi = r["realized_value_ci95"]
    print(f"  mean realized value : {r['mean_realized_value']:+.4f}/contract  CI95 [{rlo:+.4f}, {rhi:+.4f}]")
    c = r["concentration"]
    if c:
        print(f"  concentration       : top={c['top_pitcher']} {c['top_share']:.2%} of edge mass, "
              f"top3={c['top3_share']:.2%}, n_pitchers_acted={c['n_pitchers_acted']}")
    print(f"{'─'*64}")


# ─── forward ledger wiring ────────────────────────────────────────────────────

def _build_sp_lineup_index(lineups: dict, feat: dict) -> dict:
    """Kalshi-token -> {pid, opp:[batter ids], game_pk} from the lineup file.

    For each game, take the probable home/away SP name, resolve to MLBAM id, and
    pair with the OPPOSING side's posted lineup (batter ids in order). Keyed by the
    SP's Kalshi player token so it joins to KXMLBKS tickers.
    """
    out = {}
    for gpk, g in lineups.items():
        for sp_key, opp_key in [("home_pitcher", "away_lineup"),
                                ("away_pitcher", "home_lineup")]:
            name = g.get(sp_key)
            if not name:
                continue
            tok = player_token_from_name(name)
            pid = feat["name_to_id"].get(tok)
            if pid is None:
                # try "Last First" normalized
                pid = feat["name_to_id"].get(norm(name))
            opp = [b["id"] for b in g.get(opp_key, []) if b.get("id")]
            if tok and opp:
                out[tok] = {"pid": pid, "opp": opp, "game_pk": gpk}
    return out


def _parse_game_start_utc(event_ticker: str) -> datetime | None:
    """Parse UTC game start from a KXMLBKS event ticker.

    e.g. 'KXMLBKS-26JUN181610BALSEA' → datetime(2026, 6, 18, 16, 10, tzinfo=UTC)
    Returns None if the ticker format is unrecognised.
    """
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    ec = parts[1]
    # Format: 26JUN181610BALSEA (yy + MON(3) + dd(2) + HHMM(4) + teams)
    if len(ec) < 11:
        return None
    yy, mon, dd, hhmm = ec[:2], ec[2:5], ec[5:7], ec[7:11]
    if mon not in MONTHS:
        return None
    try:
        return datetime(2000 + int(yy), int(MONTHS[mon]), int(dd),
                        int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def wire_forward(date_str: str | None = None) -> int:
    """Append v2 dual predictions into the paper ledger alongside v0.

    Writes one row per (settled-or-pending) KXMLBKS ticker for `date_str`
    (default: today UTC), with model_id pitcher-k-perpa-v2 and the per-PA
    structural prob. Uses editlock (fail-open) for the shared-ledger write.
    Does NOT claim the forward N>=300 gate is met.
    """
    # Honor committee BLOCKs at the booking-function level (t232/friction #23):
    # wire_forward is invoked from multiple paths (cron, settle, manual), so gate
    # here rather than per-entrypoint. Return-early (NOT sys.exit) so a caller that
    # also settles open positions keeps running — a BLOCK freezes NEW exposure only.
    import strategy_gate
    _blk = strategy_gate.is_blocked("mlb_kprop")
    if _blk:
        print(f"[wire-forward] mlb_kprop is BLOCKED — skipping new forward entries. {_blk}")
        return 0
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Idempotency guard (no-clobber, cf t149): wire_forward appends unconditionally,
    # so a same-date re-run (cron retry / manual+scheduled overlap) would double-log
    # and corrupt the forward N-count. Skip if v2 rows for this date already exist.
    if PAPER_LEDGER.exists():
        for _line in open(PAPER_LEDGER):
            _line = _line.strip()
            if not _line:
                continue
            try:
                _r = json.loads(_line)
            except Exception:
                continue
            if (_r.get("model") or {}).get("model_id") == MODEL_ID and \
               (_r.get("game") or {}).get("game_date") == date_str:
                print(f"[wire-forward] {date_str}: v2 rows already present — skipping (idempotent no-op)")
                return 0
    asof = pd.Timestamp(date_str)
    feat = compute_features(asof)
    lineups = _load_lineups(date_str)
    quotes = load_ks_quotes(date_str)
    # Dedup: the prop_quote_logger captures MULTIPLE intraday snapshots per ticker
    # (~9/day). wire_forward writes one ledger row per quote, so without dedup each
    # market would be logged once per snapshot — inflating the forward N-count with
    # non-independent identical predictions (cf 06-16 phantom-N bug).
    # Keep exactly ONE forward decision per market_ticker: the latest PREGAME snapshot.
    # Post-game-start quotes are excluded: Kalshi keeps markets "open" during live play
    # so live prices reflect in-progress game state (a pitcher knocked out after 2 innings
    # has a near-zero K-line), manufacturing phantom +38pp edges and fake CLV. (t259)
    _by_tkr: dict[str, dict] = {}
    _skipped_postgame = 0
    for _q in quotes:
        _t = _q.get("market_ticker", "")
        if not _t:
            continue
        # Filter: discard quotes captured at or after game start.
        _et = _q.get("event_ticker", "")
        _gs = _parse_game_start_utc(_et)
        _qt_str = _q.get("as_of_utc", "")
        if _gs is not None and _qt_str:
            try:
                _qt_dt = datetime.fromisoformat(_qt_str.replace("Z", "+00:00"))
                if _qt_dt >= _gs:
                    _skipped_postgame += 1
                    continue
            except (ValueError, AttributeError):
                pass
        _prev = _by_tkr.get(_t)
        if _prev is None or (_q.get("as_of_utc") or "") >= (_prev.get("as_of_utc") or ""):
            _by_tkr[_t] = _q
    if _skipped_postgame:
        print(f"[wire-forward] skipped {_skipped_postgame} post-game-start quotes (t259 price guard)")
    if len(_by_tkr) != len(quotes) - _skipped_postgame:
        print(f"[wire-forward] deduped quotes {len(quotes)} -> {len(_by_tkr)} "
              f"(latest pregame snapshot per ticker; {_skipped_postgame} post-start dropped)")
    quotes = list(_by_tkr.values())
    pgmap = _pitcher_game_map(date_str)
    # Forward path: resolve probable SP + opposing lineup from the captured lineup
    # file (which, on game day, carries probable pitchers and posted lineups).
    sp_index = _build_sp_lineup_index(lineups, feat)
    print(f"[wire-forward] {date_str}: {len(quotes)} KS quotes, {len(lineups)} lineups, "
          f"{len(pgmap)} pitchers in same-day statcast, {len(sp_index)} SP-lineup entries")

    # advisory edit-lock (fail-open per task constraint)
    OWNER = "mlb_kprop_engine_v2"
    have_lock = False
    try:
        import editlock  # type: ignore
        have_lock = bool(editlock.acquire(str(PAPER_LEDGER), owner=OWNER, wait=10))
        if not have_lock:
            print("  [editlock] busy — proceeding fail-open (advisory)")
    except Exception as e:
        print(f"  [editlock] unavailable/fail-open: {e}")

    cache = {}
    written = 0
    out_lines = []
    now = datetime.now(timezone.utc).isoformat()
    for q in quotes:
        tkr = q.get("market_ticker", "")
        parsed = parse_ticker(tkr)
        if not parsed:
            continue
        token, team3, thr = parsed["player_token"], parsed["pitcher_team"], parsed["threshold"]
        mid = q.get("mid")
        if mid is None:
            continue
        key = (token, team3)
        if key not in cache:
            # Prefer the lineup-file SP index (forward, pregame). Fall back to the
            # same-day statcast match if the game already partly played.
            ent = sp_index.get(token)
            pid = ent["pid"] if ent else None
            opp = ent["opp"] if ent else None
            gpk = ent["game_pk"] if ent else None
            if pid is None or not opp:
                pid, opp, gpk = _match_game_for_token(lineups, token, team3, feat, pgmap)
            if pid is None or not opp:
                cache[key] = None
            else:
                bf = feat["bf"].get(pid, 22.0)
                probs, diag = per_pa_probs(feat, pid, opp, bf)
                cache[key] = (pid, probs, diag, gpk)
        c = cache[key]
        if c is None:
            continue
        pid, probs, diag, gpk = c
        model_p = poisson_binomial_tail(probs, thr)
        # t259: use real book prices (ask for YES, 1-bid for NO) as entry cost.
        # Mid-based edge manufactures phantom edge when the book is wide or empty
        # (e.g. empty-book bid=0.01/ask=0.97 gives mid=0.49, but real YES cost is 0.97).
        yes_ask = q.get("yes_ask")
        yes_bid = q.get("yes_bid")
        if yes_ask is None or yes_bid is None:
            # Can't compute real-book edge without both sides; fall back to mid.
            edge_yes = model_p - float(mid)
            edge_no = (1 - model_p) - (1.0 - float(mid))
        else:
            edge_yes = model_p - float(yes_ask)       # cost to buy YES = ask
            edge_no = (1 - model_p) - (1.0 - float(yes_bid))  # cost to buy NO = 1 - bid
        if edge_yes >= edge_no:
            side = "YES"
            entry_price = float(yes_ask) if yes_ask is not None else float(mid)
            edge = edge_yes
        else:
            side = "NO"
            entry_price = (1.0 - float(yes_bid)) if yes_bid is not None else (1.0 - float(mid))
            edge = edge_no
        spread = (round(yes_ask - yes_bid, 4)
                  if yes_ask is not None and yes_bid is not None else None)
        # Empty-book guard: bid=0/ask=1 is a market with no real quotes. Flag it so
        # the CLV evaluator doesn't treat a near-zero mid as a tradeable pregame price.
        price_corrupt = bool(spread is not None and spread > 0.5)
        if price_corrupt:
            edge = None
        # Non-security fingerprint for row dedup only; not a cryptographic digest.
        fh = hashlib.md5(
            f"{MODEL_ID}|{pid}|{date_str}|{diag['exp_k']:.3f}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:16]
        kalshi_block: dict = {
            "quote_ts": q.get("as_of_utc"),
            "yes_bid": yes_bid, "yes_ask": yes_ask,
            "mid": float(mid), "spread": spread,
            "open_interest": None, "volume": q.get("volume"),
            "quote_source": "prop_quote_logger",
        }
        if price_corrupt:
            kalshi_block["price_corrupt"] = True
            kalshi_block["price_note"] = "empty_book_wide_spread_t259"
        decision_status = "voided" if price_corrupt else "paper_open"
        rec = {
            "schema_version": "mlb-prop-paper-v1",
            "created_at": now, "decision_ts": now,
            "family": "KXMLBKS", "event_ticker": parsed["event_ticker"], "ticker": tkr,
            "game": {"game_date": date_str, "game_pk": gpk,
                     "away": parsed["away"], "home": parsed["home"], "scheduled_start": None},
            "player": {"name": None, "mlbam_id": int(pid), "team": team3,
                       "kalshi_player_token": token, "kalshi_player_uuid": q.get("player_id")},
            "contract": {"stat": "pitcher_strikeouts", "operator": ">=",
                         "threshold": thr, "participation_rule": "starting_pitcher_required",
                         "fair_market_price_on_scratch": True},
            "model": {"model_id": MODEL_ID, "prob_yes": round(model_p, 4),
                      "prob_no": round(1 - model_p, 4),
                      "expected_k": round(diag["exp_k"], 3),
                      "stuff_mult": round(diag["stuff_mult"], 4),
                      "n_pa": diag["n_pa"], "pitcher_k": round(diag["pitcher_k"], 4),
                      "engine": "per-PA poisson-binomial; log5+platoon+CSW-stuff+smooth-fatigue",
                      "features_hash": fh, "inputs_asof": now},
            "kalshi": kalshi_block,
            "decision": {"side": side,
                         "entry_price": None if price_corrupt else round(entry_price, 4),
                         "edge": round(edge, 4) if edge is not None else None,
                         "paper_contracts": 1, "status": decision_status,
                         "reject_reason": "void_corrupt_price_t259" if price_corrupt else "paper-ledger-observe-only"},
            "clv": {"close_quote_ts": None, "close_mid": None,
                    "entry_to_close_bps": None, "close_source": None},
            "settlement": {"settled_at": None, "actual_stat": None, "outcome_yes": None,
                           "kalshi_settlement": None, "pnl_contracts": None,
                           "settlement_source": None, "join_status": "pending"},
        }
        out_lines.append(json.dumps(rec))
        written += 1

    if out_lines:
        with open(PAPER_LEDGER, "a") as f:
            f.write("\n".join(out_lines) + "\n")
    if have_lock:
        try:
            import editlock  # type: ignore
            editlock.release(str(PAPER_LEDGER), owner=OWNER)
        except Exception:
            pass
    print(f"[wire-forward] appended {written} v2 predictions to {PAPER_LEDGER.name}")
    return written


# ─── fold-backtest helpers ────────────────────────────────────────────────────

def _discover_settled_dates() -> list[str]:
    """Return sorted dates that have KXMLBKS quotes AND statcast game coverage."""
    import glob as _glob
    quote_dates: set[str] = set()
    if PROP_QUOTES.exists():
        with open(PROP_QUOTES) as f:
            for line in f:
                try:
                    o = json.loads(line)
                    if o.get("series") == "KXMLBKS":
                        d = (o.get("as_of_utc") or "")[:10]
                        if d:
                            quote_dates.add(d)
                except Exception:
                    pass
    sc_dates: set[str] = set()
    for fp in _glob.glob(str(SC_CACHE / "statcast_*.parquet")):
        try:
            df = pd.read_parquet(fp, columns=["game_date"])
            sc_dates.update(str(d)[:10] for d in df["game_date"].dropna().unique())
        except Exception:
            pass
    return sorted(quote_dates & sc_dates)


def _make_fold_report(rows: list[dict], all_dates: list[str]) -> list[dict]:
    """Score 3 date-based folds + FULL and return a list of fold summary dicts."""
    sorted_dates = sorted(set(all_dates))
    n = len(sorted_dates)
    if n == 0:
        return []
    f1 = sorted_dates[: n // 3]
    f2 = sorted_dates[n // 3 : 2 * n // 3]
    f3 = sorted_dates[2 * n // 3 :]
    folds = [
        (f1, f"Fold A ({f1[0][5:] if f1 else '?'}..{f1[-1][5:] if f1 else '?'})"),
        (f2, f"Fold B ({f2[0][5:] if f2 else '?'}..{f2[-1][5:] if f2 else '?'})"),
        (f3, f"Fold C ({f3[0][5:] if f3 else '?'}..{f3[-1][5:] if f3 else '?'})"),
        (sorted_dates, f"FULL  ({sorted_dates[0][5:]}..{sorted_dates[-1][5:]})"),
    ]
    report = []
    for fd, label in folds:
        fd_set = set(fd)
        fd_rows = [r for r in rows if r.get("date", "") in fd_set]
        if not fd_rows:
            continue
        s = _score(fd_rows)
        report.append({
            "fold": label,
            "n": s["n"],
            "n_act": s["n_acted"],
            "base": s["base_rate"],
            "skill_vs_mkt": s["skill_vs_market"],
            "mean_rv": s["mean_realized_value"],
            "n_pitchers": len({r.get("token", "?") for r in fd_rows}),
        })
    return report


# ─── cli ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "backtest":
        dates = sys.argv[2:] or _discover_settled_dates()
        if not dates:
            print("no settled quote dates found — run mlb_prop_quote_logger.py + statcast refresh first")
            return 1
        res = run_backtest(dates)
        out = DATA / "mlb_kprop_v2_backtest.json"
        with open(out, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "rows"}, f, indent=2)
        # also dump rows for audit
        with open(DATA / "mlb_kprop_v2_backtest_rows.json", "w") as f:
            json.dump(res.get("rows", []), f, indent=2)
        print(f"\n  wrote {out.name} (+ rows)")
        return 0
    elif cmd == "backtest-folds":
        avail = _discover_settled_dates()
        dates = sys.argv[2:] or avail
        if not dates:
            print("no settled quote dates found — run mlb_prop_quote_logger.py + statcast refresh first")
            return 1
        print(f"  running fold backtest over {len(dates)} dates: {dates[0]} .. {dates[-1]}")
        res = run_backtest(dates)
        rows = res.get("rows", [])
        fold_report = _make_fold_report(rows, dates)
        fp = DATA / "mlb_kprop_v2_fold_report.json"
        with open(fp, "w") as f:
            json.dump(fold_report, f, indent=2)
        print(f"\n  wrote {fp.name}  ({len(fold_report)} folds, {len(rows)} total rows)")
        # keep main backtest artefacts in sync
        out = DATA / "mlb_kprop_v2_backtest.json"
        with open(out, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "rows"}, f, indent=2)
        with open(DATA / "mlb_kprop_v2_backtest_rows.json", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  wrote {out.name} (+ rows)")
        return 0
    elif cmd == "wire-forward":
        date_str = sys.argv[2] if len(sys.argv) > 2 else None
        wire_forward(date_str)
        return 0
    else:
        print(f"unknown command: {cmd}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
