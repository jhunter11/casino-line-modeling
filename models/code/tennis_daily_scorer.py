#!/usr/bin/env python3
"""tennis_daily_scorer.py — Score today's ATP Kalshi markets against the XGBoost no-odds model.

Pipeline:
  1. Load XGBoost booster from data/hpo/tennis_xgboost_noodds_booster.bin
  2. Build live Elo state by replaying ATP historical CSV data (2015-2024)
  3. Fetch open KXATPMATCH markets from Kalshi (no auth required)
  4. Extract player pairs from market pairs, look up Elo/rank/form state
  5. Score each match, compute edge vs Kalshi mid
  6. Signal if |edge| >= EDGE_THRESHOLD (default 0.08)
  7. Write signals to data/tennis_paper_signals.jsonl and data/tennis_scores/YYYY-MM-DD.json

Usage:
  python3 harness/tennis_daily_scorer.py               # score + write
  python3 harness/tennis_daily_scorer.py --dry-run     # print only, no writes
  python3 harness/tennis_daily_scorer.py --verbose     # extra diagnostics

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
    import xgboost as xgb
except ImportError as e:
    sys.exit(f"ERROR: Missing dependency — {e}\n  pip install numpy xgboost")

# ─────────────────────────────────────── paths

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "engine" / "data" / "tennis" / "tennis_atp" / "tennis_atp-master"
BOOSTER_PATH = ROOT / "data" / "hpo" / "tennis_xgboost_noodds_booster.bin"
META_PATH    = ROOT / "data" / "hpo" / "tennis_xgboost_noodds.json"
SIGNALS_OUT  = ROOT / "data" / "tennis_paper_signals.jsonl"
SCORES_DIR   = ROOT / "data" / "tennis_scores"

# ─────────────────────────────────────── constants

KALSHI_BASE    = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER  = "KXATPMATCH"
EDGE_THRESHOLD = 0.08        # |model_prob − kalshi_mid| >= this → signal
FEE_COEFF      = 0.07        # Kalshi fee: 0.07 * p * (1-p)
ELO_INIT       = 1500.0
ELO_K          = 32.0
SCHEMA_VERSION = "tennis-daily-scorer-v1"

# Structural feature names in model input order
STRUCTURAL_FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_log_advantage",
    "seed_advantage",
    "age_diff",
    "height_cm_diff",
    "h2h_win_pct_diff",
    "recent_win_pct_diff",
    "recent_match_count_diff",
    "days_rest_diff",
    "best_of_5",
    "is_grand_slam",
    "surface_hard",
    "surface_clay",
    "surface_grass",
    "surface_carpet",
    "surface_unknown",
]

# ─────────────────────────────────────── HTTP helper

_RETRY_DELAYS = (2, 4, 8)


def _get(url: str, timeout: int = 15) -> dict:
    """GET with retry/backoff."""
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json",
                 "User-Agent": "kalshi-tennis-scorer/1.0"},
    )
    last_exc: Optional[Exception] = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:
            last_exc = exc
            if delay is None:
                break
            print(f"  _get attempt {attempt} failed ({exc}); retry in {delay}s")
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────── Elo / feature helpers

def _surface_key(surface: str) -> str:
    v = (surface or "").strip().lower()
    return v if v in {"hard", "clay", "grass", "carpet"} else "unknown"


def _elo_update(winner_elo: float, loser_elo: float, k: float = ELO_K):
    e = 1.0 / (1.0 + 10.0 ** ((loser_elo - winner_elo) / 400.0))
    d = k * (1.0 - e)
    return winner_elo + d, loser_elo - d


def _fn(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _parse_date(v) -> Optional[date]:
    s = str(v or "").strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    if s:
        try:
            return datetime.fromisoformat(s).date()
        except ValueError:
            pass
    return None


def _rank_log_adv(r1, r2, default: int = 2500) -> float:
    r1v = int(float(r1)) if r1 else default
    r2v = int(float(r2)) if r2 else default
    return math.log1p(r2v) - math.log1p(r1v)


def _smoothed(wins: int, matches: int) -> float:
    return (wins + 2.5) / (matches + 5.0)


def _name_tokens(name: str) -> list[str]:
    """Return ASCII-ish name tokens for stable player matching."""
    normalized = unicodedata.normalize("NFKD", name or "")
    ascii_name = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return [t for t in re.sub(r"[^a-z0-9]+", " ", ascii_name.lower()).split() if t]


# ─────────────────────────────────────── state builder

class TennisState:
    """Incremental ATP Elo + feature state from historical CSV data."""

    def __init__(self) -> None:
        self.overall_elo: dict[str, float] = defaultdict(lambda: ELO_INIT)
        self.surface_elo: dict[tuple, float] = defaultdict(lambda: ELO_INIT)
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.last_played: dict[str, date] = {}
        self.h2h: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Player metadata (most recent seen values)
        self.player_rank: dict[str, float] = {}
        self.player_age: dict[str, float] = {}
        self.player_ht: dict[str, float] = {}
        self.player_name_to_id: dict[str, str] = {}  # lower_name → id
        self.player_id_to_name: dict[str, str] = {}  # id → display name
        self.rows_processed: int = 0

    def process_csv(self, path: Path, verbose: bool = False) -> None:
        """Replay one year's CSV, updating all state."""
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        rows.sort(key=lambda r: (
            _parse_date(r.get("tourney_date")) or date(2000, 1, 1),
            r.get("tourney_id", ""),
            _fn(r.get("match_num", 0)),
        ))

        for row in rows:
            d = _parse_date(row.get("tourney_date"))
            if d is None:
                continue

            winner = str(row.get("winner_id", "") or "").strip()
            loser  = str(row.get("loser_id", "")  or "").strip()
            if not winner or not loser:
                continue

            surface  = str(row.get("surface") or "Unknown")
            surf_key = _surface_key(surface)

            # Update name→id and id→name maps
            wname = str(row.get("winner_name", "") or "").strip()
            lname = str(row.get("loser_name",  "") or "").strip()
            if wname:
                self.player_name_to_id[wname.lower()] = winner
                self.player_id_to_name[winner] = wname
            if lname:
                self.player_name_to_id[lname.lower()] = loser
                self.player_id_to_name[loser]  = lname

            # Update metadata
            wr = _fn(row.get("winner_rank"), None)  # type: ignore[arg-type]
            lr = _fn(row.get("loser_rank"),  None)  # type: ignore[arg-type]
            if wr:
                self.player_rank[winner] = wr
            if lr:
                self.player_rank[loser]  = lr

            wa = _fn(row.get("winner_age"), None)  # type: ignore[arg-type]
            la = _fn(row.get("loser_age"),  None)  # type: ignore[arg-type]
            if wa:
                self.player_age[winner] = wa
            if la:
                self.player_age[loser]  = la

            wh = _fn(row.get("winner_ht"), None)  # type: ignore[arg-type]
            lh = _fn(row.get("loser_ht"),  None)  # type: ignore[arg-type]
            if wh:
                self.player_ht[winner] = wh
            if lh:
                self.player_ht[loser]  = lh

            # Update Elo
            we, le = _elo_update(self.overall_elo[winner], self.overall_elo[loser])
            self.overall_elo[winner] = we
            self.overall_elo[loser]  = le

            ws_e, ls_e = _elo_update(
                self.surface_elo[(winner, surf_key)],
                self.surface_elo[(loser, surf_key)],
            )
            self.surface_elo[(winner, surf_key)] = ws_e
            self.surface_elo[(loser,  surf_key)] = ls_e

            # Update h2h
            h2h_key = tuple(sorted((winner, loser)))
            self.h2h[h2h_key][winner] += 1

            # Update recent
            self.recent[winner].append(1)
            self.recent[loser].append(0)

            # Update last played
            self.last_played[winner] = d
            self.last_played[loser]  = d

            self.rows_processed += 1

        if verbose:
            print(f"    processed {path.name}: {len(rows)} rows")

    def get_features(
        self,
        p1_id: str,
        p2_id: str,
        surface: str,
        best_of: int,
        tourney_level: str,
        today: date,
    ) -> list[float]:
        """Compute the 17 structural features for p1 (YES side)."""
        surf_key = _surface_key(surface)

        p1_elo = self.overall_elo.get(p1_id, ELO_INIT)
        p2_elo = self.overall_elo.get(p2_id, ELO_INIT)
        p1_selo = self.surface_elo.get((p1_id, surf_key), ELO_INIT)
        p2_selo = self.surface_elo.get((p2_id, surf_key), ELO_INIT)

        p1_rank = self.player_rank.get(p1_id)
        p2_rank = self.player_rank.get(p2_id)
        p1_age  = self.player_age.get(p1_id, 0.0)
        p2_age  = self.player_age.get(p2_id, 0.0)
        p1_ht   = self.player_ht.get(p1_id, 0.0)
        p2_ht   = self.player_ht.get(p2_id, 0.0)

        h2h_key = tuple(sorted((p1_id, p2_id)))
        wins = self.h2h.get(h2h_key, {})
        p1_h2hw = wins.get(p1_id, 0)
        p2_h2hw = wins.get(p2_id, 0)
        h2h_tot = p1_h2hw + p2_h2hw
        h2h_p1  = (p1_h2hw + 0.5) / (h2h_tot + 1.0)

        p1_rec = list(self.recent.get(p1_id, []))
        p2_rec = list(self.recent.get(p2_id, []))
        p1_wpc = _smoothed(sum(p1_rec), len(p1_rec))
        p2_wpc = _smoothed(sum(p2_rec), len(p2_rec))

        def _days_rest(pid: str) -> float:
            lp = self.last_played.get(pid)
            if lp is None:
                return 0.0
            return float(max(min((today - lp).days, 30), 0))

        is_grand_slam = 1.0 if str(tourney_level).upper() == "G" else 0.0

        return [
            p1_elo - p2_elo,                         # elo_diff
            p1_selo - p2_selo,                        # surface_elo_diff
            _rank_log_adv(p1_rank, p2_rank),          # rank_log_advantage
            0.0,                                       # seed_advantage (unknown for today)
            p1_age - p2_age,                           # age_diff
            p1_ht - p2_ht,                             # height_cm_diff
            (2.0 * h2h_p1) - 1.0,                     # h2h_win_pct_diff
            p1_wpc - p2_wpc,                           # recent_win_pct_diff
            float(len(p1_rec) - len(p2_rec)),          # recent_match_count_diff
            _days_rest(p1_id) - _days_rest(p2_id),    # days_rest_diff
            1.0 if best_of >= 5 else 0.0,             # best_of_5
            is_grand_slam,                             # is_grand_slam
            1.0 if surf_key == "hard"    else 0.0,    # surface_hard
            1.0 if surf_key == "clay"    else 0.0,    # surface_clay
            1.0 if surf_key == "grass"   else 0.0,    # surface_grass
            1.0 if surf_key == "carpet"  else 0.0,    # surface_carpet
            1.0 if surf_key == "unknown" else 0.0,    # surface_unknown
        ]

    def resolve_player(self, display_name: str) -> Optional[str]:
        """Fuzzy-match a Kalshi display name to a player ID.

        Strategy:
          1. Exact full name match (case-insensitive)
          2. Token subset match (handles punctuation/hyphens/diacritics)
          3. Last two-token suffix match (handles 'de Minaur' → 'Alex De Minaur')
        """
        dn_lower = display_name.lower().strip()
        # Exact
        if dn_lower in self.player_name_to_id:
            return self.player_name_to_id[dn_lower]

        dn_tokens = _name_tokens(display_name)
        if not dn_tokens:
            return None

        best_id: Optional[str] = None
        best_score = 0

        for stored_name, pid in self.player_name_to_id.items():
            stored_tokens = _name_tokens(stored_name)
            if not stored_tokens:
                continue
            overlap = sum(1 for t in dn_tokens if t in stored_tokens)
            surname_match = dn_tokens[-1] == stored_tokens[-1]
            suffix_match = len(dn_tokens) >= 2 and dn_tokens[-2:] == stored_tokens[-2:]
            subset_match = overlap == len(dn_tokens)
            score = overlap + (2 if surname_match else 0) + (2 if suffix_match else 0)
            if score > best_score and (subset_match or suffix_match):
                best_score = score
                best_id = pid

        return best_id


# ─────────────────────────────────────── Kalshi fetcher

def fetch_atp_markets(verbose: bool = False) -> list[dict]:
    """Return all open KXATPMATCH markets with price fields."""
    url = (f"{KALSHI_BASE}/markets"
           f"?series_ticker={SERIES_TICKER}&status=open&limit=200")
    try:
        d = _get(url)
    except Exception as e:
        print(f"[kalshi] fetch failed: {e}")
        return []

    out = []
    for m in d.get("markets", []):
        try:
            ask = float(m.get("yes_ask_dollars") or m.get("yes_ask") or 1)
            bid = float(m.get("yes_bid_dollars") or m.get("yes_bid") or 0)
        except (TypeError, ValueError):
            continue
        out.append({
            "ticker":           m.get("ticker", ""),
            "event_ticker":     m.get("event_ticker", ""),
            "yes_sub_title":    m.get("yes_sub_title", ""),
            "title":            m.get("title", ""),
            "yes_ask":          ask,
            "yes_bid":          bid,
            "yes_mid":          round((ask + bid) / 2.0, 6),
            "occurrence_time":  m.get("occurrence_datetime", ""),
            "rules_primary":    m.get("rules_primary", ""),
        })

    if verbose:
        print(f"[kalshi] {len(out)} open KXATPMATCH markets")
    return out


def group_by_match(markets: list[dict]) -> list[tuple[dict, dict]]:
    """Pair YES-side markets by event_ticker into (p1_mkt, p2_mkt) tuples.

    Each Kalshi match event has exactly two markets (one per player).
    """
    by_event: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        by_event[m["event_ticker"]].append(m)

    pairs = []
    for et, mkts in by_event.items():
        if len(mkts) == 2:
            pairs.append((mkts[0], mkts[1]))
        else:
            print(f"  [warn] event {et} has {len(mkts)} markets (expected 2), skipping")
    return pairs


# ─────────────────────────────────────── edge model

def compute_edge(model_prob: float, kalshi_mid: float) -> dict:
    raw_edge  = model_prob - kalshi_mid
    entry_fee = FEE_COEFF * model_prob * (1.0 - model_prob)
    net_edge  = raw_edge - entry_fee
    return {
        "raw_edge":  round(raw_edge, 6),
        "entry_fee": round(entry_fee, 6),
        "net_edge":  round(net_edge, 6),
    }


def infer_match_context(mkt: dict) -> tuple[str, int, str]:
    """Extract surface, best_of, tourney_level from Kalshi market rules text.

    Returns (surface, best_of, tourney_level).
    Defaults: 'Unknown', 3, 'A' (ATP regular tournament)
    """
    rules = (mkt.get("rules_primary") or mkt.get("title") or "").lower()

    # Surface detection
    surface = "Unknown"
    if any(kw in rules for kw in ["grass", "wimbledon", "halle", "queens", "s-hertogenbosch", "rosmalen", "stuttgart", "eastbourne", "nottingham", "newport", "mallorca", "'s-hertogenbosch"]):
        surface = "Grass"
    elif "clay" in rules or "roland" in rules or "monte" in rules:
        surface = "Clay"
    elif "hard" in rules or "australian" in rules or "us open" in rules:
        surface = "Hard"

    # Also check event ticker for tournament code hints
    event = mkt.get("event_ticker", "").upper()
    if any(t in event for t in ["WIMB", "HALLE", "QUEEN", "SHERT", "MALLE", "STUT", "EAST", "NEWP", "NOTT"]):
        surface = "Grass"
    elif any(t in event for t in ["RG", "CLAY", "BARC", "MONTE", "ROME", "HAMB"]):
        surface = "Clay"
    elif any(t in event for t in ["AO", "USO", "HARD"]):
        surface = "Hard"

    # best_of: Grand Slam = 5, otherwise 3
    best_of = 3
    is_gs = any(gs in rules for gs in ["grand slam", "wimbledon", "australian open", "us open", "roland garros"])
    if is_gs:
        best_of = 5

    tourney_level = "G" if is_gs else "A"

    return surface, best_of, tourney_level


# ─────────────────────────────────────── main scorer

def score_matches(
    state: TennisState,
    booster: "xgb.Booster",
    markets: list[dict],
    today: date,
    verbose: bool = False,
) -> list[dict]:
    """Score all market pairs and return signal records."""
    pairs = group_by_match(markets)
    if not pairs:
        print("[scorer] No market pairs found")
        return []

    signals = []

    for p1_mkt, p2_mkt in pairs:
        p1_name = p1_mkt["yes_sub_title"]
        p2_name = p2_mkt["yes_sub_title"]
        event_ticker = p1_mkt["event_ticker"]

        # Resolve player IDs from historical data
        p1_id = state.resolve_player(p1_name)
        p2_id = state.resolve_player(p2_name)

        if verbose:
            print(f"\n  [{event_ticker}] {p1_name} vs {p2_name}")
            print(f"    resolved: {p1_name} → {p1_id}  |  {p2_name} → {p2_id}")

        if p1_id is None or p2_id is None:
            missing = [name for name, pid in ((p1_name, p1_id), (p2_name, p2_id)) if pid is None]
            print(f"  [skip] Cannot resolve player ID for {', '.join(missing)} — observe only")
            rec = {
                "schema_version":   SCHEMA_VERSION,
                "date":             today.isoformat(),
                "captured_at":      datetime.now(timezone.utc).isoformat(),
                "event_ticker":     event_ticker,
                "match":            f"{p1_name} vs {p2_name}",
                "p1_name":          p1_name,
                "p2_name":          p2_name,
                "p1_id":            p1_id,
                "p2_id":            p2_id,
                "p1_resolved":      p1_id is not None,
                "p2_resolved":      p2_id is not None,
                "occurrence_time":  p1_mkt["occurrence_time"],
                "model_version":    "tennis_xgboost_noodds_v1",
                "model_p1_win":     None,
                "kalshi_p1_ask":    p1_mkt["yes_ask"],
                "kalshi_p1_bid":    p1_mkt["yes_bid"],
                "kalshi_p1_mid":    p1_mkt["yes_mid"],
                "kalshi_p2_mid":    p2_mkt["yes_mid"],
                "raw_edge":         None,
                "entry_fee":        None,
                "net_edge":         None,
                "edge_threshold":   EDGE_THRESHOLD,
                "action":           "OBSERVE",
                "signal_side":      None,
                "signal_ticker":    None,
                "skip_reason":      "unresolved_player_id",
                "p1_ticker":        p1_mkt["ticker"],
                "p2_ticker":        p2_mkt["ticker"],
                "features":         {},
            }
            signals.append(rec)
            print(f"  {p1_name:<22} vs {p2_name:<22}  unresolved → OBSERVE")
            continue

        # Infer match context
        surface, best_of, tourney_level = infer_match_context(p1_mkt)

        # Compute features for p1
        feat = state.get_features(
            p1_id or "__unknown_p1__",
            p2_id or "__unknown_p2__",
            surface,
            best_of,
            tourney_level,
            today,
        )

        if verbose:
            feat_dict = dict(zip(STRUCTURAL_FEATURES, feat))
            elo_diff = feat_dict["elo_diff"]
            rla = feat_dict["rank_log_advantage"]
            surf_elo = feat_dict["surface_elo_diff"]
            print(f"    surface={surface}  best_of={best_of}  tourney_level={tourney_level}")
            print(f"    elo_diff={elo_diff:.1f}  surface_elo_diff={surf_elo:.1f}  rank_log_adv={rla:.3f}")
            print(f"    p1_elo={state.overall_elo.get(p1_id or '', ELO_INIT):.1f}  "
                  f"p2_elo={state.overall_elo.get(p2_id or '', ELO_INIT):.1f}")

        # XGBoost prediction — p1 win probability
        X = np.array([feat], dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=STRUCTURAL_FEATURES)
        p1_win_prob = float(booster.predict(dm)[0])

        # Kalshi prices
        p1_ask = p1_mkt["yes_ask"]
        p1_bid = p1_mkt["yes_bid"]
        p1_mid = p1_mkt["yes_mid"]

        edge_data_p1 = compute_edge(p1_win_prob, p1_mid)
        net_edge_p1  = edge_data_p1["net_edge"]

        # Signal decision
        if abs(net_edge_p1) >= EDGE_THRESHOLD:
            if net_edge_p1 > 0:
                action       = "BUY_YES"
                signal_side  = p1_name
                signal_ticker = p1_mkt["ticker"]
            else:
                action       = "BUY_YES"
                signal_side  = p2_name
                signal_ticker = p2_mkt["ticker"]
        else:
            action       = "OBSERVE"
            signal_side  = None
            signal_ticker = None

        # Build record
        rec = {
            "schema_version":   SCHEMA_VERSION,
            "date":             today.isoformat(),
            "captured_at":      datetime.now(timezone.utc).isoformat(),
            "event_ticker":     event_ticker,
            "match":            f"{p1_name} vs {p2_name}",
            "p1_name":          p1_name,
            "p2_name":          p2_name,
            "p1_id":            p1_id,
            "p2_id":            p2_id,
            "p1_resolved":      p1_id is not None,
            "p2_resolved":      p2_id is not None,
            "surface":          surface,
            "best_of":          best_of,
            "tourney_level":    tourney_level,
            "occurrence_time":  p1_mkt["occurrence_time"],
            "model_version":    "tennis_xgboost_noodds_v1",
            "model_p1_win":     round(p1_win_prob, 4),
            "kalshi_p1_ask":    p1_ask,
            "kalshi_p1_bid":    p1_bid,
            "kalshi_p1_mid":    p1_mid,
            "kalshi_p2_mid":    p2_mkt["yes_mid"],
            "raw_edge":         edge_data_p1["raw_edge"],
            "entry_fee":        edge_data_p1["entry_fee"],
            "net_edge":         net_edge_p1,
            "edge_threshold":   EDGE_THRESHOLD,
            "action":           action,
            "signal_side":      signal_side,
            "signal_ticker":    signal_ticker,
            "p1_elo":           round(state.overall_elo.get(p1_id or "", ELO_INIT), 1),
            "p2_elo":           round(state.overall_elo.get(p2_id or "", ELO_INIT), 1),
            "p1_ticker":        p1_mkt["ticker"],
            "p2_ticker":        p2_mkt["ticker"],
            "features": {k: round(v, 4) for k, v in zip(STRUCTURAL_FEATURES, feat)},
        }

        signals.append(rec)

        # Console summary
        edge_pct   = net_edge_p1 * 100
        flag       = " *** SIGNAL ***" if action == "BUY_YES" else ""
        unresolved = "" if (p1_id and p2_id) else " [!unresolved]"
        print(f"  {p1_name:<22} vs {p2_name:<22}  "
              f"model={p1_win_prob:.4f}  mid={p1_mid:.3f}  "
              f"net_edge={edge_pct:+.1f}pp  {action}{flag}{unresolved}")

    return signals


# ─────────────────────────────────────── write helpers

def write_signals(signals: list[dict], today_str: str) -> None:
    """Append today's signals to the JSONL ledger, replacing any existing today rows."""
    SIGNALS_OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if SIGNALS_OUT.exists():
        for line in SIGNALS_OUT.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("date") != today_str:
                    existing.append(r)
            except Exception:
                pass

    with open(SIGNALS_OUT, "w") as f:
        for r in existing:
            f.write(json.dumps(r) + "\n")
        for s in signals:
            f.write(json.dumps(s) + "\n")

    print(f"[write] {len(signals)} records → {SIGNALS_OUT}")


def write_daily_json(signals: list[dict], today_str: str) -> None:
    """Write structured daily JSON for review."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORES_DIR / f"{today_str}.json"

    payload = {
        "date":           today_str,
        "model_version":  "tennis_xgboost_noodds_v1",
        "edge_threshold": EDGE_THRESHOLD,
        "n_matches":      len(signals),
        "n_signals":      sum(1 for s in signals if s["action"] == "BUY_YES"),
        "matches":        signals,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[write] daily JSON → {path}")


# ─────────────────────────────────────── entry point

def main():
    ap = argparse.ArgumentParser(
        description="Tennis daily scorer: ATP XGBoost model vs Kalshi prices"
    )
    ap.add_argument("--dry-run",  action="store_true", help="Print signals, do not write files")
    ap.add_argument("--verbose",  action="store_true", help="Extra diagnostic output")
    ap.add_argument("--threshold", type=float, default=EDGE_THRESHOLD,
                    help=f"Edge threshold for signal (default {EDGE_THRESHOLD})")
    args = ap.parse_args()

    threshold = args.threshold
    today     = date.today()
    today_str = today.isoformat()

    print(f"\n=== Tennis Daily Scorer — {today_str}  dry_run={args.dry_run} ===\n")

    # ── 1. Load model
    if not BOOSTER_PATH.exists():
        sys.exit(f"ERROR: Booster not found at {BOOSTER_PATH}")

    print(f"[model] Loading {BOOSTER_PATH.name} …")
    booster = xgb.Booster()
    booster.load_model(str(BOOSTER_PATH))
    print(f"  OK — 17 structural features")

    # ── 2. Build Elo state from historical CSVs
    csv_files = sorted(DATA_DIR.glob("atp_matches_*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: No ATP CSV files found in {DATA_DIR}")

    print(f"[elo] Replaying {len(csv_files)} CSV files …")
    state = TennisState()
    for path in csv_files:
        state.process_csv(path, verbose=args.verbose)

    print(f"  State built: {len(state.overall_elo)} players, "
          f"{state.rows_processed} matches processed")
    print(f"  Name index: {len(state.player_name_to_id)} entries")

    # ── 3. Fetch Kalshi markets
    print(f"\n[kalshi] Fetching open {SERIES_TICKER} markets …")
    markets = fetch_atp_markets(verbose=args.verbose)
    if not markets:
        print("[kalshi] No open ATP markets found. Nothing to score.")
        return

    print(f"  {len(markets)} markets → {len(markets)//2} match pairs")

    # ── 4. Score matches
    print("\n[score] Scoring matches:\n")
    signals = score_matches(state, booster, markets, today, verbose=args.verbose)

    # ── 5. Summary
    n_signal = sum(1 for s in signals if s["action"] == "BUY_YES")
    n_unresolved = sum(1 for s in signals if not (s["p1_resolved"] and s["p2_resolved"]))
    print(f"\n[summary] {len(signals)} matches scored, "
          f"{n_signal} signals (|net_edge| >= {threshold:.0%}), "
          f"{n_unresolved} with unresolved player IDs")

    if signals and n_signal > 0:
        print("\n  SIGNALS:")
        for s in signals:
            if s["action"] == "BUY_YES":
                print(f"    BUY YES  {s['signal_side']:<24}  ticker={s['signal_ticker']}"
                      f"  net_edge={s['net_edge']*100:+.1f}pp  "
                      f"model={s['model_p1_win']:.4f}  mid={s['kalshi_p1_mid']:.3f}")

    # ── 6. Write outputs
    if args.dry_run:
        print("\n[dry-run] Files NOT written.")
        return

    write_signals(signals, today_str)
    write_daily_json(signals, today_str)


if __name__ == "__main__":
    main()
