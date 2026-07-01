#!/usr/bin/env python3
"""wc_scorer.py — score open/upcoming KXWCGAME matches with the t168 soccer model.

PAPER-ONLY pipeline, task t196 (forward CLV gate for the WC2026 soccer sleeve).
This is the PROBABILITY SOURCE consumed by harness/wc_paper.py.

For every open KXWCGAME event (a 3-way win/draw/win market on one WC2026 group match)
it identifies the two national teams, derives per-outcome model probabilities
(p_team_a / p_draw / p_team_b), and writes them to data/wc_scores/<date>.json.

----------------------------------------------------------------------------
MODEL — and the honest data gap (TRUTH IN REPORTING)
----------------------------------------------------------------------------
The validated t168 model (harness/soccer_e07_e10.py) is a POSITIONAL UNIT-rating
model: it needs the announced 23-man squad per team, split into GK/DEF/MID/ATT
lines, each player mapped to a ClubElo club rating. That squad data exists on disk
ONLY for the 5 backtested tournaments (WC2018/2022, Euro2016/2020/2024) via
research/soccer_squad_model.load_all_squads(). There is **NO wc2026_squads.csv**
on disk and no free/keyless/ToS-clean announced-WC2026-squad source wired in.

  => The full t168 UNIT model CANNOT be run for WC2026 today (no per-player squads).

Per the t196 brief's explicit fallback ("if no clean WC2026 squad source exists,
document the gap and fall back to team-level ClubElo so we can still trade"), this
scorer uses the **national-team Elo** path that the SAME base module already
implements and that t168 itself reports as a baseline:

    research/soccer_squad_model.run_elo()  -> walk-forward national-team Elo
    research/soccer_squad_model.elo_predict_3way() -> calibrated 1X2 from Elo diff

Elo is built leak-free per match: only international matches dated STRICTLY BEFORE
that match's kickoff date are used (so an already-played WC2026 group result never
informs its own prediction, and later group games benefit from earlier ones). This
is the honest, tradeable team-level model until WC2026 squads land. When squad data
arrives, swap `model_probs_for_match` to call the unit model — the JSON schema and
wc_paper.py contract do not change.

A clean post-processing hook is left for calibration (`apply_calibration`): if a
fitted calibrator lands later it can transform these raw probs in place.

----------------------------------------------------------------------------
Usage:
  python3 harness/wc_scorer.py score                 # score all open KXWCGAME today
  python3 harness/wc_scorer.py score --date 2026-06-16
  python3 harness/wc_scorer.py score --horizon-days 4 # only matches kicking off <=4d out
  python3 harness/wc_scorer.py show                   # print today's scores table
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "harness"))

import soccer_squad_model as base  # noqa: E402  (run_elo, elo_predict_3way, load_intl)
import wc_dc_model  # noqa: E402  (predict_3way — score-based DC 1X2 head, t-wc-model-improve 2026-06-16)

# Advisory edit-lock (fail-open) for shared writes.
try:
    import editlock  # noqa: E402
except Exception:  # noqa: BLE001
    editlock = None

SCORES_DIR = ROOT / "data" / "wc_scores"
TICKS_DIR = ROOT / "data" / "ticks" / "KXWCGAME"
SERIES = "KXWCGAME"

# Model identity stamped into the scores so wc_paper / CLV reports know the source.
# t152: wc2026_squads.csv now on disk (29 teams; WC2022 proxy updated to 2026 ClubElo).
# Squad-strength model beats base rates +4.12% OOS (WC2018+2022 pooled, n=128).
# Scorer uses squad model when squad data is available, DC-Elo fallback otherwise.
MODEL_TAG = "t152-squadDC"  # squad-strength Poisson + DC 1X2 head (t152 revive)


# --------------------------------------------------------------------------- #
# Market title -> intl_results canonical team name
# --------------------------------------------------------------------------- #
# The KXWCGAME title is "<TeamA> vs <TeamB> Winner?". Most names match the
# martj42 intl_results.csv names used to build Elo; a handful differ. This map
# normalizes the exceptions. Anything not in the map is passed through unchanged
# (and if it then misses Elo, the match is skipped with a logged gap).
TITLE_TO_INTL = {
    "IR Iran": "Iran",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "Turkiye": "Turkey",
    "Turkiye ": "Turkey",
    "USA": "United States",
    "Curacao": "Curaçao",
}


def title_to_intl(name: str) -> str:
    name = name.strip()
    return TITLE_TO_INTL.get(name, name)


# --------------------------------------------------------------------------- #
# Read recorded ticks: latest two-sided quote per outcome ticker
# --------------------------------------------------------------------------- #
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def latest_quotes(tick_dates: list[str]) -> dict:
    """
    Return {outcome_ticker: {ts, yes_bid, yes_ask, no_bid, no_ask, last,
                             title, close_time, event}} — the MOST RECENT tick
    seen per outcome ticker across the given dated tick files.
    """
    out: dict[str, dict] = {}
    for d in tick_dates:
        p = TICKS_DIR / f"{d}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                t = rec.get("ticker")
                if not t or not t.startswith(SERIES):
                    continue
                prev = out.get(t)
                if prev is None or rec.get("ts", "") >= prev["ts"]:
                    ev = t.rsplit("-", 1)[0]
                    out[t] = {
                        "ts": rec.get("ts", ""),
                        "yes_bid": _f(rec.get("yes_bid_dollars")),
                        "yes_ask": _f(rec.get("yes_ask_dollars")),
                        "no_bid": _f(rec.get("no_bid_dollars")),
                        "no_ask": _f(rec.get("no_ask_dollars")),
                        "last": _f(rec.get("last_price_dollars")),
                        "title": rec.get("title", ""),
                        "close_time": rec.get("close_time", ""),
                        "event": ev,
                    }
    return out


def group_events(quotes: dict) -> dict:
    """
    Group the per-outcome quotes into events.
    Returns {event_ticker: {title, close_time, outcomes: {code: quote}, kickoff_date}}.
    The two team codes + TIE are the outcome suffixes. Team A/B order = title order
    = the two codes' order in the event ticker after the date prefix.
    """
    events: dict[str, dict] = {}
    for tkr, q in quotes.items():
        ev = q["event"]
        code = tkr.rsplit("-", 1)[1]
        e = events.setdefault(ev, {
            "title": q["title"], "close_time": q["close_time"],
            "outcomes": {}, "ts": q["ts"],
        })
        e["outcomes"][code] = q
        if q["ts"] > e["ts"]:
            e["ts"] = q["ts"]
    return events


# Event ticker:  KXWCGAME-26JUN15KSAURU  -> date code 26JUN15, codes KSA+URU.
_EVT_RE = re.compile(r"^KXWCGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3,4})([A-Z]{3,4})$")
_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def parse_event(ev: str, title: str) -> dict | None:
    """
    Extract kickoff date and the (code_a, code_b, team_a, team_b) for an event.
    Team names come from the title (authoritative); codes come from the ticker
    and are matched to title order. Returns None if unparseable.
    """
    m = _EVT_RE.match(ev)
    if not m:
        return None
    yy, mon, dd, code_a, code_b = m.groups()
    mon_n = _MONTHS.get(mon)
    if not mon_n:
        return None
    kickoff = f"20{yy}-{mon_n:02d}-{int(dd):02d}"
    tm = re.match(r"(.+?)\s+vs\s+(.+?)\s+Winner", title)
    if not tm:
        return None
    team_a = title_to_intl(tm.group(1))
    team_b = title_to_intl(tm.group(2))
    return {"kickoff": kickoff, "code_a": code_a, "code_b": code_b,
            "team_a": team_a, "team_b": team_b}


# --------------------------------------------------------------------------- #
# Model: leak-free national-team Elo -> 1X2
# --------------------------------------------------------------------------- #
def _is_unplayed_wc26(m: dict) -> bool:
    """Skip the future WC2026 fixture placeholder rows (loaded as real if scored).

    load_intl() already drops NA-NA rows, so unplayed WC2026 fixtures are absent.
    Played ones (with a real score and a date < the match we are predicting) are
    legitimately included by the date filter below — no extra handling needed.
    """
    return False


def build_elo(as_of: str) -> dict:
    """
    Walk-forward national-team Elo using only competitive (non-friendly) intl
    matches dated STRICTLY BEFORE `as_of` (YYYY-MM-DD). Mirrors the t168 baseline.
    """
    intl = base.load_intl()
    train = [m for m in intl
             if "friendly" not in m["tournament"].lower() and m["date"] < as_of]
    return base.run_elo(train)


# t152: squad-strength model state (lazy-loaded once per process)
_SQUAD_STATE: dict | None = None


def _get_squad_state() -> dict:
    """Lazy-load WC2026 squad data + 2026 ClubElo snapshot for squad-model scoring."""
    global _SQUAD_STATE
    if _SQUAD_STATE is not None:
        return _SQUAD_STATE
    # ClubElo snapshot: use the most recent snapshot available before WC2026 start
    elo_date = "2026-06-01"
    elo_map = base.load_clubelo(elo_date)
    lookup = base.build_lookup(elo_map) if elo_map else {}
    all_squads = base.load_all_squads()  # now includes WC2026 (t152)
    _SQUAD_STATE = {
        "elo_map": elo_map,
        "lookup": lookup,
        "all_squads": all_squads,
        "has_data": bool(elo_map),
    }
    return _SQUAD_STATE


def squad_strength_for(team: str, state: dict) -> float | None:
    """Return squad-strength score for a WC2026 team, or None if no squad data."""
    players = state["all_squads"].get(("WC2026", team), [])
    if not players:
        return None
    return base.squad_strength(players, state["elo_map"], state["lookup"])


HOSTS = {"United States", "Canada", "Mexico"}


def host_side(team_a: str, team_b: str) -> str | None:
    """Return 'A' / 'B' if a host nation is playing (and thus at home), else None.

    WC2026 group matches are at neutral US/CA/MX venues for almost every team; the
    only non-neutral cases are the three hosts playing at home. A host can be
    listed EITHER first or second in the market title (e.g. 'Switzerland vs
    Canada'), so we must detect the host on either side, not just team_a.
    """
    if team_a in HOSTS:
        return "A"
    if team_b in HOSTS:
        return "B"
    return None


def model_probs_for_match(team_a: str, team_b: str, kickoff: str,
                          elo: dict) -> dict | None:
    """
    Per-outcome model probabilities for one match, leak-free as of `kickoff`.

    Returns {p_a, p_draw, p_b, elo_a, elo_b, neutral, squad_a, squad_b,
             model_path} or None if either team has no Elo.

    t152 (squad-strength revive): when WC2026 squad data is available for BOTH
    teams, blend the squad-strength Poisson model with the DC-Elo baseline using
    a 50/50 ensemble (equal weight; no meta-learner to avoid overfit on thin WC sample).
    Falls back to pure DC-Elo when squad data is missing for either team.
    Backtest: squad model +4.12% OOS WC-pooled vs base rates (n=128), climate layer
    negligible (+0.03% delta) — not included.
    """
    ra = elo.get(team_a)
    rb = elo.get(team_b)
    if ra is None or rb is None:
        return None
    hs = host_side(team_a, team_b)         # 'A','B' (host plays) or None (neutral)
    neutral = hs is None

    # --- DC-Elo 1X2 (base) ---
    if hs == "B":
        p_b_dc, p_draw_dc, p_a_dc = wc_dc_model.predict_3way(rb, ra, neutral=False)
    else:
        p_a_dc, p_draw_dc, p_b_dc = wc_dc_model.predict_3way(ra, rb, neutral=neutral)

    # --- Squad-strength Poisson blend (t152) ---
    squad_a = squad_b = None
    model_path = "dc-elo"
    try:
        state = _get_squad_state()
        if state["has_data"]:
            sa = squad_strength_for(team_a, state)
            sb = squad_strength_for(team_b, state)
            if sa is not None and sb is not None:
                squad_a, squad_b = sa, sb
                # Poisson params from WC-trained Poisson head (alpha from soccer_squad_model)
                # Use params fit on all competitive pre-tournament matches (WC2022 train).
                # alpha=0.0012, base_h=1.45, base_a=1.20 (from soccer_squad_report.txt WC2022 fit)
                alpha, base_h, base_a = 0.0012, 1.45, 1.20
                diff = sa - sb
                ha_mult = 0.0 if neutral else 0.08   # small home bump for host nations
                lh = max(base_h * math.exp(alpha * diff + ha_mult), 0.01)
                la = max(base_a * math.exp(-alpha * diff), 0.01)
                p_a_sq, p_draw_sq, p_b_sq = base.predict_3way(lh, la)
                # 50/50 ensemble: average probabilities, renormalize
                p_a = (p_a_dc + p_a_sq) / 2
                p_draw = (p_draw_dc + p_draw_sq) / 2
                p_b = (p_b_dc + p_b_sq) / 2
                s = p_a + p_draw + p_b
                p_a, p_draw, p_b = p_a / s, p_draw / s, p_b / s
                model_path = "squad+dc-elo"
            else:
                p_a, p_draw, p_b = p_a_dc, p_draw_dc, p_b_dc
        else:
            p_a, p_draw, p_b = p_a_dc, p_draw_dc, p_b_dc
    except Exception:  # noqa: BLE001 — fail-open to DC-Elo
        p_a, p_draw, p_b = p_a_dc, p_draw_dc, p_b_dc

    return {"p_a": p_a, "p_draw": p_draw, "p_b": p_b,
            "elo_a": ra, "elo_b": rb, "neutral": neutral,
            "squad_a": squad_a, "squad_b": squad_b,
            "model_path": model_path}


# --------------------------------------------------------------------------- #
# Calibration hook (no-op until a fitted calibrator lands)
# --------------------------------------------------------------------------- #
def apply_calibration(probs: dict) -> dict:
    """
    Post-process raw model probs. NO-OP today. When soccer_calib_oos lands a
    fitted 1X2 calibrator, transform (p_a,p_draw,p_b) here and re-normalize. The
    output schema is unchanged so wc_paper.py never needs to know.
    """
    return probs


# --------------------------------------------------------------------------- #
# Score command
# --------------------------------------------------------------------------- #
def _tick_dates_around(d: str) -> list[str]:
    """Tick files to read: the score date and the day before (quotes may span days)."""
    try:
        dt = datetime.fromisoformat(d)
    except ValueError:
        dt = datetime.now(timezone.utc)
    from datetime import timedelta
    return [(dt - timedelta(days=1)).strftime("%Y-%m-%d"),
            dt.strftime("%Y-%m-%d")]


def cmd_score(args):
    score_date = args.date or date.today().isoformat()
    horizon = args.horizon_days

    # Read the most recent recorded quotes (today + yesterday tick files).
    quotes = latest_quotes(_tick_dates_around(score_date))
    if not quotes:
        print(f"No KXWCGAME ticks found around {score_date} in {TICKS_DIR}/")
        return
    events = group_events(quotes)

    elo = build_elo(score_date)  # leak-free as of the score date

    games = []
    gaps = []
    for ev, e in sorted(events.items()):
        parsed = parse_event(ev, e["title"])
        if not parsed:
            gaps.append(f"{ev}: unparseable ({e['title']!r})")
            continue
        kickoff = parsed["kickoff"]
        # Only score upcoming/open matches (kickoff on/after score_date) within horizon.
        if kickoff < score_date:
            continue
        if horizon is not None:
            from datetime import timedelta
            limit = (datetime.fromisoformat(score_date) +
                     timedelta(days=horizon)).strftime("%Y-%m-%d")
            if kickoff > limit:
                continue

        mp = model_probs_for_match(parsed["team_a"], parsed["team_b"], kickoff, elo)
        if mp is None:
            gaps.append(f"{ev}: no Elo for "
                        f"{parsed['team_a']!r} or {parsed['team_b']!r}")
            continue
        mp = apply_calibration(mp)

        # Map per-outcome probs to the actual outcome tickers.
        code_a, code_b = parsed["code_a"], parsed["code_b"]
        outcomes = e["outcomes"]
        # Defensive: confirm the codes exist as outcomes.
        if code_a not in outcomes or code_b not in outcomes or "TIE" not in outcomes:
            gaps.append(f"{ev}: outcome tickers {sorted(outcomes)} "
                        f"!= expected [{code_a},{code_b},TIE]")
            continue

        def mkt(code):
            q = outcomes[code]
            return {"ticker": f"{ev}-{code}", "yes_bid": q["yes_bid"],
                    "yes_ask": q["yes_ask"], "no_bid": q["no_bid"],
                    "no_ask": q["no_ask"], "last": q["last"], "ts": q["ts"]}

        sq_a = mp.get("squad_a")
        sq_b = mp.get("squad_b")
        games.append({
            "event": ev,
            "title": e["title"],
            "kickoff": kickoff,
            "close_time": e["close_time"],
            "neutral": mp["neutral"],
            "team_a": parsed["team_a"], "team_b": parsed["team_b"],
            "code_a": code_a, "code_b": code_b,
            "elo_a": round(mp["elo_a"], 1), "elo_b": round(mp["elo_b"], 1),
            "squad_a": round(sq_a, 1) if sq_a else None,
            "squad_b": round(sq_b, 1) if sq_b else None,
            "model_path": mp.get("model_path", "dc-elo"),
            "model": {
                "p_a": mp["p_a"], "p_draw": mp["p_draw"], "p_b": mp["p_b"],
            },
            # Three tradeable legs, model_prob aligned to each YES contract.
            "legs": [
                {"outcome": "A", "code": code_a, "model_prob": mp["p_a"],
                 "market": mkt(code_a)},
                {"outcome": "TIE", "code": "TIE", "model_prob": mp["p_draw"],
                 "market": mkt("TIE")},
                {"outcome": "B", "code": code_b, "model_prob": mp["p_b"],
                 "market": mkt(code_b)},
            ],
        })

    payload = {
        "schema": "wc-scores-v1",
        "model_tag": MODEL_TAG,
        "score_date": score_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_games": len(games),
        "gaps": gaps,
        "games": games,
    }

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORES_DIR / f"{score_date}.json"
    lock = (editlock.hold(out_path.name, owner="wc_scorer.py", ttl=120, wait=60)
            if editlock else _nullctx())
    with lock:
        tmp = f"{out_path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(out_path))

    print(f"Scored {len(games)} upcoming KXWCGAME match(es) -> {out_path}")
    if gaps:
        print(f"  {len(gaps)} gap(s):")
        for g in gaps:
            print(f"    - {g}")
    _print_table(games)


import contextlib  # noqa: E402


@contextlib.contextmanager
def _nullctx():
    yield


def _print_table(games: list):
    if not games:
        return
    print(f"\n{'match':<34}{'kick':<11}{'A/TIE/B model':<26}{'A/TIE/B mkt-mid'}")
    print("-" * 96)
    for g in games:
        m = g["model"]
        legs = {l["outcome"]: l for l in g["legs"]}

        def mid(code):
            q = legs[code]["market"]
            b, a = q["yes_bid"], q["yes_ask"]
            if b is None or a is None:
                return "  -  "
            return f"{(b + a) / 2:.2f}"
        title = (g["title"][:32]) if len(g["title"]) > 32 else g["title"]
        print(f"{title:<34}{g['kickoff']:<11}"
              f"{m['p_a']:.2f}/{m['p_draw']:.2f}/{m['p_b']:.2f}        "
              f"{mid('A')}/{mid('TIE')}/{mid('B')}")


def cmd_show(args):
    score_date = args.date or date.today().isoformat()
    p = SCORES_DIR / f"{score_date}.json"
    if not p.exists():
        print(f"No scores file for {score_date} ({p}). Run `score` first.")
        return
    payload = json.loads(p.read_text())
    print(f"{payload['n_games']} games | model={payload['model_tag']} | "
          f"generated {payload['generated_at']}")
    _print_table(payload["games"])


def main():
    ap = argparse.ArgumentParser(description="Score KXWCGAME matches with t168 (team-Elo fallback).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score", help="score open KXWCGAME matches -> data/wc_scores/<date>.json")
    p.add_argument("--date", default=None, help="score date YYYY-MM-DD (default today)")
    p.add_argument("--horizon-days", type=int, default=None,
                   help="only score matches kicking off within N days")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("show", help="print a scores file")
    p.add_argument("--date", default=None)
    p.set_defaults(func=cmd_show)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
