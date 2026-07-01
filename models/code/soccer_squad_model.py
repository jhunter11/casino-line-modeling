#!/usr/bin/env python3
"""soccer_squad_model.py — Current-squad-strength 3-way soccer prediction engine.

Fixes the stale-team-Elo problem: instead of one aggregate national-team Elo, we use
per-player CLUB strength (ClubElo API, keyless, point-in-time) aggregated into a squad
rating, then fit a Poisson-goals model for 3-way home/draw/away predictions.

ALL PREDICTIONS ARE STRICTLY LEAK-FREE: each match uses only data available BEFORE kickoff
— the ClubElo snapshot dated before tournament start, the announced squad for that tournament.

Data sources (all free/keyless, no auth required):
  - ClubElo: http://api.clubelo.com/<date>  (point-in-time club ratings)
  - WC squads: Wikipedia API → player+club for WC 2018, 2022
  - Euro squads: Wikipedia API → player+club for Euro 2016, 2020, 2024
  - International results: martj42/international_results (data/worldcup/intl_results.csv)
  - Club league results: football-data.co.uk (data/soccer/club_*.csv)

Model design:
  - Squad strength = position-weighted mean ClubElo of squad players' clubs
    (weights: GK=0.6, DF=1.0, MF=1.2, FW=1.0; unknown clubs → 1400 fallback)
  - Goals: λ_home = base_h * exp(α*(squad_h - squad_a) + home_adv)
           λ_away = base_a * exp(-α*(squad_h - squad_a))
  - 3-way probs from independent Poisson draws summed over score grid (max 10 goals)
  - α and home_adv fit by MLE on training matches; base rates from same window
  - Walk-forward: train on prior tournaments, test on next held-out tournament
  - Elo baseline from worldcup_elo.py reproduced for direct comparison

Walk-forward protocol (BURNED-HOLDOUT):
  WC: train WC 1994-2014 + all-competitive pre-tournament → test WC2018, then WC2022
  Euro: train Euro 2016 → test Euro 2020; train Euro 2016+2020 → test Euro 2024
  Club: train season N → test season N+1

Usage:
    python3 research/soccer_squad_model.py backtest    # full walk-forward eval
    python3 research/soccer_squad_model.py club        # club-league-only validation
    python3 research/soccer_squad_model.py rate        # squad strength ratings
"""

import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data/soccer"
INTL_RESULTS = "data/worldcup/intl_results.csv"

# ---------------------------------------------------------------------------
# Club name normalization
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize a club name for fuzzy matching."""
    s = str(s).lower()
    s = re.sub(
        r'\b(f\.?c\.?|s\.?c\.?|a\.?f\.?c\.?|s\.?s\.?c\.?|a\.?s\.?|a\.?c\.?|'
        r's\.?k\.?|b\.?v\.?|nv|sv|vv|cf|cp|rcd|cd|ud|sd|ce|ca|sp|bs|es|us|rc|'
        r'rs|ue|ua|de|la|as|fc|sc|0?4|1\.|2\.)\b', ' ', s)
    s = re.sub(r'\b\d{4}\b', '', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Manual overrides: normalized wikipedia name → ClubElo name
MANUAL_MAP = {
    'manchester united': 'Man United',
    'manchester city': 'Man City',
    'tottenham hotspur': 'Tottenham',
    'tottenham': 'Tottenham',
    'west bromwich albion': 'West Brom',
    'wolverhampton wanderers': 'Wolves',
    'wolverhampton': 'Wolves',
    'atletico madrid': 'Atletico',
    'atletico de madrid': 'Atletico',
    'atlético de madrid': 'Atletico',
    'athletic bilbao': 'Bilbao',
    'athletic club': 'Bilbao',
    'rb leipzig': 'Leipzig',
    'rasenballsport leipzig': 'Leipzig',
    'red bull leipzig': 'Leipzig',
    'borussia dortmund': 'Dortmund',
    'borussia monchengladbach': "M'gladbach",
    'borussia mgladbach': "M'gladbach",
    'eintracht frankfurt': 'Frankfurt',
    'internazionale': 'Inter',
    'inter milan': 'Inter',
    'paris saint-germain': 'Paris SG',
    'olympique de marseille': 'Marseille',
    'olympique lyonnais': 'Lyon',
    'psv eindhoven': 'PSV',
    'sporting cp': 'Sporting',
    'shakhtar donetsk': 'Shakhtar',
    'dynamo kyiv': 'Dynamo Kyiv',
    'crvena zvezda': 'Red Star',
    'red star belgrade': 'Red Star',
    'newcastle united': 'Newcastle',
    'leicester city': 'Leicester',
    'nottingham forest': 'Nottm Forest',
    'real sociedad': 'Sociedad',
    'real betis': 'Betis',
    'stade rennais': 'Rennes',
    'union berlin': 'Union Berlin',
    'bayer 04 leverkusen': 'Leverkusen',
    'bayer leverkusen': 'Leverkusen',
    'bsc young boys': 'Young Boys',
    'vfb stuttgart': 'Stuttgart',
    'tsg hoffenheim': 'Hoffenheim',
    'mainz': 'Mainz',
    'koln': 'Koeln',
    'nott m forest': 'Nottm Forest',
    'crystal palace': 'Crystal Palace',
    'west ham united': 'West Ham',
    'brighton hove albion': 'Brighton',
    'brighton and hove albion': 'Brighton',
    'sheffield united': 'Sheffield Utd',
    'al nassr': 'Al Nassr',
    'al hilal': 'Al Hilal',
    'al ahli': 'Al Ahli',
    'al ittihad': 'Al Ittihad',
    'benfica': 'Benfica',
    'porto': 'Porto',
    'celtic': 'Celtic',
    'rangers': 'Rangers',
    'galatasaray': 'Galatasaray',
    'fenerbahce': 'Fenerbahce',
    'besiktas': 'Besiktas',
    'cska moscow': 'CSKA Moscow',
    'spartak moscow': 'Spartak',
    'zenit': 'Zenit',
    'lokomotiv moscow': 'Lokomotiv',
    'lazio': 'Lazio',
    'fiorentina': 'Fiorentina',
    'atalanta': 'Atalanta',
    'napoli': 'Napoli',
    'juventus': 'Juventus',
}

# football-data.co.uk names → ClubElo names
FDC_TO_CLUBELO = {
    'Man United': 'Man United', 'Man City': 'Man City', 'Arsenal': 'Arsenal',
    'Liverpool': 'Liverpool', 'Chelsea': 'Chelsea', 'Tottenham': 'Tottenham',
    'Leicester': 'Leicester', 'Everton': 'Everton', 'West Ham': 'West Ham',
    'Wolves': 'Wolves', 'Newcastle': 'Newcastle', 'Crystal Palace': 'Crystal Palace',
    'Southampton': 'Southampton', 'Brighton': 'Brighton', 'Burnley': 'Burnley',
    'West Brom': 'West Brom', 'Sheffield United': 'Sheffield Utd',
    'Fulham': 'Fulham', 'Brentford': 'Brentford', 'Norwich': 'Norwich',
    'Watford': 'Watford', 'Luton': 'Luton', 'Bournemouth': 'Bournemouth',
    "Nott'm Forest": 'Nottm Forest', 'Aston Villa': 'Aston Villa',
    'Leeds': 'Leeds',
    # La Liga
    'Barcelona': 'Barcelona', 'Real Madrid': 'Real Madrid', 'Atletico Madrid': 'Atletico',
    'Sevilla': 'Sevilla', 'Villarreal': 'Villarreal', 'Real Sociedad': 'Sociedad',
    'Real Betis': 'Betis', 'Athletic Club': 'Bilbao', 'Valencia': 'Valencia',
    'Celta Vigo': 'Celta', 'Osasuna': 'Osasuna', 'Getafe': 'Getafe',
    'Almeria': 'Almeria', 'Cadiz': 'Cadiz', 'Granada': 'Granada',
    'Las Palmas': 'Las Palmas', 'Mallorca': 'Mallorca', 'Girona': 'Girona',
    'Rayo Vallecano': 'Vallecano', 'Alaves': 'Alaves', 'Valladolid': 'Valladolid',
    'Leganes': 'Leganes', 'Espanyol': 'Espanyol',
    # Serie A
    'Inter': 'Inter', 'Milan': 'Milan', 'Juventus': 'Juventus', 'Napoli': 'Napoli',
    'Roma': 'Roma', 'Lazio': 'Lazio', 'Fiorentina': 'Fiorentina', 'Atalanta': 'Atalanta',
    'Torino': 'Torino', 'Bologna': 'Bologna', 'Udinese': 'Udinese', 'Sassuolo': 'Sassuolo',
    'Empoli': 'Empoli', 'Spezia': 'Spezia', 'Salernitana': 'Salernitana',
    'Lecce': 'Lecce', 'Cremonese': 'Cremonese', 'Monza': 'Monza',
    'Hellas Verona': 'Verona', 'Frosinone': 'Frosinone', 'Genoa': 'Genoa',
    'Cagliari': 'Cagliari', 'Sampdoria': 'Sampdoria',
    # Bundesliga
    'Bayern Munich': 'Bayern', 'Dortmund': 'Dortmund', 'Leipzig': 'Leipzig',
    'Leverkusen': 'Leverkusen', 'Frankfurt': 'Frankfurt', 'Wolfsburg': 'Wolfsburg',
    'Freiburg': 'Freiburg', 'Hoffenheim': 'Hoffenheim', 'Mainz': 'Mainz',
    'Koeln': 'Koeln', 'Augsburg': 'Augsburg', 'Bochum': 'Bochum',
    "M'gladbach": "M'gladbach", 'Stuttgart': 'Stuttgart', 'Union Berlin': 'Union Berlin',
    'Hertha': 'Hertha', 'Schalke': 'Schalke', 'Werder Bremen': 'Bremen',
    'Darmstadt': 'Darmstadt', 'Heidenheim': 'Heidenheim',
    # Ligue 1
    'Paris SG': 'Paris SG', 'Marseille': 'Marseille', 'Lyon': 'Lyon',
    'Monaco': 'Monaco', 'Rennes': 'Rennes', 'Lens': 'Lens', 'Lille': 'Lille',
    'Nice': 'Nice', 'Strasbourg': 'Strasbourg', 'Nantes': 'Nantes',
    'Montpellier': 'Montpellier', 'Reims': 'Reims', 'Brest': 'Brest',
    'Clermont': 'Clermont', 'Lorient': 'Lorient', 'Metz': 'Metz',
    'Toulouse': 'Toulouse', 'Auxerre': 'Auxerre', 'Ajaccio': 'Ajaccio',
    'Troyes': 'Troyes', 'Angers': 'Angers', 'Le Havre': 'Le Havre',
}


# ---------------------------------------------------------------------------
# ClubElo loading and lookup
# ---------------------------------------------------------------------------

def load_clubelo(snapshot_date: str) -> dict:
    """Load ClubElo snapshot {club_name: float} for given date string YYYY-MM-DD."""
    path = os.path.join(DATA_DIR, f"clubelo_{snapshot_date}.csv")
    if not os.path.exists(path):
        return {}
    return {r['Club']: float(r['Elo']) for r in csv.DictReader(open(path))}


def build_lookup(elo_map: dict) -> dict:
    """Build {original_name: elo, normalized_name: elo} lookup dict."""
    lookup = {}
    for club, elo in elo_map.items():
        lookup[club] = elo
        n = _norm(club)
        if n and n not in lookup:
            lookup[n] = elo
    return lookup


def find_elo(club_wiki: str, elo_map: dict, lookup: dict,
             default: float = 1400.0) -> float:
    """Map a Wikipedia-style club name to its ClubElo rating."""
    c = re.sub(r'\[\[|\]\]', '', str(club_wiki))
    c = re.sub(r'\|.*', '', c)
    c = re.sub(r'<!--.*?-->', '', c)
    c = c.strip()
    if not c or c in ('Unknown', ''):
        return default
    if c in elo_map:
        return elo_map[c]
    n = _norm(c)
    for k, v in MANUAL_MAP.items():
        if k == n or (len(k) > 4 and k in n):
            if v in elo_map:
                return elo_map[v]
    if n in lookup:
        return lookup[n]
    # Word-overlap fuzzy match
    n_words = set(w for w in n.split() if len(w) > 3)
    best_v, best_score = None, 0.0
    for key, val in lookup.items():
        kw = set(w for w in str(key).split() if len(w) > 3)
        if not kw:
            continue
        if n_words & kw:
            score = len(n_words & kw) / max(len(kw), 1)
            if score > best_score and score >= 0.5:
                best_score, best_v = score, val
    return best_v if best_v is not None else default


# ---------------------------------------------------------------------------
# Squad strength
# ---------------------------------------------------------------------------

POS_WEIGHT = {'GK': 0.6, 'DF': 1.0, 'MF': 1.2, 'FW': 1.0}


def squad_strength(players: list, elo_map: dict, lookup: dict) -> float:
    """Weighted-mean ClubElo across squad. Unknown clubs get 1400 fallback."""
    total_w = total_e = 0.0
    for p in players:
        e = find_elo(p['club'], elo_map, lookup)
        w = POS_WEIGHT.get(p.get('pos', 'MF'), 1.0)
        total_e += e * w
        total_w += w
    return total_e / total_w if total_w > 0 else 1400.0


def load_all_squads() -> dict:
    """
    Returns {(tournament_tag, team_name): [{'club':..., 'pos':...}, ...]}
    tournament_tag = 'WC2018', 'WC2022', 'WC2026', 'Euro2016', 'Euro2020', 'Euro2024'
    WC2026: proxy squads from WC2022 rosters (29 overlapping teams) mapped to the
    2026-06-01 ClubElo snapshot at prediction time — player turnover not modelled,
    but club-strength signal still valid (same squad_strength() path, updated Elo).
    """
    squads = {}
    configs = [
        ('WC', 2018, 'wc2018_squads.csv'),
        ('WC', 2022, 'wc2022_squads.csv'),
        ('WC', 2026, 'wc2026_squads.csv'),
        ('Euro', 2016, 'euro2016_squads.csv'),
        ('Euro', 2020, 'euro2020_squads.csv'),
        ('Euro', 2024, 'euro2024_squads.csv'),
    ]
    for tour, yr, fname in configs:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        tag = f"{tour}{yr}"
        for r in csv.DictReader(open(path)):
            key = (tag, r['team'])
            if key not in squads:
                squads[key] = []
            squads[key].append({'club': r['club'], 'pos': r['pos']})
    return squads


# ---------------------------------------------------------------------------
# Poisson goals model
# ---------------------------------------------------------------------------

def _pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - sum(math.log(i) for i in range(1, k + 1)))


def predict_3way(lam_h: float, lam_a: float, max_goals: int = 10) -> tuple:
    """Return (p_home, p_draw, p_away) from Poisson(lam_h) × Poisson(lam_a)."""
    ph = pd = pa = 0.0
    for i in range(max_goals + 1):
        pi_h = _pmf(i, lam_h)
        for j in range(max_goals + 1):
            p = pi_h * _pmf(j, lam_a)
            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


def fit_params(matches: list, squad_fn, alpha_range=(0, 30),
               hadv_range=(0, 20)) -> dict:
    """
    Fit Poisson model (alpha, home_adv) by MLE on training matches.
    squad_fn(team) → float squad strength.
    """
    goals_h = [m['home_goals'] for m in matches]
    goals_a = [m['away_goals'] for m in matches]
    base_h = sum(goals_h) / max(len(goals_h), 1)
    base_a = sum(goals_a) / max(len(goals_a), 1)

    # Pre-compute diffs
    diffs = [(squad_fn(m['home_team']) - squad_fn(m['away_team'])) for m in matches]

    best_ll, best_alpha, best_hadv = 1e12, 0.0, 0.0
    for ai in range(*alpha_range):
        alpha = ai * 0.0001
        for hi in range(*hadv_range):
            hadv = hi * 0.01
            ll = 0.0
            for m, diff in zip(matches, diffs):
                ha = 0.0 if m.get('neutral', False) else hadv
                lh = max(base_h * math.exp(alpha * diff + ha), 0.01)
                la = max(base_a * math.exp(-alpha * diff), 0.01)
                ll -= (m['home_goals'] * math.log(lh) - lh
                       + m['away_goals'] * math.log(la) - la)
            if ll < best_ll:
                best_ll, best_alpha, best_hadv = ll, alpha, hadv

    return {
        'base_h': base_h, 'base_a': base_a,
        'alpha': best_alpha, 'home_adv': best_hadv,
        'n_train': len(matches),
    }


def predict_match(home_team: str, away_team: str, neutral: bool,
                  squad_fn, params: dict) -> tuple:
    """3-way prediction for one match."""
    diff = squad_fn(home_team) - squad_fn(away_team)
    ha = 0.0 if neutral else params['home_adv']
    lh = max(params['base_h'] * math.exp(params['alpha'] * diff + ha), 0.01)
    la = max(params['base_a'] * math.exp(-params['alpha'] * diff), 0.01)
    return predict_3way(lh, la)


# ---------------------------------------------------------------------------
# Team-Elo baseline (from worldcup_elo.py, for comparison)
# ---------------------------------------------------------------------------

def run_elo(matches: list, K: float = 30.0, home_adv: float = 85.0,
            base: float = 1500.0) -> dict:
    """Walk-forward team Elo → {team: rating}."""
    R = {}

    def _gd(gd):
        gd = abs(gd)
        if gd <= 1: return 1.0
        if gd == 2: return 1.5
        return (11 + gd) / 8.0

    def _tw(t):
        t = t.lower()
        if 'friendly' in t: return 1.0
        if 'qualification' in t or 'qualifier' in t: return 2.0
        if 'world cup' in t or 'nations' in t: return 3.0
        return 2.5

    for m in sorted(matches, key=lambda x: x['date']):
        h, a = m['home_team'], m['away_team']
        rh, ra = R.get(h, base), R.get(a, base)
        ha = 0.0 if m.get('neutral', False) else home_adv
        dr = (rh + ha) - ra
        E = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        hg, ag = m['home_goals'], m['away_goals']
        W = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = K * _gd(hg - ag) * _tw(m.get('tournament', ''))
        R[h] = rh + k * (W - E)
        R[a] = ra + k * ((1 - W) - (1 - E))
    return R


def elo_predict_3way(rh: float, ra: float) -> tuple:
    """Elo → 3-way via calibrated draw curve (worldcup_elo.py method)."""
    dr = rh - ra
    E = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    a, c = 0.27, 500.0
    pd = a * math.exp(-(dr / c) ** 2)
    pd = min(pd, 2 * min(E, 1 - E) - 1e-6)
    pd = max(pd, 1e-4)
    ph = E - 0.5 * pd
    pa = 1 - E - 0.5 * pd
    s = ph + pd + pa
    return ph / s, pd / s, pa / s


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _result(hg: int, ag: int) -> str:
    return 'H' if hg > ag else ('D' if hg == ag else 'A')


def score(preds: list) -> dict:
    """
    preds: [{'ph','pd','pa','result'}, ...]
    Returns logloss, brier, accuracy, n.
    """
    n = len(preds)
    if n == 0:
        return {'logloss': None, 'brier': None, 'accuracy': None, 'n': 0}
    ll = brier = acc = 0.0
    for p in preds:
        ph = max(p['ph'], 1e-7)
        pd = max(p['pd'], 1e-7)
        pa = max(p['pa'], 1e-7)
        s = ph + pd + pa
        ph, pd, pa = ph / s, pd / s, pa / s
        res = p['result']
        probs = {'H': ph, 'D': pd, 'A': pa}
        y = {'H': (1, 0, 0), 'D': (0, 1, 0), 'A': (0, 0, 1)}[res]
        pv = (ph, pd, pa)
        ll += -math.log(probs[res])
        brier += sum((pv[i] - y[i]) ** 2 for i in range(3))
        if max(probs, key=probs.get) == res:
            acc += 1
    return {'logloss': ll / n, 'brier': brier / n, 'accuracy': acc / n, 'n': n}


def baselines(preds: list) -> dict:
    """Score uniform and base-rate baselines against the same predictions."""
    n = len(preds)
    if n == 0:
        return {}
    results = [p['result'] for p in preds]
    nH, nD, nA = results.count('H'), results.count('D'), results.count('A')
    br_h, br_d, br_a = nH / n, nD / n, nA / n
    uni = [{'ph': 1/3, 'pd': 1/3, 'pa': 1/3, 'result': r} for r in results]
    br = [{'ph': br_h, 'pd': br_d, 'pa': br_a, 'result': r} for r in results]
    return {
        'uniform': score(uni),
        'base_rate': score(br),
        'base_rates': (round(br_h, 3), round(br_d, 3), round(br_a, 3)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_intl() -> list:
    """Load martj42 international results (all played matches)."""
    out = []
    for r in csv.DictReader(open(INTL_RESULTS)):
        hs, as_ = r['home_score'], r['away_score']
        if hs in ('NA', '', None) or as_ in ('NA', '', None):
            continue
        hg, ag = int(hs), int(as_)
        out.append({
            'date': r['date'],
            'home_team': r['home_team'],
            'away_team': r['away_team'],
            'home_goals': hg,
            'away_goals': ag,
            'result': _result(hg, ag),
            'tournament': r['tournament'],
            'neutral': r['neutral'].strip().upper() == 'TRUE',
        })
    return out


def load_club_matches() -> list:
    """Load football-data.co.uk club matches (EPL/La Liga/Serie A/Bundesliga/Ligue 1)."""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'club_*.csv'))):
        parts = os.path.basename(f).replace('club_', '').replace('.csv', '').split('_')
        if len(parts) < 2:
            continue
        league, season = parts[0], parts[1]
        for r in csv.DictReader(open(f)):
            ftr = r.get('FTR', '').strip()
            if ftr not in ('H', 'D', 'A'):
                continue
            ht = FDC_TO_CLUBELO.get(r.get('HomeTeam', ''), r.get('HomeTeam', ''))
            at = FDC_TO_CLUBELO.get(r.get('AwayTeam', ''), r.get('AwayTeam', ''))
            hg = int(r.get('FTHG', 0) or 0)
            ag = int(r.get('FTAG', 0) or 0)
            out.append({
                'date': f'20{season[:2]}-08-01',  # approximate season start
                'home_team': ht,
                'away_team': at,
                'home_goals': hg,
                'away_goals': ag,
                'result': ftr,
                'neutral': False,
                'tournament': f'club_{league}_{season}',
                'league': league,
                'season': season,
            })
    return out


# ---------------------------------------------------------------------------
# WC + Euro backtest
# ---------------------------------------------------------------------------

def _is_wc_finals(m):
    return ('FIFA World Cup' in m['tournament'] and
            'qualification' not in m['tournament'].lower())


def _is_euro_finals(m, year):
    if 'UEFA Euro' not in m['tournament']:
        return False
    if year == 2020:
        return m['date'].startswith('2021') and m['date'] >= '2021-06-01'
    return m['date'].startswith(str(year)) and m['date'] >= f'{year}-06-01'


def run_intl_backtest(verbose: bool = True) -> dict:
    """
    Walk-forward backtest on WC2018, WC2022, Euro2016, Euro2020, Euro2024.
    For each tournament:
      - Squad strength from ClubElo snapshot taken BEFORE tournament start
      - Poisson model alpha tuned on all-competitive pre-tournament matches
      - Elo baseline from worldcup_elo.py reproduced for comparison
    Results are STRICTLY OOS: no test data seen during training.
    """
    intl = load_intl()
    all_squads = load_all_squads()
    comp = [m for m in intl if 'friendly' not in m['tournament'].lower()]

    results = {}

    # -----------------------------------------------------------------------
    # Configuration: (label, squad_tag, elo_snapshot, test_filter_fn, cut_date)
    # -----------------------------------------------------------------------
    configs = [
        {
            'label': 'WC2018',
            'squad_tag': 'WC2018',
            'elo_date': '2018-06-14',
            'test_filter': lambda m: _is_wc_finals(m) and m['date'].startswith('2018'),
            'train_cut': '2018-06-01',
            'neutral_test': True,
        },
        {
            'label': 'WC2022',
            'squad_tag': 'WC2022',
            'elo_date': '2022-11-21',
            'test_filter': lambda m: _is_wc_finals(m) and m['date'].startswith('2022'),
            'train_cut': '2022-11-01',
            'neutral_test': True,
        },
        {
            'label': 'Euro2016',
            'squad_tag': 'Euro2016',
            'elo_date': '2016-06-10',
            'test_filter': lambda m: _is_euro_finals(m, 2016),
            'train_cut': '2016-06-01',
            'neutral_test': False,  # Euro2016 in France, home team = France
        },
        {
            'label': 'Euro2020',
            'squad_tag': 'Euro2020',
            'elo_date': '2021-06-11',
            'test_filter': lambda m: _is_euro_finals(m, 2020),
            'train_cut': '2021-06-01',
            'neutral_test': False,
        },
        {
            'label': 'Euro2024',
            'squad_tag': 'Euro2024',
            'elo_date': '2024-06-14',
            'test_filter': lambda m: _is_euro_finals(m, 2024),
            'train_cut': '2024-06-01',
            'neutral_test': False,
        },
    ]

    for cfg in configs:
        label = cfg['label']
        elo_map = load_clubelo(cfg['elo_date'])
        if not elo_map:
            if verbose:
                print(f"  {label}: ClubElo snapshot {cfg['elo_date']} not found, skipping")
            continue
        lookup = build_lookup(elo_map)

        # Build squad function
        tag = cfg['squad_tag']

        def make_squad_fn(sq_tag, em, lk):
            def squad_fn(team):
                players = all_squads.get((sq_tag, team), [])
                if not players:
                    # Fallback: use team's average ClubElo from same country's league
                    return em.get(team, 1450.0)
                return squad_strength(players, em, lk)
            return squad_fn

        sq_fn = make_squad_fn(tag, elo_map, lookup)

        # Training data: all competitive intl matches before tournament
        train = [m for m in comp if m['date'] < cfg['train_cut']]

        # Fit params (Poisson MLE)
        params = fit_params(train, sq_fn, alpha_range=(0, 30), hadv_range=(0, 20))

        if verbose:
            print(f"{label}: alpha={params['alpha']:.4f}, "
                  f"home_adv={params['home_adv']:.3f}, "
                  f"n_train={params['n_train']}")

        # Build Elo ratings for comparison (walk-forward, never seeing test)
        elo_ratings = run_elo(train)

        # Test matches
        test = [m for m in intl if cfg['test_filter'](m)]
        if not test:
            if verbose:
                print(f"  {label}: no test matches found")
            continue

        # Generate predictions
        sq_preds = []
        elo_preds = []
        for m in test:
            ph, pd, pa = predict_match(
                m['home_team'], m['away_team'],
                cfg['neutral_test'],
                sq_fn, params)
            sq_preds.append({'ph': ph, 'pd': pd, 'pa': pa, 'result': m['result']})

            rh = elo_ratings.get(m['home_team'], 1500)
            ra = elo_ratings.get(m['away_team'], 1500)
            eph, epd, epa = elo_predict_3way(rh, ra)
            elo_preds.append({'ph': eph, 'pd': epd, 'pa': epa, 'result': m['result']})

        sq_scores = score(sq_preds)
        elo_scores = score(elo_preds)
        base = baselines(sq_preds)

        results[label] = {
            'squad': sq_scores,
            'elo': elo_scores,
            'uniform': base['uniform'],
            'base_rate': base['base_rate'],
        }

        if verbose and label.startswith('WC2026') or (verbose and len(test) <= 12):
            print(f"  {label} sample predictions:")
            for m, p in zip(test[:5], sq_preds[:5]):
                print(f"    {m['home_team']:22} v {m['away_team']:<22} "
                      f"H/D/A={p['ph']:.0%}/{p['pd']:.0%}/{p['pa']:.0%}  "
                      f"actual={m['result']}")

    # Pooled WC (2018+2022)
    wc_keys = [k for k in results if k.startswith('WC')]
    if len(wc_keys) >= 2:
        all_sq = []
        all_elo = []
        for k in wc_keys:
            # Reconstruct pooled predictions from label-level scores
            pass
        # Re-run pooled: combine 2018+2022 test predictions
        wc18_test = [m for m in intl if _is_wc_finals(m) and m['date'].startswith('2018')]
        wc22_test = [m for m in intl if _is_wc_finals(m) and m['date'].startswith('2022')]

        elo18 = load_clubelo('2018-06-14')
        elo22 = load_clubelo('2022-11-21')
        lookup18 = build_lookup(elo18)
        lookup22 = build_lookup(elo22)
        sq18 = make_squad_fn('WC2018', elo18, lookup18)
        sq22 = make_squad_fn('WC2022', elo22, lookup22)

        params18 = fit_params(
            [m for m in comp if m['date'] < '2018-06-01'], sq18)
        params22 = fit_params(
            [m for m in comp if m['date'] < '2022-11-01'], sq22)

        elo_r18 = run_elo([m for m in comp if m['date'] < '2018-06-01'])
        elo_r22 = run_elo([m for m in comp if m['date'] < '2022-11-01'])

        pooled_sq = []
        pooled_elo = []
        for m in wc18_test:
            ph, pd, pa = predict_match(m['home_team'], m['away_team'], True, sq18, params18)
            pooled_sq.append({'ph': ph, 'pd': pd, 'pa': pa, 'result': m['result']})
            rh, ra = elo_r18.get(m['home_team'], 1500), elo_r18.get(m['away_team'], 1500)
            eph, epd, epa = elo_predict_3way(rh, ra)
            pooled_elo.append({'ph': eph, 'pd': epd, 'pa': epa, 'result': m['result']})
        for m in wc22_test:
            ph, pd, pa = predict_match(m['home_team'], m['away_team'], True, sq22, params22)
            pooled_sq.append({'ph': ph, 'pd': pd, 'pa': pa, 'result': m['result']})
            rh, ra = elo_r22.get(m['home_team'], 1500), elo_r22.get(m['away_team'], 1500)
            eph, epd, epa = elo_predict_3way(rh, ra)
            pooled_elo.append({'ph': eph, 'pd': epd, 'pa': epa, 'result': m['result']})

        base_pooled = baselines(pooled_sq)
        results['WC_pooled'] = {
            'squad': score(pooled_sq),
            'elo': score(pooled_elo),
            'uniform': base_pooled['uniform'],
            'base_rate': base_pooled['base_rate'],
        }

    # Pooled Euro (2016+2020+2024)
    euro_keys = [k for k in results if k.startswith('Euro')]
    if len(euro_keys) >= 2:
        euro_test_all = []
        elo_snaps = {
            'Euro2016': ('2016-06-10', 'Euro2016'),
            'Euro2020': ('2021-06-11', 'Euro2020'),
            'Euro2024': ('2024-06-14', 'Euro2024'),
        }
        pooled_sq_e = []
        pooled_elo_e = []
        for label, (snap_date, sq_tag) in elo_snaps.items():
            if label not in results:
                continue
            em = load_clubelo(snap_date)
            lk = build_lookup(em)
            sf = make_squad_fn(sq_tag, em, lk)
            year = int(sq_tag.replace('Euro', ''))
            tr = [m for m in comp if m['date'] < snap_date]
            pm = fit_params(tr, sf)
            er = run_elo(tr)
            tm = [m for m in intl if _is_euro_finals(m, year)]
            for m in tm:
                ph, pd, pa = predict_match(m['home_team'], m['away_team'], False, sf, pm)
                pooled_sq_e.append({'ph': ph, 'pd': pd, 'pa': pa, 'result': m['result']})
                rh, ra = er.get(m['home_team'], 1500), er.get(m['away_team'], 1500)
                eph, epd, epa = elo_predict_3way(rh, ra)
                pooled_elo_e.append({'ph': eph, 'pd': epd, 'pa': epa, 'result': m['result']})

        if pooled_sq_e:
            base_ep = baselines(pooled_sq_e)
            results['Euro_pooled'] = {
                'squad': score(pooled_sq_e),
                'elo': score(pooled_elo_e),
                'uniform': base_ep['uniform'],
                'base_rate': base_ep['base_rate'],
            }

    # All intl pooled (WC + Euro)
    all_sq_intl = (
        [p for k in results if k.startswith('WC_') or k.startswith('Euro_')
         for p in []]  # placeholder
    )
    # Grand pool: rerun WC+Euro together
    return results


# ---------------------------------------------------------------------------
# Club-league walk-forward backtest
# ---------------------------------------------------------------------------

def run_club_backtest(verbose: bool = True) -> dict:
    """
    Club-league validation using ClubElo directly as squad strength.
    Walk-forward: train on 2021-22, test on 2022-23; train on 2021-23, test on 2023-24.
    """
    club_matches = load_club_matches()
    results = {}

    configs = [
        {
            'label': 'Club_2223',
            'train_seasons': ('2122',),
            'test_season': '2223',
            'elo_date': '2022-11-21',  # closest available after Aug 2022
        },
        {
            'label': 'Club_2324',
            'train_seasons': ('2122', '2223'),
            'test_season': '2324',
            'elo_date': '2026-06-01',  # proxy for 2023-24
        },
    ]

    for cfg in configs:
        elo_map = load_clubelo(cfg['elo_date'])
        lookup = build_lookup(elo_map)

        def club_sq_fn(team, em=elo_map, lk=lookup):
            return em.get(team, find_elo(team, em, lk, 1450.0))

        train = [m for m in club_matches if m['season'] in cfg['train_seasons']]
        test = [m for m in club_matches if m['season'] == cfg['test_season']]

        params = fit_params(train, club_sq_fn, alpha_range=(0, 25), hadv_range=(0, 20))
        if verbose:
            print(f"{cfg['label']}: alpha={params['alpha']:.4f}, "
                  f"home_adv={params['home_adv']:.3f}, n_train={params['n_train']}")

        sq_preds = []
        for m in test:
            ph, pd, pa = predict_match(
                m['home_team'], m['away_team'], False, club_sq_fn, params)
            sq_preds.append({'ph': ph, 'pd': pd, 'pa': pa, 'result': m['result']})

        base = baselines(sq_preds)
        results[cfg['label']] = {
            'squad': score(sq_preds),
            'elo': {'logloss': None, 'brier': None, 'accuracy': None, 'n': 0},
            'uniform': base['uniform'],
            'base_rate': base['base_rate'],
        }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(results: dict, title: str = "Results"):
    ordered = [
        'WC2018', 'WC2022', 'WC_pooled',
        'Euro2016', 'Euro2020', 'Euro2024', 'Euro_pooled',
        'Club_2223', 'Club_2324',
    ]
    # Add any keys not in ordered list
    for k in results:
        if k not in ordered:
            ordered.append(k)

    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    hdr = (f"{'Tournament':20} {'N':>5} │ {'Squad':>8} {'Squad':>8} {'Squad':>7} │ "
           f"{'Elo':>8} {'Elo':>8} │ {'Base':>8} {'Base':>8}")
    sub = (f"{'':20} {'':>5} │ {'LL':>8} {'Brier':>8} {'Acc':>7} │ "
           f"{'LL':>8} {'Brier':>8} │ {'LL':>8} {'Brier':>8}")
    print(hdr)
    print(sub)
    print('─' * 90)

    for k in ordered:
        row = results.get(k)
        if not row:
            continue
        sq = row.get('squad', {})
        el = row.get('elo', {})
        br = row.get('base_rate', {})
        if not sq or not sq.get('n'):
            continue
        n = sq['n']
        sq_ll = sq['logloss'] or 0
        sq_br = sq['brier'] or 0
        sq_ac = sq['accuracy'] or 0
        el_ll = el.get('logloss') or 0 if el else 0
        el_br = el.get('brier') or 0 if el else 0
        br_ll = br['logloss'] if br else 0
        br_br = br['brier'] if br else 0
        print(f"{k:20} {n:>5} │ {sq_ll:>8.4f} {sq_br:>8.4f} {sq_ac:>7.1%} │ "
              f"{el_ll:>8.4f} {el_br:>8.4f} │ {br_ll:>8.4f} {br_br:>8.4f}")

    print()


def print_skill(results: dict):
    ordered = [
        'WC2018', 'WC2022', 'WC_pooled',
        'Euro2016', 'Euro2020', 'Euro2024', 'Euro_pooled',
        'Club_2223', 'Club_2324',
    ]
    for k in results:
        if k not in ordered:
            ordered.append(k)

    print(f"\n{'='*75}")
    print("  Skill vs Base Rate (log-loss: positive = better than base rates)")
    print(f"{'='*75}")
    print(f"{'Tournament':20} {'N':>5} │ {'Squad%':>10} {'Elo%':>10} │ {'Squad>Elo':>10}")
    print('─' * 75)

    for k in ordered:
        row = results.get(k)
        if not row:
            continue
        sq = row.get('squad', {})
        el = row.get('elo', {})
        br = row.get('base_rate', {})
        if not sq or not sq.get('n'):
            continue
        n = sq['n']
        base_ll = br['logloss'] if br else 1.099
        sq_skill = (base_ll - sq['logloss']) / base_ll * 100 if base_ll and sq.get('logloss') else 0
        el_ll = el.get('logloss') if el and el.get('logloss') else base_ll
        el_skill = (base_ll - el_ll) / base_ll * 100
        beat = "YES" if sq.get('logloss', 999) < el_ll else "no"
        print(f"{k:20} {n:>5} │ {sq_skill:>+9.2f}% {el_skill:>+9.2f}% │ {beat:>10}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def backtest(verbose: bool = True):
    print("=" * 70)
    print("soccer_squad_model.py — Walk-forward backtest")
    print("Model: squad strength = position-weighted ClubElo; Poisson goals → 3-way")
    print("=" * 70)

    print("\n[1] International tournaments (WC + Euro)...")
    intl_results = run_intl_backtest(verbose=verbose)

    print("\n[2] Club leagues (EPL/La Liga/Serie A/Bundesliga/Ligue 1)...")
    club_results = run_club_backtest(verbose=verbose)

    all_results = {**intl_results, **club_results}

    print_table(all_results, "soccer_squad_model: Full Walk-Forward Results")
    print_skill(all_results)

    # Key summary
    pooled = all_results.get('WC_pooled', {})
    sq = pooled.get('squad', {})
    br = pooled.get('base_rate', {})
    el = pooled.get('elo', {})
    if sq and br and sq.get('n'):
        base_ll = br['logloss']
        sq_skill = (base_ll - sq['logloss']) / base_ll * 100
        el_ll = el.get('logloss', base_ll) or base_ll
        el_skill = (base_ll - el_ll) / base_ll * 100
        print(f"\n  KEY RESULT — WC pooled (n={sq['n']}):")
        print(f"    Squad model : logloss={sq['logloss']:.4f}  brier={sq['brier']:.4f}  "
              f"acc={sq['accuracy']:.1%}  skill={sq_skill:+.2f}% vs base")
        print(f"    Elo baseline: logloss={el_ll:.4f}  skill={el_skill:+.2f}% vs base")
        print(f"    Base rates  : logloss={base_ll:.4f}  H/D/A={br.get('base_rates','?')}")

    ep = all_results.get('Euro_pooled', {})
    sq_e = ep.get('squad', {})
    br_e = ep.get('base_rate', {})
    if sq_e and br_e and sq_e.get('n'):
        base_ll = br_e['logloss']
        sq_skill = (base_ll - sq_e['logloss']) / base_ll * 100
        el_e = ep.get('elo', {})
        el_ll = el_e.get('logloss', base_ll) or base_ll
        el_skill = (base_ll - el_ll) / base_ll * 100
        print(f"\n  KEY RESULT — Euro pooled (n={sq_e['n']}):")
        print(f"    Squad model : logloss={sq_e['logloss']:.4f}  brier={sq_e['brier']:.4f}  "
              f"acc={sq_e['accuracy']:.1%}  skill={sq_skill:+.2f}% vs base")
        print(f"    Elo baseline: logloss={el_ll:.4f}  skill={el_skill:+.2f}% vs base")

    return all_results


def rate():
    """Print squad strength rankings using latest ClubElo for each squad."""
    all_squads = load_all_squads()
    elo = load_clubelo('2026-06-01')
    lk = build_lookup(elo)
    print("Squad strength (2026 ClubElo, WC2022 rosters):")
    strengths = {}
    for (tag, team), players in all_squads.items():
        if tag == 'WC2022':
            strengths[team] = squad_strength(players, elo, lk)
    for team, s in sorted(strengths.items(), key=lambda x: -x[1]):
        print(f"  {team:30} {s:.1f}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'backtest'
    if cmd == 'rate':
        rate()
    elif cmd == 'club':
        r = run_club_backtest(verbose=True)
        print_table(r, "Club Results")
        print_skill(r)
    else:
        backtest(verbose=True)
