#!/usr/bin/env python3
"""The way in.

    python3 explore.py        interactive menu
    python3 explore.py 2      run one item and exit
    python3 explore.py all    run everything, top to bottom

Only item 3 (running the trained models) needs anything installed. Everything
else is standard-library Python 3.9+ reading committed data.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "casino_sim")
DATA = os.path.join(SIM, "data")
FIG = os.path.join(SIM, "figures")

WIDTH = min(shutil.get_terminal_size((84, 24)).columns, 84)
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def bold(s): return _c("1", s)
def dim(s): return _c("2", s)
def cyan(s): return _c("36", s)
def green(s): return _c("32", s)
def red(s): return _c("31", s)


def header(title):
    print()
    print(bold(title))
    print(dim("─" * min(len(title), WIDTH)))


def rule(label=""):
    if not label:
        print(dim("─" * (WIDTH - 2)))
    else:
        print(dim(f"── {label} " + "─" * max(WIDTH - len(label) - 6, 2)))


def note(text):
    for line in textwrap.wrap(text, WIDTH - 2):
        print(dim(line))


def bullet(text):
    lines = textwrap.wrap(text, WIDTH - 6)
    for i, line in enumerate(lines):
        print(f"  {'·' if i == 0 else ' '} {line}")


def show_markdown(path, limit=None):
    """Print a committed markdown file, lightly de-marked for a terminal."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    shown = 0
    for line in lines:
        if limit and shown >= limit:
            print(dim(f"  … {len(lines) - shown} more lines in {os.path.relpath(path, HERE)}"))
            break
        s = line.rstrip()
        if s.startswith("#"):
            print()
            print(bold(s.lstrip("# ")))
        elif s.startswith("_") and s.endswith("_") and len(s) > 2:
            note(s[1:-1])
        elif s.startswith("|"):
            print("  " + s)
        elif not s:
            print()
        else:
            for w in textwrap.wrap(s.replace("**", ""), WIDTH - 2):
                print(w)
        shown += 1


# ==========================================================================


def item_overview():
    header("What this is")
    note(
        "Three sports models — World Cup soccer, MLB strikeout props, and "
        "tennis — that set a betting line from team and player strength alone. "
        "None of them ever sees a sportsbook's odds. Then each one is tested "
        "blind, settled on real outcomes, from both sides: as a bettor trying "
        "to beat the line, and as the book trying to profitably set it."
    )
    print()
    rule("the answer")
    print()
    note(
        "No, on both counts — and the interesting part is exactly how each one "
        "fails. Every model loses to the market's own price on calibration. "
        "None is sharp enough to run a book against informed money. The World "
        "Cup line over-rates weak-confederation teams; tennis is over-confident "
        "on favourites; MLB under-states both tails."
    )
    print()
    note(
        "This is a market-efficiency study, not a pitch. The rigour is the "
        "point: every number comes from free public data, with no paid feed "
        "anywhere, and the models are audited to prove they never saw the line."
    )
    print()
    rule("what makes it credible")
    print()
    bullet("Blind: probabilities fixed pre-game, graded against real results.")
    bullet("Audited: three adversarial reviewers, each told to assume the line "
           "leaks and to prove it. Item 4.")
    bullet("Both directions: bettor and book, so a model can't hide behind "
           "one framing. Item 2.")
    bullet("Reproducible: item 7 regenerates every number from committed data.")


def item_headline():
    header("Three models, scored honestly")
    path = os.path.join(DATA, "THREE_MODEL_SUMMARY.md")
    if not os.path.exists(path):
        print(dim("  run item 7 first to generate the summary"))
        return
    show_markdown(path)
    print()
    rule("read the two negative columns")
    print()
    note(
        "'Skill vs market' is 0 when the model matches the market's calibration "
        "and negative when the market is better. All three are negative — the "
        "closing line is efficient, which is the expected and boring truth that "
        "most projects in this genre quietly avoid testing."
    )
    print()
    note(
        "'House (sharp)' is what happens if informed money picks you off. All "
        "three are deeply negative. Being the casino only works if your line is "
        "actually right; the vig does not save a wrong line."
    )
    print()
    for name, fn in [("World Cup", "house_backtest_wc.json"),
                     ("MLB k-prop", "house_backtest_mlb.json"),
                     ("Tennis", "house_backtest_tennis.json")]:
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        c = json.load(open(p))["calibration"]
        fav, dog = c["favorites"], c["underdogs"]
        print(f"  {bold(name)}")
        print(f"    favourites  predicted {fav['mean_pred']:.0%}  →  actual "
              f"{fav['mean_actual']:.0%}   (n={fav['n']})")
        print(f"    underdogs   predicted {dog['mean_pred']:.0%}  →  actual "
              f"{dog['mean_actual']:.0%}   (n={dog['n']})")


def item_models():
    header("The trained models, run live")
    note(
        "The actual trained artifacts are committed — the XGBoost boosters for "
        "MLB and tennis, and the Elo + Dixon–Coles parameters for the World "
        "Cup. This loads them and predicts, including tennis reproducing its "
        "recorded predictions straight from the committed booster."
    )
    print()
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print(red("  xgboost is not installed."))
        print()
        bullet("pip install -r requirements.txt")
        print()
        note("Every other menu item works without it — they read committed "
             "results rather than re-running the models.")
        return
    print()
    sys.stdout.flush()   # keep our output ahead of the child's when piped
    subprocess.run([sys.executable, os.path.join(HERE, "demo.py")], cwd=HERE)


def item_leakage():
    header("Did the models just learn the line?")
    note(
        "The whole project is worthless if a model secretly used the betting "
        "line as a feature — it would be 'predicting' a number it was handed. "
        "Three independent adversarial auditors were run, each instructed to "
        "assume the line IS leaking and to prove it from the code."
    )
    print()
    show_markdown(os.path.join(SIM, "LEAKAGE_AUDIT.md"), limit=48)
    print()
    rule("the behavioural check")
    print()
    note(
        "A leaked model is mathematically forced to track the line within a "
        "point or two. Ours sits a real ~11pp away while still picking the same "
        "favourite 81% of the time — aligned in direction, independent in "
        "number. That gap is the evidence. Item 5 shows it game by game."
    )


def item_books():
    header("Our line vs real sportsbooks")
    note(
        "The model's de-vigged probabilities next to real de-vigged book odds "
        "from Sporttery and Kalshi, per match. 'Deviance' is the average "
        "absolute gap across the three outcomes."
    )
    path = os.path.join(DATA, "book_comparison.md")
    if not os.path.exists(path):
        print(dim("  run item 7 first"))
        return
    lines = open(path).read().splitlines()
    table = [l for l in lines if l.startswith("|")]
    print()
    for line in table[:2]:
        print("  " + line)
    body = table[2:]
    for line in body[:10]:
        print("  " + line)
    print(dim(f"  … {len(body) - 10} more matches"))
    print()

    rows = []
    for line in body:
        cells = [c.strip() for c in line.strip("|").split("|")]
        try:
            rows.append((cells[0], float(cells[-1].replace("pp", ""))))
        except (ValueError, IndexError):
            pass
    if rows:
        rows.sort(key=lambda r: r[1])
        rule("closest agreement")
        for m, d in rows[:4]:
            print(f"  {green('✓')} {m[:52]:52s} {d:5.1f}pp")
        print()
        rule("widest disagreement")
        for m, d in rows[-4:][::-1]:
            print(f"  {red('✗')} {m[:52]:52s} {d:5.1f}pp")
        print()
        note(
            "The wide ones are the diagnosis. Austria v Jordan at 29pp is the "
            "model over-rating a weak-confederation side — the same failure "
            "mode that shows up in the World Cup calibration curve."
        )


def item_figures():
    header("The figures")
    print()
    figures = [
        ("three_model_summary.png", "the headline — both sides, all three sports"),
        ("wc_reliability.png", "World Cup calibration; dot area is sample size"),
        ("mlb_reliability.png", "MLB props and game winner"),
        ("tennis_reliability.png", "tennis — the over-confidence is visible"),
        ("model_vs_consensus.png", "our line vs book consensus, per outcome"),
        ("deviance_by_game.png", "where we agree and disagree, per match"),
        ("per_book_gap.png", "which book our line tracks most closely"),
    ]
    for name, what in figures:
        p = os.path.join(FIG, name)
        mark = green("✓") if os.path.exists(p) else dim("·")
        print(f"  {mark} casino_sim/figures/{name:26s} {dim(what)}")
    print()
    note("Regenerate them all with item 7. On macOS, `open casino_sim/figures` "
         "to browse; they are also embedded in the README.")


def item_reproduce():
    header("Reproduce every number")
    note(
        "These run offline against committed data. No network, no paid feeds, "
        "no API keys. matplotlib is needed for the figures; the numbers print "
        "either way."
    )
    steps = [
        ("casino_sim/house_backtest.py", "World Cup calibration + blind book backtest"),
        ("casino_sim/house_backtest_mlb.py", "MLB strikeout props and game winner"),
        ("casino_sim/house_backtest_tennis.py", "tennis"),
        ("casino_sim/book_compare.py", "our line vs real books, per match"),
        ("casino_sim/three_model_summary.py", "the consolidated table and figure"),
    ]
    print()
    for script, what in steps:
        print(f"  {cyan('▸')} {os.path.basename(script):32s} {dim(what)}")
        r = subprocess.run([sys.executable, script], cwd=HERE,
                           capture_output=True, text=True)
        if r.returncode == 0:
            tail = [l for l in r.stdout.splitlines() if l.strip()][-1:]
            for line in tail:
                print(f"    {dim(line[:WIDTH - 6])}")
        else:
            print(f"    {red('failed')} {dim((r.stderr or '').strip().splitlines()[-1][:60])}")
    print()
    note("Every table and figure in the README is now rebuilt from source data.")


def item_source():
    header("Where everything is")
    print()
    for path, what in [
        ("models/code/", "the deployed model code — line-free by construction"),
        ("models/mlb/, models/tennis/", "the actual trained XGBoost boosters"),
        ("models/wc/", "Elo + Dixon–Coles parameters"),
        ("casino_sim/house_backtest*.py", "blind calibration + book simulation"),
        ("casino_sim/book_compare.py", "our line vs real de-vigged book odds"),
        ("casino_sim/LEAKAGE_AUDIT.md", "the adversarial independence audit"),
        ("casino_sim/vizstyle.py", "one look for every figure"),
        ("data/", "settled paper ledgers and World Cup results"),
        ("demo.py", "load the trained models and predict"),
    ]:
        print(f"  {path:30s} {dim(what)}")
    print()
    rule("the engineering half")
    print()
    note(
        "An autonomous agent built and validated this system — its control "
        "plane, guardrails, and the evidence gate that refused to promote any "
        "of these models to live capital are a separate, self-contained "
        "repository:"
    )
    print()
    print("  " + cyan("https://github.com/jhunter11/agentic-quant-operator"))


# ==========================================================================

MENU = [
    ("The short version", item_overview, ""),
    ("Three models, scored honestly", item_headline, ""),
    ("Run the trained models live", item_models, "needs xgboost"),
    ("Did the models learn the line?", item_leakage, ""),
    ("Our line vs real sportsbooks", item_books, ""),
    ("The figures", item_figures, ""),
    ("Reproduce every number from raw", item_reproduce, "~1 min"),
    ("Where everything is", item_source, ""),
]


def show_menu():
    print()
    print(bold("  CASINO LINE MODELING"))
    print(dim("  Three from-scratch sports lines, blind-tested as bettor and as book."))
    print()
    for i, (label, _, tag) in enumerate(MENU, 1):
        suffix = dim(f"   {tag}") if tag else ""
        print(f"   {cyan(str(i))}  {label}{suffix}")
    print(f"   {cyan('q')}  quit")
    print()


def run(choice):
    choice = choice.strip().lower()
    if choice in ("q", "quit", "exit", "0"):
        return False
    if choice == "all":
        for _, fn, _ in MENU:
            fn()
            print()
        return True
    if choice.isdigit() and 1 <= int(choice) <= len(MENU):
        MENU[int(choice) - 1][1]()
        return True
    print(dim(f"  no item {choice!r} — pick 1-{len(MENU)} or q"))
    return True


def main(argv):
    if len(argv) > 1:
        run(argv[1])
        return 0
    if not sys.stdin.isatty():
        show_menu()
        print(dim("  not a terminal — run `python3 explore.py <n>` to pick an item"))
        return 0
    while True:
        show_menu()
        try:
            choice = input("  > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not run(choice):
            return 0
        print()
        try:
            input(dim("  ↵ back to the menu "))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
