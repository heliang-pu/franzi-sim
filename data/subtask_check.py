"""Check that a recorded run splits into single-arm subtasks.

    python -m data.subtask_check outputs/<run> [...]

The user's rule (2026-10-25): a subtask is one action of one arm on one object
(pick X, place X at Y, return the arm, press a button), so an episode can be
cut at its subtask boundaries; the finer motion stages stay in the trace's
phase field. While one arm works, the other holds: its commanded joints and
gripper do not change. Both arms may act in one subtask only on one object
together, and then its text says "both hands".

A segment is a run of control ticks with the same subtask text. For each, the
arms whose commands change in it (the trace's action: left 0-6 and 14, right
7-13 and 15) must be one arm, or both under a "both hands" subtask. Exits 1
if any run breaks the rule.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ARMS = {"left": [*range(7), 14], "right": [*range(7, 14), 15]}
BOTH = "both hands"


def segments(subtasks):
    """[(text, first tick, last tick + 1)] of consecutive equal texts."""
    out, start = [], 0
    for k in range(1, len(subtasks) + 1):
        if k == len(subtasks) or subtasks[k] != subtasks[start]:
            out.append((str(subtasks[start]), start, k))
            start = k
    return out


def check(run):
    """(problems, rows): each segment's text, ticks and moving arms."""
    trace = json.loads((Path(run) / "trace.json").read_text())
    action = np.array([t["action"] for t in trace])
    texts = [t["subtask"] for t in trace]
    problems, rows = [], []
    for text, a, b in segments(texts):
        # The command applied at tick k is action[k]; a segment's moves are
        # the changes from the tick before it to its last tick.
        window = action[max(a - 1, 0) : b]
        moving = [
            arm
            for arm, columns in ARMS.items()
            if np.abs(np.diff(window[:, columns], axis=0)).max(initial=0.0) > 1e-9
        ]
        rows.append({"subtask": text, "ticks": [a, b], "moving": moving})
        if len(moving) == 2 and BOTH not in text.lower():
            problems.append(f"both arms move in {text!r} (ticks {a}-{b})")
    return problems, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--verbose", action="store_true", help="list every segment")
    args = parser.parse_args()
    bad = 0
    for run in args.runs:
        problems, rows = check(run)
        bad += bool(problems)
        print(f"{run}: {len(rows)} subtasks, {'OK' if not problems else 'BROKEN'}")
        for row in rows if args.verbose else []:
            moving = "+".join(row["moving"]) or "none"
            print(
                f"  {row['ticks'][0]:5d}-{row['ticks'][1]:<5d} {moving:10s} {row['subtask']}"
            )
        for problem in problems:
            print("  ", problem)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
