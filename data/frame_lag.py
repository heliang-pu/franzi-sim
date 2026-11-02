"""Whether a recorded run's camera frames show the state of their own tick.

    python -m data.frame_lag outputs/<run> [...] [--reference outputs/<settled run>]

frame_lag(run) estimates the videos' delay from the run alone. Each moving
policy camera's motion per tick is measured twice: from the state
(trajectory.npz camera_pose: the translation over DEPTH plus the rotation
angle) and from its video (the mean absolute grey-level change between
frames). Their correlation, summed over the cameras, peaks at the delay. It
needs sharp motion: on tote_transfer with the ready move recorded, a MuJoCo
run gave 0 and the Isaac runs from before isaac/runtime.py waited for each
camera's frame of the current step +1, by margins of about 0.01. Started in
the default posture (task version 3), the curves are flat: MuJoCo, aligned
by construction, led by 0.0004 and the Isaac runs leaned to -1 by up to
0.002, so it decides nothing there.

frame_offset(run, reference) decides: the reference is a re-run of the same
seed with settled frames (isaac.episode --settle), whose physics is identical,
and the offset that makes the run's frames match the reference's is the
delay. On task version 3, seeds 0 and 9: 0, mean differences 1.77 against 6.4
to 7.6 at +-1 on the moving frames (2026-10-25).

Both need the full log (camera_pose, sim/episode.py since 2026-10-25).
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np

# Turns camera translation into an image motion comparable with its rotation
# (rad): the wrist and body cameras look at the bench 0.2-0.5 m away.
DEPTH = 0.3
SKIP = ("head_d435",)
# Cameras whose state motion is under this share of the most moving camera's
# are left out too: a camera that holds still correlates only render noise.
MOVING = 0.1
SPAN = 3


def video_motion(path):
    """Mean absolute grey-level change between consecutive frames."""
    previous, motion = None, []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            grey = frame.to_ndarray(format="gray").astype(np.float32)
            if previous is not None:
                motion.append(np.abs(grey - previous).mean())
            previous = grey
    return np.array(motion)


def pose_motion(pose):
    """Camera motion per tick from rows of x, y, z, qw, qx, qy, qz."""
    step = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    dot = np.abs(np.sum(pose[1:, 3:] * pose[:-1, 3:], axis=1)).clip(0, 1)
    return step / DEPTH + 2 * np.arccos(dot)


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def frame_lag(run, span=SPAN):
    """(lag, margin, scores): the videos' delay in ticks, how far its summed
    correlation leads the next lag's, and the summed correlation by lag over
    the cameras that move (MOVING; never the head)."""
    run = Path(run)
    data = np.load(run / "trajectory.npz")
    if "camera_pose" not in data:
        raise ValueError(f"{run}: no camera_pose (a run from before the full log)")
    motion = {
        str(name): pose_motion(data["camera_pose"][:, c])
        for c, name in enumerate(data["camera_names"])
        if name not in SKIP
    }
    top = max(state.sum() for state in motion.values())
    scores = dict.fromkeys(range(-span, span + 1), 0.0)
    for name, state in motion.items():
        if state.sum() < MOVING * top:
            continue
        video = video_motion(run / f"{name}.mp4")
        n = min(len(state), len(video))
        for lag in scores:
            lo, hi = max(0, -lag), min(n, n - lag)
            scores[lag] += correlation(state[lo:hi], video[lo + lag : hi + lag])
    first, second = sorted(scores.values(), reverse=True)[:2]
    lag = max(scores, key=scores.get)
    return lag, round(first - second, 4), {k: round(v, 4) for k, v in scores.items()}


def frames(path):
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="gray") for f in container.decode(video=0)]


def distance(a, b):
    return float(np.abs(a.astype(np.float32) - b).mean())


def frame_offset(run, reference, span=2):
    """(offset, ratio, distances): the frame offset o that best matches frame
    k of run to frame k + o of reference (0: frames on their own tick), how
    many times larger the next best offset's distance is, and the mean
    absolute grey-level distance by offset. Counted on the frames where each
    moving camera moves most (its top 30 % of state motion)."""
    run, reference = Path(run), Path(reference)
    data = np.load(run / "trajectory.npz")
    if not np.array_equal(
        data["state"], np.load(reference / "trajectory.npz")["state"]
    ):
        raise ValueError(f"{reference}: not the same physics as {run}")
    distances = dict.fromkeys(range(-span, span + 1), 0.0)
    motion = {
        str(name): pose_motion(data["camera_pose"][:, c])
        for c, name in enumerate(data["camera_names"])
        if name not in SKIP
    }
    top = max(state.sum() for state in motion.values())
    for name, state in motion.items():
        if state.sum() < MOVING * top:
            continue
        ours, theirs = frames(run / f"{name}.mp4"), frames(reference / f"{name}.mp4")
        cut = np.quantile(state, 0.7)
        ticks = [k for k in range(span, len(state) - span) if state[k] >= cut]
        for o in distances:
            distances[o] += np.mean([distance(ours[k], theirs[k + o]) for k in ticks])
    first, second = sorted(distances.values())[:2]
    offset = min(distances, key=distances.get)
    rounded = {k: round(float(v), 3) for k, v in distances.items()}
    return offset, round(float(second / first), 2), rounded


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path, help="settled re-run of one run")
    args = parser.parse_args()
    if args.reference:
        offset, ratio, distances = frame_offset(args.runs[0], args.reference)
        print(f"{args.runs[0]}: offset {offset} (x{ratio}) {json.dumps(distances)}")
        return
    for run in args.runs:
        lag, margin, scores = frame_lag(run)
        print(f"{run}: lag {lag} (margin {margin}) {json.dumps(scores)}", flush=True)


if __name__ == "__main__":
    main()
