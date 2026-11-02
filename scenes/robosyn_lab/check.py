"""Acceptance checks for the RoboSyn lab bench scene across seeds: expert runs,
prop pixel counts in the head and body cameras, previews; see sim/scene_check.
"""

from scenes.robosyn_lab import task as kt
from scenes.robosyn_lab.run import LabScene, run
from sim import scene_check

if __name__ == "__main__":
    scene_check.main(
        __doc__,
        kt.sample_episode,
        run,
        LabScene,
        props=("beaker", "mixer", "rack"),
        plain=("tray",),
        fields=lambda r: {"side": r["side"], "ran_s": r["ran_s"]},
    )
