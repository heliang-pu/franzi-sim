"""Acceptance checks for the two-handed tray scene across seeds: expert runs,
prop pixel counts in the head and body cameras, previews; see sim/scene_check.
"""

from scenes.robosyn_tray import task as kt
from scenes.robosyn_tray.run import TrayScene, run
from sim import scene_check

if __name__ == "__main__":
    scene_check.main(
        __doc__,
        kt.sample_episode,
        run,
        TrayScene,
        props=TrayScene.OBJECTS,
        fields=lambda r: {"side": r["side"], "tray_offset_mm": r["tray_offset_mm"]},
    )
