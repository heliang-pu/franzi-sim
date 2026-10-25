"""Acceptance checks for the RoboSyn drawer scene across seeds: expert runs,
prop pixel counts in the head and body cameras, previews; see sim/scene_check.
"""

from scenes.robosyn_drawer import task as kt
from scenes.robosyn_drawer.run import DrawerScene, run
from sim import scene_check

if __name__ == "__main__":
    scene_check.main(
        __doc__,
        kt.sample_episode,
        run,
        DrawerScene,
        props=(*kt.ITEMS, "drawer", "button"),
        plain=("handover_pad",),
        fields=lambda r: {"deepest_press_mm": r["deepest_press_mm"], "item": r["item"]},
        variants=kt.VARIANTS,
    )
