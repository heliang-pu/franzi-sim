"""Acceptance checks for the RoboSyn kitchen scene across seeds: expert runs,
prop pixel counts in the head and body cameras, previews; see sim/scene_check.
"""

from scenes.robosyn_kitchen import task as kt
from scenes.robosyn_kitchen.run import KitchenScene, run
from sim import scene_check

if __name__ == "__main__":
    scene_check.main(
        __doc__,
        kt.sample_episode,
        run,
        KitchenScene,
        props=(*KitchenScene.OBJECTS, "button"),
        plain=("lid_rest",),
        fields=lambda r: {"deepest_press_mm": r["deepest_press_mm"]},
        variants=kt.VARIANTS,
    )
