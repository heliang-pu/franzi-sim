"""Acceptance checks for the RoboSyn basket scene across seeds: expert runs,
prop pixel counts in the head and body cameras, previews; see sim/scene_check.
"""

from scenes.robosyn_basket import task as kt
from scenes.robosyn_basket.run import BasketScene, run
from sim import scene_check

if __name__ == "__main__":
    scene_check.main(
        __doc__,
        kt.sample_episode,
        run,
        BasketScene,
        props=BasketScene.OBJECTS,
        fields=lambda r: {"side": r["side"], "basket_offset_mm": r["basket_offset_mm"]},
    )
