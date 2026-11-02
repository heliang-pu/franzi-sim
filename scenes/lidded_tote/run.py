"""Lidded tote episode: the scene and a scripted IK expert.

LidScene is sim.tabletop.ObjectScene with task.py's two loose parts (the hex
blank to move and the flange that stays in the tote) and the tote's hinged
lid: the episode framework's audits and recordings, plus object tracking, the
milestones (lid open past upright, the hex blank lifted out, then lying on
the station plate, the lid shut after that) and the success test, which also
needs the flange still in its place. The expert reads the lid's angle and the
parts' poses from the simulator (privileged): no welds, no object pose or
joint writes, no contact-mask changes, no robot qpos/qvel writes.

The left hand pinches the pull tab across its 60 mm width and carries it
round the hinge while the tab turns between the pads (task.py). Every arm
move is a joint path of chained IK waypoints from task.py (lines, arcs, turns
in place), run by follow() as one minimum-jerk move, timed and rate-limited
as Scene.move times its straight lines; only the long reach to the tab and
the way home are partly joint-space moves (task.reach_path). The hex blank
is set down from its measured pose, its lowest point RELEASE_GAP over the
plate: under PhysX a carried part turns 2-4 degrees about the pads, under
MuJoCo it creeps down them. The lid is left open past its balance point and
closed from its measured angle, so the episode does not depend on the hinge
friction holding it (PhysX has none).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.lidded_tote import task as kt
from sim import tabletop as tt
from sim.episode import minimum_jerk


class LidScene(tt.ObjectScene):
    OBJECTS = ("hex", "flange")
    HELD = ("hex",)
    MILESTONES = ("lid_open", "hex_out", "hex_on_plate", "lid_closed")
    # "lower" ends with the blank RELEASE_GAP over the plate, which PhysX
    # reports as touching (its contacts count within 0.5 mm).
    CARRY_PHASES = ("carry",)

    def __init__(self, output, spec, record=False):
        super().__init__(output, spec, record)
        self.lid_body = self.m.body("tote_lid_part").id
        self.lid_qadr = int(self.m.joint("tote_lid").qposadr[0])
        self.tote_body = self.m.body("tote_part").id
        self.flange_home = np.array(spec["flange"]["xy"], float)
        self.widest = 0.0

    def build_root(self):
        return kt.build(self.spec)

    def lid_angle(self):
        return float(self.d.qpos[self.lid_qadr])

    def tote_pose(self):
        """The tote's measured place ({"xy", "yaw"}; it is fixed to the bench)."""
        R = self.d.xmat[self.tote_body].reshape(3, 3)
        return {
            "xy": [float(v) for v in self.d.xpos[self.tote_body][:2]],
            "yaw": float(np.arctan2(R[1, 0], R[0, 0])),
        }

    def lid_touching(self, pairs):
        """Robot bodies touching the lid."""
        rows = pairs[(pairs == self.lid_body).any(1)]
        return set(rows.ravel().tolist()) & self.robot_bodies

    def on_plate(self, name, pairs):
        x, y, z = self.xyz(name)
        plate = self.spec.get("plate", kt.PLATE)
        return (
            abs(x - plate[0]) < kt.PLATE_HALF[0] - 0.02
            and abs(y - plate[1]) < kt.PLATE_HALF[1] - 0.02
            and abs(z - kt.PLATE_TOP) < 0.003
            and self.settled(name, pairs)
        )

    def flange_home_ok(self, pairs):
        """The flange still lies where it started on the tote's floor."""
        x, y, z = self.xyz("flange")
        return (
            np.hypot(x - self.flange_home[0], y - self.flange_home[1]) < 0.01
            and abs(z - kt.FLOOR_Z) < 0.003
            and self.settled("flange", pairs)
        )

    def conditions(self, pairs):
        angle = self.lid_angle()
        held = self.touching("hex", pairs).issuperset(self.jaws["left"])
        now = {
            "lid_open": angle >= kt.OPENED,
            "hex_out": held and self.xyz("hex")[2] > kt.RIM_Z + 0.005,
            "hex_on_plate": self.on_plate("hex", pairs),
        }
        # Shut once the hex blank lies on the plate, the hand off the lid.
        now["lid_closed"] = (
            angle <= kt.CLOSED and now["hex_on_plate"] and not self.lid_touching(pairs)
        )
        return now

    def sample(self):
        record = super().sample()
        record["lid_angle_rad"] = self.lid_angle()
        return record

    def track_objects(self):
        super().track_objects()
        self.widest = max(self.widest, self.lid_angle())

    def complete(self):
        """The final state: the hex blank on the plate, the lid shut after it
        had been opened past upright, the flange untouched in the tote, all
        still and clear of the robot."""
        pairs = self.contact_bodies()
        now = self.status(pairs)
        return (
            now["lid_closed"]
            and self.reached["lid_open"]
            and self.reached["hex_out"]
            and self.flange_home_ok(pairs)
        )


def follow(s, path, seconds=2.0, grips=None, speed=tt.SPEED):
    """Run the left arm through the joint waypoints path (after its present
    goal) in one minimum-jerk move: timed by joint travel at speed and
    stretched to the controller's rates, as Scene.move(speed=) times the IK
    waypoints of a straight line. The right arm holds its goal."""
    paths = {side: np.array([s.goal[side], s.goal[side]]) for side in s.goal}
    paths["left"] = np.vstack([s.goal["left"], path])
    arcs = {
        side: np.r_[0, np.cumsum(np.abs(np.diff(p, axis=0)).max(1))]
        for side, p in paths.items()
    }
    seconds = max(seconds, 1.875 * max(a[-1] for a in arcs.values()) / speed)
    seconds = s.stretch(paths, arcs, seconds)
    start = s.grip.copy()
    end = start | (grips or {})
    n = int(seconds / s.m.opt.timestep)
    for i in range(n):
        p = minimum_jerk((i + 1) / n)
        if s.step_number % 20 == 0:
            for side, p_side in paths.items():
                s.goal[side] = s.along(p_side, arcs[side], p)
                s.grip[side] = start[side] + (end[side] - start[side]) * p
        s.physics_step()


def check_pinch(s):
    pairs = s.contact_bodies()
    others = set(pairs[(pairs == s.lid_body).any(1)].ravel().tolist())
    if not set(s.jaws["left"]) <= others:
        raise RuntimeError("left: no bilateral pinch on the lid's pull tab")


def hex_yaw(s):
    R = s.d.xmat[s.ids["hex"]].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def hex_low(s):
    """The held hex blank's lowest point (z, m), measured: its base's rim
    turned as the blank is (under PhysX it hangs 2-4 degrees off upright)."""
    a = np.linspace(0, 2 * np.pi, 13)[:-1]
    rim = kt.HEX_CORNER * np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
    R = s.d.xmat[s.ids["hex"]].reshape(3, 3)
    return float((s.xyz("hex") + rim @ R.T)[:, 2].min())


def place_target(s, spot):
    """The grasp-site target that puts the held blank's centre over spot and
    its lowest point RELEASE_GAP above the plate, from its measured pose."""
    site = s.d.site_xpos[s.sites["left"]]
    offset = site - s.xyz("hex")
    drop = hex_low(s) - (kt.PLATE_TOP + kt.RELEASE_GAP)
    return np.array([spot[0] + offset[0], spot[1] + offset[1], site[2] - drop])


def expert(s):
    tote = s.tote_pose()
    shut = kt.pinch_rotation(tote, 0.0)
    tab_open = tt.grip_open(kt.TAB_WIDTH)
    tab_shut = tt.grip_closed(kt.TAB_WIDTH, kt.PINCH_SQUEEZE)
    hex_open = tt.grip_open(kt.HEX_FLATS)

    # Every arm move is a joint path from task.py, run by follow(): the same
    # paths the sampler planned, chained from the arm's present goal. done
    # keeps each move's start joints and path, which the way back plays
    # backwards (task.closing).
    done = {}

    def go(path, seconds=2.0, name=None, **kw):
        if name:
            done[name] = (s.goal["left"].copy(), path)
        follow(s, path, seconds, **kw)
        return path

    def line(target, rotation, seconds=2.0, name=None, **kw):
        path = kt.line(s.kin, s.goal["left"], target, rotation, s.body())
        return go(path, seconds, name, **kw)

    def turn(start, end, seconds=1.5, name=None, **kw):
        path = kt.turn(s.kin, s.goal["left"], start, end, s.body())
        return go(path, seconds, name, **kw)

    def along_arc(start, end, seconds, name=None, **kw):
        path = kt.arc(s.kin, s.goal["left"], tote, start, end, s.body())
        return go(path, seconds, name, **kw)

    def back(name, seconds=1.5, **kw):
        """Play a move of the way out backwards, to the joints it began from."""
        start, path = done[name]
        return go(np.vstack([path[-2::-1], start]), seconds, **kw)

    # One subtask per action of the left arm (user, 2026-10-25): its text is
    # the trace's subtask, the stage name the finer phase, the hand's in-place
    # turns included; the right arm holds throughout.
    open_lid = "Open the tote's lid with the left hand."
    pick = "Pick up the hex blank with the left hand."
    place = "Put the hex blank on the station plate with the left hand."
    close_lid = "Close the tote's lid with the left hand."
    home_text = "Move the left arm back to its start pose."
    tt.start_home(s)
    home = s.goal["left"].copy()
    # Over the tab: straight up, joint-space to the stage in front of it,
    # then a line (task.reach_path).
    rise, stage, approach = kt.reach_path(s.kin, home, tote, s.body())
    s.stage("reach_tab", open_lid)
    go(rise, 2.0, "rise")
    s.move(
        joints={"left": stage},
        seconds=3.0,
        grips={"left": tab_open},
        speed=tt.SPEED,
    )
    go(approach, 1.5, "reach_tab")
    s.stage("align_tab", open_lid)
    line(kt.pinch_site(tote, s.lid_angle()), shut, 2.0, "align_tab")
    s.stage("pinch_tab", open_lid)
    s.move(grips={"left": tab_shut}, seconds=1.0)
    s.move(seconds=0.5)
    check_pinch(s)
    s.stage("open_lid", open_lid)
    along_arc(s.lid_angle(), kt.OPEN_TO, 6.0, "open_lid")
    if s.lid_angle() < kt.OPEN_TO - np.radians(5):
        raise RuntimeError(f"the lid opened only {np.degrees(s.lid_angle()):.0f} deg")
    s.stage("release_tab", open_lid)
    s.move(grips={"left": tab_open}, seconds=1.0)
    s.move(seconds=0.3)
    s.stage("back_off", open_lid)
    along_arc(kt.OPEN_TO, kt.close_start(kt.OPEN_TO), 2.0, "back_off")
    base = [float(v) for v in s.xyz("hex")]
    poses = kt.hex_poses(tote, base, hex_yaw(s), s.spec["spot"])
    down, held = poses["rotation"], kt.pinch_rotation(tote, kt.close_start(kt.OPEN_TO))
    s.stage("over_hex", pick)
    line(poses["above"], held, 2.5, "over_hex")
    s.stage("turn_down", pick)
    turn(held, down, 1.5, "turn_down", grips={"left": hex_open})
    s.stage("align_hex", pick)
    line(poses["grasp"], down, 2.0)
    s.stage("grasp_hex", pick)
    s.move(grips={"left": tt.grip_closed(kt.HEX_FLATS, kt.HEX_SQUEEZE)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, "hex", "left")
    s.stage("lift_hex", pick)
    line(poses["raise"], down, 1.0)
    line(poses["lift"], down, 2.5)
    if hex_low(s) < kt.RIM_Z:
        raise RuntimeError("lift failed for the hex blank")
    s.stage("carry", place)
    line(poses["over_rim"], down, 1.0, "over_rim")
    line(poses["rim"], down, 1.0, "rim")
    line(poses["high"], down, 2.0, "high")
    s.stage("lower", place)
    # Re-measured before setting down: the blank may have slipped in the jaws
    # (MuJoCo's soft friction) or turned about the pads (PhysX).
    above_plate = s.goal["left"].copy()
    lowered = [line(place_target(s, s.spec["spot"]), down, 1.5)]
    s.move(seconds=0.3)
    for _ in range(2):
        if hex_low(s) - kt.PLATE_TOP < 2 * kt.RELEASE_GAP + 0.0005:
            break
        lowered.append(line(place_target(s, s.spec["spot"]), down, 0.5))
        s.move(seconds=0.3)
    done["lower"] = (above_plate, np.vstack(lowered))
    s.stage("release_hex", place)
    s.move(grips={"left": hex_open}, seconds=1.0)
    s.move(seconds=0.3)
    s.stage("retreat", place)
    back("lower")
    # Close the lid from wherever it stands, the way the hand came.
    angle = s.lid_angle()
    if angle < kt.OPEN_TO - np.radians(1):
        raise RuntimeError(f"the lid fell to {np.degrees(angle):.0f} deg")
    s.stage("back_to_tote", close_lid)
    back("high", 2.0)
    back("rim", 1.0)
    back("over_rim", 1.0)
    line(poses["above"], down, 1.5)
    turned = done["turn_down"][1][-1]
    if np.abs(s.goal["left"] - turned).max() > kt.REJOIN:
        raise RuntimeError("back over the hex, the arm is too far from its joints")
    s.move(joints={"left": turned}, seconds=1.0, speed=tt.SPEED)
    way = kt.closing(s.kin, done, tote, angle, s.body())
    s.stage("tilt_to_lid", close_lid)
    go(way["tilt_to_lid"], 1.5, grips={"left": tab_open})
    s.stage("reach_lid", close_lid)
    go(way["reach_lid"], 2.5)
    go(way["up_arc"], 2.0)
    s.stage("pinch_tab_again", close_lid)
    s.move(grips={"left": tab_shut}, seconds=1.0)
    s.move(seconds=0.5)
    check_pinch(s)
    s.stage("close_lid", close_lid)
    go(way["close_lid"], 6.0)
    if s.lid_angle() > kt.CLOSED:
        raise RuntimeError(f"the lid stayed {np.degrees(s.lid_angle()):.1f} deg open")
    s.stage("let_go", close_lid)
    s.move(grips={"left": tab_open}, seconds=1.0)
    s.move(seconds=0.3)
    s.stage("back_away", close_lid)
    go(way["back_away"], 1.5)
    # Home the way the arm came: back along the reach's line to the stage,
    # joint-space over READY and straight down.
    s.stage("finish", home_text)
    go(way["return"], 1.5)
    s.move(
        joints={"left": done["rise"][1][-1]},
        seconds=3.0,
        grips={"left": 1 - tt.HOME_OPENING / tt.JAW_SPAN},
        speed=tt.SPEED,
    )
    back("rise", 2.0)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        LidScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "final_lid_angle_deg": float(np.degrees(s.lid_angle())),
            "widest_lid_angle_deg": float(np.degrees(s.widest)),
        },
        engine=engine,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", action="store_true")
    args = p.parse_args()
    result = run(args.output, args.seed, args.record)
    print(
        json.dumps({k: result[k] for k in ("success", "error", "progress")}), flush=True
    )


if __name__ == "__main__":
    main()
