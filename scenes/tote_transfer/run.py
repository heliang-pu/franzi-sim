"""Tote transfer episode: the scene and a scripted IK expert.

ToteScene is sim.tabletop.ObjectScene with task.py's two loose parts: the
episode framework's audits and recordings, plus object tracking, the
milestones (the named part lifted out, then lying in the grey tote) and the
success test, which also needs the other part still in the blue tote. The
expert reads the part's pose from the simulator (privileged): no welds, no
object pose writes, no contact-mask changes, no robot qpos/qvel writes.

The left hand does it all in three subtasks while the right arm holds: pick
the part up, put it into the grey tote, move the arm back to its start pose.
From the default posture the hand turns half a turn in place at READY (one
smooth move along task.turn_path), rises to RISE, grasps the part top-down,
lifts it, carries it over the rims in joint space (task.transfer_goal),
lowers it until its measured lowest point is SET_GAP over the grey tote's
floor, sets it down to RELEASE and lets go. Home it retraces its way: up, in
joint space back over the blue tote, straight back to RISE, down onto the
turn's last joints, and back along the turn to the start pose's joints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from scenes.tote_transfer import task as kt
from sim import tabletop as tt
from sim.episode import minimum_jerk


class ToteScene(tt.ObjectScene):
    OBJECTS = tuple(kt.PARTS)
    MILESTONES = ("lifted_out", "in_grey_tote")
    # Carried through the air: a held part touching anything but the robot
    # there is a collision. The lift starts on the blue tote's floor and the
    # set-down ends just over the grey one's, by design.
    CARRY_PHASES = ("transfer", "lower")

    def __init__(self, output, spec, record=False):
        self.HELD = (spec["target"],)
        super().__init__(output, spec, record)
        m = self.m

        def vertices(g):
            start = m.mesh_vertadr[m.geom_dataid[g]]
            return m.mesh_vert[start : start + m.mesh_vertnum[m.geom_dataid[g]]].copy()

        # Each part's collision pieces with their mesh vertices (geom frame).
        self.pieces = {
            name: [
                (g, vertices(g))
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and m.geom_contype[g] | m.geom_conaffinity[g]
            ]
            for name, bid in self.ids.items()
        }

    def build_root(self):
        return kt.build(self.spec)

    def lowest(self, name):
        """The height (m) of the part's lowest collision vertex."""
        d = self.d
        return min(
            float((d.geom_xpos[g][2] + v @ d.geom_xmat[g].reshape(3, 3)[2]).min())
            for g, v in self.pieces[name]
        )

    def follow(self, side, waypoints, grips=None, seconds=0.0, speed=None):
        """Move side's arm through joint waypoints (from its goal) in one
        minimum-jerk move. The path is the waypoints' shape-preserving cubic
        (PCHIP, which stays within each joint's waypoint values, so within its
        limits) over their joint travel, sampled densely: straight segments
        between the waypoints turned corners into accelerations that
        stretch() answered with a 14 s turn. Timed as Scene.move times a
        straight line's IK waypoints: by joint travel at speed if given, and
        stretched to the controller's rates. The other arm holds its goal and
        gripper."""
        points = [np.asarray(self.goal[side], float)]
        for q in waypoints:
            if np.abs(np.subtract(q, points[-1])).max() > 1e-9:
                points.append(np.asarray(q, float))
        points = np.array(points)
        arc = np.r_[0, np.cumsum(np.abs(np.diff(points, axis=0)).max(1))]
        if len(points) > 2:
            u = np.linspace(0, arc[-1], 20 * len(points) + 1)
            points = PchipInterpolator(arc, points, axis=0)(u)
        paths = {s: np.array([self.goal[s], self.goal[s]]) for s in self.goal}
        paths[side] = points
        arcs = {}
        if speed:
            arcs = {
                s: np.r_[0, np.cumsum(np.abs(np.diff(p, axis=0)).max(1))]
                for s, p in paths.items()
            }
            seconds = max(seconds, 1.875 * arcs[side][-1] / speed)
        seconds = self.stretch(paths, arcs, seconds)
        gs = self.grip.copy()
        ge = gs | (grips or {})
        n = int(seconds / self.m.opt.timestep)
        for i in range(n):
            p = minimum_jerk((i + 1) / n)
            if self.step_number % 20 == 0:
                for s, path in paths.items():
                    self.goal[s] = self.along(path, arcs.get(s), p)
                    self.grip[s] = gs[s] + (ge[s] - gs[s]) * p
            self.physics_step()

    def in_tote(self, name, tote, pairs):
        """Lying flat on the tote's floor, inside its walls, still."""
        x, y, z = self.xyz(name)
        return (
            abs(x - tote[0]) < kt.INNER[0]
            and abs(y - tote[1]) < kt.INNER[1]
            and abs(z - kt.FLOOR_Z) < 0.003
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        target = self.spec["target"]
        held = self.touching(target, pairs).issuperset(self.jaws["left"])
        return {
            "lifted_out": held and self.xyz(target)[2] > kt.RIM_Z,
            "in_grey_tote": self.in_tote(target, self.spec["totes"]["dest"], pairs),
        }

    def complete(self):
        pairs = self.contact_bodies()
        other = next(n for n in kt.PARTS if n != self.spec["target"])
        return self.status(pairs)["in_grey_tote"] and self.in_tote(
            other, self.spec["totes"]["source"], pairs
        )


def down_to(s, name, gap, xy):
    """The grasp-site target over xy that brings the held part's measured
    lowest point gap over the grey tote's floor."""
    site = s.d.site_xpos[s.sites["left"]]
    drop = s.lowest(name) - (kt.FLOOR_Z + gap)
    return [*xy, float(site[2] - drop)]


def expert(s):
    spec = s.spec
    target = spec["target"]
    part = kt.PARTS[target][3]
    width = kt.PARTS[target][2]
    turned = kt.hand(spec["hand_yaw"])

    def line(xyz, rotation, seconds=2.0, **kw):
        # The sampler planned these at its wider margin; a measured pose that
        # makes the IK jump branch stops the episode instead of swinging the arm.
        path = s.kin.line("left", s.goal["left"], xyz, s.body(), rotation=rotation)
        if len(path) > 1 and np.abs(np.diff(path, axis=0)).max() > kt.JUMP:
            raise RuntimeError(f"the line to {np.round(xyz, 3).tolist()} jumps")
        s.move(
            {"left": xyz},
            seconds=seconds,
            linear=True,
            rotations={"left": rotation},
            speed=tt.SPEED,
            **kw,
        )

    # One subtask per action of the left arm (user, 2026-10-25): its text is
    # the trace's subtask, the stage name the finer phase; the right arm holds.
    tt.start_home(s)
    start = s.goal["left"].copy()  # the start pose's joints
    turn = kt.turn_path(s.kin, start, spec["hand_yaw"])
    pick = f"Pick up the {part} from the blue tote with the left hand."
    # The turn runs at the controller's rate limits alone (7 s), not tt.SPEED.
    s.stage("turn", pick)
    s.follow("left", turn, grips={"left": tt.grip_open(width)}, seconds=2.0)
    s.stage("approach", pick)
    line(kt.RISE, turned, 1.5)
    rise = s.goal["left"].copy()
    base = [float(v) for v in s.xyz(target)]
    poses = kt.key_poses(base, spec["hand_yaw"], width, spec["drop"])
    rotation = poses["rotation"]
    line(poses["above"], rotation, 2.5)
    s.stage("align", pick)
    line(poses["grasp"], rotation, 2.0)
    s.stage("grasp", pick)
    s.move(grips={"left": tt.grip_closed(width)}, seconds=1.0)
    s.move(seconds=0.5)
    tt.check_grasp(s, target, "left")
    s.stage("lift", pick)
    start_z = float(s.xyz(target)[2])
    line(poses["lift"], rotation, 2.0)
    if s.xyz(target)[2] < start_z + kt.LIFT_MIN:
        raise RuntimeError(f"lift failed for the {target}")
    lifted = s.goal["left"].copy()
    place = f"Put the {part} into the grey tote with the left hand."
    s.stage("transfer", place)
    over = kt.transfer_goal(s.kin, s.goal["left"], poses)
    kt.check_transfer(s.kin, s.goal["left"], over)
    s.move(joints={"left": over}, seconds=2.0, speed=tt.SPEED)
    # The part's lowest point, measured (privileged): it creeps down the jaws
    # under MuJoCo and pivots about the pads' line under PhysX.
    s.stage("lower", place)
    line(down_to(s, target, kt.SET_GAP, poses["over"][:2]), rotation, 2.0)
    s.stage("set", place)
    line(down_to(s, target, kt.RELEASE, poses["over"][:2]), rotation, 0.8)
    print(
        f"  let go: lowest point {1000 * (s.lowest(target) - kt.FLOOR_Z):.2f} mm"
        " over the floor",
        flush=True,
    )
    s.stage("release", place)
    s.move(grips={"left": kt.release_open(width)}, seconds=0.8)
    s.move(seconds=0.4)
    back = "Move the left arm back to its start pose."
    s.stage("retreat", back)
    line(poses["clear"], rotation, 1.5)
    # Home the way the part came: back over the part, back to RISE, then down
    # onto the turn's last joints.
    s.stage("return", back)
    kt.check_transfer(s.kin, s.goal["left"], lifted)
    s.move(joints={"left": lifted}, seconds=2.0, speed=tt.SPEED)
    line(kt.RISE, turned, 2.5)
    if np.abs(s.goal["left"] - rise).max() > kt.RETURN_SLACK:
        raise RuntimeError("the way back to RISE ended on another branch")
    s.move(joints={"left": turn[-1]}, seconds=1.5, speed=tt.SPEED)
    s.stage("turn", back)
    home = 1 - tt.HOME_OPENING / tt.JAW_SPAN
    s.follow("left", [*turn[-2::-1], start], grips={"left": home}, seconds=2.0)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        ToteScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {"target": s.spec["target"]},
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
