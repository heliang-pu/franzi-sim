"""CNC tending episode: the scene and a scripted IK expert.

CNCScene is sim.tabletop.ObjectScene with task.py's parts: the finished part in
the vise and the three blanks in the tray's near row. Its milestones follow the
task (door open, finished part out of the vise, on the hand-over stand, in its
far-row pocket; a blank on the stand, in the vise; door closed; CYCLE START
pressed after that) and the success test needs the finished part in its
pocket, a blank in the vise, the door shut, CYCLE START pressed and back up and
the other blanks still in their pockets.

The expert works one arm at a time (user rule 2026-10-25: an episode splits
into single-arm subtasks; the other arm holds its goals and grip): the left
hand slides the door open by its handle, takes the finished part out of the
vise and sets it on the hand-over stand; the right hand puts it into the named
far-row pocket and brings a blank from the near row to the stand; the left
hand loads the blank into the vise and slides the door shut; the right hand
presses CYCLE START along +x with its closed fingers, a closed loop on the
cap's travel; then each arm returns to its start pose. Each subtask's text is
the trace's subtask, its fine stages the phase. Hands turn only in place,
through IK waypoints (task.turn_path), or in joint space in free space. The
expert reads part, door and button states from the simulator (privileged): no
welds, no object pose writes, no contact-mask changes, no robot qpos/qvel
writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.cnc_tending import task as kt
from sim import tabletop as tt
from sim.episode import minimum_jerk
from sim.gripper_linkage import contact_drop

BLANKS = tuple(f"blank{k}" for k in range(len(kt.NEAR_ROW)))


class CNCScene(tt.ObjectScene):
    OBJECTS = ("finished", *BLANKS)
    MILESTONES = (
        "door_open",
        "part_out",
        "part_on_stand",
        "part_in_tray",
        "blank_on_stand",
        "blank_in_vise",
        "door_closed",
        "cycle_started",
    )
    # Phases in which a held part must touch nothing but the robot: every
    # carry and turn, and each set-down until the part goes onto its support
    # (set_*) or into its pocket or between the jaws (insert_*).
    CARRY_PHASES = (
        "withdraw",
        "turn_part",
        "carry_to_stand",
        "lower_part_on_stand",
        "carry_to_tray",
        "turn_part_over",
        "over_tray",
        "lower_part_in_tray",
        "settle_part_in_tray",
        "centre_part_in_tray",
        "carry_blank",
        "turn_blank",
        "pass_panel",
        "lower_blank_on_stand",
        "turn_blank_over",
        "carry_to_vise",
        "into_vise",
        "lower_blank_in_vise",
        "settle_blank_in_vise",
        "centre_blank_in_vise",
    )

    def __init__(self, output, spec, record=False):
        self.lay = kt.Layout(spec)  # where the fixtures stand this episode
        self.HELD = ("finished", spec["blank"])
        self.deepest_press = 0.0
        self.gaps = {}  # per set-down: the part's lowest point over its support
        super().__init__(output, spec, record)
        self.door_adr = int(self.m.joint("cnc_door").qposadr[0])
        self.door_body = self.m.body("cnc_door_part").id
        self.button_adr = int(self.m.joint("cnc_cycle_start").qposadr[0])

    def build_root(self):
        return kt.build(self.spec)

    def door(self):
        """How far the door is open (m)."""
        return float(self.d.qpos[self.door_adr])

    def button(self):
        """CYCLE START's travel (m, pressed along +x)."""
        return float(self.d.qpos[self.button_adr])

    def in_pocket(self, name, xy, pairs):
        x, y, z = self.xyz(name)
        return (
            np.hypot(x - xy[0], y - xy[1]) < 0.006
            and abs(z - kt.POCKET_FLOOR) < 0.003
            and self.settled(name, pairs)
        )

    def on_stand(self, name, pairs):
        x, y, z = self.xyz(name)
        return (
            np.hypot(x - self.lay.handover[0], y - self.lay.handover[1]) < 0.02
            and abs(z - self.lay.handover[2]) < 0.003
            and self.settled(name, pairs)
        )

    def in_vise(self, name, pairs):
        """On the parallels between the open jaws, where the finished part
        was, still."""
        x, y, z = self.xyz(name)
        vx, vy = self.spec["vise"]
        return (
            abs(x - (vx + kt.LOAD_DX)) < 0.006
            and abs(y - vy) < 0.01
            and abs(z - kt.VISE_FLOOR) < 0.003
            and self.settled(name, pairs)
        )

    def conditions(self, pairs):
        blank = self.spec["blank"]
        held = self.touching("finished", pairs).issuperset(self.jaws["left"])
        now = {
            "door_open": self.door() >= kt.OPEN_MIN,
            "part_out": held and self.xyz("finished")[2] > kt.VISE_FLOOR + 0.03,
            "part_on_stand": self.on_stand("finished", pairs),
            "part_in_tray": self.in_pocket(
                "finished", self.lay.far_row[self.spec["drop"]], pairs
            ),
            "blank_on_stand": self.on_stand(blank, pairs),
            "blank_in_vise": self.in_vise(blank, pairs),
        }
        now["door_closed"] = self.door() <= kt.CLOSED and now["blank_in_vise"]
        now["cycle_started"] = (
            self.button() >= kt.PRESSED and now["door_closed"] and now["part_in_tray"]
        )
        return now

    def sample(self):
        record = super().sample()
        record["door_m"] = self.door()
        record["cycle_start_mm"] = 1000 * self.button()
        return record

    def track_objects(self):
        super().track_objects()
        self.deepest_press = max(self.deepest_press, self.button())

    def complete(self):
        """The final state: the finished part in its pocket, the blank in the
        vise, the door shut, CYCLE START pressed after that and back up, the
        other blanks untouched in their pockets."""
        pairs = self.contact_bodies()
        now = self.status(pairs)
        others = [n for n in BLANKS if n != self.spec["blank"]]
        return (
            now["part_in_tray"]
            and now["door_closed"]
            and self.reached["cycle_started"]
            and self.button() < kt.RELEASED
            and all(self.in_pocket(n, self.spec["blanks"][n], pairs) for n in others)
        )


def base(s, name):
    """A part's measured base (its body origin)."""
    return [float(v) for v in s.xyz(name)]


def check_handle(s):
    pairs = s.contact_bodies()
    others = set(pairs[(pairs == s.door_body).any(1)].ravel().tolist())
    if not set(s.jaws["left"]) <= others:
        raise RuntimeError("left: no bilateral grasp on the door handle")


def check_lift(s, name, start, rise):
    if s.xyz(name)[2] < start + rise:
        raise RuntimeError(f"lift failed for the {name}")


def follow(s, paths, seconds=1.5, grips=None, speed=tt.SPEED):
    """Run arm joint paths ({side: waypoints}; the other arm holds) the way
    Scene.move runs its own: timed by joint travel at speed, stretched to the
    controller's rate limits, minimum jerk; grips blend along."""
    paths = {k: np.asarray(v, float) for k, v in paths.items()}
    for k in s.goal:
        paths.setdefault(k, np.array([s.goal[k], s.goal[k]]))
    arcs = {
        k: np.r_[0, np.cumsum(np.abs(np.diff(p, axis=0)).max(1))]
        for k, p in paths.items()
    }
    seconds = max(seconds, 1.875 * max(a[-1] for a in arcs.values()) / speed)
    seconds = s.stretch(paths, arcs, seconds)
    start = s.grip.copy()
    end = start | (grips or {})
    n = int(seconds / s.m.opt.timestep)
    for i in range(n):
        p = minimum_jerk((i + 1) / n)
        if s.step_number % 20 == 0:
            for k, path in paths.items():
                s.goal[k] = s.along(path, arcs[k], p)
                s.grip[k] = start[k] + (end[k] - start[k]) * p
        s.physics_step()


class Hands:
    """The expert's motions: straight moves and in-place turns of the grasp
    sites, the hands' current rotations, grips and set-downs, and the stage
    labels: every fine stage of a subtask carries that subtask's text."""

    def __init__(self, s):
        self.s = s
        self.rot = {"left": np.eye(3), "right": np.eye(3)}
        self.text = ""

    def subtask(self, text):
        """Start a subtask: one arm's action on one object; the other arm
        holds its goals and grip throughout."""
        self.text = text

    def stage(self, name):
        self.s.stage(name, self.text)

    def here(self, side):
        """The grasp site the arm's goal puts the hand at."""
        return self.s.kin.fk(side, self.s.goal[side], self.s.body())[0]

    def line(self, targets, seconds=2.0, grips=None):
        """Straight moves of the grasp sites, hands keeping their rotations."""
        self.s.move(
            targets,
            seconds=seconds,
            linear=True,
            rotations={k: self.rot[k] for k in targets},
            speed=tt.SPEED,
            grips=grips,
        )

    def joints(self, side, q, seconds=2.5, grips=None):
        """A joint-space move of one arm (free space only)."""
        self.s.move(joints={side: q}, grips=grips, seconds=seconds, speed=tt.SPEED)

    def turn_at_ready(self, side, rotation):
        """The right arm's first turn, from the default posture (fingers
        down) at its ready point: one joint-space move in free space, onto the
        elbow swivel of task.SWIVEL_ROLL (turned in place through IK
        waypoints, the arm would keep its default swivel)."""
        seed = kt.swivel_seed(side, self.s.goal[side])
        self.joints(side, self.s.ik(side, kt.READY[side], near=seed, rotation=rotation))
        self.rot[side] = rotation

    def turn(self, rotations, targets=None, seconds=1.5, grips=None):
        """Turn hands in place through IK waypoints (task.turn_path): about the
        grasp site, or onto targets (the site that keeps a held part where it
        is)."""
        s, paths = self.s, {}
        for k, rotation in rotations.items():
            start = self.here(k)
            end = (targets or {}).get(k)
            end = start if end is None else end
            paths[k] = kt.turn_path(
                s.kin, k, s.goal[k], start, end, self.rot[k], rotation, s.body()
            )
        follow(s, paths, seconds, grips)
        self.rot.update(rotations)

    def turn_about(self, side, name, psi):
        """Turn the hand holding part name to psi about the point it holds
        (the closed pads' centre)."""
        width = kt.part_grip(name)[1]
        pads = self.here(side) - contact_drop(width) * (self.rot[side] @ [0, 0, 1])
        self.turn({side: kt.hand(side, psi)}, {side: kt.site(side, pads, width, psi)})

    def grip(self, side, closure, seconds=1.0):
        self.s.move(grips={side: closure}, seconds=seconds)
        self.s.move(seconds=0.5)

    def holding(self, side, name, base_at):
        """The grasp site that puts part name, as the hand holds it now
        (measured: privileged), with its base at base_at."""
        offset = self.here(side) - np.asarray(self.s.xyz(name), float)
        return np.asarray(base_at, float) + offset

    def lowest(self, name):
        """Height (m) of part name's lowest point: its bottom rim, which a
        part hanging off vertical in the grip carries below its base."""
        bid = self.s.ids[name]
        cos = float(np.clip(self.s.d.xmat[bid][8], -1.0, 1.0))
        return float(self.s.xyz(name)[2]) - kt.PART_R * np.sqrt(1.0 - cos * cos)

    def seat(self, side, name, where, support):
        """Bring part name's lowest point to within GAP_MAX of its support
        (support, m; measured: privileged), with no move when it is within
        already, then let go (release_<where>). Logs both gaps."""
        gap = self.lowest(name) - support
        log = self.s.gaps[where] = {"set_mm": round(1000 * gap, 3)}
        if gap > kt.GAP_MAX:
            self.stage(f"seat_{where}")
            self.line({side: self.here(side) - [0, 0, gap - kt.GAP_MAX]}, 0.5)
        log["release_mm"] = round(1000 * (self.lowest(name) - support), 3)
        self.stage(f"release_{where}")
        self.grip(side, tt.grip_open(kt.part_grip(name)[1]))

    def set_down(self, side, name, base_at, where, support):
        """Lower part name straight down onto a flat support until its base
        is at base_at (SET_GAP over it): lower_<where>, a carry phase, to
        SETTLE above, set_<where> the rest (the part may touch its support
        there), then seat() it and let go."""
        target = self.holding(side, name, base_at)
        self.stage(f"lower_{where}")
        self.line({side: target + [0, 0, kt.SETTLE]}, 1.5)
        self.stage(f"set_{where}")
        self.line({side: target}, 1.0)
        self.seat(side, name, where, support)

    def insert(self, side, name, centre, rim, support, where):
        """Lower part name into a pocket or between the vise jaws at centre
        (x, y): lower_<where> (carried) until its base hovers HOVER over the
        rim, settle_<where> for SETTLE_S, centre_<where> on its measured base
        (carried), insert_<where> slowly to SET_GAP over the support (the walls
        may guide it), then seat() it and let go."""
        hover = [*centre, rim + kt.HOVER]
        self.stage(f"lower_{where}")
        self.line({side: self.holding(side, name, hover)}, 1.5)
        self.stage(f"settle_{where}")
        self.s.move(seconds=kt.SETTLE_S)
        self.stage(f"centre_{where}")
        self.line({side: self.holding(side, name, hover)}, 0.5)
        self.stage(f"insert_{where}")
        target = self.holding(side, name, [*centre, support + tt.SET_GAP])
        self.line({side: target}, kt.INSERT_S)
        self.seat(side, name, where, support)


def open_door(s, h):
    """The left hand turns toward the handle (its first subtask), slides the
    door open by it and backs away behind the bar."""
    h.subtask("Open the machine door with the left hand.")
    h.stage("turn_hand")
    # In place through IK waypoints: turned in joint space, the hand swung a
    # finger 12 mm from the closed door, into it with the machine 2 cm closer
    # (a hand may touch the door, so the audit passed it, but the finger was
    # pushed past its joint limit).
    h.turn({"left": kt.hand("left", kt.HANDLE_PSI)})
    opening = kt.door_poses(s.lay, s.door(), kt.OPEN)
    h.stage("approach_handle")
    h.line(
        {"left": opening["pregrasp"]}, 2.5, grips={"left": tt.grip_open(kt.HANDLE_D)}
    )
    h.stage("align_handle")
    h.line({"left": opening["grasp"]}, 1.5)
    h.stage("grasp_handle")
    h.grip("left", tt.grip_closed(kt.HANDLE_D, kt.BAR_SQUEEZE))
    check_handle(s)
    h.stage("open_door")
    h.line({"left": opening["slid"]}, 4.0)
    if s.door() < kt.OPEN_MIN:
        raise RuntimeError(f"the door opened only {s.door():.3f} m")
    h.stage("release_handle")
    h.grip("left", tt.grip_open(kt.HANDLE_D))
    h.stage("back_off")
    h.line({"left": opening["retreat"]}, 1.5)
    h.line({"left": opening["clear"]}, 1.5)


def unload(s, h):
    """The left hand takes the finished part out of the vise and the
    machine."""
    h.subtask("Take the finished part out of the vise with the left hand.")
    h.stage("point_in")
    h.turn({"left": kt.hand("left")})
    take = kt.vise_take(s.lay, base(s, "finished"))
    h.stage("approach_part")
    h.line(
        {"left": [s.lay.door_clear_x, take["grasp"][1], take["pregrasp"][2]]},
        2.0,
        grips={"left": tt.grip_open(kt.FINISHED_W)},
    )
    h.line({"left": take["pregrasp"]}, 2.5)
    h.stage("align_part")
    h.line({"left": take["grasp"]}, 1.5)
    h.stage("grasp_part")
    h.grip("left", tt.grip_closed(kt.FINISHED_W, kt.PART_SQUEEZE))
    tt.check_grasp(s, "finished", "left")
    h.stage("lift_part")
    start = float(s.xyz("finished")[2])
    h.line({"left": take["lift"]}, 1.5)
    check_lift(s, "finished", start, 0.6 * kt.VISE_LIFT)
    h.stage("withdraw")
    h.line({"left": take["out"]}, 2.5)


def part_to_stand(s, h):
    """The left hand sets the finished part on the hand-over stand and moves
    aside."""
    h.subtask("Put the finished part on the hand-over stand with the left hand.")
    h.stage("turn_part")
    h.turn_about("left", "finished", kt.STAND_PSI["left"])
    h.stage("carry_to_stand")
    stand = s.lay.stand_set
    above = h.holding("left", "finished", stand) + [0, 0, kt.STAND_LIFT]
    h.line({"left": above}, 2.5)
    h.set_down("left", "finished", stand, "part_on_stand", s.lay.handover[2])
    h.stage("clear_stand")
    h.line({"left": above}, 1.5)
    h.stage("move_aside")
    h.line({"left": kt.WAIT["left"]}, 2.0)


def take_over(s, h):
    """The right hand turns toward the stand (its first subtask) and picks
    the finished part up."""
    rs = kt.STAND_PSI["right"]
    h.subtask("Pick up the finished part from the hand-over stand with the right hand.")
    h.stage("turn_hand_2")
    h.turn_at_ready("right", kt.hand("right", rs))
    get = kt.pick_poses("right", base(s, "finished"), "finished", rs, kt.STAND_LIFT)
    h.stage("reach_stand")
    via = [*s.lay.stand_via, get["above"][2]]
    h.line({"right": via}, 2.0, grips={"right": tt.grip_open(kt.FINISHED_W)})
    h.line({"right": get["above"]}, 1.5)
    h.stage("align_take")
    h.line({"right": get["grasp"]}, 2.0)
    h.stage("grasp_take")
    h.grip("right", tt.grip_closed(kt.FINISHED_W, kt.PART_SQUEEZE))
    tt.check_grasp(s, "finished", "right")
    h.stage("lift_from_stand")
    start = float(s.xyz("finished")[2])
    h.line({"right": get["above"]}, 2.0)
    check_lift(s, "finished", start, 0.6 * kt.STAND_LIFT)


def to_tray(s, h, pocket):
    """The right hand carries the finished part past the panel into its
    far-row pocket."""
    h.subtask(
        f"Put the finished part in the tray's {pocket} back pocket with the right hand."
    )
    h.stage("carry_to_tray")
    for xy in s.lay.route:
        h.line({"right": h.holding("right", "finished", [*xy, kt.CARRY_Z])}, 1.5)
    h.stage("turn_part_over")
    h.turn_about("right", "finished", s.lay.tray_yaw)
    pocket_xy = s.lay.far_row[s.spec["drop"]]
    above = h.holding("right", "finished", [*pocket_xy, kt.CARRY_Z])
    h.stage("over_tray")
    h.line({"right": above}, 2.0)
    h.insert(
        "right", "finished", pocket_xy, kt.TRAY_RIM, kt.POCKET_FLOOR, "part_in_tray"
    )
    h.stage("rise_from_tray")
    h.line({"right": above}, 1.5)


def pick_blank(s, h, blank):
    """The right hand takes the blank out of its near-row pocket."""
    where = kt.POCKETS[int(blank[-1])]
    h.subtask(f"Take the {where} raw blank out of the tray with the right hand.")
    pick = kt.pick_poses(
        "right", base(s, blank), "blank", s.lay.tray_yaw, kt.CARRY_Z - kt.POCKET_FLOOR
    )
    h.stage("reach_blank")
    h.line({"right": pick["above"]}, 2.0, grips={"right": tt.grip_open(kt.BLANK_W)})
    h.stage("align_blank")
    h.line({"right": pick["grasp"]}, 2.0)
    h.stage("grasp_blank")
    h.grip("right", tt.grip_closed(kt.BLANK_W, kt.PART_SQUEEZE))
    tt.check_grasp(s, blank, "right")
    h.stage("lift_blank")
    start = float(s.xyz(blank)[2])
    h.line({"right": pick["above"]}, 2.0)
    check_lift(s, blank, start, 0.06)


def blank_to_stand(s, h, blank):
    """The right hand carries the blank past the panel onto the stand and
    backs away past the panel."""
    rs = kt.STAND_PSI["right"]
    h.subtask("Put the raw blank on the hand-over stand with the right hand.")
    route, stand = s.lay.route, s.lay.stand_set
    h.stage("carry_blank")
    h.line({"right": h.holding("right", blank, [*route[-1], kt.CARRY_Z])}, 2.0)
    h.stage("turn_blank")
    h.turn_about("right", blank, rs)
    h.stage("pass_panel")
    for xy in route[-2::-1]:
        h.line({"right": h.holding("right", blank, [*xy, kt.CARRY_Z])}, 1.5)
    above = h.holding("right", blank, stand) + [0, 0, kt.STAND_LIFT]
    h.line({"right": above}, 1.5)
    h.set_down("right", blank, stand, "blank_on_stand", s.lay.handover[2])
    h.stage("clear_stand_2")
    h.line({"right": above}, 1.5)
    h.stage("leave_stand")
    for xy in route:
        h.line({"right": kt.carry_site("right", xy, "blank", rs)}, 1.5)


def take_blank(s, h, blank):
    """The left hand picks the blank up from the stand."""
    h.subtask("Pick up the raw blank from the hand-over stand with the left hand.")
    take = kt.pick_poses(
        "left", base(s, blank), "blank", kt.STAND_PSI["left"], kt.STAND_LIFT
    )
    h.stage("reach_blank_2")
    h.line({"left": take["above"]}, 2.5, grips={"left": tt.grip_open(kt.BLANK_W)})
    h.stage("align_blank_2")
    h.line({"left": take["grasp"]}, 2.0)
    h.stage("grasp_blank_2")
    h.grip("left", tt.grip_closed(kt.BLANK_W, kt.PART_SQUEEZE))
    tt.check_grasp(s, blank, "left")
    h.stage("lift_blank_2")
    start = float(s.xyz(blank)[2])
    h.line({"left": take["above"]}, 2.0)
    check_lift(s, blank, start, 0.6 * kt.STAND_LIFT)


def load(s, h, blank):
    """The left hand loads the blank into the vise and backs out of the
    machine behind the door's handle bar."""
    h.subtask("Load the raw blank into the vise with the left hand.")
    vise = s.spec["vise"]
    # Out to the machine's opening first: pointing ahead over the stand, the
    # wrist entered the torso (1 mm).
    h.stage("carry_to_vise")
    entry = [s.lay.out_x, vise[1], kt.CARRY_Z]
    h.line({"left": h.holding("left", blank, entry)}, 2.0)
    h.stage("turn_blank_over")
    h.turn_about("left", blank, 0.0)
    seat_xy = (vise[0] + kt.LOAD_DX, vise[1])
    h.stage("into_vise")
    h.line({"left": h.holding("left", blank, [*seat_xy, kt.CARRY_Z])}, 2.0)
    h.insert("left", blank, seat_xy, kt.JAW_TOP, kt.VISE_FLOOR, "blank_in_vise")
    h.stage("leave_vise")
    out = h.here("left") + [0, 0, 0.06]
    h.line({"left": out}, 1.5)
    h.line({"left": [s.lay.door_clear_x, out[1], out[2]]}, 2.0)


def close_door(s, h):
    """The left hand slides the door shut by its handle and backs away."""
    h.subtask("Close the machine door with the left hand.")
    h.stage("turn_to_handle")
    h.turn({"left": kt.hand("left", kt.HANDLE_PSI)})
    closing = kt.door_poses(s.lay, s.door(), kt.SHUT)
    h.stage("approach_handle_2")
    h.line({"left": closing["behind"]}, 2.5, grips={"left": tt.grip_open(kt.HANDLE_D)})
    h.line({"left": closing["pregrasp"]}, 1.5)
    h.stage("align_handle_2")
    h.line({"left": closing["grasp"]}, 1.5)
    h.stage("grasp_handle_2")
    h.grip("left", tt.grip_closed(kt.HANDLE_D, kt.BAR_SQUEEZE))
    check_handle(s)
    h.stage("close_door")
    h.line({"left": closing["slid"]}, 4.0)
    if s.door() > kt.CLOSED:
        raise RuntimeError(f"the door stayed {s.door():.3f} m open")
    h.stage("release_handle_2")
    h.grip("left", tt.grip_open(kt.HANDLE_D))
    h.line({"left": closing["retreat"]}, 1.5)


def press_button(s, h):
    """The right hand comes to the panel and presses CYCLE START along +x with
    its closed fingers: from 4 mm short of the cap (PRESS_TOUCH) 1 mm deeper
    at a time until the cap's travel (read from the simulator: privileged)
    reaches PRESS_DEPTH. tt.press presses downward; this button faces the
    robot. Raises RuntimeError if it never went down PRESSED."""
    poses = kt.press_poses(s.lay)
    h.subtask("Press CYCLE START with the right hand.")
    h.stage("turn_to_panel")
    h.turn({"right": poses["rotation"]})
    h.stage("reach_panel")
    h.line({"right": poses["back"]}, 2.5, grips={"right": kt.PRESS_GRIP})
    h.stage("press")
    h.line({"right": poses["touch"]}, 1.0)
    goal, deepest = np.array(poses["touch"], float), s.button()
    steps = round((kt.PRESS_DEPTH - kt.PRESS_TOUCH) / kt.PRESS_STEP) + 6
    for _ in range(steps):
        if s.button() >= kt.PRESS_DEPTH:
            break
        goal = goal + [kt.PRESS_STEP, 0, 0]
        h.line({"right": goal}, 0.3)
        deepest = max(deepest, s.button())
    s.move(seconds=0.4)
    if max(deepest, s.button()) < kt.PRESSED:
        raise RuntimeError("CYCLE START did not go down")
    h.stage("release_button")
    h.line({"right": poses["back"]}, 1.0)


def go_home(s, h, side, home, grip):
    """One arm back to the joints the episode started with: a straight move
    to HOME_VIA beside the robot (pointing ahead near the midline, a wrist
    enters the torso; swung from the wait pose, the left hand hit the shut
    door), then in joint space (turned through IK waypoints at the wait
    point, an arm took 20 s)."""
    h.subtask(f"Move the {side} arm back to its start pose.")
    h.stage(f"return_{side}")
    h.line({side: kt.HOME_VIA[side]}, 2.0)
    h.joints(side, home[side], 3.0, grips={side: grip[side]})


def expert(s):
    blank = s.spec["blank"]
    pocket = kt.POCKETS[s.spec["drop"]]
    h = Hands(s)
    tt.start_home(s)
    home = {k: s.goal[k].copy() for k in s.goal}
    grip = s.grip.copy()
    # One arm at a time: each subtask is one arm's action on one object.
    open_door(s, h)
    unload(s, h)
    part_to_stand(s, h)
    take_over(s, h)
    to_tray(s, h, pocket)
    pick_blank(s, h, blank)
    blank_to_stand(s, h, blank)
    take_blank(s, h, blank)
    load(s, h, blank)
    close_door(s, h)
    press_button(s, h)
    go_home(s, h, "left", home, grip)
    go_home(s, h, "right", home, grip)
    s.move(seconds=1.0)


def run(output, seed, record=False, kin=None, engine=None):
    """One expert episode for seed into output (must not exist). engine runs it
    on another simulator (isaac.episode.isaac_scene for Isaac Sim)."""
    spec = kt.sample_episode(seed, kin)
    return tt.run_episode(
        CNCScene,
        expert,
        spec,
        output,
        record,
        extra=lambda s: {
            "blank": s.spec["blank"],
            "pocket": kt.POCKETS[s.spec["drop"]],
            "deepest_press_mm": 1000 * s.deepest_press,
            "set_down_gaps": s.gaps,
            "final_door_m": s.door(),
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
