"""The SLA wrapper's parts that need no openpi: the shared kinematics' target
following, the hand poses and the coarse prompts."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("pinocchio")

from sla.coarse import EARLIER, LONG_SCENES, TABLES, coarsen, steps
from sla.eef import OPEN, Hands, model_state, relative, rotvec_state
from sla.eef import hands as thread_hands


@pytest.fixture(scope="module")
def hands():
    return Hands()


def row(kin, left, right, grips=(1000.0, 300.0)):
    return np.r_[kin.to_sdk("left", left), kin.to_sdk("right", right), grips]


def posture(kin, rng):
    return {
        side: rng.uniform(kin.limits[side][:, 0] + 0.2, kin.limits[side][:, 1] - 0.2)
        for side in ("left", "right")
    }


def test_origin_midway_between_the_arm_bases(hands):
    assert hands.origin[1] == pytest.approx(0, abs=1e-4)
    assert 0.8 < hands.origin[2] < 1.6


def test_pose_is_the_grasp_frame_less_the_origin(hands):
    kin = hands.kin
    q = posture(kin, np.random.default_rng(0))
    pose = hands.pose(row(kin, q["left"], q["right"]))
    assert pose.shape == (14,)
    for i, side in enumerate(("left", "right")):
        p, R = kin.fk(side, q[side])
        hand = pose[7 * i : 7 * i + 7]
        np.testing.assert_allclose(hand[:3], p - hands.origin)
        np.testing.assert_allclose(
            Rotation.from_rotvec(hand[3:6]).as_matrix(), R, atol=1e-9
        )
    assert pose[6] == 1.0 and pose[13] == 0.3


def test_poses_solve_back_into_joints_along_a_chunk(hands):
    kin = hands.kin
    rng = np.random.default_rng(1)
    q = posture(kin, rng)
    start = row(kin, q["left"], q["right"])
    # A short path from the start: each row a small move of every joint.
    path = np.cumsum(rng.normal(0, 0.003, (20, 16)), axis=0)
    path[:, 14:] = 0
    rows = start + path
    rows[:, :14] = np.clip(
        rows[:, :14],
        np.r_[kin.limits["left"][:, 0], kin.limits["right"][:, 0]] + 0.05,
        np.r_[kin.limits["left"][:, 1], kin.limits["right"][:, 1]] - 0.05,
    )
    poses = hands.pose(rows)
    solved, errors = hands.joints(poses, start)
    for side in ("left", "right"):
        assert errors[side][0] < 1e-3 and errors[side][1] < 3e-3
    np.testing.assert_allclose(hands.pose(solved)[:, :3], poses[:, :3], atol=1e-3)
    np.testing.assert_allclose(solved[:, 14:], rows[:, 14:])
    # The arms stay on their branch: no jump between rows.
    assert np.abs(np.diff(solved[:, :14], axis=0)).max() < 0.1


def test_a_target_out_of_reach_gets_the_nearest_pose_inside_the_limits(hands):
    kin = hands.kin
    q = kin.ik("left", [0.3, 0.3, 1.0], np.zeros(7))
    goal, error, _ = kin.follow("left", [3.0, 3.0, 3.0], np.eye(3), q)
    limits = kin.limits["left"]
    assert error > 1.0
    assert (goal >= limits[:, 0]).all() and (goal <= limits[:, 1]).all()


def test_follow_agrees_with_ik_on_a_reachable_target(hands):
    kin = hands.kin
    q = kin.ik("left", [0.3, 0.3, 1.0], np.zeros(7))
    goal, error, rotation = kin.follow("left", [0.31, 0.3, 1.0], np.eye(3), q)
    assert error < 2e-4 and rotation < 3e-3
    np.testing.assert_allclose(kin.fk("left", goal)[0], [0.31, 0.3, 1.0], atol=2e-4)


def test_relative_matches_slas_definition():
    rng = np.random.default_rng(2)
    state = rng.normal(0, 0.5, 14)
    poses = rng.normal(0, 0.5, (5, 14))
    delta = relative(state, poses)
    for i in (0, 7):
        np.testing.assert_allclose(
            delta[:, i : i + 3], poses[:, i : i + 3] - state[i : i + 3]
        )
        back = Rotation.from_rotvec(delta[:, i + 3 : i + 6]) * Rotation.from_rotvec(
            state[i + 3 : i + 6]
        )
        np.testing.assert_allclose(
            back.as_matrix(),
            Rotation.from_rotvec(poses[:, i + 3 : i + 6]).as_matrix(),
            atol=1e-9,
        )
    np.testing.assert_array_equal(delta[:, [6, 13]], poses[:, [6, 13]])


def test_gripper_scale_is_the_datasets():
    assert OPEN == 1000.0


def test_coarse_merges_pick_put_and_the_way_back():
    fine = [
        "Pick up parcel C with the right hand.",
        "Put parcel C into bin C with the right hand.",
        "Pick up parcel D with the right hand.",
        "Put parcel D into bin D with the right hand.",
        "Move the right arm back to its start pose.",
        "Pick up parcel A with the left hand.",
        "Put parcel A into bin A with the left hand.",
    ]
    assert steps(coarsen("parcel_sorting", fine)) == [
        "Put parcel C into bin C with the right hand.",
        "Put parcel D into bin D with the right hand.",
        "Put parcel A into bin A with the left hand.",
    ]


def test_coarse_takes_a_cartons_slot_from_the_next_text():
    fine = [
        "Pick up the right outer carton with the right hand.",
        "Stack the right outer carton in slot 1 with the right hand.",
        "Move the right arm back to its ready pose.",
        "Pick up the right inner carton with the right hand.",
        "Stack the right inner carton in slot 2 with the right hand.",
    ]
    assert steps(coarsen("palletizing", fine)) == [
        "Stack the right outer carton in slot 1 with the right hand.",
        "Stack the right inner carton in slot 2 with the right hand.",
    ]


def test_coarse_cnc_tending_keeps_seven_steps():
    fine = [
        "Open the machine door with the left hand.",
        "Take the finished part out of the vise with the left hand.",
        "Put the finished part on the hand-over stand with the left hand.",
        "Pick up the finished part from the hand-over stand with the right hand.",
        "Put the finished part in the tray's middle back pocket with the right hand.",
        "Take the right raw blank out of the tray with the right hand.",
        "Put the raw blank on the hand-over stand with the right hand.",
        "Pick up the raw blank from the hand-over stand with the left hand.",
        "Load the raw blank into the vise with the left hand.",
        "Close the machine door with the left hand.",
        "Press CYCLE START with the right hand.",
        "Move the left arm back to its start pose.",
        "Move the right arm back to its start pose.",
    ]
    assert steps(coarsen("cnc_tending", fine)) == [
        "Open the machine door with the left hand.",
        "Put the finished part on the hand-over stand with the left hand.",
        "Put the finished part in the tray's middle back pocket with the right hand.",
        "Put the right raw blank on the hand-over stand with the right hand.",
        "Load the raw blank into the vise with the left hand.",
        "Close the machine door with the left hand.",
        "Press CYCLE START with the right hand.",
    ]


def test_coarse_refuses_an_unknown_text_and_a_lone_marker():
    with pytest.raises(KeyError):
        coarsen("lidded_tote", ["Dance with the left hand."])
    with pytest.raises(ValueError):
        coarsen("lidded_tote", ["Move the left arm back to its start pose."])


def test_every_long_scene_has_a_table_with_real_steps():
    for scene in LONG_SCENES:
        assert any(v != EARLIER and "<" not in v for v in TABLES[scene].values())


def test_model_state_turns_back_into_the_hand_row_even_near_pi():
    rng = np.random.default_rng(4)
    state = rng.normal(0, 0.3, (64, 14))
    axis = rng.normal(size=(64, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    state[:, 3:6] = axis * rng.uniform(3.0, np.pi, (64, 1))  # near the wrap
    model = model_state(state)
    assert model.shape == (64, 20)
    back = rotvec_state(model + rng.normal(0, 1e-7, model.shape))
    np.testing.assert_allclose(back[:, :3], state[:, :3], atol=1e-6)
    np.testing.assert_allclose(back[:, [6, 13]], state[:, [6, 13]], atol=1e-6)
    for i in (3, 10):
        np.testing.assert_allclose(
            Rotation.from_rotvec(back[:, i : i + 3]).as_matrix(),
            Rotation.from_rotvec(state[:, i : i + 3]).as_matrix(),
            atol=1e-5,
        )


def test_model_state_is_continuous_across_the_rotation_vectors_wrap():
    a = np.zeros(14)
    b = np.zeros(14)
    a[3:6] = [0, 0, np.pi - 1e-3]
    b[3:6] = [0, 0, -(np.pi - 1e-3)]  # 0.1 degrees from a, across the wrap
    assert np.abs(a - b).max() > 6
    assert np.abs(model_state(a) - model_state(b)).max() < 3e-3


def test_follow_moves_no_joint_more_than_its_step(hands):
    kin = hands.kin
    q = kin.ik("left", [0.3, 0.3, 1.0], np.zeros(7))
    # The hand turned half a turn about the vertical: far more than one step away.
    turned = Rotation.from_euler("z", np.pi - 0.1).as_matrix()
    goal, _, rotation = kin.follow("left", [0.3, 0.3, 1.0], turned, q, step=0.05)
    assert np.abs(goal - q).max() <= 0.05 + 1e-9
    assert rotation > 0.5  # it doesn't get there in one step


def test_poses_stay_right_when_threads_convert_at_once():
    import sys
    from concurrent.futures import ThreadPoolExecutor

    # Switching threads often enough that one Hands shared by all goes wrong (it
    # did on every chunk); SLA's loader threads hit it on the A100s.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    local = thread_hands()
    kin = local.kin
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(48):
        q = posture(kin, rng)
        rows.append(row(kin, q["left"], q["right"]))
    rows = np.array(rows)
    expected = local.pose(rows)
    try:
        with ThreadPoolExecutor(8) as pool:
            got = list(
                pool.map(lambda r: thread_hands().pose(np.tile(r, (20, 1))), rows)
            )
    finally:
        sys.setswitchinterval(interval)
    for want, chunk in zip(expected, got, strict=True):
        np.testing.assert_allclose(chunk, np.tile(want, (20, 1)), atol=1e-12)


def test_mem_packs_a_cameras_frames_along_the_channels_and_back():
    from sla import mem

    frames = np.random.default_rng(6).integers(0, 255, (6, 4, 5, 3), dtype=np.uint8)
    packed = mem.pack(frames)
    assert packed.shape == (4, 5, 18)
    np.testing.assert_array_equal(
        packed[..., -3:], frames[-1]
    )  # the current frame last
    np.testing.assert_array_equal(mem.unpack(packed), frames)
    batch = np.stack([packed, packed + 1])
    np.testing.assert_array_equal(mem.repack(mem.unpack(batch)), batch)


def test_mem_history_is_stride_apart_and_repeats_the_first_frame():
    from sla import mem

    assert mem.offsets(6, 25) == [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
    assert mem.history(list(range(200)), 6, 25) == [74, 99, 124, 149, 174, 199]
    # 40 ticks into the episode: older frames are the first one.
    assert mem.history(list(range(41)), 6, 25) == [0, 0, 0, 0, 15, 40]
    assert mem.history([0], 6, 25) == [0] * 6


def test_mem_resizes_packed_images_frame_by_frame():
    from sla import mem

    def resize(images, height, width, return_mask=False):
        assert images.shape[-1] == 3
        out = images[..., :height, :width, :]
        mask = np.ones(out.shape[:-1], bool)
        return (out, mask) if return_mask else out

    frames = np.random.default_rng(7).integers(0, 255, (3, 8, 8, 3), dtype=np.uint8)
    resized, mask = mem.per_frame(resize)(mem.pack(frames), 4, 4, return_mask=True)
    assert resized.shape == (4, 4, 9) and mask.shape == (4, 4)
    np.testing.assert_array_equal(mem.unpack(resized), frames[:, :4, :4])
