"""The policy interface without a simulator: the observation and action
dictionaries, and policy.loop driving a model over XPolicyLab's websocket
server, as eval_client.episode does."""

import asyncio
import socket
import threading

import numpy as np
import pytest

from policy import contract, loop
from policy.session import Sessions

pytest.importorskip("client_server.ws")  # XPolicyLab, not in every test env


class Echo(Sessions):
    """Returns the state plus the row index, and logs the ticks it saw. With a
    barrier, each episode's first query waits there for the others'."""

    observe_every = 5
    barrier = None

    def start(self, meta):
        return {"seen": [], "seed": meta["seed"]}

    def see(self, state, obs):
        state["seen"].append(contract.info(obs, "tick"))

    def act(self, state, obs):
        rows = contract.state_vector(obs) + np.arange(10)[:, None]
        tick = contract.info(obs, "tick")
        if self.barrier and tick == 0:
            self.barrier.wait(timeout=10)
        return rows.astype(np.float32), {"subtask": f"t{tick}", "seed": state["seed"]}


def observation(tick, state=None, episode="scene/7/x", seed=7):
    frames = {
        name: np.full((4, 6, 3), tick, np.uint8) for name in contract.CAMERAS.values()
    }
    state = np.arange(16, dtype=float) if state is None else state
    return contract.observation("push it", state, frames, episode, seed, tick)


def test_contract_round_trip():
    obs = observation(3, np.linspace(-1, 1, 16))
    assert np.array_equal(contract.state_vector(obs), np.linspace(-1, 1, 16))
    assert set(obs["vision"]) == {
        "cam_head",
        "cam_waist",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    assert contract.frames(obs, ["waist"])["waist"][0, 0, 0] == 3
    chunk = np.arange(32, dtype=np.float32).reshape(2, 16)
    actions = contract.actions(chunk)
    assert set(actions[0]) == {p for p, _ in contract.PARTS}
    assert actions[0]["left_ee_joint_state"].shape == (1,)
    assert np.array_equal(contract.action_vector(actions[1]), chunk[1])
    with pytest.raises(ValueError):
        contract.actions(np.zeros((3, 14)))


class FakeEnv:
    """TASK_ENV's surface (eval_client.env.TaskEnv) without a simulator."""

    def __init__(self, ticks, observe_every, seed=7):
        self.tick, self.ticks, self.seed = 0, ticks, seed
        self.episode = f"scene/{seed}/x"
        self.policy_info = {"observe_every": observe_every}
        self.applied, self.replies = [], []

    def meta(self):
        return {"episode": self.episode, "seed": self.seed, "scene": "scene"}

    def get_obs(self):
        state = np.arange(16, dtype=float) + 100 * (self.seed - 7)
        return observation(self.tick, state, self.episode, self.seed)

    def take_action(self, action):
        self.applied.append(contract.action_vector(action))
        self.tick += 1

    def is_episode_end(self):
        return self.tick >= self.ticks

    def note(self, reply):
        self.replies.append((self.tick, reply["info"]))


@pytest.fixture
def served():
    from policy.serve import policy_server

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    model = Echo()
    server = policy_server(model, "127.0.0.1", port, None, None)
    events, stop = asyncio.new_event_loop(), asyncio.Event()

    async def serve():
        await server.start()
        await stop.wait()
        await server.stop()

    thread = threading.Thread(
        target=events.run_until_complete, args=(serve(),), daemon=True
    )
    thread.start()
    yield f"ws://127.0.0.1:{port}", model
    events.call_soon_threadsafe(stop.set)
    thread.join(timeout=30)
    events.close()


def test_loop_over_websocket(served):
    from eval_client.episode import PolicyClient

    url, model = served
    client = PolicyClient(url, "scene/7")
    try:
        env = FakeEnv(ticks=23, observe_every=5)
        loop.run(env, client, horizon=4)
    finally:
        client.close()
    # Queries every 4 ticks; observe on the multiples of 5 between them.
    assert [tick for tick, _ in env.replies] == [0, 4, 8, 12, 16, 20]
    assert env.replies[1][1] == {"subtask": "t4", "seed": 7}
    # Each chunk's first 4 rows, applied in order: the state plus the row index.
    expected = [np.arange(16) + i % 4 for i in range(23)]
    assert np.allclose(env.applied, expected)
    assert model.episodes == {}  # trial_end freed the episode


def test_errors_reach_the_episode(served):
    from eval_client.episode import PolicyClient

    url, _ = served
    client = PolicyClient(url, "scene/8")
    try:
        with pytest.raises(RuntimeError, match="was not prepared"):
            client.call(func_name="query", obs=observation(0))
        assert client.call(func_name="describe") == {"observe_every": 5}
    finally:
        client.close()


def test_episodes_share_a_server(served):
    """Two episodes at once: their calls overlap (each first query waits at a
    barrier for the other's) and their states stay apart."""
    from eval_client.episode import PolicyClient

    url, model = served
    model.barrier = threading.Barrier(2)
    envs = [FakeEnv(ticks=9, observe_every=None, seed=s) for s in (7, 8)]
    errors = []

    def episode(env):
        client = PolicyClient(url, env.episode)
        try:
            loop.run(env, client, horizon=3)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            client.close()

    threads = [threading.Thread(target=episode, args=(env,)) for env in envs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors, errors
    for env in envs:
        offset = 100 * (env.seed - 7)
        expected = [np.arange(16) + offset + i % 3 for i in range(9)]
        assert np.allclose(env.applied, expected)
        assert env.replies[0][1]["seed"] == env.seed
    assert model.episodes == {}
