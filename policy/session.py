"""The per-episode bookkeeping every model in policy/ shares.

Episodes that run at once share one server (eval_client.evaluate runs up to 8;
a VLA holds about 10 GB of GPU memory), so a model keeps each episode's state
under its id: XPolicyLab's prepare_case hook opens an episode, on_trial_end
closes it, and observe and query find it by the observation's
additional_info["episode"]. XPolicyLab's own single-episode methods (reset,
update_obs, get_action) drive one default episode, so its deploy loops run the
models too. NumPy only: the policy envs import it, and so do the tests.

Calls for different episodes run at once (policy.serve lifts XPolicyLab's
server-wide lock); an episode's own calls come one at a time from its client.
So the lock here guards only the episodes, and a model locks what must run one
at a time itself, its GPU work, as the servers before policy/ did: SLA's IK
then runs beside other episodes' inference.

A model subclasses Sessions and implements:
- start(meta): a new episode's state (meta: episode, seed, scene, instruction);
- act(state, obs): the episode's next chunk [n, 16] in SDK units, and a dict of
  what to log about the query (the subtask shown in the videos, seconds, ...);
- see(state, obs), optionally: keep what the model needs from a tick it isn't
  asked to act on (a video memory). observe_every says how often the
  simulator should show it one, in control ticks;
- describe(): what the model is, for the run's summary and the report.
"""

import threading

from policy import contract

DEFAULT = "default"


class Sessions:
    observe_every = None  # control ticks between observe() calls; None: none

    def __init__(self):
        self.lock = threading.Lock()
        self.episodes = {}
        self.pending = None

    def start(self, meta):
        return {}

    def see(self, state, obs):
        pass

    def act(self, state, obs):
        raise NotImplementedError

    def describe(self):
        return {"observe_every": self.observe_every}

    # XPolicyLab's lifecycle hooks, keyed by episode

    def prepare_case(self, meta):
        state = self.start(meta)
        with self.lock:
            self.episodes[meta["episode"]] = state
        return {"episode": meta["episode"]}

    def on_trial_end(self, result):
        with self.lock:
            self.episodes.pop(result.get("episode"), None)

    def _state(self, obs):
        episode = obs.get("additional_info", {}).get("episode", DEFAULT)
        with self.lock:
            if episode not in self.episodes:
                raise KeyError(f"episode {episode!r} was not prepared")
            return self.episodes[episode]

    def observe(self, obs):
        self.see(self._state(obs), obs)

    def query(self, obs):
        return self._query(obs, True)

    def _query(self, obs, see):
        state = self._state(obs)
        if see:
            self.see(state, obs)
        chunk, info = self.act(state, obs)
        return {"actions": contract.actions(chunk), "info": info}

    # XPolicyLab's single-episode methods: update_obs every tick, get_action at
    # a chunk's end

    def reset(self):
        self.prepare_case({"episode": DEFAULT, "seed": 0})
        self.pending = None

    def update_obs(self, obs):
        self.pending = obs
        self.observe(obs)

    def get_action(self):
        return self._query(self.pending, False)["actions"]
