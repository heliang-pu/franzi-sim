"""The learned policies, one package each, behind one websocket interface (user
request 2026-11-08: the methods in policy/, talking to Isaac or any simulator as
RoboDojo's policies do). The interface is XPolicyLab's
(github.com/XPolicyLab/XPolicyLab, commit 6baccd5, Apache-2.0), the layer RoboDojo
and RoboTwin evaluate through: its websocket transport (msgpack with NumPy, a
HELLO that pins the server process, request ids replayed on retry, keepalive)
and its adapter layout.

policy/<name>/ holds:
- model.py: Model(config), a session.Sessions served by
  ``python -m policy.serve --config policy/<name>/deploy.yml`` in the method's
  own env;
- deploy.py: eval_one_episode(TASK_ENV, model_client), the loop on the
  simulator's side (eval_client.env.TaskEnv; mostly loop.run). It runs in the
  simulator's Python, so it imports nothing heavy;
- deploy.yml: policy_name, the server's port and the model's settings, and the
  simulator side's (horizon), each overridable on the command line;
- the method's training and checks beside them.

contract.py is the data the two sides exchange (XPolicyLab's standard
observation and action dictionaries in the datasets' SDK units), session.py
the per-episode bookkeeping, loop.py the shared simulator-side loop, serve.py
the server. Nothing here imports a simulator; eval_client/ is the simulator
side.
"""
