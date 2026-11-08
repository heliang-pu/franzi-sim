"""The simulator's side of the policy interface (policy/), as RoboDojo's
src/eval_client is: env.TaskEnv is XPolicyLab's TASK_ENV over a running scene
episode on MuJoCo or Isaac Sim, episode runs one closed-loop episode against a
policy server (policy.serve) with the policy's deploy.py, and evaluate runs
the benchmark's scenes and seeds and counts the success rate.
"""
