"""SLA, the Spatial-Language-Action model, wrapped for this robot (user request
2026-11-07; arXiv 2611.08373, github.com/OpenHelix-Team/Spatial-Forcing).

SLA is pi0.5 pre-trained on six public robot datasets with its visual tokens
aligned to a 3D teacher (OmniVGGT). The flow loss trains the vision (no knowledge
insulation), 25% of the samples hide the state, and the teacher is used in
training only. It sees three cameras and each hand's absolute pose and outputs
each hand's relative pose; it knows nothing of joints.

The wrapper keeps the upstream checkout untouched and outside this repository:
- eef.py: the 16-D joint data as SLA's hand poses and back, through the shared
  kinematics (no openpi, so the tests run anywhere);
- coarse.py: the coarse prompts, one per object or interaction in the long scenes
  and the instruction itself in the short ones;
- policy.py: the transforms and data config SLA's training and policy load;
- prepare.py: the normalization and each frame's prompt, written once per dataset;
- train.py: the fine-tuning, through SLA's own train_align.py;
- replay.py: an expert run through hand poses and back, what this route costs;
- serve.py: a checkpoint behind fineart.serve's protocol, so isaac.policy and
  isaac.evaluate run it unchanged and receive joint targets;
- scheduler.py: a separate fine-tuned model that picks the long scenes' next
  coarse prompt.

These run in SLA's env (its JAX, Pinocchio 2.7, and PyTorch for the teacher), not
in the project env; only eef.py and coarse.py are imported elsewhere, by the tests.
"""
