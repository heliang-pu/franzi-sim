"""FineART-VLA on the industrial benchmark (user request 2026-10-28: train a
model, evaluate it, count the success rate).

The base checkpoint is the public FineART-VLA-subtask-midtrained: PaliGemma 2B
with a 300M action expert, 50-step chunks and subtask prediction, mid-trained
from pi0.5 on FineART's bimanual data (151 tasks, 1718 h). It is fine-tuned on
the ten scenes' Isaac demonstrations, one multi-task model:
- data.py reads the scenes' LeRobot v2.1 datasets as training samples;
- init.py writes the initialization: the base weights untouched, the features
  set to the 16-D contract and the four policy cameras, the normalization
  fitted on the training episodes;
- train.py fine-tunes it;
- recipe.py gives a checkpoint the subtask recipe with memory: the subtask of
  3 s earlier joins the prompt that asks for the subtask;
- serve.py serves a checkpoint to isaac.policy's closed-loop episodes, and
  subtasks.py scores its subtask texts on the held-out episodes.

act_train.py and act_serve.py do the same for ACT, the baseline of another kind
that tells a weak fine-tune from data nothing could succeed on.

These run in their own env (torch built for the GPU, LeRobot at the model
card's commit), not in the project env, so nothing else imports them.
"""
