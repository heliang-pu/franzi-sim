"""The subtask recipe with memory (user decision 2026-10-29), and the tool that
gives a checkpoint folder it.

The base's recipe generates the subtask from the task and the frames alone. The
benchmark's episodes start and end in the default posture, so one frame often
can't tell the first subtask from the last: at step 5000, panel_operation's
first frame got "Move the right arm back to its start pose" in 3 of 4 held-out
episodes, and in closed loop such an answer repeats, since the arm then stays
where it is. The memory recipe adds the user turn "Previous subtask: <text>"
before the answer: the subtask of SUBTASK_EVERY ticks (3 s) earlier, NO_MEMORY
in an episode's first 3 s:
- in training, the expert's label that many ticks back (fineart.data);
- in closed loop, the subtask the policy generated at its previous query, that
  many ticks back (isaac.policy).
Not the previous segment's subtask: that changes on the same tick as the label,
so no training sample would show when to move on.

    python -m fineart.recipe --model <checkpoint or initialization> --output <new folder> \
        [--source <initialization with the memory recipe>]

writes a copy of the folder (its files hard-linked) whose config and
preprocessor carry the memory recipe. For a checkpoint to --resume from,
--source is the initialization its new checkpoints take their processors from
(fineart.train's save()).
"""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

from lerobot.datasets.recipe import TrainingRecipe, render_message_turns
from lerobot.utils.constants import MESSAGES_RENDERED

from fineart.protocol import NO_MEMORY

MEMORY_TURN = {
    "role": "user",
    "content": "Previous subtask: ${memory}",
    "stream": "high_level",
    "target": False,
    "if_present": None,
    "tool_calls_from": None,
}
RENDER_STEPS = (
    "render_runtime_messages_processor",
    "render_training_messages_processor",
)


def with_memory(recipe):
    """The recipe (a dict) with MEMORY_TURN before its subtask answer."""
    recipe = copy.deepcopy(recipe)
    messages = recipe["blend"]["high_level_subtask"]["messages"]
    if not any("${memory}" in (m.get("content") or "") for m in messages):
        answer = next(i for i, m in enumerate(messages) if m.get("target"))
        messages.insert(answer, dict(MEMORY_TURN))
    return recipe


def load(model):
    """A checkpoint folder's recipe, as its runtime renderer has it."""
    processor = json.loads((Path(model) / "policy_preprocessor.json").read_text())
    step = next(s for s in processor["steps"] if s["registry_name"] == RENDER_STEPS[0])
    return TrainingRecipe.from_dict(step["config"]["recipe"])


def uses_memory(recipe):
    return "memory" in recipe.referenced_binding_names()


def subtask_messages(recipe, task, memory=None):
    """The prompt that asks for the subtask, as the runtime renderer renders a
    next_subtask query, with the memory bound too (ignored by a recipe without
    it)."""
    bindings = dict.fromkeys(recipe.referenced_binding_names())
    bindings.update(task=task, memory=memory or NO_MEMORY)
    turns = recipe.prompt_turns("subtask")
    return render_message_turns(turns, bindings)[MESSAGES_RENDERED]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="a checkpoint's new processor source"
    )
    args = parser.parse_args()
    target = args.output.resolve()
    assert not target.exists(), f"{target} exists"
    state = args.model / "training_state.json"
    assert state.exists() == (args.source is not None), (
        "--source goes with a checkpoint (training_state.json), and only with one"
    )
    if args.source:
        assert uses_memory(load(args.source)), f"{args.source} has no memory recipe"
    target.mkdir(parents=True)
    rewritten = ("config.json", "policy_preprocessor.json", "training_state.json")
    for path in args.model.iterdir():
        if path.name.startswith(".") or path.name in rewritten:
            continue
        if path.is_dir():
            shutil.copytree(path, target / path.name)
        else:
            os.link(path.resolve(), target / path.name)
    processor = json.loads((args.model / "policy_preprocessor.json").read_text())
    recipes = []
    for step in processor["steps"]:
        if step["registry_name"] in RENDER_STEPS:
            step["config"]["recipe"] = with_memory(step["config"]["recipe"])
            recipes.append(step["config"]["recipe"])
    assert len(recipes) == 2 and recipes[0] == recipes[1], "render steps differ"
    (target / "policy_preprocessor.json").write_text(json.dumps(processor, indent=2))
    config = json.loads((args.model / "config.json").read_text())
    # An initialization's config may leave the recipe to the config class's
    # default; its processors always carry it. (A checkpoint's config spells
    # out route: null, which its processors leave out.)
    if config.get("recipe") is not None:
        mine = TrainingRecipe.from_dict(with_memory(config["recipe"]))
        assert mine == TrainingRecipe.from_dict(recipes[0]), "config, processors differ"
    config["recipe"] = recipes[0]
    (target / "config.json").write_text(json.dumps(config, indent=2))
    if args.source:
        report = json.loads(state.read_text())
        report.update(
            base_checkpoint=str(args.source.resolve()),
            recipe_from=str(args.model.resolve()),
        )
        (target / "training_state.json").write_text(json.dumps(report, indent=2))
    turns = config["recipe"]["blend"]["high_level_subtask"]["messages"]
    print(json.dumps({"output": str(target), "high_level_subtask": turns}, indent=2))


if __name__ == "__main__":
    main()
