"""SLA's view of this robot, in the form its training and its Policy load: the
transforms between the datasets' rows and hand poses, and their data config.

The base image is the waist D435, which sees the bench better than the head's, whose
reachable bench fills the bottom fifth of its image; SLA takes three cameras, and
the head stays out.
"""

import dataclasses
import functools
import json
import pathlib

import numpy as np
from openpi import transforms
from openpi.training.config import DataConfigFactory, ModelTransformFactory

from sla import mem
from sla.eef import DIM, STATE, hands, model_state, rotvec_state

CAMERAS = {
    "base_0_rgb": "waist",
    "left_wrist_0_rgb": "wrist_left",
    "right_wrist_0_rgb": "wrist_right",
}
ROTATIONS = transforms.make_bool_mask(-3, 3, -1, -3, 3, -1)
RELATIVE = transforms.make_bool_mask(6, -1, 6, -1)  # the grippers stay absolute


@functools.cache
def labels(folder, scene):
    """A dataset's per-frame prompt ids (prepare.py) and the prompts."""
    folder = pathlib.Path(folder)
    return (
        np.load(folder / f"{scene}.npy", mmap_mode="r"),
        json.loads((folder / "texts.json").read_text()),
    )


def image(frame):
    """A frame as uint8 [H, W, 3]: LeRobot gives floats [3, H, W], the server uint8
    [H, W, 3]. A history of frames ([T, 3, H, W] or [T, H, W, 3], sla.mem) comes out
    packed along the channels, [H, W, 3T]."""
    frame = np.asarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        frame = (255 * frame).astype(np.uint8)
    if frame.shape[-3] == 3 and frame.shape[-1] != 3:
        frame = np.moveaxis(frame, -3, -1)
    return frame if frame.ndim == 3 else mem.pack(frame)


@dataclasses.dataclass(frozen=True)
class FranziInputs(transforms.DataTransformFn):
    """A dataset row or a server request as hand poses. In training the prompt comes
    from the frame's global index in this scene's labels; the server passes it."""

    scene: str | None = None
    labels: str | None = None

    def __call__(self, data):
        inputs = {
            "image": {
                key: image(data["images"][camera]) for key, camera in CAMERAS.items()
            },
            "image_mask": {key: np.True_ for key in CAMERAS},
            "state": hands().pose(data["state"]),
        }
        if "actions" in data:
            inputs["actions"] = hands().pose(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif self.labels is not None:
            ids, texts = labels(self.labels, self.scene)
            inputs["prompt"] = texts[int(ids[int(data["index"])])]
        return inputs


@dataclasses.dataclass(frozen=True)
class ModelState(transforms.DataTransformFn):
    """After DeltaActions: the state as the model sees it (sla.eef.model_state)."""

    def __call__(self, data):
        return {**data, "state": model_state(data["state"])}


@dataclasses.dataclass(frozen=True)
class RotvecState(transforms.DataTransformFn):
    """Before AbsoluteActions: the model's state as hand poses again."""

    def __call__(self, data):
        state = np.asarray(data["state"])[..., :STATE]
        return {**data, "state": rotvec_state(state)}


@dataclasses.dataclass(frozen=True)
class FranziOutputs(transforms.DataTransformFn):
    """The absolute hand poses; sla.serve solves them into joints."""

    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :DIM]}


@dataclasses.dataclass(frozen=True)
class FranziDataConfig(DataConfigFactory):
    """One scene's dataset (repo_id, its local path). With scene unset it serves
    inference, where the server passes each prompt."""

    scene: str | None = None
    labels: str | None = None

    def create(self, assets_dirs, model_config):
        group = (
            transforms.Group(
                inputs=[FranziInputs(self.scene, self.labels)],
                outputs=[FranziOutputs()],
            )
            .push(
                inputs=[transforms.DeltaActions(RELATIVE, ROTATIONS)],
                outputs=[transforms.AbsoluteActions(RELATIVE, ROTATIONS)],
            )
            .push(inputs=[ModelState()], outputs=[RotvecState()])
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "images": {
                                camera: f"observation.images.{camera}"
                                for camera in CAMERAS.values()
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "index": "index",
                        }
                    )
                ]
            ),
            data_transforms=group,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )
