"""MEM's short-horizon video memory (arXiv 2603.03596) for SLA's JAX pi0.5, as LeRobot's pi05
has it (lerobot/policies/pi05/memory.py; user decision 2026-11-07: "MEM 视频记忆加到底层").

Every fourth of SigLIP's 27 layers attends over time too: a causal temporal attention between
the same patch of a camera's frames, composed with the spatial one, both from that layer's
pretrained q/k/v (MEM appendix C, eq. 3). The input of those layers carries a sinusoidal time
embedding, zero for the current frame. After the last such layer only the current frame goes
on, so the language model sees as many image tokens as before; one frame gives stock SigLIP.
No parameter is added. Only the visual memory: no proprioceptive memory, no language memory.

Unlike LeRobot, frames from before an episode's start are its first frame again, not masked:
LeRobot's dataset clamps such queries to the first frame in training, and isaac.policy fills
the history the same way, so the two agree.

The frames travel through SLA's unchanged pipeline packed along each camera image's channels,
oldest first and the current frame last ([h, w, 3 * frames]). install() patches the places
that must know: the data loader asks for the cameras' history, the transforms resize frame by
frame, training's colour jitter draws once per camera and sample, the 3D teacher sees the
current frame alone, and Pi0.embed_prefix runs the SigLIP below.

    python -m sla.mem check   # in SLA's env: one frame equals stock SigLIP, history matters
"""

import functools

import numpy as np

FRAMES = 6  # the current one included, as MEM's pretraining
STRIDE = 25  # control ticks between frames: one second at 25 Hz
EVERY = 4  # every fourth SigLIP layer attends over time
FPS = 25
CAMERAS = ("waist", "wrist_left", "wrist_right")  # sla.policy.CAMERAS' frames


def offsets(frames=FRAMES, stride=STRIDE):
    """The history's times (s) before the current frame, oldest first."""
    return [-(frames - 1 - i) * stride / FPS for i in range(frames)]


def pack(frames):
    """Frames [T, h, w, 3], oldest first, as one image [h, w, 3T]."""
    return np.concatenate(list(np.asarray(frames)), axis=-1)


def unpack(image, xp=np):
    """Images [..., h, w, 3T] as frames [..., T, h, w, 3]."""
    *lead, h, w, c = image.shape
    return xp.moveaxis(xp.reshape(image, (*lead, h, w, c // 3, 3)), -2, -4)


def repack(frames, xp=np):
    """unpack()'s inverse: frames [..., T, h, w, 3] as images [..., h, w, 3T]."""
    *lead, t, h, w, _ = frames.shape
    return xp.reshape(xp.moveaxis(frames, -4, -2), (*lead, h, w, 3 * t))


def history(ticks, frames=FRAMES, stride=STRIDE):
    """For the frames seen so far (oldest first, one per tick, the current one last), the
    indices of the history the model takes: stride ticks apart, the first frame repeated where
    the episode is younger."""
    now = len(ticks) - 1
    return [max(now - (frames - 1 - i) * stride, 0) for i in range(frames)]


def time_embedding(frames, width):
    """MEM's fixed temporal embedding, exactly zero for the current frame."""
    import jax.numpy as jnp

    positions = jnp.arange(1 - frames, 1, dtype=jnp.float32)[:, None]
    rates = jnp.exp(
        jnp.arange(0, width, 2, dtype=jnp.float32) * (-np.log(10_000.0) / width)
    )
    angles = positions * rates[None]
    return jnp.stack([jnp.sin(angles), jnp.cos(angles) - 1.0], axis=-1).reshape(
        frames, width
    )


def video_tokens(module, frames):
    """SigLIP's tokens for the current frame of frames [b, T, h, w, 3] in [-1, 1], with MEM's
    space-time attention; module is SLA's bound siglip._Module (method= of its ToNNX)."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from openpi.models import siglip
    from openpi.training import sharding

    p = module.variables["params"]
    dtype = module.dtype_mm
    heads = module.num_heads
    depth = module.width // heads
    b, t = frames.shape[:2]
    # The stem in float32 as siglip's own (its nn.Conv promotes bfloat16 weights).
    x = jnp.reshape(frames, (b * t, *frames.shape[2:])).astype(jnp.float32)
    x = jax.lax.conv_general_dilated(
        x,
        p["embedding"]["kernel"].astype(jnp.float32),
        module.patch_size,
        "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = x + p["embedding"]["bias"].astype(jnp.float32)
    n = x.shape[1] * x.shape[2]
    x = jnp.reshape(x, (b * t, n, module.width)) + p["pos_embedding"]
    x = jnp.reshape(x.astype(dtype), (b, t, n, module.width))

    def layer(x, block, timed):
        """One encoder block on frames x [b, T, n, w]. Timed (MEM's layers), its input
        carries the time embedding and each patch first attends causally over its frames;
        the spatial attention then mixes those values: one composed attention, one output
        projection. Untimed, each frame attends to itself alone, which leaves the values
        as they are: stock SigLIP's block."""
        frames_now = x.shape[1]
        attention = block["MultiHeadDotProductAttention_0"]

        def norm(name, v):
            return nn.LayerNorm(dtype=dtype).apply({"params": block[name]}, v)

        def project(name, v):
            return nn.DenseGeneral((heads, depth), dtype=dtype).apply(
                {"params": attention[name]}, v
            )

        x = sharding.activation_sharding_constraint(x)
        embedding = time_embedding(frames_now, module.width).astype(x.dtype)[:, None]
        y = norm("LayerNorm_0", x + timed.astype(x.dtype) * embedding)
        q, k, v = (project(name, y) for name in ("query", "key", "value"))  # b t n h d
        causal = jnp.tril(jnp.ones((frames_now, frames_now), bool))
        mixed = nn.dot_product_attention(
            *(jnp.swapaxes(a, 1, 2) for a in (q, k, v)),
            mask=jnp.where(timed, causal, jnp.eye(frames_now, dtype=bool)),
            dtype=dtype,
        )
        attended = nn.dot_product_attention(
            q, k, jnp.swapaxes(mixed, 1, 2), dtype=dtype
        )
        y = nn.DenseGeneral(module.width, axis=(-2, -1), dtype=dtype).apply(
            {"params": attention["out"]}, attended
        )
        x = x + sharding.activation_sharding_constraint(y)
        y = siglip.MlpBlock(mlp_dim=module.mlp_dim, dtype_mm=dtype).apply(
            {"params": block["MlpBlock_0"]},
            jnp.reshape(norm("LayerNorm_1", x), (b * frames_now, n, -1)),
        )
        return x + jnp.reshape(sharding.activation_sharding_constraint(y), x.shape)

    # A scan over the layers, as siglip's own nn.scan: its backward recomputes one layer
    # at a time. Unrolled, XLA hoisted every layer's recomputation and kept them all
    # (58 GiB of temporaries for 72 frames' gradient against siglip's 5.8).
    @functools.partial(
        jax.checkpoint,
        prevent_cse=False,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    def step(x, layer_in):
        return layer(x, *layer_in), None

    blocks = p["Transformer"]["encoderblock"]
    timed = (np.arange(1, module.depth + 1) % EVERY == 0) & (t > 1)
    # Nothing mixes frames above the last timed layer: only the current frame goes on.
    last = int(np.flatnonzero(timed)[-1]) + 1 if timed.any() else module.depth
    below, above = blocks, None
    if last < module.depth:
        leaves, tree = jax.tree.flatten(blocks)
        halves = [jax.lax.split(a, (last, module.depth - last)) for a in leaves]
        below, above = (tree.unflatten([h[i] for h in halves]) for i in (0, 1))
    x, _ = jax.lax.scan(step, x, (below, jnp.asarray(timed[:last])))
    x = x[:, -1:]
    if above is not None:
        x, _ = jax.lax.scan(step, x, (above, jnp.asarray(timed[last:])))
    x = nn.LayerNorm(dtype=dtype).apply(
        {"params": p["Transformer"]["encoder_norm"]}, x[:, -1]
    )
    return nn.Dense(module.num_classes, dtype=dtype).apply({"params": p["head"]}, x)


def embed_prefix(self, obs):
    """Pi0.embed_prefix, with a packed history going through video_tokens."""
    import einops
    import jax.numpy as jnp

    input_mask, ar_mask, tokens = [], [], []
    for name in obs.images:
        image = obs.images[name]
        if image.shape[-1] == 3:
            image_tokens, _ = self.PaliGemma.img(image, train=False)
        else:
            image_tokens = self.PaliGemma.img(unpack(image, jnp), method=video_tokens)
        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
        )
        ar_mask += [False] * image_tokens.shape[1]
    if obs.tokenized_prompt is not None:
        prompt = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(prompt)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * prompt.shape[1]
    return (
        jnp.concatenate(tokens, axis=1),
        jnp.concatenate(input_mask, axis=1),
        jnp.array(ar_mask),
    )


def preprocess(original):
    """model.preprocess_observation; in training, a packed camera's frames get one draw of
    the augmentation per sample, as one image would."""

    @functools.wraps(original)
    def wrapper(rng, observation, *, train=False, sf_align=False, **kwargs):
        if not train or all(a.shape[-1] == 3 for a in observation.images.values()):
            return original(rng, observation, train=train, sf_align=sf_align, **kwargs)
        import augmax
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        images = {}
        for key in _model.IMAGE_KEYS:
            frames = unpack(observation.images[key], jnp) / 2.0 + 0.5
            steps = []
            if "wrist" not in key and not sf_align:
                h, w = frames.shape[-3:-1]
                steps += [
                    augmax.RandomCrop(int(w * 0.95), int(h * 0.95)),
                    augmax.Resize(w, h),
                    augmax.Rotate((-5, 5)),
                ]
            chain = augmax.Chain(
                *steps, augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            )
            draws = jax.random.split(rng, frames.shape[0])
            frames = jax.vmap(
                lambda r, f, chain=chain: jax.vmap(lambda g: chain(r, g))(f)
            )(draws, frames)
            images[key] = repack(frames * 2.0 - 1.0, jnp)
        batch = observation.state.shape[:-1]
        return _model.Observation(
            images=images,
            image_masks={
                key: jnp.asarray(observation.image_masks[key])
                if key in observation.image_masks
                else jnp.ones(batch, dtype=jnp.bool)
                for key in images
            },
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

    return wrapper


def per_frame(original):
    """openpi_client.image_tools.resize_with_pad, frame by frame on packed images."""

    @functools.wraps(original)
    def wrapper(images, height, width, *args, return_mask=False, **kwargs):
        if images.shape[-1] == 3:
            return original(
                images, height, width, *args, return_mask=return_mask, **kwargs
            )
        out = original(
            unpack(images), height, width, *args, return_mask=return_mask, **kwargs
        )
        if not return_mask:
            return repack(out)
        resized, mask = out
        return repack(resized), mask[..., -1, :, :]

    return wrapper


def current_frame(original):
    """The 3D teacher's call, on each camera's current frame alone."""

    @functools.wraps(original)
    def wrapper(self, observation, *args, **kwargs):
        images = {k: v[..., -3:] for k, v in observation.images.items()}
        return original(self, observation.replace(images=images), *args, **kwargs)

    return wrapper


def with_history(original, frames, stride):
    """data_loader._action_delta_timestamps, plus the cameras' history."""

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        times = dict(original(*args, **kwargs))
        for camera in CAMERAS:
            times[f"observation.images.{camera}"] = offsets(frames, stride)
        return times

    return wrapper


def query_timestamps(self, current_ts, query_indices=None):
    """LeRobotDataset._get_query_timestamps for datasets 4, whose select()[column] is a
    Column, not the list of tensors torch.stack wants; SLA never asked for a video's
    history before."""
    import torch

    times = {}
    for key in self.meta.video_keys:
        if query_indices is not None and key in query_indices:
            column = self.hf_dataset.select(query_indices[key])["timestamp"]
            times[key] = torch.stack(list(column)).tolist()
        else:
            times[key] = [current_ts]
    return times


def install(frames=FRAMES, stride=STRIDE):
    """Patch SLA's pipeline for packed histories of frames frames, stride ticks apart."""
    from lerobot.common.datasets import lerobot_dataset
    from openpi.models import model as _model
    from openpi.models import pi0
    from openpi.training import align_utils, data_loader
    from openpi_client import image_tools

    lerobot_dataset.LeRobotDataset._get_query_timestamps = query_timestamps
    pi0.Pi0.embed_prefix = embed_prefix
    _model.preprocess_observation = preprocess(_model.preprocess_observation)
    image_tools.resize_with_pad = per_frame(image_tools.resize_with_pad)
    shard = align_utils._TorchAlignFeatureShard
    shard.__call__ = current_frame(shard.__call__)
    data_loader._action_delta_timestamps = with_history(
        data_loader._action_delta_timestamps,
        frames,
        stride,
    )


def check():
    """In SLA's env: video_tokens on one frame equals stock SigLIP's tokens, its history
    changes the current frame's tokens, and its gradient reaches the frames."""
    import jax
    import jax.numpy as jnp
    from openpi.models import siglip

    module = siglip.Module(
        num_classes=2048,
        variant="So400m/14",
        pool_type="none",
        scan=True,
        dtype_mm="float32",
    )
    rng = jax.random.key(0)
    images = jax.random.uniform(rng, (2, 3, 224, 224, 3), minval=-1.0, maxval=1.0)
    variables = module.init(rng, images[:, -1], train=False)
    # SigLIP's head starts at zero, which would hide everything below it.
    head = variables["params"]["head"]
    head["kernel"] = jax.random.normal(rng, head["kernel"].shape) * 0.02
    stock, _ = module.apply(variables, images[:, -1], train=False)
    one = module.apply(variables, images[:, -1:], method=video_tokens)
    print("one frame vs stock: max abs diff", float(jnp.abs(one - stock).max()))
    three = module.apply(variables, images, method=video_tokens)
    other = module.apply(variables, images.at[:, 0].set(0.0), method=video_tokens)
    print("history changes the tokens by", float(jnp.abs(three - other).max()))
    gradient = jax.grad(
        lambda f: module.apply(variables, f, method=video_tokens).sum()
    )(images)
    print(
        "gradient per frame",
        [float(jnp.abs(gradient[:, i]).sum()) for i in range(images.shape[1])],
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["check"]:
        check()
