"""A separate model that schedules the long scenes (user decision 2026-11-07:
"专门微调别的模型调度长任务"): from the frames it picks the coarse step (sla.coarse)
that SLA should work on, among a closed list, so it can't produce another scene's
sentence or half a word.

    python -m sla.scheduler train --data <collection> --assets <assets>/<asset id> --output <new_dir>
    python -m sla.scheduler chain --data <collection> --assets <assets>/<asset id> --model <checkpoint>

DINOv2-B (facebook/dinov2-base, fine-tuned in its last blocks) reads all four
cameras. A query token, the sum of what the episode knows (the instruction, the step
in hand, the steps done), attends to their patches through a small transformer and
gives the logits. In training the step in hand is the label SUBTASK_EVERY ticks back,
as in fineart.recipe, so the model learns when to move on from what the frames show.
The steps done are those before it in the episode. Half of the samples lie within
4 s of a change of step, where the decisions are.

The closed list follows the instruction: the steps seen with it in training, less
those done. In closed loop the step in hand changes once two queries in a row give it
less than half the probability, to the likeliest of the others.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import Dinov2Config, Dinov2Model

from fineart.protocol import SUBTASK_EVERY
from sla.coarse import LONG_SCENES

BASE = "facebook/dinov2-base"
CAMERAS = ("head", "waist", "wrist_left", "wrist_right")
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
HOLDOUT = 4  # last episodes of each dataset, as fineart.data
WINDOW = 100  # ticks around a change of step that count as near it
CONFIDENT = 0.5
FPS = 25


def tensor_frame(frame):
    """uint8 [3, H, W] as DINOv2's input: resized and padded to SIZE square."""
    frame = frame.float().unsqueeze(0) / 255
    h, w = frame.shape[-2:]
    scale = SIZE / max(h, w)
    size = (round(h * scale), round(w * scale))
    frame = F.interpolate(frame, size=size, mode="bilinear", antialias=True)[0]
    frame = (frame - MEAN) / STD
    bottom, right = SIZE - size[0], SIZE - size[1]
    return F.pad(frame, (0, right, 0, bottom))


def vocabulary(assets, roots):
    """The steps, the instructions and which steps each instruction allows."""
    texts = json.loads((Path(assets) / "labels/texts.json").read_text())
    used, instructions, allowed = {}, {}, {}
    for root in roots:
        scene = root.parent.name
        ids = np.load(Path(assets) / "labels" / f"{scene}.npy")
        tasks = {}
        for line in (root / "meta/tasks.jsonl").read_text().splitlines():
            row = json.loads(line)
            tasks[row["task_index"]] = row["task"]
        meta = json.loads((root / "meta/info.json").read_text())
        for index in range(meta["total_episodes"]):
            path = root / meta["data_path"].format(
                episode_chunk=index // meta["chunks_size"], episode_index=index
            )
            table = pq.read_table(path, columns=["task_index", "index"])
            task = tasks[int(table.column("task_index")[0].as_py())]
            steps = {texts[i] for i in np.unique(ids[table.column("index").to_numpy()])}
            allowed.setdefault(task, set()).update(steps)
            instructions.setdefault(task, len(instructions))
            for step in sorted(steps):
                used.setdefault(step, len(used))
    return {
        "steps": sorted(used, key=used.get),
        "instructions": sorted(instructions, key=instructions.get),
        "allowed": {task: sorted(steps) for task, steps in allowed.items()},
    }


class Frames(Dataset):
    """The long scenes' ticks with what a query knows at each."""

    def __init__(self, roots, assets, vocab, split):
        texts = json.loads((Path(assets) / "labels/texts.json").read_text())
        index = {step: i for i, step in enumerate(vocab["steps"])}
        self.instruction = {t: i for i, t in enumerate(vocab["instructions"])}
        self.steps = len(vocab["steps"])
        self.episodes, rows, near = [], [], []
        for root in roots:
            scene = root.parent.name
            labels = np.load(Path(assets) / "labels" / f"{scene}.npy")
            tasks = {}
            for line in (root / "meta/tasks.jsonl").read_text().splitlines():
                row = json.loads(line)
                tasks[row["task_index"]] = row["task"]
            meta = json.loads((root / "meta/info.json").read_text())
            count = meta["total_episodes"]
            chosen = (
                range(count - HOLDOUT)
                if split == "train"
                else range(count - HOLDOUT, count)
            )
            for number in chosen:
                chunk = number // meta["chunks_size"]
                table = pq.read_table(
                    root
                    / meta["data_path"].format(
                        episode_chunk=chunk, episode_index=number
                    ),
                    columns=["task_index", "index"],
                )
                ids = labels[table.column("index").to_numpy()]
                codes = np.array([index[texts[i]] for i in ids], np.int16)
                changes = np.flatnonzero(np.diff(codes)) + 1
                e = len(self.episodes)
                self.episodes.append(
                    {
                        "codes": codes,
                        "changes": changes,
                        "task": tasks[int(table.column("task_index")[0].as_py())],
                        "scene": scene,
                        "videos": {
                            camera: str(
                                root
                                / meta["video_path"].format(
                                    episode_chunk=chunk,
                                    video_key=f"observation.images.{camera}",
                                    episode_index=number,
                                )
                            )
                            for camera in CAMERAS
                        },
                    }
                )
                rows.append(
                    np.stack([np.full(len(codes), e), np.arange(len(codes))], 1)
                )
                close = np.zeros(len(codes), bool)
                for change in changes:
                    close[max(change - WINDOW, 0) : change + WINDOW] = True
                near.append(close)
        self.rows = np.concatenate(rows).astype(np.int64)
        near = np.concatenate(near)
        # Half the draws near a change of step, half anywhere.
        self.weights = np.where(near, 0.5 / max(near.sum(), 1), 0.0) + 0.5 / len(near)

    def __len__(self):
        return len(self.rows)

    def query(self, data, frame):
        """(step in hand, or the extra id for none yet; multi-hot of steps done)."""
        back = frame - SUBTASK_EVERY
        done = np.zeros(self.steps, np.float32)
        if back < 0:
            return self.steps, done
        for change in data["changes"]:
            if change <= back:
                done[data["codes"][change - 1]] = 1
        return int(data["codes"][back]), done

    def __getitem__(self, i):
        from torchcodec.decoders import VideoDecoder

        episode, frame = (int(v) for v in self.rows[i])
        data = self.episodes[episode]
        current, done = self.query(data, frame)
        return {
            "frames": torch.stack(
                [
                    tensor_frame(
                        VideoDecoder(data["videos"][c], seek_mode="approximate")[frame]
                    )
                    for c in CAMERAS
                ]
            ),
            "instruction": self.instruction[data["task"]],
            "current": current,
            "done": torch.from_numpy(done),
            "target": int(data["codes"][frame]),
        }


class Model(nn.Module):
    def __init__(self, steps, instructions, backbone, tune=4):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        for block in self.backbone.encoder.layer[-tune:]:
            block.requires_grad_(True)
        self.backbone.layernorm.requires_grad_(True)
        width = backbone.config.hidden_size
        self.camera = nn.Parameter(torch.zeros(len(CAMERAS), 1, width))
        self.instruction = nn.Embedding(instructions, width)
        self.current = nn.Embedding(steps + 1, width)
        self.done = nn.Linear(steps, width)
        self.mix = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width,
                8,
                width * 2,
                batch_first=True,
                norm_first=True,
                dropout=0.0,
                activation="gelu",
            ),
            2,
        )
        self.head = nn.Linear(width, steps)
        nn.init.normal_(self.camera, std=0.02)

    def forward(self, frames, instruction, current, done):
        batch, cameras = frames.shape[:2]
        tokens = self.backbone(pixel_values=frames.flatten(0, 1)).last_hidden_state
        tokens = tokens.view(batch, cameras, tokens.shape[1], -1) + self.camera
        query = self.instruction(instruction) + self.current(current) + self.done(done)
        out = self.mix(torch.cat([query[:, None], tokens.flatten(1, 2)], 1))
        return self.head(out[:, 0])


class Schedule:
    """One episode's decisions."""

    def __init__(self, scheduler, instruction):
        self.scheduler = scheduler
        self.instruction = instruction
        self.current = None
        self.done = []
        self.proposal = None

    def step(self, frames):
        """The coarse step for this query; frames are uint8 [H, W, 3] per camera."""
        scheduler = self.scheduler
        batch = torch.stack(
            [
                tensor_frame(torch.from_numpy(np.array(frames[c])).permute(2, 0, 1))
                for c in CAMERAS
            ]
        )[None].to(scheduler.device)
        vocab = scheduler.index
        done = torch.zeros(1, len(vocab), device=scheduler.device)
        for step in self.done:
            done[0, vocab[step]] = 1
        current = len(vocab) if self.current is None else vocab[self.current]
        with torch.inference_mode():
            logits = scheduler.model(
                batch,
                torch.tensor(
                    [scheduler.instructions[self.instruction]],
                    device=scheduler.device,
                ),
                torch.tensor([current], device=scheduler.device),
                done,
            )[0]
        allowed = [
            vocab[s] for s in scheduler.allowed[self.instruction] if s not in self.done
        ]
        mask = torch.full_like(logits, -torch.inf)
        mask[allowed] = 0
        p = (logits + mask).softmax(0)
        if self.current is None:
            self.current = scheduler.steps[int(p.argmax())]
            return self.current
        # Moving on is the step in hand losing its probability, not one next step
        # winning it: where the expert drew the order (which blank, which carton),
        # the next steps share it, and a threshold on the best one kept the
        # schedule waiting (25-33% of the changes missed in cnc_tending and
        # palletizing on the held-out episodes).
        stay = float(p[vocab[self.current]])
        if stay >= CONFIDENT:
            self.proposal = None
        elif self.proposal is not None:
            p[vocab[self.current]] = 0
            self.done.append(self.current)
            self.current, self.proposal = scheduler.steps[int(p.argmax())], None
        else:
            self.proposal = True
        return self.current


class Scheduler:
    def __init__(self, model, vocab, device):
        self.model = model
        self.device = device
        self.steps = vocab["steps"]
        self.index = {step: i for i, step in enumerate(self.steps)}
        self.instructions = {t: i for i, t in enumerate(vocab["instructions"])}
        self.allowed = vocab["allowed"]

    def handles(self, instruction):
        return instruction in self.instructions

    def episode(self, instruction):
        return Schedule(self, instruction)


def load(checkpoint, device=None):
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / "meta.json").read_text())
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    backbone = Dinov2Model(Dinov2Config(**meta["backbone"]))
    model = Model(len(meta["steps"]), len(meta["instructions"]), backbone)
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu"))
    return Scheduler(model.to(device).eval(), meta, device)


def long_roots(data):
    return [Path(data) / scene / "lerobot_v21" for scene in LONG_SCENES]


def evaluate(model, loader, device):
    """Accuracy on a loader's samples, overall and where the step changes."""
    model.eval()
    hits = changed = changed_hits = total = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in loader:
            logits = model(
                b["frames"].to(device),
                b["instruction"].to(device),
                b["current"].to(device),
                b["done"].to(device),
            )
            target = b["target"].to(device)
            good = logits.argmax(1) == target
            moved = b["current"].to(device) != target
            hits += good.sum().item()
            total += len(good)
            changed += moved.sum().item()
            changed_hits += (good & moved).sum().item()
    model.train()
    return hits / max(total, 1), changed_hits / max(changed, 1), changed


def train(args):
    device = "cuda:0"
    roots = long_roots(args.data)
    vocab = vocabulary(args.assets, roots)
    args.output.mkdir(parents=True)
    backbone = Dinov2Model.from_pretrained(BASE)
    meta = vocab | {"backbone": backbone.config.to_dict(), "base": BASE}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    model = Model(len(vocab["steps"]), len(vocab["instructions"]), backbone).to(device)
    data = Frames(roots, args.assets, vocab, "train")
    held = Frames(roots, args.assets, vocab, "validation")
    print(
        f"{len(vocab['steps'])} steps, {len(vocab['instructions'])} instructions, "
        f"{len(data)} training ticks, {len(held)} held out",
        flush=True,
    )
    sampler = WeightedRandomSampler(
        data.weights,
        args.batch * args.steps,
        generator=torch.Generator().manual_seed(1),
    )
    loader = DataLoader(
        data, args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True
    )
    probe = DataLoader(
        torch.utils.data.Subset(
            held,
            np.random.default_rng(0).choice(
                len(held), 1024, p=held.weights / held.weights.sum()
            ),
        ),
        args.batch,
        num_workers=args.workers // 2,
    )
    tuned = [p for p in model.backbone.parameters() if p.requires_grad]
    rest = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [{"params": tuned, "lr": args.lr / 10}, {"params": rest, "lr": args.lr}],
        weight_decay=0.01,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, [args.lr / 10, args.lr], args.steps, pct_start=0.05
    )
    start = time.monotonic()
    log = (args.output / "metrics.jsonl").open("w")
    for step, b in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                b["frames"].to(device, non_blocking=True),
                b["instruction"].to(device),
                b["current"].to(device),
                b["done"].to(device),
            )
            loss = F.cross_entropy(logits, b["target"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            row = {"step": step, "loss": loss.item(), "s": time.monotonic() - start}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
        if step % args.check_every == 0 or step == args.steps:
            accuracy, at_change, count = evaluate(model, probe, device)
            row = {
                "step": step,
                "held_accuracy": accuracy,
                "held_at_change": at_change,
                "changes": count,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)
            torch.save(model.state_dict(), args.output / "model.pt")
    print("SCHEDULER_TRAINED", flush=True)


def chain(args):
    """Ask as the closed loop does, on the held-out expert episodes: every
    SUBTASK_EVERY ticks with its own answers as the memory.

    The expert's frames move on whatever it answers. Where the expert's order was a
    free draw (which parcel, which blank), another right choice leaves it waiting for
    a step the frames never show, which the closed loop, where SLA does what it says,
    wouldn't. So --timing tells it the expert's step and the steps done after each of
    its changes, and counts when it changes: how long after the expert, too early or
    not at all. Without it, the report is how often its answer is the expert's step."""
    from torchcodec.decoders import VideoDecoder

    scheduler = load(args.model)
    roots = long_roots(args.data)
    vocab = {
        "steps": scheduler.steps,
        "instructions": list(scheduler.instructions),
        "allowed": scheduler.allowed,
    }
    held = Frames(roots, args.assets, vocab, "validation")
    steps = scheduler.steps
    report = {}
    for data in held.episodes:
        schedule = scheduler.episode(data["task"])
        decoders = {
            c: VideoDecoder(data["videos"][c], seek_mode="approximate") for c in CAMERAS
        }
        codes, changes = data["codes"], [int(c) for c in data["changes"]]
        hits = asked = 0
        picks, delays, same, spurious = [], [], [], 0
        open_changes = list(changes)
        for t in range(0, len(codes), SUBTASK_EVERY):
            frames = {c: decoders[c][t].permute(1, 2, 0).numpy() for c in CAMERAS}
            before = schedule.current
            pick = schedule.step(frames)
            hits += pick == steps[codes[t]]
            asked += 1
            if not picks or picks[-1] != pick:
                picks.append(pick)
            if not args.timing or before is None:
                continue
            if pick != before:
                # The expert change it answers: the first open one up to one query ahead.
                match = next((c for c in open_changes if c <= t + SUBTASK_EVERY), None)
                if match is None:
                    spurious += 1
                    resync = t
                else:
                    open_changes.remove(match)
                    delays.append((t - match) / FPS)
                    same.append(pick == steps[codes[match]])
                    resync = max(t, match)
            else:
                # A change it hasn't answered within 30 s counts as missed.
                late = [c for c in open_changes if t - c > 30 * FPS]
                if not late:
                    continue
                for c in late:
                    open_changes.remove(c)
                resync = t
            schedule.current = steps[codes[resync]]
            schedule.done = [steps[codes[c - 1]] for c in changes if c <= resync]
            schedule.proposal = None
        expert = [steps[codes[0]]] + [steps[codes[c]] for c in changes]
        missed = len(changes) - len(delays) if args.timing else None
        report.setdefault(data["scene"], []).append(
            {
                "accuracy": hits / asked,
                "steps_expert": len(expert),
                "steps_picked": len(picks),
                "order_right": picks == expert,
                "delays_s": delays,
                "same_step": same,
                "spurious": spurious,
                "missed": missed,
            }
        )
    for scene, rows in report.items():
        line = [scene]
        if args.timing:
            delays = [d for r in rows for d in r["delays_s"]]
            line += [
                f"changes {sum(r['steps_expert'] - 1 for r in rows)}",
                f"answered {len(delays)}",
                f"missed {sum(r['missed'] for r in rows)}",
                f"too early {sum(d < 0 for d in delays)}",
                f"spurious {sum(r['spurious'] for r in rows)}",
                f"delay median {np.median(delays) if delays else float('nan'):.1f} s",
                f"max {max(delays) if delays else float('nan'):.1f} s",
                (
                    f"expert's step {sum(s for r in rows for s in r['same_step'])}"
                    f"/{len(delays)}"
                ),
            ]
        else:
            line += [
                f"accuracy {np.mean([r['accuracy'] for r in rows]):.3f}",
                f"order right {sum(r['order_right'] for r in rows)}/{len(rows)}",
                (
                    f"steps {[r['steps_picked'] for r in rows]} of "
                    f"{[r['steps_expert'] for r in rows]}"
                ),
            ]
        print(*line, flush=True)
    name = "chain_timing.json" if args.timing else "chain.json"
    (Path(args.model) / name).write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--assets", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--steps", type=int, default=20000)
    fit.add_argument("--batch", type=int, default=32)
    fit.add_argument("--workers", type=int, default=12)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--check-every", type=int, default=2000)
    check = commands.add_parser("chain")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--assets", type=Path, required=True)
    check.add_argument("--model", type=Path, required=True)
    check.add_argument(
        "--timing", action="store_true", help="the expert's step after each change"
    )
    args = parser.parse_args()
    (train if args.command == "train" else chain)(args)


if __name__ == "__main__":
    main()
