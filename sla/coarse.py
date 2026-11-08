"""Coarse prompts for SLA (user decision 2026-11-07: "subtask粗一点").

The experts' subtask texts name one action of one arm: pick X, put X, move the arm
back. SLA is told less:
- in a short scene the prompt is the episode's instruction, for the whole episode;
- in a long scene it names one object moved to its place or one interaction, from
  the first move toward it until the next one starts, with the arm's way back. The
  scheduler (scheduler.py) picks among these.

A table maps a scene's expert texts to coarse ones. Two markers stand for texts
that need their neighbours: EARLIER takes the coarse text before it (the way back,
or a step whose text doesn't say which blank), LATER the one after it (a pick,
whose place only the next text names).
"""

LONG_SCENES = (
    "cnc_tending",
    "lidded_tote",
    "palletizing",
    "parcel_sorting",
    "shaped_kitting",
)
SHORT_SCENES = (
    "panel_operation",
    "precision_insertion",
    "tote_carry",
    "tote_transfer",
    "tray_push",
)

EARLIER, LATER = "<earlier>", "<later>"
BACK = {
    f"Move the {side} arm back to its {pose} pose.": EARLIER
    for side in ("left", "right")
    for pose in ("start", "ready")
}


def _parcel(zone, hand):
    put = f"Put parcel {zone} into bin {zone} with the {hand} hand."
    return {f"Pick up parcel {zone} with the {hand} hand.": put, put: put}


def _kit(part, hand):
    put = f"Put the {part} into its pocket with the {hand} hand."
    return {f"Pick up the {part} with the {hand} hand.": put, put: put}


def _carton(side):
    picks = {
        f"Pick up the {side} {place} carton with the {side} hand.": LATER
        for place in ("inner", "outer")
    }
    slots = (1, 2) if side == "right" else (3, 4)
    stacks = {
        text: text
        for text in (
            f"Stack the {side} {place} carton in slot {slot} with the {side} hand."
            for place in ("inner", "outer")
            for slot in slots
        )
    }
    return picks | stacks


def _cnc():
    part = "Put the finished part on the hand-over stand with the left hand."
    load = "Load the raw blank into the vise with the left hand."
    table = {
        "Open the machine door with the left hand.": (
            "Open the machine door with the left hand."
        ),
        "Take the finished part out of the vise with the left hand.": part,
        part: part,
        "Pick up the finished part from the hand-over stand with the right hand.": (
            LATER
        ),
        "Put the raw blank on the hand-over stand with the right hand.": EARLIER,
        "Pick up the raw blank from the hand-over stand with the left hand.": load,
        load: load,
        "Close the machine door with the left hand.": (
            "Close the machine door with the left hand."
        ),
        "Press CYCLE START with the right hand.": (
            "Press CYCLE START with the right hand."
        ),
    }
    for place in ("left", "middle", "right"):
        put = f"Put the finished part in the tray's {place} back pocket with the right hand."
        table[put] = put
        table[f"Take the {place} raw blank out of the tray with the right hand."] = (
            f"Put the {place} raw blank on the hand-over stand with the right hand."
        )
    return table


_HEX = "Put the hex blank on the station plate with the left hand."
TABLES = {
    "cnc_tending": _cnc() | BACK,
    "lidded_tote": {
        "Open the tote's lid with the left hand.": (
            "Open the tote's lid with the left hand."
        ),
        "Pick up the hex blank with the left hand.": _HEX,
        _HEX: _HEX,
        "Close the tote's lid with the left hand.": (
            "Close the tote's lid with the left hand."
        ),
    }
    | BACK,
    "palletizing": _carton("left") | _carton("right") | BACK,
    "parcel_sorting": (
        _parcel("A", "left")
        | _parcel("B", "left")
        | _parcel("C", "right")
        | _parcel("D", "right")
        | BACK
    ),
    "shaped_kitting": (
        _kit("flanged bushing", "left")
        | _kit("spur gear", "left")
        | _kit("L bracket", "right")
        | _kit("hex blank", "right")
        | BACK
    ),
}


def coarsen(scene, subtasks):
    """The coarse text of each of a long scene's expert subtask texts, given in an
    episode's order. Raises KeyError on a text its table doesn't know, and
    ValueError on a marker with nothing to take."""
    table = TABLES[scene]
    coarse = [table[text] for text in subtasks]
    for i in range(len(coarse) - 2, -1, -1):
        if coarse[i] == LATER:
            coarse[i] = coarse[i + 1]
    for i in range(1, len(coarse)):
        if coarse[i] == EARLIER:
            coarse[i] = coarse[i - 1]
    if {LATER, EARLIER} & set(coarse):
        raise ValueError(f"{scene}: a marker with no neighbour in {subtasks}")
    return coarse


def steps(coarse):
    """Runs of equal coarse texts, in order."""
    return [t for i, t in enumerate(coarse) if i == 0 or t != coarse[i - 1]]
