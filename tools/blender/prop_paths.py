"""Where each prop lives (stdlib only, for Blender's Python and PROP_PYTHON alike).

A prop that only one scene uses lives in that scene's folder,
scenes/<scene>/assets/<id>/; the others, and every new prop until a scene claims
it, in the shared library, assets/props/<id>/. Ids are unique across both. A
prop names a file of another prop (a shared texture atlas) as ../<id>/<file>,
which locate() finds wherever that prop lives; "<id>/<file>" names any prop's
file (the compaction policy and reports). To build a new prop into a scene,
create its empty folder there first.

PROPS / "<id>" is the prop's folder, so the build and check scripts join ids
onto PROPS as they did onto the library folder; PROPS.glob("*/<pattern>") and
PROPS.rglob(...) cover every prop. sim/prop_assets.py finds props the same way.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "assets" / "props"
SCENES = ROOT / "scenes"


def folders():
    """Every prop's folder, by id (a scene's assets/derived/ is a cache, not a prop)."""
    found = {}
    scenes = [p for p in SCENES.glob("*/assets/*") if p.name != "derived"]
    for folder in sorted([*LIBRARY.iterdir(), *scenes]):
        if not folder.is_dir():
            continue
        if folder.name in found:
            raise RuntimeError(f"prop {folder.name} is in {found[folder.name]} and {folder}")
        found[folder.name] = folder
    return found


def prop_dir(prop_id):
    return folders().get(prop_id, LIBRARY / prop_id)


def locate(name, folder=None):
    """The file "<id>/<file>" names, or a file a prop in folder names: its own
    "<file>" or another prop's "../<id>/<file>", in a sibling folder of that
    name if there is one (a prop and its atlas owner in one place, or a test's
    scratch folder), else wherever that prop lives."""
    if folder is not None:
        if not name.startswith("../"):
            return Path(folder) / name
        sibling = Path(folder).parent / name[3:]
        if sibling.parent.is_dir():
            return sibling
    prop_id, file = name.removeprefix("../").split("/", 1)
    return prop_dir(prop_id) / file


def reference(path, folder):
    """How a prop in folder names the file at path (the inverse of locate)."""
    path = Path(path)
    if path.parent == Path(folder):
        return path.name
    return f"../{path.parent.name}/{path.name}"


def label(path):
    """A prop file's "<id>/<file>" name."""
    return f"{Path(path).parent.name}/{Path(path).name}"


class PropFolders:
    """PROPS / "<id>" is the prop's folder, wherever it lives."""

    def __truediv__(self, prop_id):
        return prop_dir(str(prop_id))

    def glob(self, pattern):
        first, _, rest = pattern.partition("/")
        if first != "*" or not rest:
            raise ValueError(f"PROPS.glob takes '*/<pattern>', not {pattern!r}")
        return iter(sorted(p for folder in folders().values() for p in folder.glob(rest)))

    def rglob(self, pattern):
        return iter(sorted(p for folder in folders().values() for p in folder.rglob(pattern)))


PROPS = PropFolders()
