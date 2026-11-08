"""Convex-hull colliders for the robot links (user request 2026-10-25: collision
detection, RoboDojo's mesh colliders as the reference).

The source model collides each link as its STL's bounding box. scene.py
swaps those boxes for the STLs' convex hulls, cut to at most VERTICES vertices
so that PhysX, which cooks hulls of at most 255 (isaac/stage.py), gets the same
polytope as MuJoCo. A cut hull lies inside the full one, at most 2.1 mm inside
it (the base). NumPy, SciPy and trimesh only.

The hulls are cached per machine in robots/franzi/assets/derived/robot_hulls.json,
keyed by VERSION, VERTICES and the STL bytes: cutting them takes about 2 s,
building the rest of the scene 0.4 s.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
MESHES = HERE / "assets/urdf/meshes"
CACHE = HERE / "assets/derived/robot_hulls.json"
VERSION = 1
VERTICES = 255
TOLERANCE = 1e-4  # m: a vertex this close to the cut hull counts as inside
# The cut hull starts from the extreme points along these directions.
DIRECTIONS = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]],
    float,
)


def outside(points, hull):
    """How far each point lies outside hull (a scipy ConvexHull), m: its largest
    signed facet distance, 0 or less inside."""
    # macOS Accelerate's matmul raises spurious floating-point warnings.
    with np.errstate(all="ignore"):
        return (points @ hull.equations[:, :3].T + hull.equations[:, 3]).max(1)


def cut(points, limit=VERTICES):
    """At most limit of the hull vertices of points, spanning a polytope inside
    their hull. From the extreme points along DIRECTIONS, each round adds, for
    up to a quarter of the remaining budget, the vertex farthest outside the cut
    hull, one per facet, until every vertex is within TOLERANCE. Returns the
    vertices and the cut's error: how far the farthest full-hull vertex lies
    outside the cut hull, m."""
    vertices = points[ConvexHull(points).vertices]
    if len(vertices) <= limit:
        return vertices, 0.0
    with np.errstate(all="ignore"):
        along = vertices @ DIRECTIONS.T
    pick = [int(i) for i in np.unique(np.r_[along.argmin(0), along.argmax(0)])]
    rest = np.setdiff1d(np.arange(len(vertices)), pick)
    while len(pick) < limit and len(rest):
        hull = ConvexHull(vertices[pick])
        with np.errstate(all="ignore"):
            dist = vertices[rest] @ hull.equations[:, :3].T + hull.equations[:, 3]
        far = dist.max(1)
        # Once inside, a vertex stays inside: the cut hull only grows.
        keep = far > TOLERANCE
        rest, dist, far = rest[keep], dist[keep], far[keep]
        if not len(rest):
            break
        facet = dist.argmax(1)
        budget = max(1, (limit - len(pick)) // 4)
        chosen, seen = [], set()
        for i in np.argsort(-far):
            if facet[i] not in seen:
                seen.add(facet[i])
                chosen.append(int(i))
                if len(chosen) == budget:
                    break
        pick += [int(rest[i]) for i in chosen]
        rest = np.delete(rest, chosen)
    hull = ConvexHull(vertices[pick])
    error = max(0.0, float(outside(vertices, hull).max()))
    return vertices[pick][hull.vertices], error


def hulls(links):
    """{link: (cut hull vertices in the link frame, m; the cut's error, m)} for
    the links' STLs (robots/franzi/assets/urdf/meshes/<link>.STL)."""
    h = hashlib.sha256(f"{VERSION} {VERTICES}".encode())
    for link in sorted(links):
        h.update(link.encode())
        h.update((MESHES / f"{link}.STL").read_bytes())
    key = h.hexdigest()[:16]
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            return {
                link: (np.array(item["vertices"]), item["error_m"])
                for link, item in cached["hulls"].items()
            }
    made = {}
    for link in sorted(links):
        mesh = trimesh.load(MESHES / f"{link}.STL", process=True, force="mesh")
        made[link] = cut(np.asarray(mesh.vertices))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Episodes run in parallel: write a private file, then rename it in place.
    tmp = CACHE.with_name(f"{CACHE.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "key": key,
                "hulls": {
                    link: {"vertices": v.tolist(), "error_m": e}
                    for link, (v, e) in made.items()
                },
            }
        )
    )
    os.replace(tmp, CACHE)
    return made
