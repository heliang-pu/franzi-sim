"""Author a scene's USD stage from isaac/export.py's output, in Isaac Sim 5.1.

build(folder) writes folder/scene.usdc and folder/paths.json (the prim paths,
drive gains and joint couplings isaac/runtime.py needs). It needs pxr with the
PhysX schemas, so import it only inside a running Isaac Sim app
(runtime.start_app()).

How the MuJoCo model maps onto PhysX:
- A body that moves on its own (a hinge or slide joint, or a freejoint) becomes
  a rigid body; the bodies welded to it become child Xforms of it, their masses
  and inertias summed into it. A body with two joints (each finger's x-slide
  and its z drop) becomes a chain of two rigid bodies, one per joint. The wheel
  and steering joints are welded (export.LOCKED_SUFFIXES).
- Each tree with hinge or slide joints is one articulation, its root body
  (welded to the world in every scene) fixed to the world by a fixed joint.
  Free props without joints are plain rigid bodies; trees without joints are
  static colliders.
- Joints keep MuJoCo's anchor, axis and range, measured from qpos0. Actuated
  joints get position drives with the actuator's kp and the joint's damping
  plus the actuator's kv; joint springs become drives towards springref, joint
  damping a damping drive. The gripper linkage's coupled joints get stiff
  drives whose targets runtime.py sets from the finger slides every physics
  step: PhysX couples joints only linearly, the linkage's fits are quartic.
  frictionloss is not carried over: PhysX's joint friction is a coefficient of
  the joint force, not a constant force.
- Robot links ignore gravity. That is the gravity feedforward sim/episode.py
  applies on the robot's joints, here on the finger slides too.
- Contacts: MuJoCo's contype/conaffinity rule becomes one collision group per
  mask pair, filtered against the groups it would not collide with, and the
  contact exclusions become filtered pairs. Mesh geoms collide as convex hulls,
  as in MuJoCo (PhysX caps a hull at 255 vertices; 26 wrist pieces have more).
  Friction combines by the maximum, MuJoCo's rule for equal priorities; contact
  priority (tray_push's tray) has no PhysX counterpart.
- Visuals: the compiled meshes with their normals and texture coordinates, and
  UsdPreviewSurface materials from MuJoCo's colours (sRGB to linear) and 2D
  textures. MuJoCo hides geom groups 3 and up, so colliders there are guides.
- Cameras keep their body frame (MuJoCo and USD cameras both look along -Z with
  +Y up), their fovy and the 45 mm near plane. Lights become distant lights
  plus a dome.
"""

import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

# Drives of the linkage's coupled joints (SI, per rad or per m): stiff, so the
# links and pads follow the finger slides closely under grip loads.
COUPLED_GAINS = {"hinge": (50.0, 0.5), "slide": (20000.0, 100.0)}
CONTACT_OFFSET, REST_OFFSET = 0.004, 0.0
HULL_VERTEX_LIMIT = 255
# Solver iterations (position, velocity): the robot's as in the speed check.
ROBOT_ITERATIONS, PROP_ITERATIONS, BODY_ITERATIONS = (64, 4), (32, 4), (16, 4)
MAX_DEPENETRATION = 5.0  # m/s, every rigid body
MIN_MASS, MIN_INERTIA = 1e-3, 1e-7
# Distant-light intensity per unit of MuJoCo diffuse, and the dome standing in
# for MuJoCo's headlight and ambient terms, with a blue-grey sky. Chosen on five
# renders of tote_transfer (2026-10-25): at 7500 and 600 the floor and the
# pearl shell washed out to white.
LIGHT_PER_DIFFUSE, DOME_INTENSITY = 3000.0, 180.0
DOME_COLOR = (0.62, 0.70, 0.80)
# Each episode's light is drawn from its seed (user request 2026-10-27: random
# lighting for the benchmark data), so a seed always renders the same way and
# its settled re-run matches its frames. A model without a seed (the scene
# check) keeps the nominal light. The lamps' intensity and colour temperature,
# the dome's intensity, and the directional lights' azimuth and elevation.
RANDOM_LIGHT = True
LIGHT_SALT = 0x11647  # keeps the light's draws apart from the layout sampler's
LIGHT_SCALE, DOME_SCALE = (0.6, 1.4), (0.6, 1.5)
# 3500 K turned the darkest light amber and 4200 K beige (tote_transfer,
# 2026-10-27), while the RealSense colour cameras' auto white balance takes most
# of a light's tint out on the robot: only a mild warm-to-neutral range.
LIGHT_KELVIN = (5000.0, 6500.0)
LIGHT_TURN, LIGHT_TILT = math.radians(30.0), math.radians(10.0)
MIN_ELEVATION = math.radians(15.0)  # a directional light stays this far above the floor
FOCAL = 10.0  # camera focal length; only its ratio to the aperture matters
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)  # MuJoCo's geom rgba default


def rot(q):
    """MuJoCo quaternion (w, x, y, z) to a rotation matrix."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_of(R):
    """Rotation matrix to a unit (w, x, y, z), on its largest diagonal term."""
    R = np.asarray(R, float)
    t = np.trace(R)
    k = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))
    if k == 0:
        s = 2 * np.sqrt(1 + t)
        q = [
            s / 4,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    elif k == 1:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [
            (R[2, 1] - R[1, 2]) / s,
            s / 4,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ]
    elif k == 2:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            s / 4,
            (R[1, 2] + R[2, 1]) / s,
        ]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            s / 4,
        ]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


class Pose:
    """A rigid transform: x_parent = p + R x_child."""

    def __init__(self, p=(0.0, 0.0, 0.0), R=None):
        self.p = np.asarray(p, float)
        self.R = np.eye(3) if R is None else np.asarray(R, float)

    @classmethod
    def of(cls, pos, quat):
        return cls(pos, rot(quat))

    def __mul__(self, other):
        return Pose(self.p + self.R @ other.p, self.R @ other.R)

    def inv(self):
        return Pose(-self.R.T @ self.p, self.R.T)


def x_frame(axis):
    """A rotation whose x axis is axis: USD joints move along or about X."""
    x = np.asarray(axis, float) / np.linalg.norm(axis)
    helper = (0.0, 0.0, 1.0) if abs(x[2]) < 0.9 else (1.0, 0.0, 0.0)
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def turned(direction, turn, tilt):
    """A light's direction turned by `turn` about z and lowered or raised by
    `tilt` (rad), staying at least MIN_ELEVATION below the horizon."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    azimuth = math.atan2(d[1], d[0]) + turn
    down = math.asin(float(np.clip(-d[2], -1.0, 1.0))) + tilt
    down = min(max(down, MIN_ELEVATION), math.radians(89.0))
    return np.array(
        [
            math.cos(down) * math.cos(azimuth),
            math.cos(down) * math.sin(azimuth),
            -math.sin(down),
        ]
    )


def looking(forward, up=(0.0, 0.0, 1.0)):
    """Axes of a USD light or camera (looks along -Z, +Y up) facing forward."""
    f = np.asarray(forward, float) / np.linalg.norm(forward)
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(f, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(right, f), -f])


def linear(c):
    """sRGB to linear: MuJoCo shows rgba as given, RTX shades in linear."""
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def prim_name(name):
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if out and not out[0].isdigit() else "_" + out


def vec3f(v):
    return Gf.Vec3f(*map(float, v))


def quatf(R):
    w, x, y, z = quat_of(R)
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def place(prim, pose, scale=None):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*map(float, pose.p))
    )
    w, x, y, z = quat_of(pose.R)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(float(w), float(x), float(y), float(z))
    )
    if scale is not None:
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, scale)))


class Builder:
    def __init__(self, folder, model=None, arrays=None):
        """Reads folder/model.json and folder/arrays.npz unless given them."""
        self.folder = Path(folder)
        self.model = model or json.loads((self.folder / "model.json").read_text())
        self.arrays = (
            arrays if arrays is not None else np.load(self.folder / "arrays.npz")
        )
        self.bodies = self.model["bodies"]
        self.joints = self.model["joints"]
        self.geoms = self.model["geoms"]
        self.world = [Pose.of(b["xpos"], b["xquat"]) for b in self.bodies]
        self.usd = self.folder / "scene.usdc"
        if self.usd.exists():
            self.usd.unlink()
        self.stage = Usd.Stage.CreateNew(str(self.usd))
        self.meshes, self.looks, self.physics = {}, {}, {}
        self.colliders = {}  # (contype, conaffinity) -> collider paths
        self.body_colliders = {}  # body -> collider paths
        self.chain = {}  # body -> its rigid bodies, one per joint
        self.used = set()
        self.paths = {"articulations": {}, "free_bodies": {}, "dofs": {}, "bodies": {}}
        # collider prim -> [MuJoCo body id, geom name]: contacts map back to bodies
        self.paths["colliders"] = {}

    def unique(self, path):
        """path, or path_<k> if a sanitised name already took it."""
        path, k = Sdf.Path(path), 1
        candidate = path
        while candidate in self.used:
            candidate = path.GetParentPath().AppendChild(f"{path.name}_{k}")
            k += 1
        self.used.add(candidate)
        return candidate

    # ---------------------------------------------------------------- planning
    def plan(self):
        B, J = self.bodies, self.joints
        for j in J:
            if j["type"] == "ball":
                raise ValueError(f"ball joint {j['name']} is not ported")
        self.moving = [
            [
                j
                for j in b["joints"]
                if J[j]["type"] in ("hinge", "slide") and not J[j]["locked"]
            ]
            for b in B
        ]
        self.free = {
            i
            for i in range(1, len(B))
            if any(J[j]["type"] == "free" for j in B[i]["joints"])
        }
        self.jointed = {B[i]["root"] for i in range(1, len(B)) if self.moving[i]}
        # link[i]: the body whose rigid body carries body i, None if static.
        self.link = {}
        for i in range(1, len(B)):
            parent = B[i]["parent"]
            if self.moving[i] or i in self.free:
                self.link[i] = i
            elif parent:
                self.link[i] = self.link[parent]
            else:
                self.link[i] = i if i in self.jointed else None
        self.tree = {}
        for i in range(1, len(B)):
            if B[i]["parent"] == 0 and (i in self.jointed or i in self.free):
                name = "franzi" if B[i]["robot"] else prim_name(B[i]["name"])
                self.tree[i] = f"/World/{name}"

    def link_path(self, i):
        root = self.bodies[i]["root"]
        if root in self.jointed:
            return f"{self.tree[root]}/{prim_name(self.bodies[i]['name'])}"
        return self.tree[root]

    def body_path(self, i):
        if i == 0:
            return "/World"
        name = prim_name(self.bodies[i]["name"])
        if self.link[i] is None:
            return f"/World/static/{name}"
        return f"{self.link_path(self.link[i])}/{name}"

    # ------------------------------------------------------------------ stage
    def build(self):
        self.plan()
        stage = self.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        # Mesh data lives once under an abstract (class) prim, never rendered,
        # and every geom using a mesh references it.
        stage.CreateClassPrim("/MeshLibrary")
        self.physics_scene()
        for scope in ("static", "Looks", "PhysicsMaterials", "CollisionGroups"):
            UsdGeom.Scope.Define(stage, f"/World/{scope}")
        for root, path in self.tree.items():
            if root in self.jointed:
                UsdGeom.Xform.Define(stage, path)
                UsdGeom.Scope.Define(stage, f"{path}/joints")
        for i in range(1, len(self.bodies)):
            self.add_body(i)
        for i in sorted(self.jointed):
            self.add_articulation(i)
        self.collision_groups()
        self.filtered_pairs()
        self.lights()
        self.cameras()
        self.sites()
        stage.GetRootLayer().Save()
        (self.folder / "paths.json").write_text(json.dumps(self.paths, indent=1))
        return self.usd

    def physics_scene(self):
        o = self.model["options"]
        g = np.asarray(o["gravity"])
        scene = UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(vec3f(g / np.linalg.norm(g)))
        scene.CreateGravityMagnitudeAttr(float(np.linalg.norm(g)))
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateTimeStepsPerSecondAttr(round(1 / o["timestep"]))
        px.CreateSolverTypeAttr("TGS")
        px.CreateEnableGPUDynamicsAttr(False)
        px.CreateBroadphaseTypeAttr("MBP")
        px.CreateEnableCCDAttr(False)
        self.paths["physics_scene"] = "/World/PhysicsScene"
        self.paths["timestep"] = o["timestep"]

    def add_body(self, i):
        b = self.bodies[i]
        link = self.link[i]
        if link == i:
            moving = self.moving[i]
            path = self.link_path(i)
            # A body with two joints: an extra rigid body per joint before the
            # last, at the body's own pose (both joints are defined in it).
            chain = [
                f"{path}__{prim_name(self.joints[j]['name'])}" for j in moving[:-1]
            ]
            for extra in chain:
                self.add_link(extra, self.world[i], [], b["robot"])
            members = [k for k, v in self.link.items() if v == i]
            self.add_link(path, self.world[i], members, b["robot"])
            self.chain[i] = chain + [path]
            if i in self.free and i not in self.jointed:
                self.paths["free_bodies"][b["name"]] = path
        xform = UsdGeom.Xform.Define(self.stage, self.body_path(i)).GetPrim()
        if link is None:
            place(xform, self.world[i])
        else:
            place(xform, self.world[link].inv() * self.world[i])
            if b["robot"]:
                offset = self.world[link].inv() * self.world[i]
                self.paths["bodies"][b["name"]] = {
                    "link": self.link_path(link),
                    "pos": offset.p.tolist(),
                    "quat": quat_of(offset.R).tolist(),
                }
        for g, geom in enumerate(self.geoms):
            if geom["body"] == i:
                self.add_geom(g, xform.GetPath())

    def add_link(self, path, pose, members, robot):
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        place(prim, pose)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        total, com, diag, axes = self.mass_properties(members, pose)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(total))
        mass.CreateCenterOfMassAttr(vec3f(com))
        mass.CreateDiagonalInertiaAttr(vec3f(diag))
        mass.CreatePrincipalAxesAttr(quatf(axes))
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if robot:
            body.CreateDisableGravityAttr(True)
        body.CreateSolverPositionIterationCountAttr(BODY_ITERATIONS[0])
        body.CreateSolverVelocityIterationCountAttr(BODY_ITERATIONS[1])
        # PhysX leaves it unlimited, which can fling a body that ends up deeply
        # interpenetrated; RoboDojo (Isaac Lab) caps it at 5 m/s. Resting and
        # grasp contacts (penetrations of microns) never come near it.
        body.CreateMaxDepenetrationVelocityAttr(MAX_DEPENETRATION)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        return prim

    def mass_properties(self, members, link_pose):
        total, first, parts = 0.0, np.zeros(3), []
        for i in members:
            b = self.bodies[i]
            if b["mass"] <= 0:
                continue
            T = link_pose.inv() * self.world[i]
            c = T.p + T.R @ np.asarray(b["ipos"])
            R = T.R @ rot(b["iquat"])
            parts.append((b["mass"], c, R @ np.diag(b["inertia"]) @ R.T))
            total += b["mass"]
            first += b["mass"] * c
        if total <= 0:
            return MIN_MASS, np.zeros(3), np.full(3, MIN_INERTIA), np.eye(3)
        com = first / total
        inertia = np.zeros((3, 3))
        for m, c, tensor in parts:
            r = c - com
            inertia += tensor + m * (r @ r * np.eye(3) - np.outer(r, r))
        w, V = np.linalg.eigh(inertia)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        return total, com, np.maximum(w, MIN_INERTIA), V

    # ------------------------------------------------------------------ geoms
    def add_geom(self, g, parent):
        geom = self.geoms[g]
        collides = bool(geom["contype"] or geom["conaffinity"])
        visible = geom["group"] < 3
        if not (collides or visible):
            return
        path = self.unique(parent.AppendChild(prim_name(geom["name"] or f"geom{g}")))
        kind, size = geom["type"], geom["size"]
        pose, scale = Pose.of(geom["pos"], geom["quat"]), None
        stage = self.stage
        if kind == "mesh":
            prim = UsdGeom.Mesh.Define(stage, path).GetPrim()
            prim.GetReferences().AddInternalReference(
                self.mesh_class(geom["mesh"], visible)
            )
        elif kind in ("box", "plane"):
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(2.0)
            prim = cube.GetPrim()
            scale = size
            if kind == "plane":  # a 5 cm slab under the plane, 50 m if infinite
                scale = [size[0] or 50.0, size[1] or 50.0, 0.025]
                pose = pose * Pose((0.0, 0.0, -0.025))
        elif kind == "sphere":
            prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
            UsdGeom.Sphere(prim).CreateRadiusAttr(size[0])
        elif kind in ("capsule", "cylinder"):
            shape = (UsdGeom.Capsule if kind == "capsule" else UsdGeom.Cylinder).Define(
                stage, path
            )
            shape.CreateRadiusAttr(size[0])
            shape.CreateHeightAttr(2 * size[1])
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
            prim = shape.GetPrim()
        elif kind == "ellipsoid" and not collides:
            prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
            UsdGeom.Sphere(prim).CreateRadiusAttr(1.0)
            scale = size
        else:
            raise ValueError(f"geom {geom['name']}: {kind} is not ported")
        place(prim, pose, scale)
        if collides:
            self.add_collider(prim, geom, kind, g)
        if visible:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.look(geom))
        else:
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)

    def mesh_class(self, k, visual):
        key = (k, visual)
        if key in self.meshes:
            return self.meshes[key]
        path = f"/MeshLibrary/{'v' if visual else 'c'}{k:04d}"
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        a = self.arrays
        v0, nv = int(a["mesh_vertadr"][k]), int(a["mesh_vertnum"][k])
        f0, nf = int(a["mesh_faceadr"][k]), int(a["mesh_facenum"][k])
        points = np.ascontiguousarray(a["mesh_vert"][v0 : v0 + nv], np.float32)
        faces = np.ascontiguousarray(a["mesh_face"][f0 : f0 + nf], np.int32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(nf, 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
        mesh.CreateExtentAttr(
            Vt.Vec3fArray([vec3f(points.min(0)), vec3f(points.max(0))])
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if visual:
            n0, nn = int(a["mesh_normaladr"][k]), int(a["mesh_normalnum"][k])
            normals = a["mesh_normal"][n0 : n0 + nn]
            corner = a["mesh_facenormal"][f0 : f0 + nf]
            if nn == nv and np.array_equal(corner, faces):
                mesh.CreateNormalsAttr(
                    Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, np.float32))
                )
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            else:
                flat = np.ascontiguousarray(normals[corner.ravel()], np.float32)
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(flat))
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            t0, nt = int(a["mesh_texcoordadr"][k]), int(a["mesh_texcoordnum"][k])
            if nt > 0:
                corner = a["mesh_facetexcoord"][f0 : f0 + nf]
                st = np.ascontiguousarray(
                    a["mesh_texcoord"][t0 : t0 + nt][corner.ravel()], np.float32
                )
                UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                ).Set(Vt.Vec2fArray.FromNumpy(st))
        self.meshes[key] = path
        return path

    def add_collider(self, prim, geom, kind, g):
        UsdPhysics.CollisionAPI.Apply(prim)
        if kind == "mesh":
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                UsdPhysics.Tokens.convexHull
            )
            hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            hull.CreateHullVertexLimitAttr(HULL_VERTEX_LIMIT)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(CONTACT_OFFSET)
        px.CreateRestOffsetAttr(REST_OFFSET)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self.physics_material(geom["friction"]),
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
        path = prim.GetPath()
        self.colliders.setdefault((geom["contype"], geom["conaffinity"]), []).append(
            path
        )
        self.body_colliders.setdefault(geom["body"], []).append(path)
        # The geom's name, or for an unnamed one its prim name, which
        # isaac/engine.py maps back to the geom (the contact audit).
        self.paths["colliders"][str(path)] = [geom["body"], geom["name"] or f"geom{g}"]

    def physics_material(self, friction):
        key = round(float(friction), 4)
        if key not in self.physics:
            path = f"/World/PhysicsMaterials/mu_{round(key * 1000):04d}"
            material = UsdShade.Material.Define(self.stage, path)
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(key)
            api.CreateDynamicFrictionAttr(key)
            api.CreateRestitutionAttr(0.0)
            px = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            px.CreateFrictionCombineModeAttr("max")
            px.CreateRestitutionCombineModeAttr("min")
            self.physics[key] = material
        return self.physics[key]

    def look(self, geom):
        """The render material: MuJoCo's material, unless the geom's own rgba
        differs from the default, which then takes precedence (as in MuJoCo)."""
        rgba, texture, shininess, emission = geom["rgba"], None, 0.5, 0.0
        if geom["material"] >= 0:
            m = self.model["materials"][geom["material"]]
            if np.allclose(rgba, DEFAULT_RGBA):
                rgba = m["rgba"]
            if m["texture"] >= 0:
                texture = self.model["textures"][m["texture"]].get("file")
            shininess, emission = m["shininess"], m["emission"]
        key = (
            tuple(np.round(rgba, 4)),
            texture,
            round(shininess, 3),
            round(emission, 3),
        )
        if key in self.looks:
            return self.looks[key]
        path = f"/World/Looks/look_{len(self.looks):04d}"
        material = UsdShade.Material.Define(self.stage, path)
        surface = UsdShade.Shader.Define(self.stage, f"{path}/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        color = linear(rgba[:3])
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(0.9 - 0.8 * shininess)
        )
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if rgba[3] < 1:
            surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
        # MuJoCo adds emission times the surface colour, the texture's where
        # there is one: a white textured lamp glows in its texture's colour.
        channels = {"diffuseColor": color}
        if emission > 0:
            channels["emissiveColor"] = color * emission
        for channel, scale in channels.items():
            target = surface.CreateInput(channel, Sdf.ValueTypeNames.Color3f)
            if not texture:
                target.Set(vec3f(scale))
                continue
            reader = UsdShade.Shader.Get(self.stage, f"{path}/UV")
            if not reader:
                reader = UsdShade.Shader.Define(self.stage, f"{path}/UV")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            image = UsdShade.Shader.Define(self.stage, f"{path}/{channel}Texture")
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{texture}")
            image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            image.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(*map(float, scale), 1.0)
            )
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            target.ConnectToSource(image.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(
            surface.ConnectableAPI(), "surface"
        )
        self.looks[key] = material
        return material

    # ----------------------------------------------------------------- joints
    def add_articulation(self, root):
        B, J = self.bodies, self.joints
        tree = self.tree[root]
        prim = self.stage.GetPrimAtPath(tree)
        robot = B[root]["robot"]
        if self.moving[root]:
            raise ValueError(
                f"{B[root]['name']}: a tree root jointed to the world is not ported"
            )
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        art.CreateEnabledSelfCollisionsAttr(robot)
        iterations = ROBOT_ITERATIONS if robot else PROP_ITERATIONS
        art.CreateSolverPositionIterationCountAttr(iterations[0])
        art.CreateSolverVelocityIterationCountAttr(iterations[1])
        art.CreateSleepThresholdAttr(0.0)
        art.CreateStabilizationThresholdAttr(0.0)
        self.paths["articulations"][B[root]["name"]] = tree
        if robot:
            self.paths["robot"] = tree
        if root not in self.free:
            fixed = UsdPhysics.FixedJoint.Define(
                self.stage, f"{tree}/joints/root_fixed"
            )
            fixed.CreateBody1Rel().SetTargets([Sdf.Path(self.link_path(root))])
            fixed.CreateLocalPos0Attr(vec3f(self.world[root].p))
            fixed.CreateLocalRot0Attr(quatf(self.world[root].R))
            fixed.CreateLocalPos1Attr(vec3f((0, 0, 0)))
            fixed.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        actuators = {a["joint"]: a for a in self.model["actuators"]}
        couplings = {c["joint1"]: c for c in self.model["couplings"]}
        for i in range(1, len(B)):
            if B[i]["root"] != root or not self.moving[i]:
                continue
            parent = self.link[B[i]["parent"]]
            parent_path = self.link_path(parent)
            parent_pose = self.world[parent]
            for k, j in enumerate(self.moving[i]):
                child_path = self.chain[i][k]
                path = f"{tree}/joints/{prim_name(J[j]['name'])}"
                joint = self.add_joint(
                    j, path, parent_path, parent_pose, child_path, self.world[i]
                )
                self.add_drive(joint, J[j], actuators.get(j), couplings.get(j), robot)
                # The next joint of a two-joint body starts from this one's body.
                parent_path, parent_pose = child_path, self.world[i]

    def add_joint(self, j, path, parent_path, parent_pose, child_path, child_pose):
        J = self.joints[j]
        frame = self.world[J["body"]] * Pose(J["pos"], x_frame(J["axis"]))
        local0 = parent_pose.inv() * frame if parent_pose is not None else frame
        local1 = child_pose.inv() * frame
        hinge = J["type"] == "hinge"
        joint = (
            UsdPhysics.RevoluteJoint if hinge else UsdPhysics.PrismaticJoint
        ).Define(self.stage, path)
        joint.CreateAxisAttr("X")
        if parent_path:
            joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
        joint.CreateLocalPos0Attr(vec3f(local0.p))
        joint.CreateLocalRot0Attr(quatf(local0.R))
        joint.CreateLocalPos1Attr(vec3f(local1.p))
        joint.CreateLocalRot1Attr(quatf(local1.R))
        if J["limited"]:
            lo, hi = (r - J["qpos0"][0] for r in J["range"])
            if hinge:
                lo, hi = math.degrees(lo), math.degrees(hi)
            joint.CreateLowerLimitAttr(float(lo))
            joint.CreateUpperLimitAttr(float(hi))
        PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateArmatureAttr(
            float(J["armature"])
        )
        return joint

    def add_drive(self, joint, J, actuator, coupling, robot):
        """Drive gains in SI (per rad or per m), recorded in paths.json for
        runtime.py, and authored in USD's per-degree units for angular drives."""
        hinge = J["type"] == "hinge"
        target, max_force = 0.0, None
        if actuator is not None:
            stiffness = actuator["gain"]
            damping = J["damping"] + max(0.0, -actuator["bias"][2])
            if actuator["forcerange"]:
                max_force = max(abs(f) for f in actuator["forcerange"])
        elif coupling is not None:
            stiffness, damping = COUPLED_GAINS[J["type"]]
        else:
            stiffness, damping = J["stiffness"], J["damping"]
            target = J["springref"] - J["qpos0"][0]
        dof = {
            "prim": joint.GetPrim().GetName(),
            "type": J["type"],
            "qpos0": J["qpos0"][0],
            "stiffness": stiffness,
            "damping": damping,
            "max_force": max_force,
            "target": target,
            "robot": robot,
        }
        if coupling is not None:
            dof["leader"] = self.joints[coupling["joint2"]]["name"]
            dof["polycoef"] = coupling["polycoef"]
        self.paths["dofs"][J["name"]] = dof
        if stiffness == 0 and damping == 0:
            return
        scale = math.pi / 180 if hinge else 1.0
        drive = UsdPhysics.DriveAPI.Apply(
            joint.GetPrim(), "angular" if hinge else "linear"
        )
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(stiffness * scale))
        drive.CreateDampingAttr(float(damping * scale))
        drive.CreateTargetPositionAttr(float(math.degrees(target) if hinge else target))
        if max_force is not None:
            drive.CreateMaxForceAttr(float(max_force))

    # -------------------------------------------------------------- contacts
    def collision_groups(self):
        """One group per (contype, conaffinity), filtered against every group
        MuJoCo's rule would not let it touch (itself included)."""
        groups = {}
        for (contype, conaffinity), members in sorted(self.colliders.items()):
            group = UsdPhysics.CollisionGroup.Define(
                self.stage, f"/World/CollisionGroups/mask_{contype}_{conaffinity}"
            )
            includes = group.GetCollidersCollectionAPI().CreateIncludesRel()
            for path in members:
                includes.AddTarget(path)
            groups[(contype, conaffinity)] = group
        for a, group in groups.items():
            filtered = group.CreateFilteredGroupsRel()
            for b, other in groups.items():
                if not (a[0] & b[1] or b[0] & a[1]):
                    filtered.AddTarget(other.GetPath())

    def filtered_pairs(self):
        def targets(i):
            if self.link.get(i) is not None:
                return [Sdf.Path(self.link_path(self.link[i]))]
            return self.body_colliders.get(i, [])

        for b1, b2 in self.model["excludes"]:
            first, second = targets(b1), targets(b2)
            if not first or not second or first == second:
                continue
            for path in first:
                prim = self.stage.GetPrimAtPath(path)
                rel = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                for other in second:
                    rel.AddTarget(other)

    # ------------------------------------------------------ lights, cameras
    def lighting(self):
        """This episode's light (see RANDOM_LIGHT), also written to
        lighting.json: intensity scales, colour temperature (None: the
        lights' own white) and the directional lights' turn and tilt (rad)."""
        seed = (self.model.get("task") or {}).get("seed")
        look = {"scale": 1.0, "dome": 1.0, "kelvin": None, "turn": 0.0, "tilt": 0.0}
        if RANDOM_LIGHT and seed is not None:
            rng = np.random.default_rng([int(seed), LIGHT_SALT])
            look = {
                "scale": float(rng.uniform(*LIGHT_SCALE)),
                "dome": float(rng.uniform(*DOME_SCALE)),
                "kelvin": float(rng.uniform(*LIGHT_KELVIN)),
                "turn": float(rng.uniform(-LIGHT_TURN, LIGHT_TURN)),
                "tilt": float(rng.uniform(-LIGHT_TILT, LIGHT_TILT)),
            }
        (self.folder / "lighting.json").write_text(json.dumps(look))
        return look

    def lights(self):
        UsdGeom.Scope.Define(self.stage, "/World/Lights")
        look = self.lighting()
        for light in self.model["lights"]:
            if not light["active"]:
                continue
            path = f"/World/Lights/{prim_name(light['name'])}"
            intensity = LIGHT_PER_DIFFUSE * float(np.mean(light["diffuse"]))
            direction = np.asarray(light["xdir"], float)
            if light["directional"]:
                prim = UsdLux.DistantLight.Define(self.stage, path)
                prim.CreateAngleAttr(0.53)
                if look["turn"] or look["tilt"]:
                    direction = turned(direction, look["turn"], look["tilt"])
            else:
                prim = UsdLux.SphereLight.Define(self.stage, path)
                prim.CreateRadiusAttr(0.05)
                shaping = UsdLux.ShapingAPI.Apply(prim.GetPrim())
                shaping.CreateShapingConeAngleAttr(float(light["cutoff"]))
            prim.CreateIntensityAttr(intensity * look["scale"])
            if look["kelvin"] is not None:
                prim.CreateEnableColorTemperatureAttr(True)
                prim.CreateColorTemperatureAttr(look["kelvin"])
            UsdLux.ShadowAPI.Apply(prim.GetPrim()).CreateShadowEnableAttr(
                light["castshadow"]
            )
            place(prim.GetPrim(), Pose(light["xpos"], looking(direction)))
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/dome")
        dome.CreateIntensityAttr(DOME_INTENSITY * look["dome"])
        dome.CreateColorAttr(vec3f(DOME_COLOR))

    def cameras(self):
        znear = self.model["options"]["znear"]
        UsdGeom.Scope.Define(self.stage, "/World/Cameras")
        self.paths["cameras"] = {}
        for cam in self.model["cameras"]:
            parent = (
                "/World/Cameras" if cam["body"] == 0 else self.body_path(cam["body"])
            )
            path = f"{parent}/{prim_name(cam['name'])}"
            camera = UsdGeom.Camera.Define(self.stage, path)
            place(camera.GetPrim(), Pose.of(cam["pos"], cam["quat"]))
            vertical = 2 * FOCAL * math.tan(math.radians(cam["fovy"]) / 2)
            camera.CreateFocalLengthAttr(FOCAL)
            camera.CreateVerticalApertureAttr(float(vertical))
            camera.CreateHorizontalApertureAttr(float(vertical * 4 / 3))
            camera.CreateClippingRangeAttr(Gf.Vec2f(float(znear), 100.0))
            self.paths["cameras"][cam["name"]] = path

    def sites(self):
        self.paths["sites"] = {}
        for site in self.model["sites"]:
            path = f"{self.body_path(site['body'])}/site_{prim_name(site['name'])}"
            place(
                UsdGeom.Xform.Define(self.stage, path).GetPrim(),
                Pose.of(site["pos"], site["quat"]),
            )
            self.paths["sites"][site["name"]] = path


def build(folder, model=None, arrays=None):
    """Write folder/scene.usdc and folder/paths.json; returns the USD path."""
    return Builder(folder, model, arrays).build()
