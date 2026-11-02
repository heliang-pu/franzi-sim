"""Build section 6.2 electronics props, without mutating the task 2 library.

Usage: blender -b --factory-startup -t 6 -P tools/blender/build_task15_electronics.py
       -- [--props ID ...] [--geometry-only | --export-only | --render-exports]
All construction dimensions are millimetres; shared prop-frame exports are metres.
Artwork workers run in PROP_PYTHON and use only the bundled DejaVu Sans fonts.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
OUT = ROOT / "assets/props"
QA = HERE / "_qa/task15"
IDS = ("plc_module", "module_tray", "test_fixture", "ng_bin", "product_box",
       "accessory_box", "stack_light", "line_console")


def qr_matrix(text: str) -> list[list[bool]]:
    """Version 1-L byte-mode QR, mask zero, for a short fictional model ID."""
    payload = text.encode("ascii")
    if len(payload) > 17:
        raise ValueError("Version 1-L payload too long")
    bits = [0, 1, 0, 0] + [int(v) for v in f"{len(payload):08b}"]
    bits += [int(v) for byte in payload for v in f"{byte:08b}"]
    bits += [0] * min(4, 152-len(bits))
    bits += [0] * (-len(bits) % 8)
    data = [sum(bits[i+j] << (7-j) for j in range(8)) for i in range(0,len(bits),8)]
    while len(data)<19: data.append(0xEC if (len(data)-(len(bits)//8))%2==0 else 0x11)
    def mul(a,b):
        result=0
        while b:
            if b&1: result ^= a
            a <<= 1
            if a&256: a ^= 0x11D
            b >>= 1
        return result
    generator=[1]; root=1
    for _ in range(7):
        nxt=[0]*(len(generator)+1)
        for i,c in enumerate(generator):
            nxt[i] ^= c; nxt[i+1] ^= mul(c,root)
        generator=nxt; root=mul(root,2)
    rem=data+[0]*7
    for i in range(19):
        factor=rem[i]
        for j,c in enumerate(generator): rem[i+j] ^= mul(c,factor)
    bits=[int(v) for byte in data+rem[-7:] for v in f"{byte:08b}"]
    grid=[[False]*21 for _ in range(21)]; fixed=[[False]*21 for _ in range(21)]
    def setcell(x,y,value):
        if 0<=x<21 and 0<=y<21: grid[y][x]=bool(value);fixed[y][x]=True
    for cx,cy in ((3,3),(17,3),(3,17)):
        for dy in range(-4,5):
            for dx in range(-4,5): setcell(cx+dx,cy+dy,max(abs(dx),abs(dy)) not in (2,4))
    for i in range(8,13): setcell(6,i,i%2==0);setcell(i,6,i%2==0)
    fmt=0x77C4
    for i in range(6): setcell(8,i,(fmt>>i)&1)
    setcell(8,7,(fmt>>6)&1);setcell(8,8,(fmt>>7)&1);setcell(7,8,(fmt>>8)&1)
    for i in range(9,15): setcell(14-i,8,(fmt>>i)&1)
    for i in range(8): setcell(20-i,8,(fmt>>i)&1)
    for i in range(8,15): setcell(8,6+i,(fmt>>i)&1)
    setcell(8,13,True)
    k=0;right=20;up=True
    while right>=1:
        if right==6: right=5
        for y in (range(20,-1,-1) if up else range(21)):
            for x in (right,right-1):
                if not fixed[y][x]:
                    value=bits[k] if k<len(bits) else 0
                    grid[y][x]=bool(value ^ ((x+y)%2==0));k+=1
        right-=2;up=not up
    return grid


def artwork_worker() -> None:
    """Raster art with explicit physical-size typography; no platform fonts."""
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo
    p = json.load(sys.stdin)
    w, h = p["size"]
    im = Image.new("RGB", (w, h), p.get("background", "#F2F2F0"))
    d = ImageDraw.Draw(im)
    ink = p.get("foreground", "#173F77")
    def text(s, x, y, height, fill=ink, max_width=None):
        font = ImageFont.truetype(str(HERE / "fonts/DejaVuSans-Bold.ttf"), round(height * 1.4))
        b = font.getbbox(s)
        font = ImageFont.truetype(str(HERE / "fonts/DejaVuSans-Bold.ttf"),
                                 max(1, round(font.size * height / (b[3] - b[1]))))
        if max_width and font.getlength(s) > max_width:
            font = ImageFont.truetype(str(HERE / "fonts/DejaVuSans-Bold.ttf"),
                                     max(1, int(font.size * max_width / font.getlength(s))))
        d.text((x, y), s, font=font, fill=fill, anchor="mt")
    kind = p.get("kind", "label")
    if kind == "plc":
        color, model = p["color"], p["model"]
        d.rectangle((0, 0, w * .075, h), fill=color)
        # 90 x 65 mm label: glyph height 12 mm. Split the long brand to fit.
        for s, y in (("IMC", .035), ("Controls", .25), (model, .475)):
            text(s, w * .50, h * y, h * 12.1 / 64)
        text("24 VDC  |  I/O MODULE", w * .43, h * .735, h * .047, max_width=w * .7)
        text("IMC-15  /  ESD SAFE", w * .43, h * .825, h * .043, max_width=w * .7)
        qr = Image.new("RGB", (29, 29), "white")
        for y,row in enumerate(qr_matrix(model)):
            for x,value in enumerate(row):
                if value: qr.putpixel((x+4,y+4),(0,0,0))
        qr = qr.resize((round(h * .23), round(h * .23)), Image.Resampling.NEAREST)
        im.paste(qr, (w - qr.width - 8, h - qr.height - 8))
    elif kind == "front":
        d.rectangle((0, 0, w, h * 8 / 29), fill=p["color"])
        for i in range(16):
            x = w * (.07 + (i % 8) * .12)
            y = h * (.43 + (i // 8) * .31)
            d.rounded_rectangle((x, y, x+w*.035, y+h*.12), 3,
                                fill="#267144" if i < 3 else "#52605A")
    elif kind == "vents":
        for i in range(11):
            x = w * (.07 + i * .082)
            d.rounded_rectangle((x, h*.22, x+w*.035, h*.78), 5, fill="#737B80")
    else:
        lines = p.get("lines", [])
        for i, s in enumerate(lines):
            text(s, w/2, h * (.12 + i * .72 / max(len(lines), 1)),
                 (h * (.70 if len(lines)==1 else .48) / max(len(lines), 1)), max_width=w * .94)
        if p.get("barcode"):
            import random
            rng = random.Random(15)
            x = int(w*.08)
            while x < w*.92:
                bw = rng.randint(1, 3) * max(1, w//260)
                d.rectangle((x, h*.76, x+bw, h*.96), fill=ink)
                x += bw + rng.randint(1, 3) * max(1, w//260)
    info = PngInfo()
    info.add(b"sRGB", b"\x00")
    im.save(p["path"], pnginfo=info)


if __name__ == "__main__" and "--artwork-worker" in sys.argv:
    artwork_worker()
    raise SystemExit(0)

import bpy
import bmesh
from mathutils import Matrix, Vector
import prop_lib as props
import build_task02_panel_operation as console

PLASTIC = {"specular": .3, "shininess": .4, "reflectance": 0.0}
METAL = {"specular": .7, "shininess": .7, "reflectance": .05}
PAPER = {"specular": .05, "shininess": .1, "reflectance": 0.0}
TOP = Matrix(((0, 1, 0), (-1, 0, 0), (0, 0, 1)))
FRONT = Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0)))
COMMON = ("Metres, Z up, front -X. Applied transforms in the shared prop frame. "
          "Cycles CPU 64-sample diffuse COLOR times (0.4 + 0.6 AO), sRGB8 albedo; "
          "PBR and artwork packed in source.blend. DejaVu Sans Bold, English only. "
          "Task 15 uses isolated finalization to protect existing props. ")


def log(*args: object) -> None:
    print("TASK15", *args, flush=True)


def start(pid: str) -> Path:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    props.reset_scene()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.context.scene.render.threads_mode = "FIXED"
    bpy.context.scene.render.threads = 6
    folder = OUT / pid
    folder.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    log("START", pid)
    return folder


def mat(name: str, color: str, rough: float = .55, metallic: float = 0) -> bpy.types.Material:
    return props.pbr_material(name, color, rough, metallic, wear=.025, grime=.12,
                              machining="brushed" if metallic else None)


def box(name, size, pos, material=None, edge=.4):
    return props.box(name, size, pos, bevel_mm=edge, segments=3, material=material)


def cyl(name, r, h, pos, material=None, edge=.4, n=48):
    return props.cylinder(name, r, h, pos, bevel_mm=edge, segments=3, vertices=n, material=material)


def merge(items, name):
    return props.join_objects(items, name) if len(items) > 1 else items[0]


def union(items, name):
    obj = items[0]
    for other in items[1:]:
        console.boolean(obj, other, "UNION")
    obj.name = name
    return obj


def label(folder, stem, size, pos, lines=(), background="#F2F2F0", foreground="#173F77",
          rotation=TOP, **extra):
    """A shallow printed patch; source UV is separate from the baked atlas."""
    path = folder / (stem + "_print.png")
    payload = {"path": str(path), "size": [1200, max(120, round(1200*size[1]/size[0]))],
               "lines": list(lines), "background": background, "foreground": foreground, **extra}
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(Path(__file__)), "--artwork-worker"],
                   input=json.dumps(payload), text=True, check=True)
    material = props.pbr_material(stem + " print", background, .65, 0)
    tree = material.node_tree
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path), check_existing=False)
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "PrintUV"
    tree.links.new(uv.outputs["UV"], image.inputs["Vector"])
    tree.links.new(image.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
    obj = box(stem, (*size, .04), (0, 0, 0), material, edge=0)
    console.print_uv(obj, *size)
    console.transform(obj, pos, rotation)
    return obj, image


def part(name, visual, collisions, mass, parent=None, joint=None, rgba=None, friction=.5,
         material=None, size=2048):
    visual = merge(visual if isinstance(visual, list) else [visual], name)
    visual.name = name
    console.clean(visual)
    props.uv_unwrap(visual)
    for i, obj in enumerate(collisions):
        obj.name = f"COL__{name}__{i:02d}"
        obj.hide_render = True
        obj.hide_set(True)
    rec = {"name": name, "visual": name+".obj", "texture": name+"_albedo.png" if rgba is None else None,
           "texture_px": size if rgba is None else None, "collision": [], "mass_kg": mass,
           "friction": friction, "mujoco_material": dict(material or PLASTIC),
           "parent": parent, "joint": joint, "source_collision_objects": [o.name for o in collisions]}
    if rgba is not None:
        rec["rgba"] = rgba
        rec["emission"] = 0.0
    return rec


def preview(objects, folder, name="preview.png"):
    """Render at the binding 1024 size; retain it alongside delivery previews."""
    def settings(scene, *_):
        scene.render.resolution_x = scene.render.resolution_y = 1024
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 6
    bpy.app.handlers.render_init.append(settings)
    try:
        for obj in objects:
            obj.hide_render = False
            obj.hide_set(False)
        props.render_preview(objects, folder / name)
        shutil.copyfile(folder / name, folder / (name.replace("preview", "render1024")))
    finally:
        bpy.app.handlers.render_init.remove(settings)


def finish(folder, parts, dimensions, notes="", variants=None, static=False, **extra):
    objects = [bpy.data.objects[p["name"]] for p in parts]
    data = {"id": folder.name, "task": "electronics_line", "units": "m", "origin": "bottom_center",
            "bbox_mm": props.bbox_mm(objects), "triangles": props.triangle_count(objects),
            "budget_class": "small" if folder.name in ("plc_module", "accessory_box") else "medium",
            "static": static, "dimensions_mm": dimensions, "parts": parts,
            "recommended_instance_color_gain_range": [.97, 1.03],
            "variants": variants or [], "notes": COMMON + notes, **extra}
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in objects
            obj.hide_set(obj not in objects)
    bpy.context.scene["task15_metadata"] = json.dumps(data)
    props._select(objects)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "source.blend"), check_existing=False)
    if not ARGS.geometry_only:
        export_current(folder)
    log("GEOMETRY", folder.name, data["bbox_mm"], data["triangles"])


def export_current(folder: Path) -> None:
    data = json.loads(bpy.context.scene["task15_metadata"])
    objects = [bpy.data.objects[p["name"]] for p in data["parts"]]
    for p, obj in zip(data["parts"], objects):
        log("EXPORT", folder.name, p["name"])
        if p["texture"]:
            console.isolated_bake(obj, p["texture_px"], folder / p["texture"])
        props.export_obj(obj, folder / p["visual"])
        p["collision"] = props.export_collision(p["name"],
            [bpy.data.objects[n] for n in p["source_collision_objects"]], folder)
    for variant in data["variants"]:
        source_changes = variant.pop("source_changes", [])
        if not source_changes:
            continue
        saved = []
        for change in source_changes:
            material = bpy.data.materials[change["material"]]
            if "image" in change:
                node = next(n for n in material.node_tree.nodes if n.type == "TEX_IMAGE")
                saved.append((node, "image", node.image))
                node.image = bpy.data.images.load(str(folder / change["image"]), check_existing=False)
            else:
                node = next(n for n in material.node_tree.nodes if n.type == "RGB")
                saved.append((node.outputs[0], "default_value", tuple(node.outputs[0].default_value)))
                node.outputs[0].default_value = props._linear_color(change["color"])
        for pname, texture in variant["textures"].items():
            record = next(p for p in data["parts"] if p["name"] == pname)
            console.isolated_bake(bpy.data.objects[pname], record["texture_px"], folder / texture)
        preview(objects, folder, "preview_" + variant["name"] + ".png")
        for obj, attribute, value in saved:
            setattr(obj, attribute, value)
    preview(objects, folder)
    if folder.name == "product_box":
        lid = bpy.data.objects["lid"]
        anchor = Vector(data["parts"][1]["joint"]["anchor_m"])
        lid.matrix_world = Matrix.Translation(anchor) @ Matrix.Rotation(1.75, 4, "Y") @ Matrix.Translation(-anchor)
        preview(objects, folder, "preview_open.png")
        lid.matrix_world = Matrix.Identity(4)
    props.write_prop_json(data, folder / "prop.json")


def button(name, pos, color, stroke, parent, rotation=Matrix.Identity(3), emergency=False):
    material = console.hand_material(name + " handled cap", color)
    # Same stepped mushroom construction as the library; full-travel clearance.
    obj = console.revolve(name, [(0, 3), (7, 3), (7, 8), (18.8, 8), (20, 9),
                                (20, 23), (19.5, 25), (18, 26), (0, 26)], material, sectors=48)
    col = cyl(name+" head proxy", 19.7, 18, (0, 0, 17), edge=0, n=32)
    console.transform(obj, pos, rotation)
    console.transform(col, pos, rotation)
    axis = list(rotation @ Vector((0, 0, -1)))
    joint = {"type": "slide", "parent": parent, "axis": axis, "anchor_m": list(Vector(pos)/1000),
             "range": [0.0, stroke/1000], "damping": .6, "frictionloss": .08,
             "spring": {"stiffness": 400.0, "reference": 0.0}}
    if emergency:
        joint["latch"] = {"engage_at_m": .0048, "hold_at_m": .005,
                          "implementation": "Simulator state machine overrides spring after latching"}
    record = part(name, obj, [col], .01, parent, joint, friction=.8, size=1024)
    meta = {"part": name, "press_direction": axis, "travel_m": stroke/1000,
            "stroke_m": stroke/1000, "cap_diameter_mm": 40, "activation_m": stroke*.8/1000}
    return record, meta


def plc_module() -> None:
    f = start("plc_module")
    shell = mat("PC light grey", "#D9DDE0")
    dark = mat("Removable terminal charcoal", "#3A3D40")
    body = box("PC housing", (65, 90, 30), (0, 0, 15), shell)
    # DIN recess stays outside the central 50 mm jaw contact region.
    for y in (-37, 37):
        console.boolean(body, box("DIN latch rebate", (4, 9, 19), (32, y, 15), edge=0))
    console.boolean(body, box("Terminal seating recess", (2.1,12.2,21.2), (-31.6,37.5,15), edge=0))
    items = [body]
    # Thin terminal/flip-label representations at the ends preserve both grip planes.
    items += [box("Terminal block", (1.8, 12, 21), (-31.6, 37.5, 15), dark, .3)]
    top, node = label(f, "top_DI", (89, 64), (0, 0, 29.99), kind="plc",
                       color="#1565C0", model="DI16", background="#D9DDE0")
    front, front_node = label(f, "front_DI", (66, 29), (-32.49, -7, 15), rotation=FRONT,
                              kind="front", color="#1565C0", background="#D9DDE0")
    items += [top, front]
    for side in (-1, 1):
        rot = Matrix(((side, 0, 0), (0, 0, side), (0, -1, 0)))
        vent, _ = label(f, "vents_"+str(side), (52, 20), (0, side*44.99, 15), rotation=rot,
                        kind="vents", background="#D9DDE0")
        items.append(vent)
    flip, _ = label(f, "flip_tag", (11, 9), (-32.49, 37.5, 23), ["I/O"], rotation=FRONT)
    items.append(flip)
    variants = [{"name": "DI", "model": "DI16", "color": "#1565C0", "textures": {"body":"body_albedo.png"}}]
    for i, (name, model, color) in enumerate((("DO", "DO16", "#EF6C00"), ("AI", "AI4", "#2E7D32")), 1):
        changes=[]
        for stem, size, kind, target in (("top_", (89,64), "plc", top), ("front_", (66,29), "front", front)):
            tmp, _ = label(f, stem+name, size, (0,0,0), kind=kind, color=color, model=model, background="#D9DDE0")
            changes.append({"material": target.data.materials[0].name, "image":stem+name+"_print.png"})
            bpy.data.objects.remove(tmp, do_unlink=True)
        variants.append({"name":name, "model":model, "color":color,
                         "textures":{"body":f"body_albedo_v{i}.png"}, "source_changes":changes})
    p = part("body", items, [box("Module box collision", (65,90,30), (0,0,15), edge=0)], .15, size=1024)
    finish(f, [p], {"xyz": [65,90,30], "grip_width":65, "flat_grip_region_y":50,
                    "front_color_band_height":8, "model_letter_height":12, "DIN_recess_depth":2.5},
           "Brand is split into IMC / Controls to retain 12 mm glyph height on a 90 mm face. "
           "DIN rebates and terminal are outside the central 50 mm jaw contact strip. "
           "Terminal, flip tag, vents and LEDs are visual details; nominal one-box collision.", variants)


def chamfer_block(name, bounds, inner_edges=(), material=None, bevel=.25):
    """Convex block with exact 2/3 x 45 degree top inward-edge chamfers."""
    x0,x1,y0,y1,z0,z1,ch = bounds
    lower=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    tx0=x0+ch if "x0" in inner_edges else x0
    tx1=x1-ch if "x1" in inner_edges else x1
    ty0=y0+ch if "y0" in inner_edges else y0
    ty1=y1-ch if "y1" in inner_edges else y1
    upper=[(tx0,ty0),(tx1,ty0),(tx1,ty1),(tx0,ty1)]
    vertices=[(x,y,z) for z in (z0,z1-ch) for x,y in lower]+[(x,y,z1) for x,y in upper]
    faces=[(3,2,1,0),(8,9,10,11)]
    for level in (0,4):
        faces += [(level+i,level+(i+1)%4,level+(i+1)%4+4,level+i+4) for i in range(4)]
    return props._mesh_object(name,vertices,faces,(0,0,0),bevel,2,material)


def module_tray() -> None:
    f=start("module_tray")
    material=mat("Conductive PP tray", "#1A1A1A", .65)
    cols=[box("Base",(150,230,5),(0,0,2.5),edge=0),
          box("Pocket support floors",(71,208,5),(0,0,7.5),edge=0)]
    # Continuous outer rails bridge transverse chamfers outside the pocket mouths.
    cols += [box("Continuous outside rim", (17.5,230,20), (side*66.25,0,15), edge=0)
             for side in (-1,1)]
    # 15 convex pieces, preserving every pocket and finger channel.
    for y0,y1,edges in ((-115,-104,("y1",)),(-8,8,("y0","y1")),(104,115,("y0",))):
        cols.append(chamfer_block("Transverse rim",(-75,75,y0,y1,5,25,2),edges,bevel=0))
    for cy in (-56,56):
        for y0,y1 in ((cy-50,cy-15),(cy+15,cy+50)):
            for s in (-1,1):
                x0,x1=(-75,-35.5) if s<0 else (35.5,75)
                cols.append(chamfer_block("Pocket side",(x0,x1,y0,y1,5,25,2),
                                          ("x1",) if s<0 else ("x0",),bevel=0))
    visuals=[]
    for c in cols:
        cp=console.duplicate(c,"Moulded "+c.name)
        cp.data.materials.append(material)
        visuals.append(cp)
    body=union(visuals,"Tray continuous moulding")
    console.bevel(body,.35,2)
    tag, node=label(f,"tray_in",(36,9),(-64,0,25.01),["IN"],background="#1A1A1A",foreground="#FFFFFF")
    tmp,_=label(f,"tray_ok",(36,9),(0,0,0),["OK / PASS"],background="#2E7D32",foreground="#FFFFFF")
    bpy.data.objects.remove(tmp,do_unlink=True)
    changes=[{"material":material.name,"color":"#2E7D32"},
             {"material":tag.data.materials[0].name,"image":"tray_ok_print.png"}]
    p=part("body",[body,tag],cols,.3)
    finish(f,[p],{"xyz":[150,230,25],"pocket_xyz":[71,96,15],"pocket_centres_y":[-56,56],
                  "pocket_floor_z":10,"mouth_chamfer":2,"finger_width_y":30,"finger_extension_x":22,
                  "finger_depth":20,"finger_floor_z":5},
           "Open finger channels on both X sides of each pocket. 15 convex pieces preserve cavities. "
           "Pocket-floor centre between finger slots is supported; no hull spans an opening.",
           [{"name":"in","textures":{"body":"body_albedo.png"}},
            {"name":"ok","textures":{"body":"body_albedo_v1.png"},"source_changes":changes}],static=True,
           recommended_placement_m={"in":[.40,.30,.90],"ok":[.44,-.32,.90]})


def test_fixture() -> None:
    f=start("test_fixture")
    black=mat("Black anodized fixture", "#24282B",.35,1)
    silver=mat("Milled locating aluminium","#C9CDD1",.30,1)
    grey=mat("Button enclosure","#D7D7D7")
    vis=[box("Fixture base",(240,300,20),(0,0,10),black,1),
         box("Locating platform",(75,100,8),(0,0,24),silver,.5),
         box("TEST enclosure",(60,60,60),(-20,120,50),grey,1)]
    cols=[box("Base proxy",(240,300,20),(0,0,10),edge=0),
          box("Platform proxy",(75,100,8),(0,0,24),edge=0),
          box("Enclosure proxy",(60,60,60),(-20,120,50),edge=0)]
    for sx in (-1,1):
        for sy in (-1,1):
            for axis in ("x","y"):
                if axis=="x":
                    xa,xb=35.5,43.5; ya,yb=30,48
                else:
                    xa,xb=23.5,43.5; ya,yb=48,56
                x0,x1=sorted((sx*xa,sx*xb)); y0,y1=sorted((sy*ya,sy*yb))
                edges=(("x0" if sx>0 else "x1"),) if axis=="x" else (("y0" if sy>0 else "y1"),)
                col=chamfer_block("L stop arm",(x0,x1,y0,y1,20,46,3),edges,bevel=0)
                cols.append(col)
                obj=console.duplicate(col,"Chamfered L stop")
                obj.data.materials.append(silver)
                console.bevel(obj,.25,2)
                vis.append(obj)
    for x in (-106,106):
        for y in (-136,136):
            vis.append(cyl("Mounting screw",4,1,(x,y,20.1),silver,.2,n=24))
    # Discrete spring pins are visible from the front under the raised platform edge.
    gold=mat("Gold test contacts","#B08D57",.35,1)
    for y in (-30,-10,10,30):
        vis.append(cyl("Pogo pin collar",1.7,4,(-39,y,22),gold,.15,n=16))
    vis.append(cyl("Result lamp post",10,200,(88,-100,120),silver,.5))
    cols.append(cyl("Post proxy",10,200,(88,-100,120),edge=0,n=32))
    cablemat=mat("Cable elastomer","#151515",.9)
    for a,b in (((38,0,23),(62,-30,23)),((62,-30,23),(88,-100,38))):
        delta=Vector(b)-Vector(a)
        cable=cyl("Signal cable",2.3,delta.length,(0,0,0),cablemat,.2,n=16)
        console.transform(cable,(Vector(a)+Vector(b))/2,delta.to_track_quat("Z","Y").to_matrix())
        vis.append(cable)
    plaque,_=label(f,"fixture_name",(132,13),(-88,0,20.03),["IMC FT-02 FUNCTION TESTER"],background="#24282B",foreground="#FFFFFF")
    testtag,_=label(f,"test_legend",(46,10),(-44,120,80.03),["TEST"],background="#D7D7D7")
    vis += [plaque,testtag]
    parts=[part("body",vis,cols,3.91,material=METAL)]
    btn, meta=button("test_button",(-15,120,80),"#1565C0",6,"body")
    parts.append(btn)
    rgba=[.85,.85,.82,1]
    lampmat=props.pbr_material("Result lamp off","#D9D9D1",.25,0)
    profile=[(0,220),(30,220)]+[(30*math.cos(a),220+30*math.sin(a)) for a in [i*math.pi/16 for i in range(1,8)]]+[(0,250)]
    dome=console.revolve("Result hemisphere",profile,lampmat,sectors=48)
    dome.data.transform(Matrix.Translation((.088,-.100,0)))
    # A single coarse convex dome stays within 1 mm of the visual hemisphere.
    proxy=console.revolve("Dome proxy",[(0,220),(30.3,220),(28.0,231.6),(21.43,241.43),(11.60,247.99),(0,250.3)],sectors=15)
    proxy.data.transform(Matrix.Translation((.088,-.100,0)))
    parts.append(part("result_lamp",dome,[proxy],.08,parent="body",rgba=rgba))
    finish(f,parts,{"base_xyz":[240,300,20],"platform_xyz":[75,100,8],"platform_top_z":28,
                   "locating_inner_xy":[71,96],"stop_height_above_platform":18,"stop_thickness":8,
                   "stop_arms_xy":[20,18],"entry_chamfer":3,"jaw_clear_span_y":60,
                   "button_box_xyz":[60,60,60],"button_box_centre_xy":[-20,120],"button_box_top_z":80,
                   "result_post_diameter":20,"result_post_height":200,"result_lamp_diameter":60,
                   "result_post_centre_xy":[88,-100]},
           "The post rises 200 mm above the 20 mm base; hemisphere top is Z250. "
           "Pogo pins and flexible cable are appearance-only. All eight L-stop arms have 3 mm entry chamfers. "
           "Lamp mast at [88,-100] mm avoids the gripper housing envelope while remaining on the +X side.",
           static=True,recommended_placement_m=[.38,0,.90],buttons=[meta],
           lamps=[{"part":"result_lamp","on_rgba":[.18,.8,.25,1],"emission":.8,
                   "state_colors":{"testing":[1,.65,0,1],"pass":[.18,.8,.25,1],"fail":[.9,.08,.05,1]}}],
           collision_exclusions=[["body","test_button"]])


def ng_bin() -> None:
    f=start("ng_bin")
    red=mat("NG red PP","#C62828")
    dims=[((160,160,2),(0,0,1)),((2,160,30),(-79,0,17)),((2,160,30),(79,0,17)),
          ((156,2,30),(0,-79,17)),((156,2,30),(0,79,17))]
    vis=union([box("NG moulding",s,p,red,.3) for s,p in dims],"NG shallow bin")
    floor,_=label(f,"ng_floor",(90,70),(0,0,2.03),["NG"],background="#C62828",foreground="#FFFFFF")
    front,_=label(f,"ng_front",(65,23),(-80.01,0,17),["NG"],background="#C62828",foreground="#FFFFFF",rotation=FRONT)
    p=part("body",[vis,floor,front],[box("NG proxy",s,p,edge=0) for s,p in dims],.3)
    finish(f,[p],{"xyz":[160,160,32],"wall_height":30,"base_thickness":2,"wall_thickness":2},
           static=True,recommended_placement_m=[.30,-.18,.90])


def board(name, size, pos, face_material, core_material):
    """Coated corrugated board with visible kraft cut edges and softened corners."""
    obj=box(name,size,pos,face_material,.25)
    obj.data.materials.append(core_material)
    thin=min(range(3),key=lambda i:size[i])
    for face in obj.data.polygons:
        axis=max(range(3),key=lambda i:abs(face.normal[i]))
        if axis != thin:
            face.material_index=1
    return obj


def paper_core() -> bpy.types.Material:
    material=props.pbr_material("Corrugated cut edge", "#B88B5C",.85,0)
    tree=material.node_tree
    coord=tree.nodes.new("ShaderNodeTexCoord")
    wave=tree.nodes.new("ShaderNodeTexWave")
    wave.wave_type="BANDS";wave.bands_direction="DIAGONAL"
    wave.inputs["Scale"].default_value=320
    wave.inputs["Distortion"].default_value=.15
    tree.links.new(coord.outputs["Object"],wave.inputs["Vector"])
    ramp=tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color=props._linear_color("#795536")
    ramp.color_ramp.elements[1].color=props._linear_color("#D4B289")
    tree.links.new(wave.outputs["Fac"],ramp.inputs[0])
    tree.links.new(ramp.outputs["Color"],tree.nodes["Principled BSDF"].inputs["Base Color"])
    return material


def product_box() -> None:
    f=start("product_box")
    white=mat("White coated folding board","#F2F2F0",.85)
    core=paper_core()
    # Nominal BODY is X100 x Y168 x Z50. Shift +5 mm so the 10 mm front lip
    # leaves the entire closed assembly centred at X=0 as required by section 1.2.
    cx=5
    shapes=[((100,168,3),(cx,0,1.5)),((3,168,47),(cx-48.5,0,26.5)),
            ((3,168,47),(cx+48.5,0,26.5)),((94,3,47),(cx,-82.5,26.5)),
            ((94,3,47),(cx,82.5,26.5)),((94,2,47),(cx,-20,26.5))]
    body=union([board("Carton board",s,p,white,core) for s,p in shapes],"Compartmented body")
    front,_=label(f,"box_front",(130,35),(cx-50.01,0,27),["IMC Controls","PLC I/O MODULE"],rotation=FRONT)
    side_rot=Matrix(((1,0,0),(0,0,-1),(0,1,0)))
    side,_=label(f,"box_barcode",(65,25),(cx,-84.01,26),["IMC-15  /  PLC"],rotation=side_rot @ Matrix.Rotation(math.radians(2),3,"Z"),barcode=True)
    # 47 mm clear interior requires underside Z50 and top Z53, unlike a 50 mm
    # total closed bbox. The discrepancy is explicit in metadata/report.
    lidshapes=[((110,168,3),(0,0,51.5)),((3,168,8),(-53.5,0,46))]
    lid=union([board("Folded lid board",s,p,white,core) for s,p in lidshapes],"Lid with return flange")
    top,_=label(f,"lid_brand",(150,77),(0,0,53.01),["IMC Controls","PLC I/O MODULE","FACTORY TESTED  /  ESD SAFE"])
    parts=[part("body",[body,front,side],[box("Body proxy",s,p,edge=0) for s,p in shapes],.06,friction=.6,material=PAPER),
           part("lid",[lid,top],[box("Lid proxy",s,p,edge=0) for s,p in lidshapes],.02,parent="body",
                joint={"type":"hinge","parent":"body","axis":[0,1,0],"anchor_m":[.055,0,.05],
                       "range":[0.0,1.75],"damping":.01,"frictionloss":.001},friction=.6,material=PAPER)]
    finish(f,parts,{"body_xyz":[100,168,50],"closed_assembly_xyz":[110,168,53],"body_center_x":5,
                   "board_thickness":3,"divider_thickness":2,"module_compartment_xy":[94,100],
                   "accessory_compartment_xy":[94,60],"clear_height":47,"divider_centre_y":-20,
                   "lid_main_xy":[100,168],"lip_projection":10,"lip_drop":8,"lip_thickness":3},
           "Body is 50 mm high, clear interior 47 mm; 3 mm lid above it makes the closed assembly 53 mm. "
           "Body centre is X+5 mm so the 10 mm front projection yields an assembly bottom-centre origin. "
           "Main lid and horizontal 10 mm extension are one 110 mm slab, plus the 8 mm downward return. "
           "Positive +Y hinge rotation lifts the -X front. No contact exclusion disables lid closing contact.",
           recommended_placement_m=[.385,0,.93],
           packing={"module_center_m":[.005,.031,.003],"accessory_center_m":[.005,-.051,.003],
                    "lid_lift_point_m":[-.05,0,.05],"open_stop_rad":1.75})


def accessory_box() -> None:
    f=start("accessory_box")
    kraft=mat("Kraft paperboard","#B88B5C",.85)
    base=box("Accessory carton",(60,50,24.5),(0,0,12.25),kraft,.4)
    flaps=[box("Carton top flap",(29.8,50,.5),(x,0,24.75),kraft,.15) for x in (-15.1,15.1)]
    body=merge([base,*flaps],"Accessory carton with flap seam")
    top,_=label(f,"accessory_label",(46,53),(0,0,25.01),
                ["IMC Controls","ACCESSORY KIT","TERMINAL BLOCKS","+ MANUAL"],background="#E6CEAC",
                rotation=TOP @ Matrix.Rotation(math.radians(2),3,"Z"))
    p=part("body",[body,top],[box("Accessory proxy",(60,50,25),(0,0,12.5),edge=0)],.05,
           friction=.6,material=PAPER,size=1024)
    finish(f,[p],{"xyz":[60,50,25]},"Long accessory description wraps over two lines for readability.")


def stack_light() -> None:
    f=start("stack_light")
    black=mat("Tower black polymer","#202327")
    silver=mat("Tower bracket metal","#A7ADB3",.35,1)
    bracket=union([box("Foot plate",(100,60,8),(0,0,4),silver,1),
                   box("L return",(8,60,24),(46,0,16),silver,1)],"L mounting bracket")
    post=cyl("Tower mast",12.5,200,(0,0,108),silver,.5)
    collar=cyl("Tower lower collar",35,22,(0,0,219),black,.5)
    cap=cyl("Buzzer cap",35,30,(0,0,425),black,1)
    # Buzzer ventilation is inset engraving, never transparent texture.
    for y in (-20,-10,0,10,20):
        console.boolean(cap,box("Buzzer slot",(24,2,2),(0,y,440),edge=.4))
    vis=[bracket,post,collar,cap]
    tag,_=label(f,"tower_brand",(43,15),(-23,0,8.02),["IMC Controls"],background="#A7ADB3")
    vis.append(tag)
    cols=[box("Foot proxy",(100,60,8),(0,0,4),edge=0),box("L return proxy",(8,60,24),(46,0,16),edge=0),
          cyl("Mast proxy",12.5,200,(0,0,108),edge=0,n=32),
          cyl("Collar proxy",35,22,(0,0,219),edge=0,n=16),cyl("Buzzer proxy",35,30,(0,0,425),edge=0,n=16)]
    parts=[part("body",vis,cols,1.05,material=METAL)]
    lamps=[]
    for name,z,rgba,on in (("green",260,[.06,.30,.10,1],[.18,.85,.25,1]),
                           ("amber",320,[.40,.30,.05,1],[1,.68,.02,1]),
                           ("red",380,[.35,.06,.06,1],[.95,.06,.04,1])):
        material=props.pbr_material(name+" unlit lens","#555555",.25,0)
        material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=rgba
        # Remove default RGB link so the explicit off-state RGBA drives preview.
        socket=material.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
        for link in list(socket.links): material.node_tree.links.remove(link)
        obj=cyl("lens_"+name,35,60,(0,0,z),material,.5)
        parts.append(part("lens_"+name,obj,[cyl("Lens proxy",35,60,(0,0,z),edge=0,n=16)],.15,
                          parent="body",rgba=rgba))
        lamps.append({"part":"lens_"+name,"on_rgba":on,"emission":.8})
    finish(f,parts,{"bracket_xy":[100,60],"bracket_thickness":8,"mast_diameter":25,"mast_height":200,
                   "lens_diameter":70,"lens_height":60,"buzzer_height":30,"height":440,"lower_collar_height":22},
           "A 22 mm lower socket collar reconciles the specified 200 mm mast and 440 mm approximate height. "
           "Two bracket collision boxes preserve the L rather than filling its notch. Collar has its own proxy.",
           static=True,lamps=lamps,recommended_placement_m=[.56,.08,.90])


def line_console() -> None:
    f=start("line_console")
    grey=mat("Console RAL7035","#D7D7D7",.6)
    dark=mat("Console charcoal panel","#3A3D40",.6)
    metal=mat("Chrome trim","#BFC3C7",.3,1)
    black=mat("Gland black","#222326",.65)
    layout=[("reset","btn_blue",105,140,"RESET"),("mode","selector_knob",35,140,"MAN / AUTO"),
            ("start","btn_green",-35,140,"START"),("stop","btn_red",-105,140,"STOP"),
            ("emergency","estop",-100,68,"EMERGENCY STOP")]
    # Reuse task 2 shell construction and coordinate transforms exactly.
    body=console.shell_geometry("body",grey,layout)
    plate=console.rr_box("panel",280,190,1.5,3,material=dark,edge=0)
    console.transform(plate,console.pose(0,console.SLOPE_LENGTH/2,.75))
    for _,_,u,v,_ in layout:
        console.boolean(plate,console.mounting_cutter(u,v))
    console.bevel(plate,.3,2)
    # New English/bundled-font nameplate; never call task 2's artwork writer.
    side_rotation=Matrix(((1,0,0),(0,0,-1),(0,1,0)))
    nameplate,_=label(f,"console_name",(70,18),(console.BASE_X-5,-159.7,30),["IMC Controls","OP-03"],rotation=side_rotation)
    gland=cyl("Rear gland",6,12,(0,0,0),black,.4,n=32)
    console.transform(gland,(console.BASE_X+106,0,35),Matrix.Rotation(math.pi/2,3,"Y"))
    gc=cyl("Gland proxy",6,12,(0,0,0),edge=0,n=32)
    console.transform(gc,(console.BASE_X+106,0,35),Matrix.Rotation(math.pi/2,3,"Y"))
    library=json.loads((OUT/"panel_devices/prop.json").read_text())
    wanted=["bezel","selector_knob"]
    with bpy.data.libraries.load(str(OUT/"panel_devices/source.blend"),link=False) as (src,dst):
        dst.objects=wanted
    templates={}
    for obj in dst.objects:
        bpy.context.collection.objects.link(obj)
        templates[obj.name]=obj
        obj.hide_render=True
        obj.hide_set(True)
    panel_items=[plate]
    parts=[part("body",[body,nameplate,gland],console.shell_collisions()+[gc],3.635,material=console.SHEET_MJ)]
    buttons=[]; placements=[]; moving=[]
    for name,template,u,v,text in layout:
        pos=console.pose(u,v)
        if name=="mode":
            obj=console.duplicate(templates["selector_knob"],name)
            # Fixed local coordinate Empty keeps the library's procedural white stripe.
            frame=bpy.data.objects.new("Selector local shader frame",None)
            bpy.context.collection.objects.link(frame)
            frame.matrix_world=console.ROTATION.to_4x4(); frame.matrix_world.translation=pos/1000
            for i,original in enumerate(obj.data.materials):
                own=original.copy(); obj.data.materials[i]=own
                for node in own.node_tree.nodes:
                    if node.type=="TEX_COORD": node.object=frame
            console.transform(obj,pos)
            proxy=box("Selector lever proxy",(9,32,14),(0,0,13),edge=0)
            console.transform(proxy,pos)
            joint=json.loads(json.dumps(next(p for p in library["parts"] if p["name"]=="selector_knob")["joint"]))
            joint.update(parent="panel",axis=list(console.NORMAL),anchor_m=list(pos/1000))
            record=part("mode",obj,[proxy],.025,parent="panel",joint=joint,size=1024)
            record["recommended_initial_qpos"]=-math.pi/4
            moving.append(record)
        else:
            if name=="emergency":
                yellow=mat("Emergency yellow disc","#F4C20D")
                disc=cyl("Emergency disc",30,1.5,(0,0,.75),yellow,.3)
                console.transform(disc,pos)
                panel_items.append(disc)
            else:
                bezel=console.duplicate(templates["bezel"],name+" bezel")
                console.transform(bezel,pos)
                panel_items.append(bezel)
            record,meta=button(name+"_button",pos,{"reset":"#1565C0","start":"#2E9E44","stop":"#D32F2F","emergency":"#D32F2F"}[name],
                               5 if name=="emergency" else 4,"panel",console.ROTATION,name=="emergency")
            moving.append(record); buttons.append(meta)
        legendpos=console.pose(u,v-(40 if name=="emergency" else 29),1.6)
        legend,_=label(f,name+"_legend",(61 if name=="emergency" else 58,10),legendpos,[text],
                       background="#E5E7E9",foreground="#171A1D",rotation=console.ROTATION)
        panel_items.append(legend)
        placements.append({"name":name,"library_part":template,"panel_uv_mm":[u,v],"position_m":list(pos/1000)})
    parts.append(part("panel",panel_items,[],.3,parent="body",material=console.SHEET_MJ))
    parts+=moving
    for obj in templates.values():
        bpy.data.objects.remove(obj,do_unlink=True)
    finish(f,parts,{"enclosure_xy":[200,320],"front_height":60,"rear_height":160,
                   "slope_degrees":math.degrees(math.atan(.5)),"panel_uv":[280,190],"mushroom_diameter":40,
                   "button_travel":4,"emergency_travel":5},
           "Task 2 shell and panel frame reused, selector/bezel meshes appended read-only from panel_devices. "
           "RESET/START/STOP enlarged to 40 mm mushrooms per section 6.0 and user instruction. "
           "Row left-to-right is +Y to -Y: RESET, MAN/AUTO, START, STOP; emergency is lower right. "
           "Emergency spring is overridden by simulator latch once engaged. No built-in lamps. "
           "Body includes fixed rear gland, hence seven body proxies instead of six shell-only proxies.",
           static=True,buttons=buttons,lamps=[],devices=placements,recommended_placement_m=[.436,-.20,.90],
           selector={"part":"mode","positions_rad":[-math.pi/4,math.pi/4],"labels":["MAN","AUTO"],"initial_qpos":-math.pi/4},
           collision_exclusions=[[p["name"],root] for p in moving for root in ("body","panel")])


def render_exports(pid: str) -> None:
    """Independently render each delivered OBJ/albedo variant, including open lid."""
    data=json.loads((OUT/pid/"prop.json").read_text())
    for variant in [None,*data["variants"]]:
        f=start(pid)
        objects=[]
        by_name={}
        textures=variant.get("textures",{}) if variant else {}
        for p in data["parts"]:
            before=set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=str(f/p["visual"]),forward_axis="Y",up_axis="Z")
            obj=next(o for o in bpy.data.objects if o not in before)
            material=props.pbr_material("Delivered "+p["name"],"#FFFFFF",.55,0)
            bsdf=material.node_tree.nodes["Principled BSDF"]
            for link in list(bsdf.inputs["Base Color"].links): material.node_tree.links.remove(link)
            texture=textures.get(p["name"],p["texture"])
            if texture:
                node=material.node_tree.nodes.new("ShaderNodeTexImage")
                node.image=bpy.data.images.load(str(f/texture))
                material.node_tree.links.new(node.outputs["Color"],bsdf.inputs["Base Color"])
            else:
                bsdf.inputs["Base Color"].default_value=p["rgba"]
            obj.data.materials.clear();obj.data.materials.append(material)
            objects.append(obj)
            by_name[p["name"]]=obj
        suffix="_"+variant["name"] if variant else ""
        preview(objects,f,"preview_export"+suffix+".png")
        if pid=="product_box" and variant is None:
            anchor=Vector(next(p for p in data["parts"] if p["name"]=="lid")["joint"]["anchor_m"])
            by_name["lid"].matrix_world=(Matrix.Translation(anchor) @ Matrix.Rotation(1.75,4,"Y")
                                         @ Matrix.Translation(-anchor))
            preview(objects,f,"preview_export_open.png")


def isolated_finalize(ids: list[str]) -> None:
    """Existing compactor has global manifest/dedup passes even with --props.

    Stage only our outputs, call the mandatory public finalizer, then copy back.
    Never expose control_console, panel_devices, or concurrent builders to it.
    """
    subprocess.run([str(props.VALIDATOR_PYTHON), "-B", str(HERE/"verify_task15_electronics.py"),
                    "--repair-collisions", *ids], check=True)
    stage=QA/"finalize_root"
    if stage.exists(): shutil.rmtree(stage)
    for pid in ids:
        manifest=json.loads((OUT/pid/"prop.json").read_text())
        used={name for part in manifest["parts"] for name in part["collision"]}
        for path in (OUT/pid).glob("*_col_*.obj"):
            if path.name not in used:
                path.unlink()
        shutil.copytree(OUT/pid,stage/"assets/props"/pid)
    original=props.ROOT
    try:
        props.ROOT=stage
        props._DELIVERED_PROPS=set(ids)
        props.finalize_assets()
    finally:
        props.ROOT=original
    for pid in ids:
        source=stage/"assets/props"/pid
        # All compactor rewrites remain within the selected Task 15 set.
        for path in source.iterdir():
            if path.is_file() and path.name!="source.blend":
                shutil.copy2(path,OUT/pid/path.name)
    shutil.rmtree(stage)


def main() -> None:
    global ARGS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--props",nargs="+",choices=IDS,default=list(IDS))
    parser.add_argument("--geometry-only",action="store_true")
    parser.add_argument("--export-only",action="store_true")
    parser.add_argument("--render-exports",action="store_true")
    parser.add_argument("--finalize-only",action="store_true")
    ARGS=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    for pid in ARGS.props:
        if ARGS.finalize_only:
            continue
        if ARGS.render_exports:
            render_exports(pid)
        elif ARGS.export_only:
            bpy.ops.wm.open_mainfile(filepath=str(OUT/pid/"source.blend"))
            export_current(OUT/pid)
        else:
            globals()[pid]()
    if not ARGS.geometry_only:
        isolated_finalize(ARGS.props)


if __name__=="__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
