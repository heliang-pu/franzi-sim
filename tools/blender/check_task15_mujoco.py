"""Compile delivered Task 15 assets through the real loader; exercise passive joints."""
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# PROP_PYTHON is the tools venv. Reuse the matching Python 3.12 project package
# read-only when MuJoCo is not installed in that tools venv: PROJECT_SITE_PACKAGES
# names the project env's site-packages on this machine.
try:
    import mujoco
except ImportError:
    sys.path.append(os.environ['PROJECT_SITE_PACKAGES'])
    import mujoco
import numpy as np
# prop_assets needs only sim's ROOT and fmt, not the robot (Pinocchio).
from sim import prop_assets

IDS = ('plc_module','module_tray','test_fixture','ng_bin','product_box','accessory_box','stack_light','line_console')
QA = ROOT / 'tools/blender/_qa/task15'


def make_model(pid, variant=None, free=False):
    root = ET.Element('mujoco', model='task15_'+pid)
    ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep='0.002',integrator='implicitfast')
    ET.SubElement(root,'asset')
    world = ET.SubElement(root,'worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .1',rgba='.4 .4 .4 1',friction='.6 .005 .0001')
    prop_assets.add(root,pid,'prop',pos=(0,0,.005 if free else 0),free=free,variant_name=variant)
    xml = ET.tostring(root,encoding='unicode')
    suffix = '_'+variant if variant else ''
    (QA/(pid+suffix+'_loader.xml')).write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    return model,mujoco.MjData(model)


def main(ids=IDS):
    rows=[]
    for pid in ids:
        spec=prop_assets.spec(pid)
        variants=[None]+[v['name'] for v in spec['variants']]
        for variant in variants:
            free=not spec.get('static',False)
            model,data=make_model(pid,variant,free)
            mujoco.mj_forward(model,data)
            for _ in range(400): mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            assert max((max(0,-c.dist) for c in data.contact),default=0)<.001
            row={'prop':pid,'variant':variant,'bodies':model.nbody,'joints':model.njnt,
                 'geoms':model.ngeom,'max_rest_penetration_mm':1000*max((max(0,-c.dist) for c in data.contact),default=0)}
            tests=[]
            if variant is None:
                for btn in spec.get('buttons',[]):
                    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'prop_'+btn['part'])
                    qadr=model.jnt_qposadr[jid];dadr=model.jnt_dofadr[jid]
                    # Nominal full-stroke spring load, not an arbitrary overload.
                    force=float(model.jnt_stiffness[jid]*btn['travel_m']+model.dof_frictionloss[dadr])
                    data.qfrc_applied[dadr]=force
                    for _ in range(150): mujoco.mj_step(model,data)
                    pressed=float(data.qpos[qadr])
                    data.qfrc_applied[dadr]=0
                    for _ in range(300): mujoco.mj_step(model,data)
                    released=float(data.qpos[qadr])
                    assert pressed>.7*btn['travel_m'],(pid,btn['part'],'press',pressed)
                    assert released<.001,(pid,btn['part'],'return',released)
                    assert pressed<btn['travel_m']+.001,(pid,btn['part'],'limit softness',pressed)
                    tests.append({'button':btn['part'],'force_n':force,'pressed_m':pressed,'released_m':released,
                                  'note':'Passive spring tested; emergency latch is scene state-machine behavior'})
                if pid=='product_box':
                    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'prop_lid')
                    qadr=model.jnt_qposadr[jid]
                    data.qpos[qadr]=.4;data.qvel[:]=0
                    for _ in range(1200):mujoco.mj_step(model,data)
                    closed=float(data.qpos[qadr])
                    assert abs(closed)<.09,('lid close',closed)
                    data.qpos[qadr]=1.74;data.qvel[:]=0
                    for _ in range(1200):mujoco.mj_step(model,data)
                    opened=float(data.qpos[qadr])
                    assert 1.70<opened<1.80,('lid open stop',opened)
                    tests.append({'lid_released_from_0_4_rad':closed,'lid_released_near_open_stop':opened})
            row['tests']=tests;rows.append(row)
            print(pid,variant or 'default','PASS',tests,flush=True)
    destination='mujoco_checks.json' if tuple(ids)==IDS else 'mujoco_partial.json'
    (QA/destination).write_text(json.dumps({'mujoco_version':mujoco.__version__,'cases':rows},indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props',nargs='+',choices=IDS,default=IDS)
    main(parser.parse_args().props)
