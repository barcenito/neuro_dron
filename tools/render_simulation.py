"""Render recorded Crazyflow states with its native MuJoCo drone and scene.

No new dynamics or invented trajectories: positions, orientations and obstacles
come from flight_trace.json. Uses software OpenGL without opening a window.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')

import crazyflow
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT/'docs/benchmark_20261031/flight_trace.json')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/benchmark_20261031/simulacion.gif')
    parser.add_argument('--preview', action='store_true')
    a = parser.parse_args()
    trace = json.loads(a.trace.read_text())
    assets = Path(crazyflow.__file__).parent/'models/cf2'
    spec = mujoco.MjSpec.from_file(str(assets/'scene.xml'))
    spec.copy_during_attach = True
    drone = mujoco.MjSpec.from_file(str(assets/'cf2.xml'))
    spec.worldbody.add_frame(name='world').attach_body(drone.body('drone'), '', ':0').add_freejoint()
    model = spec.compile()
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=960, max_geom=2000)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Display the real drone mesh, not its collision ball.
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 120
    camera.elevation = -25
    camera.distance = 1.6
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = small = ImageFont.load_default()
    goal = np.asarray(trace['goal'][0][0])
    frames = []

    def geom(kind, pos, size, color):
        scene = renderer.scene
        item = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(item, kind, np.array(size, dtype=float), np.asarray(pos, dtype=float),
                           np.eye(3).ravel(), np.array(color, dtype=np.float32))
        scene.ngeom += 1

    rows = trace['frames']
    indices = [min(60,len(rows)-1)] if a.preview else range(len(rows))
    for index in indices:
        row = rows[index]
        position = np.asarray(row['position'][0][0])
        quat = np.asarray(row['quaternion'][0][0])  # Crazyflow XYZW -> MuJoCo WXYZ.
        data.qpos[:3] = position
        data.qpos[3:7] = quat[[3,0,1,2]]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = .9*position + .1*goal
        renderer.update_scene(data, camera=camera, scene_option=option)
        # Same scenario geometry as the project's on-screen viewer.
        for lo, hi in zip(trace['boxes_lo'][0], trace['boxes_hi'][0]):
            lo, hi = np.asarray(lo), np.asarray(hi)
            if np.linalg.norm(lo)<30:
                geom(mujoco.mjtGeom.mjGEOM_BOX, (lo+hi)/2, (hi-lo)/2, [.45,.5,.6,.12])
        for p in trace['static'][0]:
            if np.linalg.norm(p)<30:
                geom(mujoco.mjtGeom.mjGEOM_SPHERE,p,[trace['obstacle_radius']]*3,[.8,.2,.2,1])
        for p in row['dynamic'][0]:
            if np.linalg.norm(p)<30:
                geom(mujoco.mjtGeom.mjGEOM_SPHERE,p,[trace['obstacle_radius']]*3,[.8,.2,.8,1])
        geom(mujoco.mjtGeom.mjGEOM_SPHERE,goal,[trace['region_radius']]*3,[1,.6,0,.07])
        geom(mujoco.mjtGeom.mjGEOM_SPHERE,goal,[.035]*3,[1,.6,0,1])
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0,0,960,46],fill='#111c2d')
        draw.text((20,11),'NEURO DRON  /  Crazyflow + MuJoCo',font=font,fill='#edf6ff')
        draw.text((780,15),f"t = {row['time']:04.1f} s",font=small,fill='#c5d5e8')
        draw.rectangle([0,498,960,540],fill='#111c2d')
        state = 'Destino alcanzado' if row['success'][0] else 'Vuelo autonomo · control directo de motores'
        draw.text((20,510),state,font=small,fill='#edf6ff')
        draw.text((625,510),'Pesos congelados · mision 0 / 500031',font=small,fill='#c5d5e8')
        frames.append(frame)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.preview:
        frames[0].save(a.output.with_suffix('.png'))
    else:
        durations = [max(10,round((rows[i+1]['time']-r['time'])*1000)) for i,r in enumerate(rows[:-1])]+[1600]
        frames[0].save(a.output,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)
        frames[min(60,len(frames)-1)].save(a.output.with_suffix('.png'))
        if shutil.which('ffmpeg'):
            optimized = a.output.with_name(a.output.stem+'.optimized.gif')
            subprocess.run(['ffmpeg','-v','error','-i',str(a.output),'-filter_complex',
                'fps=10,scale=720:-1:flags=lanczos,split[s0][s1];'
                '[s0]palettegen=max_colors=96:stats_mode=diff[p];[s1][p]paletteuse=dither=none',
                '-loop','0','-y',str(optimized)], check=True)
            optimized.replace(a.output)
    renderer.close()
    output = a.output.with_suffix('.png') if a.preview else a.output
    public_path = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output.name
    print(public_path, len(frames), 'rendered frames',flush=True)


if __name__ == '__main__':
    main()
