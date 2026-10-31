"""Evaluate frozen reference weights and record two real simulated flights.

Optional visualization dependencies: matplotlib and Pillow. Run from any folder.
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import jax
import numpy as np
from drone_swarm.direct.artifacts import load
from drone_swarm.direct.environment import DirectEnvironment, summarize, CAUSES
from drone_swarm.direct.worlds import make_worlds
from drone_swarm.learning.network import mlp_forward
from drone_swarm.runtime import configure_device, device_info
from drone_swarm.simulation.scenarios import obstacle_states


def wilson(k, n):
    z = 1.95996398454
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [float(c-h), float(c+h)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT/'models/separacion_reference/best.pkl'))
    parser.add_argument('--output', default=str(ROOT/'docs/benchmark_20261031'))
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='auto')
    parser.add_argument('--episodes', type=int, default=200)
    a = parser.parse_args()
    if a.episodes < 2:
        parser.error('At least two episodes required')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    payload, base = load(a.checkpoint)
    if payload.get('smoke_only'):
        raise ValueError('Smoke checkpoint is not a benchmark candidate')
    base, device = configure_device(replace(base, device=a.device))
    names = ['hover', 'goal', 'static', 'moving_a', 'moving_b', 'moving_c', 'maze', 'group4', 'group8', 'hold30']
    settings = [(0,1,3,20,500031), (1,1,3,20,500031), (2,1,3,20,500031),
                (3,1,3,20,500031), (3,1,3,20,500032), (3,1,3,20,500033),
                (4,1,3,20,500031), (1,4,3,20,500031), (1,8,3,20,500031),
                (1,1,30,40,500031)]
    active_files = {str(p.relative_to(ROOT/'src/drone_swarm')): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT/'src/drone_swarm').rglob('*.py')}
    original = payload.get('source_sha256', {})
    changed = [f for f,h in active_files.items() if original.get(f) != h]
    checkpoint_path = Path(a.checkpoint).resolve()
    public_path = str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path.name
    report = dict(checkpoint=public_path,
                  sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                  checkpoint_generation_index=payload.get('generation'), runtime=device_info(device),
                  config=base.to_dict(), source_sha256=active_files, changed_active_sources=changed,
                  protocol='Frozen weights; full horizons; fresh diagnostic seeds; no training or tuning.',
                  seeds_now_consumed=[500031,500032,500033], cases={})
    cache = {}
    trace_s = None
    for name, (stage, size, hold, horizon, seed) in zip(names, settings):
        cfg = replace(base, swarm_size=size, hold_seconds=hold, episode_seconds=horizon, batch_worlds=a.episodes)
        key = (size, hold, horizon)
        if key not in cache:
            cache[key] = DirectEnvironment(cfg, a.episodes)
        env = cache[key]
        s = make_worlds(seed, a.episodes, cfg, stage)
        started = time.perf_counter()
        ep = env.rollout(payload['weights'], s)
        jax.block_until_ready(ep.success)
        metrics = summarize(ep, True)
        success = np.asarray(ep.success)
        report['cases'][name] = dict(stage=stage, size=size, hold=hold, horizon=horizon, seed=seed,
            wall_seconds=time.perf_counter()-started, metrics=metrics,
            successes=int(success.sum()), success_ci95=wilson(int(success.sum()), a.episodes),
            success=success.tolist(), failed=np.asarray(ep.failed).tolist(),
            causes=np.asarray(ep.causes).tolist(), cause_names=list(CAUSES))
        (out/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, json.dumps(metrics), flush=True)
        if name == 'moving_a':
            good = np.flatnonzero(success)
            bad = np.flatnonzero(np.asarray(ep.failed))
            ids = [int(good[0]) if len(good) else 0, int(bad[0]) if len(bad) else 1]
            trace_s = jax.tree_util.tree_map(lambda x: x[np.array(ids)], s)
            trace_cfg = cfg
            report['gif_selection'] = dict(seed=seed, mission_ids=ids,
                rule='First successful and first failed mission in moving_a; illustrative, not random performance estimate.')
    moving = [report['cases'][n] for n in ['moving_a','moving_b','moving_c']]
    k = sum(c['successes'] for c in moving)
    report['moving_aggregate'] = dict(episodes=3*a.episodes, successes=k, success_rate=k/(3*a.episodes),
                                      success_ci95=wilson(k,3*a.episodes),
                                      collision_rate=float(np.mean([c['metrics']['collision_rate'] for c in moving])))
    (out/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    render_chart(report, out/'benchmark.png')
    record_flights(payload['weights'], trace_cfg, trace_s, out, report['gif_selection'])


def record_flights(weights, cfg, s, out, selection):
    env = DirectEnvironment(cfg, 2)
    ep = env.reset(s)
    frames = []
    def capture():
        dynamic, _ = obstacle_states(s, np.full(2, int(ep.step)*cfg.dt))
        frames.append(dict(time=float(ep.step)*cfg.dt, position=np.asarray(ep.data.states.pos).tolist(),
                           dynamic=np.asarray(dynamic).tolist(), success=np.asarray(ep.success).tolist(),
                           failed=np.asarray(ep.failed).tolist(), hold=np.asarray(ep.hold).tolist(),
                           quaternion=np.asarray(ep.data.states.quat).tolist(),
                           causes=np.asarray(ep.causes).tolist()))
    capture()
    while bool(ep.alive.any()):
        action = mlp_forward(ep.history.reshape((2,cfg.swarm_size,-1)), weights)
        ep = env.step(ep, action, s)
        if int(ep.step) % 10 == 0 or not bool(ep.alive.any()):
            capture()
    data = dict(selection=selection, frames=frames, dt=cfg.dt,
                goal=np.asarray(s.routes[:,:,-1]).tolist(), static=np.asarray(s.static).tolist(),
                boxes_lo=np.asarray(s.boxes_lo).tolist(), boxes_hi=np.asarray(s.boxes_hi).tolist(),
                drone_radius=cfg.drone_radius, obstacle_radius=cfg.obstacle_radius,
                region_radius=cfg.region_radius, cause_names=list(CAUSES))
    (out/'flight_trace.json').write_text(json.dumps(data, indent=2)+'\n')
    render_gif(data, out/'vuelo.gif')


def render_gif(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    rows = data['frames']
    positions = np.asarray([r['position'] for r in rows])[:,:,0,:]
    goals = np.asarray(data['goal'])[:,0,:]
    fig, axes = plt.subplots(1,2,figsize=(11,6),dpi=90)
    images = []
    xmin = min(float(positions[:,:,0].min()), float(goals[:,0].min()))-.3
    xmax = max(float(positions[:,:,0].max()), float(goals[:,0].max())+data['region_radius'])+.3
    for idx, row in enumerate(rows):
        for j, ax in enumerate(axes):
            ax.clear()
            ax.set_facecolor('#f3f6fa')
            goal = goals[j]
            ax.add_patch(Circle(goal[:2], data['region_radius'], color='#e9ad35', alpha=.15))
            ax.plot(*goal[:2], marker='+', color='#b37400', markersize=12)
            for p in data['static'][j]:
                if np.linalg.norm(p)<30:
                    ax.add_patch(Circle(p[:2], data['obstacle_radius'], color='#778397', alpha=.6))
            for p in row['dynamic'][j]:
                if np.linalg.norm(p)<30:
                    ax.add_patch(Circle(p[:2], data['obstacle_radius'], color='#db4d8c', alpha=.85))
                    ax.text(p[0],p[1]+.14,f'z={p[2]:.1f}',fontsize=7,color='#a62f64',ha='center')
            p = positions[idx,j]
            ax.plot(positions[:idx+1,j,0],positions[:idx+1,j,1],color='#148caa',lw=1.6)
            ax.add_patch(Circle(p[:2],data['drone_radius'],color='#087d9b'))
            # Rotor locations are a visual symbol; collision radius remains the circle above.
            ax.plot([p[0]-.07,p[0]+.07],[p[1]-.07,p[1]+.07],color='#07536b',lw=2)
            ax.plot([p[0]-.07,p[0]+.07],[p[1]+.07,p[1]-.07],color='#07536b',lw=2)
            state = 'ÉXITO' if row['success'][j] else 'COLISIÓN' if row['failed'][j] else 'EN VUELO'
            ax.set_title(f"Misión {data['selection']['mission_ids'][j]} · {state}",fontsize=11)
            ax.text(.03,.03,f"Altura: {p[2]:.2f} m | Hold: {row['hold'][j]:.1f} s",transform=ax.transAxes,fontsize=9)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(-1.9,1.9)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.grid(alpha=.2)
        fig.suptitle(f"Vuelo simulado · best de separación · t = {row['time']:.1f} s",fontsize=14)
        fig.subplots_adjust(bottom=.27,top=.84,wspace=.25)
        if idx == 0:
            fig.text(.5,.05,'Vista superior: azul = dron/traza · rosa = móviles · gris = estáticos · ámbar = destino\n'
                     'Proyección 2D de trayectorias 3D reales. La altura puede evitar un contacto aparente.\n'
                     'Primer éxito y primer fallo del benchmark; al terminar cada misión su estado queda congelado.',
                     ha='center',fontsize=8,color='#445066')
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    durations=[max(10,round((rows[i+1]['time']-r['time'])*1000)) for i,r in enumerate(rows[:-1])]+[1600]
    images[0].save(path,save_all=True,append_images=images[1:],duration=durations,loop=0,optimize=True)
    images[min(len(images)-1,60)].save(path.with_suffix('.png'))
    plt.close(fig)
    public_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
    print('GIF',public_path, len(images),'frames',flush=True)


def render_chart(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['hover','goal','hold30','static','moving','maze','group4','group8']
    labels = ['Vuelo estacionario','Llegada · 3 s','Llegada · 30 s','Obstáculos estáticos',
              'Obstáculos móviles','Laberinto (sin entrenar)','Grupo de 4 drones','Grupo de 8 drones']
    items = [report['moving_aggregate'] if k=='moving' else report['cases'][k] for k in names]
    values = [100*(c['success_rate'] if 'success_rate' in c else c['metrics']['success_rate']) for c in items]
    cis = np.array([c['success_ci95'] for c in items])*100
    fig, ax = plt.subplots(figsize=(10,5.5),dpi=140)
    y = np.arange(len(names))
    ax.barh(y,values,color=['#148caa']*4+['#d8a035']*2+['#c95871']*2,height=.6)
    ax.errorbar(values,y,xerr=[np.maximum(0,np.array(values)-cis[:,0]),
                             np.maximum(0,cis[:,1]-np.array(values))],
                fmt='none',ecolor='#25354b',capsize=3,lw=1)
    for i,v in enumerate(values):
        ax.text(cis[i,1]+3,i,f'{v:.2f}'.rstrip('0').rstrip('.')+' %',va='center',fontsize=10)
    ax.set_yticks(y,labels)
    ax.invert_yaxis()
    ax.set_xlim(0,112)
    ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Misiones completadas con éxito (%) · intervalo de Wilson del 95 %')
    ax.set_title('Estado del piloto · benchmark del 31/10/2026',loc='left',fontweight='bold',pad=20)
    ax.grid(axis='x',alpha=.15)
    ax.set_axisbelow(True)
    for side in ['top','right','left']:
        ax.spines[side].set_visible(False)
    total = sum(c['metrics']['episodes'] for c in report['cases'].values())
    fig.text(.02,.02,f"{total:,} misiones · pesos congelados · Crazyflow · "
             f"{report['moving_aggregate']['episodes']} con móviles; {report['cases']['hover']['metrics']['episodes']} por cada otra condición\n"
             'En grupos se mide éxito del conjunto. Los intervalos no incluyen variación entre entrenamientos.',
             fontsize=9,color='#445066')
    fig.tight_layout(rect=[0,.08,1,1])
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
