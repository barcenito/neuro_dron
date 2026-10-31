"""Evaluate frozen direct-motor policies. Reports simulated-navigation assumptions explicitly."""
import sys,argparse,json,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import replace
from drone_swarm.direct.artifacts import load,export
from drone_swarm.direct.training import evaluate
from drone_swarm.runtime import configure_device,device_info

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint')
    p.add_argument('--output',required=True);p.add_argument('--device',choices=('cpu','gpu','auto'),default='auto')
    p.add_argument('--episodes',type=int,default=200);p.add_argument('--seed',type=int,default=500000)
    p.add_argument('--stage',type=int,choices=range(5),default=0)
    p.add_argument('--sizes',nargs='+',type=int,default=[1,4,8]);p.add_argument('--batch-worlds',type=int,default=128)
    p.add_argument('--export');p.add_argument('--render',action='store_true')
    p.add_argument('--trace',help='JSON trajectory of the first size/mission, including direct motor actions')
    p.add_argument('--hold-seconds',type=float,help='Required continuous hold for this evaluation only')
    p.add_argument('--episode-seconds',type=float,help='Physical evaluation horizon; checkpoint stays unchanged')
    a=p.parse_args()
    if a.episodes<1 or min(a.sizes)<1:p.error('Positive episodes and sizes required')
    payload,cfg=load(a.checkpoint)
    overrides={name:getattr(a,name) for name in ('hold_seconds','episode_seconds') if getattr(a,name) is not None}
    if any(not math.isfinite(value) or value<=0 for value in overrides.values()):
        p.error('Evaluation durations must be positive and finite')
    cfg=replace(cfg,**overrides)
    if overrides and cfg.hold_seconds>cfg.episode_seconds:
        p.error('Episode horizon must cover the required hold')
    if payload.get('smoke_only'):p.error('Smoke weights are not eligible for performance evaluation')
    report={'checkpoint':a.checkpoint,'seed':a.seed,'stage':a.stage,'evaluation_overrides':overrides,'navigation':'simulated relative goal and velocity; no real navigation provider','cases':{}}
    for size in a.sizes:
        variant,device=configure_device(replace(cfg,device=a.device,swarm_size=size,batch_worlds=a.batch_worlds))
        metrics,_=evaluate(payload['weights'],variant,a.stage,a.seed,a.episodes)
        report['cases'][str(size)]={'config':variant.to_dict(),**metrics};report['runtime']=device_info(device)
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
        print(size,json.dumps(metrics),flush=True)
    if a.export:print(export(a.checkpoint,a.export))
    if a.render or a.trace:
        from types import SimpleNamespace
        import numpy as np
        from drone_swarm.direct.environment import DirectEnvironment
        from drone_swarm.direct.worlds import make_worlds
        from drone_swarm.learning.network import mlp_forward
        from drone_swarm.simulation.rendering import render_episode
        variant,_=configure_device(replace(cfg,device=a.device,swarm_size=a.sizes[0],batch_worlds=a.batch_worlds))
        env=DirectEnvironment(variant,1);s=make_worlds(a.seed,1,variant,a.stage);ep=env.reset(s)
        rows=[]
        while bool(ep.alive.any()):
            action=mlp_forward(ep.history.reshape((1,variant.swarm_size,-1)),payload['weights'])
            ep=env.step(ep,action,s)
            if int(ep.step)%5==0 or not bool(ep.alive.any()):
                rows.append({'time':float(ep.step)*variant.dt,'position':np.asarray(ep.data.states.pos[0]).tolist(),
                    'velocity_m_s':np.asarray(ep.data.states.vel[0]).tolist(),
                    'quaternion_xyzw':np.asarray(ep.data.states.quat[0]).tolist(),
                    'angular_velocity_rad_s':np.asarray(ep.data.states.ang_vel[0]).tolist(),
                    'goal_distance_m':np.asarray(ep.error[0]).tolist(),
                    'hold_seconds':float(ep.hold[0]),'failure_causes':np.asarray(ep.causes[0]).tolist(),
                    'action':np.asarray(action[0]).tolist(),'motor_thrust_N':np.asarray(ep.motors[0]).tolist(),
                    'success':bool(ep.success[0]),'failed':bool(ep.failed[0])})
                if a.render:
                    # Adapter for display only; direct policy never receives this estimate object.
                    view=SimpleNamespace(data=ep.data,step=ep.step,sensors=ep.sensors,
                                         estimate=SimpleNamespace(quat=ep.data.states.quat))
                    render_episode(env,view,s);time.sleep(variant.dt*5)
        if a.trace:
            from drone_swarm.direct.environment import CAUSES
            path=Path(a.trace);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'checkpoint':a.checkpoint,'seed':a.seed,'stage':a.stage,
                'evaluation_config':variant.to_dict(),
                'goal':np.asarray(s.routes[0,:,-1]).tolist(),'failure_cause_names':list(CAUSES),
                'frames':rows},indent=2)+'\n')

if __name__=='__main__':main()
