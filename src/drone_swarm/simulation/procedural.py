"""Random mapped rooms and two-gate routes, not unknown-maze exploration.

One independent RNG per mission ID. CRN twins are exact copies. Moving spheres
have varied directions and frequencies but do not react to the controlled drone.
"""
import numpy as np
import jax.numpy as jnp
from drone_swarm.simulation.scenarios import (Scenario,formation_offsets,
    validate_scenario,path_sphere_clearance,_segment_valid)


def make_procedural_scenarios(seed,worlds,cfg,stage=0,paired=False,replay=0.,start_index=0,practice=0.):
    if paired and worlds%2: raise ValueError('CRN requires even worlds')
    n=worlds//2 if paired else worlds
    K=cfg.swarm_size; fields={name:[] for name in Scenario._fields}
    for w in range(n):
        rng=np.random.default_rng(np.random.SeedSequence([seed,start_index+w]))
        level=int(rng.integers(0,stage)) if stage and rng.random()<replay else stage
        active_count=K if level>=4 else 1
        # Launch queue is only an initial condition, never a final formation.
        spawn=np.tile([-2.,0.,1.],(K,1)).astype(np.float32)
        for k in range(active_count):
            spawn[k]=[-2.-.45*(k//3),.45*((k%3)-1) if active_count>1 else 0.,1.]
        goal=np.array([rng.uniform(2.6,3.3),rng.uniform(-.3,.3),1.],np.float32)
        lo=np.full((4,3),100.,np.float32);hi=lo+.1
        gate_x=np.array([-.5,.8])+rng.uniform(-.12,.12,2)
        gate_y=rng.uniform(-.55,.55,2)
        widths=rng.uniform(.85,1.4,2)
        # Repeated waypoints pad a fixed seven-point route for all levels.
        if level==0:
            goal=spawn[0]+[rng.uniform(1.3,2.1),rng.uniform(-.3,.3),rng.uniform(-.08,.08)]
            center_route=np.tile(goal,(7,1))
        else:
            center_route=np.array([[gate_x[0]-.3,gate_y[0],1.],
                [gate_x[0]+.3,gate_y[0],1.],[gate_x[1]-.3,gate_y[1],1.],
                [gate_x[1]+.3,gate_y[1],1.],goal+[-1.15,0.,0.],goal+[-.95,0.,0.],goal])
            for i in range(2):
                x,y,gap=gate_x[i],gate_y[i],widths[i]
                lo[2*i]=[x-.06,-1.7,0.];hi[2*i]=[x+.06,y-gap/2,2.5]
                lo[2*i+1]=[x-.06,y+gap/2,0.];hi[2*i+1]=[x+.06,1.7,2.5]
        routes=np.broadcast_to(center_route,(K,7,3)).copy().astype(np.float32)
        if cfg.mission_mode=='slots':
            routes[:active_count,-1]+=formation_offsets(active_count,cfg)
        paths=np.concatenate([spawn[:active_count,None],routes[:active_count]],1)
        static=np.full((4,3),100.,np.float32)
        if level>=2:
            for i in range(int(rng.integers(1,5))):
                for _ in range(100):
                    candidate=np.array([rng.uniform(-1.3,2.5),rng.uniform(-1.35,1.35),rng.uniform(.85,1.15)],np.float32)
                    if (_segment_valid(candidate,candidate,lo,hi,cfg.obstacle_radius) and
                        path_sphere_clearance(paths,candidate[None],cfg.drone_radius+cfg.obstacle_radius)>.12 and
                        np.linalg.norm(candidate-goal)>cfg.region_radius+.2 and
                        np.linalg.norm(static-candidate,axis=-1).min()>.3):
                        static[i]=candidate;break
        dc=np.full((4,3),100.,np.float32);da=np.zeros((4,3),np.float32)
        omega=rng.uniform(.4,1.8,(4,1)).astype(np.float32)*cfg.dynamic_speed_scale
        phase=rng.uniform(0,2*np.pi,(4,1)).astype(np.float32)
        if level>=3:
            for i in range(int(rng.integers(1,5))):
                for _ in range(100):
                    center=np.array([rng.uniform(-.8,2.6),rng.uniform(-.65,.65),rng.uniform(.9,1.1)],np.float32)
                    angle=rng.uniform(-np.pi,np.pi);amplitude=rng.uniform(.3,.9)
                    delta=np.array([np.cos(angle),np.sin(angle),0.],np.float32)*amplitude
                    if (np.max(np.abs((center[None]+np.array([-1,1])[:,None]*delta)[:,1]))<1.5 and
                        _segment_valid(center-delta,center+delta,lo,hi,cfg.obstacle_radius) and
                        path_sphere_clearance(np.array([[center-delta,center+delta]]),spawn[:active_count],cfg.drone_radius+cfg.obstacle_radius)>.15 and
                        path_sphere_clearance(np.array([[center-delta,center+delta]]),static,2*cfg.obstacle_radius)>.02):
                        dc[i]=center;da[i]=delta;break
        if cfg.dynamic_scale==0:dc[:]=100.;da[:]=0.
        else:da*=cfg.dynamic_scale
        # Quarter-turn transforms preserve exact AABB geometry.
        angle=float(rng.integers(4)*np.pi/2)
        R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]],np.float32)
        shift=np.array([*rng.uniform(-.4,.4,2),0.],np.float32)
        if cfg.enclosed_room:
            x0=float(spawn[:active_count,0].min()-cfg.room_margin);x1=float(goal[0]+cfg.room_margin)
            room_lo=np.array([[x0-.12,-1.82,0],[x1,-1.82,0],[x0,-1.82,0],[x0,1.7,0]])
            room_hi=np.array([[x0,1.82,2.5],[x1+.12,1.82,2.5],[x1,-1.7,2.5],[x1,1.82,2.5]])
            lo=np.concatenate([lo,room_lo]);hi=np.concatenate([hi,room_hi])
        centers=(lo+hi)/2@R.T+shift;half=(hi-lo)/2@np.abs(R.T)
        release=np.zeros(K,np.float32)
        # Explicit queue only for the narrow-stage diagnostic, never hidden IDs in policy.
        if level==5:release[:active_count]=np.arange(active_count)*2.5
        length=np.linalg.norm(np.diff(paths,axis=1),axis=-1).sum(-1)
        horizon=float(np.max(length/(cfg.max_speed*.7)+release[:active_count])+cfg.hold_seconds+cfg.preflight_seconds+rng.uniform(5.,7.))
        values=(spawn@R.T+shift,routes@R.T+shift,np.arange(K)<active_count,
            centers-half,centers+half,static@R.T+shift,dc@R.T+shift,da@R.T,cfg.preflight_seconds,omega,phase,
            rng.uniform(-.03,.03,(K,3)).astype(np.float32)@R.T,angle+rng.uniform(-.6,.6,K),start_index+w,
            release,horizon,level,angle,rng.integers(0,2**32,(2,),dtype=np.uint32))
        for name,value in zip(Scenario._fields,values):fields[name].append(value)
    arrays=[np.asarray(fields[name]) for name in Scenario._fields]
    validate_scenario(Scenario(*arrays),cfg)
    if paired:arrays=[np.concatenate([a,a]) for a in arrays]
    return Scenario(*(jnp.asarray(a) for a in arrays))
