"""Host-side validated mission generation; each ES mirror gets the whole world."""
from typing import NamedTuple
import numpy as np
import jax.numpy as jnp

STAGES = ('arrive_hold', 'route', 'parked', 'moving', 'swarm', 'narrow')

class Scenario(NamedTuple):
    spawn: object
    routes: object             # W,K,P,3; final slots or repeated common region center
    active: object             # W,K
    boxes_lo: object           # W,B,3, unused boxes at 100
    boxes_hi: object
    static: object             # W,O,3
    dynamic_center: object     # W,D,3
    dynamic_amplitude: object
    dynamic_start: object       # W; obstacles use mission time after optional preflight
    dynamic_frequency: object
    dynamic_phase: object
    initial_velocity: object   # W,K,3; physical launch motion
    initial_yaw: object        # W,K; known launch orientation
    episode_id: object         # W; reproducible across batch partitions
    release: object            # W,K seconds
    horizon: object            # W seconds
    stage: object              # W
    rotation: object           # W radians
    noise_keys: object          # W,2 independent worlds or explicit CRN


def formation_offsets(k, cfg):
    if k == 1: return np.zeros((1,3),np.float32)
    sep=max(cfg.separation,4*cfg.drone_radius)
    radius=sep/(2*np.sin(np.pi/k))
    theta=np.arange(k)*2*np.pi/k
    return np.stack([radius*np.cos(theta),radius*np.sin(theta),np.zeros(k)],-1).astype(np.float32)


def _segment_valid(a,b,lo,hi,radius):
    # Dense validation complements exact runtime sweep; <=1 cm grid.
    p=np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.01)+2))
    d=np.maximum(np.maximum(lo[None]-p[:,None],p[:,None]-hi[None]),0)
    return np.linalg.norm(d,axis=-1).min() > radius+.02


def path_sphere_clearance(points,centers,radius):
    a=points[:,:-1,None,:]; delta=points[:,1:,None,:]-a
    u=np.clip(np.sum((centers[None,None]-a)*delta,-1)/(np.sum(delta*delta,-1)+1e-12),0,1)
    return np.linalg.norm(a+u[...,None]*delta-centers[None,None],axis=-1).min()-radius


def validate_scenario(s,cfg):
    for w in range(len(s.spawn)):
        active=np.asarray(s.active[w],bool)
        spawn=np.asarray(s.spawn[w])[active]
        slots=np.asarray(s.routes[w])[active,-1]
        for points in ((spawn,slots) if cfg.mission_mode=='slots' else (spawn,)):
            if len(points)>1:
                d=np.linalg.norm(points[:,None]-points[None],axis=-1)
                np.fill_diagonal(d,np.inf)
                if d.min() < 2*cfg.drone_radius+.02: raise ValueError('Overlapping spawn/slots')
        for k in np.flatnonzero(active):
            points=np.concatenate([s.spawn[w,k:k+1],s.routes[w,k]])
            for a,b in zip(points[:-1],points[1:]):
                if not _segment_valid(a,b,s.boxes_lo[w],s.boxes_hi[w],cfg.drone_radius):
                    raise ValueError('Route intersects inflated maze')
            if path_sphere_clearance(points[None],np.asarray(s.static[w]),cfg.drone_radius+cfg.obstacle_radius)<=.02:
                raise ValueError('Route intersects parked agent')
            for p in (points[0],points[-1]):
                if np.linalg.norm(s.static[w]-p,axis=-1).min() <= cfg.drone_radius+cfg.obstacle_radius+.02:
                    raise ValueError('Spawn/slot intersects parked agent')


def make_scenarios(seed, worlds, cfg, stage=0, paired=False, replay=0., start_index=0, practice=0.):
    if cfg.scenario_family=='procedural':
        from drone_swarm.simulation.procedural import make_procedural_scenarios
        return make_procedural_scenarios(seed,worlds,cfg,stage,paired,replay,start_index,practice)
    if paired and worlds%2: raise ValueError('CRN requires even worlds')
    n=worlds//2 if paired else worlds
    K=cfg.swarm_size
    fields={name:[] for name in Scenario._fields}
    for w in range(n):
        rng=np.random.default_rng(np.random.SeedSequence([seed,start_index+w]))
        level=int(rng.integers(0,stage)) if stage and rng.random()<replay else stage
        k_active=K if level>=4 else 1
        offsets=formation_offsets(k_active,cfg)
        spawn=np.tile([0.,0.,1.],(K,1)).astype(np.float32)
        routes=np.zeros((K,4,3),np.float32)
        release=np.zeros(K,np.float32)
        initial_velocity=np.zeros((K,3),np.float32)
        lo=np.full((4,3),100.,np.float32); hi=lo+.1
        static=np.full((4,3),100.,np.float32)
        dc=np.full((2,3),100.,np.float32); da=np.zeros((2,3),np.float32)
        dp=rng.uniform(0,2*np.pi,(2,1)).astype(np.float32)
        for k in range(k_active):
            o=offsets[k]
            if level==5:
                # Preloaded queue before a 1x1 m passage; no peer messages.
                spawn[k]=[-1.8-k*.6,0,1]
                routes[k]=[[-1.4,0,1],[0,0,1],[1.4,0,1],np.array([2.8,0,1])+o]
                release[k]=k*2.5
            else:
                spawn[k]=np.array([-1.,0,1])+o
                end=np.array([0.2 if level==0 else 1.5,0,1])+o
                if level==0:
                    # One actual target, repeated only to keep a fixed tensor shape.
                    distance=rng.uniform(.6,1.8)
                    if rng.random()<practice: distance=rng.uniform(.02,.35)
                    end=spawn[k]+[distance,rng.uniform(-.35,.35),rng.uniform(-.12,.12)]
                    routes[k]=end
                    initial_velocity[k]=rng.uniform(-.08,.08,3)
                else:
                    routes[k]=[spawn[k]+[.5,.3,0],spawn[k]+[1.,.3,0],end+[-.3,0,0],end]
        if cfg.mission_mode=='region':
            routes[:k_active,-1]=routes[:k_active,-1].mean(axis=0)
        if level==5:
            lo=np.array([[-1,-.6,.4],[-1,-.6,1.5],[-1,-.6,.5],[-1,.5,.5]],np.float32)
            hi=np.array([[1,.6,.5],[1,.6,1.6],[1,-.5,1.5],[1,.6,1.5]],np.float32)
        if level>=2:
            goal=routes[0,-1]
            # Occupied secondary slots, outside all assigned slots and planned path.
            static[:]=goal+np.array([[0,1.2,0],[0,-1.2,0],[1.2,1.2,0],[1.2,-1.2,0]])
            for i in range(4):
                while (np.linalg.norm(routes[:k_active,-1]-static[i],axis=-1).min()<cfg.separation or
                       path_sphere_clearance(np.concatenate([spawn[:k_active,None],routes[:k_active]],1),static[i:i+1],cfg.drone_radius+cfg.obstacle_radius)<=.02):
                    static[i,1]+=np.sign(static[i,1]-goal[1])*.3
        if level in (3,4):
            dc=np.array([[.25,.8,1],[.8,-.8,1]],np.float32)
            da=np.array([[0,.6,0],[0,.6,0]],np.float32)
        if cfg.dynamic_scale == 0:
            dc[:]=100.; da[:]=0.
        else:
            da*=cfg.dynamic_scale
        # All spatial entities receive the SAME arbitrary yaw and translation.
        angle=float(rng.uniform(-np.pi,np.pi))
        R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]],np.float32)
        # AABBs remain AABBs under quarter turns; arbitrary yaw for open scenes.
        if level==5:
            angle=float(rng.integers(4)*np.pi/2)
            R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]],np.float32)
        shift=np.array([*rng.uniform(-.3,.3,2),0],np.float32)
        bcenter=(lo+hi)/2@R.T+shift; bh=(hi-lo)/2@np.abs(R.T)
        path=np.concatenate([spawn[:,None],routes],axis=1)
        lengths=np.linalg.norm(np.diff(path,axis=1),axis=-1).sum(-1)
        speed=.25 if level==5 else cfg.max_speed*.7
        horizon=float(np.max((lengths/speed+release)[:k_active])+cfg.hold_seconds+cfg.preflight_seconds+rng.uniform(5.,7.))
        if cfg.enclosed_room:
            # A mapped indoor room, built around the transformed mission. Four
            # actual walls give ToF position information, not invisible beacons.
            mission=np.concatenate([spawn[:k_active],routes[:k_active].reshape(-1,3)])@R.T+shift
            low=mission[:,:2].min(0)-cfg.room_margin; high=mission[:,:2].max(0)+cfg.room_margin
            x0,y0=low; x1,y1=high; thick=.12
            room_lo=np.array([[x0-thick,y0-thick,0],[x1,y0-thick,0],[x0,y0-thick,0],[x0,y1,0]],np.float32)
            room_hi=np.array([[x0,y1+thick,3],[x1+thick,y1+thick,3],[x1,y0,3],[x1,y1+thick,3]],np.float32)
            world_lo=np.concatenate([bcenter-bh,room_lo]); world_hi=np.concatenate([bcenter+bh,room_hi])
        else:
            world_lo,world_hi=bcenter-bh,bcenter+bh
        values=(spawn@R.T+shift,routes@R.T+shift,np.arange(K)<k_active,
                world_lo,world_hi,static@R.T+shift,dc@R.T+shift,da@R.T,cfg.preflight_seconds,np.full((2,1),.7*cfg.dynamic_speed_scale,np.float32),dp,
                initial_velocity@R.T,angle+rng.uniform(-.6,.6,K),start_index+w,
                release,horizon,level,angle,rng.integers(0,2**32,(2,),dtype=np.uint32))
        for name,value in zip(Scenario._fields,values): fields[name].append(value)
    arrays=[np.asarray(fields[name]) for name in Scenario._fields]
    host=Scenario(*arrays)
    validate_scenario(host,cfg)
    if paired: arrays=[np.concatenate([a,a],axis=0) for a in arrays]
    return Scenario(*(jnp.asarray(a) for a in arrays))


def obstacle_states(s,t):
    elapsed=jnp.maximum(t-s.dynamic_start,0.)
    phase=elapsed[...,None,None]*s.dynamic_frequency+s.dynamic_phase
    pos=s.dynamic_center+s.dynamic_amplitude*jnp.sin(phase)
    vel=jnp.where((t>=s.dynamic_start)[...,None,None],s.dynamic_frequency*s.dynamic_amplitude*jnp.cos(phase),0.)
    return pos,vel
