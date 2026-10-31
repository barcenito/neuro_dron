"""Local missions, no route supplied to the policy. Deterministic per mission ID."""
import numpy as np
import jax.numpy as jnp
from drone_swarm.simulation.scenarios import Scenario

STAGES=('hover','open_goal','static_obstacles','moving_obstacles','swarm')
ENCOUNTER_HOLD_START=12.  # last first-pass crossing is at 9..11 s; then regain hold

def targeted_encounters(cfg):
    return cfg.direct_curriculum_version>=3 and cfg.dynamic_scale>0 and cfg.dynamic_speed_scale>0

def hold_ready(s,time,cfg):
    """Do not finish before late encounters; accumulate a fresh hold afterwards."""
    if targeted_encounters(cfg):return (s.stage<3)|(time>=ENCOUNTER_HOLD_START)
    return jnp.ones_like(s.stage,dtype=bool)

def encounter_motion(rng,spawn,goal,cfg):
    """Prescribed paths, not pursuit or privileged policy input. Frequency is rad/s."""
    fraction=max(rng.uniform(.65,.8),(.25-spawn[0])/(goal[0]-spawn[0]))
    center=np.stack([spawn+fraction*(goal-spawn),goal,goal.copy()]).astype(np.float32)
    # Sweep an off-center part of the allowed region as well as its center.
    center[2,0]-=rng.uniform(.25,.5)
    amplitude=np.zeros((3,3),np.float32)
    amplitude[0,1]=rng.uniform(.9,1.1)*cfg.dynamic_scale
    amplitude[1,1]=rng.uniform(1.,1.2)*cfg.dynamic_scale
    amplitude[2,2]=rng.uniform(.45,.6)*cfg.dynamic_scale
    # Keep complete sphere paths inside the room, including the late maze's far side.
    margin=cfg.obstacle_radius+.05
    for j,axis,low,high in [(0,1,-2.9,3.),(1,1,-2.9,3.),(2,2,0.,3.)]:
        amplitude[j,axis]=min(amplitude[j,axis],center[j,axis]-low-margin,high-margin-center[j,axis])
        amplitude[j,axis]*=rng.choice([-1.,1.])
    crossing=np.array([rng.uniform(1.5,2.5),rng.uniform(6.,8.),rng.uniform(9.,11.)])
    frequency=np.array([rng.uniform(.65,.95),rng.uniform(.22,.3),rng.uniform(.15,.23)])
    # Avoid unintended earlier zero crossings even with a changed speed scale.
    frequency=np.minimum(frequency*cfg.dynamic_speed_scale,(np.pi-.1)/crossing).astype(np.float32)[:,None]
    phase=(-frequency*crossing[:,None]).astype(np.float32)
    return center,amplitude,frequency,phase

def varied_encounter_motion(rng,spawn,goal,cfg,progress=None):
    """v4: bounded oblique sweeps across varied parts of the arrival region.

    None means full evaluation (three movers). Training retains a mixture of
    one, two and three movers, increasing speed and available count gradually.
    No trajectory depends on the drone's evolving position.
    """
    center=np.tile(goal,(3,1)).astype(np.float32)
    if rng.random()<.5:
        fraction=max(rng.uniform(.65,.9),(.4-spawn[0])/(goal[0]-spawn[0]))
        center[0]=spawn+fraction*(goal-spawn)
    # Sample a disk rather than always sweeping the same side of the target.
    for k in range(3):
        angle=rng.uniform(-np.pi,np.pi);radius=.45*np.sqrt(rng.random())
        center[k,:2]+=radius*np.array([np.cos(angle),np.sin(angle)])
        center[k,2]+=rng.uniform(-.15,.15)
    margin=cfg.obstacle_radius+.06
    low=np.array([.05+margin,-2.9+margin,margin])
    high=np.array([3.-margin,3.-margin,3.-margin])
    center=np.clip(center,low+.06,high-.06).astype(np.float32)
    direction=rng.normal(size=(3,3));direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    extent=rng.uniform(.7,1.3,3)*cfg.dynamic_scale
    room=np.minimum(center-low,high-center)
    limit=np.min(room/np.maximum(np.abs(direction),1e-8),axis=1)
    amplitude=(direction*np.minimum(extent,limit)[:,None]).astype(np.float32)
    crossing=np.array([rng.uniform(2.,5.),rng.uniform(5.,8.),rng.uniform(8.,11.)])
    speed=1. if progress is None else .5+.5*progress
    frequency=np.minimum(rng.uniform(.3,.9,3)*cfg.dynamic_speed_scale*speed,
                         (np.pi-.1)/crossing).astype(np.float32)[:,None]
    phase=(-frequency*crossing[:,None]).astype(np.float32)
    if progress is not None:
        maximum=1+int(progress>=.33)+int(progress>=.67)
        count=rng.integers(1,maximum+1)
        # Keep late encounters even in single-obstacle training.
        active=rng.permutation(3)[:count]
        disabled=np.ones(3,bool);disabled[active]=False
        center[disabled]=100.;amplitude[disabled]=0.
    return center,amplitude,frequency,phase

def curriculum_stages(cfg):
    return STAGES if cfg.direct_curriculum_version==1 else STAGES[:4]+('maze',)

def make_worlds(seed,worlds,cfg,stage=0,start_index=0,paired=False,encounter_progress=None):
    if encounter_progress is not None and (cfg.direct_curriculum_version!=4 or not 0<=encounter_progress<=1):
        raise ValueError('Encounter progress requires v4 and a value in [0,1]')
    if stage not in range(len(STAGES)) or worlds<1 or (paired and worlds%2):
        raise ValueError('Invalid stage/world count')
    if targeted_encounters(cfg) and stage>=3 and cfg.episode_seconds<=ENCOUNTER_HOLD_START+cfg.hold_seconds:
        raise ValueError('Encounter horizon must exceed 12 seconds plus required hold')
    n=worlds//2 if paired else worlds
    fields={name:[] for name in Scenario._fields}; K=cfg.swarm_size
    for i in range(n):
        rng=np.random.default_rng(np.random.SeedSequence([seed,start_index+i]))
        # All configured drones are active: collective evaluation never hides extras.
        spawn=np.array([[-1.2-.5*(k//3),.5*((k%3)-1) if K>1 else 0.,1.] for k in range(K)],np.float32)
        spawn+=rng.uniform(-.025,.025,spawn.shape)
        goal=np.array([rng.uniform(.8,1.8),rng.uniform(-.5,.5),rng.uniform(.85,1.25)],np.float32)
        if stage==0: goal=spawn[0]+rng.uniform(-.05,.05,3)
        routes=np.broadcast_to(goal,(K,1,3)).copy()
        # Room boundary + two optional wall segments with a traversable gap.
        lo=np.array([[-4.,-3.,0.],[3.,-3.,0.],[-4.,-3.,0.],[-4.,3.,0.],[-4.,-3.,3.],
                     [100.,100.,100.],[100.,100.,100.]],np.float32)
        hi=np.array([[-3.9,3.1,3.],[3.1,3.1,3.],[3.,-2.9,3.],[3.,3.1,3.], [3.,3.,3.1],
                     [101.,101.,101.],[101.,101.,101.]],np.float32)
        static=np.full((3,3),100.,np.float32)
        if stage>=2:
            gap_y=rng.uniform(-.6,.6); gap=rng.uniform(.9,1.5)
            # Keep random draws stable across curriculum versions for paired comparisons.
            # v2 separates open obstacle avoidance from the late maze stage.
            if cfg.direct_curriculum_version==1 or stage>=4:
                lo[5]=[-.05,-2.9,0.];hi[5]=[.05,gap_y-gap/2,3.]
                lo[6]=[-.05,gap_y+gap/2,0.];hi[6]=[.05,3.,3.]
            for j in range(3):
                for _ in range(50):
                    p=np.array([rng.uniform(.35,2.1),rng.uniform(-1.5,1.5),rng.uniform(.5,1.8)])
                    if np.linalg.norm(p-goal)>.95 and np.min(np.linalg.norm(spawn-p,axis=-1))>.4:
                        static[j]=p;break
        dc=np.full((3,3),100.,np.float32);da=np.zeros((3,3),np.float32)
        frequency=rng.uniform(.4,1.8,(3,1)).astype(np.float32)*cfg.dynamic_speed_scale
        phase=rng.uniform(0,2*np.pi,(3,1)).astype(np.float32)
        if stage>=3 and cfg.dynamic_scale:
            for j in range(3):
                dc[j]=[.4+j*.5,rng.uniform(-.3,.3),rng.uniform(.7,1.4)]
                da[j]=[0.,rng.uniform(.4,.9)*cfg.dynamic_scale,0.]
        yaw=rng.uniform(-np.pi,np.pi,K).astype(np.float32)
        velocity=rng.uniform(-.04,.04,(K,3)).astype(np.float32)
        values=(spawn,routes,np.ones(K,bool),lo,hi,static,dc,da,0.,frequency,phase,
                velocity,yaw,start_index+i,np.zeros(K,np.float32),cfg.episode_seconds,stage,0.,
                rng.integers(0,2**32-1,2,dtype=np.uint32))
        if targeted_encounters(cfg) and stage>=3:
            # Separate stream preserves all other scenario draws and v1/v2 checkpoints.
            encounter_rng=np.random.default_rng(np.random.SeedSequence([seed,start_index+i,303]))
            if cfg.direct_curriculum_version==4:
                dc,da,frequency,phase=varied_encounter_motion(encounter_rng,spawn[0],goal,cfg,encounter_progress)
            else:
                dc,da,frequency,phase=encounter_motion(encounter_rng,spawn[0],goal,cfg)
            values=list(values)
            values[6:11]=[dc,da,0.,frequency,phase]
        for name,value in zip(Scenario._fields,values):fields[name].append(value)
    arrays={name:np.asarray(value) for name,value in fields.items()}
    if paired:arrays={name:np.concatenate([a,a]) for name,a in arrays.items()}
    return Scenario(**{name:jnp.asarray(a) for name,a in arrays.items()})
