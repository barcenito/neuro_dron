"""Sensors + simulated relative navigation -> four motor thrusts. No flight controller."""
from typing import NamedTuple
import math
import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control
from drone_swarm.runtime import configure_device
from drone_swarm.simulation.geometry import inverse_rotate
from drone_swarm.simulation.environment import contacts,arrival_mask,CAUSES
from drone_swarm.simulation.scenarios import obstacle_states
from drone_swarm.direct.worlds import hold_ready
from drone_swarm.perception.sensors import cone_directions,measure_ranges,init_sensors,update_sensors,world_random
from drone_swarm.perception.estimation import sample_imu
from drone_swarm.learning.network import mlp_forward

class DirectEpisode(NamedTuple):
    data: object
    sensors: object
    history: object
    motors: object
    previous: object
    step: object
    alive: object
    success: object
    failed: object
    timed_out: object
    hold: object
    max_hold: object
    reward: object
    exposure: object
    error: object
    speed: object
    causes: object
    effort: object

def motor_thrust(action,cfg):
    """Static actuator conversion, not stabilization. Full individual motor authority."""
    return (jnp.clip(action,-1.,1.)+1.)*.5*cfg.max_thrust_per_motor

def direct_reward_v2(density,alive,success,failed,timeout,time,horizon,dt):
    """Fixed-horizon absorbing costs; early failure cannot avoid future costs.

    Remove the negative distance-progress sum that rewarded dying near spawn.
    Successful endpoints receive the remaining optimal occupancy value.
    """
    remaining=jnp.maximum(horizon-time,0.)
    return alive*jnp.clip(density,0.,2.)*dt+success*(8.+2.*remaining)-failed*(8.+2.*remaining)-timeout

def direct_reward_v3(density,alive,success,failed,timeout,time,horizon,dt):
    """Nonnegative stable-flight rate; failure cannot erase negative future terms.

    No survival payment for the remaining horizon. Successful termination keeps
    its optimal remaining value so delaying success cannot improve the return.
    """
    remaining=jnp.maximum(horizon-time,0.)
    return alive*jnp.clip(density,0.,2.)*dt+success*(8.+2.*remaining)-8.*failed-timeout

def stability_density(upright,angular_speed,error,speed,effort,smooth,cfg):
    near=jnp.exp(-(jnp.maximum(error-(cfg.region_radius-cfg.slot_tolerance),0.)/.7)**2)
    slow=jnp.exp(-(speed/.6)**2)
    stable=jnp.exp(-(angular_speed/2.)**2)
    return jnp.clip(2.*upright**2*stable*(.25+.75*near*slow)
                    -cfg.effort_weight*effort-.05*smooth,0.,2.)

def preclaude_density(upright,angular_speed,error,speed,cfg):
    """Legacy broad proximity + settling incentive, rescaled to [0,2] per second.

    Use distance outside the inner free region: no forced center or fixed slots.
    A near-goal speed/attitude defect does not erase the entire proximity signal.
    """
    distance=jnp.maximum(error-(cfg.region_radius-cfg.slot_tolerance),0.)
    proximity=10.-jnp.clip(distance,0.,10.)
    penalty=.05*angular_speed**2+(.005+.1*jnp.exp(-2.*distance))*speed**2
    hold=2.*jnp.exp(-(distance/.5)**2-(speed/.3)**2-angular_speed**2)*upright**2
    return (jnp.maximum(proximity-penalty,0.)+hold)/6.

def direct_reward_v4(density,alive,success,failed,time,horizon,dt):
    """No negative live return to escape; failed states receive no further score."""
    return jnp.where(alive & ~failed,jnp.clip(density,0.,2.)*dt,0.)+success*2.*jnp.maximum(horizon-time,0.)

def clearance_density(density,clearance):
    """Training-only physical surface distance; never added to policy observations.

    Retain a nonnegative flight incentive, but reduce it near walls/obstacles/peers.
    A safe maneuver can earn more than holding still in the path of an obstacle.
    """
    proximity=jnp.clip(1.-clearance/.35,0.,1.)
    return density*(1.-.75*proximity**2)

def direct_reward_v5(density,alive,success,failed,time,horizon,dt):
    return direct_reward_v4(density,alive,success,failed,time,horizon,dt)+8.*success-8.*failed

def observe(imu,relative_goal,body_velocity,sensors,previous,cfg):
    """Explicit contract: relative goal/velocity are simulator navigation, not IMU estimates."""
    gravity=inverse_rotate(imu.attitude,jnp.broadcast_to(jnp.array([0.,0.,-1.]),relative_goal.shape))
    if cfg.direct_observation_version==2:
        base=jnp.concatenate([gravity,jnp.clip(imu.gyro,-20.,20.),
            jnp.clip(relative_goal/2.,-5.,5.),jnp.clip(body_velocity/cfg.max_speed,-3,3),previous],-1)
        # One channel per ToF: 0 means free to max range, 1 near contact,
        # -1 unavailable/stale. No exact obstacle velocity or position is added.
        fresh=sensors.valid & (sensors.age <= (cfg.sensor_period_steps+cfg.sensor_latency_steps)*cfg.dt)
        ranges=jnp.where(fresh,1.-jnp.clip(sensors.distance/cfg.lidar_max_range,0.,1.),-1.)
        return jnp.concatenate([base,ranges],-1)
    base=jnp.concatenate([gravity,jnp.clip(imu.gyro/10.,-3,3),
                          jnp.clip(relative_goal/5.,-2,2),jnp.clip(body_velocity/cfg.max_speed,-3,3),previous],-1)
    return jnp.concatenate([base,sensors.distance/cfg.lidar_max_range,sensors.valid.astype(jnp.float32),jnp.clip(sensors.age,0,1)],-1)

def navigation(states,goal,keys,step,cfg,noise):
    # Deliberate oracle at the simulator boundary, authorized for this research phase.
    delta=goal-states.pos+noise*cfg.navigation_position_noise*world_random(keys,step,61,states.pos.shape[1:],True)
    velocity=states.vel+noise*cfg.navigation_velocity_noise*world_random(keys,step,62,states.vel.shape[1:],True)
    return inverse_rotate(states.quat,delta),inverse_rotate(states.quat,velocity)

class DirectEnvironment:
    def __init__(self,cfg,worlds):
        self.cfg,self.device=configure_device(cfg);self.worlds=worlds
        self.sim=Sim(n_worlds=worlds,n_drones=cfg.swarm_size,physics='analytical',
                     control=Control.thrust,freq=cfg.physics_freq,thrust_freq=cfg.physics_freq,device=self.cfg.device)
        self.cones=cone_directions(cfg);self.step=jax.jit(self._step)
        self._rollouts={}

    def reset(self,s,noise=1.,difficulty=1.):
        cfg=self.cfg;W,K=s.active.shape;t=self.sim.data
        # Small real attitude perturbation; no programmatic recovery on reset.
        rollpitch=world_random(s.noise_keys,0,63,(K,2),True)*jnp.deg2rad(cfg.initial_tilt_degrees)*difficulty/2
        axis=jnp.concatenate([rollpitch,jnp.zeros((W,K,1))],-1)
        norm=jnp.linalg.norm(axis,axis=-1,keepdims=True)
        axis=axis*jnp.minimum(1.,.3/jnp.maximum(norm,1e-6))
        q0=jnp.concatenate([axis,jnp.sqrt(1-jnp.sum(axis**2,-1,keepdims=True))],-1)
        z=jnp.sin(s.initial_yaw/2);c=jnp.cos(s.initial_yaw/2)
        q=jnp.stack([c*q0[...,0]-z*q0[...,1],c*q0[...,1]+z*q0[...,0],z*q0[...,3],c*q0[...,3]],-1)
        mass=cfg.nominal_mass*(1+cfg.mass_variation*difficulty*(2*world_random(s.noise_keys,0,1,(K,1))-1))
        factor=1+cfg.inertia_variation*difficulty*(2*world_random(s.noise_keys,0,2,(K,1,1))-1)
        states=t.states.replace(pos=s.spawn,vel=s.initial_velocity*difficulty,quat=q,ang_vel=jnp.zeros((W,K,3)))
        params=t.params.replace(mass=mass,J=t.params.J*factor,J_INV=t.params.J_INV/factor)
        data=t.replace(states=states,params=params)
        previous=jnp.zeros((W,K,4))
        # Start in flight with nominal-hover rotor thrust, not an airborne motor-off reset.
        motors=jnp.full((W,K,4),cfg.nominal_mass*9.81/4)
        data=data.replace(controls=data.controls.replace(thrust=motors))
        sensors=init_sensors(s.noise_keys,K,cfg)
        imu=sample_imu(states,states,s.noise_keys,0,cfg,noise)
        delta,vel=navigation(states,s.routes[:,:,-1],s.noise_keys,0,cfg,noise)
        obs=observe(imu,delta,vel,sensors,previous,cfg)
        z=jnp.zeros(W);false=jnp.zeros(W,bool)
        return DirectEpisode(data,sensors,jnp.repeat(obs[:,:,None],cfg.num_frames,axis=2),motors,previous,jnp.int32(0),
            ~false,false,false,false,z,z,z,z,jnp.linalg.norm(delta,axis=-1),jnp.linalg.norm(states.vel,axis=-1),
            jnp.zeros((W,len(CAUSES)),bool),z)

    def _step(self,ep,action,s,noise=1.):
        cfg=self.cfg;W,K=s.active.shape
        bad=jnp.any(~jnp.isfinite(action),axis=(1,2))
        action=jnp.clip(jnp.nan_to_num(action),-1,1);desired=motor_thrust(action,cfg)
        wind=world_random(s.noise_keys,ep.step,3,(K,3),True)*cfg.wind_force*noise
        def substep(i,carry):
            data,motors,causes,clearance=carry
            motors=motors+(1-jnp.exp(-1/(cfg.physics_freq*cfg.motor_tau)))*(desired-motors)
            commanded=data.replace(controls=data.controls.replace(thrust=motors))
            nxt=self.sim._step(commanded,n_steps=1)
            acc=wind/data.params.mass
            nxt=nxt.replace(states=nxt.states.replace(vel=nxt.states.vel+acc/cfg.physics_freq,
                pos=nxt.states.pos+.5*acc/cfg.physics_freq**2,force=nxt.states.force+wind))
            hit,gap,_=contacts(data.states,nxt.states,s,ep.step*cfg.dt+(i+1)/cfg.physics_freq,cfg)
            stopped=jnp.any(causes,-1)|~ep.alive
            newstates=jax.tree_util.tree_map(lambda old,new:jnp.where(stopped[:,None,None],old,new),data.states,nxt.states)
            return nxt.replace(states=newstates),motors,causes|(jnp.any(hit,axis=1)&~stopped[:,None]),jnp.minimum(clearance,gap)
        initial_causes=jnp.zeros_like(ep.causes).at[:,7].set(bad)
        data,motors,causes,clearance=jax.lax.fori_loop(0,cfg.physics_freq//cfg.env_freq,substep,
            (ep.data,ep.motors,initial_causes,jnp.full((W,K),jnp.inf)))
        dyn,_=obstacle_states(s,jnp.full((W,),(ep.step+1)*cfg.dt))
        ideal=measure_ranges(data.states.pos,data.states.quat,s,dyn,cfg,self.cones)
        sensors=update_sensors(ep.sensors,ideal,s.noise_keys,ep.step+1,cfg,noise)
        imu=sample_imu(ep.data.states,data.states,s.noise_keys,ep.step+1,cfg,noise)
        delta,velocity=navigation(data.states,s.routes[:,:,-1],s.noise_keys,ep.step+1,cfg,noise)
        obs=observe(imu,delta,velocity,sensors,action,cfg)
        history=jnp.concatenate([ep.history[:,:,1:],obs[:,:,None]],axis=2)
        error=jnp.linalg.norm(data.states.pos-s.routes[:,:,-1],axis=-1)
        speed=jnp.linalg.norm(data.states.vel,axis=-1)
        failed=jnp.any(causes,-1)&ep.alive
        occupied=arrival_mask(error,speed,data.states.pos,s.active,cfg)
        angular_speed=jnp.linalg.norm(data.states.ang_vel,axis=-1)
        if cfg.direct_reward_version>=2:
            occupied &= angular_speed<1.
        holding=jnp.all(occupied|~s.active,-1)&~failed&hold_ready(s,(ep.step+1)*cfg.dt,cfg)
        hold=jnp.where(holding,ep.hold+cfg.dt,0.)
        success=(hold>=cfg.hold_seconds)&ep.alive
        timeout=((ep.step+1)*cfg.dt>=s.horizon)&ep.alive&~success&~failed
        count=jnp.maximum(s.active.sum(-1),1)
        mean=lambda x:(x*s.active).sum(-1)/count
        inner=cfg.region_radius-cfg.slot_tolerance
        # Dense, bounded score: progress + staying upright/near/slow, no route obedience.
        progress=mean(ep.error-error)
        upright=jnp.clip(1-2*(data.states.quat[...,0]**2+data.states.quat[...,1]**2),0,1)
        settle=jnp.exp(-(jnp.maximum(error-inner,0)/.7)**2-(speed/.4)**2)
        effort=jnp.mean((motors/cfg.max_thrust_per_motor)**2,-1)
        smooth=jnp.mean(((action-ep.previous)/2)**2,-1)
        density=mean(.2*upright+settle-cfg.effort_weight*effort-.05*smooth)
        reward=ep.reward+ep.alive*(2*progress+density*cfg.dt)+8*success-8*failed-timeout
        if cfg.direct_reward_version>=2:
            stable=jnp.exp(-(angular_speed/5.)**2)
            density=mean(.2+.3*upright*stable+1.5*settle*stable-cfg.effort_weight*effort-.05*smooth)
            reward=ep.reward+direct_reward_v2(density,ep.alive,success,failed,timeout,
                                             (ep.step+1)*cfg.dt,s.horizon,cfg.dt)
        if cfg.direct_reward_version==3:
            density=mean(stability_density(upright,angular_speed,error,speed,effort,smooth,cfg))
            reward=ep.reward+direct_reward_v3(density,ep.alive,success,failed,timeout,
                                             (ep.step+1)*cfg.dt,s.horizon,cfg.dt)
        if cfg.direct_reward_version>=4:
            density=preclaude_density(upright,angular_speed,error,speed,cfg)
            if cfg.direct_reward_version==5:density=clearance_density(density,clearance)
            reward_fn=direct_reward_v5 if cfg.direct_reward_version==5 else direct_reward_v4
            reward=ep.reward+reward_fn(mean(density),ep.alive,success,failed,
                                             (ep.step+1)*cfg.dt,s.horizon,cfg.dt)
        return DirectEpisode(data,sensors,history,motors,action,ep.step+1,ep.alive&~failed&~success&~timeout,
            ep.success|success,ep.failed|failed,ep.timed_out|timeout,jnp.where(ep.alive,hold,ep.hold),
            jnp.maximum(ep.max_hold,jnp.where(ep.alive,hold,0)),reward,ep.exposure+ep.alive*cfg.dt,
            jnp.where(ep.alive[:,None],error,ep.error),jnp.where(ep.alive[:,None],speed,ep.speed),
            ep.causes|(causes&ep.alive[:,None]),ep.effort+mean(effort)*cfg.dt*ep.alive)

    def rollout(self,weights,s,mutations=None,noise=1.,max_steps=None,difficulty=1.):
        initial=self.reset(s,noise,difficulty);has_mut=mutations is not None
        if has_mut not in self._rollouts:
            def run(ep,weights,s,mutations,noise,limit):
                def body(ep):
                    obs=ep.history.reshape((self.worlds,self.cfg.swarm_size,-1))
                    action=(jax.vmap(mlp_forward,in_axes=(0,None,0))(obs,weights,mutations) if has_mut else mlp_forward(obs,weights))
                    return self._step(ep,action,s,noise)
                return jax.lax.while_loop(lambda e:(e.step<limit)&jnp.any(e.alive),body,ep)
            self._rollouts[has_mut]=jax.jit(run)
        limit=max_steps if max_steps is not None else math.ceil(self.cfg.episode_seconds/self.cfg.dt)
        return self._rollouts[has_mut](initial,weights,s,mutations,noise,jnp.int32(limit))

def summarize(ep,include_stability=False):
    ep=jax.device_get(ep)
    result={'episodes':len(ep.success),'success_rate':float(np.mean(ep.success)),
            'collision_rate':float(np.mean(ep.causes[:,:5].any(-1))), 'failure_rate':float(np.mean(ep.failed)),
            'timeout_rate':float(np.mean(ep.timed_out)),'truncated_rate':float(np.mean(ep.alive)),
            'fitness_mean':float(np.mean(ep.reward)),'goal_distance_mean':float(np.mean(ep.error)),
            'speed_mean':float(np.mean(ep.speed)),'hold_max_mean':float(np.mean(ep.max_hold)),
            'flight_seconds_mean':float(np.mean(ep.exposure)),
            **{'hit_'+cause:float(np.mean(ep.causes[:,i])) for i,cause in enumerate(CAUSES)}}
    if include_stability:
        result['angular_speed_terminal_mean']=float(np.mean(np.linalg.norm(ep.data.states.ang_vel,axis=-1)))
    return result
