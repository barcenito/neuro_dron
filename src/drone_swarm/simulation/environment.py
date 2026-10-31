"""One pure transition for training, evaluation and rendering, with K agents/world."""
from drone_swarm.runtime import configure_device
from typing import NamedTuple
import math
import jax
import jax.numpy as jnp
from crazyflow.sim import Sim
from crazyflow.control import Control
from drone_swarm.config import SwarmConfig
from drone_swarm.simulation.geometry import inverse_rotate,box_distance,ray_boxes,segment_spheres
from drone_swarm.perception.sensors import (cone_directions,measure_ranges,init_sensors,update_sensors,world_random)
from drone_swarm.perception.estimation import init_estimate,update_estimate,sample_imu
from drone_swarm.control.control import init_controller,velocity_control
from drone_swarm.simulation.scenarios import obstacle_states

CAUSES=('ground','maze','neighbor','dynamic','static','bounds','spin','nonfinite','localization')

class Episode(NamedTuple):
    data: object
    estimate: object
    sensors: object
    controller: object
    history: object
    waypoint: object
    previous_action: object
    step: object
    alive: object
    success: object
    failed: object
    hold_clock: object
    hold_integral: object
    cohesion_integral: object
    smooth_integral: object
    risk_integral: object
    progress: object
    causes: object
    min_clearance: object
    end_error: object
    end_speed: object
    time_to_success: object
    exposure: object
    compact_time: object
    timed_out: object
    max_hold: object
    settling_integral: object
    action_integral: object
    route_waypoint: object
    min_neighbor_clearance: object
    saturation_integral: object


def route_progress(pos,s,waypoint=None):
    points=jnp.concatenate([s.spawn[:,:,None],s.routes],axis=2)
    start,end=points[:,:,:-1],points[:,:,1:]
    delta=end-start
    lengths=jnp.linalg.norm(delta,axis=-1)
    u=jnp.clip(jnp.sum((pos[:,:,None]-start)*delta,-1)/(lengths**2+1e-8),0,1)
    nearest=start+u[...,None]*delta
    error=jnp.linalg.norm(pos[:,:,None]-nearest,axis=-1)
    if waypoint is not None:
        error=jnp.where(jnp.arange(s.routes.shape[2])<=waypoint[...,None],error,jnp.inf)
    idx=jnp.argmin(error,axis=-1)
    arclength=jnp.cumsum(lengths,axis=-1)-lengths+u*lengths
    travelled=jnp.take_along_axis(arclength,idx[...,None],-1)[...,0]
    cross=jnp.take_along_axis(error,idx[...,None],-1)[...,0]
    total=jnp.sum(lengths,axis=-1)
    return jnp.clip((travelled-cross)/jnp.maximum(total,.1),-1,1),travelled,total


def arrival_error(distance,cfg):
    """Reward error is zero throughout the inner arrival region; no center reward."""
    if cfg.mission_mode=='region':
        return jnp.maximum(distance-(cfg.region_radius-cfg.slot_tolerance),0.)
    return distance


def arrival_mask(distance,speed,pos,active,cfg):
    occupied=(distance<cfg.arrival_tolerance)&(speed<cfg.hold_speed)
    if cfg.mission_mode=='region':
        pair=jnp.linalg.norm(pos[:,:,None]-pos[:,None,:],axis=-1)-2*cfg.drone_radius
        pair_mask=active[:,:,None]&active[:,None,:]&~jnp.eye(pos.shape[1],dtype=bool)[None]
        occupied &= jnp.all(jnp.where(pair_mask,pair>=cfg.region_clearance,True),axis=-1)
    return occupied


def mission_target(estimate,s,waypoint,step,cfg,advance=True):
    target=jnp.take_along_axis(s.routes,waypoint[...,None,None],axis=2)[:,:,0]
    # Keep a margin for localization error. Otherwise a measured 15 cm pass
    # can miss the physical 15 cm route gate and strand a drone at the final slot.
    margin=jnp.minimum(estimate.uncertainty,cfg.waypoint_tolerance/3) if cfg.range_localization else 0.
    reached=jnp.linalg.norm(target-estimate.pos,axis=-1)<cfg.waypoint_tolerance-margin
    if advance:
        reached &= step*cfg.dt>=s.release+cfg.preflight_seconds
        waypoint=jnp.minimum(waypoint+reached.astype(jnp.int32),s.routes.shape[2]-1)
    target=jnp.take_along_axis(s.routes,waypoint[...,None,None],axis=2)[:,:,0]
    waiting=step*cfg.dt<s.release+cfg.preflight_seconds
    target=jnp.where(waiting[...,None],s.spawn,target)
    final=(waypoint==s.routes.shape[2]-1)&(~waiting)
    altitude=target[...,2]
    if cfg.mission_mode=='region':
        delta=target-estimate.pos
        distance=jnp.linalg.norm(delta,axis=-1,keepdims=True)
        inner=cfg.region_radius-cfg.slot_tolerance
        projected=estimate.pos+delta*jnp.maximum(1-inner/jnp.maximum(distance,1e-6),0.)
        target=jnp.where(final[...,None],projected,target)
    if cfg.level_flight:
        target=target.at[...,2].set(altitude)
    error=target-estimate.pos
    speed=jnp.where(s.stage==5,.25,cfg.max_speed*.7)[:,None,None]
    reference=error*cfg.guidance_kp
    reference=reference/jnp.maximum(1.,jnp.linalg.norm(reference,axis=-1,keepdims=True)/speed)
    phase=jnp.stack([waiting,~waiting&~final,final],-1).astype(jnp.float32)
    return waypoint,target,reference,phase


def observation(estimate,sensors,target,reference,phase,previous,cfg):
    gravity=inverse_rotate(estimate.quat,jnp.broadcast_to(jnp.array([0.,0.,-1.]),estimate.pos.shape))
    base=jnp.concatenate([gravity,jnp.clip(inverse_rotate(estimate.quat,estimate.vel)/cfg.max_speed,-3,3),
                          jnp.clip(estimate.gyro/5,-3,3),
                          jnp.clip(inverse_rotate(estimate.quat,target-estimate.pos)/3,-1,1),
                          inverse_rotate(estimate.quat,reference)/cfg.max_speed,
                          estimate.confidence[...,None],jnp.clip(estimate.uncertainty[...,None]/cfg.odometry_failure,0,1),
                          phase,previous],-1)
    # Channel-major order, explicitly versioned in checkpoints/export manifest.
    return jnp.concatenate([base,sensors.distance/cfg.lidar_max_range,sensors.valid.astype(jnp.float32),
                            jnp.clip(sensors.age,0,1)],-1)


def contacts(before,after,s,t,cfg):
    a,b=before.pos,after.pos
    W,K=a.shape[:2]
    delta=b-a; length=jnp.linalg.norm(delta,axis=-1)
    direction=delta/jnp.maximum(length[...,None],1e-8)
    bd=box_distance(b[:,:,None],s.boxes_lo[:,None],s.boxes_hi[:,None]).min(-1)-cfg.drone_radius
    ray=ray_boxes(a[:,:,None],direction[:,:,None],s.boxes_lo[:,None]-cfg.drone_radius,
                  s.boxes_hi[:,None]+cfg.drone_radius).min(-1)
    maze=(bd<=0)|(ray<=length)
    # Relative segment detects two agents moving through each other.
    pa=a[:,:,None]-a[:,None,:]; pb=b[:,:,None]-b[:,None,:]
    pmask=s.active[:,None,:]&~jnp.eye(K,dtype=bool)[None]
    neighbor=jnp.any(segment_spheres(pa,pb,0.,2*cfg.drone_radius)&pmask,-1)
    pd=jnp.where(pmask,jnp.linalg.norm(pb,axis=-1)-2*cfg.drone_radius,jnp.inf).min(-1)
    dyn0,_=obstacle_states(s,jnp.full((W,),t-1/cfg.physics_freq))
    dyn1,dynvel=obstacle_states(s,jnp.full((W,),t))
    hitdyn=jnp.any(segment_spheres(a[:,:,None]-dyn0[:,None],b[:,:,None]-dyn1[:,None],0.,cfg.drone_radius+cfg.obstacle_radius),-1)
    hitstatic=jnp.any(segment_spheres(a[:,:,None],b[:,:,None],s.static[:,None],cfg.drone_radius+cfg.obstacle_radius),-1)
    centers=jnp.concatenate([dyn1,s.static],1)
    sphere_delta=b[:,:,None]-centers[:,None]
    sd=jnp.linalg.norm(sphere_delta,axis=-1)-cfg.drone_radius-cfg.obstacle_radius
    sphere_min=sd.min(-1)
    ground=b[...,2]<=cfg.drone_radius
    bounds=(jnp.linalg.norm(b[...,:2],axis=-1)>15)|(b[...,2]>4)
    spin=jnp.linalg.norm(after.ang_vel,axis=-1)>22
    nonfinite=~(jnp.all(jnp.isfinite(b),-1)&jnp.all(jnp.isfinite(after.vel),-1)&jnp.all(jnp.isfinite(after.quat),-1))
    causes=jnp.stack([ground,maze,neighbor,hitdyn,hitstatic,bounds,spin,nonfinite,jnp.zeros_like(ground)],-1)&s.active[...,None]
    clearance=jnp.minimum(jnp.minimum(bd,pd),jnp.minimum(sphere_min,b[...,2]-cfg.drone_radius))
    clearance=jnp.where(s.active,clearance,jnp.inf)
    # True normal closing speed, used ONLY in training reward, never observation.
    closest=jnp.clip(b[:,:,None],s.boxes_lo[:,None],s.boxes_hi[:,None])
    normal=b[:,:,None]-closest
    normal=normal/jnp.maximum(jnp.linalg.norm(normal,axis=-1,keepdims=True),1e-6)
    wall_close=jnp.maximum(-jnp.sum(after.vel[:,:,None]*normal,-1),0)
    wall_d=box_distance(b[:,:,None],s.boxes_lo[:,None],s.boxes_hi[:,None])-cfg.drone_radius
    wall_risk=(jnp.maximum(1-wall_d/cfg.safe_clearance,0)**2*jnp.clip(wall_close/cfg.max_speed,0,1)).max(-1)
    rv=after.vel[:,:,None]-jnp.concatenate([dynvel,jnp.zeros_like(s.static)],1)[:,None]
    sn=sphere_delta/jnp.maximum(jnp.linalg.norm(sphere_delta,axis=-1,keepdims=True),1e-6)
    srisk=(jnp.maximum(1-sd/cfg.safe_clearance,0)**2*jnp.clip(-jnp.sum(rv*sn,-1)/cfg.max_speed,0,1)).max(-1)
    pn=pb/jnp.maximum(jnp.linalg.norm(pb,axis=-1,keepdims=True),1e-6)
    pclose=-jnp.sum((after.vel[:,:,None]-after.vel[:,None])*pn,-1)
    prisk=jnp.where(pmask,jnp.maximum(1-(jnp.linalg.norm(pb,axis=-1)-2*cfg.drone_radius)/cfg.safe_clearance,0)**2*jnp.clip(pclose/cfg.max_speed,0,1),0).max(-1)
    risk=jnp.clip(jnp.maximum(jnp.maximum(wall_risk,srisk),prisk),0,1)
    return causes,clearance,risk


def fitness(ep,s,cfg=None):
    cfg=cfg or SwarmConfig()
    active=s.active.astype(jnp.float32); count=jnp.maximum(active.sum(-1),1)
    progress=(ep.progress*active).sum(-1)/count
    terminal_settle=jnp.exp(-(arrival_error(ep.end_error,cfg)/cfg.reward_distance_scale)**2-(ep.end_speed/cfg.reward_speed_scale)**2)
    terminal_settle=(terminal_settle*active).sum(-1)/count
    efficiency=jnp.where(ep.success&jnp.isfinite(ep.time_to_success),
                         jnp.clip(1-jnp.nan_to_num(ep.time_to_success)/s.horizon,0,1),0.)
    return (cfg.success_time_weight*efficiency+progress+ep.settling_integral+.5*terminal_settle+2*ep.hold_integral+
            .5*ep.cohesion_integral-.1*ep.smooth_integral-.5*ep.risk_integral-
            cfg.effort_weight*ep.action_integral-6*ep.failed+4*ep.success-.5*ep.timed_out)


class SwarmEnvironment:
    def __init__(self,cfg: SwarmConfig,worlds=None):
        cfg,self.device=configure_device(cfg)
        self.cfg=cfg; self.worlds=worlds or cfg.es_population
        self.sim=Sim(n_worlds=self.worlds,n_drones=cfg.swarm_size,physics='analytical',
                     control=Control.attitude,freq=cfg.physics_freq,attitude_freq=cfg.physics_freq,device=cfg.device)
        self.template=self.sim.data
        self.cones=cone_directions(cfg)
        self.step=jax.jit(self._step)

    def reset(self,s,noise_scale=1.):
        cfg=self.cfg; W,K=s.active.shape
        q=jnp.zeros((W,K,4)).at[...,2].set(jnp.sin(s.initial_yaw/2)).at[...,3].set(jnp.cos(s.initial_yaw/2))
        pos=jnp.where(s.active[...,None],s.spawn,100.)
        states=self.template.states.replace(pos=pos,quat=q,vel=s.initial_velocity,ang_vel=jnp.zeros_like(pos))
        mass=cfg.nominal_mass*(1+cfg.mass_variation*(2*world_random(s.noise_keys,0,1,(K,1))-1))
        factor=1+cfg.inertia_variation*(2*world_random(s.noise_keys,0,2,(K,1,1))-1)
        params=self.template.params.replace(mass=mass,J=self.template.params.J*factor,J_INV=self.template.params.J_INV/factor)
        controls=self.template.controls.replace(last_rpy=jnp.stack([jnp.zeros((W,K)),jnp.zeros((W,K)),s.initial_yaw],-1))
        data=self.template.replace(states=states,params=params,controls=controls)
        estimate=init_estimate(pos,q,s.initial_velocity,s.noise_keys,cfg,noise_scale)
        sensors=init_sensors(s.noise_keys,K,cfg)
        controller=init_controller((W,K),s.initial_yaw)
        previous=jnp.zeros((W,K,4)); waypoint=jnp.broadcast_to(jnp.where(s.stage[:,None]==0,s.routes.shape[2]-1,0),(W,K)).astype(jnp.int32)
        _,target,ref,phase=mission_target(estimate,s,waypoint,0,cfg)
        obs=observation(estimate,sensors,target,ref,phase,previous,cfg)
        history=jnp.repeat(obs[:,:,None,:],cfg.num_frames,axis=2)
        z=jnp.zeros(W); false=jnp.zeros(W,bool)
        return Episode(data,estimate,sensors,controller,history,waypoint,previous,jnp.int32(0),
                       ~false,false,false,z,z,z,z,z,jnp.zeros((W,K)),jnp.zeros((W,len(CAUSES)),bool),
                       jnp.full((W,),jnp.inf),jnp.linalg.norm(s.routes[:,:,-1]-pos,axis=-1),
                       jnp.zeros((W,K)),jnp.full((W,),jnp.nan),z,z,
                       false,z,z,z,waypoint,jnp.full((W,),jnp.inf),z)

    def _step(self,ep,action,s,noise_scale):
        cfg=self.cfg; W,K=s.active.shape
        action=jnp.where(s.active[...,None],action,0.)
        attitude,controller=velocity_control(action,ep.estimate,ep.controller,cfg)
        data=ep.data.replace(controls=ep.data.controls.replace(staged_attitude=attitude))
        wind=world_random(s.noise_keys,ep.step,3,(K,3),True)*cfg.wind_force*noise_scale
        def substep(i,carry):
            data,causes,clearance,risk=carry
            nxt=self.sim._step(data,n_steps=1)
            # Analytical wrench overwrites force; integrate exogenous wind as an
            # acceleration split at every physical substep (not a velocity command).
            wa=wind/data.params.mass
            nxt=nxt.replace(states=nxt.states.replace(
                vel=nxt.states.vel+wa/cfg.physics_freq,
                pos=nxt.states.pos+.5*wa/cfg.physics_freq**2,
                force=nxt.states.force+wind))
            c,d,r=contacts(data.states,nxt.states,s,(ep.step*cfg.dt+(i+1)/cfg.physics_freq),cfg)
            # Freeze worlds at their first fatal substep (no runaway states or survivor reward).
            already=jnp.any(causes,axis=-1)|~ep.alive
            newstates=jax.tree_util.tree_map(lambda old,new:jnp.where(already[:,None,None],old,new),data.states,nxt.states)
            nxt=nxt.replace(states=newstates)
            c=jnp.any(c,axis=1)&(~already[:,None])
            return nxt,causes|c,jnp.minimum(clearance,jnp.where(already,jnp.inf,d.min(-1))),jnp.maximum(risk,jnp.where(already,0.,r.max(-1)))
        data,causes,clearance,risk=jax.lax.fori_loop(0,cfg.physics_freq//cfg.env_freq,substep,
                         (data,jnp.zeros((W,len(CAUSES)),bool),jnp.full((W,),jnp.inf),jnp.zeros(W)))
        dyn,_=obstacle_states(s,jnp.full((W,), (ep.step+1)*cfg.dt))
        ideal=measure_ranges(data.states.pos,data.states.quat,s,dyn,cfg,self.cones)
        sensor=update_sensors(ep.sensors,ideal,s.noise_keys,ep.step+1,cfg,noise_scale)
        imu=sample_imu(ep.data.states,data.states,s.noise_keys,ep.step+1,cfg,noise_scale)
        estimate=update_estimate(ep.estimate,imu,sensor,s,cfg,noise_scale,self.cones)
        estimate=jax.tree_util.tree_map(lambda old,new:jnp.where(ep.alive.reshape((W,)+(1,)*(new.ndim-1)),new,old),ep.estimate,estimate)
        waypoint,target,ref,phase=mission_target(estimate,s,ep.waypoint,ep.step+1,cfg)
        obs=observation(estimate,sensor,target,ref,phase,action,cfg)
        history=jnp.concatenate([ep.history[:,:,1:],obs[:,:,None]],axis=2)
        lost=jnp.any(((estimate.uncertainty>cfg.odometry_failure)|(estimate.altitude_age>cfg.altitude_timeout*2))&s.active,axis=-1)
        causes=causes.at[:,-1].set(lost)
        failed=jnp.any(causes,axis=-1)&ep.alive
        error=jnp.linalg.norm(data.states.pos-s.routes[:,:,-1],axis=-1)
        speed=jnp.linalg.norm(data.states.vel,axis=-1)
        physical_target=jnp.take_along_axis(s.routes,ep.route_waypoint[...,None,None],2)[:,:,0]
        reached=(jnp.linalg.norm(physical_target-data.states.pos,axis=-1)<cfg.waypoint_tolerance)&((ep.step+1)*cfg.dt>=s.release+cfg.preflight_seconds)
        route_waypoint=jnp.minimum(ep.route_waypoint+reached.astype(jnp.int32),s.routes.shape[2]-1)
        holding=arrival_mask(error,speed,data.states.pos,s.active,cfg)&(route_waypoint==s.routes.shape[2]-1)&((ep.step+1)*cfg.dt>=s.release+cfg.preflight_seconds)
        settling=jnp.exp(-(arrival_error(error,cfg)/cfg.reward_distance_scale)**2-(speed/cfg.reward_speed_scale)**2)
        settling*=route_waypoint==s.routes.shape[2]-1
        all_hold=jnp.all(holding|~s.active,axis=-1)&~failed
        clock=jnp.where(all_hold,ep.hold_clock+cfg.dt,0.)
        success=(clock>=cfg.hold_seconds)&ep.alive
        progress,travelled,_=route_progress(data.states.pos,s,route_waypoint)
        if cfg.mission_mode=='region':
            progress=jnp.where((route_waypoint==s.routes.shape[2]-1)&(error<cfg.arrival_tolerance),1.,progress)
        spread=jnp.max(jnp.where(s.active,travelled,-jnp.inf),-1)-jnp.min(jnp.where(s.active,travelled,jnp.inf),-1)
        # Narrow passage has deliberate queue offsets; no rigid-formation penalty.
        cohesion=jnp.where((s.stage==5)|(s.active.sum(-1)<=1),0.,jnp.exp(-spread))
        count=jnp.maximum(s.active.sum(-1),1)
        hold=(holding*s.active).sum(-1)/count
        smooth=(jnp.mean(((action-ep.previous_action)/2)**2,axis=-1)*s.active).sum(-1)/count
        alive=ep.alive.astype(jnp.float32)
        fraction=cfg.dt/s.horizon*alive
        timeout=((ep.step+1)*cfg.dt>=s.horizon)&ep.alive&~success&~failed
        pair=jnp.linalg.norm(data.states.pos[:,:,None]-data.states.pos[:,None,:],axis=-1)-2*cfg.drone_radius
        pair_mask=s.active[:,:,None]&s.active[:,None,:]&~jnp.eye(K,dtype=bool)[None]
        neighbor=jnp.min(jnp.where(pair_mask,pair,jnp.inf),axis=(1,2))
        effort=(jnp.mean(action**2,-1)*s.active).sum(-1)/count
        saturation=(jnp.mean(jnp.abs(action)>.99,-1)*s.active).sum(-1)/count
        return Episode(data,estimate,sensor,controller,history,waypoint,action,ep.step+1,
                       ep.alive&~failed&~success&~timeout,ep.success|success,ep.failed|failed,
                       jnp.where(ep.alive,clock,ep.hold_clock),ep.hold_integral+hold*fraction,
                       ep.cohesion_integral+cohesion*fraction,ep.smooth_integral+smooth*fraction,
                       ep.risk_integral+risk*fraction,jnp.where(ep.alive[:,None],progress,ep.progress),
                       ep.causes|(causes&ep.alive[:,None]),jnp.minimum(ep.min_clearance,clearance),
                       jnp.where(ep.alive[:,None],error,ep.end_error),jnp.where(ep.alive[:,None],speed,ep.end_speed),
                       jnp.where(success,(ep.step+1)*cfg.dt,ep.time_to_success),ep.exposure+alive*cfg.dt,
                       ep.compact_time+alive*(spread<=1.)*cfg.dt,
                       ep.timed_out|timeout,jnp.maximum(ep.max_hold,jnp.where(ep.alive,clock,0.)),
                       ep.settling_integral+(settling*s.active).sum(-1)/count*fraction,
                       ep.action_integral+effort*fraction,jnp.where(ep.alive[:,None],route_waypoint,ep.route_waypoint),
                       jnp.minimum(ep.min_neighbor_clearance,jnp.where(ep.alive,neighbor,jnp.inf)),
                       ep.saturation_integral+saturation*cfg.dt*alive)

    def rollout(self,initial,weights,s,mutations=None,noise_scale=1.,max_steps=None,baseline=False):
        from drone_swarm.learning.network import mlp_forward
        limit=max_steps if max_steps is not None else int(math.ceil(float(jnp.max(s.horizon))/self.cfg.dt))
        # The compiled function is cached by topology, not regenerated every generation.
        key=(mutations is not None,baseline)
        if not hasattr(self,'_rollouts'): self._rollouts={}
        if key not in self._rollouts:
            has_mut=mutations is not None
            def run(ep,weights,s,mutations,noise,limit):
                def body(ep):
                    if baseline:
                        action=baseline_action(ep,s,self.cfg)
                    else:
                        obs=ep.history.reshape((self.worlds,self.cfg.swarm_size,-1))
                        if has_mut: action=jax.vmap(mlp_forward,in_axes=(0,None,0))(obs,weights,mutations)
                        else: action=mlp_forward(obs,weights)
                        action=policy_action(ep,action,s,self.cfg)
                    return self._step(ep,action,s,noise)
                return jax.lax.while_loop(lambda ep:(ep.step<limit)&jnp.any(ep.alive),body,ep)
            self._rollouts[key]=jax.jit(run)
        return self._rollouts[key](initial,weights,s,mutations,jnp.float32(noise_scale),jnp.int32(limit))


def action_components(ep,residual,s,cfg):
    from drone_swarm.perception.sensors import get_lidar_directions
    from drone_swarm.control.avoidance import guard_velocity
    _,target,reference,phase=mission_target(ep.estimate,s,ep.waypoint,ep.step,cfg,advance=False)
    error=jnp.linalg.norm(ep.estimate.pos-target,axis=-1)
    guide=inverse_rotate(ep.estimate.quat,reference)
    directions=get_lidar_directions(cfg.num_lidars)
    near=jnp.maximum(1-ep.sensors.distance/.4,0)*ep.sensors.valid
    repulsion=-jnp.sum(near[...,:-2,None]*directions[:-2],axis=-2)*.3
    gate=jnp.where((phase[...,2]>0)|cfg.range_localization,jnp.clip(error/.5,.1,1.),1.)
    if cfg.mission_mode=='region':
        fresh=ep.sensors.valid&(ep.sensors.age<(cfg.sensor_period_steps+cfg.sensor_latency_steps+1)*cfg.dt)
        threat=jnp.max(jnp.where(fresh&(ep.sensors.distance<.6),jnp.clip(-ep.sensors.range_rate/.5,0,1),0.),axis=-1)
        gate=jnp.where(phase[...,2]>0,jnp.maximum(jnp.clip(error/.5,0,1),threat),gate)
    if cfg.preflight_seconds>0:
        gate=jnp.where(phase[...,0]>0,0.,gate)
    residual_scale=jnp.array([cfg.residual_speed,cfg.residual_speed,
        0. if cfg.level_flight else (cfg.residual_vertical_speed if cfg.control_profile=='guarded' else cfg.residual_speed)])
    residual_velocity=jnp.clip(residual[...,:3],-1,1)*gate[...,None]*residual_scale
    limits=jnp.array([cfg.max_speed,cfg.max_speed,cfg.max_vertical_speed])
    if cfg.control_profile=='legacy':
        base=jnp.clip((guide+repulsion)/limits,-1,1)*limits
        candidate=base+residual_velocity
        guarded=candidate
    else:
        damping=cfg.hold_damping*jnp.clip(1-error/.5,0,1)*(phase[...,2]>0)
        guide=guide-damping[...,None]*inverse_rotate(ep.estimate.quat,ep.estimate.vel)
        # Avoid summing overlapping cone returns into a large repulsive kick.
        candidate=guide+residual_velocity
        if cfg.mission_mode=='region':
            # No peer identity or perfect position: reserve personal space from ToF.
            fresh=ep.sensors.valid[...,:-2]&(ep.sensors.age[...,:-2]<(cfg.sensor_period_steps+cfg.sensor_latency_steps+1)*cfg.dt)
            pressure=jnp.where(fresh,jnp.clip((.32-ep.sensors.distance[...,:-2])/.32,0,1),0.)
            separation=-.15*jnp.sum(pressure[...,None]*directions[:-2],axis=-2)/jnp.maximum(1.,jnp.sum(pressure>0,axis=-1,keepdims=True))
            candidate=candidate+separation
        guarded=guard_velocity(candidate,ep.estimate,ep.sensors,cfg)
        repulsion=guarded-candidate
    yaw=jnp.clip(residual[...,3],-1,1)*gate*cfg.residual_yaw_rate/cfg.max_yaw_rate
    action=jnp.clip(jnp.concatenate([guarded/limits,yaw[...,None]],-1),-1,1)
    return action,guide,repulsion,residual_velocity,candidate


def baseline_action(ep,s,cfg):
    return action_components(ep,jnp.zeros_like(ep.previous_action),s,cfg)[0]


def policy_action(ep,residual,s,cfg):
    return action_components(ep,residual,s,cfg)[0]


def summarize(ep,s,cfg=None):
    cfg=cfg or SwarmConfig()
    import numpy as np
    ep=jax.device_get(ep); active=np.asarray(s.active)
    occupied=np.asarray(arrival_mask(ep.end_error,ep.end_speed,ep.data.states.pos,s.active,cfg))
    result={'success_rate':float(np.mean(ep.success)),
            'collision_rate':float(np.mean(np.any(ep.causes[:,:5],-1))),
            'failure_rate':float(np.mean(ep.failed)),
            'fitness_mean':float(jnp.mean(fitness(ep,s,cfg))),
            'slot_error_mean':float(ep.end_error[active].mean()),
            'speed_mean':float(ep.end_speed[active].mean()),
            'min_clearance':float(ep.min_clearance.min()),
            'slot_occupancy':float(np.mean(occupied[active])),
            'compact_fraction':float(np.mean(ep.compact_time/np.maximum(ep.exposure,1e-6))),
            'time_to_success':float(np.nanmean(ep.time_to_success)) if np.any(ep.success) else None}
    result.update(timeout_rate=float(np.mean(ep.timed_out)),truncated_rate=float(np.mean(ep.alive)),
                  hold_max_mean=float(np.mean(ep.max_hold)),hold_integral_mean=float(np.mean(ep.hold_integral)),
                  settling_mean=float(np.mean(ep.settling_integral)),progress_mean=float(ep.progress[active].mean()),
                  cohesion_mean=float(np.mean(ep.cohesion_integral)),risk_mean=float(np.mean(ep.risk_integral)),
                  smooth_mean=float(np.mean(ep.smooth_integral)),effort_mean=float(np.mean(ep.action_integral)),
                  slot_error_p95=float(np.quantile(ep.end_error[active],.95)),speed_p95=float(np.quantile(ep.end_speed[active],.95)),
                  simultaneous_occupancy=float(np.mean(np.all(occupied|~active,-1))),
                  route_completed_rate=float(np.mean(np.all(((ep.route_waypoint==s.routes.shape[2]-1)&(ep.end_error<cfg.arrival_tolerance))|~active,-1))),
                  command_saturation=float(np.mean(ep.saturation_integral/np.maximum(ep.exposure,1e-6))),
                  odometry_error_mean=float(np.linalg.norm(ep.estimate.pos-ep.data.states.pos,axis=-1)[active].mean()),
                  min_neighbor_clearance=float(ep.min_neighbor_clearance.min()) if np.isfinite(ep.min_neighbor_clearance).any() else None)
    if cfg.mission_mode=='region':
        result.update(goal_distance_mean=result['slot_error_mean'],region_occupancy=result['slot_occupancy'],
                      effort_proxy_mean=result['effort_mean'])
    result.update({f'hit_{cause}' :float(np.mean(ep.causes[:,i])) for i,cause in enumerate(CAUSES)})
    return result
