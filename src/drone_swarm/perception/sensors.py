"""Finite-range ToF measurements only: no object velocity or identity channels."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from drone_swarm.simulation.geometry import rotate, inverse_rotate, ray_boxes, ray_spheres

rotate_body_to_world=rotate
rotate_world_to_body=inverse_rotate

class SensorState(NamedTuple):
    distance: object
    valid: object
    age: object
    pending_distance: object
    pending_valid: object
    pending_age: object
    bias: object
    range_rate: object


def world_random(keys,step,stream,shape,normal=False):
    """Same world key + time + stream => CRN independent of policy decisions."""
    fn=jax.random.normal if normal else jax.random.uniform
    return jax.vmap(lambda key: fn(jax.random.fold_in(jax.random.fold_in(key,step),stream),shape))(keys)


def get_lidar_directions(num_lidars=16):
    if num_lidars==26:
        # Experimental three-ring mount: equator, +45 and -45 degrees, poles.
        az=jnp.arange(8)*2*jnp.pi/8
        rings=[jnp.stack([jnp.cos(az)*jnp.cos(el),jnp.sin(az)*jnp.cos(el),
                         jnp.full_like(az,jnp.sin(el))],-1) for el in (0.,jnp.pi/4,-jnp.pi/4)]
        return jnp.concatenate([*rings,jnp.array([[0.,0.,1.],[0.,0.,-1.]])])
    if num_lidars not in (10,16): raise ValueError('Supported physical layouts: 10 and 16')
    a=jnp.arange(num_lidars-2)*2*jnp.pi/(num_lidars-2)
    ring=jnp.stack([jnp.cos(a),jnp.sin(a),jnp.zeros_like(a)],-1)
    return jnp.concatenate([ring,jnp.array([[0.,0.,1.],[0.,0.,-1.]])])


def cone_directions(cfg):
    # Nine deterministic sub-rays approximate the return from a finite FoV.
    axes=np.asarray(get_lidar_directions(cfg.num_lidars))
    cones=[]
    for d in axes:
        ref=np.array([0,0,1]) if abs(d[2])<.9 else np.array([1,0,0])
        u=np.cross(d,ref); u/=np.linalg.norm(u); v=np.cross(d,u)
        half=np.deg2rad(cfg.lidar_fov_deg/2)
        cones.append(np.stack([d]+[np.cos(half)*d+np.sin(half)*(np.cos(a)*u+np.sin(a)*v)
                                  for a in np.arange(8)*np.pi/4]))
    return jnp.array(cones)


def range_cast(pos,quat,scenario,other_pos,cfg,cones):
    # W,K,L,S,3 and W,K,L,S,O/B,3. Mask self and inactive agents.
    dirs=rotate(quat[:,:,None,None,:],cones[None,None])
    mount=rotate(quat[:,:,None,:],get_lidar_directions(cfg.num_lidars)[None,None]*cfg.drone_radius*.8)
    origin=pos[:,:,None,None,:]+mount[:,:,:,None,:]
    b=ray_boxes(origin[...,None,:],dirs[...,None,:],
                scenario.boxes_lo[:,None,None,None,:,:],scenario.boxes_hi[:,None,None,None,:,:]).min(-1)
    W,K=pos.shape[:2]
    diff=other_pos[:,None,:,:]-jnp.zeros((W,K,1,3))
    peer_mask=scenario.active[:,None,:] & ~jnp.eye(K,dtype=bool)[None]
    diff=jnp.where(peer_mask[...,None],diff,1000.)
    from drone_swarm.simulation.scenarios import obstacle_states
    # other_pos contains current live agents; external geometry is supplied separately below.
    return origin,dirs,b,diff


def measure_ranges(pos,quat,scenario,dynamic,cfg,cones):
    origin,dirs,b,peers=range_cast(pos,quat,scenario,pos,cfg,cones)
    external=jnp.concatenate([scenario.static,dynamic],axis=1)
    external=jnp.broadcast_to(external[:,None],(pos.shape[0],pos.shape[1],external.shape[1],3))
    centers=jnp.concatenate([peers,external],axis=2)
    radii=jnp.concatenate([jnp.full((pos.shape[1],),cfg.drone_radius),jnp.full((external.shape[2],),cfg.obstacle_radius)])
    sphere=ray_spheres(origin[...,None,:],dirs[...,None,:],centers[:,:,None,None,:,:],radii).min(-1)
    dz=dirs[...,2]
    floor=jnp.where(dz < -1e-6,-origin[...,2]/jnp.where(dz < -1e-6,dz,-1.),jnp.inf)
    distance=jnp.minimum(jnp.minimum(b,sphere),floor).min(-1)
    return jnp.clip(distance,cfg.lidar_min_range,cfg.lidar_max_range)


def init_sensors(keys,K,cfg):
    shape=(keys.shape[0],K,cfg.num_lidars)
    distance=jnp.full(shape,cfg.lidar_max_range)
    valid=jnp.zeros(shape,dtype=bool); age=jnp.ones(shape)*10
    qshape=(cfg.sensor_latency_steps,*shape)
    bias=world_random(keys,0,20,(K,cfg.num_lidars),True)*cfg.sensor_bias
    return SensorState(distance,valid,age,jnp.broadcast_to(distance,qshape),
                       jnp.broadcast_to(valid,qshape),jnp.broadcast_to(age,qshape),bias,jnp.zeros(shape))


def update_sensors(state,ideal,keys,step,cfg,noise_scale):
    shape=ideal.shape[1:]
    noise=world_random(keys,step,21,shape,True)*cfg.sensor_noise*(1+ideal)
    valid=world_random(keys,step,22,shape) >= cfg.sensor_dropout*noise_scale
    measured=jnp.round(jnp.clip(ideal+noise_scale*(noise+state.bias),cfg.lidar_min_range,cfg.lidar_max_range)*1000)/1000
    measured=jnp.where(valid,measured,cfg.lidar_max_range)
    scheduled=(step+jnp.arange(cfg.num_lidars))%cfg.sensor_period_steps==0
    # Sample-and-hold, then explicit delivery latency. Age never resets for stale samples.
    last_d=state.pending_distance[-1]; last_v=state.pending_valid[-1]
    sampled=jnp.where(scheduled,measured,last_d)
    sampled_v=jnp.where(scheduled,valid,last_v)
    sampled_age=jnp.where(scheduled,0.,state.pending_age[-1]+cfg.dt)
    delivered=state.pending_distance[0]
    fresh=state.pending_age[0]<(cfg.sensor_latency_steps-.5)*cfg.dt
    usable=fresh&state.pending_valid[0]&state.valid&(state.age<(cfg.sensor_period_steps+cfg.sensor_latency_steps+1)*cfg.dt)
    # Angular/surface changes also affect ranges: this is NOT object velocity.
    rate=jnp.clip((delivered-state.distance)/(cfg.sensor_period_steps*cfg.dt),-1.,1.)
    filtered=jnp.where(usable,state.range_rate+.25*(rate-state.range_rate),state.range_rate)
    filtered=jnp.where(state.pending_valid[0],filtered,0.)
    return SensorState(state.pending_distance[0],state.pending_valid[0],state.pending_age[0]+cfg.dt,
                       jnp.concatenate([state.pending_distance[1:],sampled[None]]),
                       jnp.concatenate([state.pending_valid[1:],sampled_v[None]]),
                       jnp.concatenate([state.pending_age[1:]+cfg.dt,sampled_age[None]]),state.bias,filtered)


def mapped_horizontal_ranges(pos,vel,quat,age,scenario,cfg,cones):
    """Predicted ToF against the PRELOADED wall map, never live agents/obstacles.

    Latency compensation is per sensor. This is a measurement prediction, not
    a ground-truth position correction; all pose inputs come from the estimator.
    """
    axes=get_lidar_directions(cfg.num_lidars)[:-2]
    directions=rotate(quat[:,:,None,None,:],cones[None,None,:-2])
    mount=rotate(quat[:,:,None,:],axes[None,None]*cfg.drone_radius*.8)
    origin=(pos[:,:,None]-vel[:,:,None]*age[...,:-2,None]+mount)[:,:,:,None,:]
    ranges=ray_boxes(origin[...,None,:],directions[...,None,:],scenario.boxes_lo[:,None,None,None],
                     scenario.boxes_hi[:,None,None,None]).min((-1,-2))
    return jnp.clip(ranges,cfg.lidar_min_range,cfg.lidar_max_range)
