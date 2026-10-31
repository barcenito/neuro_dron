"""Local velocity constraints from measured ToF only, after learned residuals."""
import jax
import jax.numpy as jnp
from drone_swarm.perception.sensors import get_lidar_directions
from drone_swarm.simulation.geometry import inverse_rotate


def guard_velocity(velocity,estimate,sensors,cfg):
    axes=get_lidar_directions(cfg.num_lidars)[:-2,:2]
    if cfg.avoidance_cones:
        # ToF identifies a cone, not a bearing: constrain its center and edges.
        angle=jnp.arctan2(axes[:,1],axes[:,0])[:,None]+jnp.deg2rad(cfg.lidar_fov_deg/2)*jnp.array([-1.,0.,1.])
        axes=jnp.stack([jnp.cos(angle),jnp.sin(angle)],-1).reshape((-1,2))
    measured_velocity=inverse_rotate(estimate.quat,estimate.vel)[...,:2]
    distance=sensors.distance[...,:-2]
    valid=sensors.valid[...,:-2]&(sensors.age[...,:-2]<(cfg.sensor_period_steps+cfg.sensor_latency_steps+1)*cfg.dt)
    # Relative range derivative includes ego motion; do not count it twice.
    approach=jnp.clip(-sensors.range_rate[...,:-2],0.,.6)
    if cfg.avoidance_cones:
        distance=jnp.repeat(distance,3,axis=-1)
        valid=jnp.repeat(valid,3,axis=-1)
        approach=jnp.repeat(approach,3,axis=-1)
    ego=jnp.einsum('...i,li->...l',measured_velocity,axes)
    bound=ego-approach+cfg.avoidance_gain*(distance-cfg.avoidance_margin)
    bound=jnp.where(valid&(distance<.8),bound,10.)
    # Cyclic projections onto sensed half-planes. Existing separation and
    # actuator limits may make constraints infeasible; no safety guarantee.
    def project(i,v):
        k=i%axes.shape[0];n=axes[k]
        excess=jnp.maximum(jnp.sum(v*n,-1)-bound[...,k],0.)
        return v-excess[...,None]*n
    xy=jax.lax.fori_loop(0,axes.shape[0]*2,project,velocity[...,:2])
    xy=xy/jnp.maximum(1.,jnp.linalg.norm(xy,axis=-1,keepdims=True)/cfg.max_speed)
    return velocity.at[...,:2].set(xy)
