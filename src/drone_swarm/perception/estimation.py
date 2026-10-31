"""Camera-free IMU + ToF localization against a preloaded wall/floor map.

The launch pose is calibrated in the preloaded mission frame. Horizontal position is corrected only when the measured wall geometry is
observable against the preloaded map. Otherwise uncertainty grows in empty space. IMUSample is
an explicit simulated measurement, not ground-truth position/velocity input to
the estimator. The AHRS attitude model assumes a calibrated IMU; not raw-image
or optical-flow processing and not a claim of hardware navigation accuracy.
"""
from typing import NamedTuple
import jax.numpy as jnp
from drone_swarm.simulation.geometry import rotate,inverse_rotate
from drone_swarm.perception.sensors import world_random,mapped_horizontal_ranges

class IMUSample(NamedTuple):
    specific_force: object
    attitude: object
    gyro: object

class Estimate(NamedTuple):
    pos: object
    vel: object
    quat: object
    gyro: object
    confidence: object
    uncertainty: object
    altitude_age: object
    elapsed: object
    horizontal_age: object
    anchor_uncertainty: object


def sample_imu(before,after,keys,step,cfg,noise_scale):
    """Measurement-level accelerometer/gyro/AHRS simulation, in body axes.

    Delta velocity represents the integrated accelerometer over a policy tick.
    AHRS yaw error accumulates from a constant gyro bias after launch calibration.
    """
    K=after.vel.shape[1]; shape=(K,3)
    gravity=jnp.array([0.,0.,9.81])
    force=inverse_rotate(after.quat,(after.vel-before.vel)/cfg.dt+gravity)
    force+=noise_scale*(world_random(keys,0,30,shape,True)*cfg.accel_bias+
                        world_random(keys,step,31,shape,True)*cfg.accel_noise)
    bias=world_random(keys,0,32,shape,True)*cfg.gyro_bias*noise_scale
    # Crazyflow angular velocity is ALREADY in body coordinates.
    gyro=after.ang_vel+bias+world_random(keys,step,33,shape,True)*cfg.gyro_noise*noise_scale
    tilt=world_random(keys,step,34,(K,2),True)*cfg.attitude_noise*noise_scale
    yaw=bias[...,2]*step*cfg.dt
    axis=jnp.concatenate([tilt,jnp.zeros_like(yaw[...,None])],-1)*.5
    w=jnp.sqrt(jnp.maximum(1-jnp.sum(axis*axis,-1),0))
    qv=after.quat[...,:3]; qw=after.quat[...,3]
    q=jnp.concatenate([w[...,None]*qv+qw[...,None]*axis+jnp.cross(axis,qv),
                        (w*qw-jnp.sum(axis*qv,-1))[...,None]],-1)
    z=jnp.sin(yaw/2); c=jnp.cos(yaw/2)
    axis=jnp.stack([jnp.zeros_like(z),jnp.zeros_like(z),z],-1)
    attitude=jnp.concatenate([c[...,None]*q[...,:3]+q[...,3:]*axis+jnp.cross(axis,q[...,:3]),
                              (c*q[...,3]-jnp.sum(axis*q[...,:3],-1))[...,None]],-1)
    return IMUSample(force,attitude/jnp.linalg.norm(attitude,axis=-1,keepdims=True),gyro)


def init_estimate(launch_pos,launch_quat,launch_velocity,keys,cfg,noise_scale):
    shape=launch_pos.shape[1:]; z=jnp.zeros(launch_pos.shape[:2])
    pos=launch_pos+world_random(keys,0,35,shape,True)*cfg.launch_position_noise*noise_scale
    vel=launch_velocity+world_random(keys,0,36,shape,True)*cfg.launch_velocity_noise*noise_scale
    return Estimate(pos,vel,launch_quat,jnp.zeros_like(pos),jnp.ones_like(z),
                    z+cfg.launch_position_noise*noise_scale,z,z,z,z+cfg.launch_position_noise*noise_scale)


def mapped_floor_height(pos,s):
    """Known support surfaces from the preloaded map, evaluated at ESTIMATED XY."""
    inside=jnp.all((pos[:,:,None,:2]>=s.boxes_lo[:,None,:,:2])&
                   (pos[:,:,None,:2]<=s.boxes_hi[:,None,:,:2]),-1)
    below=s.boxes_hi[:,None,:,2]<pos[...,2,None]-.2
    return jnp.maximum(0.,jnp.max(jnp.where(inside&below,s.boxes_hi[:,None,:,2],0.),-1))


def update_estimate(old,imu,sensors,s,cfg,noise_scale,cones):
    acc=rotate(imu.attitude,imu.specific_force)-jnp.array([0.,0.,9.81])
    vel=old.vel+acc*cfg.dt
    pos=old.pos+(old.vel+vel)*(.5*cfg.dt)
    # ToF measures from its mount, with a cone and delivery age. Only mapped
    # horizontal support is used; no peer identities or simulator pose fixes.
    directions=rotate(imu.attitude[:,:,None,:],cones[-1][None,None])
    cosine=jnp.max(-directions[...,2],-1)
    mount=rotate(imu.attitude,jnp.broadcast_to(jnp.array([0.,0.,-cfg.drone_radius*.8]),pos.shape))
    height=mapped_floor_height(pos,s)+sensors.distance[...,-1]*cosine-mount[...,2]
    height+=vel[...,2]*sensors.age[...,-1]
    innovation=height-pos[...,2]
    fresh=sensors.age[...,-1] <= (cfg.sensor_latency_steps+.5)*cfg.dt
    valid=sensors.valid[...,-1]&fresh&(sensors.distance[...,-1]<cfg.lidar_max_range-.02)&(cosine>.7)
    # Reject returns from unknown objects/edges rather than snapping onto them.
    valid &= jnp.abs(innovation)<.2
    pos=pos.at[...,2].add(jnp.where(valid,.08*innovation,0.))
    vel=vel.at[...,2].add(jnp.where(valid,.003*innovation/cfg.dt,0.))
    age=jnp.where(valid,0.,old.altitude_age+cfg.dt)
    t=old.elapsed+cfg.dt
    horizontal_age=old.horizontal_age+cfg.dt
    anchor=old.anchor_uncertainty
    if cfg.range_localization:
        pos,vel,observed,range_sigma=correct_horizontal_ranges(pos,vel,imu.attitude,sensors,s,cfg,cones)
        horizontal_age=jnp.where(observed,0.,horizontal_age)
        anchor=jnp.where(observed,jnp.maximum(.002,range_sigma*noise_scale),anchor)
    # No observed geometry => no reset, and the same inertial uncertainty grows.
    age_xy=horizontal_age
    uncertainty=jnp.sqrt(anchor**2+noise_scale**2*((cfg.launch_velocity_noise*age_xy)**2+
                    (.5*cfg.accel_bias*age_xy*age_xy)**2+(.5*9.81*cfg.gyro_bias*age_xy*age_xy)**2+
                    cfg.accel_noise**2*cfg.dt*age_xy**3/3.))
    confidence=jnp.exp(-uncertainty/cfg.odometry_warn)*jnp.exp(-age/cfg.altitude_timeout)
    return Estimate(pos,vel,imu.attitude,imu.gyro,confidence,uncertainty,age,t,horizontal_age,anchor)


def correct_horizontal_ranges(pos,vel,quat,sensors,s,cfg,cones):
    """Robust 2-D scan correction; reject unobservable or inconsistent scans."""
    predict=lambda p:mapped_horizontal_ranges(p,vel,quat,sensors.age,s,cfg,cones)
    prediction=predict(pos)
    eps=.01
    dx=jnp.array([eps,0.,0.]); dy=jnp.array([0.,eps,0.])
    jac=jnp.stack([(predict(pos+dx)-predict(pos-dx))/(2*eps),
                   (predict(pos+dy)-predict(pos-dy))/(2*eps)],-1)
    residual=sensors.distance[...,:-2]-prediction
    valid=(sensors.valid[...,:-2]&(sensors.age[...,:-2]<=(cfg.sensor_latency_steps+cfg.sensor_period_steps)*cfg.dt)&
           (prediction<cfg.lidar_max_range-.02)&(sensors.distance[...,:-2]<cfg.lidar_max_range-.02)&
           (jnp.abs(residual)<cfg.range_innovation_gate)&s.active[...,None])
    # A Huber-like bounded weight reduces influence of unmodelled moving peers.
    weight=valid*jnp.minimum(1.,.05/jnp.maximum(jnp.abs(residual),1e-6))
    normal=jnp.einsum('...li,...l,...lj->...ij',jac,weight,jac)
    rhs=jnp.einsum('...li,...l,...l->...i',jac,weight,residual)
    a,b,c=normal[...,0,0],normal[...,0,1],normal[...,1,1]
    eigen_min=.5*(a+c-jnp.sqrt((a-c)**2+4*b*b))
    observed=(eigen_min>.2)&(valid.sum(-1)>=3)
    det=jnp.maximum((a+.01)*(c+.01)-b*b,1e-6)
    correction=jnp.stack([((c+.01)*rhs[...,0]-b*rhs[...,1])/det,
                          ((a+.01)*rhs[...,1]-b*rhs[...,0])/det],-1)
    correction=correction/jnp.maximum(1.,jnp.linalg.norm(correction,axis=-1,keepdims=True)/.05)
    correction=jnp.where(observed[...,None],correction,0.)
    pos=pos.at[...,:2].add(.2*correction)
    vel=vel.at[...,:2].add(.001*correction/cfg.dt)
    sigma=cfg.sensor_bias+2*cfg.sensor_noise*(1+jnp.sum(jnp.where(valid,prediction,0.),-1)/jnp.maximum(valid.sum(-1),1))
    return pos,vel,observed,sigma
