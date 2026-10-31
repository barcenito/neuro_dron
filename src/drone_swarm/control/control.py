"""Velocity PI outer loop; Crazyflow's attitude/rate/motor controller stays active."""
from typing import NamedTuple
import jax.numpy as jnp
from drone_swarm.simulation.geometry import rotate,inverse_rotate

class ControllerState(NamedTuple):
    integral: object
    yaw: object
    command: object


def init_controller(shape,yaw):
    return ControllerState(jnp.zeros((*shape,3)),yaw,jnp.zeros((*shape,4)))


def velocity_control(action,estimate,state,cfg):
    action=jnp.clip(jnp.nan_to_num(action),-1.,1.)
    limits=jnp.array([cfg.max_speed,cfg.max_speed,cfg.max_vertical_speed,cfg.max_yaw_rate])
    cmd=action*limits
    # Norm limit, not a per-axis sqrt(2) overspeed.
    cmd=cmd.at[...,:2].set(cmd[...,:2]/jnp.maximum(1.,jnp.linalg.norm(cmd[...,:2],axis=-1,keepdims=True)/cfg.max_speed))
    lost=(estimate.uncertainty>cfg.odometry_warn)|(estimate.altitude_age>cfg.altitude_timeout)
    cmd=jnp.where(lost[...,None],0.,cmd)
    # Mission supervisor brakes on unreliable odometry; no fictitious camera timeout.
    filt=state.command+(1-jnp.exp(-cfg.dt/cfg.actuator_tau))*(cmd-state.command)
    desired=rotate(estimate.quat,filt[...,:3])
    error=desired-estimate.vel
    integral=jnp.clip(state.integral+error*cfg.dt,-3.,3.)
    acc=cfg.velocity_kp*error+cfg.velocity_ki*integral
    # Vertical PI must compensate mass uncertainty, not settle in a descent.
    acc=acc.at[...,2].set(cfg.vertical_kp*error[...,2]+cfg.vertical_ki*integral[...,2])
    yaw=state.yaw+filt[...,3]*cfg.dt
    c,s=jnp.cos(yaw),jnp.sin(yaw)
    ax=c*acc[...,0]+s*acc[...,1]; ay=-s*acc[...,0]+c*acc[...,1]
    tilt=jnp.deg2rad(cfg.max_tilt_deg)
    pitch=jnp.clip(jnp.arctan2(ax,9.81),-tilt,tilt)
    roll=jnp.clip(jnp.arctan2(-ay,9.81),-tilt,tilt)
    thrust=cfg.nominal_mass*jnp.clip(9.81+acc[...,2],2.,18.)/(jnp.cos(roll)*jnp.cos(pitch))
    thrust=jnp.clip(thrust,0.,4*cfg.max_thrust_per_motor)
    return jnp.stack([thrust,roll,pitch,yaw],-1),ControllerState(integral,yaw,filt)
