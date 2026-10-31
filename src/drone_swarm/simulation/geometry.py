"""Geometry shared by sensor simulation, scenario validation and contacts."""
import jax.numpy as jnp

def rotate(q, v):
    xyz, w = q[..., :3], q[..., 3:]
    t = 2 * jnp.cross(xyz, v)
    return v + w * t + jnp.cross(xyz, t)

def inverse_rotate(q, v):
    return rotate(q * jnp.array([-1., -1., -1., 1.]), v)

def box_distance(p, lo, hi):
    return jnp.linalg.norm(jnp.maximum(jnp.maximum(lo-p, p-hi), 0.), axis=-1)

def ray_boxes(origin, direction, lo, hi):
    """Broadcasting slab test, including parallel/on-boundary and inside origins."""
    parallel = jnp.abs(direction) < 1e-7
    inv = 1. / jnp.where(parallel, 1., direction)
    t0, t1 = (lo-origin)*inv, (hi-origin)*inv
    near = jnp.where(parallel, -jnp.inf, jnp.minimum(t0,t1))
    far = jnp.where(parallel, jnp.inf, jnp.maximum(t0,t1))
    valid = ~jnp.any(parallel & ((origin < lo) | (origin > hi)), axis=-1)
    a, b = jnp.max(near,axis=-1), jnp.min(far,axis=-1)
    return jnp.where(valid & (a <= b) & (b >= 0), jnp.maximum(a,0), jnp.inf)

def ray_spheres(origin, direction, center, radius):
    oc = origin-center
    b = jnp.sum(oc*direction,axis=-1)
    c = jnp.sum(oc*oc,axis=-1)-radius**2
    d = b*b-c
    t = -b-jnp.sqrt(jnp.maximum(d,0))
    return jnp.where(c <= 0, 0., jnp.where((d >= 0) & (t >= 0),t,jnp.inf))

def segment_spheres(a,b,center,radius):
    ab=b-a
    t=jnp.clip(jnp.sum((center-a)*ab,axis=-1)/(jnp.sum(ab*ab,axis=-1)+1e-12),0,1)
    return jnp.linalg.norm(a+t[...,None]*ab-center,axis=-1) <= radius
