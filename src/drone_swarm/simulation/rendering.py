"""Display the actual collision geometry and mission used by the pure step."""
import numpy as np
import mujoco

# Fix para incompatibilidad entre versiones recientes de mujoco y gymnasium (< 0.30)
# gymnasium pasa 'scene' a mjv_moveCamera, pero mujoco>=3.0 eliminó ese argumento.
_orig_move = mujoco.mjv_moveCamera
def _patched_move(*args, **kwargs):
    if len(args) == 6:
        args = args[:4] + args[5:]
    return _orig_move(*args, **kwargs)
mujoco.mjv_moveCamera = _patched_move
from drone_swarm.simulation.scenarios import obstacle_states
from drone_swarm.simulation.geometry import rotate
from drone_swarm.perception.sensors import get_lidar_directions


def render_episode(env,ep,s):
    env.sim.data=ep.data
    if env.sim.viewer is None: env.sim.render()
    viewer=env.sim.viewer
    if viewer is not None and hasattr(viewer,'viewer'):
        v=viewer.viewer
        if hasattr(v,'_markers'): v._markers.clear()
        def sphere(p,r,color,label=''):
            v.add_marker(pos=np.asarray(p),size=np.full(3,r),rgba=np.array(color),type=2,label=label)
        for lo,hi in zip(np.asarray(s.boxes_lo[0]),np.asarray(s.boxes_hi[0])):
            if np.linalg.norm(lo)<30:
                v.add_marker(pos=(lo+hi)/2,size=(hi-lo)/2,rgba=np.array([.4,.4,.5,.4]),type=6,label='')
        if env.cfg.mission_mode=='region':
            sphere(s.routes[0,0,-1],env.cfg.region_radius,[1.,.6,0,.10],'arrival region')
            sphere(s.routes[0,0,-1],.04,[1.,.6,0,1.],'common goal')
        else:
            for i,target in enumerate(np.asarray(s.routes[0,:,-1])):
                if bool(s.active[0,i]): sphere(target,.05,[1.,.5,0,1.],f'slot {i}')
        for obstacle in np.asarray(s.static[0]):
            if np.linalg.norm(obstacle)<30: sphere(obstacle,env.cfg.obstacle_radius,[.8,.2,.2,1.])
        dynamic,_=obstacle_states(s,np.array([int(ep.step)*env.cfg.dt]))
        for obstacle in np.asarray(dynamic[0]):
            if np.linalg.norm(obstacle)<30: sphere(obstacle,env.cfg.obstacle_radius,[.8,.2,.8,1.])
        # Measured ray endpoints and separation radius, without privileged returns.
        for i,pos in enumerate(np.asarray(ep.data.states.pos[0])):
            if not bool(s.active[0,i]): continue
            sphere(pos,env.cfg.separation/2,[.2,1.,.2,.08])
            directions=np.asarray(rotate(ep.estimate.quat[0,i],get_lidar_directions(env.cfg.num_lidars)))
            for d,r,valid in zip(directions,np.asarray(ep.sensors.distance[0,i]),np.asarray(ep.sensors.valid[0,i])):
                if valid and r<env.cfg.lidar_max_range: sphere(pos+d*r,.015,[0.,1.,1.,.8])
    env.sim.render()
