from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase
import numpy as np
import jax
import jax.numpy as j
from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.worlds import make_worlds,hold_ready
from drone_swarm.direct.environment import DirectEnvironment
from drone_swarm.simulation.scenarios import obstacle_states
from drone_swarm.simulation.environment import contacts
from drone_swarm.perception.sensors import measure_ranges,cone_directions
from drone_swarm.direct.artifacts import save,load
from drone_swarm.learning.network import init_mlp


def config():
    return DirectConfig(device='cpu',es_population=4,hidden_layers=(8,),episode_seconds=20.,direct_curriculum_version=3)


def test_encounters_really_cross_route_and_goal_at_planned_times():
    cfg=config();s=make_worlds(910013,64,cfg,3)
    center=np.asarray(s.dynamic_center);goal=np.asarray(s.routes[:,0,-1]);spawn=np.asarray(s.spawn[:,0])
    crossing=-np.asarray(s.dynamic_phase[:,:,0])/np.asarray(s.dynamic_frequency[:,:,0])
    for k,(low,high) in enumerate([(1.5,2.5),(6,8),(9,11)]):
        assert np.all((crossing[:,k]>=low)&(crossing[:,k]<=high))
        pos,vel=obstacle_states(s,j.asarray(crossing[:,k]))
        np.testing.assert_allclose(np.asarray(pos)[:,k],center[:,k],atol=1e-6)
        assert np.all(np.linalg.norm(np.asarray(vel)[:,k],axis=-1)>0)
    np.testing.assert_allclose(center[:,1],goal)
    fraction=(center[:,0,0]-spawn[:,0])/(goal[:,0]-spawn[:,0])
    np.testing.assert_allclose(center[:,0],spawn+fraction[:,None]*(goal-spawn),atol=2e-7)
    offset=np.linalg.norm(center[:,2]-goal,axis=-1)
    assert np.all((offset>=.25)&(offset<=.5))
    ext=np.abs(np.asarray(s.dynamic_amplitude));low=center-ext;high=center+ext
    assert np.all(low[:,:,1]>-2.9+cfg.obstacle_radius) and np.all(high[:,:,1]<3-cfg.obstacle_radius)
    assert np.all(low[:,:,2]>cfg.obstacle_radius) and np.all(high[:,:,2]<3-cfg.obstacle_radius)
    assert np.all(low[:,:,0]>.05+cfg.obstacle_radius) # no path through the late maze walls
    assert not bool(hold_ready(s,11.99,cfg).any()) and bool(hold_ready(s,12.,cfg).all())


def test_encounters_preserve_earlier_scenes_and_batch_pairing():
    cfg=config();old=replace(cfg,direct_curriculum_version=2)
    for stage in range(5):
        a=make_worlds(51,4,cfg,stage);b=make_worlds(51,4,old,stage)
        for name in a._fields:
            if stage>=3 and name in ('dynamic_center','dynamic_amplitude','dynamic_frequency','dynamic_phase'):continue
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    full=make_worlds(51,4,cfg,3);left=make_worlds(51,2,cfg,3);right=make_worlds(51,2,cfg,3,start_index=2)
    twin=make_worlds(51,8,cfg,3,paired=True)
    for a,b,c,d in zip(full,left,right,twin):
        np.testing.assert_array_equal(a,np.concatenate([b,c]))
        np.testing.assert_array_equal(a,d[:4]);np.testing.assert_array_equal(d[:4],d[4:])
    disabled=replace(cfg,dynamic_scale=0.)
    assert bool(hold_ready(make_worlds(51,4,disabled,3),0.,disabled).all())
    with TestCase().assertRaisesRegex(ValueError,'horizon'):
        make_worlds(51,1,replace(cfg,episode_seconds=15.),3)


def test_goal_crossing_is_sensed_and_collision_checked():
    cfg=replace(config(),initial_tilt_degrees=0.)
    s=make_worlds(11,1,cfg,3);env=DirectEnvironment(cfg,1);ep=env.reset(s,0.)
    time=float(-s.dynamic_phase[0,1,0]/s.dynamic_frequency[0,1,0])
    goal=s.routes[:,:,-1]
    state=ep.data.states.replace(pos=goal,quat=j.array([[[0.,0.,0.,1.]]]))
    dynamic,_=obstacle_states(s,j.array([time-.4]))
    ranges=measure_ranges(goal,state.quat,s,dynamic,cfg,cone_directions(cfg))
    assert float(ranges.min())<.3
    causes,_,_=contacts(state,state,s,time,cfg)
    assert bool(causes[0,0,3])


def test_late_hold_cannot_use_time_accumulated_before_encounters():
    cfg=replace(config(),hold_seconds=.04,initial_tilt_degrees=0.,mass_variation=0.,inertia_variation=0.,wind_force=0.)
    env=DirectEnvironment(cfg,1);s=make_worlds(3,1,cfg,3)
    s=s._replace(routes=s.spawn[:,:,None],initial_velocity=j.zeros_like(s.initial_velocity),
        static=j.full_like(s.static,100.),dynamic_center=j.full_like(s.dynamic_center,100.),dynamic_amplitude=j.zeros_like(s.dynamic_amplitude))
    ep=env.reset(s,0.);hover=j.full((1,1,4),2*(cfg.nominal_mass*9.81/4)/cfg.max_thrust_per_motor-1)
    ep=ep._replace(step=j.int32(1198),hold=j.array([10.]))
    ep=env.step(ep,hover,s,0.)
    assert not bool(ep.success.any()) and float(ep.hold[0])==0.
    for _ in range(6):ep=env.step(ep,hover,s,0.)
    assert bool(ep.success.all()) and not bool(ep.failed.any())


def test_encounter_version_roundtrips_without_changing_old_checkpoints():
    cfg=config();weights=init_mlp(jax.random.PRNGKey(1),cfg.layers)
    with TemporaryDirectory() as d:
        p=Path(d)/'p.pkl';save(p,weights,cfg)
        assert load(p)[1].direct_curriculum_version==3
        old=replace(cfg,direct_curriculum_version=2);save(p,weights,old)
        assert load(p)[1].direct_curriculum_version==2
