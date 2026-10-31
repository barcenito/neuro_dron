from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as j
from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.worlds import make_worlds
from drone_swarm.direct.environment import DirectEnvironment,motor_thrust,observe
from drone_swarm.direct.artifacts import save,load,export
from drone_swarm.learning.network import init_mlp,generate_and_apply_mutations
from drone_swarm.perception.sensors import measure_ranges,cone_directions

@lru_cache(None)
def setup():
    cfg=DirectConfig(device='cpu',es_population=4,hidden_layers=(8,),initial_tilt_degrees=0.,
                     mass_variation=0.,inertia_variation=0.,wind_force=0.,hold_seconds=.04)
    return cfg,DirectEnvironment(cfg,4)

def test_direct_worlds_are_batch_independent_and_twins_share_noise():
    cfg,_=setup()
    full=make_worlds(9,4,cfg,3);first=make_worlds(9,2,cfg,3);last=make_worlds(9,2,cfg,3,start_index=2)
    twin=make_worlds(9,8,cfg,3,paired=True)
    for a,b,c,d in zip(full,first,last,twin):
        np.testing.assert_array_equal(a,np.concatenate([b,c]))
        np.testing.assert_array_equal(d[:4],d[4:])
    group=make_worlds(9,2,replace(cfg,swarm_size=4),0)
    assert bool(group.active.all())

def test_direct_motor_authority_and_no_hidden_stabilizer():
    from unittest.mock import patch
    cfg,env=setup();s=make_worlds(1,4,cfg)
    np.testing.assert_allclose(motor_thrust(j.array([-1.,0.,1.,2.]),cfg),
                               [0.,cfg.max_thrust_per_motor/2,cfg.max_thrust_per_motor,cfg.max_thrust_per_motor])
    ep=env.reset(s,0.)
    # Trace while legacy stabilizer and map estimator are disabled outright.
    env._rollouts.clear()
    with (patch('drone_swarm.control.control.velocity_control',side_effect=AssertionError('hidden PI')),
          patch('drone_swarm.perception.estimation.update_estimate',side_effect=AssertionError('hidden map'))):
        symmetric=env._step(ep,j.zeros((4,1,4)),s,0.)
        differential=env._step(ep,j.zeros((4,1,4)).at[...,0].set(.8),s,0.)
    assert float(j.linalg.norm(differential.data.states.ang_vel-symmetric.data.states.ang_vel))>.01
    assert float(differential.motors[0,0,0])>float(differential.motors[0,0,1])

def test_direct_arrival_uses_goal_only_and_requires_sustained_hold():
    cfg,env=setup();s=make_worlds(3,4,cfg)
    s=s._replace(routes=s.spawn[:,:,None],initial_velocity=j.zeros_like(s.initial_velocity))
    ep=env.reset(s,0.)
    hover=j.full((4,1,4),2*(cfg.nominal_mass*9.81/4)/cfg.max_thrust_per_motor-1)
    first=env.step(ep,hover,s,0.)
    assert not bool(first.success.any())
    for _ in range(6):ep=env.step(ep,hover,s,0.)
    assert bool(ep.success.all())
    assert not bool(ep.failed.any())
    distant=s._replace(routes=s.routes+j.array([2.,0.,0.]))
    ep=env.reset(distant,0.)
    for _ in range(6):ep=env.step(ep,hover,distant,0.)
    assert not bool(ep.success.any())

def test_direct_sensor_mount_detects_previous_diagonal_blind_case():
    cfg=DirectConfig(es_population=2,swarm_size=2)
    s=make_worlds(1,1,cfg)
    s=s._replace(boxes_lo=j.full_like(s.boxes_lo,100.),boxes_hi=j.full_like(s.boxes_hi,101.))
    pos=j.array([[[0.,0.,1.],[.065,0.,1.065]]]);q=j.array([[[0.,0.,0.,1.],[0.,0.,0.,1.]]])
    new=measure_ranges(pos,q,s,j.full_like(s.dynamic_center,100.),cfg,cone_directions(cfg))
    oldcfg=replace(cfg,num_lidars=16)
    old=measure_ranges(pos,q,s,j.full_like(s.dynamic_center,100.),oldcfg,cone_directions(oldcfg))
    np.testing.assert_allclose(old[...,:-2],cfg.lidar_max_range)
    assert bool((new[...,:-2].min(-1)<.15).all())

def test_direct_batch_rollouts_keep_global_mutation_and_world_order():
    cfg,env=setup();s=make_worlds(15,4,cfg,paired=True)
    w=init_mlp(jax.random.PRNGKey(3),cfg.layers)
    mu=generate_and_apply_mutations(jax.random.PRNGKey(4),cfg.layers,1,.005,4)
    full=env.rollout(w,s,mu,noise=0.,max_steps=4)
    small=DirectEnvironment(cfg,2);rewards=[]
    for start in (0,2):
        scene=jax.tree_util.tree_map(lambda x:x[start:start+2],s)
        mutation=jax.tree_util.tree_map(lambda x:x[start:start+2],mu)
        result=small.rollout(w,scene,mutation,noise=0.,max_steps=4)
        rewards.append(result.reward)
    np.testing.assert_allclose(full.reward,j.concatenate(rewards),atol=1e-5)

def test_direct_checkpoint_contract_and_export_cannot_be_confused_with_residual():
    from drone_swarm.artifacts.checkpoints import load_checkpoint,save_checkpoint
    from drone_swarm.config import SwarmConfig
    cfg,_=setup();w=init_mlp(jax.random.PRNGKey(0),cfg.layers)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'direct.pkl';save(path,w,cfg)
        _,read=load(path);assert read.layers==cfg.layers
        export(path,Path(directory)/'export')
        try:load_checkpoint(path)
        except ValueError:pass
        else:raise AssertionError('Residual loader accepted direct motor actions')
        legacy=SwarmConfig(es_population=4,hidden_layers=(8,))
        save_checkpoint(path,init_mlp(jax.random.PRNGKey(0),legacy.layers),legacy)
        try:load(path)
        except ValueError:pass
        else:raise AssertionError('Direct loader accepted residual actions')
