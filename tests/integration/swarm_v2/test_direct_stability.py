from dataclasses import replace
import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.environment import DirectEnvironment, observe, stability_density, direct_reward_v3
from drone_swarm.direct.training import training_difficulty, checkpoint_score, evaluate
from drone_swarm.direct.worlds import make_worlds
from drone_swarm.direct.artifacts import save, load, export
from drone_swarm.learning.network import init_mlp


def test_stability_reward_orders_hover_spin_and_early_failure():
    cfg=DirectConfig(direct_reward_version=3)
    rate=lambda omega:float(stability_density(1.,omega,0.,0.,.25,0.,cfg))
    assert rate(0.)>rate(1.)>rate(3.)>=rate(11.5)==0.
    assert float(stability_density(1.,0.,0.,.3,.25,0.,cfg))<rate(0.)
    assert float(stability_density(1.,0.,2.,0.,.25,0.,cfg))<rate(0.)
    assert float(stability_density(.8,0.,0.,0.,.25,0.,cfg))<rate(0.)
    score=lambda density,t,failed,success=False:float(
        density*t+direct_reward_v3(0.,True,success,failed,False,t,12.,.01))
    assert score(rate(0.),3.,True)>score(rate(11.5),10.,True)
    assert score(rate(0.),2.,True)>score(rate(0.),.2,True)
    assert score(rate(0.),3.,False,True)>score(rate(0.),8.,False,True)
    # Keeping the same physical trajectory at another decision rate preserves score.
    a=sum(float(direct_reward_v3(1.,True,False,False,False,i*.01,12.,.01)) for i in range(100))
    b=sum(float(direct_reward_v3(1.,True,False,False,False,i*.002,12.,.002)) for i in range(500))
    np.testing.assert_allclose(a,b,rtol=1e-6)


def test_compact_observation_preserves_missing_ranges_and_small_gyro():
    cfg=DirectConfig(device='cpu',direct_observation_version=2)
    imu=SimpleNamespace(attitude=jnp.array([[[0.,0.,0.,1.]]]),gyro=jnp.array([[[.1,-.2,.3]]]))
    sensors=SimpleNamespace(distance=jnp.full((1,1,26),cfg.lidar_max_range),
        valid=jnp.ones((1,1,26),bool).at[...,0].set(False),
        age=jnp.zeros((1,1,26)).at[...,1].set(1.))
    obs=observe(imu,jnp.zeros((1,1,3)),jnp.zeros((1,1,3)),sensors,jnp.zeros((1,1,4)),cfg)
    assert obs.shape==(1,1,42) and cfg.input_size==168
    np.testing.assert_allclose(obs[...,3:6],imu.gyro)
    np.testing.assert_array_equal(obs[...,16:18],-jnp.ones((1,1,2)))
    np.testing.assert_array_equal(obs[...,18:],jnp.zeros((1,1,24)))


def test_stability_checkpoint_and_export_do_not_reinterpret_old_models():
    cfg=DirectConfig(device='cpu',hidden_layers=(8,),direct_observation_version=2,direct_reward_version=3)
    weights=init_mlp(jax.random.PRNGKey(7),cfg.layers)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'new.pkl';save(p,weights,cfg)
        _,loaded=load(p);assert loaded==cfg
        export(p,Path(tmp)/'export')
        manifest=json.loads((Path(tmp)/'export/manifest.json').read_text())
        assert sum(n for _,n in manifest['fields'])==cfg.frame_size
        old=DirectConfig(device='cpu',hidden_layers=(8,));save(p,init_mlp(jax.random.PRNGKey(8),old.layers),old)
        payload=pickle.loads(p.read_bytes())
        for key in ['direct_observation_version','direct_curriculum_generations']:payload['config'].pop(key)
        p.write_bytes(pickle.dumps(payload));_,restored=load(p)
        assert restored.input_size==376 and restored.direct_curriculum_generations==0


def test_curriculum_changes_training_only_and_selection_rewards_stability():
    cfg=DirectConfig(device='cpu',es_population=2,batch_worlds=2,hidden_layers=(8,),
        episode_seconds=.04,direct_observation_version=2,direct_reward_version=3,direct_curriculum_generations=10)
    assert training_difficulty(cfg,0,0)==.25
    assert training_difficulty(cfg,9,0)==1. and training_difficulty(cfg,0,1)==1.
    assert training_difficulty(replace(cfg,direct_curriculum_generations=1),0,0)==1.
    env=DirectEnvironment(cfg,2);s=make_worlds(7,2,cfg,0)
    full=env.reset(s);easy=env.reset(s,difficulty=.25)
    np.testing.assert_allclose(easy.data.states.vel,full.data.states.vel*.25)
    np.testing.assert_allclose(easy.data.params.mass-cfg.nominal_mass,
                               (full.data.params.mass-cfg.nominal_mass)*.25,atol=1e-8)
    w=init_mlp(jax.random.PRNGKey(7),cfg.layers)
    metrics,rewards=evaluate(w,cfg,0,7,2)
    full_ep=env.rollout(w,s)
    np.testing.assert_allclose(rewards,full_ep.reward,rtol=1e-6)
    stable=dict(success_rate=0.,failure_rate=1.,hold_max_mean=0.,fitness_mean=-3.,flight_seconds_mean=2.)
    spinning={**stable,'fitness_mean':-8.,'flight_seconds_mean':10.}
    assert checkpoint_score(cfg,0,stable)>checkpoint_score(cfg,0,spinning)
