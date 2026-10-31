from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.environment import clearance_density,direct_reward_v5,DirectEnvironment
from drone_swarm.direct.training import learning_scales,validate_policy,train
from drone_swarm.direct.worlds import make_worlds
from drone_swarm.direct.artifacts import save,load
from drone_swarm.learning.network import init_mlp


def test_safety_reward_prefers_margin_and_does_not_pay_early_death():
    cfg=DirectConfig()
    d=clearance_density(j.full((5,),2.),j.array([1.,.35,.2,0.,-.1]))
    assert float(d[0])==float(d[1])==2.
    assert float(d[1])>float(d[2])>float(d[3])==float(d[4])>=0
    # Early failure cannot escape negative per-step costs: live density stays >=0.
    fail=lambda t:float(direct_reward_v5(0.,True,False,True,t,20.,.01))+t*.5
    assert fail(2)<fail(10)
    assert float(direct_reward_v5(2.,False,False,False,10.,20.,.01))==0
    success=lambda t:2*(t-.01)+float(direct_reward_v5(2.,True,True,False,t,20.,.01))
    np.testing.assert_allclose(success(3),success(15),atol=1e-5)
    assert success(15)>fail(19)


def test_schedule_duration_is_independent_of_training_budget():
    c=DirectConfig(num_generations=1000,direct_schedule_generations=100)
    for gen in [0,50,99,500]:
        assert learning_scales(c,gen)==learning_scales(replace(c,num_generations=2000),gen)
    np.testing.assert_allclose(learning_scales(c,99),(c.alpha_end,c.sigma_end))
    assert learning_scales(c,99)==learning_scales(c,500)
    old=replace(c,direct_schedule_generations=0)
    np.testing.assert_allclose(learning_scales(old,499),
      (old.alpha_start+(old.alpha_end-old.alpha_start)*499/999,old.sigma_start+(old.sigma_end-old.sigma_start)*499/999))


def test_multiple_validation_groups_are_separate_and_weighted():
    c=DirectConfig(direct_validation_groups=3,validation_episodes=4,validation_seed=123)
    calls=[]
    def fake(w,cfg,stage,seed,episodes,**kw):
        calls.append(seed);return {'episodes':episodes,'success_rate':(seed-123)/2},[]
    with patch('drone_swarm.direct.training.evaluate',side_effect=fake):
        m,groups=validate_policy(None,c,3)
    assert calls==[123,124,125] and m=={'episodes':12,'success_rate':.5}
    assert len(groups)==3 and groups[-1]['metrics']['success_rate']==1.


def test_new_reward_changes_score_not_physics_or_observations():
    c=DirectConfig(device='cpu',es_population=2,hidden_layers=(8,),episode_seconds=.04,direct_reward_version=4)
    s=make_worlds(3,2,c,1)
    s=s._replace(static=s.static.at[:,0,:].set(s.spawn[:,0]+j.array([.25,0.,0.])))
    w=init_mlp(jax.random.PRNGKey(1),c.layers)
    old=DirectEnvironment(c,2).rollout(w,s)
    new=DirectEnvironment(replace(c,direct_reward_version=5),2).rollout(w,s)
    for name in ['history','motors','success','failed','causes','error']:
        np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
    np.testing.assert_array_equal(old.data.states.pos,new.data.states.pos)
    assert not bool(old.failed.any()) and bool((new.reward<old.reward).all())


def test_warm_start_keeps_baseline_when_first_update_is_worse():
    c=DirectConfig(device='cpu',es_population=4,batch_worlds=4,num_generations=1,eval_interval=1,
        validation_episodes=2,hidden_layers=(8,),episode_seconds=.04)
    good=dict(episodes=2,success_rate=1.,failure_rate=0.,hold_max_mean=3.,flight_seconds_mean=4.,fitness_mean=10.,truncated_rate=0.,hit_spin=0.)
    bad={**good,'success_rate':0.,'failure_rate':1.,'hold_max_mean':0.}
    validations=[(m,[{'seed':c.validation_seed,'metrics':m}]) for m in [good,bad]]
    with TemporaryDirectory() as d:
        source=Path(d)/'source.pkl';out=Path(d)/'run';w=init_mlp(jax.random.PRNGKey(1),c.layers);save(source,w,c)
        with patch('drone_swarm.direct.training.validate_policy',side_effect=validations):train(c,out,init_from=source)
        best,_=load(out/'best.pkl');latest,_=load(out/'latest.pkl')
        assert best['generation']==-1 and latest['generation']==0
        assert (out/'baseline.json').exists()
        for x,y in zip(jax.tree_util.tree_leaves(w),jax.tree_util.tree_leaves(best['weights'])):np.testing.assert_array_equal(x,y)
