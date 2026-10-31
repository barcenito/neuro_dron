from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import hashlib
import jax
import numpy as np
from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.artifacts import save,load
from drone_swarm.direct.training import initial_policy,train
from drone_swarm.learning.network import init_mlp


def test_initialization_preserves_weights_and_rejects_semantic_changes():
    cfg=DirectConfig(device='cpu',hidden_layers=(8,),direct_observation_version=2)
    weights=init_mlp(jax.random.PRNGKey(7),cfg.layers)
    with TemporaryDirectory() as d:
        source=Path(d)/'source.pkl';save(source,weights,cfg,generation=199)
        result,origin=initial_policy(source,replace(cfg,num_generations=2,training_seed=17))
        for a,b in zip(jax.tree_util.tree_leaves(result),jax.tree_util.tree_leaves(weights)):
            np.testing.assert_array_equal(a,b)
        assert origin['generation']==199 and origin['optimizer_reset']
        assert origin['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        for changed in [replace(cfg,hidden_layers=(16,)),replace(cfg,max_speed=3.),replace(cfg,direct_observation_version=1)]:
            with TestCase().assertRaisesRegex(ValueError,'Incompatible'):initial_policy(source,changed)
        save(source,weights,cfg,smoke_only=True)
        with TestCase().assertRaisesRegex(ValueError,'Incompatible'):initial_policy(source,cfg)


def test_warm_start_resets_budget_and_resume_keeps_origin():
    cfg=DirectConfig(device='cpu',hidden_layers=(8,),es_population=4,batch_worlds=4,
        num_generations=1,validation_episodes=2,eval_interval=1,episode_seconds=.04)
    weights=init_mlp(jax.random.PRNGKey(7),cfg.layers)
    with TemporaryDirectory() as d:
        source=Path(d)/'source.pkl';out=Path(d)/'new'
        save(source,weights,cfg,generation=199,physical_steps=999999)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        result=train(cfg,out,stage=3,max_stage=3,smoke=True,init_from=source)
        initial,_=load(out/'initial.pkl');last,_=load(result)
        assert last['generation']==0 and last['physical_steps']<999999
        assert last['initialization']==initial['initialization']
        assert last['evaluated_stage']==3
        for a,b in zip(jax.tree_util.tree_leaves(initial['weights']),jax.tree_util.tree_leaves(weights)):
            np.testing.assert_array_equal(a,b)
        train(cfg,out,stage=3,max_stage=3,smoke=True,resume=result)
        assert load(result)[0]['initialization']==last['initialization']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        with TestCase().assertRaisesRegex(ValueError,'Choose resume'):
            train(cfg,out,resume=result,init_from=source)
