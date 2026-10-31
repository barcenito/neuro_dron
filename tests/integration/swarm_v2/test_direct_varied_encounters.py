"""Physical invariants for v4 encounters and separation of training/evaluation."""
from dataclasses import replace
import numpy as np
import jax.numpy as j
from unittest import TestCase
from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.worlds import make_worlds,hold_ready
from drone_swarm.direct.training import encounter_progress
from drone_swarm.simulation.scenarios import obstacle_states


def config():
    return DirectConfig(device='cpu',es_population=4,hidden_layers=(8,),episode_seconds=20.,
                        direct_curriculum_version=4,direct_encounter_generations=100)


def test_paths_cross_at_varied_times_and_stay_inside_late_maze_room():
    cfg=config();s=make_worlds(911020,256,cfg,4)
    center=np.asarray(s.dynamic_center);amplitude=np.asarray(s.dynamic_amplitude)
    crossing=-np.asarray(s.dynamic_phase[:,:,0])/np.asarray(s.dynamic_frequency[:,:,0])
    assert np.all((crossing>=2)&(crossing<=11.00001))
    for k in range(3):
        pos,vel=obstacle_states(s,j.asarray(crossing[:,k]))
        np.testing.assert_allclose(np.asarray(pos)[:,k],center[:,k],atol=2e-6)
        assert np.all(np.linalg.norm(np.asarray(vel)[:,k],axis=-1)>0)
    low=center-np.abs(amplitude);high=center+np.abs(amplitude)
    assert np.all(low>np.array([.05+cfg.obstacle_radius,-2.9+cfg.obstacle_radius,cfg.obstacle_radius]))
    assert np.all(high<3.-cfg.obstacle_radius)
    # Both sides of the target and oblique directions, not a single safe offset.
    offsets=center[:,1:,:2]-np.asarray(s.routes[:,0,-1])[:,None,:2]
    assert np.all(np.min(offsets,axis=(0,1))<-.3)
    assert np.all(np.max(offsets,axis=(0,1))>.3)
    assert np.all(np.count_nonzero(np.abs(amplitude)>1e-5,axis=-1)==3)
    assert not bool(hold_ready(s,11.99,cfg).any())
    assert bool(hold_ready(s,12.,cfg).all())


def test_training_progress_does_not_leak_into_evaluation():
    cfg=config()
    active=lambda s:np.sum(np.linalg.norm(np.asarray(s.dynamic_center),axis=-1)<30,axis=-1)
    easy=make_worlds(17,256,cfg,3,encounter_progress=0.)
    full=make_worlds(17,256,cfg,3)
    mixed=make_worlds(17,256,cfg,3,encounter_progress=1.)
    assert np.all(active(easy)==1) and np.all(active(full)==3)
    assert set(active(mixed))=={1,2,3}
    # Single-obstacle scenes include late passages, not just early transit.
    assert np.any(np.asarray(easy.dynamic_center)[:,2,0]<30)
    assert encounter_progress(cfg,0)==0 and encounter_progress(cfg,99)==1
    assert encounter_progress(cfg,1000)==1
    assert encounter_progress(replace(cfg,direct_curriculum_version=3,direct_encounter_generations=0),0) is None
    with TestCase().assertRaises(ValueError):make_worlds(1,2,cfg,3,encounter_progress=float('nan'))


def test_pairs_chunks_and_early_stages_remain_reproducible():
    cfg=config();old=replace(cfg,direct_curriculum_version=3,direct_encounter_generations=0)
    for stage in range(3):
        a=make_worlds(1,4,cfg,stage);b=make_worlds(1,4,old,stage)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    for progress in [None,0.,.5,1.]:
        a=make_worlds(3,4,cfg,3,encounter_progress=progress)
        b=make_worlds(3,8,cfg,3,paired=True,encounter_progress=progress)
        c=make_worlds(3,2,cfg,3,start_index=2,encounter_progress=progress)
        for x,y,z in zip(a,b,c):
            np.testing.assert_array_equal(x,y[:4]);np.testing.assert_array_equal(x,y[4:])
            np.testing.assert_array_equal(x[2:],z)
