from dataclasses import replace
import jax.numpy as j
import numpy as np
from drone_swarm.direct.environment import direct_reward_v2
from drone_swarm.direct.training import physical_progress,collapsed_policy

def test_early_failure_cannot_escape_the_remaining_horizon_cost():
    # Even maximum accrued density before immediate failure must lose to a
    # substantially longer flight at minimum density. No terminal distance reward.
    def failed_at(t,density):
        return density*(t-.01)+float(direct_reward_v2(density,True,False,True,False,t,12.,.01))
    assert failed_at(.24,2.)<failed_at(1.75,0.)
    assert failed_at(.24,0.)<failed_at(1.75,0.)<failed_at(10.,0.)
    # Succeeding sooner does not lose the positive occupancy integral.
    a=2.*2.99+float(direct_reward_v2(2.,True,True,False,False,3.,12.,.01))
    b=2.*10.99+float(direct_reward_v2(2.,True,True,False,False,11.,12.,.01))
    np.testing.assert_allclose(a,b,atol=1e-5)
    assert a>failed_at(11.,2.)
    assert float(direct_reward_v2(2.,False,False,False,False,4.,12.,.01))==0.

def test_collapse_and_progress_ignore_tiny_fitness_improvements():
    m={'success_rate':0.,'failure_rate':1.,'hold_max_mean':.06,'flight_seconds_mean':.237,'hit_spin':1.}
    assert collapsed_policy(m)
    assert physical_progress(m)==physical_progress({**m,'flight_seconds_mean':.239})
    assert physical_progress({**m,'flight_seconds_mean':1.5})>physical_progress(m)
    assert not collapsed_policy({**m,'hit_spin':.2})
    from drone_swarm.direct.training import training_stop_reason
    assert training_stop_reason(3,75,75,200)=='repeated_early_spin_failure'
    assert training_stop_reason(0,225,25,200)=='no_physical_progress'
    assert training_stop_reason(0,225,25,200,promoted=True) is None
    assert training_stop_reason(3,1000,0,0) is None  # historical exact resume

def test_old_direct_checkpoints_keep_original_reward():
    import tempfile,pickle
    from pathlib import Path
    import jax
    from drone_swarm.direct.config import DirectConfig
    from drone_swarm.direct.artifacts import load,save
    from drone_swarm.learning.network import init_mlp
    cfg=DirectConfig(hidden_layers=(8,),es_population=2)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'model.pkl';save(p,init_mlp(jax.random.PRNGKey(4),cfg.layers),cfg)
        payload,loaded=load(p);assert loaded.direct_reward_version==2
        del payload['config']['direct_reward_version'];del payload['config']['direct_stall_generations']
        with p.open('wb') as f:pickle.dump(payload,f)
        _,old=load(p);assert old.direct_reward_version==1 and old.direct_stall_generations==0
