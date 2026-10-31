from dataclasses import replace

import jax
import numpy as np
from unittest import TestCase

from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.training import evaluate
from drone_swarm.learning.network import init_mlp


def test_cached_evaluation_resets_worlds_and_rejects_changed_physics():
    cfg = DirectConfig(device='cpu', es_population=4, batch_worlds=2,
                       hidden_layers=(8,), episode_seconds=.04)
    weights = init_mlp(jax.random.PRNGKey(42), cfg.layers)
    cache = {}
    first, rewards = evaluate(weights, cfg, 0, 17, 3, envs=cache)
    identities = {n: id(env) for n, env in cache.items()}
    evaluate(weights, cfg, 1, 18, 3, envs=cache)
    repeated, repeated_rewards = evaluate(weights, cfg, 0, 17, 3, envs=cache)
    assert identities == {n: id(env) for n, env in cache.items()}
    assert first == repeated
    np.testing.assert_array_equal(rewards, repeated_rewards)
    with TestCase().assertRaisesRegex(ValueError, 'configuration mismatch'):
        evaluate(weights, replace(cfg, motor_tau=.05), 0, 17, 3, envs=cache)
