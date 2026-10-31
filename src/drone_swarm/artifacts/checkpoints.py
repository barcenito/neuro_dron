"""Trusted local checkpoints, schema checks and portable inference export."""
from pathlib import Path
import json,pickle,os
import numpy as np
import jax
from drone_swarm.config import SwarmConfig,CONTRACT_VERSION,OBSERVATION_VERSION,ACTION_VERSION,REWARD_VERSION

OBSERVATION_FIELDS=[('gravity_body',3),('velocity_body_normalized',3),('gyro_body_normalized',3),
                    ('target_body_normalized',3),('reference_velocity_body_normalized',3),
                    ('pose_confidence',1),('odometry_uncertainty_normalized',1),('phase_wait_transit_final',3),('previous_action',4)]


def save_checkpoint(path,weights,cfg,**state):
    import hashlib,importlib.metadata
    root=Path(__file__).resolve().parents[1]
    provenance={'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')},
                'dependencies':{name:importlib.metadata.version(name) for name in ('jax','jaxlib','crazyflow')}}
    payload={'provenance':provenance,'contract_version' :CONTRACT_VERSION,'observation_version':OBSERVATION_VERSION,'action_version':ACTION_VERSION,'reward_version':REWARD_VERSION,
             'config':cfg.to_dict(),'weights':jax.device_get(weights),**jax.device_get(state)}
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f: pickle.dump(payload,f)
    os.replace(tmp,p)


def load_checkpoint(path):
    # Pickle is only for checkpoints you trust; use npz export for deployment.
    with open(path,'rb') as f: p=pickle.load(f)
    if not isinstance(p,dict) or p.get('contract_version')!=CONTRACT_VERSION:
        raise ValueError('Legacy/incompatible checkpoint. Camera-free v3 requires a new training run; v2 weights remain untouched.')
    for field,version in [('observation_version',OBSERVATION_VERSION),('action_version',ACTION_VERSION),('reward_version',REWARD_VERSION)]:
        if p.get(field)!=version: raise ValueError(f'Incompatible {field}')
    # Early v3 pilots predate these configurable schedules. Preserve their
    # recorded reward/noise semantics for evaluation and exact resume.
    settings=dict(p['config'])
    settings.setdefault('preflight_seconds',0.)
    settings.setdefault('level_flight',False)
    settings.setdefault('mission_mode','slots')
    settings.setdefault('scenario_family','legacy')
    settings.setdefault('effort_weight',.05)
    settings.setdefault('dynamic_speed_scale',1.)
    settings.setdefault('control_profile','legacy')
    settings.setdefault('dynamic_scale',1.0)
    settings.setdefault('residual_vertical_speed',.15)
    settings.setdefault('avoidance_cones',False)
    settings.setdefault('checkpoint_safety_first',False)
    settings.setdefault('range_localization',False)
    settings.setdefault('enclosed_room',False)
    settings.setdefault('success_time_weight',0.)
    settings.setdefault('training_noise_start',.25)
    settings.setdefault('training_noise_stage_increment',.15)
    cfg=SwarmConfig(**settings)
    expected=list(zip(cfg.layers[:-1],cfg.layers[1:]))
    if len(p['weights'])!=len(expected): raise ValueError('Wrong layer count')
    for (w,b),(n,m) in zip(p['weights'],expected):
        if w.shape!=(n,m) or b.shape!=(m,): raise ValueError('Checkpoint tensor shape mismatch')
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)): raise ValueError('Nonfinite checkpoint')
    return p,cfg


def numpy_forward(obs,weights):
    x=np.asarray(obs,dtype=np.float32)
    for i,(w,b) in enumerate(weights):
        x=x@w+b
        if i<len(weights)-1: x=np.where(x>=0,x,.01*x)
    return np.tanh(x)


def export_checkpoint(path,directory):
    from drone_swarm.perception.sensors import get_lidar_directions
    p,cfg=load_checkpoint(path); out=Path(directory); out.mkdir(parents=True,exist_ok=True)
    tensors={f'{kind}{i}':np.asarray(t) for i,layer in enumerate(p['weights']) for kind,t in zip(('W','b'),layer)}
    np.savez(out/'policy.npz',**tensors)
    test=np.random.default_rng(7).normal(size=(8,cfg.input_size)).astype(np.float32)
    from drone_swarm.learning.network import mlp_forward
    runtime_actions=np.asarray(mlp_forward(test,p['weights']))
    with jax.default_matmul_precision('highest'):
        reference_actions=np.asarray(mlp_forward(test,p['weights']))
    np.testing.assert_allclose(numpy_forward(test,p['weights']),reference_actions,atol=2e-5,rtol=2e-5)
    runtime_error=float(np.max(np.abs(runtime_actions-numpy_forward(test,p['weights']))))
    np.savez(out/'inference_vectors.npz',observation=test,action=numpy_forward(test,p['weights']))
    manifest={'contract_version':CONTRACT_VERSION,'config':cfg.to_dict(),'fields':OBSERVATION_FIELDS+
              [('distance_over_range',cfg.num_lidars),('valid',cfg.num_lidars),('age_seconds_clipped_0_1',cfg.num_lidars)],
              'stack':'oldest to newest; concatenate frames','quaternion':'xyzw; body to world',
              'actions':['residual_vx','residual_vy','residual_vz','residual_yaw'],
              'composition':'environment.policy_action: profile-specific estimated guidance and bounded residual; guarded uses measured-range velocity constraints AFTER residual; clamp command to [-1,1]',
              'sensor_axes':np.asarray(get_lidar_directions(cfg.num_lidars)).tolist(),
              'reference_matmul_precision':'highest (float32); runtime default may use TF32',
              'runtime_reference_max_abs_difference':runtime_error,
              'activation':'leaky_relu slope 0.01; tanh output','dtype':'float32',
              'requires':'calibrated launch pose, IMU + ToF estimator, preloaded mission and wall map, policy_action composer, velocity controller, odometry supervisor'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
