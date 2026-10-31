"""Trusted local direct-policy checkpoints; incompatible with v3 residual actions."""
from pathlib import Path
import json,pickle,hashlib,importlib.metadata
import jax
import numpy as np
from drone_swarm.direct.config import DirectConfig,CONTRACT

FIELDS=[('gravity_body',3),('gyro_body_over_10',3),('sim_relative_goal_body_over_5',3),
        ('sim_velocity_body_over_max_speed',3),('previous_motor_action',4)]

def save(path,weights,cfg,**state):
    root=Path(__file__).resolve().parents[1]
    payload={'contract_version':CONTRACT,'config':cfg.to_dict(),'weights':jax.device_get(weights),
             'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')},
             'dependencies':{n:importlib.metadata.version(n) for n in ('jax','jaxlib','crazyflow')},**jax.device_get(state)}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(payload,f)
    tmp.replace(path)

def load(path):
    with Path(path).open('rb') as f:payload=pickle.load(f)
    if payload.get('contract_version')!=CONTRACT:
        raise ValueError('Expected direct-motor v4 checkpoint; residual v3 weights are incompatible')
    settings=dict(payload['config']);settings['hidden_layers']=tuple(settings['hidden_layers'])
    settings.setdefault('direct_curriculum_version',1)
    settings.setdefault('direct_reward_version',1)
    settings.setdefault('direct_stall_generations',0)
    settings.setdefault('direct_observation_version',1)
    settings.setdefault('direct_curriculum_generations',0)
    cfg=DirectConfig(**settings)
    shapes=list(zip(cfg.layers[:-1],cfg.layers[1:]))
    if len(payload['weights'])!=len(shapes):raise ValueError('Layer count mismatch')
    for (w,b),(n,m) in zip(payload['weights'],shapes):
        if w.shape!=(n,m) or b.shape!=(m,) or not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError('Invalid direct policy tensors')
    return payload,cfg

def export(path,directory):
    from drone_swarm.perception.sensors import get_lidar_directions
    from drone_swarm.artifacts.checkpoints import numpy_forward
    from drone_swarm.learning.network import mlp_forward
    p,cfg=load(path);out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    obs=np.random.default_rng(4).normal(size=(8,cfg.input_size)).astype(np.float32)
    with jax.default_matmul_precision('highest'):
        np.testing.assert_allclose(numpy_forward(obs,p['weights']),mlp_forward(obs,p['weights']),rtol=2e-5,atol=2e-5)
    np.savez(out/'policy.npz',**{f'{kind}{i}':np.asarray(t) for i,layer in enumerate(p['weights']) for kind,t in zip(('W','b'),layer)})
    np.savez(out/'inference_vectors.npz',observation=obs,action=numpy_forward(obs,p['weights']))
    fields=FIELDS+ [('distance_over_range',cfg.num_lidars),('valid',cfg.num_lidars),('age_seconds_clipped_0_1',cfg.num_lidars)]
    if cfg.direct_observation_version==2:
        fields=[('gravity_body',3),('gyro_body_rad_s_clipped_20',3),('sim_relative_goal_body_over_2',3),
                ('sim_velocity_body_over_max_speed',3),('previous_motor_action',4),
                ('tof_proximity_0_free_1_contact_minus1_invalid_or_stale',cfg.num_lidars)]
    manifest={'contract_version':CONTRACT,'config':cfg.to_dict(),'fields':fields,
        'stack':'oldest to newest; concatenate frames','actions':['motor_0','motor_1','motor_2','motor_3'],
        'conversion':'thrust_N = (clip(action,-1,1)+1)*0.5*max_thrust_per_motor; motor order is Crazyflow order',
        'sensor_axes':np.asarray(get_lidar_directions(cfg.num_lidars)).tolist(),
        'requires':'IMU/AHRS, ToF, relative goal and body velocity supplied by simulator navigation; real navigation provider NOT implemented',
        'composition':'No guidance, obstacle filter, PI, attitude or rate stabilizer; motor_tau models actuator lag',
        'activation':'leaky_relu slope .01; tanh output','deployment_status':'research simulation only'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
