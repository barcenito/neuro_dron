"""Global antithetic ES ranks across GPU batches; batching does not split evolution."""
import csv,json,time,hashlib
from pathlib import Path
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter
from drone_swarm.runtime import configure_device,device_info
from drone_swarm.learning.network import init_mlp,generate_and_apply_mutations,es_update
from drone_swarm.direct.worlds import make_worlds,curriculum_stages
from drone_swarm.direct.environment import DirectEnvironment,summarize
from drone_swarm.direct.artifacts import save,load

def combine(summaries):
    total=sum(s['episodes'] for s in summaries)
    return {k:(total if k=='episodes' else sum(s[k]*s['episodes'] for s in summaries)/total) for k in summaries[0]}

def physical_progress(metrics):
    return (metrics['success_rate'],1-metrics['failure_rate'],
            int(metrics['hold_max_mean']*10),int(metrics['flight_seconds_mean']*10))

def collapsed_policy(metrics):
    return metrics['success_rate']==0 and metrics['hit_spin']>=.95 and metrics['flight_seconds_mean']<.5

def training_stop_reason(collapse_streak,generation,last_progress,stall_budget,promoted=False):
    if not stall_budget:return None
    if collapse_streak>=3:return 'repeated_early_spin_failure'
    if generation-last_progress>=stall_budget and not promoted:return 'no_physical_progress'
    return None

def evaluate(weights,cfg,stage,seed,episodes=200,max_steps=None,envs=None):
    # Keep compiled rollouts across validations, while resetting each scenario.
    cfg,_=configure_device(cfg)
    summaries=[];rewards=[]
    if envs is None:envs={}
    for start in range(0,episodes,cfg.batch_worlds):
        n=min(cfg.batch_worlds,episodes-start)
        if n not in envs:envs[n]=DirectEnvironment(cfg,n)
        elif envs[n].cfg!=cfg:raise ValueError('Evaluation cache configuration mismatch')
        s=make_worlds(seed,n,cfg,stage,start_index=start)
        ep=envs[n].rollout(weights,s,max_steps=max_steps)
        summaries.append(summarize(ep,include_stability=cfg.direct_reward_version>=3));rewards.extend(np.asarray(ep.reward).tolist())
    return combine(summaries),rewards

def training_difficulty(cfg,generation,stage):
    if stage!=0 or cfg.direct_curriculum_generations<=1:return 1.
    return min(1.,.25+.75*generation/max(cfg.direct_curriculum_generations-1,1))

def encounter_progress(cfg,generation):
    if cfg.direct_curriculum_version!=4:return None
    return min(1.,generation/max(cfg.direct_encounter_generations-1,1)) if cfg.direct_encounter_generations else 1.

def learning_scales(cfg,generation):
    duration=cfg.direct_schedule_generations or cfg.num_generations
    fraction=min(1.,generation/max(duration-1,1))
    return (cfg.alpha_start+(cfg.alpha_end-cfg.alpha_start)*fraction,
            cfg.sigma_start+(cfg.sigma_end-cfg.sigma_start)*fraction)

def validate_policy(weights,cfg,stage,smoke=False,envs=None):
    groups=[]
    for offset in range(cfg.direct_validation_groups):
        seed=cfg.validation_seed+offset
        metrics,_=evaluate(weights,cfg,stage,seed,cfg.validation_episodes,
                           max_steps=20 if smoke else None,envs=envs)
        groups.append({'seed':seed,'metrics':metrics})
    return (groups[0]['metrics'] if len(groups)==1 else combine([g['metrics'] for g in groups])),groups

def checkpoint_score(cfg,stage,val):
    if cfg.direct_reward_version>=3:
        return (stage,val['success_rate'],1-val['failure_rate'],
                int(val['hold_max_mean']*10),val['fitness_mean'])
    if cfg.direct_reward_version==2:return (stage,*physical_progress(val),val['fitness_mean'])
    return (stage,val['success_rate'],1-val['failure_rate'],val['fitness_mean'])

def initial_policy(path,cfg):
    """Transfer weights only; preserve observation/action semantics, not optimizer state."""
    payload,old=load(path)
    fields=('hidden_layers','num_frames','num_lidars','direct_observation_version',
            'max_speed','max_thrust_per_motor','lidar_max_range','env_freq')
    changed=[name for name in fields if getattr(old,name)!=getattr(cfg,name)]
    if payload.get('smoke_only') or changed:
        raise ValueError(f'Incompatible initialization checkpoint: smoke or changed {changed}')
    provenance={'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'generation':payload.get('generation'),'optimizer_reset':True}
    return jax.tree_util.tree_map(jnp.asarray,payload['weights']),provenance

def train(cfg,output,stage=0,max_stage=3,resume=None,smoke=False,init_from=None):
    if resume and init_from:raise ValueError('Choose resume or init_from, not both')
    cfg,device=configure_device(cfg);out=Path(output)
    last_stage=4 if cfg.direct_curriculum_version>=2 else 3
    if not 0<=stage<=max_stage<=last_stage:raise ValueError(f'Individual curriculum supports stages 0..{last_stage}')
    if cfg.swarm_size!=1:raise ValueError('Train individual worlds; evaluate shared weights collectively')
    if out.exists() and any(out.iterdir()) and not resume:raise FileExistsError(f'Preserve existing run: {out}')
    out.mkdir(parents=True,exist_ok=True)
    key=jax.random.PRNGKey(cfg.training_seed);key,init_key=jax.random.split(key)
    weights=init_mlp(init_key,cfg.layers);initialization=None
    if init_from:weights,initialization=initial_policy(init_from,cfg)
    m=jax.tree_util.tree_map(jnp.zeros_like,weights);v=m
    first=0;best=None;streak=0;physical_steps=0
    best_progress=None;last_progress=0;collapse_streak=0
    if resume:
        old,old_cfg=load(resume)
        initialization=old.get('initialization')
        if replace(old_cfg,device=cfg.device)!=cfg or bool(old.get('smoke_only'))!=smoke:
            raise ValueError('Resume needs identical configuration and smoke status')
        if old.get('max_stage')!=max_stage:raise ValueError('Resume must preserve max_stage')
        weights=jax.tree_util.tree_map(jnp.asarray,old['weights']);m=old['adam_m'];v=old['adam_v'];key=jnp.asarray(old['rng_key'])
        first=old['generation']+1;stage=old['next_stage'];streak=old['streak'];best=old['best_score'];physical_steps=old['physical_steps']
        best_progress=old.get('best_progress');last_progress=old.get('last_progress',first);collapse_streak=old.get('collapse_streak',0)
    else:save(out/'initial.pkl',weights,cfg,generation=-1,smoke_only=smoke,initialization=initialization)
    (out/'experiment.json').write_text(json.dumps({'initialization':initialization,'start_stage':stage,
        'max_stage':max_stage,'resume':str(resume) if resume else None},indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps(device_info(device),indent=2)+'\n')
    print('Direct motors:',json.dumps(device_info(device)),'population',cfg.es_population,'batch',cfg.batch_worlds,flush=True)
    envs={};validation_envs={};update=jax.jit(lambda w,mu,r,m,v,i,a,sd:es_update(w,mu,r,m,v,i,a,sd,cfg))
    if init_from and not smoke:
        val,groups=validate_policy(weights,cfg,stage,envs=validation_envs)
        best=checkpoint_score(cfg,stage,val);best_progress=(stage,*physical_progress(val))
        state=dict(generation=-1,adam_m=m,adam_v=v,rng_key=key,best_score=best,streak=0,next_stage=stage,
            evaluated_stage=stage,max_stage=max_stage,validation=val,validation_groups=groups,physical_steps=0,
            smoke_only=False,best_progress=best_progress,last_progress=0,collapse_streak=0,stop_reason=None,
            initialization=initialization)
        save(out/'best.pkl',weights,cfg,**state);save(out/f'best_stage_{stage}.pkl',weights,cfg,**state)
        (out/'baseline.json').write_text(json.dumps({'stage':stage,**val,'groups':groups},indent=2)+'\n')
        print('baseline',json.dumps(val),flush=True)
    with (out/'metrics.csv').open('a' if resume else 'w',newline='') as log, SummaryWriter(str(out)) as tb:
        writer=None
        for gen in range(first,cfg.num_generations):
            begin=time.perf_counter();key,mk,sk=jax.random.split(key,3)
            alpha,sigma=learning_scales(cfg,gen)
            seed=int(jax.random.randint(sk,(),0,2**30))
            progress=encounter_progress(cfg,gen)
            s=make_worlds(seed,cfg.es_population,cfg,stage,paired=True,encounter_progress=progress)
            mu=generate_and_apply_mutations(mk,cfg.layers,cfg.mutation_rank,sigma,cfg.es_population)
            rewards=[];summaries=[]
            difficulty=training_difficulty(cfg,gen,stage)
            for start in range(0,cfg.es_population,cfg.batch_worlds):
                end=min(start+cfg.batch_worlds,cfg.es_population);n=end-start
                if n not in envs:envs[n]=DirectEnvironment(cfg,n)
                scene=jax.tree_util.tree_map(lambda a:a[start:end],s)
                mutation=jax.tree_util.tree_map(lambda a:a[start:end],mu)
                ep=envs[n].rollout(weights,scene,mutation,noise=difficulty,
                    max_steps=20 if smoke else None,difficulty=difficulty)
                rewards.append(ep.reward);summaries.append(summarize(ep,include_stability=cfg.direct_reward_version>=3))
                physical_steps+=round(float(jnp.sum(ep.exposure))*cfg.physics_freq)
            rewards=jnp.concatenate(rewards)
            if not bool(jnp.isfinite(rewards).all()):raise RuntimeError('Nonfinite ES returns')
            weights,m,v,norm=update(weights,mu,rewards,m,v,gen+1,alpha,sigma);jax.block_until_ready(weights)
            row={'generation':gen,'stage':stage,'alpha':alpha,'sigma':sigma,'gradient_norm':float(norm),
                 'seconds':time.perf_counter()-begin,'physical_steps':physical_steps,**combine(summaries)}
            if cfg.direct_curriculum_generations:row['training_difficulty']=difficulty
            if progress is not None:
                row['encounter_progress']=progress
                row['moving_obstacles_mean']=float(np.mean(np.sum(np.linalg.norm(np.asarray(s.dynamic_center),axis=-1)<30,axis=-1)))
            if writer is None:
                writer=csv.DictWriter(log,fieldnames=list(row))
                if not resume:writer.writeheader()
            writer.writerow(row);log.flush()
            for k,value in row.items():tb.add_scalar('train/'+k,value,gen)
            print('generation',gen,'stage',curriculum_stages(cfg)[stage],json.dumps(row),flush=True)
            budget=cfg.training_step_budget>0 and physical_steps>=cfg.training_step_budget
            if (gen+1)%cfg.eval_interval==0 or gen==cfg.num_generations-1 or budget:
                val,groups=validate_policy(weights,cfg,stage,smoke,validation_envs)
                score=checkpoint_score(cfg,stage,val)
                improved=best is None or score>tuple(best)
                if improved:best=score
                eligible=all(g['metrics']['success_rate']>=cfg.promotion_success and
                    g['metrics']['failure_rate']<=cfg.promotion_collision and g['metrics']['truncated_rate']==0 for g in groups)
                streak=streak+1 if eligible and not smoke else 0
                next_stage=stage
                if streak>=cfg.promotion_streak and stage<max_stage:next_stage=stage+1;streak=0
                progress=(stage,*physical_progress(val))
                if best_progress is None or progress>tuple(best_progress):
                    best_progress=progress;last_progress=gen+1
                collapse_streak=collapse_streak+1 if collapsed_policy(val) else 0
                stop_reason=None
                if cfg.direct_stall_generations and not smoke:
                    stop_reason=training_stop_reason(collapse_streak,gen+1,last_progress,
                                                     cfg.direct_stall_generations,next_stage!=stage)
                state=dict(generation=gen,adam_m=m,adam_v=v,rng_key=key,best_score=best,streak=streak,next_stage=next_stage,
                           evaluated_stage=stage,max_stage=max_stage,validation=val,physical_steps=physical_steps,smoke_only=smoke,
                           best_progress=best_progress,last_progress=last_progress,collapse_streak=collapse_streak,stop_reason=stop_reason,
                           initialization=initialization,validation_groups=groups)
                if improved:save(out/'best.pkl',weights,cfg,**state);save(out/f'best_stage_{stage}.pkl',weights,cfg,**state)
                save(out/'latest.pkl',weights,cfg,**state)
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps({'generation':gen,'stage':stage,'next_stage':next_stage,**val,'groups':groups})+'\n')
                for k,value in val.items():tb.add_scalar('val/'+k,value,gen)
                print('validation',json.dumps(val),'next_stage',next_stage,'stop_reason',stop_reason,flush=True)
                stage=next_stage
                if stop_reason:
                    (out/'stop.json').write_text(json.dumps({'generation':gen,'reason':stop_reason,'validation':val},indent=2)+'\n')
                    break
            if budget:break
    return out/'latest.pkl'
