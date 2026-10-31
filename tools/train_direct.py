"""Evolve a direct-motor pilot in independent worlds, without mapped navigation or guidance."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import replace
from drone_swarm.direct.config import DirectConfig
from drone_swarm.direct.training import train
from drone_swarm.direct.artifacts import load

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--resume')
    source.add_argument('--init-from',help='New experiment from compatible weights; inherit model/physics, reset optimizer and budget')
    p.add_argument('--device',choices=('cpu','gpu','auto'),default='auto')
    p.add_argument('--population',type=int,default=8192);p.add_argument('--batch-worlds',type=int,default=256)
    p.add_argument('--generations',type=int,default=1000);p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--stage',type=int,choices=range(5),default=0);p.add_argument('--max-stage',type=int,choices=range(5),default=3)
    p.add_argument('--episode-seconds',type=float,default=12.)
    p.add_argument('--eval-interval',type=int,default=25);p.add_argument('--validation-episodes',type=int,default=40)
    p.add_argument('--validation-seed',type=int,help='Independent validation bank for a new experiment')
    p.add_argument('--validation-groups',type=int,help='Validate on this many fixed independent groups, each with validation-episodes missions')
    p.add_argument('--schedule-generations',type=int,help='Generations over which alpha/sigma decrease; 0 uses total budget')
    p.add_argument('--stall-generations',type=int,help='Stop after this many generations without progress; 0 disables early stop')
    p.add_argument('--physics-budget',type=int,default=0);p.add_argument('--smoke',action='store_true')
    p.add_argument('--alpha-start',type=float);p.add_argument('--alpha-end',type=float)
    p.add_argument('--sigma-start',type=float);p.add_argument('--sigma-end',type=float)
    p.add_argument('--profile',choices=('baseline','stability','preclaude'),help='preclaude: broad reward v4, 128x3 MLP, experimental legacy ES tuning')
    p.add_argument('--reward-version',type=int,choices=(1,2,3,4,5))
    p.add_argument('--observation-version',type=int,choices=(1,2))
    p.add_argument('--curriculum-generations',type=int)
    p.add_argument('--encounter-generations',type=int,help='v4: ramp mover count/speed over this many run generations; validation remains full difficulty')
    p.add_argument('--hidden-layers',type=int,nargs='+',help='Explicit MLP capacity for ablations')
    p.add_argument('--curriculum-version',type=int,choices=(1,2,3,4),help='v2: late maze; v3: fixed-axis crossings; v4: varied oblique crossings with optional training ramp')
    a=p.parse_args()
    tuning={name:getattr(a,name) for name in ('alpha_start','alpha_end','sigma_start','sigma_end')
            if getattr(a,name) is not None}
    if any(not (0.<value<float('inf')) for value in tuning.values()):
        p.error('Learning rates and mutation scales must be finite and positive')
    if a.resume and (tuning or a.profile is not None or a.reward_version is not None or
                     a.observation_version is not None or a.curriculum_generations is not None or
                     a.hidden_layers is not None or a.curriculum_version is not None or a.validation_seed is not None or
                     a.encounter_generations is not None or a.validation_groups is not None or a.schedule_generations is not None or a.stall_generations is not None):
        p.error('Resume preserves checkpoint settings; tuning requires a new experiment')
    if a.init_from and a.profile is not None:
        p.error('Initialization inherits the source profile; use explicit tuning overrides')
    if a.curriculum_generations is not None and a.curriculum_generations<0:
        p.error('Curriculum generations must be nonnegative')
    if a.hidden_layers is not None and min(a.hidden_layers)<1:p.error('Hidden layer sizes must be positive')
    if a.profile=='stability':
        tuning.update(direct_reward_version=3,direct_observation_version=2,
                      direct_curriculum_generations=min(100,max(1,a.generations//2)))
    if a.profile=='preclaude':
        # Explicit experimental profile, not a claim of transfer or convergence.
        defaults=dict(alpha_start=.02,alpha_end=.001,sigma_start=.05,sigma_end=.005)
        tuning={**defaults,**tuning}
        tuning.update(hidden_layers=(128,128,128),grad_clip=1.,direct_reward_version=4,
                      direct_observation_version=2,
                      direct_curriculum_generations=min(100,max(1,a.generations//2)))
    if a.hidden_layers is not None:tuning['hidden_layers']=tuple(a.hidden_layers)
    if a.curriculum_version is not None or not a.init_from:
        tuning['direct_curriculum_version']=a.curriculum_version if a.curriculum_version is not None else 2
    if a.validation_seed is not None:tuning['validation_seed']=a.validation_seed
    for arg,field,minimum in [('validation_groups','direct_validation_groups',1),
                             ('encounter_generations','direct_encounter_generations',0),('schedule_generations','direct_schedule_generations',0),('stall_generations','direct_stall_generations',0)]:
        value=getattr(a,arg)
        if value is not None:
            if value<minimum:p.error(f'{arg} must be at least {minimum}')
            tuning[field]=value
    for arg,field in [('reward_version','direct_reward_version'),('observation_version','direct_observation_version'),
                      ('curriculum_generations','direct_curriculum_generations')]:
        if getattr(a,arg) is not None:tuning[field]=getattr(a,arg)
    if a.resume:
        _,cfg=load(a.resume);cfg=replace(cfg,device=a.device)
    else:
        base=load(a.init_from)[1] if a.init_from else DirectConfig()
        cfg=replace(base,device=a.device,es_population=a.population,batch_worlds=a.batch_worlds,
            num_generations=a.generations,training_seed=a.seed,episode_seconds=a.episode_seconds,
            eval_interval=a.eval_interval,validation_episodes=a.validation_episodes,training_step_budget=a.physics_budget,**tuning)
    print(train(cfg,a.output,a.stage,a.max_stage,a.resume,a.smoke,init_from=a.init_from))

if __name__=='__main__':main()
