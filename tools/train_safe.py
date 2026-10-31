"""Launch training in a bounded Linux user service; one run at a time."""
import fcntl,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
lock=(root/'.training.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Ya hay un entrenamiento protegido en ejecución.')
cpus=sorted(os.sched_getaffinity(0))[:2]
env_args=['XLA_PYTHON_CLIENT_PREALLOCATE=false','XLA_PYTHON_CLIENT_ALLOCATOR=platform',
          'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
command=['systemd-run','--user','--wait','--pipe','--collect',
         '-p','MemoryMax=12G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
         '-p','TasksMax=256','-p','Nice=10',f'--working-directory={root}',
         '/usr/bin/env',*env_args,'/usr/bin/taskset','-c',','.join(map(str,cpus)),
         sys.executable,'-u',str(root/'tools/train_direct.py'),*sys.argv[1:]]
print('Entrenamiento limitado: RAM 12 GiB, sin swap, 2 CPU; JAX sin reserva previa de VRAM.',flush=True)
raise SystemExit(subprocess.call(command))
