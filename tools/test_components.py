"""Run test modules sequentially in separate processes to release JAX caches."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if len(sys.argv)>1:
    import importlib.util,unittest
    sys.path.insert(0,str(root/'src'))
    path=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite=unittest.TestSuite(unittest.FunctionTestCase(v,description=n)
        for n,v in sorted(vars(module).items()) if n.startswith('test_') and callable(v))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
for path in sorted((root/'tests/integration/swarm_v2').glob('test_*.py')):
    result=subprocess.run([sys.executable,__file__,str(path)],cwd=root)
    if result.returncode:raise SystemExit(result.returncode)
