"""One reusable fresh-process safety contract for stdlib-only planning CLIs."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def assert_lightweight_cli(case, module, calls, assertions=""):
    # These are our own entry points, not mocked third-party dependencies.
    program = r'''
import builtins, contextlib, importlib, io, json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
blocked = {'torch','numpy','sionna','tensorflow','models','utils','main_train','baseline_evaluate'}
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.',1)[0] in blocked:
        raise AssertionError('Heavy import: '+name)
    return real_import(name,*args,**kwargs)
def audit(event,args):
    if event == 'os.mkdir':
        raise AssertionError('Planning created a directory')
    if event == 'open':
        path,mode,flags=args
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            raise AssertionError('Planning wrote a file')
        if isinstance(path,(str,bytes)) and os.fsdecode(path).lower().endswith(('.pth','.pt','.ckpt','.safetensors')):
            raise AssertionError('Planning read weights')
builtins.__import__=guarded
sys.addaudithook(audit)
entry=importlib.import_module(sys.argv[2])
with contextlib.redirect_stdout(io.StringIO()) as help_out:
    assert entry.main([])==0
assert '--dry-run' in help_out.getvalue()
plans=[]
for arguments in json.loads(sys.argv[3]):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        assert entry.main(arguments)==0
    plans.append(json.loads(output.getvalue()))
assert not blocked.intersection(sys.modules)
'''
    import json
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-S", "-c", program + "\n" + assertions,
             str(ROOT), module, json.dumps(calls)], cwd=directory,
            capture_output=True, text=True, timeout=45)
        case.assertEqual(list(Path(directory).iterdir()), [])
    case.assertEqual(result.returncode, 0, result.stdout + result.stderr)
