"""Real local-Git launcher guards; tmux/GPU/training are never started."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

LAUNCHER = Path(__file__).resolve().parents[1]/'tools/start_twc_v1_k8_ce_refine_new.sh'


class LauncherGuards(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/'repo with spaces';self.root.mkdir()
        (self.root/'tools').mkdir();shutil.copy2(LAUNCHER,self.root/'tools'/LAUNCHER.name)
        (self.root/'configs').mkdir();(self.root/'configs/__init__.py').write_text('')
        (self.root/'configs/twc_v1_k8_ce_refine_config.py').write_text(
            "CORE_FILES=('science.py','configs/twc_v1_k8_ce_refine_config.py')\n")
        (self.root/'science.py').write_text('original scientific code\n')
        (self.root/'.gitignore').write_text('runs/\nterminal_logs/\n')
        self.git('init','--initial-branch=main');self.git('add','.')
        self.git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture')
        source=self.root/'runs/twc_v1_k8_icc_aligned_savefix_v1/training/ce';source.mkdir(parents=True)
        for name in ('checkpoint_latest.pt','terminal_receipt.json','manifest.json','initialization.json'):
            (source/name).write_text('shell fixture only; no real model\n')
        self.bin=self.root.parent/'stubs';self.bin.mkdir()
        env_bin=self.root.parent/'environment/bin';env_bin.mkdir(parents=True)
        self.calls=self.root.parent/'calls.log'
        self.executable(self.bin/'id','#!/bin/sh\nprintf "%s\\n" rong\n')
        self.executable(self.bin/'nvidia-smi','#!/bin/sh\nexit 91\n')
        self.executable(self.bin/'tmux',
            '#!/bin/sh\nif test "$1" = has-session; then exit 1; fi\n'
            'printf "tmux:%s\\n" "$*" >> "$TEST_CALLS"\n')
        self.executable(env_bin/'python',
            '#!/bin/sh\nif test "$2" = -S && test "$3" = -c; then\n'
            ' exec '+shlex.quote(sys.executable)+' "$@"\nfi\n'
            'printf "python:%s\\n" "$*" >> "$TEST_CALLS"\n')
        self.env=dict(os.environ,PATH=str(self.bin)+os.pathsep+os.environ['PATH'],
            CONDA_PREFIX=str(env_bin.parent),TEST_CALLS=str(self.calls))

    def executable(self,path,text):path.write_text(text);path.chmod(0o755)

    def git(self,*args):
        return subprocess.run(['git',*args],cwd=self.root,check=True,capture_output=True,text=True).stdout

    def launch(self):
        return subprocess.run(['bash',str(self.root/'tools'/LAUNCHER.name)],
            cwd=self.root,env=self.env,capture_output=True,text=True)

    def test_unrelated_paper_and_gitignore_changes_are_preserved_and_launch_allowed(self):
        (self.root/'overleaf').mkdir();paper=self.root/'overleaf/main.tex';paper.write_text('private paper\n')
        (self.root/'.gitignore').write_text('runs/\nterminal_logs/\n*.private\n')
        result=self.launch()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        calls=self.calls.read_text()
        self.assertIn('-m unittest discover -s tests -p test_twc_v1_k8_ce_refine*.py',calls)
        self.assertIn('run_twc_v1_k8_ce_refine.py --dry-run',calls)
        self.assertIn('tmux:new-session',calls)
        self.assertIn('--min-free-mib 12000 --max-gpu-util 80',calls)
        self.assertEqual(paper.read_text(),'private paper\n')
        self.assertIn('*.private',(self.root/'.gitignore').read_text())

    def test_dirty_scientific_core_stops_before_any_training_launch(self):
        file=self.root/'science.py';file.write_text('uncommitted science edit\n')
        result=self.launch()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('scientific core has local changes',result.stdout+result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertEqual(file.read_text(),'uncommitted science edit\n')
        self.git('add','science.py')
        result=self.launch()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.calls.exists())

    def test_existing_output_is_retained_and_only_summary_requested(self):
        output=self.root/'runs/twc_v1_k8_ce_refine_v1';output.mkdir()
        marker=output/'do-not-touch';marker.write_text('saved evidence\n')
        result=self.launch()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('existing CE run is preserved',result.stdout+result.stderr)
        calls=self.calls.read_text()
        self.assertIn('--summary',calls)
        self.assertNotIn('--run',calls)
        self.assertNotIn('tmux:',calls)
        self.assertEqual(marker.read_text(),'saved evidence\n')


if __name__=='__main__':unittest.main()
