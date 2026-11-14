"""Local synthetic profiles, real PTYs and simulated service transitions."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import pty
import shutil
import stat
import subprocess
import sys
import tempfile
import termios
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('uk_proxy_setup',ROOT/'tools/setup_uk_proxy.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(username='FAKE_USER',domain='FAKE_DOMAIN',auth='NTLMv2',workstation='')
        self.ca = Path('/etc/ssl/certs/ca-certificates.crt').read_bytes()
        self.native = self.root/'fake-native'
        self.native.write_text('''#!/usr/bin/python3
import os,sys,termios
assert os.isatty(0)
assert not termios.tcgetattr(0)[3] & (termios.ECHO|termios.ECHONL)
assert '-H' in sys.argv and '-p' not in sys.argv and '-v' not in sys.argv and '-T' not in sys.argv
assert '-l' not in sys.argv and 'proxyuk.huawei.com:8080' not in sys.argv
secret=sys.stdin.buffer.readline().rstrip(b'\\n')
if secret != b'FAKE_PASSWORD': raise SystemExit(7)
print('Password: ')
print('PassLM '+'B'*32)
print('PassNT '+'C'*32)
print("PassNTLMv2 "+'A'*32+" # Only for user 'FAKE_USER', domain 'FAKE_DOMAIN'")
print('FAKE_PRIVATE_STDERR',file=sys.stderr)
''')
        self.native.chmod(0o700)

    def build(self,path=None):
        return setup.build_profile(self.identity,path or self.root/'profile',self.ca,
            password_reader=lambda:b'FAKE_PASSWORD',native=str(self.native),probe=False)

    def test_real_pty_hash_capture_and_private_profile(self):
        output,error=io.StringIO(),io.StringIO()
        with redirect_stdout(output),redirect_stderr(error):
            result=self.build()
        self.assertEqual(output.getvalue()+error.getvalue(),'')
        target=Path(result['directory'])
        self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o700)
        self.assertEqual({p.name for p in target.iterdir()},setup.FILES)
        for p in target.iterdir(): self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o600)
        text=(target/'cntlm-3132.conf').read_text()
        self.assertIn('PassNTLMv2 '+'A'*32,text)
        self.assertIn('Listen 127.0.0.1:3132',text)
        self.assertIn('Proxy proxyuk.huawei.com:8080',text)
        self.assertNotIn('PassNT ',text)
        self.assertNotIn('PassLM ',text)
        self.assertNotIn('FAKE_PASSWORD',text)
        self.assertNotIn('FAKE_PRIVATE_STDERR',text)
        self.assertTrue(result['strict_tls'])
        self.assertFalse(result['existing_proxy_changed'])

    def test_verify_only_needs_no_live3129_no_password_and_no_write(self):
        result=self.build(); path=Path(result['directory'])
        before={p.name:p.read_bytes() for p in path.iterdir()}
        with mock.patch.object(setup,'working_profile',side_effect=AssertionError()),mock.patch.object(setup,'native_hash',side_effect=AssertionError()),mock.patch.object(setup,'private_write',side_effect=AssertionError()):
            setup.verify_profile(path)
        self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})

    def test_existing_profile_is_not_overwritten_or_reprompted(self):
        result=self.build(); path=Path(result['directory'])
        reader=mock.Mock(side_effect=AssertionError())
        with self.assertRaisesRegex(setup.SetupError,'ALREADY_EXISTS'):
            setup.build_profile(self.identity,path,self.ca,password_reader=reader,probe=False)
        reader.assert_not_called()

    def test_private_permissions_are_exact_even_with_restrictive_umask(self):
        previous=os.umask(0o777)
        try:
            result=self.build()
        finally:
            os.umask(previous)
        target=Path(result['directory'])
        self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o700)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode)==0o600 for p in target.iterdir()))

    def test_failed_private_write_cleans_only_this_new_profile(self):
        target=self.root/'profile'
        untouched=self.root/'keep'; untouched.write_text('preserve')
        write=setup.private_write
        calls=[]
        def fail_second(path,data):
            calls.append(path)
            if len(calls)==2: raise OSError('synthetic failure')
            write(path,data)
        with mock.patch.object(setup,'private_write',side_effect=fail_second):
            with self.assertRaises(OSError): self.build(target)
        self.assertFalse(target.exists())
        self.assertEqual(untouched.read_text(),'preserve')

    def test_tampering_permissions_extra_files_and_links_are_rejected(self):
        for change in ('bytes','mode','extra','symlink'):
            with self.subTest(change=change):
                path=self.root/change; self.build(path)
                conf=path/'cntlm-3132.conf'
                if change=='bytes': conf.write_bytes(conf.read_bytes()+b'Gateway yes\n')
                if change=='mode': conf.chmod(0o644)
                if change=='extra': (path/'extra').write_text('fixture')
                if change=='symlink':
                    backup=self.root/'linked-source'; backup.write_bytes(conf.read_bytes()); conf.unlink(); conf.symlink_to(backup)
                with self.assertRaises(setup.SetupError): setup.verify_profile(path)

    def test_bad_native_hash_never_echoes_captured_private_output(self):
        self.native.write_text('#!/usr/bin/python3\nimport sys\nprint("FAKE_PASSWORD FAKE_HASH")\nprint("FAKE_PRIVATE_STDERR",file=sys.stderr)\nsys.exit(9)\n')
        out,err=io.StringIO(),io.StringIO()
        with redirect_stdout(out),redirect_stderr(err):
            with self.assertRaisesRegex(setup.SetupError,'NATIVE_HASH_FAILED'):
                setup.native_hash(self.identity,password_reader=lambda:b'FAKE_PASSWORD',native=str(self.native))
        self.assertEqual(out.getvalue()+err.getvalue(),'')
        self.assertFalse((self.root/'profile').exists())

    def test_password_boundaries_and_private_ca_material(self):
        for value in (b'',b'X'*49,b'X\0Y',b'X\nY',b'X\rY'):
            with self.subTest(size=len(value)):
                with self.assertRaises(setup.SetupError):
                    setup.native_hash(self.identity,password_reader=lambda:value,native=str(self.native))
        with self.assertRaises(setup.SetupError):
            setup.build_profile(self.identity,self.root/'badca',b'PRIVATE KEY',password_reader=lambda:b'FAKE_PASSWORD',probe=False)
        self.assertFalse((self.root/'badca').exists())

    def test_terminal_echo_is_restored_even_when_password_read_is_interrupted(self):
        master,slave=pty.openpty()
        self.addCleanup(os.close,master); self.addCleanup(os.close,slave)
        before=termios.tcgetattr(slave)
        class Terminal:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def fileno(self): return slave
            def write(self,value): return len(value)
            def flush(self): pass
            def readline(self):
                assert not termios.tcgetattr(slave)[3] & termios.ECHO
                raise KeyboardInterrupt()
        with mock.patch('builtins.open',return_value=Terminal()):
            with self.assertRaises(KeyboardInterrupt): setup.read_password()
        self.assertEqual(termios.tcgetattr(slave),before)

    def test_working_argv_identity_is_consumed_privately_and_de_is_refused(self):
        argv=['cntlm','-c','/dev/null','-f','-v','-I','-l','127.0.0.1:3129',
            '-u','FAKE_USER@FIRST_DOMAIN','-d','FAKE_DOMAIN','-a','NTLMv2','proxyuk.huawei.com:8080']
        def read(args,**kwargs):
            if 'ss' in args: return 'LISTEN 0 5 127.0.0.1:3129 0.0.0.0:* users:(("cntlm",pid=12,fd=5))'
            if 'readlink' in args: return '/usr/sbin/cntlm\n'
            return ('\0'.join(argv)+'\0').encode()
        with mock.patch.object(setup,'run',side_effect=read):
            self.assertEqual(setup.working_profile(),self.identity)
            argv[-1]='proxyde.huawei.com:8080'
            with self.assertRaises(setup.SetupError): setup.working_profile()

    def test_wrong_account_and_unknown_arguments_refuse_before_private_reads(self):
        for args in (['--prepare'],['--FAKE_PRIVATE_VALUE']):
            out,err=io.StringIO(),io.StringIO()
            with mock.patch.object(setup,'account_home',side_effect=setup.SetupError('NEW_RONG_ACCOUNT_REQUIRED')),mock.patch.object(setup,'working_profile',side_effect=AssertionError()),redirect_stdout(out),redirect_stderr(err):
                self.assertEqual(setup.main(args),1)
            self.assertNotIn('FAKE_PRIVATE_VALUE',out.getvalue()+err.getvalue())


class ServiceTransitionTests(unittest.TestCase):
    """Exercise real Bash cleanup/config restore with fake system interfaces."""
    def fixture(self,mode,active=False,enabled=False):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root=Path(temporary.name); home=root/'home'; repo=root/'repo'; commands=root/'bin'
        for p in (home,repo,commands,root/'units'): p.mkdir()
        (repo/'.git').mkdir()
        config=repo/'.git/config'; config.write_text('[remote "origin"]\n url = https://github.com/Charlie-Su-55/03_PRECODER.git\n proxy = http://127.0.0.1:3129\n')
        original=config.read_bytes()
        profile=home/'precoder_proxy_uk_v1'; profile.mkdir()
        conda=home/'miniforge3/etc/profile.d'; conda.mkdir(parents=True)
        envroot=root/'env'; (envroot/'bin').mkdir(parents=True)
        (conda/'conda.sh').write_text('conda() { :; }\nexport CONDA_PREFIX='+str(envroot)+'\n')
        state=root/'state.json'; state.write_text(json.dumps(dict(active=active,enabled=enabled,sets=0)))
        driver=commands/'driver'
        driver.write_text('''#!/usr/bin/python3
import json,os,shutil,subprocess,sys
from pathlib import Path
p=Path(os.environ['UK_TEST_STATE']); s=json.loads(p.read_text()); a=sys.argv[1:]; name=Path(sys.argv[0]).name
def save(): p.write_text(json.dumps(s))
if name=='id': print('rong'); raise SystemExit()
if name=='git':
 if a[:3]==['remote','get-url','origin']: print('https://github.com/Charlie-Su-55/03_PRECODER.git'); raise SystemExit()
 if 'ls-remote' in a:
  if os.environ['UK_TEST_MODE']=='readfail': raise SystemExit(1)
  print('a'*40+'\\trefs/heads/main'); raise SystemExit()
 if a[:2]==['rev-parse','--git-path']: print(os.environ['UK_TEST_CONFIG']); raise SystemExit()
 if a[:2]==['config','--local']:
  s['sets']+=1; save()
  if os.environ['UK_TEST_MODE']=='gitfail' and s['sets']==3: raise SystemExit(1)
  raise SystemExit(subprocess.run([os.environ['UK_REAL_GIT'],'config','--file',os.environ['UK_TEST_CONFIG'],*a[2:]]).returncode)
 raise SystemExit(2)
if name=='sudo':
 if a==['-v']: raise SystemExit()
 if a and a[0]=='-n': a=a[1:]
 if a[0]=='systemctl':
  op=a[1]
  if op=='show': print(''); raise SystemExit()
  if op=='is-active': raise SystemExit(0 if s['active'] else 1)
  if op=='is-enabled': raise SystemExit(0 if s['enabled'] else 1)
  if op in ('start','stop'): s['active']=op=='start'
  if op in ('enable','disable'): s['enabled']=op=='enable'
  save(); raise SystemExit()
 if a[0]=='install': shutil.copyfile(a[-2],a[-1]); raise SystemExit()
 raise SystemExit(subprocess.run(a).returncode)
raise SystemExit(3)
'''); driver.chmod(0o700)
        for name in ('id','sudo','git'): (commands/name).symlink_to(driver)
        python=envroot/'bin/python'
        python.write_text('''#!/usr/bin/python3
import subprocess,sys
a=sys.argv[1:]
if 'tools/setup_uk_proxy.py' in a: raise SystemExit()
source=sys.stdin.read()
if 'uk_proxy_restore_' in source:
 raise SystemExit(subprocess.run([sys.executable,*a],input=source,text=True).returncode)
raise SystemExit()
'''); python.chmod(0o700)
        script=(ROOT/'tools/start_uk_proxy_service.sh').read_text()
        script=script.replace('/home/rong',str(home)).replace('/home/r50037942/project/03_PRECODER',str(repo)).replace('/etc/systemd/system/',str(root/'units')+'/')
        path=root/'start.sh'; path.write_text(script)
        env=dict(os.environ,HOME=str(home),PATH=str(commands)+':'+os.environ['PATH'],
            UK_TEST_STATE=str(state),UK_TEST_MODE=mode,UK_TEST_CONFIG=str(config),UK_REAL_GIT=shutil.which('git'))
        result=subprocess.run(['bash',str(path)],env=env,capture_output=True,text=True,timeout=15)
        return result,json.loads(state.read_text()),config.read_bytes(),original

    def test_failed_read_stops_only_new_service_and_keeps_git(self):
        result,state,after,before=self.fixture('readfail')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(state['active']); self.assertFalse(state['enabled'])
        self.assertEqual(after,before)

    def test_git_failure_restores_exact_bytes_and_service_enable_state(self):
        result,state,after,before=self.fixture('gitfail')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(after,before)
        self.assertFalse(state['active']); self.assertFalse(state['enabled'])

    def test_preexisting_active_service_is_not_stopped_on_failed_read(self):
        result,state,after,before=self.fixture('readfail',active=True,enabled=True)
        self.assertNotEqual(result.returncode,0)
        self.assertTrue(state['active']); self.assertTrue(state['enabled'])
        self.assertEqual(after,before)

    def test_success_enables_uk_service_and_selects_repo_proxy(self):
        result,state,after,before=self.fixture('success')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(state['active']); self.assertTrue(state['enabled'])
        self.assertIn(b'127.0.0.1:3132',after)
        self.assertIn('Read access is verified',result.stdout)


if __name__=='__main__':
    unittest.main()
