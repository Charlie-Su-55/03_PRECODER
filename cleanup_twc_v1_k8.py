#!/usr/bin/env python3
"""Preview, or explicitly remove, untracked generated caches inside this 03 only."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parent
ORIGIN='https://github.com/Charlie-Su-55/03_PRECODER.git'
SOURCE_DIRS=('configs','models','tests','utils','paper/scripts')
CACHE_NAMES={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}
EXCLUDED={'.git','.venv','venv','env','envs','.tox','.nox','site-packages','overleaf',
    'data','datasets','figures','checkpoints','checkpoint','ckp','logs','terminal_logs',
    'archive','archives','cleanup_reports','manual','results','evaluation','training',
    'batches','outputs','output','locks','old'}
PROTECTED=('.gitignore','paper/plot_specialist_envelope.py',
    'paper/figures/fig_k4_pure_as_feedback_snr.pdf',
    'paper/figures/fig_k4_pure_as_feedback_snr.png',
    'paper/figures/fig_specialist_envelope.pdf','paper/figures/fig_specialist_envelope.png')
CHAIN=('twc_v1_k8_ce','twc_v1_k8_codec_warmup','twc_v1_k8_schedule_screen_v1',
    'twc_v1_k8_ordered_encoder_screen_v2','twc_v1_k8_task_loss_screen_v1',
    'twc_v1_k8_rzf_probe_v1','twc_v1_k8_rzf_matched_training_v1','twc_v1_k8_decoder_screen_v1')
CACHEDIR_SIGNATURE=b'Signature: 8a477f597d28d172789f06886806bc55'

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root).decode('utf-8','surrogateescape').rstrip('\n')

def linked(path):
    info=Path(path).lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)

def safe_path(root,path):
    root=Path(root).resolve();path=Path(path)
    relative=path.relative_to(root)
    cursor=root
    for part in relative.parts:
        cursor=cursor/part
        if cursor.exists() or cursor.is_symlink():
            if linked(cursor):raise ValueError('Link/reparse path is protected: '+str(cursor))
    if not path.resolve().is_relative_to(root):raise ValueError('Path escaped this 03')
    return path

def snapshot(root):
    return dict(protected_sha256={name:sha256(root/name) if (root/name).is_file() else None for name in PROTECTED},
        git_status=git(root,'status','--porcelain=v1','--untracked-files=all'),
        tracked_status=git(root,'status','--porcelain=v1','--untracked-files=no'),
        tracked_deletions=[name for name in git(root,'ls-files','--deleted','-z').split('\0') if name])

def process_guard(root,*,platform_name=None):
    """Uncertain relevant processes cause a conservative all-cache skip."""
    found=[]
    try:
        if (os.name if platform_name is None else platform_name)=='nt':
            command="@((Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(?:[0-9.]*)?(?:w)?\\.exe$' } | Select-Object ProcessId,Name,CommandLine)) | ConvertTo-Json -Compress"
            raw=subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',command],text=True)
            records=json.loads(raw) if raw.strip() else []
            if isinstance(records,dict):records=[records]
            for item in records:
                if item['ProcessId']==os.getpid():continue
                # Windows CIM does not expose cwd. Do not infer it from an exe path.
                found.append(dict(pid=item['ProcessId'],name=item['Name'],
                    reason='active repository Python' if str(root).lower() in (item.get('CommandLine') or '').lower()
                    else 'Python cwd cannot be proved outside this repository'))
        elif Path('/proc').is_dir():
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
                try:
                    if proc.stat().st_uid!=os.getuid():continue
                    name=(proc/'comm').read_text().strip()
                    try:executable=(proc/'exe').resolve().name
                    except OSError:executable=name
                    if not re.fullmatch(r'python[0-9.]*',name) and not re.fullmatch(r'python[0-9.]*',executable):continue
                    cwd=(proc/'cwd').resolve(strict=True)
                    command=(proc/'cmdline').read_bytes().decode(errors='replace')
                    if cwd==root or cwd.is_relative_to(root) or str(root) in command:
                        found.append(dict(pid=int(proc.name),name=name,reason='active repository Python'))
                except FileNotFoundError:continue
                except (PermissionError,OSError) as error:
                    found.append(dict(pid=int(proc.name),reason='process inspection uncertain: '+type(error).__name__))
        else:found.append(dict(reason='No supported reliable process inventory'))
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        found.append(dict(reason='Process inspection unavailable: '+type(error).__name__))
    return dict(clear=not found,blockers=found,self_pid_excluded=os.getpid(),
        conservative_windows_cwd_policy=True)

def generated(path,cache):
    if path.suffix.lower() in ('.pyc','.pyo'):
        try:
            source=(Path(importlib.util.source_from_cache(str(path))) if path.parent.name=='__pycache__'
                    else path.with_suffix('.py'))
        except ValueError:return False
        if source.is_symlink() or linked(source) or not source.is_file():return False
        with path.open('rb') as stream:header=stream.read(16)
        return (len(header)==16 and header[:4]==importlib.util.MAGIC_NUMBER
                and int.from_bytes(header[4:8],'little') in (0,1,3))
    name=path.name.lower()
    if cache=='.pytest_cache':
        if name=='cachedir.tag':return path.read_bytes().startswith(CACHEDIR_SIGNATURE)
        if name not in {'nodeids','lastfailed','stepwise'} or path.parts[-3:-1]!=('v','cache'):return False
        try:value=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,UnicodeError):return False
        return isinstance(value,dict if name=='lastfailed' else list)
    if cache=='.mypy_cache':
        if name=='cachedir.tag':return path.read_bytes().startswith(CACHEDIR_SIGNATURE)
        if name=='cache.db':
            with path.open('rb') as stream:return stream.read(16)==b'SQLite format 3\0'
        if not name.endswith(('.data.json','.meta.json')):return False
        try:return isinstance(json.loads(path.read_text(encoding='utf-8')),dict)
        except (ValueError,UnicodeError):return False
    if cache=='.ruff_cache':
        if name=='cachedir.tag':return path.read_bytes().startswith(CACHEDIR_SIGNATURE)
        if not re.fullmatch(r'[0-9a-f]{4,}',name):return False
        directory=next((p for p in path.parents if p.name=='.ruff_cache'),None)
        marker=directory/'CACHEDIR.TAG' if directory is not None else None
        return (marker is not None and marker.is_file() and not linked(marker)
                and marker.read_bytes().startswith(CACHEDIR_SIGNATURE))
    return False

def allocation(info):
    if hasattr(info,'st_blocks'):return info.st_blocks*512,'st_blocks'
    # Do not claim a logical byte count is the exact NTFS allocation.
    return None,'filesystem allocation unavailable; logical bytes reported separately'

def inventory(root):
    tracked=set(git(root,'ls-files','-z').split('\0'));tracked.discard('')
    candidates=[];directories=[];skips=[]
    def walk(folder,cache=None,root_only=False):
        try:entries=list(os.scandir(safe_path(root,folder)))
        except (OSError,ValueError) as error:skips.append(dict(path=str(folder),reason=str(error)));return
        for entry in entries:
            path=Path(entry.path);relative=path.relative_to(root).as_posix()
            try:
                if linked(path):skips.append(dict(path=relative,reason='link/reparse protected'));continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in EXCLUDED or entry.name.startswith('runs'):continue
                    next_cache=entry.name if entry.name in CACHE_NAMES else cache
                    if root_only and next_cache is None:continue
                    if next_cache:directories.append(path)
                    walk(path,next_cache)
                elif entry.is_file(follow_symlinks=False) and relative not in tracked and generated(path,cache):
                    info=path.stat();allocated,method=allocation(info)
                    candidates.append(dict(path=relative,size_bytes=info.st_size,mtime_ns=info.st_mtime_ns,
                        file_id=[info.st_dev,info.st_ino],hard_links=info.st_nlink,
                        allocated_estimate_bytes=allocated,allocation_method=method))
            except (OSError,ValueError) as error:skips.append(dict(path=relative,reason=str(error)))
    walk(root,root_only=True)
    for name in SOURCE_DIRS:
        folder=root/name
        if folder.is_dir():walk(folder)
    return candidates,directories,tracked,skips

def held_on_windows(path):
    if os.name!='nt':
        if not Path('/proc').is_dir():return True
        expected=path.stat()
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:
                if proc.stat().st_uid!=os.getuid():continue
                for fd in (proc/'fd').iterdir():
                    try:
                        actual=fd.stat()
                        if (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino):return True
                    except FileNotFoundError:continue
            except FileNotFoundError:continue
            except (PermissionError,OSError):return True
        return False
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW;create.restype=wintypes.HANDLE
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE]
    handle=create(str(path),0x80000000,0,None,3,0x00200000,None)
    if handle==ctypes.c_void_p(-1).value:return True
    close(handle);return False

def heavy_candidates(root):
    """Metadata only; these paths are never cleanup targets or hashed weights."""
    result=[]
    for name in CHAIN:
        folder=root/'runs'/name
        if not folder.is_dir() or linked(folder):continue
        logical=0;ids={};count=0;truncated=False
        for current,dirs,files in os.walk(folder,followlinks=False):
            dirs[:]=[d for d in dirs if not linked(Path(current)/d)]
            for filename in files:
                path=Path(current)/filename
                if linked(path):continue
                info=path.stat();logical+=info.st_size;ids.setdefault((info.st_dev,info.st_ino),info.st_size);count+=1
                if count>=100000:truncated=True;break
            if truncated:break
        result.append(dict(path=str(folder),logical_bytes=logical,unique_file_bytes=sum(ids.values()),files=count,
            truncated=truncated,action='retain',classification='current source/recovery or unique historical evidence',
            reason='Protected K8 source chain; size/name alone does not prove a disposable smoke or duplicate',
            reclaimable_bytes_authorized=0,weights_hashed=False))
    return sorted(result,key=lambda item:item['logical_bytes'],reverse=True)

def run_cleanup(root=ROOT,*,apply=False,report_path=None):
    root=Path(root).resolve()
    if Path(git(root,'rev-parse','--show-toplevel')).resolve()!=root:raise ValueError('Exact repository root required')
    if git(root,'branch','--show-current')!='main' or git(root,'remote','get-url','origin')!=ORIGIN:
        raise ValueError('Only the authorized 03 main/origin is allowed')
    directory=safe_path(root,root/'cleanup_reports');directory.mkdir(exist_ok=True)
    target=Path(report_path) if report_path is not None else directory/('cleanup_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]+'.json')
    if not target.is_absolute():target=root/target
    target=safe_path(root,target)
    if target.parent!=directory:raise ValueError('Report must be a new file directly inside cleanup_reports')
    if target.exists():raise FileExistsError('Cleanup report is never overwritten: '+str(target))
    before=snapshot(root);disk_before=shutil.disk_usage(root)._asdict();guard=process_guard(root)
    candidates,dirs,tracked,skips=inventory(root);removed=[];removed_dirs=[]
    if apply and guard['clear']:
        for item in candidates:
            path=safe_path(root,root/item['path'])
            try:
                if item['path'] in set(git(root,'ls-files','-z').split('\0')):raise ValueError('Now Git tracked')
                info=path.lstat()
                if ([info.st_dev,info.st_ino]!=item['file_id'] or info.st_size!=item['size_bytes']
                        or info.st_mtime_ns!=item['mtime_ns'] or linked(path)):raise ValueError('Candidate changed')
                if held_on_windows(path):raise ValueError('Open/held or exclusive-access status uncertain')
                path.unlink();removed.append(item)
            except (OSError,ValueError) as error:skips.append(dict(path=item['path'],reason=str(error)))
        for path in sorted(set(dirs),key=lambda p:len(p.parts),reverse=True):
            relative=path.relative_to(root).as_posix()
            if any(name.startswith(relative+'/') for name in tracked):continue
            try:safe_path(root,path).rmdir();removed_dirs.append(relative)
            except OSError:pass
    elif apply:skips.append(dict(path='all generated caches',reason='Active/uncertain Python processes; no deletion'))
    after=snapshot(root)
    unchanged=(before['protected_sha256']==after['protected_sha256'] and before['tracked_status']==after['tracked_status']
        and before['tracked_deletions']==after['tracked_deletions'])
    disk_after=shutil.disk_usage(root)._asdict();links=Counter(tuple(v['file_id']) for v in removed);unique={tuple(v['file_id']):v for v in removed}
    reclaimable=[v['allocated_estimate_bytes'] for key,v in unique.items() if links[key]==v['hard_links']]
    reclaimed=sum(reclaimable) if all(v is not None for v in reclaimable) else None
    try:heavy=heavy_candidates(root)
    except OSError as error:
        heavy=[dict(action='retain',classification='inventory uncertain',reason=type(error).__name__+': '+str(error),weights_hashed=False)]
    report=dict(format='twc_v1_k8_safe_cleanup_v1',status='completed' if unchanged else 'failed',created=datetime.now(timezone.utc).isoformat(),
        machine=platform.node(),platform=platform.platform(),repository=str(root),mode='apply' if apply else 'preview',
        process_guard=guard,source_directory_whitelist=['root immediate bytecode/tool caches',*SOURCE_DIRS],
        before=before,after=after,protected_and_tracked_state_unchanged=unchanged,candidates=candidates,deleted=removed,
        removed_directories=removed_dirs,skipped=skips,
        disk_before=disk_before,disk_after=disk_after,observed_free_delta_bytes=disk_after['free']-disk_before['free'],
        logical_deleted_bytes=sum(v['size_bytes'] for v in removed),unique_unlinked_file_bytes=sum(v['size_bytes'] for v in unique.values()),
        reclaimable_allocation_estimate_bytes=reclaimed,
        space_note='Hard links counted once and only when every link was removed; free-space delta includes concurrent system writes, not proof of exact attributable recovery.',
        heavy_candidates=heavy,only_cache_files_deleted=True,checkpoints_outputs_archives_locks_untouched=True)
    report['report_path']=str(target)
    if not unchanged:report['error']='Protected files or tracked Git state changed during cleanup'
    with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    if not unchanged:raise RuntimeError(report['error']+'; evidence: '+str(target))
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--apply',action='store_true');modes.add_argument('--preview',action='store_true')
    parser.add_argument('--report',type=Path);args=parser.parse_args(argv)
    report=run_cleanup(apply=args.apply,report_path=args.report)
    print(json.dumps({name:report[name] for name in ('mode','repository','report_path','logical_deleted_bytes',
        'reclaimable_allocation_estimate_bytes','observed_free_delta_bytes','protected_and_tracked_state_unchanged')},indent=2))
    print('deleted_files='+str(len(report['deleted']))+'; process_clear='+str(report['process_guard']['clear']))
    return 0

if __name__=='__main__':raise SystemExit(main())
