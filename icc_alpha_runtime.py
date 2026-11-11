"""Read-only model bridge and immutable calibration lock for alpha inference."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile

from configs import twc_v1_k8_icc_alpha_config as cfg
import train_twc_v1_k8_icc_rate as rate

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
now = rate.now
sha256 = rate.sha256
read_json = rate.read_json
configure_cuda = rate.configure_cuda
environment = rate.environment
state_digest = rate.state_digest


def source_identity():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,
        text=True).strip(),sha256={name:sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1',sha256=source_identity()['sha256'])


def guard_core(expected):
    rate.guard_core(cfg.RATE_CORE)
    if core_identity()!=expected:
        raise ValueError('Alpha experiment source changed')


def inside(path):
    path=Path(path); resolved=path.resolve(); allowed=OUTPUT_ROOT.resolve()
    if resolved==allowed or not resolved.is_relative_to(allowed):
        raise ValueError('Owned alpha experiment child path required')
    for item in (path,*path.parents):
        if item.is_symlink():raise ValueError('Linked alpha output paths are forbidden')
        if item.resolve()==ROOT:break
    return resolved


def atomic_json(path,value,*,exclusive=False):
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(cfg.plain(value),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        return
    fd,name=tempfile.mkstemp(prefix='record_',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


@contextmanager
def process_lock(path):
    import fcntl
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def verify_files(record):
    rate.verify_files(record)


def load_model(kind,arm,device='cpu'):
    if kind not in ('source','rate') or arm not in cfg.ARMS:
        raise ValueError('Declared source/rate model and physical arm required')
    rate.guard_core(cfg.RATE_CORE)
    if kind=='rate':
        directory=cfg.RATE_ROOT/'training'/arm
        for path,digest in ((directory/'checkpoint_latest.pt',cfg.RATE_CHECKPOINT_SHA256[arm]),
                            (directory/'terminal_receipt.json',cfg.RATE_TERMINAL_SHA256[arm])):
            if path.is_symlink() or not path.is_file() or sha256(path)!=digest:
                raise ValueError('Pinned rate terminal missing/changed: '+str(path))
        terminal=read_json(directory/'terminal_receipt.json')
        if (terminal.get('training_code',{}).get('commit')!=cfg.RATE_COMMIT
                or terminal.get('core')!=cfg.RATE_CORE or terminal.get('step')!=5000
                or terminal.get('status')!='completed'):
            raise ValueError('Pinned rate training provenance changed')
        codec,stats,ce,record=rate.load_finetuned(arm,device)
        if record['sha256']!=cfg.RATE_CHECKPOINT_SHA256[arm] or record['step']!=5000:
            raise ValueError('Fixed rate source identity changed')
        files=list(record['frozen_files'])
        for name in ('manifest.json','initialization.json'):
            path=directory/name
            files.append(dict(path=str(path),sha256=sha256(path)))
        checkpoint=record['sha256']; model_core=record['core']; step=record['step']
    else:
        codec,stats,ce,record=rate.load_source(arm,device)
        files=list(record['frozen_files']); checkpoint=record['checkpoint_sha256']
        model_core=record['core']; step=20000
    codec.eval().requires_grad_(False);ce.eval().requires_grad_(False);stats.eval().requires_grad_(False)
    result=dict(kind=kind,arm=arm,model_commit=cfg.RATE_COMMIT if kind=='rate' else cfg.SOURCE_COMMIT,
        checkpoint_sha256=checkpoint,core=model_core,step=step,
        state_sha256=state_digest(codec),CE_state_sha256=state_digest(ce),
        statistics_state_sha256=state_digest(stats),frozen_files=files)
    validate_model_record(result,kind,arm)
    return codec,stats,ce,result


def validate_model_record(record,kind,arm):
    expected=dict(kind=kind,arm=arm,model_commit=cfg.RATE_COMMIT if kind=='rate' else cfg.SOURCE_COMMIT,
        checkpoint_sha256=(cfg.RATE_CHECKPOINT_SHA256 if kind=='rate' else cfg.SOURCE_CHECKPOINT_SHA256)[arm],
        core=cfg.RATE_CORE if kind=='rate' else cfg.SOURCE_CORE,step=5000 if kind=='rate' else 20000)
    for key,value in expected.items():
        if record.get(key)!=value:raise ValueError('Pinned model record changed: '+key)
    verify_files(record)


def make_data(device):
    source=rate.original.load_local('utils/twc_v1_k8_replayable_data.py','_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4,codec_train=4,ce_calibration=4,validation=4,test=1),
        device=str(device),cdl_model='C',speed=1.,seed_namespace=cfg.DATA_NAMESPACE)


def batch_random(role,index,device,*,gamma):
    import torch
    if role not in cfg.ROLE_RANGES:raise ValueError('Only alpha calibration/final noise roles allowed')
    gamma=torch.as_tensor(gamma,device=device,dtype=torch.float32)
    if gamma.ndim==0:gamma=gamma.expand(cfg.MICRO_BATCH)
    gen=torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'noise'))
    noise=torch.view_as_complex(torch.randn(4,32,256,2,device=device,generator=gen).contiguous())/2**.5
    gen=torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'polarity'))
    polarity=(2*torch.randint(2,(4,8),device=device,generator=gen)-1).float()
    return gamma,noise,polarity


def forward_codec(codec,stats,ce,sample,gamma_db,noise_key):
    import torch
    role,batch,gi=noise_key
    phy=rate.original.physics()
    with torch.no_grad():
        gamma,noise,polarity=batch_random(role,batch*10+gi,sample['H_ul'].device,gamma=gamma_db)
        raw=codec.encoder(sample['H_dl'])
        wave=phy.build_waveform(raw,stats.layout,polarity=polarity)
        Y=phy.transmit(sample['H_ul'],wave.x_grid,noise,gamma)
        initial=phy.fourier_ls(Y,wave.pilot)
        estimated=ce(Y,wave.pilot,initial,gamma_db=gamma)
        cevar=stats.ce_variance(gamma)
        recovered=phy.recover(Y,estimated,stats.layout,stats.nominal_power(),cevar,gamma)
        result=codec.decoder(Y_rx=Y,H_ul_est=estimated,soft=recovered['soft'],
            reliability=recovered['reliability'],pilots=wave.pilot,gamma_db=gamma,
            layout=stats.layout,prior=stats.nominal_power(),ce_error=cevar)
        if torch.is_tensor(result):result=dict(H_hat=result)
        return dict(result,waveform=wave,soft=recovered['soft'],H_ul_est=estimated,
            reliability=recovered['reliability'],gamma_db=gamma,
            input_hashes=dict(standard_noise=state_digest({'noise':noise}),
                pilot_polarity=state_digest({'polarity':polarity})))


def _check_calibration_fingerprints(path,protocol):
    entries=read_json(path);index={}
    metadata=make_data(protocol['execution']['device'])
    expected={(arm,bi,float(g)) for arm in cfg.ARMS
              for bi in range(cfg.CALIBRATION_BATCHES) for g in cfg.GAMMAS}
    for row in entries:
        key=(row['arm'],row['batch_index'],float(row['gamma_db']))
        if key not in expected or key in index:raise ValueError('Calibration fingerprint identity changed')
        if row.get('address')!=metadata.address('validation',cfg.ROLE_RANGES['calibration'][0]+row['batch_index']):
            raise ValueError('Calibration must use only its declared validation addresses')
        value=row['fingerprint']
        if set(value)!={'H_dl','H_ul','standard_noise','pilot_polarity'}:
            raise ValueError('Complete actual calibration input hashes required')
        if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
               for v in value.values()):raise ValueError('Invalid calibration hash')
        index[key]=value
    if set(index)!=expected:raise ValueError('Calibration actual input coverage incomplete')
    for bi in range(cfg.CALIBRATION_BATCHES):
        for g in cfg.GAMMAS:
            if index[cfg.ARMS[0],bi,float(g)]!=index[cfg.ARMS[1],bi,float(g)]:
                raise ValueError('Calibration arms did not use identical inputs')


def calibration_evidence(directory):
    """Recalculate the choice using only sealed calibration evidence."""
    import twc_v1_k8_icc_alpha_results as results
    directory=inside(directory)
    if not directory.is_relative_to((OUTPUT_ROOT/'calibration').resolve()):
        raise ValueError('Selection evidence must be in the calibration tree')
    manifest=read_json(directory/'manifest.json')
    expected=dict(status='passed',phase='calibration',core=core_identity(),
        recipe_fingerprint=cfg.fingerprint(),alpha_grid=list(cfg.ALPHA_GRID),gamma_db=list(cfg.GAMMAS),
        batch_size=cfg.MICRO_BATCH,n_batches=cfg.CALIBRATION_BATCHES,
        samples_per_condition=cfg.CALIBRATION_BATCHES*cfg.MICRO_BATCH)
    for key,value in expected.items():
        if manifest.get(key)!=value:raise ValueError('Calibration identity changed: '+key)
    protocol=manifest['data_protocol']
    device=protocol.get('execution',{}).get('device')
    if protocol!=make_data(device).protocol:raise ValueError('Calibration namespace/data protocol changed')
    required={'samples.csv.gz','scores.csv','input_fingerprints.json','selection_candidate.json'}
    files=manifest.get('files',{})
    if not required<=set(files):raise ValueError('Calibration selection evidence missing')
    evidence={str(directory/'manifest.json'):sha256(directory/'manifest.json')}
    for name,digest in files.items():
        if Path(name).name!=name:raise ValueError('Calibration filename escapes owned directory')
        path=inside(directory/name)
        if not path.is_file() or sha256(path)!=digest:raise ValueError('Calibration evidence changed: '+name)
        evidence[str(path)]=digest
    records=manifest.get('source_records',{})
    if set(records)!=set(cfg.ARMS):raise ValueError('Both source model identities required for calibration')
    for arm,record in records.items():validate_model_record(record,'source',arm)
    _check_calibration_fingerprints(directory/'input_fingerprints.json',protocol)
    with gzip.open(directory/'samples.csv.gz','rt',encoding='utf-8',newline='') as stream:
        rows=[]
        for row in csv.DictReader(stream):
            rows.append({key:(value if key=='arm' else int(value) if key in
                ('sample_index','batch_index','within_batch') else float(value)) for key,value in row.items()})
    recomputed=results.select_alphas(rows,samples=cfg.CALIBRATION_BATCHES*cfg.MICRO_BATCH,
                                     batch_size=cfg.MICRO_BATCH)
    with (directory/'scores.csv').open(encoding='utf-8',newline='') as stream:
        scores=[{key:(value if key=='arm' else None if value=='' else int(value)
                     if key in ('n_samples','n_batches','batch_size') else float(value))
                 for key,value in row.items()} for row in csv.DictReader(stream)]
    if cfg.plain(scores)!=cfg.plain(recomputed['scores']):
        raise ValueError('Calibration score table does not match its raw samples')
    candidate=read_json(directory/'selection_candidate.json')
    for key,value in dict(core=core_identity(),data_protocol=protocol,
                          recipe_fingerprint=cfg.fingerprint()).items():
        if candidate.get(key)!=value:raise ValueError('Calibration candidate identity changed: '+key)
    for key,value in recomputed.items():
        if cfg.plain(candidate.get(key))!=cfg.plain(value):
            raise ValueError('Calibration candidate was not selected from declared samples: '+key)
    return dict(directory=str(directory),files=evidence,selected=recomputed['selected'],
        scores=recomputed['scores'],data_protocol=protocol,source_records=records)


def final_attempt_exists():
    root=OUTPUT_ROOT/'evaluation'/'final'
    return root.exists() and any(root.iterdir())


def _selection_path(path=None):
    expected=inside(OUTPUT_ROOT/'selection.json')
    if path is not None and inside(path)!=expected:
        raise ValueError('Only the own immutable alpha selection lock is accepted')
    return expected


def load_selection(path=None):
    path=_selection_path(path)
    record=read_json(path)
    if record.get('record_id')!=cfg.digest({k:v for k,v in record.items() if k!='record_id'}):
        raise ValueError('Alpha selection lock hash changed')
    expected=dict(format=cfg.NAMESPACE+'_selection_v1',core=core_identity(),
        recipe_fingerprint=cfg.fingerprint(),status='locked')
    for key,value in expected.items():
        if record.get(key)!=value:raise ValueError('Alpha selection identity changed: '+key)
    evidence=calibration_evidence(record['calibration_dir'])
    for key in ('selected','scores','data_protocol','source_records'):
        if cfg.plain(record.get(key))!=cfg.plain(evidence[key]):
            raise ValueError('Locked alpha/calibration identity changed: '+key)
    if record.get('calibration_files')!=evidence['files']:
        raise ValueError('Locked calibration artifacts changed')
    return record


def lock_selection(candidate_dir):
    """Create once after calibration; never reselect after a final attempt."""
    path=_selection_path()
    if path.exists():
        record=load_selection(path)
        if inside(candidate_dir)!=inside(record['calibration_dir']):
            raise ValueError('A different alpha selection is already locked')
        return record
    if final_attempt_exists():
        raise ValueError('Final evidence exists without a valid selection lock; reselection forbidden')
    guard_core(core_identity())
    evidence=calibration_evidence(candidate_dir)
    record=dict(format=cfg.NAMESPACE+'_selection_v1',status='locked',core=core_identity(),
        recipe_fingerprint=cfg.fingerprint(),selected=evidence['selected'],scores=evidence['scores'],
        calibration_dir=evidence['directory'],calibration_files=evidence['files'],
        data_protocol=evidence['data_protocol'],source_records=evidence['source_records'],
        locked_at=now(),selection='calibration-only global alpha; no final/test selection')
    record['record_id']=cfg.digest(record)
    atomic_json(path,record,exclusive=True)
    return load_selection(path)
