"""Pinned rate-terminal bridge and paired inputs for the residual-W experiment."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile

from configs import twc_v1_k8_w_residual_config as cfg
import icc_alpha_runtime as pinned
import train_twc_v1_k8_icc_rate as rate

ROOT, OUTPUT_ROOT = cfg.ROOT, cfg.OUTPUT_ROOT
now, sha256, read_json = rate.now, rate.sha256, rate.read_json
state_digest, capture_rng, restore_rng = rate.state_digest, rate.capture_rng, rate.restore_rng
configure_cuda, environment = rate.configure_cuda, rate.environment
helpers = rate.helpers


def source_identity():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,
        text=True).strip(),sha256={name:sha256(ROOT/name) for name in cfg.CORE_FILES})


def core_identity():
    return dict(schema=cfg.NAMESPACE+'_core_v1',sha256=source_identity()['sha256'])


def guard_core(expected):
    rate.guard_core(cfg.SOURCE_CORE)
    if pinned.core_identity()!=cfg.LEGACY_CORE:
        raise ValueError('Pinned legacy alpha core changed')
    if core_identity()!=expected:
        raise ValueError('Residual-W experiment core changed')


def inside(path):
    path=Path(path);target=path.resolve();allowed=OUTPUT_ROOT.resolve()
    if target==allowed or not target.is_relative_to(allowed):
        raise ValueError('Owned residual-W child path required')
    for item in (path,*path.parents):
        if item.is_symlink():raise ValueError('Linked residual-W output paths are forbidden')
        if item.resolve()==ROOT:break
    return target


def atomic_json(path,value,*,exclusive=False):
    path=inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(cfg.plain(value),indent=2,sort_keys=True,allow_nan=False)+'\n'
    fd,name=tempfile.mkstemp(prefix='record_',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        if exclusive:
            # The fully written record appears once; an existing identity is never replaced.
            os.link(name,path)
        else:os.replace(name,path)
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


def model_module():
    from models import twc_v1_k8_w_residual as model
    return model


def load_initial(arm,method,device='cpu'):
    if arm not in cfg.ARMS or method not in cfg.TRAIN_METHODS:
        raise ValueError('Declared residual-W arm and training method required')
    codec,stats,ce,teacher=pinned.load_model('rate',arm,device)
    pinned.validate_model_record(teacher,'rate',arm)
    teacher=dict(teacher,base_decoder_state_sha256=state_digest(codec.decoder))
    codec.decoder=model_module().PrecodingDecoder(codec.decoder,method,head_seed=cfg.HEAD_SEED).to(device)
    if state_digest(codec.decoder.base_decoder)!=teacher['base_decoder_state_sha256']:
        raise ValueError('Wrapping changed the initial BS decoder')
    codec.eval().requires_grad_(False);stats.eval().requires_grad_(False);ce.eval().requires_grad_(False)
    verify_files(teacher)
    return codec,stats,ce,teacher


def make_data(device):
    source=rate.original.load_local('utils/twc_v1_k8_replayable_data.py','_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4,codec_train=4,ce_calibration=4,validation=4,test=1),
        device=str(device),cdl_model='C',speed=1.,seed_namespace=cfg.DATA_NAMESPACE)


def batch_random(role,index,device,*,gamma=None):
    import torch
    if role not in cfg.ROLE_RANGES:raise ValueError('Declared residual-W random role required')
    if gamma is None:
        generator=torch.Generator(device='cpu').manual_seed(cfg.noise_seed(role,index,'gamma'))
        gamma=(25*torch.rand(cfg.MICRO_BATCH,generator=generator)).to(device)
    else:
        gamma=torch.as_tensor(gamma,device=device,dtype=torch.float32)
        if gamma.ndim==0:gamma=gamma.expand(cfg.MICRO_BATCH)
    generator=torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'noise'))
    noise=torch.view_as_complex(torch.randn(4,32,256,2,device=device,generator=generator).contiguous())/2**.5
    generator=torch.Generator(device=device).manual_seed(cfg.noise_seed(role,index,'polarity'))
    polarity=(2*torch.randint(2,(4,8),device=device,generator=generator)-1).float()
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
    return dict(result,waveform=wave,soft=recovered['soft'],reliability=recovered['reliability'],
        H_ul_est=estimated,gamma_db=gamma,input_hashes=dict(
            standard_noise=state_digest({'noise':noise}),pilot_polarity=state_digest({'polarity':polarity}),
            gamma=state_digest({'gamma':gamma})))


def training_pairing_evidence():
    """Audit actual inputs retained at every complete checkpoint boundary."""
    expected_steps=set(range(cfg.SAVE_EVERY,cfg.UPDATES+1,cfg.SAVE_EVERY))
    metadata=make_data('cuda:0');traces={};files={}
    required={'micro_index','address','channel','standard_noise','pilot_polarity','gamma'}
    for arm in cfg.ARMS:
        for method in cfg.TRAIN_METHODS:
            key=cfg.key(arm,method)
            path=inside(OUTPUT_ROOT/'training'/key/'progress.jsonl')
            if not path.is_file():raise ValueError('Training pairing trace missing: '+key)
            index={}
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);step=row.get('step');values=row.get('input_fingerprints')
                    if type(step)is not int or step not in expected_steps or not isinstance(values,list) or len(values)!=cfg.ACCUMULATION:
                        raise ValueError('Invalid retained training boundary: '+key)
                    for micro,value in enumerate(values):
                        micro_index=cfg.micro_index(step,micro)
                        if (set(value)!=required or value['micro_index']!=micro_index
                                or value['address']!=metadata.address('codec_train',micro_index)):
                            raise ValueError('Training input address/schema changed: '+key)
                        for name in ('channel','standard_noise','pilot_polarity','gamma'):
                            digest=value[name]
                            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                                raise ValueError('Invalid actual training tensor hash: '+name)
                    if step in index and index[step]!=values:
                        raise ValueError('Repeated training boundary used different inputs: '+key)
                    index[step]=values
            if set(index)!=expected_steps:raise ValueError('Training trace boundary coverage incomplete: '+key)
            traces[key]=index;files[str(path)]=sha256(path)
    reference=traces[cfg.key(cfg.ARMS[0],cfg.TRAIN_METHODS[0])]
    if any(trace!=reference for trace in traces.values()):
        raise ValueError('Methods/arms did not use identical retained training inputs')
    return dict(status='passed',scope='actual input hashes at every saved checkpoint boundary; unsaved steps are not exported',
        jobs=list(traces),boundaries_per_job=len(expected_steps),
        microbatches_checked_per_job=len(expected_steps)*cfg.ACCUMULATION,
        data_protocol=metadata.protocol,files=files,test_used=False)


def load_model(policy,arm,device='cpu'):
    if policy not in cfg.POLICIES:raise ValueError('Declared evaluation policy required')
    method='rzf_continue' if policy=='rate_source' else policy
    codec,stats,ce,teacher=load_initial(arm,method,device)
    if policy=='rate_source':
        record=dict(policy=policy,arm=arm,step=0,source_identity=teacher,
            state_sha256=state_digest(codec.decoder),frozen_files=teacher['frozen_files'])
    else:
        import torch
        import train_twc_v1_k8_w_residual as train
        directory=inside(OUTPUT_ROOT/'training'/cfg.key(arm,method))
        path=directory/'checkpoint_latest.pt';manifest=read_json(directory/'manifest.json')
        if path.is_symlink() or manifest.get('checkpoint_sha256')!=sha256(path):
            raise ValueError('Residual-W terminal checkpoint bytes changed')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        frozen=helpers().frozen_state(codec,ce,stats)
        train.verify_saved(directory,payload,manifest,arm=arm,method=method,data=make_data(device),
            core=core_identity(),source=teacher,frozen=frozen,require_final=True)
        rate.original.strict_load(codec.decoder,payload['model'])
        helpers().assert_frozen_state(codec,ce,stats,frozen)
        files=list(teacher['frozen_files'])
        for name in ('checkpoint_latest.pt','terminal_receipt.json','manifest.json','initialization.json'):
            files.append(dict(path=str(directory/name),sha256=sha256(directory/name)))
        record=dict(policy=policy,arm=arm,step=payload['step'],source_identity=teacher,
            state_sha256=state_digest(codec.decoder),core=payload['core'],frozen_files=files)
    codec.eval().requires_grad_(False);stats.eval().requires_grad_(False);ce.eval().requires_grad_(False)
    verify_files(record)
    return codec,stats,ce,record


def preflight_sources(output_dir,*,require_cuda=True):
    import gc
    import torch
    from evaluate_twc_v1_k8_icc_aligned import frozen_evaluation,checked_input_hashes,_tensor_hash
    output=inside(output_dir);subtree=(OUTPUT_ROOT/'preflight').resolve()
    if output==subtree or not output.is_relative_to(subtree):
        raise ValueError('Preflight must be a proper child of the own preflight tree')
    core=core_identity();guard_core(core)
    device=configure_cuda() if require_cuda else torch.device('cpu')
    records={};checks=[];fingerprints={};protocol=None
    for arm in cfg.ARMS:
        codec,stats,ce,teacher=load_initial(arm,'w_residual',device)
        data=make_data(device)
        if protocol is not None and protocol!=data.protocol:
            raise ValueError('Warmstart arms changed the data protocol')
        protocol=data.protocol
        frozen=helpers().frozen_state(codec,ce,stats)
        sample=data.sample('validation',cfg.ROLE_RANGES['monitor'][0])
        if sample['address']!=data.address('validation',cfg.ROLE_RANGES['monitor'][0]):
            raise ValueError('Warmstart sample address changed')
        with frozen_evaluation({arm:(codec,stats)},ce):
            for gi,gamma in enumerate(cfg.GAMMAS):
                out=forward_codec(codec,stats,ce,sample,gamma,('monitor',0,gi))
                if bool((out['delta_W']!=0).any()):raise ValueError('Residual initial projection is not zero')
                error=float((out['W']-out['W_base']).abs().max())
                if error>2e-6:raise ValueError('Zero residual changed the initial RZF precoder')
                module=model_module()
                _,actual=module.precoding_objective(out['W'],sample['H_dl'],downlink_snr_db=cfg.DL_SNR)
                _,base=module.precoding_objective(out['W_base'],sample['H_dl'],downlink_snr_db=cfg.DL_SNR)
                rate_error=float((actual['sum_rate_bps_hz']-base['sum_rate_bps_hz']).abs().max())
                if rate_error>1e-4:raise ValueError('Zero residual changed initial sum rates')
                checks.append(dict(arm=arm,gamma_db=gamma,max_precoder_error=error,max_rate_error=rate_error))
                fingerprint=checked_input_hashes(out['input_hashes'],H_dl=_tensor_hash(sample['H_dl']),H_ul=_tensor_hash(sample['H_ul']))
                if gamma in fingerprints and fingerprints[gamma]!=fingerprint:
                    raise ValueError('Warmstart arms did not use identical actual inputs')
                fingerprints[gamma]=fingerprint
        helpers().assert_frozen_state(codec,ce,stats,frozen)
        if state_digest(codec.decoder.base_decoder)!=teacher['base_decoder_state_sha256']:
            raise ValueError('Warmstart changed the source BS decoder')
        verify_files(teacher)
        records[arm]=dict(teacher=teacher,wrapper=codec.decoder.metadata())
        del codec,stats,ce,data,sample,out,actual,base
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    guard_core(core)
    output.mkdir(parents=True,exist_ok=False)
    evidence=dict(source_records=records,warmstart_checks=checks,data_protocol=protocol,
        core=core,recipe_fingerprint=cfg.study_fingerprint(),optimizer_updates=0,test_used=False,
        paired_input_fingerprints=fingerprints,
        environment=environment())
    atomic_json(output/'source_checks.json',evidence,exclusive=True)
    manifest=dict(status='passed',phase='preflight',core=core,
        recipe_fingerprint=cfg.study_fingerprint(),files={'source_checks.json':sha256(output/'source_checks.json')},
        data_protocol=protocol,source_records=records,warmstart_checks=checks,
        optimizer_updates=0,test_used=False,finished=now())
    atomic_json(output/'manifest.json',manifest,exclusive=True)
    return manifest
