#!/usr/bin/env python3
"""Independent Joint+Swin256 arms; reuse the accepted trainer and strict guards."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from configs.joint_swin_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT, REPO_ROOT, joint_swin_config, validate_config,
    optimizer_group_spec, make_channel_data, channel_data_api,
)
from preflight_proposal_as_direct import (
    DEFAULT_MANIFEST, _backend_info, sha256_file, validate_checkpoint_metadata,
    verify_checkpoint_hash,
)
from train_proposal_as_direct import _safe_path, validate_model_state
from train_proposal_as_receiver import (
    fingerprint, _json, validate_output_paths as _output_paths,
    validate_best_metadata as _best_metadata, run_lock,
)
import train_proposal_as_spatial as _group_tools

SOURCE_FILES = (
    'train_proposal_joint_swin.py','configs/joint_swin_train_config.py',
    'models/adaptive_hybrid_joint_swin.py','models/adaptive_hybrid.py',
    'models/adaptive_hybrid_as_receiver.py','models/adaptive_hybrid_as_spatial.py',
    'models/swin_precoder.py','models/baseline_common.py',
    'main_train.py','configs/train_config.py','utils/replayable_channel_data.py',
    'utils/sionna_rng.py','utils/channel_generator.py','train_proposal_as_spatial.py',
    'configs/as_spatial_train_config.py','train_proposal_as_receiver.py',
    'configs/as_receiver_train_config.py','train_proposal_as_direct.py','preflight_proposal_as_direct.py',
)


class _JointGroupPolicy:
    """Read-only adapter for existing joint UE/BS *group policy* helpers.

    Their sole legacy arm branch selects frozen-versus-joint parameter groups.
    Both new arms are joint. Only that local policy reads ``arm='joint'``;
    serialization, fingerprints, checkpoint identities, paths and model configs
    remain the untouched real config. This view is NEVER given to the model or
    trainer and is never serialized as a replacement experiment configuration.
    """
    __slots__ = ('_cfg',)

    def __init__(self,cfg):
        object.__setattr__(self,'_cfg',cfg)

    @property
    def arm(self):
        return 'joint'

    def __getattr__(self,name):
        return getattr(self._cfg,name)

    def __setattr__(self,name,value):
        raise AttributeError('Group policy view is read-only')

    def to_dict(self):
        return self._cfg.to_dict()


def parameter_groups(cfg,model):
    validate_config(cfg)
    return _group_tools.parameter_groups(_JointGroupPolicy(cfg),model)


def optimizer_scheduler_factory(cfg,model):
    validate_config(cfg)
    return _group_tools.optimizer_scheduler_factory(_JointGroupPolicy(cfg),model)


def validate_resume_payload(payload,cfg,*,allow_completed=False):
    validate_config(cfg)
    return _group_tools.validate_resume_payload(payload,_JointGroupPolicy(cfg),allow_completed=allow_completed)


def validate_best_metadata(best,latest,cfg):
    validate_config(cfg)
    return _best_metadata(best,latest,cfg)


def validate_runtime_resume(torch,model,payload,cfg):
    validate_config(cfg)
    validate_model_identity(cfg,model)
    return _group_tools.validate_runtime_resume(torch,model,payload,_JointGroupPolicy(cfg))


def validate_output_paths(cfg,*,allowed_root=OUTPUT_ROOT):
    return _output_paths(cfg,allowed_root=allowed_root)


def check_run_state(cfg,*,resume,allowed_root=OUTPUT_ROOT):
    validate_config(cfg)
    validate_output_paths(cfg,allowed_root=allowed_root)
    if not resume:
        for path in (cfg.SAVE_DIR,cfg.LOG_DIR):
            if Path(path).exists():
                raise FileExistsError(f'Refusing to overwrite/reinitialize {path}')
        return
    for field in ('SAVE_PATH','BEST_PATH','CONFIG_PATH','STATUS_PATH','METRICS_PATH'):
        if not Path(getattr(cfg,field)).is_file():
            raise FileNotFoundError(f'Resume requires {field}: {getattr(cfg,field)}')
    if _json(json.loads(Path(cfg.CONFIG_PATH).read_text(encoding='utf-8'))) != _json(
            dict(cfg.to_dict(),config_fingerprint=fingerprint(cfg))):
        raise ValueError('Resume full config/source/data/architecture mismatch')
    status=json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
    if status.get('experiment')!=cfg.EXP_NAME or status.get('state') not in ('paused','running'):
        raise ValueError('Resume requires this unfinished arm and seed')
    if status.get('fingerprint',fingerprint(cfg))!=fingerprint(cfg):
        raise ValueError('Resume status fingerprint mismatch')


def validate_model_identity(cfg,model):
    actual=model.architecture_metadata
    for key,value in cfg.architecture_metadata.items():
        if _json(actual.get(key))!=_json(value):
            raise ValueError(f'Actual Joint+Swin256 architecture differs: {key}')
    if model.ue_frozen is not False:
        raise ValueError('Both Joint+Swin256 arms require joint UE training')
    return actual


def make_model_factory(cfg,model,*,source_state=None,resume_state=None):
    if (source_state is None)==(resume_state is None):
        raise ValueError('Choose exactly one source UE initialization or strict own-arm resume')
    used=False
    def factory(actual_cfg,device):
        nonlocal used,source_state,resume_state
        if used or _json(actual_cfg.to_dict())!=_json(cfg.to_dict()):
            raise RuntimeError('Factory reused or experiment identity changed')
        used=True
        fresh=source_state is not None
        if fresh:
            model.initialize_ue_from_full(source_state)
        else:
            model.load_state_dict(resume_state,strict=True)
        model.to(device)
        validate_model_identity(cfg,model)
        parameter_groups(cfg,model)
        source_state=resume_state=None
        if fresh:
            from main_train import set_seed
            set_seed(cfg.SEED)
        # Original trainer restores resumed RNG after model/optimizer creation.
        return model
    return factory


def write_record(cfg,name,record):
    path=Path(cfg.LOG_DIR)/name
    _safe_path(path,OUTPUT_ROOT)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,sort_keys=True,allow_nan=False)


def run_training(cfg,args):
    check_run_state(cfg,resume=args.resume)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Joint+Swin256 training requires CUDA; no CPU fallback')
    from main_train import RunOptions,set_seed,train_proposal
    from models.adaptive_hybrid_joint_swin import AdaptiveHybridPrecoderJointSwin
    data=make_channel_data(cfg)
    channel_data_api().validate_channel_data(cfg,data)
    with run_lock(cfg,resume=args.resume,allowed_root=OUTPUT_ROOT):
        check_run_state(cfg,resume=args.resume)
        set_seed(cfg.SEED)
        model=AdaptiveHybridPrecoderJointSwin(cfg)
        actual=validate_model_identity(cfg,model)
        parameter_groups(cfg,model)
        record=dict(arm=cfg.arm,seed=cfg.SEED,config=cfg.to_dict(),config_fingerprint=fingerprint(cfg),
            actual_model_metadata=actual,mode='resume' if args.resume else 'source_UE_random_BS',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip(),
            source_code_sha256={p:sha256_file(REPO_ROOT/p) for p in SOURCE_FILES},
            manifest_sha256=sha256_file(DEFAULT_MANIFEST),backend_before=_backend_info(torch),
            python=sys.version,platform=platform.platform(),scheduler_total_steps=20000,
            optimizer_groups=optimizer_group_spec(cfg),source_bs_weights_loaded=False,
            screening_only=True,full_training_bitwise_resume_verified=False)
        if args.resume:
            payload=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            validate_runtime_resume(torch,model,payload,cfg)
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
            validate_best_metadata(best,payload,cfg)
            validate_model_state(torch,model,best['state'])
            del best
            status=json.loads(Path(cfg.STATUS_PATH).read_text(encoding='utf-8'))
            if status['state']=='paused' and status.get('step')!=payload['step']:
                raise ValueError('Paused status/latest update count mismatch')
            start=payload['step']+1
            record['resume_sha256']=sha256_file(cfg.SAVE_PATH)
            factory=make_model_factory(cfg,model,resume_state=payload['state'])
        else:
            path=cfg.source_identity['path'] if args.checkpoint is None else str((REPO_ROOT/args.checkpoint).resolve())
            with Path(path).open('rb') as stream:
                verify_checkpoint_hash(path,cfg.source_identity['checkpoint_sha256'],stream=stream)
                stream.seek(0)
                payload=torch.load(stream,map_location='cpu',weights_only=False)
            validate_checkpoint_metadata(payload,cfg.source_identity)
            start=1
            record.update(source_path=path,source_sha256=cfg.source_identity['checkpoint_sha256'])
            factory=make_model_factory(cfg,model,source_state=payload['state'])
        record.update(start_step=start,ue_state_sha256_before=_group_tools.ue_state_digest(torch,payload['state']),
            planned_end_step=min(20000,start+args.stop_after-1) if args.stop_after is not None else 20000)
        del payload
        if not args.resume:
            Path(cfg.SAVE_DIR).mkdir(parents=True,exist_ok=False)
            Path(cfg.LOG_DIR).mkdir(parents=True,exist_ok=False)
        token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')+uuid.uuid4().hex[:8]
        write_record(cfg,f'invocation_{token}.json',record)
        try:
            train_proposal(cfg,RunOptions(resume=args.resume,skip_existing=False,
                stop_after=args.stop_after,init_checkpoint=None),model_factory=factory,
                channel_data=data,optimizer_scheduler_factory=optimizer_scheduler_factory)
            latest=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
            validate_resume_payload(latest,cfg,allow_completed=True)
            if latest['step']!=record['planned_end_step']:
                raise RuntimeError('Trainer stopped outside the fixed update budget')
            best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
            validate_best_metadata(best,latest,cfg)
            if not args.resume:
                verify_checkpoint_hash(record['source_path'],cfg.source_identity['checkpoint_sha256'])
            result=dict(completed_updates=latest['step'],best_step=best['step'],best_metric=best['best_metric'],
                latest_sha256=sha256_file(cfg.SAVE_PATH),best_sha256=sha256_file(cfg.BEST_PATH),
                backend_after=_backend_info(torch),screening_only=True)
            write_record(cfg,f'completion_{token}.json',result)
            print(json.dumps(result,indent=2))
        except Exception as error:
            write_record(cfg,f'failure_{token}.json',dict(error_type=type(error).__name__,error=str(error)))
            raise


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=SEEDS)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--train',action='store_true')
    mode.add_argument('--stop-after',type=int)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--checkpoint',type=Path)
    return parser


def main(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    if not(args.dry_run or args.train or args.stop_after is not None):
        if args.resume or args.checkpoint is not None:parser.error('Choose explicit execution mode')
        parser.print_help();return 0
    if args.arm is None or args.seed is None:parser.error('Choose arm and seed explicitly')
    if args.resume and (args.dry_run or args.checkpoint is not None):parser.error('Resume forbids fresh source overrides/dry-run')
    if args.stop_after is not None and (not 0<args.stop_after<=20000 or args.stop_after%500):
        parser.error('stop-after must be a positive multiple of500 within the fixed20k budget')
    cfg=joint_swin_config(args.arm,args.seed);validate_config(cfg);validate_output_paths(cfg)
    if args.dry_run:
        print(json.dumps(dict(action='plan only; no model imports/weights/output writes',config=cfg.to_dict(),
            fingerprint=fingerprint(cfg),best=cfg.BEST_PATH,latest=cfg.SAVE_PATH,
            source_path=str(args.checkpoint) if args.checkpoint else cfg.source_identity['path'],
            scheduler_total_steps=20000,optimizer_groups=optimizer_group_spec(cfg)),indent=2));return 0
    run_training(cfg,args);return 0


if __name__=='__main__':raise SystemExit(main())
