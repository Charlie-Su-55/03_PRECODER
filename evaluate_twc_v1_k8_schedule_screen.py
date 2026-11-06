#!/usr/bin/env python3
"""Paired validation of the four fixed3024-update K8 schedule-screen arms."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from configs import twc_v1_k8_schedule_screen_config as cfg

METRICS = ('dl_nmse', 'sum_rate_bps_hz')
POLICIES = {'hybrid_1_3': (1, 3), 'hybrid_1_1': (1, 1)}
ARTIFACTS = ('summary.csv', 'samples.csv.gz', 'paired_deltas.csv', 'results.json')


def conditions():
    return tuple((arm, g, policy, gamma) for arm in cfg.ARMS
        for g in ((1,) if arm == 'fdma_control' else (4,) if arm == 'specialist_g4' else (1, 2, 4))
        for policy in (('fdma',) if arm == 'fdma_control' else tuple(POLICIES))
        for gamma in cfg.GAMMAS)


def statistics_for(values):
    if not values or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
        raise ValueError('Finite nonempty signed values required')
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return dict(mean=mean, se=se, ci95_low=None if se is None else mean-1.96*se,
                ci95_high=None if se is None else mean+1.96*se)


def summarize_samples(samples, count=256, batch_size=4):
    if type(count) is not int or count < batch_size or count % batch_size or batch_size != 4:
        raise ValueError('Complete batches of four multi-user channels required')
    index = {condition: {} for condition in conditions()}
    fingerprints = {}
    for row in samples:
        key = tuple(row.get(k) for k in ('arm', 'group_count', 'energy_policy', 'gamma_db'))
        i = row.get('sample_index')
        if key not in index or type(i) is not int or not 0 <= i < count or i in index[key]:
            raise ValueError('Unexpected, missing-identity or duplicate sample key')
        if (row.get('seed') != cfg.SEED or row.get('checkpoint_step') != cfg.UPDATES
                or row.get('cumulative_updates') != cfg.SOURCE_UPDATES+cfg.UPDATES
                or row.get('batch_index') != i//batch_size
                or row.get('pilot_energy_per_ue') != 16.0 or row.get('downlink_snr_db') != 25.0):
            raise ValueError('Sample condition/source/update identity mismatch')
        fingerprint = row.get('channel_fingerprint')
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError('Actual common channel fingerprint required')
        if fingerprints.setdefault(i//batch_size, fingerprint) != fingerprint:
            raise ValueError('Methods or conditions did not share actual channels')
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Nonfinite/negative metric: '+metric)
        for metric, tolerance in (('energy_max_abs_error', 1e-4), ('rzf_power_max_abs_error', 1e-5)):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= tolerance:
                raise ValueError('Physical constraint check failed: '+metric)
        index[key][i] = row
    summary = []
    for key, rows in index.items():
        if set(rows) != set(range(count)):
            raise ValueError('Incomplete method/G/energy/gamma sample coverage')
        row = dict(zip(('arm', 'group_count', 'energy_policy', 'gamma_db'), key))
        row.update(seed=cfg.SEED, num_samples=count, num_batches=count//batch_size,
                   batch_size=batch_size, checkpoint_step=cfg.UPDATES,
                   cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,
                   role='validation', test_used=False)
        for metric in METRICS:
            row.update({metric+'_'+name: value for name, value in
                        statistics_for([rows[i][metric] for i in range(count)]).items()})
        summary.append(row)
    return summary, index


def contrasts():
    result = []
    for policy in POLICIES:
        for g in (1, 2, 4):
            result.append((f'interleaved_minus_blocked_G{g}_{policy}',
                ('interleaved_universal', g, policy), ('blocked_universal', g, policy)))
        for control in ('blocked_universal', 'interleaved_universal'):
            result.append((f'specialist_minus_{control}_G4_{policy}',
                ('specialist_g4', 4, policy), (control, 4, policy)))
        for arm in cfg.ARMS:
            if arm == 'fdma_control': continue
            for g in ((4,) if arm == 'specialist_g4' else (1, 2, 4)):
                result.append((f'{arm}_minus_FDMA_G{g}_{policy}',
                    (arm, g, policy), ('fdma_control', 1, 'fdma')))
    return result


def paired_statistics(samples, count=256, batch_size=4):
    _, index = summarize_samples(samples, count, batch_size)
    result = []
    scopes = [(str(int(g)), (g,)) for g in cfg.GAMMAS]+[
        ('low_0_5_10', cfg.GAMMAS[:3]), ('all_six', cfg.GAMMAS)]
    for name, left, right in contrasts():
        for scope, gammas in scopes:
            for metric in METRICS:
                values = [statistics.fmean(index[(*left,g)][i][metric]-index[(*right,g)][i][metric]
                                          for g in gammas) for i in range(count)]
                batches = [statistics.fmean(values[i:i+batch_size]) for i in range(0,count,batch_size)]
                row = dict(contrast=name, left_arm=left[0], right_arm=right[0], group_count=left[1],
                    energy_policy=left[2], estimand=scope, gamma_points=','.join(str(int(g)) for g in gammas),
                    metric=metric, seed=cfg.SEED, num_samples=count, num_batches=len(batches), batch_size=batch_size,
                    primary=(left[:2]==('interleaved_universal',4) and right[0]=='blocked_universal'),
                    sample_unit='one_complete_K8_channel', batch_unit='sample_index//4',
                    ci_scope='fixed-checkpoint validation sampling; not training-seed stability',
                    interpretation='negative NMSE delta is favorable; positive rate delta is favorable')
                for prefix, value in (('sample_', statistics_for(values)), ('batch_', statistics_for(batches))):
                    row.update({prefix+k:v for k,v in value.items()})
                row['batch_ci_estimable'] = len(batches)>1
                result.append(row)
    return result


def _write_json(path, data):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _write_csv(path, rows, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, 'wt', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def artifact_hashes(directory):
    return {name:cfg.sha256_file(Path(directory)/name) for name in ARTIFACTS}


def _output(directory):
    path=Path(directory).resolve()
    path.relative_to(cfg.OUTPUT_ROOT.resolve())
    if path==cfg.OUTPUT_ROOT.resolve() or path.exists():raise FileExistsError('New isolated evaluation directory required')
    path.mkdir(parents=True,exist_ok=False)
    return path


def _tensor_sha(tensor):
    value=tensor.detach().contiguous().cpu()
    digest=hashlib.sha256((str(value.dtype)+str(tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes());return digest.hexdigest()


def _state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(_tensor_sha(value).encode())
    return digest.hexdigest()


def _frozen_snapshot(module):
    return dict(buffers={n:_tensor_sha(v) for n,v in module.named_buffers()},
                modes={n:m.training for n,m in module.named_modules()},
                versions={n:p._version for n,p in module.named_parameters()})


def run_evaluation(checkpoints, output_dir, *, device='cuda:0'):
    """Run only final-checkpoint validation, returning small report paths/data."""
    if set(checkpoints)!=set(cfg.ARMS) or device!='cuda:0':
        raise ValueError('Four explicit paths and CUDA:0 are required')
    output=_output(output_dir)
    manifest=dict(status='running',role='validation',test_used=False,optimizer_updates=0,
        seed=cfg.SEED,checkpoint_selection='predeclared final3024; no validation reselection',
        samples=cfg.VALIDATION_SAMPLES,batch_size=cfg.BATCH_SIZE,
        validation_batch_addresses=[cfg.VALIDATION_START_INDEX+i for i in range(cfg.VALIDATION_BATCHES)],
        pilot_energy_per_ue=16.,downlink_snr_db=25.,gamma_db=list(cfg.GAMMAS),
        energy_ratios=POLICIES,shared_inputs=[],methods={},numerical_failures=[],
        interpretation='single-seed schedule screening; no independent test or architecture claim')
    inputs={}; chain=None;ce_before=None;failure=None
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        import train_twc_v1_k8_codec_warmup as old
        from train_twc_v1_k8_schedule_screen import verify_completed_payload, source_code_record, ce_identity
        code=source_code_record();ce_record=ce_identity()
        manifest['code_commit']=code['commit'];manifest['source_sha256']=code['sha256']
        manifest['ce_identity']=ce_record
        manifest['backend']=dict(torch=torch.__version__,cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),matmul_precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        for relative in (old.cfg.CE_CHECKPOINT,old.cfg.CE_CALIBRATION):
            p=(cfg.REPO_ROOT/relative).resolve();inputs[str(p)]=cfg.sha256_file(p)
        chain=old.build_chain(torch.device(device)).eval()
        ce_before=_state_sha(chain.ce_model.state_dict())
        states={}; state_hashes={};data_protocol=None
        for arm in cfg.ARMS:
            path=Path(checkpoints[arm]).resolve();inputs[str(path)]=cfg.sha256_file(path)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            verify_completed_payload(torch,payload,arm)
            if payload['ce_identity']!=ce_record:raise ValueError('Frozen CE/calibration identity differs from training')
            if data_protocol is None:data_protocol=payload['data_protocol']
            elif data_protocol!=payload['data_protocol']:raise ValueError('Training data protocols differ')
            expected=chain.codec_model.state_dict();state=payload['codec_model']
            if set(state)!=set(expected):raise ValueError('Strict full model keys required')
            for name,value in state.items():
                if value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('State shape/dtype/finite mismatch '+name)
            chain.codec_model.load_state_dict(state,strict=True);chain.eval()
            states[arm]=state;state_hashes[arm]=_state_sha(state)
            manifest['methods'][arm]=dict(path=str(path),sha256=inputs[str(path)],step=cfg.UPDATES,
                cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,source=payload['source'],
                parameters=sum(p.numel() for p in chain.codec_model.parameters()),strict_loading=True)
            del payload
        sizes={k:cfg.BATCH_SIZE for k in ('ce_train','codec_train','ce_calibration','validation')};sizes['test']=1
        data=old.TWCv1K8ReplayableData(base_seed=old.cfg.BASE_DATA_SEED,batch_sizes=sizes,
            device=device,cdl_model=old.cfg.CDL_MODEL,speed=old.cfg.SPEED_MPS,seed_namespace=old.cfg.SEED_NAMESPACE)
        if data.protocol!=data_protocol:raise ValueError('Evaluation must use the matched original data identity')
        manifest['data_protocol']=data.protocol
        raw=[];torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for bi in range(cfg.VALIDATION_BATCHES):
                sample=data.sample('validation',cfg.VALIDATION_START_INDEX+bi)
                hdl,hul=sample['H_dl'],sample['H_ul']
                original_channels=(_tensor_sha(hdl),_tensor_sha(hul))
                input_row=dict(batch_index=bi,address=sample['address'],H_dl_sha256=original_channels[0],
                               H_ul_sha256=original_channels[1],noise={})
                noises={}
                for gi,gamma in enumerate(cfg.GAMMAS):
                    tensors=[]
                    for kind in ('pilot','data'):
                        generator=torch.Generator(device='cpu').manual_seed(cfg.evaluation_noise_seed(bi,gi,kind))
                        noise=torch.view_as_complex(torch.randn(cfg.BATCH_SIZE,32,256,2,generator=generator)).to(device)
                        tensors.append(noise)
                        input_row['noise'][f'{gamma}_{kind}']=_tensor_sha(noise)
                    noises[gamma]=tuple(tensors)
                ul_reference={}
                for arm in cfg.ARMS:
                    chain.codec_model.load_state_dict(states[arm],strict=True);chain.eval()
                    before=_frozen_snapshot(chain)
                    for _,groups,policy,gamma in (c for c in conditions() if c[0]==arm):
                        if arm=='fdma_control':layout,split=old.protocol.FDMALayout(),None
                        else:
                            ratio=POLICIES[policy];layout=old.protocol.default_hybrid_layout(groups)
                            split=old.protocol.EnergySplit(16.,48.*ratio[0]/sum(ratio),48.*ratio[1]/sum(ratio))
                        pilot_noise,data_noise=noises[gamma]
                        out=chain(hdl,hul,feedback_gamma_db=gamma,pilot_energy_per_ue=16.,
                            pilot_standard_noise=pilot_noise,data_standard_noise=data_noise,layout=layout,energy_split=split)
                        ul_digest=_tensor_sha(out['H_ul_est'])
                        if ul_reference.setdefault(gamma,ul_digest)!=ul_digest:raise ValueError('UL estimate differs across methods/layouts')
                        metrics=chain.metrics(out,hdl,rzf_alpha=.01,downlink_snr_db=25.)
                        nmse=metrics['dl_nmse_per_user'].mean(-1)
                        rate=metrics['sum_rate_bps_hz']
                        energy=(out['energy']['actual_data_per_ue']-48.).abs().amax(-1)
                        power=(metrics['rzf_per_subcarrier_power']-1).abs().amax(-1)
                        arrays={k:v.detach().cpu().tolist() for k,v in dict(dl_nmse=nmse,sum_rate_bps_hz=rate,
                            energy_max_abs_error=energy,rzf_power_max_abs_error=power,
                            desired_signal_power=metrics['desired_signal_power_mean'],
                            interference_power=metrics['interference_power_mean']).items()}
                        for i in range(cfg.BATCH_SIZE):
                            raw.append(dict(arm=arm,seed=cfg.SEED,group_count=groups,energy_policy=policy,
                                gamma_db=gamma,pilot_energy_per_ue=16.,downlink_snr_db=25.,
                                sample_index=bi*cfg.BATCH_SIZE+i,batch_index=bi,
                                channel_fingerprint=sample['sample_fingerprint'],checkpoint_step=cfg.UPDATES,
                                cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,
                                **{k:v[i] for k,v in arrays.items()}))
                    if before!=_frozen_snapshot(chain):raise ValueError('Evaluation modified buffers/modes/parameter versions')
                    if bi in (0,cfg.VALIDATION_BATCHES-1) and _state_sha(chain.codec_model.state_dict())!=state_hashes[arm]:
                        raise ValueError('Evaluation modified model state')
                if original_channels!=(_tensor_sha(hdl),_tensor_sha(hul)):raise ValueError('Shared channel inputs changed')
                for gi,gamma in enumerate(cfg.GAMMAS):
                    for kind,noise in zip(('pilot','data'),noises[gamma]):
                        if _tensor_sha(noise)!=input_row['noise'][f'{gamma}_{kind}']:raise ValueError('Shared noise changed')
                manifest['shared_inputs'].append(input_row)
                print(f'Paired schedule validation {bi+1}/{cfg.VALIDATION_BATCHES}',flush=True)
        summary,_=summarize_samples(raw,cfg.VALIDATION_SAMPLES,cfg.BATCH_SIZE)
        paired=paired_statistics(raw,cfg.VALIDATION_SAMPLES,cfg.BATCH_SIZE)
        _write_csv(output/'summary.csv',summary);_write_csv(output/'samples.csv.gz',raw,True)
        _write_csv(output/'paired_deltas.csv',paired)
        _write_json(output/'results.json',dict(summary=summary,paired=paired))
        manifest.update(status='completed',summary_rows=len(summary),sample_rows=len(raw),paired_rows=len(paired),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),results_verified=True)
    except BaseException as error:
        failure=error;manifest.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        errors=[]
        for path,digest in inputs.items():
            try:
                if cfg.sha256_file(path)!=digest:errors.append('Input changed: '+path)
            except OSError as error:errors.append('Input unreadable: '+path+': '+str(error))
        for path,digest in manifest.get('source_sha256',{}).items():
            try:
                if cfg.sha256_file(cfg.REPO_ROOT/path)!=digest:errors.append('Code changed: '+path)
            except OSError as error:errors.append('Code unreadable: '+path+': '+str(error))
        if chain is not None and ce_before is not None and _state_sha(chain.ce_model.state_dict())!=ce_before:
            errors.append('Frozen CE state changed')
        manifest.update(input_sha256=inputs,freeze_checks=dict(passed=not errors,errors=errors),
                        checkpoint_files_unchanged=not any(e.startswith('Input') for e in errors))
        if errors:
            failure=failure or RuntimeError('; '.join(errors));manifest.update(status='failed',error=str(failure))
        if failure is None:manifest['output_sha256']=artifact_hashes(output)
        _write_json(output/'manifest.json',manifest)
    if failure is not None:raise RuntimeError(f'Validation failed; evidence retained at {output}') from failure
    return dict(output_directory=str(output),manifest_path=str(output/'manifest.json'),
                summary=summary,paired=paired,artifacts=manifest['output_sha256'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',action='append',default=[],metavar='ARM=PATH')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(arms=list(cfg.ARMS),conditions=len(conditions()),samples=256,batch_size=4,
            validation_start_index=1024,role='validation',test_used=False,models_loaded=False,
            checkpoint_step=3024,device='cuda:0',ci='sample and batch paired; mean+-1.96SE'),indent=2));return 0
    if not args.evaluate:parser.print_help();return 0
    checkpoints={}
    for value in args.checkpoint:
        arm,sep,path=value.partition('=')
        if not sep or arm in checkpoints:parser.error('Each --checkpoint must be a distinct ARM=PATH')
        checkpoints[arm]=path
    if args.output_dir is None:parser.error('--output-dir required')
    run_evaluation(checkpoints,args.output_dir);return 0


if __name__=='__main__':raise SystemExit(main())
