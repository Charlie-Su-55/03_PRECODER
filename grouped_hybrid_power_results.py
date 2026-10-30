"""CSV-only reporting for energy gates; no weights/model execution or paper edits."""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

from as_receiver_batch_support import save_json, sha256
from as_receiver_results import rows, finite
from grouped_codec_results import _uncertainty
from grouped_codec_fdma_results import _safe_path, _read_json, _copy_small, _write_csv
from evaluate_grouped_codec_fdma import csv_rows
from evaluate_grouped_hybrid_power import (
    ARMS, SEEDS, METHODS, GAMMAS, TEST_SEED, TAG, CONTRASTS, LAYOUTS, PRIMARY,
    OPTIONAL_METRICS, METRICS, FROZEN, SOURCE_PRIOR_OFFSET, _available,
    protocol, artifact_hashes, verify_shared_inputs, paired_rows, source_identity)

LABELS = {'fdma_control': 'FDMA +20k control', 'h32_control': 'H32 +20k control',
    'h32_static_power': 'H32 constant gate +20k', 'h32_gamma_power': 'H32 gamma gate +20k',
    'frozen_fdma_source': 'Frozen FDMA source', 'frozen_h32_source': 'Frozen H32 source'}
RULE = 'low gamma mean>=0.5; low batch95CI lower>0; gamma25 batch95CI lower>=-0.1'
VALIDATION_STEPS = tuple(range(0,20001,500))


def _integer(row, key):
    value = finite(row,key)
    if value != int(value): raise ValueError('Nonintegral '+key)
    return int(value)


def _units(row, seed):
    expected = dict(training_seed=seed,seed=TEST_SEED,num_samples=1600,num_batches=200,batch_size=8)
    if any(_integer(row,k)!=v for k,v in expected.items()) or finite(row,'downlink_snr_db')!=25:
        raise ValueError('Formal1600 channels /200 batches/test20270108/DL25 required')
    if any(n in row for n in ('feedback_snr_db','allocation_df','allocation_da')):
        raise ValueError('Do not mix historical received-SNR protocols')


def interpret(summary, paired, training_seed):
    if type(training_seed) is not int or training_seed not in SEEDS:
        raise ValueError('Two training seeds must be interpreted separately')
    expected = {(m,g) for m in METHODS for g in GAMMAS}
    found = []
    for row in summary:
        _units(row,training_seed)
        method,gamma=row['method_key'],finite(row,'gamma_db');found.append((method,gamma))
        if (method,gamma) not in expected: raise ValueError('Unexpected method/gamma')
        d,a=LAYOUTS[method]
        if (_integer(row,'Df_per_user')!=d or _integer(row,'Da_per_group')!=a or
                _integer(row,'occupied_feedback_RE')!=256 or _integer(row,'active_symbols_per_ue')!=d+a):
            raise ValueError('Exclusive/shared support identity changed')
        if str(row.get('preflight','')).lower() not in ('false','0') or str(row.get('prior_frozen','')).lower() not in ('true','1'):
            raise ValueError('Frozen formal results only; no preflight rows')
        if not math.isclose(finite(row,'noise_variance_N0'),.25*10**(-gamma/10),rel_tol=1e-10):
            raise ValueError('Absolute N0 changed')
        for metric in METRICS:
            values=[row.get(metric+'_'+part) for part in ('mean','se','ci95_low','ci95_high')]
            if metric in OPTIONAL_METRICS and not _available(method,metric):
                if any(v not in ('',None) for v in values): raise ValueError('Empty support diagnostic must remain NA')
            else:
                if any(v in ('',None) for v in values): raise ValueError('Missing available diagnostic '+metric)
                _uncertainty(row,metric+'_')
    if len(found)!=36 or set(found)!=expected: raise ValueError('All six methods and six gamma points required')
    scopes=[('per_gamma',str(g)) for g in GAMMAS]+[
        ('low_gamma_mean','0,5,10'),('all_gamma_mean','0,5,10,15,20,25')]
    expected_pairs={(name,scope,points) for name in CONTRASTS for scope,points in scopes}
    indexed={}
    for row in paired:
        _units(row,training_seed)
        key=(row['contrast_id'],row['estimand'],row['gamma_points'])
        if key not in expected_pairs or key in indexed: raise ValueError('Unexpected/duplicate signed contrast')
        coefficients=row['coefficients']
        if isinstance(coefficients,str): coefficients=json.loads(coefficients)
        if coefficients!=CONTRASTS[key[0]]: raise ValueError('Predeclared contrast changed')
        if (str(row.get('training_seeds_pooled','')).lower() not in ('false','0') or
                str(row.get('batch_ci_estimable','')).lower() not in ('true','1') or
                row.get('sample_ci_unit')!='complete_multi_user_channel' or row.get('batch_ci_unit')!='sample_index//8'):
            raise ValueError('Sample/batch paired units changed or training seeds pooled')
        for prefix in ('delta_','batch_delta_'): _uncertainty(row,prefix)
        if not math.isclose(finite(row,'delta_mean'),finite(row,'batch_delta_mean'),rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError('Batch mean must preserve signed estimate')
        indexed[key]=row
    if set(indexed)!=expected_pairs: raise ValueError('All eight contrasts and eight estimands required')
    comparisons={name:dict(per_gamma=[indexed[name,'per_gamma',str(g)] for g in GAMMAS],
        low_gamma=indexed[name,'low_gamma_mean','0,5,10'],
        all_gamma=indexed[name,'all_gamma_mean','0,5,10,15,20,25']) for name in CONTRASTS}
    criteria={}
    for name in PRIMARY:
        low,high=comparisons[name]['low_gamma'],comparisons[name]['per_gamma'][-1]
        criteria[name]=(finite(low,'delta_mean')>=.5 and finite(low,'batch_delta_ci95_low')>0 and
                        finite(high,'batch_delta_ci95_low')>=-.1)
    gates=[dict(method_key=r['method_key'],gamma_db=finite(r,'gamma_db'),
        g_log_energy_ratio=finite(r,'g_log_energy_ratio_mean')) for r in summary]
    energy=[]
    for r in summary:
        energy.append(dict(method_key=r['method_key'],gamma_db=finite(r,'gamma_db'),
            **{n:r[n] for n in ('exclusive_energy_mean_per_ue_mean','shared_energy_mean_per_ue_mean',
                 'exclusive_energy_per_active_symbol_mean','shared_energy_per_active_symbol_mean')}))
    return dict(training_seed=training_seed,test_seed=TEST_SEED,all_rates=summary,
        comparisons=comparisons,engineering_criteria=criteria,engineering_rule=RULE,gates=gates,energy=energy,
        training_seeds_pooled=False,
        criterion_scope='Per-seed engineering screen against continued FDMA, not journal acceptance',
        gate_attribution='Gamma adaptation must be judged against independently trained constant gate, not only H32 control',
        reference_scope='Frozen sources explain additional training, not primary performance comparators',
        proxy_scope='Historical diagonal-prior linear error; not calibrated posterior uncertainty',
        ci_scope='Sampling conditional on selected checkpoints, not multiple-training-seed significance')


def _typed_rows(path, raw):
    result=[]
    integer=('sample_index','batch_index','training_seed','seed','Df_per_user','Da_per_group',
        'occupied_feedback_RE','active_symbols_per_ue','num_samples','num_batches','batch_size')
    for item in csv_rows(path):
        row=dict(item)
        for name in integer:
            if name in row: row[name]=int(row[name])
        for name in ('gamma_db','downlink_snr_db','noise_variance_N0','rzf_regularization'):
            if name in row:row[name]=float(row[name])
        for metric in METRICS:
            for name in (metric,) if raw else tuple(metric+'_'+s for s in ('mean','se','ci95_low','ci95_high')):
                row[name]=None if row[name]=='' else float(row[name])
        for name in ('preflight','prior_frozen'):
            if name in row:
                if row[name] not in ('True','False','true','false'):raise ValueError('Invalid boolean '+name)
                row[name]=row[name].lower()=='true'
        result.append(row)
    return result


def training_curves(training_records):
    if set(training_records)!={str(s) for s in SEEDS}:raise ValueError('Both training seeds required')
    per_seed,files={},{}
    for seed in SEEDS:
        records=training_records[str(seed)]
        if set(records)!=set(ARMS):raise ValueError('Exactly four new arm logs per seed required')
        per_seed[str(seed)]={}
        for arm,record in records.items():
            if _integer(record,'completed_updates')!=20000:raise ValueError('Each continuation must complete20k')
            path=_safe_path(record['metrics_path'],file=True);digest=sha256(path)
            if digest!=record['metrics_sha256']:raise ValueError('Validation log hash changed')
            points=[]
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():continue
                    item=json.loads(line)
                    if item.get('event')!='validation':
                        if item.get('event')!='train_gamma':raise ValueError('Unknown metrics event')
                        continue
                    if item.get('selection_metric_name')!='mean_all' or item.get('gamma_db')!=list(GAMMAS) or finite(item,'downlink_snr_db')!=25:
                        raise ValueError('Validation protocol changed')
                    table=item['table']
                    if set(table)!={f'layout0_gamma{g}' for g in GAMMAS}:raise ValueError('Six validation nodes required')
                    rates={str(g):finite(table,f'layout0_gamma{g}') for g in GAMMAS}
                    mean=finite(item,'selection_metric')
                    if not math.isclose(mean,sum(rates.values())/6,rel_tol=1e-10,abs_tol=1e-10):
                        raise ValueError('Checkpoint selection not six-node mean_all')
                    points.append(dict(step=_integer(item,'step'),mean_all=mean,rates=rates))
            if sha256(path)!=digest:raise ValueError('Training log changed during reading')
            if tuple(p['step'] for p in points)!=VALIDATION_STEPS:raise ValueError('Need step0 and all forty500-step validations')
            best=max(points,key=lambda p:p['mean_all'])
            if _integer(record,'best_step')!=best['step']:raise ValueError('Best must be first strict maximum including step0')
            tail=[p for p in points if p['step']>=15000]
            xs,ys=[p['step']/1000 for p in tail],[p['mean_all'] for p in tail]
            xm,ym=sum(xs)/len(xs),sum(ys)/len(ys)
            slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
            per_seed[str(seed)][arm]=dict(best_step=best['step'],best_mean_all=best['mean_all'],
                selected_prior_count=SOURCE_PRIOR_OFFSET+best['step'],latest_prior_count=40000,
                completed_updates=20000,points=points,last_five_validation_points=points[-5:],
                last_5000_updates=dict(start_step=15000,end_step=20000,num_points=11,
                    mean_all_start=ys[0],mean_all_end=ys[-1],end_minus_start=ys[-1]-ys[0],
                    ols_slope_per_1000_updates=slope,minimum=min(ys),maximum=max(ys)),
                metrics_path=str(path),metrics_sha256=digest,convergence_established=False)
            files[str(path)]=digest
    if len(files)!=8:raise ValueError('Eight distinct metrics paths required')
    return dict(per_seed=per_seed,source_files=files,validation_samples=256,validation_batch_size=8,
                training_seeds_pooled=False,convergence_established=False)


def _figures(output, interpretations, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(zip(METHODS,('#2864a5','#529f72','#b77d37','#9064a8','#727272','#9baca1')))
    def finish(fig,name):
        for suffix in ('pdf','png'):fig.savefig(output/(name+'.'+suffix),dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True,layout='constrained')
    for axis,seed in zip(axes,SEEDS):
        for method in METHODS:
            selected=sorted((r for r in interpretations[str(seed)]['all_rates'] if r['method_key']==method),key=lambda r:finite(r,'gamma_db'))
            axis.errorbar(GAMMAS,[finite(r,'sum_rate_mean') for r in selected],
                yerr=[1.96*finite(r,'sum_rate_se') for r in selected],color=colors[method],
                marker='o',markersize=3,capsize=2,linewidth=1.1,
                linestyle='--' if method in FROZEN else '-',label=LABELS[method])
        axis.set(xlabel='Transmit-reference gamma (dB)',xticks=GAMMAS,title=f'Training seed {seed}')
        axis.grid(alpha=.18)
    axes[0].set_ylabel('Sum rate (bps/Hz); sample95% CI');axes[-1].legend(frameon=False,fontsize=7)
    finish(fig,'power_six_methods_all_gamma')
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,seed in enumerate(SEEDS):
        for row,names in enumerate((tuple(CONTRASTS)[:4],tuple(CONTRASTS)[4:])):
            axis=axes[row,col]
            for name in names:
                selected=interpretations[str(seed)]['comparisons'][name]['per_gamma']
                axis.errorbar(GAMMAS,[finite(r,'delta_mean') for r in selected],
                    yerr=[1.96*finite(r,'batch_delta_se') for r in selected],marker='o',markersize=3,
                    capsize=2,linewidth=1,label=name)
            axis.axhline(0,color='.5',linestyle='--',linewidth=.8)
            axis.set(xlabel='Transmit-reference gamma (dB)',xticks=GAMMAS,title=f'Training seed {seed}',
                     ylabel='Signed rate difference; batch95% CI')
            axis.grid(alpha=.18);axis.legend(frameon=False,fontsize=7)
    finish(fig,'power_all_paired_contrasts')
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for col,seed in enumerate(SEEDS):
        for method in METHODS:
            selected=sorted((r for r in interpretations[str(seed)]['all_rates'] if r['method_key']==method),key=lambda r:finite(r,'gamma_db'))
            for row,field in enumerate(('g_log_energy_ratio_mean','exclusive_energy_mean_per_ue_mean','exclusive_energy_per_active_symbol_mean')):
                axes[row,col].plot(GAMMAS,[finite(r,field) for r in selected],color=colors[method],
                    marker='o',markersize=3,label=LABELS[method],linestyle='--' if method in FROZEN else '-')
            if LAYOUTS[method][1]>0:
                axes[2,col].plot(GAMMAS,[finite(r,'shared_energy_per_active_symbol_mean') for r in selected],
                    color=colors[method],linestyle=':',label=LABELS[method]+' shared')
        for row in range(3):
            axes[row,col].set(xlabel='Transmit-reference gamma (dB)',xticks=GAMMAS,title=f'Training seed {seed}')
            axes[row,col].grid(alpha=.18)
        axes[0,col].axhline(0,color='.5',linewidth=.6)
    for row,label in enumerate(('Gate g (natural log energy ratio)','Exclusive energy per UE (shared=64-exclusive)',
                                'Energy per active symbol (solid exclusive / dotted shared)')):axes[row,0].set_ylabel(label)
    axes[0,-1].legend(frameon=False,fontsize=7)
    finish(fig,'power_gate_and_energy_all_gamma')
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,layout='constrained')
    for axis,seed in zip(axes,SEEDS):
        for arm in ARMS:
            points=curves['per_seed'][str(seed)][arm]['points']
            axis.plot([p['step'] for p in points],[p['mean_all'] for p in points],color=colors[arm],label=LABELS[arm])
        axis.set(xlabel='Additional optimizer updates',title=f'Training seed {seed}');axis.grid(alpha=.18)
    axes[0].set_ylabel('Fixed256-channel validation mean_all (bps/Hz)');axes[-1].legend(frameon=False,fontsize=8)
    finish(fig,'power_validation_mean_all')


def result_paragraphs(interpretations, curves):
    en=['Internal energy-gate continuation screen. All four arms receive the same additional20000-update budget; the primary comparator is continued FDMA, not either frozen source. The channel/energy/noise protocol is separate from historical received-feedback-SNR experiments. No historical Swin curve is overlaid.']
    zh=['能量门控内部筛选：四个新臂均追加相同的20000次更新，主要对照为继续训练后的FDMA，而不是冻结源模型。本批使用独立信道、能量和噪声协议，不与历史接收反馈SNR曲线拼接，也不叠加旧Swin曲线。']
    for seed in SEEDS:
        item=interpretations[str(seed)]
        for name in CONTRASTS:
            for scope,key in (('low gamma0/5/10','low_gamma'),('all six gamma','all_gamma')):
                r=item['comparisons'][name][key]
                text=f"{finite(r,'delta_mean'):+.6f} bps/Hz; sample95% CI [{finite(r,'delta_ci95_low'):+.6f},{finite(r,'delta_ci95_high'):+.6f}], batch95% CI [{finite(r,'batch_delta_ci95_low'):+.6f},{finite(r,'batch_delta_ci95_high'):+.6f}]"
                en.append(f'Training seed {seed}, {name}, {scope}: {text}.')
                zh.append(f'训练seed {seed}，{name}，{scope}：{text}。')
        for name in PRIMARY:
            r=item['comparisons'][name]['per_gamma'][-1]
            en.append(f"Seed {seed}, {name}: engineering criterion {'met' if item['engineering_criteria'][name] else 'not met'}; gamma25 delta {finite(r,'delta_mean'):+.6f}, batch95% CI [{finite(r,'batch_delta_ci95_low'):+.6f},{finite(r,'batch_delta_ci95_high'):+.6f}]. Rule: {RULE}; not a journal acceptance threshold.")
            zh.append(f"seed {seed}，{name}：工程判据{'满足' if item['engineering_criteria'][name] else '未满足'}；gamma25差值 {finite(r,'delta_mean'):+.6f}，batch95%区间[{finite(r,'batch_delta_ci95_low'):+.6f},{finite(r,'batch_delta_ci95_high'):+.6f}]。该判据不是期刊验收标准。")
        for arm in ARMS:
            r=curves['per_seed'][str(seed)][arm];tail=r['last_5000_updates']
            points=', '.join(f"{p['step']}:{p['mean_all']:.6f}" for p in r['last_five_validation_points'])
            en.append(f"Seed {seed}, {arm}: best local step {r['best_step']}, inherited prior20000, selected cumulative prior {r['selected_prior_count']}; last five validation points {points}; last5k end-minus-start {tail['end_minus_start']:+.6f}, slope per1000 updates {tail['ols_slope_per_1000_updates']:+.6f}. No convergence is inferred.")
            zh.append(f"seed {seed}，{arm}：选中本轮step {r['best_step']}，继承先验计数20000，选中累计计数{r['selected_prior_count']}；最后五次验证{points}；最后5k末减初{tail['end_minus_start']:+.6f}，每1000步斜率{tail['ols_slope_per_1000_updates']:+.6f}。这不证明训练已收敛。")
    en.append('A gamma-gated improvement over H32 alone does not establish gamma adaptation: the independently trained constant gate comparison is also reported. All six gamma points and negative differences are retained. Low/all-gamma estimates average within each complete multi-user channel before sample/batch aggregation. Training seeds remain separate; intervals describe conditional evaluation sampling only. Gates multiply exclusive amplitude before joint energy64 normalization and do not change support or total UE energy. Turning shared energy off would not produce the full FDMA64 layout. Soft/corrected codeword errors and raw H_hat NMSE are task-representation diagnostics, not extra denoising objectives. Reliability proxies are not calibrated posterior errors. No post-training gate-disable intervention is represented as retrained ablation. Parameter counts are not measured latency. Raw samples and weights stay on the server; this report does not modify the manuscript.')
    zh.append('仅相对H32控制臂改善，不能证明收益来自gamma自适应，必须同时讨论独立训练常数门的比较。六个gamma点及所有负差值完整保留；低三点与全六点先在同一完整多用户信道内平均，再计算sample和batch区间。两训练seed分别报告，区间仅描述固定checkpoint下的评估采样。门控先改变独占幅度，再联合归一化到每UE能量64，不改变支持位置或总能量；即使关闭共享部分，H32也不会变成完整FDMA64布局。软码字、修正码字误差及原始H_hat NMSE仅为任务表示诊断，不添加去噪目标；proxy不称校准后验误差。不把训练后关门干预当作重新训练消融。参数量不代表实测时延。原始样本和权重留服务器，本报告不修改论文。')
    return en,zh


def organize_results(folder,evaluations,training_records,source_records,*,make_plots=True):
    keys={str(s) for s in SEEDS}
    if set(evaluations)!=keys or set(source_records)!=keys:raise ValueError('Both explicit seed records required')
    curves=training_curves(training_records)
    interpretations,manifests,sources,parameters={},{},{},{}
    source_files=dict(curves['source_files']);reference=None
    for seed in SEEDS:
        directory=_safe_path(evaluations[str(seed)]);manifest,digest=_read_json(directory/'manifest.json')
        if (manifest.get('status')!='completed' or manifest.get('results_verified') is not True or
                manifest.get('freeze_checks',{}).get('passed') is not True or manifest['freeze_checks'].get('errors')!=[] or
                set(manifest['freeze_checks'].get('loaded_methods',()))!=set(METHODS) or
                manifest.get('protocol')!=protocol(seed) or manifest.get('output_sha256')!=artifact_hashes(directory) or
                manifest.get('shared_channel_noise_inputs_unchanged') is not True or
                manifest.get('samples_hashed_pairing') is not True or manifest.get('prior_frozen') is not True):
            raise ValueError('Completed formal protocol/freeze/artifact checks required')
        for before,after in (('model_buffers_before','model_buffers_after'),('model_state_before','model_state_after'),
                             ('input_sha256_before','input_sha256_after')):
            if not manifest.get(before) or manifest[before]!=manifest.get(after):raise ValueError('Frozen state evidence mismatch: '+before)
        if set(manifest['model_buffers_before'])!=set(METHODS):raise ValueError('Six frozen model snapshots required')
        if any(any(s['training'].values()) for s in manifest['model_buffers_before'].values()):raise ValueError('All models must remain eval')
        reference=verify_shared_inputs(manifest,reference)
        code_hashes=manifest.get('runtime',{}).get('source_sha256')
        if not isinstance(code_hashes,dict) or not code_hashes:
            raise ValueError('Evaluation code provenance missing')
        for name,digest_value in code_hashes.items():
            path=_safe_path(Path(__file__).resolve().parent/name,file=True)
            if sha256(path)!=digest_value:raise ValueError('Evaluation code changed before report: '+name)
            source_files[str(path)]=digest_value
        for name,digest_value in manifest['output_sha256'].items():
            source_files[str(directory/name)]=digest_value
        sources_for_seed=source_records[str(seed)]
        if set(sources_for_seed)!=set(('fdma_control','h32_control')):raise ValueError('Two exact frozen source families required')
        for arm,source in sources_for_seed.items():
            identity=source_identity(arm,seed)
            if source['best_sha256']!=identity['best_sha256']:raise ValueError('Pinned source best mismatch')
            for name in ('best_path','latest_path','config_path','status_path','best_sha256','latest_sha256',
                         'config_sha256','status_sha256','best_step','latest_step'):
                if source[name]!=manifest['source_models'][arm][name]:raise ValueError('Source provenance differs: '+name)
        for arm in ARMS:
            trained,loaded=training_records[str(seed)][arm],manifest['methods'][arm]
            if (trained['best_step']!=loaded['selected_best_step'] or
                    loaded['prior_update_count']!=SOURCE_PRIOR_OFFSET+trained['best_step'] or
                    loaded['latest_step']!=20000 or loaded['latest_prior_update_count']!=40000 or
                    trained['best_prior_update_count']!=loaded['prior_update_count'] or
                    trained['latest_prior_update_count']!=40000 or loaded['strict_loading'] is not True):
                raise ValueError('New arm selection/cumulative prior/completion mismatch')
            for name in ('best_sha256','latest_sha256'):
                if trained[name]!=loaded[name]:raise ValueError('Trained/evaluated checkpoint differs')
        for method,arm in FROZEN.items():
            loaded=manifest['methods'][method]
            if (loaded['best_sha256']!=sources_for_seed[arm]['best_sha256'] or loaded['prior_update_count']!=20000 or
                    loaded['selected_best_step']!=20000 or loaded['strict_loading'] is not True):
                raise ValueError('Frozen source is not exact completed anchor best')
        # Independently recompute paired statistics from preserved raw complete-channel samples.
        raw=_typed_rows(directory/f'samples_{TAG}.csv.gz',True)
        typed_summary=_typed_rows(directory/f'summary_{TAG}.csv',False)
        recomputed,_=paired_rows(typed_summary,raw,1600,seed)
        paired=rows(directory/f'paired_deltas_{TAG}.csv')
        keyed={(r['contrast_id'],r['estimand'],r['gamma_points']):r for r in paired}
        for row in recomputed:
            candidate=keyed.get((row['contrast_id'],row['estimand'],row['gamma_points']),{})
            for name,value in row.items():
                if value is None:
                    if candidate.get(name) not in (None,''):raise ValueError('Paired raw/CSV mismatch: '+name)
                elif isinstance(value,(float,int)) and not isinstance(value,bool):
                    if not math.isclose(float(value),finite(candidate,name),rel_tol=1e-6,abs_tol=1e-6):
                        raise ValueError('Paired raw/CSV mismatch: '+name)
                elif str(candidate.get(name)).lower()!=str(value).lower():raise ValueError('Paired raw/CSV identity changed: '+name)
        interpretations[str(seed)]=interpret(rows(directory/f'summary_{TAG}.csv'),paired,seed)
        table=rows(directory/f'parameters_{TAG}.csv')
        if len(table)!=6 or {r['method_key'] for r in table}!=set(METHODS):raise ValueError('Six actual parameter records required')
        counts={}
        for r in table:
            full,ue,bs=(_integer(r,n) for n in ('full_parameters','ue_parameters','bs_parameters'))
            if _integer(r,'training_seed')!=seed or min(full,ue,bs)<=0 or full!=ue+bs:raise ValueError('Parameter accounting mismatch')
            expected_gate=1 if r['method_key']=='h32_static_power' else 6 if r['method_key']=='h32_gamma_power' else 0
            if _integer(r,'gate_parameters')!=expected_gate:raise ValueError('Actual gate numel differs from prescribed count')
            counts[r['method_key']]=(full,ue,bs)
            if r['method_key'] in ARMS:
                prior_counts=training_records[str(seed)][r['method_key']]['parameters']
                if prior_counts!=dict(full=full,ue=ue,bs=bs,gate=expected_gate):
                    raise ValueError('Training and evaluation actual parameter counts differ')
        base=counts['h32_control']
        for arm,extra in (('h32_static_power',1),('h32_gamma_power',6)):
            if counts[arm]!=(base[0]+extra,base[1]+extra,base[2]):raise ValueError('Gate changed more than prescribed UE parameters')
        if counts['frozen_h32_source']!=base or counts['fdma_control']!=counts['frozen_fdma_source']:
            raise ValueError('Control/source capacity changed')
        gate_rows=rows(directory/f'gates_{TAG}.csv')
        if len(gate_rows)!=36 or {(r['method_key'],finite(r,'gamma_db')) for r in gate_rows}!={(m,g) for m in METHODS for g in GAMMAS}:
            raise ValueError('Complete fixed six-node gate table required')
        for r in gate_rows:
            expected=next(x for x in typed_summary if x['method_key']==r['method_key'] and x['gamma_db']==finite(r,'gamma_db'))
            if not math.isclose(finite(r,'g_log_energy_ratio'),expected['g_log_energy_ratio_mean'],rel_tol=1e-6,abs_tol=1e-6):
                raise ValueError('Gate summary disagrees with actual forward')
            if r['method_key'] in ARMS:
                trained_gate=training_records[str(seed)][r['method_key']]['selected_gate_log_energy_ratio']
                if not math.isclose(finite(r,'g_log_energy_ratio'),float(trained_gate[str(int(finite(r,'gamma_db')))]),rel_tol=1e-6,abs_tol=1e-6):
                    raise ValueError('Evaluation gate differs from selected checkpoint inspection')
            n=1 if r['method_key']=='h32_static_power' else 6 if r['method_key']=='h32_gamma_power' else 0
            theta=json.loads(r['theta'])
            if len(theta)!=n or any(not math.isfinite(float(v)) for v in theta):raise ValueError('Invalid trained gate parameter record')
        interpretations[str(seed)]['gate_parameters']=gate_rows
        parameters[str(seed)]=table;sources[str(seed)]=directory;manifests[str(seed)]=(manifest,digest)
    parent=_safe_path(folder)
    if not parent.is_dir():raise FileNotFoundError(parent)
    output=parent/('export_'+uuid.uuid4().hex);output.mkdir(exist_ok=False)
    for seed in SEEDS:
        source=sources[str(seed)];manifest,digest=manifests[str(seed)]
        for prefix in ('summary','paired_deltas','parameters','gates'):
            name=f'{prefix}_{TAG}.csv'
            actual=_copy_small(source/name,output/f'seed{seed}_{name}',manifest['output_sha256'][name])
            source_files[str(source/name)]=actual
        _copy_small(source/'manifest.json',output/f'seed{seed}_evaluation_manifest.json',digest)
        source_files[str(source/'manifest.json')]=digest
    save_json(output/'comparison_summary.json',dict(per_seed=interpretations,training_seeds_pooled=False))
    save_json(output/'training_trends.json',curves)
    save_json(output/'source_evidence.json',dict(source_records=source_records,training_records=training_records,
        source_files=source_files,report_source_sha256=sha256(Path(__file__)),
        plots_generated=bool(make_plots),weights_loaded_by_report=False))
    learning,trends,diagnostics=[],[],[]
    for seed in SEEDS:
        for arm in ARMS:
            item=curves['per_seed'][str(seed)][arm]
            learning.extend(dict(training_seed=seed,method_key=arm,step=p['step'],mean_all=p['mean_all'],
                **{f'gamma{g}_rate':p['rates'][str(g)] for g in GAMMAS}) for p in item['points'])
            trends.append(dict(training_seed=seed,method_key=arm,best_step=item['best_step'],
                best_mean_all=item['best_mean_all'],**item['last_5000_updates'],convergence_established=False))
        for row in interpretations[str(seed)]['all_rates']:
            names=('g_log_energy_ratio_mean','exclusive_energy_mean_per_ue_mean','shared_energy_mean_per_ue_mean',
                'exclusive_energy_per_active_symbol_mean','shared_energy_per_active_symbol_mean',
                'recovered_mse_active_mean','corrected_mse_active_mean','proxy_mse_active_mean',
                'recovered_mse_exclusive_mean','proxy_mse_exclusive_mean','recovered_mse_shared_mean',
                'proxy_mse_shared_mean','downlink_desired_mean_mean','downlink_interference_mean_mean')
            diagnostics.append(dict(training_seed=seed,method_key=row['method_key'],gamma_db=row['gamma_db'],
                                    **{name:row[name] for name in names}))
    _write_csv(output/'validation_mean_all_all_points.csv',learning)
    _write_csv(output/'training_last5000_trends.csv',trends)
    _write_csv(output/'receiver_energy_diagnostics_summary.csv',diagnostics)
    if make_plots:_figures(output,interpretations,curves)
    english,chinese=result_paragraphs(interpretations,curves)
    for name,paragraphs in (('results_draft_en.md',english),('results_draft_zh.md',chinese)):
        with (output/name).open('x',encoding='utf-8') as stream:stream.write('\n\n'.join(paragraphs)+'\n')
    for path,digest in source_files.items():
        if sha256(_safe_path(path,file=True))!=digest:raise ValueError('Report input changed during export')
    files=sorted(p for p in output.iterdir() if p.is_file())
    return dict(export_directory=str(output),per_seed=interpretations,training_trends=curves,parameters=parameters,
        training_seeds_pooled=False,files={str(p):sha256(p) for p in files},_evidence=files)
