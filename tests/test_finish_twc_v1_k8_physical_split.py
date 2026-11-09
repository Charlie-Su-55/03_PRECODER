"""Targeted CPU postprocessing regressions; no checkpoint/model/CUDA execution."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import finish_twc_v1_k8_physical_split as finish
from configs import twc_v1_k8_physical_split_config as cfg


def acceptance_fixture(directory):
    """Explicit synthetic rows, not an accepted scientific run."""
    source=directory/'old_source';source.mkdir()
    original=source/'batch_summary.json';original.write_text('{"status":"failed","do_not_rewrite":true}',encoding='utf-8')
    partial=source/'monitor_full_grid.csv';partial.write_text('old,partial\nunchanged,1\n',encoding='utf-8')
    result=dict(training={},training_code={'commit':finish.validation.TRAINING_COMMIT,'sha256':{'old.py':'a'*64}},
        postprocessing_code={'commit':'b'*40,'sha256':{'new.py':'c'*64}},frozen_files={})
    for path in (original,partial):result['frozen_files'][str(path)]=finish.sha256(path)
    progress_root=source/'training';progress_root.mkdir()
    for arm in cfg.ARMS:
        family='fdma' if cfg.is_fdma(arm) else 'g4';armdir=progress_root/arm;armdir.mkdir()
        policy='fdma' if family=='fdma' else 'hybrid_1_3'
        row=dict(energy_policy=policy,gamma_db=0.,num_samples=64,alpha=.03,sum_rate_bps_hz_mean=1.)
        if family=='fdma':row['physical_head_codeword_nmse_linear_fdma_mean']=0.
        else:
            row['physical_head_codeword_nmse_linear_key_mean']=.1
            row['physical_head_codeword_nmse_linear_shared_mean']=.2
        mean={k:v for k,v in row.items() if k!='gamma_db'}
        report=dict(shared_inputs=[{'schema_fixture':True}],summary=[row],all_six=[mean])
        monitor=armdir/'monitor.json';monitor.write_text(json.dumps(report),encoding='utf-8')
        event=dict(step=6048,update_counts=cfg.update_counts(6048),csi_nmse_linear=0.,head_nmse_linear=0.,
            rate=1.,condition=cfg.schedule(arm,6048))
        progress=armdir/'progress.jsonl';progress.write_text(json.dumps(event)+'\n',encoding='utf-8')
        for path in (monitor,progress):result['frozen_files'][str(path)]=finish.sha256(path)
        result['training'][arm]=dict(status='completed',step=6048,update_counts=cfg.update_counts(6048),
            training_input_history='d'*64,layout_transmit_history=('e' if family=='fdma' else 'f')*64,
            monitor_records={'6048':dict(path=str(monitor),sha256=finish.sha256(monitor))})
    return result,progress_root,original,partial


class FinishAggregationTests(unittest.TestCase):
    def test_actual_aggregation_path_preserves_union_for_both_monitor_tables(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            directory=Path(name);accepted,progress_root,original,partial=acceptance_fixture(directory)
            before={p:p.read_bytes() for p in (original,partial)}
            output=directory/'finish'/'aggregation';output.mkdir(parents=True)
            with patch.object(finish,'OUTPUT_ROOT',directory/'finish'):
                result=finish.write_training_aggregation(output,accepted,progress_root=progress_root)
            for filename in ('monitor_full_grid.csv','monitor_all_six.csv'):
                with (output/filename).open(newline='',encoding='utf-8') as stream:
                    reader=csv.DictReader(stream);rows=list(reader);fields=reader.fieldnames
                self.assertEqual(fields,sorted(fields));self.assertEqual(len(rows),4)
                f=next(r for r in rows if r['arm']=='fdma_control');g=next(r for r in rows if r['arm']=='g4_split')
                self.assertEqual(f['physical_head_codeword_nmse_linear_fdma_mean'],'0.0')
                self.assertEqual(f['physical_head_codeword_nmse_linear_key_mean'],'')
                self.assertEqual(g['physical_head_codeword_nmse_linear_fdma_mean'],'')
                self.assertEqual(g['physical_head_codeword_nmse_linear_key_mean'],'0.1')
                self.assertEqual(g['physical_head_codeword_nmse_linear_shared_mean'],'0.2')
            receipt=json.loads((output/'aggregation_receipt.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['additional_training_updates'],0)
            self.assertEqual(receipt['training_code']['commit'],finish.validation.TRAINING_COMMIT)
            self.assertNotEqual(receipt['training_code']['commit'],receipt['postprocessing_code']['commit'])
            self.assertEqual(result['row_counts']['monitor_full_grid'],4)
            for path,data in before.items():self.assertEqual(path.read_bytes(),data)

    def test_completed_stage_is_reused_only_with_matching_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            output=Path(name)/'finish';output.mkdir()
            source=Path(name)/'old.csv';source.write_bytes(b'read-only source')
            summary=dict(status='running',stages={});calls=[]
            def action():
                calls.append(1);return dict(training_updates_added=0,_evidence=[source])
            with patch.object(finish,'OUTPUT_ROOT',output):
                first=finish.execute_stage(summary,output,'CPU_aggregation',action)
                second=finish.execute_stage(summary,output,'CPU_aggregation',action)
                self.assertEqual(first,second);self.assertEqual(calls,[1])
                source.write_bytes(b'changed fixture source')
                with self.assertRaises(ValueError):finish.execute_stage(summary,output,'CPU_aggregation',action)
            self.assertEqual(calls,[1])

    def test_alpha_guard_reuses_no_partial_attempt_and_reaches_only_gpu_boundary(self):
        import evaluate_twc_v1_k8_physical_split_finish as evaluator
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            root=Path(name)/'finish';folder=root/'finishes/fixture';folder.mkdir(parents=True)
            attempt=root/'selection_attempt.json';selection=root/'alpha_selection.json'
            with patch.object(finish,'OUTPUT_ROOT',root),patch.object(evaluator,'SELECTION_PATH',selection),patch.object(
                    evaluator,'ATTEMPT_PATH',attempt),patch.object(finish,'gpu_sharing_snapshot',
                    side_effect=RuntimeError('unit boundary before any GPU access')):
                with self.assertRaisesRegex(RuntimeError,'unit boundary'):
                    finish.evaluate_stage(folder,'calibration')
                attempt.write_text('{"started":true}',encoding='utf-8')
                original=attempt.read_bytes()
                with self.assertRaisesRegex(RuntimeError,'automatic reselection'):
                    finish.evaluate_stage(folder,'calibration')
                self.assertEqual(attempt.read_bytes(),original)
                self.assertFalse(selection.exists())

    def test_failed_stage_keeps_attempt_and_captures_its_own_exception(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            output=Path(name)/'finish';output.mkdir();summary=dict(status='running',stages={})
            def fail():raise ValueError('Hybrid field is present after the FDMA row')
            with patch.object(finish,'OUTPUT_ROOT',output):
                with self.assertRaises(ValueError):finish.execute_stage(summary,output,'CPU_aggregation',fail)
                entry=deepcopy(summary['stages']['CPU_aggregation'])
                self.assertIn('Hybrid field',entry['error']);self.assertIn('ValueError',entry['traceback'])
                receipt=output/'new.json';receipt.write_text('{"passed":true}',encoding='utf-8')
                finish.execute_stage(summary,output,'CPU_aggregation',lambda:dict(_evidence=[receipt]))
            history=summary['stages']['CPU_aggregation']['attempt_history']
            self.assertEqual(history[0]['error'],entry['error']);self.assertEqual(len(history),1)


class FinishSummaryAndSafetyTests(unittest.TestCase):
    def test_summary_uses_failed_stage_not_newest_training_log(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            root=Path(name)/'finish';folder=root/'finishes/fixture';folder.mkdir(parents=True)
            (folder/'newest_training.log').write_text('UNRELATED_TRAINING_LOG_NOT_THE_ERROR',encoding='utf-8')
            summary=dict(status='failed',error='generic top-level failure',active_stage='CPU_aggregation',
                stages={'CPU_aggregation':dict(status='failed',error='ValueError: actual union failure',
                    traceback='Traceback\nCPU aggregation\nValueError: actual union failure'),
                    'train_last':dict(status='completed',result='not a finish stage')})
            (root/'last_finish.json').write_text(json.dumps({'directory':str(folder)}),encoding='utf-8')
            (folder/'finish_summary.json').write_text(json.dumps(summary),encoding='utf-8')
            output=io.StringIO()
            with patch.object(finish,'OUTPUT_ROOT',root),patch.object(finish,'POINTER',root/'last_finish.json'),redirect_stdout(output):
                finish.print_summary()
            text=output.getvalue()
            self.assertIn('actual union failure',text);self.assertIn('CPU_aggregation',text)
            self.assertNotIn('UNRELATED_TRAINING_LOG_NOT_THE_ERROR',text)
            self.assertIn('training_updates_added=0',text)

    def test_full_six_point_summary_keeps_core_ci_and_final_failure_line(self):
        methods=list(cfg.METHODS);rows=[]
        for method in methods:
            for policy in (('fdma',) if cfg.is_fdma(method) else ('hybrid_1_3','hybrid_1_1')):
                for mode in ('fixed','tuned'):
                    for gamma in cfg.GAMMAS:rows.append(dict(arm=method,energy_policy=policy,rzf_mode=mode,
                        gamma_db=gamma,sum_rate_bps_hz_mean=10.+gamma/100.))
        paired=[dict(contrast=name,metric='sum_rate_bps_hz',estimand=scope,sample_mean=-.1,
                     batch_ci95_low=-.2,batch_ci95_high=0.) for name in (
            'fdma_split_minus_control_fdma_fixed','g4_split_minus_control_hybrid_1_3_fixed',
            'g4_split_minus_control_hybrid_1_1_fixed','split_g4_minus_fdma_hybrid_1_3_fixed',
            'split_g4_minus_fdma_hybrid_1_1_fixed') for scope in ('low_0_5_10','all_six')]
        diagnostic=[dict(arm=arm,energy_policy=policy,estimand='all_six',metric=metric,sample_mean=.01)
            for arm in cfg.ARMS for policy in (('fdma',) if cfg.is_fdma(arm) else ('hybrid_1_3','hybrid_1_1'))
            for metric in ('physical_soft_codeword_nmse_linear_all','physical_head_codeword_nmse_linear_all')]
        accepted={'training':{arm:{'update_counts':cfg.update_counts(6048)} for arm in cfg.ARMS}}
        summary=dict(status='failed',active_stage='frozen_validation',stages={'frozen_validation':dict(status='failed',
            error='proper stage error',traceback='first\nsecond\nthird\nfinal targeted failure')},
            result_package='exact_result_package.zip',summary_path='exact_finish_summary.json')
        result=dict(summary=rows,paired=paired,receiver_diagnostics=diagnostic)
        lines=finish.summary_lines(summary,accepted,result)
        text='\n'.join(lines)
        self.assertLessEqual(len(lines),40);self.assertLessEqual(len(text.encode()),8000)
        for method in methods:self.assertIn(method+'/',text)
        self.assertIn('final targeted failure',text);self.assertIn('exact_result_package.zip',text)
        self.assertIn('physical_linear_all=',text);self.assertIn('g4_split_minus_control_hybrid_1_3_fixed',text)
        for line in lines:
            if ' fixed=' in line:self.assertEqual(len(line.split('fixed=',1)[1].split()[0].split(',')),6)

    def test_child_startup_error_reads_only_the_recorded_failing_stage_log(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            root=Path(name)/'finish';root.mkdir()
            exact=root/'calibration_failure.log';exact.write_text('ImportError: exact child startup error\n',encoding='utf-8')
            unrelated=root/'newest_train.log';unrelated.write_text('UNRELATED_TRAINING_OUTPUT\n',encoding='utf-8')
            summary=dict(status='failed',active_stage='alpha_calibration',stages={
                'alpha_calibration':dict(status='failed',error='calibration exit1',traceback='parent traceback',
                    log_path=str(exact),child_traceback=None)})
            with patch.object(finish,'OUTPUT_ROOT',root):
                tail=finish.exact_failure_log_tail(summary)
            self.assertEqual(tail,['ImportError: exact child startup error'])
            text='\n'.join(finish.summary_lines(summary,stage_log_tail=tail))
            self.assertIn('exact child startup error',text);self.assertNotIn('UNRELATED_TRAINING_OUTPUT',text)

    def test_only_frozen_subprocess_modes_and_no_training_update_calls(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            root=Path(name)
            with patch.object(finish,'OUTPUT_ROOT',root):
                for stage in ('calibration','evaluation'):
                    command=finish.frozen_command(stage,root/stage)
                    self.assertEqual(Path(command[3]).name,'evaluate_twc_v1_k8_physical_split_finish.py')
                    self.assertNotIn('--train',command);self.assertNotIn('--resume',command)
                for stage in ('train','108','cleanup','preflight'):
                    with self.assertRaises(ValueError):finish.frozen_command(stage,root/'invalid')
        # A focused architectural regression required by this repair: new
        # postprocessing code may inspect optimizer state, never update it.
        for filename in ('finish_twc_v1_k8_physical_split.py','physical_split_finish_validation.py',
                         'evaluate_twc_v1_k8_physical_split_finish.py'):
            tree=ast.parse((ROOT/filename).read_text(encoding='utf-8'))
            forbidden=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute) and node.func.attr in ('step','backward','zero_grad','train')]
            self.assertEqual(forbidden,[],filename)

    def test_finish_lease_receipts_preserve_original_locks_and_never_unlink(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            root=Path(name);old_lock=root/'original_running.lock';old_lock.write_bytes(b'preserve old lock')
            output=root/'finish';output.mkdir()
            with patch.object(finish,'OUTPUT_ROOT',output):
                with finish._finish_lease_record(False):
                    self.assertEqual(json.loads((output/'finish_lease.json').read_text())['status'],'active')
                    with self.assertRaises(RuntimeError):
                        with finish._finish_lease_record(True):pass
                self.assertTrue((output/'finish_lease.json').exists())
                with finish._finish_lease_record(True):pass
                self.assertEqual(json.loads((output/'finish_lease.json').read_text())['status'],'inactive')
                self.assertEqual(len(list((output/'lease_history').iterdir())),1)
            self.assertEqual(old_lock.read_bytes(),b'preserve old lock')

    def test_default_help_dryrun_and_summary_are_stdlib_only_and_write_nothing(self):
        before=finish.OUTPUT_ROOT.exists()
        for filename in ('finish_twc_v1_k8_physical_split.py','evaluate_twc_v1_k8_physical_split_finish.py'):
            source=("import sys,runpy\nsys.argv=['tool','--dry-run']\ntry:\n"
                " runpy.run_path("+repr(filename)+",run_name='__main__')\nexcept SystemExit as e:\n"
                " assert e.code in (None,0)\nassert not any(m in sys.modules for m in ('torch','numpy','sionna'))\n")
            result=subprocess.run([sys.executable,'-B','-S','-c',source],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(finish.OUTPUT_ROOT.exists(),before)


if __name__=='__main__':unittest.main()
