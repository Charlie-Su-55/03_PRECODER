"""Small standard-library contracts for the bounded K8 schedule experiment."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import twc_v1_k8_schedule_screen_config as cfg


def schema_resume(arm="interleaved_universal", step=108):
    """Schema fixture, explicitly not a numerical/model checkpoint."""
    return dict(format=cfg.CHECKPOINT_FORMAT, arm=arm, seed=cfg.SEED,
        step=step, cumulative_updates=cfg.SOURCE_UPDATES+step,
        recipe=cfg.recipe(arm), config_fingerprint=cfg.fingerprint(arm),
        source=cfg.source_identity(arm), codec_model={"schema": "only"},
        optimizer={"schema": "only"}, scheduler={"last_epoch": step},
        rng={name: "schema only" for name in ("python", "numpy", "torch_cpu", "torch_cuda")},
        data_protocol={"schema": "only"}, ce_identity={"schema": "only"},
        code_sha256={"test-fixture.py": "a"*64})


class ScheduleContracts(unittest.TestCase):
    def test_both_universal_orders_cover_identical_full_joint_grid(self):
        old = cfg.schedule_counts("blocked_universal")
        new = cfg.schedule_counts("interleaved_universal")
        self.assertEqual(old, new)
        self.assertEqual(len(old), 108)
        self.assertEqual({r["count"] for r in old}, {28})
        self.assertEqual(sum(r["count"] for r in old), cfg.UPDATES)
        for arm in cfg.ARMS[:2]:
            # Check all28 cycles, not just a full-run count hiding collisions.
            for start in range(1, cfg.UPDATES+1, cfg.CYCLE_UPDATES):
                points = [cfg.schedule(arm, s) for s in range(start, start+108)]
                identities = {(r["group_count"], r["gamma_db"], r["pilot_energy_per_ue"],
                               tuple(r["energy_ratio"])) for r in points}
                self.assertEqual(len(identities), 108)

    def test_interleaving_breaks_blocks_without_group_gamma_aliasing(self):
        old = [cfg.schedule("blocked_universal", s)["group_count"] for s in range(1,109)]
        new = [cfg.schedule("interleaved_universal", s)["group_count"] for s in range(1,109)]
        self.assertGreater(sum(a!=b for a,b in zip(new,new[1:])),
                           sum(a!=b for a,b in zip(old,old[1:])))
        for gamma in cfg.GAMMAS:
            groups = {cfg.schedule("interleaved_universal", s)["group_count"]
                      for s in range(1,109)
                      if cfg.schedule("interleaved_universal", s)["gamma_db"] == gamma}
            self.assertEqual(groups, {1,2,4})

    def test_fixed_g4_has_threefold_exposure_but_same_total_budget(self):
        fixed = cfg.schedule_counts("specialist_g4")
        universal = [r for r in cfg.schedule_counts("blocked_universal") if r["group_count"] == 4]
        self.assertEqual(len(fixed), 36)
        self.assertEqual({r["count"] for r in fixed}, {84})
        self.assertEqual([dict(r,count=r["count"]*3) for r in universal], fixed)
        self.assertEqual(sum(r["count"] for r in fixed), 3024)
        fdma = cfg.schedule_counts("fdma_control")
        self.assertEqual(len(fdma), 18)
        self.assertEqual({r["count"] for r in fdma}, {168})

    def test_all_arms_share_global_data_gamma_energy_and_noise_addresses(self):
        for step in (1,2,18,108,756,3024):
            rows = [cfg.schedule(a,step) for a in cfg.ARMS]
            for key in ("local_step","global_step","gamma_db","pilot_energy_per_ue"):
                self.assertEqual(len({r[key] for r in rows}), 1)
            self.assertEqual(rows[0]["global_step"], 10000+step)
            self.assertEqual(cfg.training_noise_seed(step), cfg.SEED*10_000_000+10000+step)
        self.assertEqual(cfg.schedule("blocked_universal",1)["group_count"],
                         (1,2,4)[(10000//18)%3])
        for invalid in (0,3025,True,1.0):
            with self.subTest(step=invalid), self.assertRaises(ValueError):
                cfg.schedule("blocked_universal",invalid)

    def test_pinned_sources_distinguish_same_hybrid_start_from_fdma(self):
        shared = cfg.source_identity("blocked_universal")
        self.assertEqual(shared,cfg.source_identity("interleaved_universal"))
        self.assertEqual(shared,cfg.source_identity("specialist_g4"))
        fdma = cfg.source_identity("fdma_control")
        self.assertNotEqual(shared["sha256"],fdma["sha256"])
        self.assertEqual(shared["step"],10000)
        self.assertEqual(shared["seed"],20261107)
        self.assertIn("seed_20261107_jointv1",shared["path"])
        self.assertEqual(len({cfg.fingerprint(a) for a in cfg.ARMS}),4)

    def test_validation_addresses_are_common_fresh_and_noise_keys_disjoint(self):
        self.assertEqual(cfg.VALIDATION_BATCHES*cfg.BATCH_SIZE,256)
        self.assertGreaterEqual(cfg.VALIDATION_START_INDEX,4)
        seeds = [cfg.evaluation_noise_seed(b,g,k)
                 for b in range(64) for g in range(6) for k in ("pilot","data")]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertFalse(set(seeds) & {cfg.training_noise_seed(s) for s in range(1,3025)})
        for args in ((-1,0,"pilot"),(64,0,"pilot"),(0,6,"pilot"),(0,0,"other")):
            with self.subTest(args=args),self.assertRaises(ValueError):cfg.evaluation_noise_seed(*args)

    def test_resume_json_roundtrip_passes_but_identity_count_rng_or_code_changes_fail(self):
        payload=schema_resume()
        actual=json.loads(json.dumps(payload))
        self.assertEqual(cfg.validate_resume_metadata(actual,"interleaved_universal",
                         code_sha256=payload["code_sha256"]),108)
        for kind in ("arm","seed","source","recipe","fingerprint","count","step","rng","scheduler","optimizer","code"):
            bad=deepcopy(actual)
            if kind=="arm":bad["arm"]="blocked_universal"
            elif kind=="seed":bad["seed"]+=1
            elif kind=="source":bad["source"]["sha256"]="0"*64
            elif kind=="recipe":bad["recipe"]["learning_rate"]=1e-3
            elif kind=="fingerprint":bad["config_fingerprint"]="0"*64
            elif kind=="count":bad["cumulative_updates"]=108
            elif kind=="step":bad["step"]=3025
            elif kind=="rng":bad["rng"].pop("torch_cuda")
            elif kind=="scheduler":bad["scheduler"]["last_epoch"]=0
            elif kind=="optimizer":bad.pop("optimizer")
            else:bad["code_sha256"]["test-fixture.py"]="b"*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                cfg.validate_resume_metadata(bad,"interleaved_universal",code_sha256=payload["code_sha256"])

    def test_output_protection_and_standard_library_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"new_run_root"
            path=cfg.prepare_output(root/"arm",root=root)
            with self.assertRaises(FileExistsError):cfg.prepare_output(path,root=root)
            with self.assertRaises(FileNotFoundError):cfg.prepare_output(path,resume=True,root=root)
            with self.assertRaises(ValueError):cfg.prepare_output(root,root=root)
            with self.assertRaises(ValueError):cfg.prepare_output(root.parent/"outside",root=root)
            self.assertEqual(list(path.iterdir()),[])
        script=("import sys; from configs import twc_v1_k8_schedule_screen_config as c; "
                "assert not any(n=='torch' or n.startswith('sionna') for n in sys.modules); "
                "assert c.UPDATES==3024; print(c.fingerprint('blocked_universal'))")
        result=subprocess.run([sys.executable,"-B","-S","-c",script],cwd=ROOT,
                              capture_output=True,text=True,check=True)
        self.assertEqual(len(result.stdout.strip()),64)


class EvaluationContracts(unittest.TestCase):
    """Synthetic complete-channel samples, never checkpoint/model substitutes."""
    def rows(self, count=8, delta=None):
        import evaluate_twc_v1_k8_schedule_screen as evaluator
        result=[]
        for arm,groups,policy,gamma in evaluator.conditions():
            for i in range(count):
                difference=delta(gamma,i) if delta and arm=='interleaved_universal' else 0.
                result.append(dict(arm=arm,group_count=groups,energy_policy=policy,gamma_db=gamma,
                    seed=20261107,checkpoint_step=3024,cumulative_updates=13024,
                    sample_index=i,batch_index=i//4,pilot_energy_per_ue=16.,downlink_snr_db=25.,
                    channel_fingerprint=str(i//4).zfill(64),dl_nmse=.1,
                    sum_rate_bps_hz=10+i+difference,energy_max_abs_error=0.,rzf_power_max_abs_error=0.))
        return result

    def selected(self, rows, scope='low_0_5_10', count=8):
        import evaluate_twc_v1_k8_schedule_screen as evaluator
        return next(r for r in evaluator.paired_statistics(rows,count=count)
            if r['contrast']=='interleaved_minus_blocked_G4_hybrid_1_3'
            and r['metric']=='sum_rate_bps_hz' and r['estimand']==scope)

    def test_complete_conditions_and_all_signed_negative_pairs_preserved(self):
        import evaluate_twc_v1_k8_schedule_screen as evaluator
        rows=self.rows(delta=lambda gamma,i:-.5)
        summary,_=evaluator.summarize_samples(rows,count=8)
        paired=evaluator.paired_statistics(rows,count=8)
        self.assertEqual((len(rows),len(summary),len(paired)),(720,90,384))
        self.assertEqual(sum(r['arm']=='fdma_control' for r in summary),6)
        selected=self.selected(rows,'all_six')
        self.assertEqual((selected['sample_mean'],selected['batch_mean']),(-.5,-.5))
        self.assertEqual((selected['sample_ci95_low'],selected['sample_ci95_high']),(-.5,-.5))
        self.assertTrue(selected['primary'])

    def test_cross_gamma_average_uses_within_channel_covariance(self):
        rows=self.rows(delta=lambda gamma,i:(i-3.5)*(1 if gamma==0 else -1 if gamma==5 else 0))
        low=self.selected(rows)
        self.assertEqual((low['sample_mean'],low['sample_se'],low['batch_se']),(0.,0.,0.))
        self.assertGreater(self.selected(rows,'0')['sample_se'],0)
        self.assertGreater(self.selected(rows,'5')['sample_se'],0)

    def test_batch_clusters_and_one_batch_uncertainty_are_distinct(self):
        rows=self.rows(delta=lambda gamma,i:4.*(i//4))
        result=self.selected(rows)
        self.assertEqual(result['batch_mean'],2.)
        self.assertAlmostEqual(result['batch_se'],2.)
        self.assertGreater(result['batch_se'],result['sample_se'])
        one=self.selected(self.rows(count=4,delta=lambda gamma,i:float(i)),count=4)
        self.assertFalse(one['batch_ci_estimable'])
        self.assertIsNone(one['batch_se'])
        self.assertIsNone(one['batch_ci95_low'])
        self.assertIsNone(one['batch_ci95_high'])
        self.assertGreater(one['sample_se'],0)

    def test_missing_duplicate_nonfinite_identity_and_physical_failures_rejected(self):
        import evaluate_twc_v1_k8_schedule_screen as evaluator
        original=self.rows()
        for kind in ('missing','duplicate','nonfinite','wrong_seed','wrong_step','wrong_batch',
                     'different_channel','energy','power'):
            bad=deepcopy(original)
            if kind=='missing':bad.pop()
            elif kind=='duplicate':bad.append(deepcopy(bad[0]))
            elif kind=='nonfinite':bad[0]['dl_nmse']=float('nan')
            elif kind=='wrong_seed':bad[0]['seed']+=1
            elif kind=='wrong_step':bad[0]['cumulative_updates']=3024
            elif kind=='wrong_batch':bad[0]['batch_index']=9
            elif kind=='different_channel':bad[1]['channel_fingerprint']='f'*64
            elif kind=='energy':bad[0]['energy_max_abs_error']=1.
            else:bad[0]['rzf_power_max_abs_error']=1.
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                evaluator.summarize_samples(bad,count=8)

    def test_unfavorable_group_cannot_be_omitted_or_relabelled(self):
        import evaluate_twc_v1_k8_schedule_screen as evaluator
        rows=self.rows()
        bad=[r for r in rows if not(r['arm']=='interleaved_universal' and r['group_count']==1)]
        with self.assertRaises(ValueError):evaluator.paired_statistics(bad,count=8)
        rows[0]['energy_policy']='a_more_favorable_policy'
        with self.assertRaises(ValueError):evaluator.paired_statistics(rows,count=8)


class ControllerContracts(unittest.TestCase):
    def test_entry_points_help_and_dry_run_are_stdlib_only_and_do_not_write(self):
        # Guard the real CLI imports and deny any actual checkpoint reads.
        guard = r'''
import builtins, json, pathlib, runpy, sys
from configs import twc_v1_k8_schedule_screen_config as cfg
before = cfg.OUTPUT_ROOT.exists()
original_import, original_open = builtins.__import__, builtins.open
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch','numpy','sionna'):
        raise AssertionError('Heavy dependency imported during plan/help: '+name)
    return original_import(name,*args,**kwargs)
def guarded_open(path, *args, **kwargs):
    if str(path).lower().endswith(('.pt','.pth','.ckpt')):
        raise AssertionError('Checkpoint read during plan/help: '+str(path))
    return original_open(path,*args,**kwargs)
builtins.__import__, builtins.open = guarded_import, guarded_open
sys.argv = json.loads(sys.argv[1])
try:
    runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as error:
    if error.code not in (None,0): raise
assert cfg.OUTPUT_ROOT.exists() == before
assert not any(n.split('.')[0] in ('torch','numpy','sionna') for n in sys.modules)
'''
        jobs = [
            ['run_twc_v1_k8_schedule_screen.py'],
            ['run_twc_v1_k8_schedule_screen.py','--dry-run'],
            ['run_twc_v1_k8_schedule_screen.py','--summary'],
            ['evaluate_twc_v1_k8_schedule_screen.py'],
            ['evaluate_twc_v1_k8_schedule_screen.py','--dry-run'],
            ['train_twc_v1_k8_schedule_screen.py'],
        ] + [['train_twc_v1_k8_schedule_screen.py','--arm',arm,'--dry-run'] for arm in cfg.ARMS]
        for args in jobs:
            with self.subTest(args=args):
                done=subprocess.run([sys.executable,'-B','-S','-c',guard,json.dumps(args)],
                                    cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(done.returncode,0,done.stdout+'\n'+done.stderr)
                self.assertTrue(done.stdout.strip())

    def test_actual_controller_smoke_resumes_within3024_and_skips_completed(self):
        from unittest.mock import patch
        import run_twc_v1_k8_schedule_screen as entry
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            arm='interleaved_universal'
            destination=root/'training'/arm
            commands=[]
            def child(folder,name,command):
                commands.append(command)
                destination.mkdir(parents=True,exist_ok=True)
                if command[-2:]==['--stop-after','108']:
                    self.assertFalse((destination/'checkpoint_latest.pt').exists())
                    state,step='paused',108
                else:
                    self.assertEqual(command[-2:],['--resume','--train'])
                    self.assertEqual(entry.read_json(destination/'manifest.json')['step'],108)
                    state,step='completed',3024
                # Only a subprocess-control sentinel; no model/checkpoint is fabricated.
                (destination/'checkpoint_latest.pt').write_bytes(b'control-flow fixture, not a checkpoint')
                (destination/'manifest.json').write_text(json.dumps(dict(status=state,step=step,
                    cumulative_updates=10000+step)),encoding='utf-8')
                log=root/(name+'.log');log.write_text('synthetic subprocess receipt',encoding='utf-8')
                return log
            def verified(_arm):
                self.assertEqual(_arm,arm)
                record=entry.read_json(destination/'manifest.json')
                self.assertEqual((record['step'],record['cumulative_updates']),(3024,13024))
                return dict(report=record,_evidence=[])
            with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(entry,'_child',side_effect=child),\
                    patch.object(entry,'verify_training',side_effect=verified):
                result=entry.run_arm(root,arm)
                self.assertEqual(len(commands),2)
                self.assertEqual(commands[0][-2:],['--stop-after','108'])
                self.assertEqual(commands[1][-2:],['--resume','--train'])
                self.assertEqual(result['report']['step'],3024)
                entry.run_arm(root,arm)
                self.assertEqual(len(commands),2)
                (destination/'manifest.json').write_text(json.dumps(dict(status='paused',step=108)),encoding='utf-8')
                entry.run_arm(root,arm)
                self.assertEqual(len(commands),3)
                self.assertEqual(commands[-1][-2:],['--resume','--train'])

    def test_unowned_or_incomplete_output_is_preserved_and_never_reinitialized(self):
        from unittest.mock import patch,Mock
        import run_twc_v1_k8_schedule_screen as entry
        for case in ('directory_only','checkpoint_without_manifest'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);destination=root/'training'/'specialist_g4'
                destination.mkdir(parents=True)
                if case=='checkpoint_without_manifest':
                    (destination/'checkpoint_latest.pt').write_bytes(b'control-flow fixture only')
                before={p.name:p.read_bytes() for p in destination.iterdir()}
                child,verified=Mock(),Mock()
                with patch.object(cfg,'OUTPUT_ROOT',root),patch.object(entry,'_child',child),\
                        patch.object(entry,'verify_training',verified):
                    with self.assertRaises((FileExistsError,FileNotFoundError)):
                        entry.run_arm(root,'specialist_g4')
                child.assert_not_called();verified.assert_not_called()
                self.assertEqual(before,{p.name:p.read_bytes() for p in destination.iterdir()})

    def test_failed_smoke_stops_before_continuation_or_completion_inspection(self):
        from unittest.mock import patch,Mock
        import run_twc_v1_k8_schedule_screen as entry
        with tempfile.TemporaryDirectory() as directory:
            child=Mock(side_effect=RuntimeError('synthetic subprocess failure'))
            verified=Mock()
            with patch.object(cfg,'OUTPUT_ROOT',Path(directory)),patch.object(entry,'_child',child),\
                    patch.object(entry,'verify_training',verified):
                with self.assertRaises(RuntimeError):entry.run_arm(Path(directory),'fdma_control')
            self.assertEqual(child.call_count,1)
            self.assertEqual(child.call_args.args[-1][-2:],['--stop-after','108'])
            verified.assert_not_called()


class ResumeReceiptRegression(unittest.TestCase):
    def test_another_interruption_before_next_save_retains_valid_latest_identity(self):
        import train_twc_v1_k8_schedule_screen as train
        initial=train.training_manifest('blocked_universal',0,data_protocol={'data':'schema'},
            ce={'ce':'schema'},code={'sha256':{'code':'schema'}})
        self.assertNotIn('checkpoint_sha256',initial)
        digest='a'*64
        resumed=train.training_manifest('blocked_universal',108,data_protocol={'data':'schema'},
            ce={'ce':'schema'},code={'sha256':{'code':'schema'}},resume_hash=digest)
        resumed.update(status='failed',error='Synthetic interruption before next save')
        roundtrip=json.loads(json.dumps(resumed))
        self.assertEqual(roundtrip['checkpoint_sha256'],digest)
        self.assertEqual(roundtrip['step'],108)
        self.assertEqual(roundtrip['cumulative_updates'],10108)
        for missing in (None,'bad'):
            with self.assertRaises(ValueError):
                train.training_manifest('blocked_universal',108,data_protocol={},ce={},code={},resume_hash=missing)


if __name__ == "__main__":
    unittest.main()
