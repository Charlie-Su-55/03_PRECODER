"""Metadata-only contract checks for the new joint-training comparison."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

from configs import twc_v1_k8_ue_bs_rate_config as cfg
from configs import twc_v1_k8_ul_adapt_config as old


class JointRateConfigTests(unittest.TestCase):
    def test_fixed_budget_and_architecture_are_explicit(self):
        recipe=cfg.study_recipe()
        self.assertEqual(cfg.METHODS,('source','bs_only','joint'))
        self.assertEqual(cfg.TRAIN_METHODS,('joint',))
        self.assertEqual((cfg.UPDATES,cfg.MICRO_BATCH,cfg.ACCUMULATION),(10000,4,8))
        self.assertEqual(recipe['physical'],old.study_recipe()['physical'])
        self.assertFalse(recipe['architecture']['UE_SNR_input'])
        self.assertFalse(recipe['architecture']['user_projection'])
        self.assertEqual(recipe['archived_controls']['new_optimizer_updates'],0)
        self.assertTrue(recipe['prior']['benchmark_policy_differs'])
        self.assertIn('not UE-gradient-only',recipe['prior']['attribution'])
        self.assertEqual((cfg.ALPHA,cfg.DL_SNR),(.03,25.))

    def test_legacy_bytes_and_exact_archived_controls_are_pinned(self):
        self.assertEqual(len(cfg.LEGACY_CORE_FILES),61)
        self.assertEqual(len(cfg.CORE_FILES),68)
        self.assertEqual(set(cfg.CONTROL_CORE['sha256']),set(cfg.LEGACY_CORE_FILES))
        for name,digest in cfg.CONTROL_CORE['sha256'].items():
            self.assertEqual(hashlib.sha256((cfg.ROOT/name).read_bytes()).hexdigest(),digest,name)
        for arm in cfg.ARMS:
            pins=cfg.CONTROL_PINS[arm]
            self.assertEqual(pins['step'],10000)
            self.assertEqual(pins['training_trace'],cfg.TRAINING_TRACE_PIN)
            self.assertEqual(pins['counters']['EMA_commits'],0)
            self.assertEqual(set(cfg.CONTROL_FILE_SHA256S[arm]),
                {'checkpoint_latest.pt','manifest.json','terminal_receipt.json',
                 'initialization.json','progress.jsonl'})

    def test_full_training_schedule_exactly_replays_archived_control(self):
        count={str(g):0 for g in cfg.GAMMAS}
        for index in range(1,cfg.UPDATES*cfg.ACCUMULATION+1):
            scheduled=cfg.gamma_schedule(index)
            self.assertEqual(scheduled,old.gamma_schedule(index))
            for gamma in scheduled:count[str(gamma)]+=1
        self.assertEqual(count,cfg.gamma_counts(cfg.UPDATES))
        self.assertEqual(sum(count.values()),320000)
        for start in (0,3,9996):
            local={g:0 for g in cfg.GAMMAS}
            for update in range(start+1,start+4):
                for micro in range(cfg.ACCUMULATION):
                    for gamma in cfg.gamma_schedule(cfg.micro_index(update,micro)):local[gamma]+=1
            self.assertEqual(set(local.values()),{16})

    def test_training_random_stream_is_replayed_evaluation_is_fresh(self):
        self.assertEqual(cfg.TRAIN_DATA_NAMESPACE,old.DATA_NAMESPACE)
        for index in (1,8,80000):
            for kind in ('noise','polarity'):
                self.assertEqual(cfg.noise_seed('codec_train',index,kind),old.noise_seed('codec_train',index,kind))
        for role in ('monitor','final'):
            self.assertNotEqual(cfg.noise_seed(role,0),old.noise_seed(role,0))
        self.assertEqual(cfg.ROLE_RANGES['final'],[4096,4351])
        self.assertLess(cfg.ROLE_RANGES['monitor'][1],cfg.ROLE_RANGES['final'][0])
        for args in (('final',-1),('final',1536),('test',0),('monitor',192)):
            with self.assertRaises(ValueError):cfg.noise_seed(*args)

    def test_counters_preserve_inherited_ema_and_reject_partial_resume(self):
        step=100
        source={'base_ema_updates':20000};frozen={'ce':'a'*64};core={'sha256':{}}
        protocol={'namespace':'old replay'}
        payload=dict(format=cfg.NAMESPACE,arm='fdma',method='joint',step=step,
            recipe=cfg.recipe('fdma'),fingerprint=cfg.fingerprint('fdma'),core=core,
            data_protocol=protocol,source_identity=source,frozen_identity=frozen,
            model={'a':1},statistics={'ema_updates':20100},optimizer={'a':1},
            scheduler={'last_epoch':100},rng={'fixture':True},counters=cfg.counters(step,20000))
        kwargs=dict(core=core,data_protocol=protocol,source_identity=source,
            frozen_identity=frozen,base_ema_updates=20000)
        self.assertEqual(cfg.validate_resume(payload,'fdma',**kwargs),100)
        self.assertEqual(payload['counters']['EMA_commits'],100)
        self.assertEqual(payload['counters']['total_EMA_updates'],20100)
        for field,value in (('step',101),('method','bs_only'),
                ('counters',cfg.counters(100,0)),('fingerprint','b'*64)):
            changed=copy.deepcopy(payload);changed[field]=value
            with self.assertRaises(ValueError):cfg.validate_resume(changed,'fdma',**kwargs)

    def test_stdlib_only_recipe_import_has_no_sampling_or_writes(self):
        code=('import sys; from configs import twc_v1_k8_ue_bs_rate_config as c; '
            'assert "torch" not in sys.modules and "sionna" not in sys.modules; '
            'assert c.study_recipe()["final"]["test_used"] is False; print(c.NAMESPACE)')
        result=subprocess.run([sys.executable,'-B','-S','-c',code],cwd=cfg.ROOT,
            capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),cfg.NAMESPACE)


if __name__=='__main__':unittest.main()
