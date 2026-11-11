"""Real file/hash selection locks; no Torch, source weights or GPU needed."""
import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_icc_alpha_config as cfg
import icc_alpha_runtime as runtime
import twc_v1_k8_icc_alpha_results as results


class AlphaSelectionLockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'alpha_outputs'
        pins=dict(cfg.SOURCE_CHECKPOINT_SHA256)
        self.artifacts={}
        for arm in cfg.ARMS:
            p=self.root/(arm+'_original.bin');p.write_bytes(('unchanged-source-'+arm).encode())
            self.artifacts[arm]=p;pins[arm]=runtime.sha256(p)
        for patcher in (mock.patch.object(runtime,'OUTPUT_ROOT',self.output),
                        mock.patch.object(cfg,'OUTPUT_ROOT',self.output),
                        mock.patch.object(cfg,'CALIBRATION_BATCHES',1),
                        mock.patch.object(cfg,'SOURCE_CHECKPOINT_SHA256',pins)):
            patcher.start();self.addCleanup(patcher.stop)
        self.data=runtime.make_data('cpu')
        self.directory=self.output/'calibration'/'attempt_fixture'
        self.directory.mkdir(parents=True)
        self.build(self.directory)

    def build(self,directory):
        rows=[];fingerprints=[];records={}
        for arm in cfg.ARMS:
            optimum=.05 if arm=='fdma' else .01
            p=self.artifacts[arm]
            records[arm]=dict(kind='source',arm=arm,model_commit=cfg.SOURCE_COMMIT,
                checkpoint_sha256=cfg.SOURCE_CHECKPOINT_SHA256[arm],core=cfg.SOURCE_CORE,step=20000,
                frozen_files=[dict(path=str(p),sha256=runtime.sha256(p))])
            for gi,gamma in enumerate(cfg.GAMMAS):
                fingerprints.append(dict(arm=arm,batch_index=0,gamma_db=gamma,
                    address=self.data.address('validation',0),
                    fingerprint={k:format(gi+1,'064x') for k in ('H_dl','H_ul','standard_noise','pilot_polarity')}))
                for alpha in cfg.ALPHA_GRID:
                    for i in range(cfg.MICRO_BATCH):
                        rows.append(dict(arm=arm,alpha=alpha,gamma_db=gamma,sample_index=i,
                            batch_index=0,within_batch=i,sum_rate_bps_hz=10.+gamma/20+i/100-(alpha-optimum)**2))
        with gzip.open(directory/'samples.csv.gz','wt',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        chosen=results.select_alphas(rows,samples=cfg.MICRO_BATCH,batch_size=cfg.MICRO_BATCH)
        candidate=dict(chosen,core=runtime.core_identity(),data_protocol=self.data.protocol,
                       recipe_fingerprint=cfg.fingerprint())
        runtime.atomic_json(directory/'selection_candidate.json',candidate,exclusive=True)
        with (directory/'scores.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(chosen['scores'][0]));writer.writeheader();writer.writerows(chosen['scores'])
        runtime.atomic_json(directory/'input_fingerprints.json',fingerprints,exclusive=True)
        manifest=dict(status='passed',phase='calibration',core=runtime.core_identity(),
            recipe_fingerprint=cfg.fingerprint(),alpha_grid=list(cfg.ALPHA_GRID),gamma_db=list(cfg.GAMMAS),
            data_protocol=self.data.protocol,batch_size=cfg.MICRO_BATCH,n_batches=1,
            samples_per_condition=cfg.MICRO_BATCH,source_records=records,
            files={name:runtime.sha256(directory/name) for name in (
                'samples.csv.gz','scores.csv','selection_candidate.json','input_fingerprints.json')})
        runtime.atomic_json(directory/'manifest.json',manifest,exclusive=True)

    def test_lock_uses_only_calibration_is_repeatable_and_preserves_sources(self):
        before={a:p.read_bytes() for a,p in self.artifacts.items()}
        record=runtime.lock_selection(self.directory)
        self.assertEqual(record['selected'],{'fdma':.05,'hybrid_g4':.01})
        path=self.output/'selection.json';raw=path.read_bytes()
        self.assertEqual(runtime.load_selection(path),record)
        self.assertEqual(runtime.lock_selection(self.directory),record)
        self.assertEqual(path.read_bytes(),raw)
        self.assertFalse((self.output/'evaluation').exists())
        self.assertEqual(before,{a:p.read_bytes() for a,p in self.artifacts.items()})

    def test_changed_calibration_file_invalidates_existing_lock_without_rewriting(self):
        runtime.lock_selection(self.directory)
        path=self.output/'selection.json';before=path.read_bytes()
        with (self.directory/'samples.csv.gz').open('ab') as stream:stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'Calibration evidence changed'):
            runtime.load_selection()
        self.assertEqual(path.read_bytes(),before)

    def test_rehashed_but_forged_selected_alpha_is_rejected(self):
        record=runtime.lock_selection(self.directory)
        record['selected']['fdma']=.3
        record['record_id']=cfg.digest({k:v for k,v in record.items() if k!='record_id'})
        (self.output/'selection.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'Locked alpha/calibration identity changed'):
            runtime.load_selection()

    def test_existing_final_attempt_forbids_reselection_when_lock_missing(self):
        final=self.output/'evaluation'/'final'/'attempt_partial';final.mkdir(parents=True)
        (final/'unknown_partial.txt').write_text('preserve partial final evidence')
        with self.assertRaisesRegex(ValueError,'reselection forbidden'):
            runtime.lock_selection(self.directory)
        self.assertFalse((self.output/'selection.json').exists())
        self.assertEqual((final/'unknown_partial.txt').read_text(),'preserve partial final evidence')

    def test_different_calibration_attempt_cannot_replace_valid_lock(self):
        runtime.lock_selection(self.directory)
        other=self.output/'calibration'/'attempt_other';other.mkdir();self.build(other)
        with self.assertRaisesRegex(ValueError,'different alpha selection'):
            runtime.lock_selection(other)

    def test_final_addresses_cannot_be_relabelled_as_calibration(self):
        path=self.directory/'input_fingerprints.json';rows=runtime.read_json(path)
        for row in rows:row['address']=self.data.address('validation',cfg.ROLE_RANGES['final'][0])
        path.write_text(json.dumps(rows))
        manifest=runtime.read_json(self.directory/'manifest.json')
        manifest['files'][path.name]=runtime.sha256(path)
        (self.directory/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'only its declared validation addresses'):
            runtime.lock_selection(self.directory)

    def test_source_mutation_blocks_lock_and_unsafe_output_paths_are_rejected(self):
        self.artifacts['fdma'].write_bytes(b'changed-original')
        with self.assertRaisesRegex(ValueError,'source artifact changed'):
            runtime.lock_selection(self.directory)
        with self.assertRaisesRegex(ValueError,'Owned alpha'):
            runtime.atomic_json(self.root/'outside.json',{'forbidden':True})
        with self.assertRaisesRegex(ValueError,'own immutable'):
            runtime.load_selection(self.output/'other_selection.json')

    def test_rate_checkpoint_pin_is_checked_before_any_tensor_loading(self):
        with mock.patch.object(cfg,'RATE_ROOT',self.root/'fake_rate'), \
             mock.patch.object(runtime.rate,'load_finetuned') as loader:
            directory=cfg.RATE_ROOT/'training'/'fdma';directory.mkdir(parents=True)
            (directory/'checkpoint_latest.pt').write_bytes(b'wrong-source-checkpoint')
            with self.assertRaisesRegex(ValueError,'Pinned rate terminal'):
                runtime.load_model('rate','fdma','cpu')
            loader.assert_not_called()


if __name__=='__main__':unittest.main()
