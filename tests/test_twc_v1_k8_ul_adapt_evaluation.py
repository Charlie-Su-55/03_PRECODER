"""Real CPU receiver/decoder/RZF replay with explicit nonproduction sources."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from configs import twc_v1_k8_ul_adapt_config as cfg
import evaluate_twc_v1_k8_ul_adapt as evaluate
import twc_v1_k8_ul_adapt_results as results
import ul_adapt_runtime as runtime
from twc_v1_k8_ul_calibration import fit_model


class FinalAdaptationReplayTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.output=self.root/'adapt'
        self.cal=self.root/'frozen_calibration';self.cal.mkdir()
        self.source=self.root/'explicit_fixture.bin'
        self.source.write_bytes(b'CPU fixture; not a scientific checkpoint')
        calrows=[dict(role='calibration',gamma_db=g,feature=float(i),n_coordinates=2048,
            h2_sum=2048*32.,he_real_sum=0.,he_imag_sum=0.,parallel_scaled_sum=2048*3.,
            he_scaled_real_sum=0.,he_scaled_imag_sum=0.,h2_scaled_sum=2048*32./.02,
            perp_scaled_sum=2048*.8) for g in cfg.GAMMAS for i in range(4)]
        self.model=fit_model(calrows)
        (self.cal/'model.json').write_text(json.dumps(self.model,indent=2,sort_keys=True)+'\n')
        (self.cal/'manifest.json').write_text('{"explicit_cpu_fixture":true}\n')
        (self.cal/'input_fingerprints.json').write_text(json.dumps([
            dict(fingerprint=dict(H_dl='a'*64,H_ul='b'*64))]))
        self.directory=self.output/'evaluation'/'attempt_fixture'
        for patch in (mock.patch.object(cfg,'OUTPUT_ROOT',self.output),
                mock.patch.object(runtime,'OUTPUT_ROOT',self.output),
                mock.patch.object(cfg,'FINAL_BATCHES',1),
                mock.patch.dict(cfg.ROLE_RANGES,final=[2048,2048]),
                mock.patch.object(cfg,'CALIBRATION_DIR',self.cal),
                mock.patch.object(cfg,'CALIBRATION_MODEL_SHA256',runtime.sha256(self.cal/'model.json')),
                mock.patch.object(cfg,'CALIBRATION_MANIFEST_SHA256',runtime.sha256(self.cal/'manifest.json'))):
            patch.start();self.addCleanup(patch.stop)
        gen=torch.Generator().manual_seed(123891)
        self.hdl=torch.randn(4,8,32,256,dtype=torch.complex64,generator=gen)
        self.hul=torch.randn(4,8,32,1,dtype=torch.complex64,generator=gen).expand(-1,-1,-1,256).clone()
        codes=torch.randn(8,216,dtype=torch.complex64,generator=gen)
        template=torch.randn(8,32,256,dtype=torch.complex64,generator=gen)
        outer=self
        self.events=[];self.data_reads=[];self.factories=[];self.instances=[]
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('codes',codes.clone())
                self.scale=nn.Parameter(torch.tensor(1.))
            def forward(self,H):return self.scale*self.codes[None].expand(H.shape[0],-1,-1)
        class Decoder(nn.Module):
            def __init__(self,phase,arm,method):
                super().__init__();self.register_buffer('template',template.clone())
                self.scale=nn.Parameter(torch.tensor(.7+.03*(arm=='hybrid_g4')+.04*(phase=='final')))
                self.label=(phase,arm,method)
            def forward(self,Y_rx,H_ul_est,soft,reliability,pilots,*,gamma_db,layout,prior,ce_error):
                outer.events.append(('decoder',self.label,tuple(gamma_db.tolist())))
                gain=soft.real.mean(-1)+.01*torch.log1p(reliability).mean(-1)
                return dict(H_hat=self.scale*self.template[None]*(1+.01*gain[...,None,None]))
        class CE(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.96))
            def forward(self,Y,pilots,initial,*,gamma_db):return initial*self.scale
        self.CE=CE
        self.metadata=runtime.make_data('cpu')
        class Data:
            protocol=outer.metadata.protocol
            def sample(self,role,index):
                outer.data_reads.append((role,index))
                return dict(H_dl=outer.hdl.clone(),H_ul=outer.hul.clone(),
                    address=outer.metadata.address(role,index))
        self.Data=Data
        self.cal_record=dict(model_sha256=cfg.CALIBRATION_MODEL_SHA256,
            manifest_sha256=cfg.CALIBRATION_MANIFEST_SHA256,
            common_CE_signature=runtime.state_digest(CE()),
            frozen_files=[dict(path=str(p),sha256=runtime.sha256(p))
                for p in (self.cal/'model.json',self.cal/'manifest.json',self.cal/'input_fingerprints.json')])
        patch=mock.patch.object(runtime,'load_calibration',side_effect=lambda:(copy.deepcopy(self.model),copy.deepcopy(self.cal_record)))
        patch.start();self.addCleanup(patch.stop)
        self.terminal_files={}
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                p=self.root/(cfg.key(arm,method)+'.fixture')
                p.write_bytes(('explicit fixture '+cfg.key(arm,method)).encode())
                self.terminal_files[arm,method]=p
        def loader(phase,arm,method,device):
            outer.events.append(('load',phase,arm,method))
            codec=nn.Module();codec.encoder=Encoder();codec.decoder=Decoder(phase,arm,method)
            phy=runtime.physics()
            stats=phy.ReceiverStatistics(phy.make_layout(arm),ce_errors=torch.full((6,8,256),.02))
            stats.ema_updates.fill_(20000)
            ce=CE()
            for module in (codec,stats,ce):module.eval().requires_grad_(False)
            outer.instances.append((codec,stats,ce))
            files=[dict(path=str(outer.source),sha256=runtime.sha256(outer.source)),*outer.cal_record['frozen_files']]
            record=dict(arm=arm,method=method,production_evidence=False,step=5000 if phase=='source' else cfg.UPDATES,
                state_sha256=runtime.state_digest(codec),CE_state_sha256=runtime.state_digest(ce),
                statistics_state_sha256=runtime.state_digest(stats),frozen_files=files)
            if phase=='final':
                terminal=outer.terminal_files[arm,method]
                record.update(checkpoint_sha256=runtime.sha256(terminal),
                    frozen_files=[*files,dict(path=str(terminal),sha256=runtime.sha256(terminal))])
            return codec,stats,ce,record
        self.source_loader=lambda a,m,d:loader('source',a,m,d)
        self.terminal_loader=lambda a,m,d:loader('final',a,m,d)
        def factory(device):
            outer.events.append(('factory',));outer.factories.append(device);return Data()
        self.factory=factory

    def run_eval(self,**kwargs):
        return evaluate.evaluate(self.directory,require_cuda=False,source_loader=self.source_loader,
            terminal_loader=kwargs.pop('terminal_loader',self.terminal_loader),
            data_factory=kwargs.pop('data_factory',self.factory),**kwargs)

    def test_real_final_replay_accepts_all_terminals_before_data_and_preserves_sources(self):
        old=self.source.read_bytes();model=(self.cal/'model.json').read_bytes()
        manifest=self.run_eval()
        self.assertEqual(manifest['status'],'test_fixture')
        self.assertFalse(manifest['production_evidence'])
        first_factory=next(i for i,e in enumerate(self.events) if e[0]=='factory')
        self.assertEqual(self.events[:first_factory],
            [('load','final',a,m) for a in cfg.ARMS for m in cfg.TRAIN_METHODS]+[('load','source','fdma','original')])
        self.assertEqual(set(self.data_reads),{('validation',2048)})
        self.assertEqual(len(self.factories),8)
        rows=results.read_raw_csv(self.directory/'samples.csv.gz')
        self.assertEqual(len(rows),2*2*2*6*4)
        self.assertTrue(all(r['sum_rate_bps_hz']>=0 and r['rzf_power_max_error']<=1e-5 for r in rows))
        self.assertTrue(all('soft_actual_to_reported_mse_ratio' in r for r in rows))
        fingerprints=runtime.read_json(self.directory/'input_fingerprints.json')
        self.assertEqual(len(fingerprints),2*6)
        self.assertTrue(all(len(r['conditions'])==4 for r in fingerprints))
        self.assertEqual(self.source.read_bytes(),old)
        self.assertEqual((self.cal/'model.json').read_bytes(),model)
        for states in manifest['frozen_signatures'].values():
            self.assertTrue(all(v['before']==v['after'] for v in states.values()))
        evaluate.verify_manifest(self.directory,allow_fixture=True)
        with self.assertRaisesRegex(ValueError,'fixture'):
            evaluate.verify_manifest(self.directory)
        self.assertFalse(list(self.output.rglob('*.zip')))

    def test_last_missing_terminal_stops_before_final_factory(self):
        original=self.terminal_loader
        def wrong(arm,method,device):
            codec,stats,ce,record=original(arm,method,device)
            if (arm,method)==('hybrid_g4','moment_receiver'):record['step']=cfg.UPDATES-1
            return codec,stats,ce,record
        with self.assertRaisesRegex(ValueError,'four fixed'):
            self.run_eval(terminal_loader=wrong)
        self.assertFalse(self.factories);self.assertFalse(self.data_reads)
        self.assertFalse(self.directory.exists())

    def test_pinned_calibration_bytes_are_checked_before_terminals_or_final_factory(self):
        path=self.cal/'model.json';path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'Pinned calibration'):
            self.run_eval()
        self.assertFalse(self.events);self.assertFalse(self.factories)
        self.assertFalse(self.directory.exists())

    def test_missing_production_training_journal_blocks_before_final_factory(self):
        # Fault injection enters the production admission path, but cannot
        # create a production result: missing immutable training evidence must
        # fail before the final-data factory is reached.
        trainer=evaluate._trainer()
        base_metadata=runtime.make_data
        protocol=base_metadata('cuda:0').protocol
        calls=[]
        def metadata_only(device):
            calls.append(str(device))
            if str(device)!='cuda:0':
                raise AssertionError('final data factory was reached')
            return base_metadata(device)
        original=self.terminal_loader
        def claimed(arm,method,device):
            codec,stats,ce,record=original(arm,method,device)
            record.update(production_evidence=True,core=runtime.core_identity(),data_protocol=protocol)
            return codec,stats,ce,record
        with (mock.patch.object(runtime,'configure_cuda',return_value=torch.device('cpu')),
                mock.patch.object(runtime,'load_initial',side_effect=self.source_loader),
                mock.patch.object(trainer,'load_finetuned',side_effect=claimed),
                mock.patch.object(runtime,'make_data',side_effect=metadata_only)):
            with self.assertRaisesRegex(ValueError,'immutable training progress'):
                evaluate.evaluate(self.directory)
        self.assertTrue(calls)
        self.assertTrue(all(device=='cuda:0' for device in calls))
        self.assertFalse(self.factories);self.assertFalse(self.data_reads)
        self.assertFalse(self.directory.exists())

    def test_wrong_final_address_cannot_read_monitor_or_test_data(self):
        parent=self.Data
        class Wrong(parent):
            def sample(inner,role,index):
                sample=super().sample(role,index)
                sample['address']=self.metadata.address('validation',0)
                return sample
        with mock.patch.object(runtime,'forward_codec',side_effect=AssertionError('must not forward')):
            with self.assertRaisesRegex(ValueError,'address'):
                self.run_eval(data_factory=lambda device:Wrong())

    def test_actual_paired_inputs_cannot_change_between_methods(self):
        original=runtime.forward_codec
        def wrong(*args,**kwargs):
            out=original(*args,**kwargs)
            if kwargs['method']=='moment_receiver':out['input_hashes']['standard_noise']='f'*64
            return out
        with self.assertRaisesRegex(ValueError,'identical'):
            self.run_eval(forward=wrong)
        self.assertEqual(runtime.read_json(self.directory/'manifest.json')['status'],'failed')

    def test_moving_a_common_CE_parameter_is_rejected(self):
        original=self.source_loader
        def wrong(arm,method,device):
            codec,stats,ce,record=original(arm,method,device)
            ce.scale.fill_(.8)
            return codec,stats,ce,record
        self.source_loader=wrong
        with self.assertRaisesRegex(ValueError,'Frozen CE'):
            self.run_eval()

    def test_in_memory_calibrator_mutation_is_rejected(self):
        original=runtime.forward_codec
        def wrong(*args,**kwargs):
            out=original(*args,**kwargs)
            kwargs['calibration']['nodes'][0]['strata'][0]['parallel']+=.1
            return out
        with self.assertRaisesRegex(ValueError,'calibration'):
            self.run_eval(forward=wrong)

    def test_read_only_scoring_cannot_mutate_actual_decoder_output(self):
        original=evaluate.score_out
        def wrong(out,truth):
            result=original(out,truth);out['H_hat'].add_(.1);return result
        with mock.patch.object(evaluate,'score_out',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'Offline scoring'):
                self.run_eval()

    def test_actual_calibration_channel_reuse_is_rejected(self):
        path=self.cal/'input_fingerprints.json'
        path.write_text(json.dumps([dict(fingerprint=dict(
            H_dl=evaluate._tensor_hash(self.hdl),H_ul=evaluate._tensor_hash(self.hul)))]))
        self.cal_record['frozen_files'][2]['sha256']=runtime.sha256(path)
        with self.assertRaisesRegex(ValueError,'overlap'):
            self.run_eval()

    def test_artifact_mutation_and_condition_coverage_are_rejected(self):
        self.run_eval()
        path=self.directory/'summary.csv';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'artifact'):
            evaluate.verify_manifest(self.directory,allow_fixture=True)
        entries=runtime.read_json(self.directory/'input_fingerprints.json')
        entries[0]['conditions'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):
            evaluate.fingerprint_map(entries)
        entries=runtime.read_json(self.directory/'input_fingerprints.json')
        # Changing both arms keeps cross-arm pairing intact but must still
        # fail the stronger same-MU pairing across gamma views.
        for item in entries:
            if item['gamma_db']==cfg.GAMMAS[0]:
                item['input_hashes']['H_dl']='f'*64
        with self.assertRaisesRegex(ValueError,'same complete-MU channel'):
            evaluate.fingerprint_map(entries)

    def test_actual_evaluation_scalar_payloads_pass_public_schema(self):
        self.run_eval()
        import publish_twc_v1_k8_ul_adapt as publisher
        for name in ('samples.csv.gz','summary.csv','comparisons.csv','input_fingerprints.json'):
            publisher._check_scalar_file(self.directory/name)
        publisher._check_scalar_file(self.cal/'model.json')


class ActualTrainingPairingTests(unittest.TestCase):
    """Synthetic journals exercise real SHA chains without running training."""
    def setUp(self):
        import train_twc_v1_k8_ul_adapt as trainer
        self.trainer=trainer
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)/'adapt'
        for patch in (mock.patch.object(cfg,'OUTPUT_ROOT',self.root),
                mock.patch.object(runtime,'OUTPUT_ROOT',self.root),
                mock.patch.object(cfg,'UPDATES',200)):
            patch.start();self.addCleanup(patch.stop)
        self.metadata=runtime.make_data('cuda:0')
        self.protocol=self.metadata.protocol
        trace=trainer.initial_training_trace(self.protocol)
        common={}
        snapshots={}
        for step in range(1,cfg.UPDATES+1):
            before=dict(trace);items=[]
            for micro in range(cfg.ACCUMULATION):
                index=cfg.micro_index(step,micro)
                def digest(name):return hashlib.sha256((name+str(index)).encode()).hexdigest()
                item=dict(micro_index=index,address=self.metadata.address('codec_train',index),
                    channel=digest('channel'),standard_noise=digest('noise'),pilot_polarity=digest('polarity'),
                    gamma=evaluate._tensor_hash(torch.tensor(cfg.gamma_schedule(index),dtype=torch.float32)),
                    wave_active=digest('wave'),Y_rx=digest('Y'),estimated_UL=digest('UL'),
                    CE_variance=digest('variance'),prior=digest('prior'))
                trace=trainer.advance_training_trace(trace,item,self.protocol)
                items.append(item)
            if step%cfg.SAVE_EVERY==0:
                snapshots[step]=dict(step=step,counters=cfg.counters(step),input_fingerprints=items,
                    training_trace_before_update=before,training_trace=dict(trace))
        self.records={};self.rows={}
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                key=cfg.key(arm,method)
                self.rows[key]=[dict(copy.deepcopy(row),arm=arm,method=method,
                    attempt_id='fixture_attempt',resumed_from=0) for row in snapshots.values()]
                self.records[key]=dict(data_protocol=self.protocol,training_trace=dict(trace))
                self.write(key)

    def write(self,key):
        path=self.root/'training'/key/'progress.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in self.rows[key]))
        self.records[key]['progress_sha256']=runtime.sha256(path)

    def check(self):return evaluate.verify_training_pairing(self.records)

    def test_complete_micro_hash_chain_and_saved_snapshots_pass(self):
        result=self.check()
        self.assertTrue(result['full_micro_trace_chain_verified'])
        self.assertTrue(result['snapshot_pairing_verified'])
        self.assertEqual(result['hashed_microbatches'],1600)
        self.assertEqual(result['hashed_MU_frames'],6400)
        self.assertEqual(result['logged_boundary_updates'],2)
        self.assertEqual(result['logged_microbatches'],16)
        self.assertEqual(len(result['trajectories']),4)
        for row in result['trajectories'].values():self.assertEqual(row['superseded_journal_rows'],0)

    def test_failed_save_journal_row_and_successful_retry_are_retained(self):
        key=cfg.key('fdma','original')
        failed=copy.deepcopy(self.rows[key][0]);failed['attempt_id']='failed_before_checkpoint_commit'
        self.rows[key].insert(0,failed)
        self.rows[key][1]['attempt_id']='successful_retry'
        self.write(key)
        result=self.check()
        self.assertEqual(result['trajectories'][key]['retained_journal_rows'],3)
        self.assertEqual(result['trajectories'][key]['superseded_journal_rows'],1)
        self.assertEqual(result['logged_boundary_updates'],2)

    def test_missing_training_boundary_refused(self):
        key=cfg.key('hybrid_g4','moment_receiver')
        self.rows[key].pop(0);self.write(key)
        with self.assertRaisesRegex(ValueError,'complete saved-boundary'):
            self.check()

    def test_actual_gamma_must_match_declared_schedule_not_only_other_trajectories(self):
        for key in self.rows:
            self.rows[key][0]['input_fingerprints'][0]['gamma']='0'*64
            self.write(key)
        with self.assertRaisesRegex(ValueError,'balanced schedule'):
            self.check()

    def test_snapshot_micro_payload_must_close_rolling_endpoint(self):
        key=cfg.key('fdma','moment_receiver')
        self.rows[key][0]['input_fingerprints'][0]['standard_noise']='f'*64
        self.write(key)
        with self.assertRaisesRegex(ValueError,'close their rolling'):
            self.check()

    def test_wrong_source_data_role_is_rejected(self):
        key=cfg.key('fdma','original')
        self.rows[key][0]['input_fingerprints'][0]['address']=self.metadata.address('validation',0)
        self.write(key)
        with self.assertRaisesRegex(ValueError,'address'):
            self.check()

    def test_complete_stream_must_match_across_trajectories(self):
        key=cfg.key('fdma','moment_receiver')
        # Change an unseen middle-step digest, then rebuild the logged update
        # ending at200 from a different incoming trace. It is self-consistent
        # locally, but fails the common full-stream endpoint comparison.
        before=copy.deepcopy(self.rows[key][-1]['training_trace_before_update'])
        before['sha256']='f'*64
        after=dict(before)
        for item in self.rows[key][-1]['input_fingerprints']:
            after=self.trainer.advance_training_trace(after,item,self.protocol)
        self.rows[key][-1]['training_trace_before_update']=before
        self.rows[key][-1]['training_trace']=after
        self.records[key]['training_trace']=after
        self.write(key)
        with self.assertRaisesRegex(ValueError,'complete actual training input stream'):
            self.check()

    def test_terminal_progress_hash_and_fixture_labels_are_preserved(self):
        key=cfg.key('fdma','original')
        path=self.root/'training'/key/'progress.jsonl'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'immutable training progress'):
            self.check()
        result=evaluate.verify_training_pairing({},fixture=True)
        self.assertFalse(result['production_evidence'])
        self.assertFalse(result['full_micro_trace_chain_verified'])
        self.assertEqual(result['hashed_microbatches'],0)


if __name__=='__main__':unittest.main()
