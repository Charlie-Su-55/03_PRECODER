"""Owned-file and actual small-CPU warmstart checks; no server weights/CDL."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from configs import twc_v1_k8_w_residual_config as cfg
import w_residual_runtime as runtime

try:
    import torch
    from torch import nn
except ImportError:
    torch=None


class RuntimeFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'new_outputs'
        for patch in (mock.patch.object(runtime,'OUTPUT_ROOT',self.output),
                      mock.patch.object(cfg,'OUTPUT_ROOT',self.output),
                      mock.patch.object(cfg,'UPDATES',200)):
            patch.start();self.addCleanup(patch.stop)

    def traces(self):
        metadata=runtime.make_data('cuda:0')
        records=[]
        for step in (100,200):
            values=[]
            for micro in range(cfg.ACCUMULATION):
                index=cfg.micro_index(step,micro)
                values.append(dict(micro_index=index,address=metadata.address('codec_train',index),
                    **{name:format(index,'064x') for name in ('channel','standard_noise','pilot_polarity','gamma')}))
            records.append(dict(step=step,input_fingerprints=values))
        paths={}
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                path=self.output/'training'/cfg.key(arm,method)/'progress.jsonl'
                path.parent.mkdir(parents=True)
                path.write_text(''.join(json.dumps(row)+'\n' for row in records))
                paths[cfg.key(arm,method)]=path
        return records,paths

    def test_atomic_exclusive_records_and_failed_replacement_preserve_bytes(self):
        path=self.output/'batch'/'identity.json'
        runtime.atomic_json(path,dict(value=1),exclusive=True);before=path.read_bytes()
        with self.assertRaises(FileExistsError):runtime.atomic_json(path,dict(value=2),exclusive=True)
        with mock.patch.object(runtime.os,'replace',side_effect=OSError('replace failure')):
            with self.assertRaises(OSError):runtime.atomic_json(path,dict(value=3))
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(list(path.parent.iterdir()),[path])

    def test_owned_paths_reject_root_external_and_symlink(self):
        self.output.mkdir();external=self.root/'protected';external.mkdir()
        link=self.output/'link';link.symlink_to(external,target_is_directory=True)
        for path in (self.output,external/'file',link/'file'):
            with self.assertRaises(ValueError):runtime.atomic_json(path,dict(value=1))
        self.assertEqual(list(external.iterdir()),[])

    def test_real_file_hash_refuses_changed_teacher(self):
        path=self.root/'teacher';path.write_bytes(b'original')
        record=dict(frozen_files=[dict(path=str(path),sha256=runtime.sha256(path))])
        runtime.verify_files(record);path.write_bytes(b'changed')
        with self.assertRaises(ValueError):runtime.verify_files(record)

    def test_retained_actual_training_inputs_pair_across_four_jobs(self):
        records,paths=self.traces();result=runtime.training_pairing_evidence()
        self.assertEqual(result['boundaries_per_job'],2)
        self.assertEqual(result['microbatches_checked_per_job'],2*cfg.ACCUMULATION)
        self.assertEqual(len(result['files']),4)
        # A failed checkpoint followed by a replay may duplicate the same input record.
        path=next(iter(paths.values()))
        with path.open('a') as stream:stream.write(json.dumps(records[0])+'\n')
        self.assertEqual(runtime.training_pairing_evidence()['status'],'passed')

    def test_changed_random_inputs_or_wrong_role_or_missing_boundary_rejected(self):
        records,paths=self.traces();path=next(iter(paths.values()))
        for mode in ('random','address','coverage','schema'):
            bad=copy.deepcopy(records)
            if mode=='random':bad[0]['input_fingerprints'][0]['gamma']='f'*64
            elif mode=='address':bad[0]['input_fingerprints'][0]['address']=runtime.make_data('cuda:0').address('validation',0)
            elif mode=='coverage':bad.pop()
            else:bad[0]['input_fingerprints'][0]['unpaid_truth']='not allowed'
            path.write_text(''.join(json.dumps(row)+'\n' for row in bad))
            with self.assertRaises(ValueError,msg=mode):runtime.training_pairing_evidence()


@unittest.skipIf(torch is None,'Real PyTorch unavailable; no fake tensor dependency')
class WarmstartPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'experiment'
        self.source=self.root/'teacher';self.source.write_bytes(b'preserved original weights')
        for patch in (mock.patch.object(runtime,'OUTPUT_ROOT',self.output),
                      mock.patch.object(cfg,'OUTPUT_ROOT',self.output),
                      mock.patch.object(runtime,'core_identity',return_value=dict(schema='test',sha256={}))):
            patch.start();self.addCleanup(patch.stop)
        self.saved=self.source.read_bytes();self.latest={}
        generator=torch.Generator().manual_seed(971)
        self.H_dl=torch.randn(4,8,32,256,dtype=torch.complex64,generator=generator)
        self.H_ul=torch.randn(4,8,32,256,dtype=torch.complex64,generator=generator)
        metadata=runtime.make_data('cpu');outer=self

        class Base(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.8))
            def forward(self,*,H_ul_est,**kwargs):return dict(H_hat=H_ul_est*self.scale)

        class Frozen(nn.Module):
            def __init__(self):
                super().__init__();self.weight=nn.Parameter(torch.tensor(1.));self.register_buffer('tracked',torch.tensor(0.))

        class Data:
            protocol=metadata.protocol
            def address(self,role,index):return metadata.address(role,index)
            def sample(self,role,index):
                return dict(H_dl=outer.H_dl.clone(),H_ul=outer.H_ul.clone(),address=self.address(role,index))

        self.Data=Data
        def loader(arm,method,device):
            codec=nn.Module();codec.encoder=Frozen();base=Base()
            teacher=dict(base_decoder_state_sha256=runtime.state_digest(base),
                frozen_files=[dict(path=str(self.source),sha256=runtime.sha256(self.source))])
            codec.decoder=runtime.model_module().PrecodingDecoder(base,method,head_seed=cfg.HEAD_SEED)
            stats,ce=Frozen(),Frozen()
            for module in (codec,stats,ce):module.eval().requires_grad_(False)
            self.latest.update(codec=codec,stats=stats,ce=ce)
            return codec,stats,ce,teacher
        self.loader=loader

        def forward(codec,stats,ce,sample,gamma_db,noise_key):
            gamma,noise,polarity=runtime.batch_random('monitor',noise_key[1]*10+noise_key[2],'cpu',gamma=gamma_db)
            out=codec.decoder(Y_rx=sample['H_ul'][:,0],H_ul_est=sample['H_ul'],soft=None,
                reliability=None,pilots=None,gamma_db=gamma)
            return dict(out,input_hashes=dict(standard_noise=runtime.state_digest({'noise':noise}),
                pilot_polarity=runtime.state_digest({'polarity':polarity}),gamma=runtime.state_digest({'gamma':gamma})))
        self.forward=forward
        for patch in (mock.patch.object(runtime,'load_initial',side_effect=loader),
                      mock.patch.object(runtime,'make_data',side_effect=lambda device:Data())):
            patch.start();self.addCleanup(patch.stop)

    def run_preflight(self,forward=None):
        with mock.patch.object(runtime,'forward_codec',side_effect=forward or self.forward):
            return runtime.preflight_sources(self.output/'preflight'/'attempt_fixture',require_cuda=False)

    def test_real_cpu_zero_head_matches_rates_and_keeps_source_unchanged(self):
        result=self.run_preflight()
        self.assertEqual(result['status'],'passed');self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual(len(result['warmstart_checks']),12)
        self.assertTrue(all(r['max_precoder_error']<2e-6 for r in result['warmstart_checks']))
        self.assertEqual(self.source.read_bytes(),self.saved)

    def test_preflight_rejects_nonzero_head_and_leaves_no_passed_manifest(self):
        def changed(*args,**kwargs):
            out=self.forward(*args,**kwargs);out['delta_W']=torch.ones_like(out['delta_W'])*.01;return out
        with self.assertRaisesRegex(ValueError,'not zero'):self.run_preflight(changed)
        self.assertFalse((self.output/'preflight'/'attempt_fixture'/'manifest.json').exists())

    def test_frozen_grad_flag_mutation_and_changed_teacher_are_rejected(self):
        for kind in ('grad','source','base'):
            self.source.write_bytes(self.saved)
            def changed(codec,stats,ce,*args,**kwargs):
                out=self.forward(codec,stats,ce,*args,**kwargs)
                if kind=='grad':ce.weight.requires_grad_(True)
                elif kind=='source':self.source.write_bytes(b'tamper')
                else:
                    with torch.no_grad():codec.decoder.base_decoder.scale.add_(.1)
                return out
            with self.assertRaises((ValueError,RuntimeError),msg=kind):self.run_preflight(changed)
            self.assertFalse((self.output/'preflight'/'attempt_fixture'/'manifest.json').exists())

    def test_wrong_phase_path_rejected_before_allocation(self):
        with mock.patch.object(runtime,'configure_cuda') as configure:
            with self.assertRaises(ValueError):runtime.preflight_sources(self.output/'evaluation'/'attempt_bad')
            configure.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
