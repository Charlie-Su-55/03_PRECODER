"""Real CPU radio/decoder graph, fixed calibration, replay and owned IO checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from configs import twc_v1_k8_ul_adapt_config as cfg
import ul_adapt_runtime as runtime


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def fixture_model():
    math = runtime._calibration_math()
    rows = []
    for gamma in cfg.GAMMAS:
        for sample in range(3):
            rows.append(dict(role='calibration', gamma_db=gamma, feature=float(sample),
                n_coordinates=2048, h2_sum=2048., he_real_sum=0., he_imag_sum=0.,
                parallel_scaled_sum=2048., he_scaled_real_sum=0., he_scaled_imag_sum=0.,
                h2_scaled_sum=2048., perp_scaled_sum=2048.))
    model = math.fit_model(rows)
    for node in model['nodes']:
        for cell in node['strata']:
            cell.update(alpha_real=.04, parallel=3., perpendicular=.75)
    return math.validate_model(model)


class ULAdaptRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'adapt'
        self.calibration = self.root/'locked_calibration'
        self.calibration.mkdir()
        self.model = fixture_model()
        (self.calibration/'model.json').write_bytes(canonical(self.model))
        (self.calibration/'source_records.json').write_text('{"fixture":true}\n')
        self.ce_signature = None
        manifest = dict(model_sha256=runtime.sha256(self.calibration/'model.json'),
            common_CE_signature='f'*64, files={n:runtime.sha256(self.calibration/n)
                for n in ('model.json','source_records.json')})
        (self.calibration/'manifest.json').write_bytes(canonical(manifest))
        self.manifest = manifest
        for patch in (mock.patch.object(cfg,'CALIBRATION_DIR',self.calibration),
                mock.patch.object(cfg,'CALIBRATION_MODEL_SHA256',manifest['model_sha256']),
                mock.patch.object(cfg,'CALIBRATION_MANIFEST_SHA256',runtime.sha256(self.calibration/'manifest.json')),
                mock.patch.object(runtime,'OUTPUT_ROOT',self.output)):
            patch.start()
            self.addCleanup(patch.stop)
        generator = torch.Generator().manual_seed(9086)
        self.codes = torch.randn(8,216,dtype=torch.complex64,generator=generator)
        self.template = torch.randn(8,32,256,dtype=torch.complex64,generator=generator)
        self.sample = {name:torch.randn(4,8,32,256,dtype=torch.complex64,generator=generator)
            for name in ('H_dl','H_ul')}

    def modules(self, arm='fdma'):
        codes, template = self.codes, self.template
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.))
                self.register_buffer('codes',codes.clone())
            def forward(self, H):
                return self.scale*self.codes[None].expand(H.shape[0],-1,-1)
        class CE(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(.96))
            def forward(self,Y,pilots,initial,*,gamma_db):
                return self.scale*initial
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(.9))
                self.offset = nn.Parameter(torch.tensor(.1))
                self.register_buffer('template',template.clone())
            def forward(self,*,Y_rx,H_ul_est,soft,reliability,pilots,gamma_db,layout,prior,ce_error):
                feature = soft.real.mean(-1)+.01*torch.log1p(reliability).mean(-1)
                return dict(H_hat=self.template[None]*(self.weight+self.offset*feature[...,None,None]))
        codec = nn.Module()
        codec.encoder, codec.decoder = Encoder(), Decoder()
        ce = CE()
        phy = runtime.physics()
        stats = phy.ReceiverStatistics(phy.make_layout(arm),
            ce_errors=torch.full((6,8,256),.02))
        stats.ema_updates.fill_(20000)
        for module in (codec,stats,ce):
            module.eval().requires_grad_(False)
        return codec,stats,ce

    def test_real_cpu_streams_are_balanced_paired_and_do_not_touch_default_rng(self):
        state = torch.random.get_rng_state().clone()
        gamma,noise,polarity = runtime.batch_random('codec_train',1,'cpu')
        self.assertEqual(gamma.tolist(),[0.,5.,10.,15.])
        self.assertEqual(runtime.batch_random('codec_train',2,'cpu')[0].tolist(),[20.,25.,0.,5.])
        again = runtime.batch_random('codec_train',1,'cpu')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip((gamma,noise,polarity),again)))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual(noise.shape,(4,32,256))
        self.assertEqual(noise.dtype,torch.complex64)
        self.assertEqual(set(polarity.unique().tolist()),{-1.,1.})
        self.assertFalse(torch.equal(noise,runtime.batch_random('monitor',1,'cpu',gamma=0.)[1]))
        for role,index,values in (('codec_train',1,25.),('codec_train',0,None),
                ('monitor',0,None),('final',0,2.),('monitor',0,True)):
            with self.assertRaises(ValueError):
                runtime.batch_random(role,index,'cpu',gamma=values)

    def test_real_frozen_receivers_build_gradients_only_through_decoder_for_both_methods(self):
        codec,stats,ce = self.modules('hybrid_g4')
        codec.decoder.train().requires_grad_(True)
        before = runtime.helpers().frozen_state(codec,ce,stats)
        model_bytes = canonical(self.model)
        outputs = {}
        with mock.patch.object(runtime,'load_calibration',side_effect=AssertionError('must use cached model')):
            for method in cfg.TRAIN_METHODS:
                codec.decoder.zero_grad(set_to_none=True)
                out = runtime.forward_codec(codec,stats,ce,self.sample,None,
                    ('codec_train',1,0),method=method,calibration=self.model)
                self.assertTrue(out['H_hat'].requires_grad)
                self.assertFalse(out['soft'].requires_grad)
                self.assertFalse(out['H_ul_est'].requires_grad)
                self.assertEqual(out['gamma_db'].tolist(),[0.,5.,10.,15.])
                loss = out['H_hat'].abs().square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in codec.decoder.parameters()))
                self.assertTrue(all(p.grad is None for m in (codec.encoder,ce,stats) for p in m.parameters()))
                runtime.helpers().assert_frozen_state(codec,ce,stats,before)
                outputs[method] = out
        self.assertEqual(outputs['original']['input_hashes'],outputs['moment_receiver']['input_hashes'])
        self.assertTrue(torch.equal(outputs['original']['H_ul_est'],outputs['original']['source_estimated_UL']))
        self.assertFalse(torch.equal(outputs['moment_receiver']['H_ul_est'],outputs['moment_receiver']['source_estimated_UL']))
        self.assertEqual(canonical(self.model),model_bytes)

    def test_moment_deployment_receives_only_public_inputs_and_eval_gamma_is_explicit(self):
        codec,stats,ce = self.modules()
        math = runtime._calibration_math()
        with mock.patch.object(math,'recover_variants',wraps=math.recover_variants) as recover:
            out = runtime.forward_codec(codec,stats,ce,self.sample,10.,('monitor',0,2),
                method='moment_receiver',calibration=self.model)
        self.assertEqual(set(recover.call_args.args[0]),{'Y','estimated','ce_variance','prior','pilots'})
        self.assertEqual(out['gamma_db'].tolist(),[10.]*4)
        self.assertIs(out['layout'],stats.layout)
        self.assertEqual(set(out['input_hashes']),{'standard_noise','pilot_polarity','gamma',
            'wave_active','Y_rx','estimated_UL','CE_variance','prior'})
        self.assertTrue(all(len(value)==64 for value in out['input_hashes'].values()))
        with self.assertRaisesRegex(ValueError,'declared node index'):
            runtime.forward_codec(codec,stats,ce,self.sample,25.,('monitor',0,2),
                method='original',calibration=self.model)

    def test_cached_model_and_frozen_module_mutations_are_rejected(self):
        codec,stats,ce = self.modules()
        changed = copy.deepcopy(self.model)
        changed['nodes'][0]['strata'][0]['alpha_real'] += .1
        with self.assertRaisesRegex(ValueError,'not the pinned'):
            runtime.forward_codec(codec,stats,ce,self.sample,None,('codec_train',1,0),
                method='moment_receiver',calibration=changed)
        for bad_module in (codec.encoder,stats,ce):
            bad_module.train()
            with self.assertRaisesRegex(ValueError,'must remain frozen'):
                runtime.forward_codec(codec,stats,ce,self.sample,None,('codec_train',1,0),
                    method='original',calibration=self.model)
            bad_module.eval()

    def test_calibration_hash_pins_checked_before_original_production_verifier(self):
        with mock.patch('evaluate_twc_v1_k8_ul_calibration.verify_calibration',return_value=self.manifest) as verify:
            model,record = runtime.load_calibration()
            self.assertEqual(model,self.model)
            self.assertEqual(record['common_CE_signature'],'f'*64)
            self.assertEqual(len(record['frozen_files']),3)
            verify.assert_called_once_with(self.calibration)
            runtime.verify_files(record)
        path = self.calibration/'model.json'
        path.write_bytes(path.read_bytes()+b'\n')
        with mock.patch('evaluate_twc_v1_k8_ul_calibration.verify_calibration') as verify:
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                runtime.load_calibration()
            verify.assert_not_called()

    def test_both_methods_start_from_identical_frozen_source_decoder(self):
        source = self.root/'source.bin'
        source.write_bytes(b'explicit test fixture; not a source checkpoint')
        loaded = []
        def loader(arm,device):
            codec,stats,ce = self.modules(arm)
            record = dict(kind='rate',arm=arm,frozen_files=[dict(path=str(source),sha256=runtime.sha256(source))])
            loaded.append((codec,stats,ce))
            return codec,stats,ce,record
        ce_signature = runtime.state_digest(self.modules()[2])
        calibration_record = dict(common_CE_signature=ce_signature,frozen_files=[],
            model_sha256=cfg.CALIBRATION_MODEL_SHA256,manifest_sha256=cfg.CALIBRATION_MANIFEST_SHA256)
        with mock.patch.object(runtime,'load_calibration',return_value=(self.model,calibration_record)), \
                mock.patch.object(runtime.previous,'load_source',side_effect=loader), \
                mock.patch.object(runtime.previous,'verify_source_record'):
            records = [runtime.load_initial('fdma',method,'cpu')[3] for method in cfg.TRAIN_METHODS]
        self.assertEqual(records[0]['base_decoder_state_sha256'],records[1]['base_decoder_state_sha256'])
        for codec,stats,ce in loaded:
            self.assertTrue(all(not child.training for module in (codec,stats,ce) for child in module.modules()))
            self.assertTrue(all(not p.requires_grad for module in (codec,stats,ce) for p in module.parameters()))

    def test_real_atomic_io_lock_and_owned_paths_preserve_prior_records(self):
        path = self.output/'training'/'record.json'
        runtime.atomic_json(path,dict(value=1),exclusive=True)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            runtime.atomic_json(path,dict(value=2),exclusive=True)
        with mock.patch.object(runtime.os,'replace',side_effect=OSError('failure')):
            with self.assertRaises(OSError):
                runtime.atomic_json(path,dict(value=2))
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(list(path.parent.iterdir()),[path])
        foreign = self.root/'old_results'
        foreign.mkdir()
        (self.output/'linked').symlink_to(foreign,target_is_directory=True)
        for target in (self.output,foreign/'record.json',self.output/'linked'/'record.json'):
            with self.assertRaises(ValueError):
                runtime.atomic_json(target,dict(value=3))
        lock = self.output/'controller.lock'
        with runtime.process_lock(lock):
            with self.assertRaises(BlockingIOError):
                with runtime.process_lock(lock): pass
        with runtime.process_lock(lock):
            self.assertTrue(lock.is_file())

    def test_real_54_file_source_guard_rejects_changes_to_a_pinned_hash(self):
        local = ('configs/twc_v1_k8_ul_adapt_config.py','ul_adapt_runtime.py')
        with mock.patch.object(cfg,'CORE_FILES',local):
            identity = runtime.core_identity()
            runtime.guard_core(identity)
            changed = copy.deepcopy(cfg.PINNED_CALIBRATION_CORE)
            changed['sha256'][next(iter(changed['sha256']))] = '0'*64
            with mock.patch.object(cfg,'PINNED_CALIBRATION_CORE',changed):
                with self.assertRaises(ValueError):
                    runtime.guard_core(identity)


if __name__ == '__main__':
    unittest.main()
