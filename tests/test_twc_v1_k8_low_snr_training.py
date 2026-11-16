"""Actual CPU UE/MAC/recovery/BS/RZF gradients and causal checkpoint replay."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import train_twc_v1_k8_low_snr as train
from evaluate_twc_v1_k8_icc_aligned import _tensor_hash


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.angle = torch.nn.Parameter(torch.tensor(.17))
        self.register_buffer('shape', torch.cos(torch.arange(216).float()/5.))

    def forward(self, h):
        real = (1.+self.angle*self.shape)[None,None].expand(h.shape[0],8,-1)
        return torch.complex(real, .07*real.flip(-1))


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coupling = torch.nn.Parameter(torch.tensor(.22))
        self.snr_slope = torch.nn.Parameter(torch.tensor(.03))
        base = torch.zeros(1,8,32,256,dtype=torch.complex64)
        base[:,torch.arange(8),torch.arange(8),:] = 1.
        mix = torch.zeros_like(base)
        mix[:,0,1,:] = 1.
        mix[:,1,0,:] = 1.
        self.register_buffer('base',base)
        self.register_buffer('mix',mix)

    def forward(self, soft, gamma):
        observed = soft[...,0].real.mean(-1)
        gain = self.coupling + .06*observed + self.snr_slope*(gamma/25.-.5)
        return self.base+self.mix*gain[:,None,None,None]


class LowSNRTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = train.capture_rng()
        self.addCleanup(train.restore_rng,rng)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root/'joint'
        self.core = dict(schema=train.cfg.NAMESPACE+'_core_v1',sha256={'CPU_fixture':'a'*64})
        self.patches = []
        for patcher in (
            mock.patch.object(train.cfg,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.runtime,'OUTPUT_ROOT',self.output),
            mock.patch.object(train.cfg,'UPDATES',4),
            mock.patch.object(train.cfg,'SAVE_EVERY',2),
            mock.patch.object(train.cfg,'ACCUMULATION',8),


            mock.patch.object(train.cfg,'ROLE_RANGES',dict(codec_train=[1,32],final=[4096,4351])),
            mock.patch.object(train.runtime,'core_identity',return_value=self.core),
            mock.patch.object(train.runtime,'source_identity',return_value=dict(commit='b'*40,sha256=self.core['sha256'])),
            mock.patch.object(train.runtime,'guard_core',side_effect=lambda value:self.assertEqual(value,self.core))):
            patcher.start()
            self.addCleanup(patcher.stop)
        torch.manual_seed(379)
        codec = torch.nn.Module()
        codec.encoder,codec.decoder = TinyEncoder(),TinyDecoder()
        phy = train.runtime.physics()
        stats = phy.ReceiverStatistics(phy.make_layout('hybrid_g4'),torch.full((6,8,256),.002))
        stats.ema_updates.fill_(40000)
        ce = torch.nn.Linear(2,2)
        for module in (codec,stats,ce):
            module.eval().requires_grad_(False)
        self.base = codec,stats,ce
        self.source_path = self.root/'explicit_CPU_fixture.pt'
        torch.save(dict(model=codec.state_dict(),statistics=stats.state_dict()),self.source_path)
        self.source_bytes = self.source_path.read_bytes()
        self.source = dict(arm='hybrid_g4',base_ema_updates=40000,
            base_encoder_state_sha256=train.state_digest(codec.encoder),
            base_decoder_state_sha256=train.state_digest(codec.decoder),
            base_statistics_state_sha256=train.state_digest(stats),
            frozen_files=[dict(path=str(self.source_path),sha256=train.sha256(self.source_path))])
        self.trajectory = train.cfg.TRAJECTORIES[0]
        self.loaded,self.requests,self.forward_calls = [],[],[]
        outer = self
        class Data:
            def __init__(self):
                self.protocol = dict(fixture='actual_CPU_MAC_RZF',stream='train')
            def address(self,role,index):
                return dict(namespace=self.protocol['stream'],role=role,index=index)
            def sample(self,role,index):
                if role not in ('codec_train','validation'):
                    raise AssertionError('Unexpected data role')
                outer.requests.append((self.protocol['stream'],role,index))
                h = outer.base[0].decoder.base.expand(4,-1,-1,-1).clone()
                h[:,0,1,:] = .01*(index%7)
                ul = outer.base[0].decoder.base.expand_as(h).clone()
                ul[:,torch.arange(8),torch.arange(8),:] *= complex(1.,.003*(index%5))
                return dict(H_dl=h,H_ul=ul,address=self.address(role,index),
                    sample_fingerprint=train.cfg.digest([_tensor_hash(h),_tensor_hash(ul)]))
        self.data = Data()
        patcher = mock.patch.object(train.runtime,'verify_files',side_effect=self.verify_files)
        patcher.start()
        self.addCleanup(patcher.stop)

    def verify_files(self, record):
        for item in record['frozen_files']:
            if train.sha256(item['path']) != item['sha256']:
                raise ValueError('Fixture source bytes changed')

    def loader(self, arm, device):
        self.assertEqual((arm,str(device)),('hybrid_g4','cpu'))
        result = copy.deepcopy(self.base)
        self.loaded.append(result)
        return (*result,copy.deepcopy(self.source))

    def data_factory(self, device):
        self.assertEqual(str(device),'cpu')
        return self.data

    def forward(self, codec, stats, ce, sample, gamma_db, noise_key):
        phy = train.runtime.physics()
        role,index,gi = noise_key
        self.forward_calls.append((role,index,int(stats.ema_updates),train.state_digest(stats)))
        if role=='codec_train':
            self.assertTrue(codec.encoder.training and codec.decoder.training and stats.training)
            self.assertTrue(torch.is_grad_enabled())
        else:
            self.assertFalse(codec.encoder.training or codec.decoder.training or stats.training)
            self.assertFalse(torch.is_grad_enabled())
        self.assertFalse(ce.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in ce.parameters()))
        random.random();np.random.rand();torch.rand(1)
        ri = index if role=='codec_train' else index*len(train.cfg.GAMMAS)+gi
        gamma,noise,polarity = train.runtime.batch_random(role,ri,'cpu',gamma=gamma_db)
        raw = codec.encoder(sample['H_dl'])
        wave = phy.build_waveform(raw,stats.layout,polarity=polarity)
        Y = phy.transmit(sample['H_ul'],wave.x_grid,noise,gamma)
        # This declared fixture uses a synthetic known estimated channel;
        # the production forward's pilot-only CE path is tested separately.
        estimated = sample['H_ul'].detach().clone()
        prior,cevar = stats.nominal_power(),stats.ce_variance(gamma)
        recovered = phy.recover(Y,estimated,stats.layout,prior,cevar,gamma)
        hhat = codec.decoder(recovered['soft'],gamma)
        return dict(H_hat=hhat,soft=recovered['soft'],reliability=recovered['reliability'],
            H_ul_est=estimated,source_estimated_UL=estimated,recovered=recovered,
            waveform=wave,layout=stats.layout,gamma_db=gamma,
            input_hashes=dict(standard_noise=_tensor_hash(noise),pilot_polarity=_tensor_hash(polarity),
                gamma=_tensor_hash(gamma),wave_active=_tensor_hash(wave.active),Y_rx=_tensor_hash(Y),
                estimated_UL=_tensor_hash(estimated),CE_variance=_tensor_hash(cevar),prior=_tensor_hash(prior)))

    def run_train(self, target=2, resume=False, **kwargs):
        kwargs.setdefault('forward', self.forward)
        return train.train(self.trajectory,target,resume,require_cuda=False,loader=self.loader,
            data_factory=self.data_factory,**kwargs)

    def directory(self):
        return train.cfg.OUTPUT_ROOT/'training'/self.trajectory

    def payload(self):
        return torch.load(self.directory()/'checkpoint_latest.pt',map_location='cpu',weights_only=False)

    def verify(self,payload,manifest,**kwargs):
        codec,stats,ce = self.loaded[-1]
        kwargs.setdefault('allow_fixture', True)
        return train.verify_saved(self.directory(),payload,manifest,trajectory=self.trajectory,data=self.data,
            core=self.core,source=self.source,frozen=train.runtime.frozen_state(codec,ce,stats),
            **kwargs)

    def zero_boundary(self):
        def stop(*args, **kwargs):
            raise RuntimeError('intentional first-forward stop')
        with self.assertRaisesRegex(RuntimeError, 'first-forward'):
            self.run_train(forward=stop)
        return train.read_json(self.directory()/'manifest.json')

    def test_actual_joint_update_and_exact_full_state_resume(self):
        first = self.run_train()
        saved = self.payload()
        self.assertEqual((first['step'], first['status']), (2, 'paused_boundary'))
        self.assertFalse(first['production_evidence'])
        self.assertEqual(saved['counters'], train.cfg.counters(2, 40000))
        self.assertEqual(saved['counters']['FDMA_updates'], 0)
        self.assertEqual(int(saved['statistics']['ema_updates']), 40002)
        self.assertNotEqual(saved['model']['encoder.angle'], self.base[0].encoder.angle)
        self.assertNotEqual(saved['model']['decoder.coupling'], self.base[0].decoder.coupling)
        self.assertEqual(self.verify(saved, first), 2)
        self.assertEqual(train.state_digest(self.loaded[-1][2]), train.state_digest(self.base[2]))
        self.assertTrue(torch.equal(saved['statistics']['ce_errors'], self.base[1].ce_errors))
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], 1e-5)
        journal = [json.loads(line) for line in (self.directory()/'progress.jsonl').read_text().splitlines()]
        self.assertGreater(journal[-1]['direction_error'], 0)
        self.assertGreater(journal[-1]['dl_leakage_penalty'],0.)
        self.assertAlmostEqual(journal[-1]['weighted_direction_loss'], 100*journal[-1]['direction_error'])
        self.assertAlmostEqual(journal[-1]['loss'], -journal[-1]['sum_rate_bps_hz']+journal[-1]['weighted_direction_loss']+journal[-1]['weighted_interference_loss'], places=6)
        self.assertEqual(journal[-1]['weighted_interference_loss'],0.)
        final = self.run_train(target=4, resume=True)
        resumed = self.payload()
        self.assertEqual(self.verify(resumed, final, require_final=True), 4)
        self.assertEqual(int(resumed['statistics']['ema_updates']), 40004)
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)
        self.assertEqual([i for stream, role, i in self.requests if role == 'codec_train'], list(range(1, 33)))
        for offset in range(0, len(self.forward_calls), 8):
            self.assertTrue(all(self.forward_calls[offset][2:] == self.forward_calls[offset+j][2:] for j in range(8)))
        with mock.patch.object(train.cfg, 'OUTPUT_ROOT', self.root/'continuous'), mock.patch.object(
                train.runtime, 'OUTPUT_ROOT', self.root/'continuous'):
            self.run_train(target=4)
            continuous = self.payload()
        for name in ('model', 'statistics'):
            self.assertEqual(train.state_digest(resumed[name]), train.state_digest(continuous[name]))
        self.assertEqual(resumed['training_trace'], continuous['training_trace'])
        self.assertTrue(torch.equal(resumed['rng']['torch_cpu'], continuous['rng']['torch_cpu']))
        self.assertEqual(resumed['rng']['python'], continuous['rng']['python'])
        self.assertEqual(resumed['scheduler'], continuous['scheduler'])
        for key, state in resumed['optimizer']['state'].items():
            for field, value in state.items():
                self.assertTrue(torch.equal(value, continuous['optimizer']['state'][key][field]))

    def test_step_zero_source_and_fresh_optimizer_survive_first_forward_failure(self):
        manifest = self.zero_boundary()
        payload = self.payload()
        self.assertEqual(payload['optimizer']['state'], {})
        self.assertEqual(int(payload['statistics']['ema_updates']), 40000)
        self.assertEqual(self.verify(payload, manifest), 0)
        self.assertEqual(self.requests[0][1:], ('codec_train', 1))
        self.assertEqual(self.source_bytes, self.source_path.read_bytes())
        self.run_train(resume=True)
        self.assertEqual(int(self.payload()['statistics']['ema_updates']), 40002)

    def test_paired_trajectories_share_source_inputs_rng_and_only_auxiliary_changes(self):
        control = self.run_train()
        control_payload = self.payload()
        control_init = train.read_json(self.directory()/'initialization.json')
        self.trajectory = train.cfg.TRAJECTORIES[1]
        experiment = self.run_train()
        experiment_payload = self.payload()
        experiment_init = train.read_json(self.directory()/'initialization.json')
        self.assertEqual(control_payload['training_trace'], experiment_payload['training_trace'])
        self.assertEqual(control_payload['source_identity'], experiment_payload['source_identity'])
        self.assertEqual(control_payload['frozen_identity'], experiment_payload['frozen_identity'])
        self.assertEqual(control_payload['counters'], experiment_payload['counters'])
        for name in ('initial_model_sha256', 'source_encoder_sha256', 'source_decoder_sha256',
                'source_statistics_sha256', 'training_trace_initial_sha256', 'data_protocol'):
            self.assertEqual(control_init[name], experiment_init[name])
        self.assertNotEqual(control['fingerprint'], experiment['fingerprint'])
        self.assertEqual(control['method'], 'low_snr_control')
        self.assertEqual(experiment['method'], 'low_snr_interference')
        journal = [json.loads(line) for line in (self.directory()/'progress.jsonl').read_text().splitlines()]
        self.assertGreater(journal[-1]['weighted_interference_loss'], 0.)
        self.assertAlmostEqual(journal[-1]['loss'], -journal[-1]['sum_rate_bps_hz']+
            journal[-1]['weighted_direction_loss']+journal[-1]['weighted_interference_loss'], places=6)
        with self.assertRaises(ValueError):
            self.verify(control_payload, control)
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)

    def test_incomplete_accumulation_and_nonfinite_gradients_do_not_commit_ema(self):
        def stopped(codec, stats, ce, sample, gamma_db, noise_key):
            if noise_key[1] == 2: raise RuntimeError('partial accumulation')
            return self.forward(codec, stats, ce, sample, gamma_db, noise_key)
        with self.assertRaisesRegex(RuntimeError, 'partial accumulation'):
            self.run_train(forward=stopped)
        self.assertEqual(int(self.payload()['statistics']['ema_updates']), 40000)
        self.assertEqual(int(self.loaded[-1][1].ema_updates), 40000)
        codec, stats, ce = copy.deepcopy(self.base)
        train.runtime.configure_joint(codec, ce, stats)
        optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        for p in codec.parameters(): p.grad = torch.full_like(p, float('nan'))
        before = (train.state_digest(codec), train.state_digest(stats))
        with self.assertRaises(FloatingPointError):
            train.commit_update(codec, optimizer, scheduler, stats=stats,
                power_sum=torch.full_like(stats.power_ema, 48.*8/44), frames=8,
                previous_updates=0, base_ema_updates=40000)
        self.assertEqual(before, (train.state_digest(codec), train.state_digest(stats)))
        self.assertEqual(optimizer.state, {})

    def test_real_serialization_failure_preserves_saved_checkpoint_and_cleans_temporary_file(self):
        self.run_train()
        target = self.directory()/'checkpoint_latest.pt'
        before = target.read_bytes()
        manifest_before = (self.directory()/'manifest.json').read_bytes()
        actual_save = train.runtime.atomic_torch_save
        def serialization_failure(payload, path):
            # Actual torch.save is called and fails on an unpicklable payload.
            actual_save(dict(payload, unpicklable=lambda: None), path)
        with mock.patch.object(train.runtime, 'atomic_torch_save', side_effect=serialization_failure):
            with self.assertRaises((AttributeError, TypeError, RuntimeError)):
                self.run_train(target=4, resume=True)
        self.assertEqual(before, target.read_bytes())
        self.assertEqual(manifest_before, (self.directory()/'manifest.json').read_bytes())
        self.assertEqual(list(self.directory().glob('checkpoint_*.pt')), [target])
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)
        self.assertEqual(torch.load(target, map_location='cpu', weights_only=False)['step'], 2)

    def test_strict_resume_refuses_changed_recipe_state_optimizer_rng_trace_and_fixture_promotion(self):
        manifest = self.run_train()
        payload = self.payload()
        mutations = [lambda p: p.update(arm='fdma'), lambda p: p.update(method='rate_only'),
            lambda p: p['recipe'].update(namespace='changed'),
            lambda p: p['statistics']['ema_updates'].fill_(40001),
            lambda p: p['statistics']['power_ema'].fill_(float('nan')),
            lambda p: p['model'].update({'encoder.angle': torch.zeros(2)}),
            lambda p: p['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
            lambda p: p['scheduler'].update(last_epoch=999),
            lambda p: p['rng'].update(torch_cpu=torch.tensor([1], dtype=torch.uint8)),
            lambda p: p['training_trace'].update(sha256='0'*64)]
        for i, mutate in enumerate(mutations):
            changed = copy.deepcopy(payload); mutate(changed)
            with self.subTest(mutation=i), self.assertRaises(ValueError): self.verify(changed, manifest)
        with self.assertRaisesRegex(ValueError, 'fixture'):
            self.verify(payload, manifest, allow_fixture=False)
        self.run_train(target=4, resume=True)
        with mock.patch.object(train.runtime, 'load_source', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'fixture'): train.load_finetuned(self.trajectory,'cpu')
        with self.assertRaisesRegex(ValueError, 'production'):
            train.verify_terminal_record(self.trajectory)

    def test_only_hybrid_entrypoint_and_immutable_terminal_receipt(self):
        with self.assertRaises(SystemExit): train.main(['--train', '--trajectory', 'fdma'])
        self.run_train(target=4)
        manifest = train.read_json(self.directory()/'manifest.json')
        self.assertEqual(train.read_json(self.directory()/'terminal_receipt.json'), manifest)
        before = (self.directory()/'terminal_receipt.json').read_bytes()
        train.ensure_boundary_receipts(self.directory(), manifest)
        self.assertEqual((self.directory()/'terminal_receipt.json').read_bytes(), before)
        changed = copy.deepcopy(manifest); changed['checkpoint_sha256'] = '0'*64
        with self.assertRaises(ValueError): train.ensure_boundary_receipts(self.directory(), changed)


if __name__ == '__main__': unittest.main()
