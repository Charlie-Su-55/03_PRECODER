"""Physical-split component tests only: no full codec, real weights or CDL."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
TORCH_PRESENT=importlib.util.find_spec('torch') is not None
IMPORT_ERROR=None
if TORCH_PRESENT:
    import torch
    try:
        from models import twc_v1_k8_physical_split as physical
        from models.twc_v1_k8_neural import _protocol as protocol
    except ModuleNotFoundError as error:IMPORT_ERROR=str(error)
else:IMPORT_ERROR='PyTorch absent; physical-split tiny numerical tests not executed'


@unittest.skipIf(IMPORT_ERROR is not None,IMPORT_ERROR or '')
class DetachedInterfaceTests(unittest.TestCase):
    def _fixture(self):
        generator=torch.Generator().manual_seed(20261109)
        head,bridge=physical.make_physical_modules(width=8)
        context=torch.randn(2,2,3,8,generator=generator,requires_grad=True)
        soft=torch.complex(torch.randn(2,2,3,generator=generator),torch.randn(2,2,3,generator=generator)).requires_grad_(True)
        semantic_head=torch.nn.Linear(8,2)
        raw=semantic_head(context)
        mask=torch.tensor([[True,True,False],[True,True,False]])
        z_sem=torch.complex(raw[...,0],raw[...,1]).masked_fill(~mask[None],0)
        reliability=torch.full_like(soft.real,.7)
        def recover(value):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(value.requires_grad)
            return dict(correction=value*(.1+.2j),residual_energy=value.abs().square().mean((1,2)))
        return head,bridge,context,soft,semantic_head,z_sem,reliability,mask,recover

    def test_production_mlp_parameter_counts_zero_outputs_and_normal_hidden_initialization(self):
        head,bridge=physical.make_physical_modules()
        self.assertEqual(sum(p.numel() for p in head.parameters()),8386)
        self.assertEqual(sum(p.numel() for p in bridge.parameters()),514)
        for module in (head,bridge):
            self.assertGreater(float(module[0].weight.abs().sum()),0.)
            self.assertEqual(float(module[-1].weight.abs().sum()+module[-1].bias.abs().sum()),0.)

    def test_zero_initialization_replays_semantic_values_and_masks_padding(self):
        head,bridge,context,soft,_,z_sem,reliability,mask,recover=self._fixture()
        values=physical.physical_split_values(head,bridge,soft,context,z_sem,reliability,mask,recover,bridge_enabled=True)
        self.assertTrue(torch.equal(values['x_phy'],soft.detach().masked_fill(~mask[None],0)))
        self.assertTrue(torch.equal(values['z_recon'],z_sem))
        self.assertFalse(values['d_phy'].requires_grad)
        with torch.no_grad():head[-1].bias.copy_(torch.tensor([.1,-.2]))
        values=physical.physical_split_values(head,bridge,soft,context,z_sem,reliability,mask,recover,bridge_enabled=True)
        self.assertEqual(float(values['x_phy'][...,2].abs().max()),0.)
        self.assertEqual(float(values['z_recon'][...,2].abs().max()),0.)

    def test_two_losses_have_disjoint_gradients_without_retaining_shared_graph(self):
        head,bridge,context,soft,semantic_head,z_sem,reliability,mask,recover=self._fixture()
        values=physical.physical_split_values(head,bridge,soft,context,z_sem,reliability,mask,recover,bridge_enabled=True)
        physical_loss=(values['x_phy']-(1+.5j)).abs().square().mean()
        physical_loss.backward()
        self.assertGreater(float(head[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in bridge.parameters()))
        self.assertTrue(all(p.grad is None for p in semantic_head.parameters()))
        self.assertIsNone(context.grad);self.assertIsNone(soft.grad)
        head_grad=[p.grad.clone() if p.grad is not None else None for p in head.parameters()]
        csi_surrogate=(values['z_recon']-(.2+.4j)).abs().square().mean()
        csi_surrogate.backward()  # No retain_graph: the intentional detach separates both graphs.
        for before,parameter in zip(head_grad,head.parameters()):
            self.assertTrue(torch.equal(before,parameter.grad) if before is not None else parameter.grad is None)
        self.assertGreater(float(bridge[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(bool(torch.isfinite(bridge[-1].weight.grad).all()))
        self.assertEqual(float(bridge[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(semantic_head.weight.grad.abs().sum()),0.)
        self.assertTrue(bool(torch.isfinite(context.grad).all()))

    def test_nonzero_bridge_uses_physical_inputs_but_bridge_off_is_independent(self):
        head,bridge,context,soft,_,z_sem,reliability,mask,recover=self._fixture()
        with torch.no_grad():
            head[-1].bias.copy_(torch.tensor([.1,-.2]))
            bridge[-1].weight.fill_(.05)
        normal=physical.physical_split_values(head,bridge,soft,context,z_sem,reliability,mask,recover,bridge_enabled=True)
        replaced=physical.physical_split_values(head,bridge,soft,context,z_sem,reliability,mask,recover,
                                               bridge_enabled=True,physical_from_soft=True)
        self.assertGreater(float((normal['z_recon']-replaced['z_recon']).abs().max()),0.)
        for from_soft in (False,True):
            disabled=physical.physical_split_values(head,bridge,soft,context,z_sem,reliability,mask,recover,
                                                   bridge_enabled=False,physical_from_soft=from_soft)
            self.assertTrue(torch.equal(disabled['z_recon'],z_sem))

    def test_explicit_diagnostic_options_restore_on_exception_and_control_cannot_enable_bridge(self):
        options=SimpleNamespace(variant='split',bridge_override=None,physical_from_soft=False)
        with self.assertRaises(RuntimeError):
            with physical.K8PhysicalSplitReceiver.diagnostic_options(options,bridge_enabled=False,physical_from_soft=True):
                self.assertFalse(options.bridge_override);self.assertTrue(options.physical_from_soft)
                raise RuntimeError('synthetic diagnostic exit')
        self.assertIsNone(options.bridge_override);self.assertFalse(options.physical_from_soft)
        control=SimpleNamespace(variant='control',bridge_override=None,physical_from_soft=False)
        with self.assertRaises(ValueError):
            with physical.K8PhysicalSplitReceiver.diagnostic_options(control,bridge_enabled=True):pass


@unittest.skipIf(IMPORT_ERROR is not None,IMPORT_ERROR or '')
class PhysicalLabelAndMigrationTests(unittest.TestCase):
    def _output(self,layout,batch=1):
        from tests.test_twc_v1_k8_decoder_screen import OfflineMetricAndFreezeTests
        output,truth=OfflineMetricAndFreezeTests()._output(layout)
        if batch>1:
            output['tx']={key:value.repeat(batch,1,1) for key,value in output['tx'].items()}
            truth=truth.repeat(batch,1,1)
            output['Y_rx']=output['Y_rx'].repeat(batch,1,1)
            output['H_ul_est']=output['H_ul_est'].repeat(batch,1,1,1)
            output['corrected_symbols']=output['corrected_symbols'].repeat(batch,1,1)
            output['soft_symbols']={key:value.repeat(batch,1,1) for key,value in output['soft_symbols'].items()}
        _,kind,_,active,_,_=physical.residual_ops.packed_layout_map(layout)
        output['receiver_execution']['stages']=dict(soft=.5*truth,x_phy=.5*truth,z_sem=3*truth,z_recon=2*truth,
            active_mask=active,kind=kind)
        return output,truth

    def test_fdma_uses_all24_data_slots_and_detaches_actual_transmitted_labels(self):
        output,truth=self._output(protocol.FDMALayout())
        label=output['tx']['data_symbols_tx'].requires_grad_(True)
        estimate=(.5*truth).detach().clone().requires_grad_(True)
        output['receiver_execution']['stages']['x_phy']=estimate
        loss=physical.physical_codeword_loss(output)
        self.assertAlmostEqual(float(loss),.25,places=6)
        loss.backward()
        self.assertIsNone(label.grad)
        self.assertTrue(bool(torch.isfinite(estimate.grad).all()))
        self.assertGreater(float(estimate.grad.abs().sum()),0.)
        self.assertEqual(float(estimate.grad[...,24:].abs().sum()),0.)

    def test_hybrid_averages_normalized_key_shared_errors_equally_then_complete_mu_batch(self):
        output,truth=self._output(protocol.default_hybrid_layout(4),batch=2)
        estimate=truth.clone()
        estimate[0,:,:4]*=2.;estimate[0,:,4:44]*=1.1
        estimate[1,:,:4]*=3.;estimate[1,:,4:44]*=1.2
        estimate[...,44:]=999.  # Padding cannot contribute to the supervised loss.
        estimate.requires_grad_(True)
        output['receiver_execution']['stages']['x_phy']=estimate
        loss=physical.physical_codeword_loss(output)
        self.assertAlmostEqual(float(loss),(.5*(1.+.01)+.5*(4.+.04))/2,places=5)
        loss.backward()
        self.assertEqual(float(estimate.grad[...,44:].abs().sum()),0.)

    def test_diagnostics_separate_linear_physical_error_from_semantic_latent_distance(self):
        for layout in (protocol.FDMALayout(),protocol.default_hybrid_layout(4)):
            output,_=self._output(layout)
            metrics=physical.physical_diagnostics(output)
            self.assertAlmostEqual(float(metrics['physical_soft_codeword_nmse_linear_all']),.25,places=6)
            self.assertAlmostEqual(float(metrics['physical_head_codeword_nmse_linear_all']),.25,places=6)
            self.assertAlmostEqual(float(metrics['latent_z_sem_distance_linear_all']),4.,places=6)
            self.assertAlmostEqual(float(metrics['latent_z_recon_distance_linear_all']),1.,places=6)
            self.assertTrue(all(not value.requires_grad for value in metrics.values()))
            self.assertEqual(len(metrics),10 if hasattr(layout,'data_positions') else 14)
            output['receiver_execution']['stages']={}
            old=physical.physical_diagnostics(output)
            self.assertFalse(any(name.startswith('physical_head_') for name in old))
            self.assertNotIn('rx_residual_physical_mean_power_unnormalized',old)
            self.assertEqual(len(old),7 if hasattr(layout,'data_positions') else 10)

    def _codecs(self):
        from tests.test_twc_v1_k8_decoder_screen import NewModuleAndMigrationTests
        source=NewModuleAndMigrationTests()._codec('base2')
        target=deepcopy(source)
        target.joint_decoder.physical_head,target.joint_decoder.physical_bridge=physical.make_physical_modules(width=8)
        return source,target

    def test_strict_migration_preserves_all_old_keys_and_decoder_aliases(self):
        source,target=self._codecs();state=deepcopy(source.state_dict())
        record=physical.migrate_source_state(target,state)
        self.assertTrue(record['strict']);self.assertTrue(record['source_values_preserved'])
        self.assertEqual(set(record['added_keys']),set(physical.new_state_keys()))
        self.assertEqual(len(record['added_keys']),8)
        self.assertIs(target.decoder,target.joint_decoder.base_decoder)
        for key,value in state.items():self.assertTrue(torch.equal(target.state_dict()[key],value),key)
        conflicting=deepcopy(state)
        conflicting['joint_decoder.base_decoder.weight']=conflicting['joint_decoder.base_decoder.weight']+1
        with self.assertRaises(ValueError):physical.migrate_source_state(target,conflicting)
        for issue in ('missing','shape','dtype','nonfinite'):
            changed=deepcopy(state)
            if issue=='missing':changed.pop('encoder.weight')
            elif issue=='shape':changed['encoder.weight']=torch.zeros(7)
            elif issue=='dtype':changed['encoder.weight']=changed['encoder.weight'].double()
            else:changed['encoder.weight'].fill_(float('nan'))
            with self.subTest(issue=issue),self.assertRaises(ValueError):physical.migrate_source_state(target,changed)

    def test_normal_strict_resume_keeps_trained_head_and_bridge_and_never_fills_missing_keys(self):
        source,target=self._codecs();physical.migrate_source_state(target,deepcopy(source.state_dict()))
        with torch.no_grad():
            target.joint_decoder.physical_head[-1].weight.fill_(.2)
            target.joint_decoder.physical_bridge[-1].weight.fill_(-.1)
        checkpoint=deepcopy(target.state_dict());_,restored=self._codecs()
        restored.load_state_dict(checkpoint,strict=True)
        for key in physical.new_state_keys():self.assertTrue(torch.equal(restored.state_dict()[key],checkpoint[key]))
        with self.assertRaises(ValueError):physical.migrate_source_state(restored,checkpoint)
        missing=deepcopy(checkpoint);missing.pop('joint_decoder.physical_bridge.2.bias')
        with self.assertRaises(RuntimeError):restored.load_state_dict(missing,strict=True)


if __name__=='__main__':unittest.main()
