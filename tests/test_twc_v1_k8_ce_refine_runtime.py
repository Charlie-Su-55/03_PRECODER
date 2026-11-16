"""Real CPU source loading and strict source/mode/random boundary checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
import ce_refine_runtime as rt
from models.twc_v1_k8_icc_aligned import K8ICCAlignedPilotUNet


class CEOnlyRuntimeTests(unittest.TestCase):
    def test_random_frames_match_between_objectives_and_gamma_cannot_be_replaced(self):
        a=rt.batch_random('ce_train',2,'cpu')
        b=rt.batch_random('ce_train',2,'cpu')
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(a[0].tolist(),[20.,25.,0.,5.])
        with self.assertRaises(ValueError):rt.batch_random('ce_train',2,'cpu',gamma=0)
        with self.assertRaises(ValueError):rt.batch_random('final',0,'cpu')
        midpoint=rt.batch_random('midpoint',1,'cpu',gamma=2.5)
        self.assertEqual(midpoint[0].tolist(),[2.5]*4)

    def test_output_paths_cannot_escape_or_follow_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'study';root.mkdir()
            with mock.patch.object(rt.cfg,'OUTPUT_ROOT',root):
                self.assertEqual(rt.inside(root/'record.json'),root/'record.json')
                for path in (root,root/'..'/'outside.json'):
                    with self.assertRaises(ValueError):rt.inside(path)
                (root/'linked').symlink_to(root,target_is_directory=True)
                with self.assertRaises(ValueError):rt.inside(root/'linked'/'record.json')
                rt.atomic_json(root/'record.json',{'x':1},exclusive=True)
                with self.assertRaises(FileExistsError):rt.atomic_json(root/'record.json',{'x':2},exclusive=True)

    def test_strict_loader_reads_only_CE_and_refuses_changed_source_files(self):
        # Explicitly constructed CPU source fixture exercises the real loader.
        # It is never persisted in runs/ or represented as server training.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9321);model=K8ICCAlignedPilotUNet()
            payload=dict(format=rt.original.cfg.NAMESPACE,component='ce',step=5000,
                core=rt.cfg.SOURCE_CORE,training_code={'commit':rt.cfg.SOURCE_COMMIT},
                recipe=rt.original.cfg.recipe('ce'),fingerprint=rt.original.cfg.fingerprint('ce'),
                counters=rt.original.cfg.counters('ce',5000),statistics=None,ce_identity=None,
                data_protocol={'seed_namespace':rt.original.cfg.NAMESPACE},model=model.state_dict())
            torch.save(payload,root/'checkpoint_latest.pt')
            record=dict(status='completed',component='ce',step=5000,
                checkpoint_sha256=rt.sha256(root/'checkpoint_latest.pt'),core=rt.cfg.SOURCE_CORE,
                fingerprint=payload['fingerprint'],counters=payload['counters'])
            for name in ('manifest.json','terminal_receipt.json'):
                (root/name).write_text(json.dumps(record))
            initial=dict(component='ce',core=rt.cfg.SOURCE_CORE,fingerprint=payload['fingerprint'],
                recipe=payload['recipe'],CE_identity=None)
            (root/'initialization.json').write_text(json.dumps(initial))
            with mock.patch.multiple(rt.cfg,SOURCE_ROOT=root,
                    SOURCE_CHECKPOINT_SHA256=rt.sha256(root/'checkpoint_latest.pt'),
                    SOURCE_TERMINAL_SHA256=rt.sha256(root/'terminal_receipt.json'),
                    SOURCE_STATE_SHA256=rt.state_digest(model)):
                with mock.patch.object(rt,'make_data',side_effect=AssertionError('source loader must not build Sionna')):
                    loaded,identity=rt.load_source('cpu')
                self.assertEqual(rt.state_digest(loaded),rt.state_digest(model))
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                self.assertFalse(loaded.training)
                self.assertEqual(len(identity['frozen_files']),4)
                rt.verify_source(identity)
                (root/'initialization.json').write_text('{}')
                with self.assertRaises(ValueError):rt.verify_source(identity)
                with self.assertRaises(ValueError):rt.load_source('cpu')


if __name__=='__main__':unittest.main()
