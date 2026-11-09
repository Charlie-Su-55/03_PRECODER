"""Light admission-helper tests; no complete model or CUDA execution locally."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import preflight_twc_v1_k8_icc_aligned as preflight


class AdmissionPlanTests(unittest.TestCase):
    def test_plan_has_zero_updates_and_no_claim_of_training_peak(self):
        plan=preflight.plan()
        self.assertEqual(plan["optimizer_updates"],0)
        self.assertFalse(plan["performance_claim"])
        self.assertFalse(plan["full_model_local_execution_allowed"])
        self.assertEqual(plan["synthetic_MU_frames"],4)

    def test_dry_run_is_standard_library_only(self):
        command=[sys.executable,"-B","-S",str(ROOT/"preflight_twc_v1_k8_icc_aligned.py"),"--dry-run"]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
        plan=json.loads(result.stdout)
        self.assertEqual(plan["optimizer_updates"],0)
        self.assertIn("CUDA",plan["device"])

    def test_default_only_help(self):
        result=subprocess.run([sys.executable,"-B","-S",str(ROOT/"preflight_twc_v1_k8_icc_aligned.py")],
            cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertIn("--dry-run",result.stdout)
        self.assertIn("--run",result.stdout)


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch absent; tiny common-initialization/gradient checks not executed")
class TinyAdmissionTests(unittest.TestCase):
    def test_actual_storage_and_parameter_values_checked(self):
        import torch
        a=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.LayerNorm(3))
        b=deepcopy(a)
        result=preflight.validate_common_initialization(a,b)
        self.assertTrue(result["all_storage_independent"])
        with self.assertRaisesRegex(ValueError,"share"):
            preflight.validate_common_initialization(a,a)
        with torch.no_grad():
            b[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError,"differs"):
            preflight.validate_common_initialization(a,b)

    def test_gradient_record_rejects_missing_and_zero_accepts_finite_path(self):
        import torch
        layer=torch.nn.Linear(2,3)
        with self.assertRaises(ValueError):
            preflight._gradient_record(layer)
        layer(torch.tensor([[1.,2.]])).square().sum().backward()
        report=preflight._gradient_record(layer)
        self.assertGreater(report["L2_norm"],0)
        self.assertTrue(report["all_present_gradients_finite"])
        for p in layer.parameters():
            p.grad.zero_()
        with self.assertRaisesRegex(ValueError,"zero"):
            preflight._gradient_record(layer)


if __name__=="__main__":
    unittest.main()
