"""Synthetic stdlib archival tests; no real checkpoints or experiments used."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import as_spatial_archive as archive


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    else:
        path.write_bytes(value)
    return path


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(seed):
    return dict(training_seed=seed, seed=20261103, preflight=False, batch_size=8,
                num_batches=200, num_samples=1600, feedback_snr_db=[0,5,10,15,20,25],
                downlink_snr_db=25, channel_profile="CDL-C",
                scenario=dict(K=4,Nt=32,Nsc=256,Dtot=256,Nsb=4),
                dataset_role="independent_screening_test",
                channel_data_protocol=dict(protocol_version="training-channel-data-v1",
                    base_seed=20261103,sionna_version="2.0.1",batch_sizes=dict(training=8,validation=8),
                    physical=dict(k_users=4,nt=32,nsc=256,cdl_model="C"),data_id="e"*64))


def rows(seed):
    summary = [dict(method_key=m,feedback_snr_db=fb,training_seed=seed,seed=20261103,
                    num_samples=1600,preflight=False,sum_rate_mean="21.123456789012345",
                    sum_rate_se="0.123456789012345",sum_rate_ci95_low="20.881481482548",
                    sum_rate_ci95_high="21.365432095476")
               for m in archive.METHODS for fb in (0,5,10,15,20,25)]
    comparisons = [(arm,ref) for arm in archive.ARMS for ref in ("full_control","as_receiver","swin")]
    comparisons.append(("joint","frozen_ue"))
    scopes = [("per_snr",str(fb)) for fb in (0,5,10,15,20,25)] + [
        ("low_snr_mean","0,5,10"),("all_snr_mean","0,5,10,15,20,25")]
    paired = [dict(method_key=m,reference_method_key=ref,estimand=scope,feedback_snr_points=points,
                   training_seed=seed,seed=20261103,num_samples=1600,num_batches=200,training_seeds_pooled=False,
                   delta_mean="-0.250000000000001",delta_se=.1,delta_ci95_low=-.446,delta_ci95_high=-.054,
                   batch_delta_mean=-.25,batch_delta_se=.2,batch_delta_ci95_low=-.642,batch_delta_ci95_high=.142)
              for m,ref in comparisons for scope,points in scopes]
    return summary, paired


def evaluation(root, directory, seed, training=None):
    training = training or {}
    summary, paired = rows(seed)
    contents = {
        f"summary_{archive.TAG}.csv": csv_bytes(summary),
        f"paired_deltas_{archive.TAG}.csv": csv_bytes(paired),
        f"parameters_{archive.TAG}.csv": csv_bytes([
            dict(method_key=m,training_seed=seed,full_parameters=25 if m in archive.ARMS else 100,
                 encoder_plus_condition_parameters=5 if m in archive.ARMS else 20)
            for m in archive.METHODS]),
        f"spatial_parameters_{archive.TAG}.csv": csv_bytes([
            dict(method_key=m,name="decoder.weight",numel=20,l2_norm=1.25) for m in archive.ARMS]),
        f"soft_codeword_summary_{archive.TAG}.csv": b"method_key,soft_mse\nfrozen_ue,0.123456789123456\njoint,0.25\n",
    }
    for name in archive.ARTIFACTS:
        if name not in contents:
            contents[name] = b"raw gzip sentinel: must remain at source" if name.endswith(".gz") else b'{"fixture": true}\n'
    for name,value in contents.items():
        write(directory/name,value)
    methods, inputs = {}, {}
    for method in archive.METHODS:
        if method in training:
            data = training[method]
            record = copy.deepcopy(data["record"])
            checkpoint = Path(record["best_path"])
            methods[method] = dict(config=data["snapshot"],config_fingerprint=data["fingerprint"],
                                  completed_updates=20000,latest_step=20000,selected_best_step=record["best_step"],
                                  latest_sha256=record["latest_sha256"],latest_path=record["latest_path"])
            for path in data["paths"]:
                inputs[str(path)] = digest(path)
        else:
            checkpoint = write(root/"controls"/f"{method}_{seed}.pth",b"NOT A PICKLE; never deserialize")
            methods[method] = {}
        methods[method].update(best_path=str(checkpoint),best_sha256=digest(checkpoint),strict_loading=True)
        inputs[str(checkpoint)] = digest(checkpoint)
    states = {method:dict(buffers={},training={"":False,"decoder":False}) for method in archive.METHODS}
    manifest = dict(status="completed",results_verified=True,protocol=protocol(seed),output_directory=str(directory),
                    freeze_checks=dict(passed=True,errors=[]),model_buffers_before=states,model_buffers_after=copy.deepcopy(states),
                    input_sha256_before=inputs,input_sha256_after=dict(inputs),methods=methods,
                    runtime=dict(source_sha256={"models/historical.py":"a"*64},python="historical"),
                    diagnostic_rng_checks=dict(passed=True),
                    output_sha256={name:digest(directory/name) for name in archive.ARTIFACTS})
    write(directory/"manifest.json",manifest)
    return manifest


def complete_batch(root):
    proposal = root/"runs_as_screen/proposal_as_spatial_v1"
    batch = proposal/"batches/batch_fixture"
    batch.mkdir(parents=True)
    original = write(root/"source/robust.pth",b"synthetic pinned source bytes, not a checkpoint")
    identity = dict(path=str(original),checkpoint_sha256=digest(original),step=49500)
    summary = dict(status="completed",source=dict(commit="b"*40,sha256={"models/historical.py":"a"*64}),
                   stages={},training={})
    def stage(name, report=None, paths=(), result=None):
        if report is not None:
            path = write(batch/(name+".json"),report)
            paths = [path,*paths]
            result = dict(report=report,report_path=str(path))
        summary["stages"][name] = dict(status="completed",result=result or {},
                                      evidence_sha256={str(path):digest(path) for path in paths})
    stage("environment_and_old_controls",dict(passed=True))
    stage("new_model_checks",dict(passed=True,source_sha256=digest(original),
          same_initial_state_between_arms=True,source_ue_exact=True,
          actual_parameters={arm:dict(total=25,ue_encoder_condition=5,new_bs=20) for arm in archive.ARMS}))
    training_by_seed = {}
    for seed in archive.SEEDS:
        training_by_seed[seed] = {}
        for arm in archive.ARMS:
            checkpoint_dir = proposal/"checkpoints"/f"{arm}_{seed}"
            best = write(checkpoint_dir/"best.pth",b"best fixture "+f"{arm}{seed}".encode())
            latest = write(checkpoint_dir/"latest.pth",b"latest fixture "+f"{arm}{seed}".encode())
            snapshot = dict(arm=arm,experiment=dict(seed=seed),allocations=[[0,256]],
                            training_recipe=dict(total_steps=20000),
                            architecture_metadata=dict(architecture="as_spatial_v1"),source_identity=identity)
            fp = hashlib.sha256(json.dumps(snapshot,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
            config = write(checkpoint_dir/"config.json",dict(snapshot,config_fingerprint=fp))
            status_data = dict(state="completed",step=20000,experiment=f"{arm}_{seed}")
            status = write(checkpoint_dir/"status.json",status_data)
            record = dict(passed=True,arm=arm,training_seed=seed,completed_updates=20000,best_step=19500,
                          best_path=str(best),latest_path=str(latest),best_sha256=digest(best),
                          latest_sha256=digest(latest),status=status_data)
            key = f"train_{arm}_{seed}"
            paths = [best,latest,config,status]
            stage(key,record,paths)
            summary["training"][key] = record
            training_by_seed[seed][arm] = dict(record=record,paths=paths,snapshot=snapshot,fingerprint=fp)
    evaluations = {}
    for seed in archive.SEEDS:
        directory = root/"runs_as_screen/evaluation/k4_as_spatial_v1"/batch.name/f"seed_{seed}"/"attempt_fixture"
        evaluation(root,directory,seed,training_by_seed[seed])
        paths = list(directory.iterdir())
        stage(f"evaluate_{seed}",paths=paths,result=dict(output_directory=str(directory)))
        evaluations[str(seed)] = directory
    report = write(batch/"export/engineering_summary.json",{"complete":True})
    stage("organize_results",paths=[report],result=dict(export_directory=str(report.parent)))
    write(batch/"batch_summary.json",summary)
    write(proposal/"last_batch.json",dict(directory=str(batch)))
    # Deliberately different current implementation: historical provenance must
    # be preserved rather than compared against today's files.
    write(root/"models/historical.py",b"current changed implementation")
    return batch, evaluations, summary


class GenericArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root/"evaluations/seed"
        self.record = evaluation(self.root,self.directory,20261030)
        self.options = dict(root=self.root,archive_root=self.root/"paper/data/generic",
                            archive_id="batch_test",expected_protocols={"20261030":self.record["protocol"]},
                            required_artifacts=archive.ARTIFACTS)

    def tearDown(self):
        self.temporary.cleanup()

    def run_archive(self, **updates):
        options = dict(self.options,**updates)
        return archive.archive_evaluations({"20261030":self.directory},**options)

    def update_manifest(self):
        write(self.directory/"manifest.json",self.record)

    def test_complete_original_bytes_and_no_raw_or_weights(self):
        before = {str(p):p.read_bytes() for p in self.directory.iterdir()}
        result = self.run_archive()
        destination = Path(result["archive_directory"])
        for name in archive.ARTIFACTS:
            target = destination/"evaluations/20261030"/name
            if name.endswith(".gz"):
                self.assertFalse(target.exists())
            else:
                self.assertEqual(target.read_bytes(),(self.directory/name).read_bytes())
        self.assertFalse(any(p.suffix in (".pth",".gz") for p in destination.rglob("*")))
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.directory.iterdir()})
        self.assertIn(b"-0.250000000000001",(destination/f"evaluations/20261030/paired_deltas_{archive.TAG}.csv").read_bytes())

    def test_idempotent_archive_never_rewrites(self):
        first = self.run_archive()
        directory = Path(first["archive_directory"])
        times = {str(p):p.stat().st_mtime_ns for p in directory.rglob("*") if p.is_file()}
        second = self.run_archive()
        self.assertTrue(second["reused"])
        self.assertEqual(times,{str(p):p.stat().st_mtime_ns for p in directory.rglob("*") if p.is_file()})

    def test_current_code_hash_need_not_equal_historical(self):
        write(self.root/"models/historical.py",b"newer changed source")
        result = self.run_archive()
        self.assertNotEqual(digest(self.root/"models/historical.py"),"a"*64)
        self.assertEqual(result["historical_source_provenance"]["20261030"]["source_sha256"],
                         {"models/historical.py":"a"*64})

    def test_missing_or_tampered_output_including_raw_is_rejected(self):
        path = self.directory/f"samples_{archive.TAG}.csv.gz"
        path.write_bytes(b"tampered raw")
        with self.assertRaises(ValueError):
            self.run_archive()
        path.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.run_archive()
        self.assertFalse(self.options["archive_root"].exists())

    def test_changed_checkpoint_or_freeze_evidence_is_rejected(self):
        input_path = Path(next(iter(self.record["input_sha256_after"])))
        input_path.write_bytes(b"changed checkpoint")
        with self.assertRaises(ValueError):
            self.run_archive()
        self.record["input_sha256_before"][str(input_path)] = digest(input_path)
        self.record["input_sha256_after"][str(input_path)] = digest(input_path)
        self.record["model_buffers_after"]["joint"]["training"][""] = True
        self.update_manifest()
        with self.assertRaises(ValueError):
            self.run_archive()

    def test_protocol_missing_artifacts_and_preflight_are_rejected(self):
        self.record["protocol"]["preflight"] = True
        self.update_manifest()
        expected = copy.deepcopy(self.record["protocol"]); expected["preflight"] = False
        with self.assertRaises(ValueError):
            self.run_archive(expected_protocols={"20261030":expected})
        self.record["output_sha256"].pop(f"parameters_{archive.TAG}.csv")
        self.update_manifest()
        with self.assertRaises(ValueError):
            self.run_archive()

    def test_partial_or_tampered_existing_archive_is_never_overwritten(self):
        result = self.run_archive()
        manifest = Path(result["archive_directory"])/"archive_manifest.json"
        manifest.write_bytes(b"incomplete")
        with self.assertRaises(ValueError):
            self.run_archive()
        self.assertEqual(manifest.read_bytes(),b"incomplete")

    def test_existing_unknown_empty_directory_is_rejected(self):
        destination = self.options["archive_root"]/self.options["archive_id"]
        destination.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            self.run_archive()
        self.assertEqual(list(destination.iterdir()),[])

    def test_forbidden_extra_sources_and_path_escape(self):
        checkpoint = Path(next(iter(self.record["input_sha256_after"])))
        raw = self.directory/f"samples_{archive.TAG}.csv.gz"
        for mapping in ({"evil.pth":checkpoint},{"evil.csv":checkpoint},{"raw.json":raw},
                        {"../escape.csv":self.directory/f"summary_{archive.TAG}.csv"}):
            with self.subTest(mapping=mapping),self.assertRaises(ValueError):
                self.run_archive(evidence_files=mapping)
        with self.assertRaises(ValueError):
            self.run_archive(archive_id="../escape")
        with self.assertRaises(ValueError):
            self.run_archive(required_artifacts=tuple(archive.ARTIFACTS)+("best.pth",))

    def test_extra_csv_json_evidence_is_byte_preserving(self):
        source = write(self.root/"reports/custom.json",b'{"value": 0.123456789012345678}\n')
        result = self.run_archive(evidence_files={"reports/custom.json":source})
        self.assertEqual((Path(result["archive_directory"])/"reports/custom.json").read_bytes(),source.read_bytes())

    def test_no_large_file_truncation_or_duplicate_csv_acceptance(self):
        with mock.patch.object(archive,"MAX_SMALL_BYTES",1),self.assertRaises(ValueError):
            self.run_archive()
        with self.assertRaises(ValueError):
            self.run_archive(required_artifacts=tuple(archive.ARTIFACTS)+(archive.ARTIFACTS[0],))

    def test_symlink_guard_rejects_mocked_ancestor(self):
        original = Path.is_symlink
        target = self.directory.parent
        def check(path):
            return path == target or original(path)
        with mock.patch.object(Path,"is_symlink",check),self.assertRaises(ValueError):
            self.run_archive()


class CompletedSpatialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.batch,self.evaluations,self.summary = complete_batch(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def refresh_stage(self, key):
        stage = self.summary["stages"][key]
        stage["evidence_sha256"] = {name:digest(Path(name)) for name in stage["evidence_sha256"]}
        write(self.batch/"batch_summary.json",self.summary)

    def test_completed_batch_verification_and_archive(self):
        plan = archive.verify_completed_as_spatial(root=self.root)
        self.assertFalse((self.root/"paper").exists())
        self.assertEqual(set(plan["evaluations"]),set(map(str,archive.SEEDS)))
        result = archive.archive_completed_as_spatial(root=self.root)
        destination = Path(result["archive_directory"])
        self.assertEqual(destination,self.root/"paper/data/as_spatial_verified/batch_fixture")
        self.assertTrue((destination/"evidence/batch_summary.json").is_file())
        self.assertEqual(len(list((destination/"training").rglob("config.json"))),4)
        self.assertEqual(len(list((destination/"evaluations").rglob("*.csv"))),10)
        self.assertEqual(result["historical_source_provenance"]["20261030"]["source_sha256"],
                         {"models/historical.py":"a"*64})
        self.assertTrue(archive.archive_completed_as_spatial(root=self.root)["reused"])

    def test_incomplete_batch_or_stage_rejected_before_copy(self):
        self.summary["status"] = "failed"
        write(self.batch/"batch_summary.json",self.summary)
        with self.assertRaises(ValueError):
            archive.archive_completed_as_spatial(root=self.root)
        self.summary["status"] = "completed"
        self.summary["stages"]["train_joint_20261030"]["status"] = "running"
        write(self.batch/"batch_summary.json",self.summary)
        with self.assertRaises(ValueError):
            archive.verify_completed_as_spatial(root=self.root)
        self.assertFalse((self.root/"paper").exists())

    def test_training_budget_source_or_hash_tamper_rejected(self):
        stage = self.summary["stages"]["train_joint_20261030"]
        Path(stage["result"]["report"]["latest_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            archive.verify_completed_as_spatial(root=self.root)

    def test_parameter_count_must_match_actual_instantiation(self):
        directory = self.evaluations["20261030"]
        path = directory/f"spatial_parameters_{archive.TAG}.csv"
        path.write_bytes(csv_bytes([dict(method_key=arm,name="decoder.weight",numel=21,l2_norm=1.25)
                                   for arm in archive.ARMS]))
        manifest_path = directory/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["output_sha256"][path.name] = digest(path)
        write(manifest_path,manifest)
        self.refresh_stage("evaluate_20261030")
        with self.assertRaisesRegex(ValueError,"parameter count"):
            archive.verify_completed_as_spatial(root=self.root)

    def test_evaluation_training_identity_must_match(self):
        directory = self.evaluations["20261031"]
        path = directory/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest["methods"]["joint"]["selected_best_step"] = 19000
        write(path,manifest)
        self.refresh_stage("evaluate_20261031")
        with self.assertRaisesRegex(ValueError,"selected_best_step"):
            archive.verify_completed_as_spatial(root=self.root)

    def test_changed_config_with_stale_fingerprint_rejected(self):
        key = "train_frozen_ue_20261030"
        checkpoint = Path(self.summary["stages"][key]["result"]["report"]["best_path"])
        path = checkpoint.parent/"config.json"
        config = json.loads(path.read_text())
        config["source_identity"]["step"] += 1
        write(path,config)
        self.refresh_stage(key)
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            archive.verify_completed_as_spatial(root=self.root)


if __name__ == "__main__":
    unittest.main()
