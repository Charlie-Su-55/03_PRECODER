"""Metadata-only resume regression tests; fixtures are not server or weight evidence."""
from __future__ import annotations

import ast
from contextlib import contextmanager, ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, Path(__file__).resolve().parent):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from configs.grouped_codec_fdma_train_config import fdma_config
from configs.grouped_hybrid_anchor_train_config import (
    ARMS, SEEDS, SOURCE_BEST_SHA256, anchor_config,
)
from test_train_grouped_codec import resume_metadata
import train_grouped_hybrid_anchor as training
import grouped_hybrid_resume_compat as compatibility
from cli_contracts import assert_lightweight_cli


def actual_source_identity(seed=SEEDS[0]):
    """Run the real source-metadata verifier on explicitly synthetic schema fixtures."""
    source_cfg = fdma_config(seed)
    best = resume_metadata(source_cfg, 19500)
    latest = resume_metadata(source_cfg, 20000)
    sidecars = dict(
        config=dict(source_cfg.to_dict(), config_fingerprint=training.fingerprint(source_cfg)),
        status=dict(state="completed", step=20000, experiment=source_cfg.EXP_NAME,
                    best_metric_name="mean_all", best_metric=12.),
    )
    hashes = dict(best_sha256=SOURCE_BEST_SHA256[seed], latest_sha256="1" * 64,
                  config_sha256="2" * 64, status_sha256="3" * 64)
    return training.verify_source_payloads(seed, best, latest, sidecars, hashes)


def provenance_fixture(arm=ARMS[0], seed=SEEDS[0]):
    """No checkpoint is opened; only public source files supply actual source hashes."""
    cfg = anchor_config(arm, seed)
    record = dict(
        arm=arm, seed=seed, source=actual_source_identity(seed),
        config=cfg.to_dict(), config_fingerprint=training.fingerprint(cfg),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in training.SOURCE_FILES},
        git_commit="synthetic-provenance-test-not-server-evidence",
    )
    initial = json.loads(json.dumps(record, allow_nan=False))
    return cfg, initial, record


@contextmanager
def registered_fixture(prepared=True):
    """Temporary schema evidence, not CUDA/checkpoint proof; only Git proof is mocked."""
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory).resolve()
        output = root / 'runs_as_screen' / 'grouped_hybrid_anchor_v1'
        batch = output / 'batches' / compatibility.BATCH_NAME
        registration = batch / 'resume_compat_266d9e5'
        batch.mkdir(parents=True)
        if prepared:registration.mkdir()
        paths = dict(ROOT=root, OUTPUT_ROOT=output, BATCH=batch, REGISTRATION=registration,
            RECORD=registration/'record.json', SNAPSHOT=registration/'batch_summary.before.json',
            SEAL=registration/'record.sha256')
        for name, value in paths.items():
            stack.enter_context(mock.patch.object(compatibility, name, value))
        stack.enter_context(mock.patch.object(compatibility, 'anchor_config',
            side_effect=lambda arm,seed: anchor_config(arm,seed,output)))
        environment_check=stack.enter_context(mock.patch.object(compatibility, '_environment_matches'))

        def write_json(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

        repair = '4'*40
        train_name, controller_name = 'train_grouped_hybrid_anchor.py', 'run_grouped_hybrid_anchor_batch.py'
        old = {name: pair[0] for name, pair in compatibility.APPROVED_REPLACEMENTS.items()}
        new = {name: pair[1] for name, pair in compatibility.APPROVED_REPLACEMENTS.items()}
        old_trainer = {train_name: old[train_name], 'models/grouped_hybrid_anchor.py': 'a'*64}
        new_trainer = {train_name: new[train_name], 'models/grouped_hybrid_anchor.py': 'a'*64,
                       compatibility.TOOL: 'b'*64}
        old_controller = dict(old_trainer, **{controller_name: old[controller_name]})
        new_controller = dict(new_trainer, **{controller_name: new[controller_name]})
        release = dict(old_commit=compatibility.BASE_COMMIT, repair_commit=repair,
            reviewed_changes={name: dict(old=old[name], new=new[name]) for name in old},
            old_controller_sha256=old_controller, new_controller_sha256=new_controller,
            old_trainer_sha256=old_trainer, new_trainer_sha256=new_trainer,
            release_files_sha256=new_controller, compatibility_tool_sha256='b'*64)

        def checked_release(commit):
            compatibility.validate_release_description(commit, repair, [compatibility.BASE_COMMIT],
                compatibility.CHANGED_PATHS, old, new)
            return deepcopy(release)

        stack.enter_context(mock.patch.object(compatibility, 'release_identity', side_effect=checked_release))
        cfg = anchor_config(ARMS[0], SEEDS[0], output)
        _, _, current = provenance_fixture()
        current.update(config=cfg.to_dict(), config_fingerprint=training.fingerprint(cfg),
                       git_commit=repair, source_sha256=new_trainer)
        initial = deepcopy(current)
        initial.update(git_commit=compatibility.BASE_COMMIT, source_sha256=old_trainer)
        initial_path = Path(cfg.LOG_DIR)/'initialization.json'
        write_json(initial_path, initial)
        initial = json.loads(initial_path.read_text(encoding='utf-8'))
        immutable = {str(initial_path): compatibility.sha256(initial_path)}
        stages, reports = {}, {}
        for name in compatibility.COMPLETED:
            report = dict(passed=True, fixture='schema only; no numerical verification')
            if name == 'environment': report['source'] = dict(commit=compatibility.BASE_COMMIT)
            elif name == 'cuda_preflight': report['commit'] = compatibility.BASE_COMMIT
            path = batch/(name+'.json')
            write_json(path, report)
            digest = compatibility.sha256(path)
            immutable[str(path)] = digest
            reports[name] = path
            stages[name] = dict(status='completed', result=dict(report=report), evidence_sha256={str(path): digest})
        stages[compatibility.FAILED_TRAIN] = dict(status='failed', error='synthetic boundary failure')
        summary = dict(status='failed', source=dict(commit=compatibility.BASE_COMMIT, sha256=old_controller),
                       stages=stages, error='Resume code/source checkpoint provenance changed')
        original_package=batch/'original_package.schema_marker'
        original_package.write_text('Synthetic preserved package marker, not an actual server archive',encoding='utf-8')
        summary.update(result_package=str(original_package),result_package_sha256=compatibility.sha256(original_package))
        immutable[str(original_package)]=compatibility.sha256(original_package)
        summary_path = batch/'batch_summary.json'
        write_json(summary_path, summary)
        if prepared:paths['SNAPSHOT'].write_bytes(summary_path.read_bytes())
        write_json(output/'last_batch.json', dict(directory=str(batch)))
        active = {}
        for name in ('latest', 'best', 'status', 'metrics'):
            path = batch/('active_'+name+'.schema_marker')
            path.write_text('Synthetic step500 marker; not weights or server evidence', encoding='utf-8')
            active[str(path)] = compatibility.sha256(path)
        record = dict(schema='grouped-hybrid-266d9e5-resume-repair-v1', batch_name=compatibility.BATCH_NAME,
            release=release, snapshot_sha256=compatibility.sha256(summary_path),
            completed_stages={name: stages[name] for name in compatibility.COMPLETED}, immutable_files=immutable,
            verification=dict(arm='fdma_control',seed=20261030,best_step=500,latest_step=500,
                next_step=501,total_steps=20000,prior_count=500,strict_state=True,
                optimizer_scheduler_rng_verified=True,optimizer_constructed=False,weights_written=False,
                other_five_unstarted=True,active_files_before=active,source_record=deepcopy(current['source'])))
        if prepared:
            write_json(paths['RECORD'], record)
            paths['SEAL'].write_text(compatibility.sha256(paths['RECORD'])+'\n', encoding='ascii')
        yield SimpleNamespace(cfg=cfg, initial=initial, current=current, record=record, release=release,
            batch=batch, summary=summary, summary_path=summary_path, paths=paths, reports=reports,
            active=active, write_json=write_json, environment_check=environment_check,
            controller_current=dict(commit=repair, sha256=new_controller))


class InitializationProvenanceTests(unittest.TestCase):
    def check_provenance(self, cfg, initial, record, compat_path=None):
        return training.validate_initialization_provenance(
            cfg, initial, record, compat_path=compat_path)

    def test_real_source_recipe_json_roundtrip_reproduces_old_failure(self):
        cfg, initial, record = provenance_fixture()
        source = record["source"]
        self.assertIsInstance(source["config"]["recipe"]["stage1_snr"], tuple)
        self.assertIsInstance(initial["source"]["config"]["recipe"]["stage1_snr"], list)
        self.assertNotEqual(initial["source"], source)
        self.assertEqual(json.loads(json.dumps(source)), initial["source"])
        before_initial, before_record = deepcopy(initial), deepcopy(record)
        self.check_provenance(cfg, initial, record)
        self.assertEqual(initial, before_initial)
        self.assertEqual(record, before_record)

    def test_same_semantics_accepts_each_arm_seed_without_mutating_recipe(self):
        for seed in SEEDS:
            for arm in ARMS:
                with self.subTest(seed=seed, arm=arm):
                    cfg, initial, record = provenance_fixture(arm, seed)
                    self.check_provenance(cfg, initial, record)
                    self.check_provenance(cfg, initial, json.loads(json.dumps(record)))
                    self.assertIsInstance(cfg.to_dict()["recipe"]["val_snr_list"], tuple)

    def test_changed_source_snr_values_or_order_rejected_with_field_location(self):
        cfg, initial, record = provenance_fixture()
        for field in ("stage1_snr", "stage2_snr", "stage3_snr", "val_snr_list"):
            with self.subTest(field=field):
                changed = deepcopy(record)
                values = list(changed["source"]["config"]["recipe"][field])
                values[0] = 1. if field != "val_snr_list" else 5
                changed["source"]["config"]["recipe"][field] = values
                with self.assertRaisesRegex(ValueError, field):
                    self.check_provenance(cfg, initial, changed)
        changed = deepcopy(record)
        changed["source"]["config"]["recipe"]["val_snr_list"] = [25,20,15,10,5,0]
        with self.assertRaisesRegex(ValueError, "val_snr_list"):
            self.check_provenance(cfg, initial, changed)

    def test_changed_source_seed_paths_hashes_steps_and_configuration_rejected(self):
        cfg, initial, record = provenance_fixture()
        mutations = {
            "training_seed": SEEDS[1], "best_path": "different/best.pth",
            "latest_path": "different/latest.pth", "config_path": "different/config.json",
            "status_path": "different/status.json", "best_sha256": "0" * 64,
            "latest_sha256": "0" * 64, "config_sha256": "0" * 64,
            "status_sha256": "0" * 64, "best_step": 19000, "latest_step": 19500,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = deepcopy(record)
                changed["source"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self.check_provenance(cfg, initial, changed)
        changed = deepcopy(record)
        changed["source"]["complete_source_configuration"]["recipe"]["lr"] = 2e-5
        with self.assertRaisesRegex(ValueError, "lr"):
            self.check_provenance(cfg, initial, changed)

    def test_missing_extra_and_scalar_type_changes_are_not_json_compatibility(self):
        cfg, initial, record = provenance_fixture()
        for issue in ("missing", "extra", "string"):
            with self.subTest(issue=issue):
                changed = deepcopy(record)
                if issue == "missing":
                    changed["source"].pop("best_sha256")
                elif issue == "extra":
                    changed["source"]["unapproved_source_override"] = True
                else:
                    changed["source"]["training_seed"] = str(SEEDS[0])
                with self.assertRaises(ValueError):
                    self.check_provenance(cfg, initial, changed)

    def test_initialization_full_config_or_fingerprint_changes_rejected(self):
        cfg, initial, record = provenance_fixture()
        changed = deepcopy(initial)
        changed["config"]["recipe"]["lr"] = 2e-5
        with self.assertRaisesRegex(ValueError, "config"):
            self.check_provenance(cfg, changed, record)
        changed = deepcopy(initial)
        changed["config_fingerprint"] = "0" * 16
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.check_provenance(cfg, changed, record)

    def test_source_code_hash_change_still_fails_without_explicit_compatibility(self):
        cfg, initial, record = provenance_fixture()
        for name in ("train_grouped_hybrid_anchor.py", "models/grouped_hybrid_anchor.py"):
            with self.subTest(name=name):
                changed = deepcopy(record)
                changed["source_sha256"][name] = "0" * 64
                with self.assertRaises(ValueError):
                    self.check_provenance(cfg, initial, changed)

    def test_missing_compatibility_record_cannot_authorize_source_change(self):
        cfg, initial, record = provenance_fixture()
        record["source_sha256"]["train_grouped_hybrid_anchor.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-created-compatibility.json"
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.check_provenance(cfg, initial, record, compat_path=missing)
            self.assertFalse(missing.exists())

    def test_real_training_resume_calls_provenance_validation_before_checkpoint_load(self):
        tree = ast.parse((ROOT / "train_grouped_hybrid_anchor.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_training")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        validation = [node for node in calls if isinstance(node.func, ast.Name)
                      and node.func.id == "validate_initialization_provenance"]
        self.assertEqual(len(validation), 1)
        resume_loads = [node for node in calls if isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                        and node.func.attr == "load"]
        self.assertTrue(resume_loads)
        self.assertLess(validation[0].lineno, min(node.lineno for node in resume_loads))


class ExactReleasePolicyTests(unittest.TestCase):
    def test_direct_child_exact_changed_files_and_executable_bytes_required(self):
        repair = '4'*40
        old = {name: pair[0] for name, pair in compatibility.APPROVED_REPLACEMENTS.items()}
        new = {name: pair[1] for name, pair in compatibility.APPROVED_REPLACEMENTS.items()}
        kwargs = dict(repair_commit=repair, head=repair, parents=[compatibility.BASE_COMMIT],
                      changed=set(compatibility.CHANGED_PATHS), old_digests=old, new_digests=new)
        compatibility.validate_release_description(**kwargs)
        mutations = {
            'not exact sha': dict(repair_commit='HEAD'), 'wrong head': dict(head='5'*40),
            'wrong parent': dict(parents=['6'*40]),
            'merge parent': dict(parents=[compatibility.BASE_COMMIT, '6'*40]),
            'unrelated model edit': dict(changed=set(compatibility.CHANGED_PATHS)|{'models/grouped_hybrid_anchor.py'}),
            'missing reviewed file': dict(changed=set(compatibility.CHANGED_PATHS)-{compatibility.TOOL}),
            'unreviewed executable': dict(new_digests=dict(new, **{'train_grouped_hybrid_anchor.py': '0'*64})),
            'wrong baseline bytes': dict(old_digests=dict(old, **{'train_grouped_hybrid_anchor.py': '0'*64})),
        }
        for issue, replacement in mutations.items():
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                compatibility.validate_release_description(**dict(kwargs, **replacement))

    def test_help_and_dry_run_remain_stdlib_and_side_effect_free(self):
        assert_lightweight_cli(self, 'grouped_hybrid_resume_compat', [['--dry-run']],
            "assert plans[0]['checkpoint_reads'] is False\nassert plans[0]['writes'] is False")


class RegisteredTransitionTests(unittest.TestCase):
    def test_actual_prepare_preserves_all_original_bytes_is_idempotent_and_refuses_partial_registration(self):
        # The real exclusive file writes/load/seal/snapshot run. Only this
        # project's server-specific verification/locking boundaries are mocked.
        with registered_fixture(prepared=False) as f:
            originals={path:path.read_bytes() for path in f.paths['ROOT'].rglob('*') if path.is_file()}
            with mock.patch.object(compatibility,'os',SimpleNamespace(name='posix')), \
                 mock.patch.object(compatibility,'verify_paused_checkpoint',return_value=deepcopy(f.record['verification'])) as verified, \
                 mock.patch('train_proposal_as_receiver.run_lock',side_effect=lambda *a,**kw:nullcontext()) as locks, \
                 redirect_stdout(io.StringIO()):
                first=compatibility.prepare(f.release['repair_commit'])
                self.assertTrue(f.paths['RECORD'].is_file())
                self.assertEqual(f.paths['SNAPSHOT'].read_bytes(),originals[f.summary_path])
                self.assertEqual(f.paths['SEAL'].read_text(encoding='ascii').strip(),compatibility.sha256(f.paths['RECORD']))
                saved={name:f.paths[name].read_bytes() for name in ('RECORD','SNAPSHOT','SEAL')}
                second=compatibility.prepare(f.release['repair_commit'])
                self.assertEqual(first,second)
                self.assertEqual(verified.call_count,1)
                self.assertEqual(locks.call_count,14)
                for name,contents in saved.items():self.assertEqual(f.paths[name].read_bytes(),contents)
            for path,contents in originals.items():self.assertEqual(path.read_bytes(),contents)
        with registered_fixture(prepared=False) as f:
            f.paths['REGISTRATION'].mkdir()
            with mock.patch.object(compatibility,'os',SimpleNamespace(name='posix')), \
                 mock.patch.object(compatibility,'verify_paused_checkpoint') as verified, \
                 mock.patch('train_proposal_as_receiver.run_lock',side_effect=lambda *a,**kw:nullcontext()):
                with self.assertRaisesRegex(FileExistsError,'Incomplete registration'):
                    compatibility.prepare(f.release['repair_commit'])
                verified.assert_not_called()
                self.assertFalse(f.paths['RECORD'].exists())
                self.assertFalse(f.paths['SNAPSHOT'].exists())

    def test_controller_and_trainer_share_record_without_rewriting_old_provenance(self):
        with registered_fixture() as f:
            frozen = {name: f.paths[name].read_bytes() for name in ('RECORD','SNAPSHOT','SEAL')}
            initial_path = Path(f.cfg.LOG_DIR)/'initialization.json'
            original_initial, original_source = initial_path.read_bytes(), deepcopy(f.summary['source'])
            accepted = compatibility.validate_controller_transition(f.batch,f.summary,f.controller_current,f.paths['RECORD'])
            f.summary['resume_compatibility'] = accepted
            f.write_json(f.summary_path,f.summary)
            actual = training.validate_initialization_provenance(f.cfg,f.initial,f.current,f.paths['RECORD'])
            self.assertEqual(actual,accepted)
            f.environment_check.assert_called_once_with(f.summary)
            self.assertEqual(actual['version_chain'],[compatibility.BASE_COMMIT,f.release['repair_commit']])
            self.assertEqual(f.summary['source'],original_source)
            self.assertEqual(initial_path.read_bytes(),original_initial)
            for name,contents in frozen.items():self.assertEqual(f.paths[name].read_bytes(),contents)

    def test_first_resume_rejects_active_change_then_later_allows_progress_not_frozen_drift(self):
        with registered_fixture() as f:
            active = Path(next(iter(f.active)))
            original = active.read_bytes()
            active.write_text('Synthetic later update',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'before first compatible resume'):
                compatibility.validate_controller_transition(f.batch,f.summary,f.controller_current,f.paths['RECORD'])
            active.write_bytes(original)
            accepted = compatibility.validate_controller_transition(f.batch,f.summary,f.controller_current,f.paths['RECORD'])
            f.summary.update(resume_compatibility=accepted,status='running')
            f.write_json(f.summary_path,f.summary)
            active.write_text('Synthetic later update; strict checkpoint validation remains separate',encoding='utf-8')
            self.assertEqual(compatibility.validate_controller_transition(
                f.batch,f.summary,f.controller_current,f.paths['RECORD']),accepted)
            self.assertEqual(training.validate_initialization_provenance(
                f.cfg,f.initial,f.current,f.paths['RECORD']),accepted)
            f.reports['cuda_preflight'].write_text('Changed frozen acceptance',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Immutable|Completed evidence'):
                compatibility.validate_controller_transition(f.batch,f.summary,f.controller_current,f.paths['RECORD'])

    def test_record_seal_snapshot_and_fixed_path_are_enforced(self):
        for issue in ('record','seal','snapshot','initialization','alternate_path'):
            with self.subTest(issue=issue), registered_fixture() as f:
                path = f.paths['RECORD']
                if issue=='record':path.write_bytes(path.read_bytes()+b' ')
                elif issue=='seal':f.paths['SEAL'].write_text('0'*64,encoding='ascii')
                elif issue=='snapshot':f.paths['SNAPSHOT'].write_bytes(f.paths['SNAPSHOT'].read_bytes()+b' ')
                elif issue=='initialization':
                    initial_path=Path(f.cfg.LOG_DIR)/'initialization.json'
                    initial_path.write_bytes(initial_path.read_bytes()+b' ')
                else:
                    path=f.batch/'different_record.json';path.write_bytes(f.paths['RECORD'].read_bytes())
                with self.assertRaises(ValueError):compatibility.load_registration(path)

    def test_wrong_batch_commit_pointer_completed_stage_and_protected_source_rejected(self):
        for issue in ('batch','commit','pointer','completed_stage','protected_source'):
            with self.subTest(issue=issue), registered_fixture() as f:
                folder,current=f.batch,deepcopy(f.controller_current)
                if issue=='batch':folder=f.batch.parent/'some_other_batch'
                elif issue=='commit':current['commit']='6'*40
                elif issue=='pointer':f.write_json(f.paths['OUTPUT_ROOT']/'last_batch.json',dict(directory=str(f.batch.parent/'other')))
                elif issue=='completed_stage':f.summary['stages']['cuda_preflight']['result']['report']['commit']='6'*40
                else:current['sha256']['models/grouped_hybrid_anchor.py']='0'*64
                with self.assertRaises(ValueError):
                    compatibility.validate_controller_transition(folder,f.summary,current,f.paths['RECORD'])

    def test_registered_code_transition_never_authorizes_real_source_or_recipe_changes(self):
        with registered_fixture() as f:
            accepted=compatibility.validate_controller_transition(f.batch,f.summary,f.controller_current,f.paths['RECORD'])
            f.summary['resume_compatibility']=accepted;f.write_json(f.summary_path,f.summary)
            changed=deepcopy(f.current);changed['source']['best_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'best_sha256'):
                training.validate_initialization_provenance(f.cfg,f.initial,changed,f.paths['RECORD'])
            changed=deepcopy(f.initial);changed['config']['recipe']['lr']=2e-5
            with self.assertRaisesRegex(ValueError,'config'):
                training.validate_initialization_provenance(f.cfg,changed,f.current,f.paths['RECORD'])

    def test_only_original_arm_can_carry_old_initialization_source_map(self):
        with registered_fixture() as f:
            accepted=compatibility.validate_controller_transition(f.batch,f.summary,f.controller_current,f.paths['RECORD'])
            f.summary['resume_compatibility']=accepted;f.write_json(f.summary_path,f.summary)
            other=anchor_config(ARMS[1],SEEDS[0],f.paths['OUTPUT_ROOT'])
            with self.assertRaisesRegex(ValueError,'Only the existing500-step arm'):
                compatibility.validate_trainer_transition(other,f.initial,f.current,f.paths['RECORD'])
            self.assertEqual(compatibility.validate_trainer_transition(
                other,deepcopy(f.current),f.current,f.paths['RECORD']),accepted)

    def test_completed_evidence_report_must_match_not_only_its_digest(self):
        with registered_fixture() as f:
            self.assertEqual(len(compatibility.inspect_completed_stages(f.summary)),3)
            changed=deepcopy(f.summary)
            changed['stages']['cuda_preflight']['result']['report']['extra_unrecorded_claim']=True
            with self.assertRaisesRegex(ValueError,'No intact report file'):
                compatibility.inspect_completed_stages(changed)

    def test_resealed_records_still_require_all_frozen_evidence_and_checkpoint_claim_fields(self):
        for issue in ('missing_evidence','wrong_prior_count','optimizer_created','wrong_batch'):
            with self.subTest(issue=issue), registered_fixture() as f:
                changed=deepcopy(f.record)
                if issue=='missing_evidence':changed['immutable_files'].pop(str(f.reports['cuda_preflight']))
                elif issue=='wrong_prior_count':changed['verification']['prior_count']=499
                elif issue=='optimizer_created':changed['verification']['optimizer_constructed']=True
                else:changed['batch_name']='other_batch'
                f.write_json(f.paths['RECORD'],changed)
                f.paths['SEAL'].write_text(compatibility.sha256(f.paths['RECORD'])+'\n',encoding='ascii')
                with self.assertRaises(ValueError):compatibility.load_registration(f.paths['RECORD'])


class ControllerContinuationTests(unittest.TestCase):
    def test_actual_controller_resume_boundary_keeps_normal_strict_checks(self):
        import run_grouped_hybrid_anchor_batch as controller
        with registered_fixture() as f:
            self.assertIsNone(controller.validate_batch_resume(f.batch,f.summary,deepcopy(f.summary['source'])))
            with self.assertRaisesRegex(ValueError,'source-code SHA256 changed'):
                controller.validate_batch_resume(f.batch,f.summary,f.controller_current)
            accepted=controller.validate_batch_resume(f.batch,f.summary,f.controller_current,f.paths['RECORD'])
            f.summary['resume_compatibility']=accepted
            with self.assertRaisesRegex(ValueError,'every resume'):
                controller.validate_batch_resume(f.batch,f.summary,deepcopy(f.summary['source']))

    def test_real_train_one_does_not_repeat_existing500_and_forwards_same_compatibility_record(self):
        import run_grouped_hybrid_anchor_batch as controller
        for already_paused in (True,False):
            with self.subTest(already_paused=already_paused), tempfile.TemporaryDirectory() as directory:
                folder=Path(directory)
                cfg=SimpleNamespace(SAVE_DIR=str(folder/'weights'),LOG_DIR=str(folder/'logs'),
                    SAVE_PATH=str(folder/'weights/latest.schema_marker'),BEST_PATH=str(folder/'weights/best.schema_marker'),
                    STATUS_PATH=str(folder/'logs/status.json'),CONFIG_PATH=str(folder/'logs/config.json'),
                    METRICS_PATH=str(folder/'logs/metrics.jsonl'))
                def progress(state):
                    Path(cfg.SAVE_DIR).mkdir(exist_ok=True);Path(cfg.LOG_DIR).mkdir(exist_ok=True)
                    Path(cfg.SAVE_PATH).write_text('Synthetic marker, no weights',encoding='utf-8')
                    Path(cfg.STATUS_PATH).write_text(json.dumps(dict(state=state,step=500 if state=='paused' else 20000)),encoding='utf-8')
                if already_paused:progress('paused')
                calls=[];registered=folder/'registered_record.json'
                def child(_folder,_label,command):
                    calls.append(command)
                    progress('paused' if '--stop-after' in command else 'completed')
                    return str(folder/'synthetic_child.log')
                with mock.patch.object(controller,'anchor_config',return_value=cfg), \
                     mock.patch.object(controller,'_child',side_effect=child), \
                     mock.patch.object(controller,'stage_child',side_effect=lambda *a: dict(report={'completed_updates':20000},_evidence=[])):
                    controller.train_one(folder,ARMS[0],SEEDS[0],registered)
                    self.assertEqual(len(calls),1 if already_paused else 2)
                    self.assertEqual(calls[-1][-4:],['--resume','--train','--resume-compat',str(registered)])
                    if already_paused:self.assertTrue(all('--stop-after' not in command for command in calls))
                    else:self.assertEqual(calls[0][-2:],['--stop-after','500'])
                    controller.train_one(folder,ARMS[0],SEEDS[0],registered)
                    self.assertEqual(len(calls),1 if already_paused else 2)

    def test_actual_stage_runner_reuses_all_three_intact_completed_stages_without_actions(self):
        import run_grouped_hybrid_anchor_batch as controller
        with registered_fixture() as f:
            action=mock.Mock(side_effect=AssertionError('Completed stages must not rerun'))
            for name in compatibility.COMPLETED:
                result=controller._run_stage(f.summary,f.batch,name,action)
                self.assertTrue(result['report']['passed'])
            action.assert_not_called()
            f.reports['cuda_preflight'].write_text('Different preserved report',encoding='utf-8')
            with self.assertRaises(RuntimeError):controller._run_stage(f.summary,f.batch,'cuda_preflight',action)
            action.assert_not_called()


if __name__ == '__main__':
    unittest.main()
