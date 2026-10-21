"""Lightweight regression checks; no checkpoints, CUDA or Uni-Core required.

Task methods are loaded from their AST to exercise real dispatch/scoring/I/O
without importing the training stack. Neural-network outputs are test doubles;
these tests do not establish checkpoint-level or GPU numerical equivalence.
"""
import argparse
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_file('inference_config', 'unimol/inference_config.py')
setup = load_file('inference_utils', 'unimol/inference_utils.py')


def task_method(name, **globals_extra):
    tree = ast.parse((ROOT / 'unimol/tasks/drugclip.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DrugCLIP')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(np=np, os=os, pickle=pickle, fold_paths=config.fold_paths,
                     write_ranked_results=config.write_ranked_results, tqdm=lambda x: x)
    namespace.update(globals_extra)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'unimol/tasks/drugclip.py'), 'exec'), namespace)
    return namespace[name]


@contextlib.contextmanager
def temp_cwd():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(previous)


class ConfigurationTests(unittest.TestCase):
    def test_explicit_booleans(self):
        for value in (True, 'True', 'YES', '1', 't'):
            self.assertIs(config.str2bool(value), True)
        for value in (False, 'False', 'NO', '0', 'f'):
            self.assertIs(config.str2bool(value), False)
        with self.assertRaises(argparse.ArgumentTypeError):
            config.str2bool('maybe')

    def test_fold_paths_match_original_layout(self):
        for version, count, weights in [('6_folds', 6, '6_folds'), ('8_folds', 8, '8_folds'), ('6_folds_filtered', 6, '6_folds')]:
            checkpoints, caches = config.fold_paths(version)
            self.assertEqual(checkpoints, [f'./data/model_weights/{weights}/fold_{i}.pt' for i in range(count)])
            self.assertEqual(caches, [f'./data/encoded_mol_embs/{version}/fold{i}.pkl' for i in range(count)])
        with self.assertRaises(ValueError):
            config.fold_paths('typo')

    def test_preflight_distinguishes_cache_from_custom_library(self):
        with temp_cwd():
            checkpoints, caches = config.fold_paths('6_folds')
            for name in checkpoints + ['pocket.lmdb', 'mols.lmdb']:
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            config.validate_retrieval_inputs('pocket.lmdb', 'mols.lmdb', '6_folds', False, 'out/result.csv')
            self.assertTrue(Path('out').is_dir())
            with self.assertRaisesRegex(FileNotFoundError, 'molecule cache'):
                config.validate_retrieval_inputs('pocket.lmdb', '', '6_folds', True, 'out/result.csv')
            for name in caches:
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            config.validate_retrieval_inputs('pocket.lmdb', '', '6_folds', True, 'out/result.csv')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                config.validate_retrieval_inputs('pocket.lmdb', 'mols.lmdb', '6_folds', False, 'mols.lmdb')

    def test_ranked_results_preserve_normal_library_behavior(self):
        rng = np.random.default_rng(42)
        with temp_cwd():
            for size in (50, 100, 999):
                scores = rng.normal(size=size)
                names = [f'mol_{i}' for i in range(size)]
                old = sorted(zip(scores, names), key=lambda x: x[0], reverse=True)[:int(size * .02)]
                expected = ''.join(f'{name},{score}\n' for score, name in old)
                config.write_ranked_results(scores, names, 'out.csv')
                self.assertEqual(Path('out.csv').read_text(), expected)

    def test_small_empty_and_tied_libraries(self):
        with temp_cwd():
            config.write_ranked_results([2., 2., 1.], ['first', 'second', 'last'], 'out.csv')
            self.assertEqual(Path('out.csv').read_text(), 'first,2.0\n')
            with self.assertRaisesRegex(ValueError, 'empty'):
                config.write_ranked_results([], [], 'out.csv')
            with self.assertRaisesRegex(ValueError, 'same length'):
                config.write_ranked_results([1], [], 'out.csv')


class ModelSetupTests(unittest.TestCase):
    def test_device_precision_and_checkpoint_policy(self):
        for available, cpu, fp16, half_on_cpu, checkpoint in [
            (True, False, True, False, True), (True, True, True, False, False),
            (False, False, True, True, False), (True, False, False, True, False),
        ]:
            with self.subTest(available=available, cpu=cpu, fp16=fp16):
                torch, unicore = MagicMock(), MagicMock()
                torch.cuda.is_available.return_value = available
                model = unicore.tasks.setup_task.return_value.build_model.return_value
                args = SimpleNamespace(cpu=cpu, fp16=fp16, device_id=0, path='weights.pt')
                with patch.dict(sys.modules, {'torch': torch, 'unicore': unicore}):
                    _, returned, use_cuda = setup.setup_inference(args, checkpoint, half_on_cpu)
                self.assertIs(returned, model)
                self.assertEqual(use_cuda, available and not cpu)
                self.assertEqual(model.half.called, fp16 and (use_cuda or half_on_cpu))
                self.assertEqual(model.cuda.called, use_cuda)
                self.assertEqual(model.load_state_dict.called, checkpoint)
                model.eval.assert_called_once()


class TaskRegressionTests(unittest.TestCase):
    def test_both_benchmarks_honor_fold_selection(self):
        metric = {key: 1. for key in ('0.005', '0.01', '0.02', '0.05')}
        for benchmark in ('dude', 'pcba'):
            for folds in (False, True):
                with self.subTest(benchmark=benchmark, folds=folds), temp_cwd():
                    data = Path('data/DUD-E' if benchmark == 'dude' else 'data/lit_pcba')
                    (data / 'target').mkdir(parents=True)
                    fake = MagicMock()
                    result = (0.8, 0.6, metric, metric)
                    if benchmark == 'dude':
                        result += (np.array([1., 0.]), np.array([1, 0]))
                    single = getattr(fake, f'test_{benchmark}_target')
                    ensemble = getattr(fake, f'test_{benchmark}_target_ensemble')
                    single.return_value = ensemble.return_value = result
                    with patch.dict(sys.modules, {'pandas': MagicMock()}), contextlib.redirect_stdout(io.StringIO()):
                        task_method(f'test_{benchmark}')(fake, object(), use_folds=folds)
                    self.assertEqual(single.call_count, int(not folds))
                    self.assertEqual(ensemble.call_count, int(folds))

    def test_dude_report_skips_missing_targets_and_rejects_empty_run(self):
        metric = {key: 1. for key in ('0.005', '0.01', '0.02', '0.05')}
        with temp_cwd():
            Path('data/DUD-E/valid').mkdir(parents=True)
            Path('data/DUD-E/missing').mkdir()
            fake, pandas = MagicMock(), MagicMock()
            fake.test_dude_target.side_effect = lambda name, model: (
                (.8, .6, metric, metric, np.array([1., 0.]), np.array([1, 0]))
                if name == 'valid' else None)
            with patch.dict(sys.modules, {'pandas': pandas}), contextlib.redirect_stdout(io.StringIO()):
                task_method('test_dude')(fake, None, use_folds=False)
            self.assertEqual(pandas.DataFrame.call_args.args[0]['target'], ['valid'])
            fake.test_dude_target.side_effect = lambda *args: None
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'No DUD-E targets'):
                task_method('test_dude')(fake, None, use_folds=False)

    def test_custom_retrieval_preserves_cache_and_matches_cached_scoring(self):
        # Execute the actual retrieval method with deterministic encoder outputs.
        rng = np.random.default_rng(7)
        mols, pockets = rng.normal(size=(100, 4)), rng.normal(size=(2, 4))
        names = [f'mol_{i}' for i in range(100)]
        model, torch, checkpoint_utils = MagicMock(), MagicMock(), MagicMock()
        for projector, values in [(model.mol_project, mols), (model.pocket_project, pockets)]:
            projector.return_value.__truediv__.return_value.detach.return_value.cpu.return_value.numpy.return_value = values
        sample = {'net_input': MagicMock(), 'smi_name': names}
        torch.utils.data.DataLoader.return_value = [sample]
        fake = MagicMock()
        method = task_method('retrieval_multi_folds', torch=torch, checkpoint_utils=checkpoint_utils)
        with temp_cwd():
            _, caches = config.fold_paths('6_folds')
            before = {}
            for name in caches:
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps([mols, names]))
                before[name] = path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                method(fake, model, 'pocket.lmdb', 'custom.csv', 'custom.lmdb', '6_folds', use_cache=False, use_cuda=False)
                method(fake, model, 'pocket.lmdb', 'cached.csv', '', '6_folds', use_cache=True, use_cuda=False)
            self.assertEqual(Path('custom.csv').read_bytes(), Path('cached.csv').read_bytes())
            for name in caches:
                self.assertEqual(Path(name).read_bytes(), before[name])
            scores = pockets @ mols.T
            median = np.median(scores, axis=1, keepdims=True)
            mad = np.median(np.abs(scores - median), axis=1, keepdims=True)
            expected = (0.6745 * (scores - median) / (mad + 1e-6)).max(axis=0)
            rows = Path('custom.csv').read_text().splitlines()
            self.assertEqual([row.split(',')[0] for row in rows], [names[i] for i in np.argsort(expected)[::-1][:2]])


if __name__ == '__main__':
    unittest.main()
