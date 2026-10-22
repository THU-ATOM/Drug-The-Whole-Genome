"""Checks using real arrays, HDF5 files and report writers."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import h5py
import numpy as np

from test_inference import checks, config, setup, benchmark_utils, temp_cwd, task_method


class ArrayChecks(unittest.TestCase):
    def test_bad_cache_arrays(self):
        good = np.ones((2, 4), dtype=np.float32)
        for reps, names, width in [
            (good, ['a'], 4), (good, ['a', 'b'], 3),
            (np.array([[np.nan] * 4] * 2), ['a', 'b'], 4),
            (np.ones((2, 4), dtype=int), ['a', 'b'], 4),
            (good, ['a', ''], 4), (good[0], ['a', 'b'], 4),
        ]:
            with self.subTest(names=names, width=width), self.assertRaisesRegex(ValueError, 'cache.pkl'):
                checks.validate_embeddings(reps, names, 'cache.pkl', dimension=width)

    def test_invalid_scores_do_not_overwrite_output(self):
        with temp_cwd():
            Path('scores.csv').write_text('existing')
            for score in [np.nan, np.inf, -np.inf]:
                with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
                    config.write_ranked_results([score], ['mol'], 'scores.csv')
                self.assertEqual(Path('scores.csv').read_text(), 'existing')

    def test_cuda_requirement_precedes_model_setup(self):
        import sys
        from unittest.mock import patch
        torch, unicore = MagicMock(), MagicMock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {'torch': torch, 'unicore': unicore}):
            with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
                setup.setup_inference(SimpleNamespace(cpu=False), require_cuda=True)
        unicore.tasks.setup_task.assert_not_called()


class ResumeChecks(unittest.TestCase):
    def test_prefix_masks(self):
        for mask, count in [([False, False], 0), ([True, False], 1), ([True, True], 2)]:
            self.assertEqual(checks.completed_prefix(np.array(mask), 'fold0'), count)
        with self.assertRaisesRegex(ValueError, 'gaps'):
            checks.completed_prefix(np.array([True, False, True]), 'fold0')

    def test_manifest_changes_with_input_order_weights_and_settings(self):
        with temp_cwd():
            Path('mols').write_bytes(b'first,second')
            Path('weight').write_bytes(b'weight1')
            Path('dict').write_bytes(b'atom dictionary')
            def manifest(seed=1):
                return checks.encoding_manifest('mols', ['weight'], {'seed': seed}, ['dict'])
            baseline = manifest()
            Path('mols').write_bytes(b'second,first')
            self.assertNotEqual(baseline, manifest())
            Path('mols').write_bytes(b'first,second')
            self.assertEqual(baseline, manifest())
            Path('weight').write_bytes(b'weight2')
            self.assertNotEqual(baseline, manifest())
            Path('weight').write_bytes(b'weight1')
            self.assertNotEqual(baseline, manifest(2))

    def test_hdf5_rejects_old_changed_and_inconsistent_resume(self):
        with temp_cwd():
            with h5py.File('resume.h5', 'w') as handle:
                checks.check_resume_file(handle, {'input': 'A'}, 3)
                handle.create_dataset('mol_reps', data=np.ones((3, 768), dtype=np.float32))
                handle.create_dataset('fold0', data=np.array([True, False, False]))
                checks.check_resume_file(handle, {'input': 'A'}, 3)
                before = handle['mol_reps'][:]
                with self.assertRaisesRegex(ValueError, 'changed'):
                    checks.check_resume_file(handle, {'input': 'B'}, 3)
                np.testing.assert_array_equal(before, handle['mol_reps'][:])
                handle['fold0'][:] = [True, False, True]
                with self.assertRaisesRegex(ValueError, 'gaps'):
                    checks.check_resume_file(handle, {'input': 'A'}, 3)
            with h5py.File('old.h5', 'w') as handle:
                handle.create_dataset('mol_reps', data=np.zeros((3, 768)))
                with self.assertRaisesRegex(ValueError, 'no encoding manifest'):
                    checks.check_resume_file(handle, {'input': 'A'}, 3)
            with h5py.File('missing.h5', 'w') as handle:
                checks.check_resume_file(handle, {'input': 'A'}, 3)
                handle.create_dataset('fold2', data=np.array([True, False, False]))
                with self.assertRaisesRegex(ValueError, 'without embeddings'):
                    checks.check_resume_file(handle, {'input': 'A'}, 3)

    def test_actual_encoder_resumes_interruption(self):
        class Dataset:
            def __init__(self, rows):
                self.rows = list(rows)
                self.collater = lambda samples: samples
            def __len__(self):
                return len(self.rows)

        with temp_cwd():
            Path('mols.lmdb').write_bytes(b'fixed ordered input')
            Path('dict').mkdir()
            for name in ('dict_mol.txt', 'dict_pkt.txt'):
                Path('dict', name).write_text('dictionary')
            for weight in config.fold_paths('6_folds')[0]:
                Path(weight).parent.mkdir(parents=True, exist_ok=True)
                Path(weight).write_bytes(b'fixed weight')
            Path('out').mkdir()
            fake, torch, model, checkpoint = MagicMock(), MagicMock(), MagicMock(), MagicMock()
            fake.args = SimpleNamespace(data='dict', num_workers=0, seed=1, fp16=False)
            fake.load_mols_dataset_dtwg.side_effect = lambda *a, **k: Dataset(range(3))
            torch.__version__ = 'test'
            torch.utils.data.Subset.side_effect = lambda data, indices: Dataset(data.rows[i] for i in indices)
            state = {'interrupt': True, 'fold': 0, 'rows': []}
            def load_weight(path):
                state['fold'] = int(Path(path).stem.split('_')[-1])
                return {'model': {}}
            checkpoint.load_checkpoint_to_cpu.side_effect = load_weight
            def batches(data, batch_size, **kwargs):
                for start in range(0, len(data), batch_size):
                    state['rows'] = data.rows[start:start + batch_size]
                    yield {'net_input': MagicMock()}
                    if state['interrupt']:
                        state['interrupt'] = False
                        raise RuntimeError('simulated interruption')
            torch.utils.data.DataLoader.side_effect = batches
            def project(*args):
                output = MagicMock()
                values = np.array([[row + 10 * state['fold']] * 128 for row in state['rows']], dtype=np.float32)
                output.__truediv__.return_value.detach.return_value.cpu.return_value.numpy.return_value = values
                return output
            model.mol_project.side_effect = project
            method = task_method('encode_mols_multi_folds', torch=torch, checkpoint_utils=checkpoint, h5py=h5py)
            def run():
                with contextlib.redirect_stdout(io.StringIO()):
                    method(fake, model, 2, 'mols.lmdb', 'out', False, write_npy=True, write_h5=True, start=0, end=None)
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                run()
            with h5py.File('out/mol_reps0None.h5', 'r') as handle:
                np.testing.assert_array_equal(handle['fold0'][:], [True, True, False])
            run()
            expected = np.array([[[row + 10 * fold] * 128 for fold in range(6)] for row in range(3)], dtype=np.float32)
            np.testing.assert_array_equal(np.load('out/mol_reps0None.npy'), expected)
            # Fully completed run follows the readback path, including NumPy export.
            run()
            np.testing.assert_array_equal(np.load('out/mol_reps0None.npy'), expected)


class BenchmarkChecks(unittest.TestCase):
    def create_targets(self):
        for target in ('b', 'a', 'missing'):
            folder = Path('data/lit_pcba') / target
            folder.mkdir(parents=True)
            if target != 'missing':
                (folder / 'mols.lmdb').touch()
                (folder / 'pockets.lmdb').touch()

    def test_selection_reports_and_unique_output(self):
        with temp_cwd():
            self.create_targets()
            metric = {key: 2.0 for key in benchmark_utils.RATIOS}
            evaluate = MagicMock(return_value=(.8, .6, metric, metric))
            with contextlib.redirect_stdout(io.StringIO()):
                first = benchmark_utils.run_benchmark('PCBA', evaluate, True, targets=['b', 'missing'])
                second = benchmark_utils.run_benchmark('PCBA', evaluate, False, targets=['a'])
            self.assertNotEqual(first, second)
            self.assertEqual([call.args[0] for call in evaluate.call_args_list], ['b', 'a'])
            report = json.loads((first / 'summary.json').read_text())
            self.assertEqual(report['means']['ef_0.01'], 2.0)
            self.assertTrue(report['use_folds'])
            self.assertIn('mols.lmdb', report['records'][1]['reason'])
            with self.assertRaisesRegex(ValueError, 'Unknown targets'):
                benchmark_utils.select_targets('data/lit_pcba', ['typo'])

    def test_failure_saves_report_and_propagates(self):
        with temp_cwd():
            self.create_targets()
            evaluate = MagicMock(side_effect=RuntimeError('broken input'))
            with self.assertRaisesRegex(RuntimeError, 'PCBA/a failed'):
                benchmark_utils.run_benchmark('PCBA', evaluate, False, targets=['a'])
            report = json.loads(next(Path('results/benchmarks').glob('*/summary.json')).read_text())
            self.assertEqual(report['counts']['failed'], 1)
            self.assertIn('broken input', report['records'][0]['reason'])
