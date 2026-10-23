import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import run_drugclip as runner


class EasyStartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_existing_output_is_preserved(self):
        out = self.root / 'run'
        out.mkdir()
        (out / 'valuable.txt').write_text('keep')
        with self.assertRaises(runner.UserError):
            runner.reserve_output(out)
        self.assertEqual((out / 'valuable.txt').read_text(), 'keep')

    def test_fresh_output_is_created(self):
        out = runner.reserve_output(self.root / 'run')
        self.assertTrue(out.is_dir())

    def test_resource_hash_rejects_same_size_corruption(self):
        path = self.root / 'asset'
        path.write_bytes(b'bad')
        expected = {'size': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}
        with self.assertRaises(runner.UserError):
            runner.verify_resource(path, expected)

    def test_resource_size_is_checked(self):
        path = self.root / 'asset'
        path.write_bytes(b'yes')
        with self.assertRaises(runner.UserError):
            runner.verify_resource(path, {'size': 5, 'sha256': hashlib.sha256(b'yes').hexdigest()})

    def test_verified_resource_is_accepted(self):
        path = self.root / 'asset'
        path.write_bytes(b'yes')
        self.assertEqual(runner.verify_resource(path, {'size': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}), hashlib.sha256(b'yes').hexdigest())

    def archive(self, members):
        path = self.root / 'archive.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, value in members:
                archive.writestr(name, value)
        return path

    def test_zip_traversal_rejected_before_any_write(self):
        archive = self.archive([('safe.txt', 'a'), ('../escaped.txt', 'b')])
        target = self.root / 'unpacked'
        with self.assertRaises(runner.UserError):
            runner.safe_extract(archive, target)
        self.assertFalse((self.root / 'escaped.txt').exists())
        self.assertFalse((target / 'safe.txt').exists())

    def test_windows_style_zip_traversal_is_rejected(self):
        archive = self.archive([('..\\escaped.txt', 'b')])
        with self.assertRaises(runner.UserError):
            runner.safe_extract(archive, self.root / 'unpacked')

    def test_zip_absolute_paths_are_rejected(self):
        archive = self.archive([('/absolute.txt', 'b')])
        with self.assertRaises(runner.UserError):
            runner.safe_extract(archive, self.root / 'unpacked')

    def test_valid_archive_extracts(self):
        archive = self.archive([('weights/fold_0.pt', b'bytes')])
        target = self.root / 'unpacked'
        runner.safe_extract(archive, target)
        self.assertEqual((target / 'weights/fold_0.pt').read_bytes(), b'bytes')

    def test_ambiguous_ligand_requires_explicit_instance(self):
        candidates = [{'code': 'LIG', 'chain': 'A', 'residue': '201'},
                      {'code': 'LIG', 'chain': 'B', 'residue': '201'}]
        with self.assertRaisesRegex(runner.UserError, 'ambiguous'):
            runner.select_ligand(candidates, 'LIG')
        self.assertEqual(runner.select_ligand(candidates, 'LIG', 'B')['chain'], 'B')

    def test_absent_ligand_fails(self):
        with self.assertRaisesRegex(runner.UserError, 'No matching'):
            runner.select_ligand([{'code': 'ABC', 'chain': 'A', 'residue': '1'}], 'LIG')

    def test_complex_requires_chains_and_ligand(self):
        with self.assertRaises(runner.UserError):
            runner.parse_cli(['screen', '--complex', 'complex.pdb'])

    def test_custom_inputs_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            runner.parse_cli(['screen', '--pocket-lmdb', 'a', '--pocket-pdb', 'b'])

    def test_demo_does_not_require_custom_files(self):
        args = runner.parse_cli(['demo', '--device', '2'])
        self.assertEqual(args.command, 'demo')
        self.assertEqual(args.device, '2')

    def test_missing_model_parameters_are_not_silently_random(self):
        with self.assertRaises(runner.UserError):
            runner.check_model_keys(['pocket_project.dense.weight'], [])
        runner.check_model_keys([], ['unused_training_head.weight'])

    def test_results_reject_nonfinite_scores(self):
        raw = self.root / 'raw.txt'
        raw.write_text('CCO,nan\n')
        with self.assertRaises(runner.UserError):
            runner.read_results(raw, 50)

    def test_results_reject_unexpected_count(self):
        raw = self.root / 'raw.txt'
        raw.write_text('CCO,4.0\n')
        with self.assertRaises(runner.UserError):
            runner.read_results(raw, 100)

    def test_results_require_descending_numeric_order(self):
        raw = self.root / 'raw.txt'
        raw.write_text('CCO,2.0\nCCN,10.0\n')
        with self.assertRaises(runner.UserError):
            runner.read_results(raw, 100)

    def test_result_names_and_scores_are_preserved(self):
        raw = self.root / 'raw.txt'
        raw.write_text('compound,with,comma,10.0\nCCO,2.0\n')
        rows = runner.read_results(raw, 100)
        self.assertEqual(rows[0], ('compound,with,comma', 10.0))
        self.assertEqual(rows[1], ('CCO', 2.0))

    def test_cropping_requires_an_explicitly_defined_smaller_pocket(self):
        import numpy as np
        record = {'pocket': 'too_big', 'pocket_atoms': ['C'] * 512,
                  'pocket_coordinates': np.zeros((512, 3), dtype=np.float32)}
        with self.assertRaises(runner.UserError):
            runner.validate_record(record)

    def test_atom_coordinate_mismatch_rejected(self):
        import numpy as np
        record = {'pocket': 'bad', 'pocket_atoms': ['C', 'N'],
                  'pocket_coordinates': np.zeros((1, 3), dtype=np.float32)}
        with self.assertRaises(runner.UserError):
            runner.validate_record(record)

    def test_valid_heavy_atom_count_ignores_stored_hydrogen(self):
        import numpy as np
        record = {'pocket': 'good', 'pocket_atoms': ['C', 'N', 'H'],
                  'pocket_coordinates': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)}
        self.assertEqual(runner.validate_record(record)['heavy_atoms'], 2)

    def test_container_command_limits_mount_writes_and_gpu_visibility(self):
        cmd = runner.container_command('image', 'GPU-abc', '/repo', '/cache', '/output')
        self.assertIn('device=GPU-abc', cmd)
        self.assertIn('/repo:/app:ro', cmd)
        self.assertIn('/cache:/assets:ro', cmd)
        self.assertIn('/output:/out:rw', cmd)
        self.assertNotIn('--privileged', cmd)

    def test_ordered_library_identity_mismatch_is_rejected(self):
        import numpy as np
        first = runner.validate_cache(np.ones((2, 128)), ['A', 'B'])
        with self.assertRaisesRegex(runner.UserError, 'order'):
            runner.validate_cache(np.ones((2, 128)), ['B', 'A'], first)

    def test_invalid_library_embeddings_are_rejected(self):
        import numpy as np
        for array in [np.ones((2, 127)), np.full((2, 128), float('nan'))]:
            with self.assertRaises(runner.UserError):
                runner.validate_cache(array, ['A', 'B'])

    def test_duplicate_pocket_names_are_rejected(self):
        records = [dict(pocket='same', pocket_atoms=['C'], pocket_coordinates=[[0,0,0]])] * 2
        with self.assertRaisesRegex(runner.UserError, 'Duplicate'):
            runner.validate_records(records)

    def test_download_complete_partial_is_verified_without_network(self):
        destination = self.root / 'asset.zip'
        destination.with_name('asset.zip.part').write_bytes(b'complete')
        runner.download_resource('https://invalid.invalid/asset', destination,
                                 {'size': 8, 'sha256': hashlib.sha256(b'complete').hexdigest()})
        self.assertEqual(destination.read_bytes(), b'complete')

    def test_server_ignoring_range_restarts_partial_instead_of_appending(self):
        path = self.root / 'asset'
        path.with_name('asset.part').write_bytes(b'abc')
        response = io.BytesIO(b'abcdef')
        response.status, response.headers = 200, {}
        with patch('urllib.request.urlopen', return_value=response):
            runner.download_resource('https://example.com', path,
                                     dict(size=6,sha256=hashlib.sha256(b'abcdef').hexdigest()))
        self.assertEqual(path.read_bytes(), b'abcdef')

    def test_valid_partial_response_resumes_exactly(self):
        path = self.root / 'asset'
        path.with_name('asset.part').write_bytes(b'abc')
        response = io.BytesIO(b'def')
        response.status, response.headers = 206, {'Content-Range':'bytes 3-5/6'}
        with patch('urllib.request.urlopen', return_value=response):
            runner.download_resource('https://example.com', path,
                                     dict(size=6,sha256=hashlib.sha256(b'abcdef').hexdigest()))
        self.assertEqual(path.read_bytes(), b'abcdef')

    def test_wrong_resume_range_preserves_existing_partial(self):
        path = self.root / 'asset'
        path.with_name('asset.part').write_bytes(b'abc')
        response = io.BytesIO(b'def')
        response.status, response.headers = 206, {'Content-Range':'bytes 1-3/6'}
        with patch('urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(runner.UserError, 'resume range'):
                runner.download_resource('https://example.com', path, dict(size=6,sha256='unused'))
        self.assertFalse(path.exists())
        self.assertEqual(path.with_name('asset.part').read_bytes(),b'abc')

    def test_environment_cache_key_changes_with_dependency_definition(self):
        (self.root/'docker').mkdir()
        (self.root/'docker/Dockerfile.inference').write_text('FROM base')
        requirements = self.root/'requirements-inference.txt'
        requirements.write_text('numpy==1.24.4')
        before = runner.environment_fingerprint(self.root)
        self.assertEqual(before,runner.environment_fingerprint(self.root))
        requirements.write_text('numpy==1.26.4')
        self.assertNotEqual(before,runner.environment_fingerprint(self.root))

    def test_nonmapping_record_has_an_actionable_error(self):
        with self.assertRaises(runner.UserError):
            runner.validate_record(['not','a','record'])


if __name__ == '__main__':
    unittest.main()
