"""Exercise the example writer with real RDKit/LMDB and a synthetic test pocket."""
import ast
import contextlib
import io
import logging
from pathlib import Path
import pickle
import unittest
from unittest.mock import MagicMock

import lmdb
import numpy as np

from test_inference import ROOT, load_file, temp_cwd

prepare = load_file('mini_prepare', 'examples/mini_screen/prepare.py')
checker = load_file('mini_checker', 'examples/mini_screen/check_result.py')


class ExampleChecks(unittest.TestCase):
    def test_prepare_and_check(self):
        with temp_cwd(), contextlib.redirect_stdout(io.StringIO()):
            prepare.write_records('source.lmdb', [{
                'pocket': 'synthetic-test-pocket', 'pocket_atoms': ['C', 'N', 'O'],
                'pocket_coordinates': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            }])
            manifest = prepare.prepare('source.lmdb', 'mini')
            self.assertEqual(manifest['molecule_count'], 12)
            self.assertEqual(manifest['expected_result_rows'], 1)
            env = lmdb.open('mini/mols.lmdb', subdir=False, readonly=True, lock=False)
            try:
                with env.begin() as txn:
                    self.assertEqual(txn.stat()['entries'], 12)
                    for index in range(12):
                        record = pickle.loads(txn.get(str(index).encode()))
                        self.assertEqual(record['coordinates'].shape, (len(record['atoms']), 3))
                        self.assertTrue(np.isfinite(record['coordinates']).all())
            finally:
                env.close()
            # This score is a format-test value, not a neural-network result.
            Path('mini/result.csv').write_text(manifest['smiles'][0] + ',1.0\n')
            checker.check_result('mini/manifest.json', 'mini/result.csv')
            for value in ('unknown,1.0\n', manifest['smiles'][0] + ',nan\n', ''):
                Path('mini/result.csv').write_text(value)
                with self.assertRaises(ValueError):
                    checker.check_result('mini/manifest.json', 'mini/result.csv')

    def test_pdb_errors_are_reported_and_empty_output_is_rejected(self):
        import os
        source = ast.parse((ROOT / 'unimol/encode_pockets.py').read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'process_one_pdbdir')
        logger = logging.getLogger('pdb-test')
        writer = MagicMock()
        namespace = dict(os=os, logger=logger, PDBParser=MagicMock(), write_lmdb=writer,
                         extract_lig_recpt=lambda model, ligand: (None, ['ligand']),
                         get_binding_pockets=lambda chain, ligands: [('pocket', None)],
                         pocket2lmdb=lambda *args: {'pocket_atoms': ['C']})
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<pdb-test>', 'exec'), namespace)
        process = namespace['process_one_pdbdir']
        with temp_cwd():
            Path('bad.pdb').touch()
            with self.assertLogs(logger, level='INFO') as logs:
                with self.assertRaisesRegex(ValueError, 'No pockets extracted'):
                    process('.')
            self.assertIn('bad.pdb', '\n'.join(logs.output))
            writer.assert_not_called()
            Path('protein_with_underscores_LIG.pdb').touch()
            with self.assertLogs(logger, level='INFO') as logs:
                process('.')
            self.assertIn('1 succeeded, 1 failed, 1 pockets', '\n'.join(logs.output))
            writer.assert_called_once()
