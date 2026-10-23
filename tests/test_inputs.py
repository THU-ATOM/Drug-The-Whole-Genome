"""Input integration checks; run inside the declared inference environment."""
from pathlib import Path
import pickle
import tempfile
import unittest
import lmdb
import numpy as np
import run_drugclip as runner


def atom(serial, name, residue, chain, number, xyz, element, hetero=False):
    x,y,z = xyz
    return f'{"HETATM" if hetero else "ATOM  "}{serial:5d} {name:^4s} {residue:3s} {chain}{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{20.:6.2f}          {element:>2s}\n'


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pdb = self.root / 'input.pdb'
        self.protein = atom(1,'CA','ALA','A',1,(0,0,0),'C') + atom(2,'N','ALA','A',1,(1,0,0),'N') + atom(3,'CA','ALA','A',2,(20,0,0),'C')

    def tearDown(self):
        self.tmp.cleanup()

    def test_pocket_pdb_keeps_coordinates_and_removes_hydrogen(self):
        self.pdb.write_text(self.protein + atom(4,'H','ALA','A',2,(20,1,0),'H') + 'END\n')
        records = runner.prepare_pdb(self.pdb, {}, self.root)
        np.testing.assert_array_equal(records[0]['pocket_coordinates'], [[0,0,0],[1,0,0],[20,0,0]])
        self.assertEqual(records[0]['pocket_atoms'], ['C','N','C'])

    def test_complex_requires_correct_chain_and_extracts_whole_residues(self):
        self.pdb.write_text(self.protein + atom(4,'C1','LIG','A',501,(-5.5,0,0),'C',True) + 'END\n')
        options = dict(complex=True,chains='A',ligand='LIG')
        result = runner.prepare_pdb(self.pdb, options, self.root)[0]
        # N is 6.5 A from the ligand, but belongs to the selected ALA residue.
        self.assertEqual(result['pocket_atoms'], ['C','N'])
        with self.assertRaisesRegex(runner.UserError, 'chains'):
            runner.prepare_pdb(self.pdb, dict(options,chains='Z'), self.root)

    def test_ligand_cannot_be_silently_discarded_in_pocket_mode(self):
        self.pdb.write_text(self.protein + atom(4,'C1','LIG','A',501,(0,1,0),'C',True) + 'END\n')
        with self.assertRaisesRegex(runner.UserError, 'only standard'):
            runner.prepare_pdb(self.pdb, {}, self.root)

    def test_complex_matches_reference_cutoff_when_protein_contains_hydrogen(self):
        self.pdb.write_text(self.protein + atom(4,'H','ALA','A',2,(18,0,0),'H') +
                            atom(5,'C1','LIG','A',501,(12.8,0,0),'C',True) + 'END\n')
        records = runner.prepare_pdb(self.pdb, dict(complex=True,chains='A',ligand='LIG'), self.root)
        self.assertEqual(records[0]['pocket_atoms'],['C'])
        np.testing.assert_array_equal(records[0]['pocket_coordinates'], [[20,0,0]])

    def test_multiple_ligand_instances_require_disambiguation(self):
        self.pdb.write_text(self.protein + atom(4,'C1','LIG','A',501,(0,1,0),'C',True) + atom(5,'C1','LIG','B',501,(0,2,0),'C',True) + 'END\n')
        with self.assertRaisesRegex(runner.UserError, 'ambiguous'):
            runner.prepare_pdb(self.pdb, dict(complex=True,chains='A',ligand='LIG'), self.root)

    def test_multi_model_input_requires_selection(self):
        self.pdb.write_text('MODEL        1\n' + self.protein + 'ENDMDL\nMODEL        2\n' + self.protein + 'ENDMDL\n')
        with self.assertRaisesRegex(runner.UserError, 'single-model'):
            runner.prepare_pdb(self.pdb, {}, self.root)

    def test_alternate_atom_uses_selected_occupancy_and_is_not_duplicated(self):
        def alternate(code, xyz, occupancy):
            line = atom(1 if code=='A' else 2,'CA','ALA','A',1,xyz,'C')
            return line[:16] + code + line[17:54] + f'{occupancy:6.2f}' + line[60:]
        self.pdb.write_text(alternate('A',(0,0,0),0.3) + alternate('B',(10,0,0),0.7) + 'END\n')
        result = runner.prepare_pdb(self.pdb, {}, self.root)[0]
        np.testing.assert_array_equal(result['pocket_coordinates'],[[10,0,0]])


    def test_lmdb_roundtrip_preserves_multiple_templates(self):
        source = self.root / 'input.lmdb'
        records = [dict(pocket=name,pocket_atoms=['C','N'],pocket_coordinates=np.array([[0,0,0],[1,2,3]],np.float32)) for name in ('one','two')]
        env = lmdb.open(str(source), subdir=False, map_size=1024**2)
        with env.begin(write=True) as txn:
            for i,record in enumerate(records): txn.put(str(i).encode(),pickle.dumps(record))
        env.close()
        summaries = runner.prepare_input(dict(input_type='lmdb',input_suffix='.lmdb'), self.root)
        self.assertEqual([s['name'] for s in summaries], ['one','two'])
        env = lmdb.open(str(self.root/'prepared_pocket.lmdb'),subdir=False,readonly=True,lock=False)
        with env.begin() as txn:
            result = pickle.loads(txn.get(b'1'))
        env.close()
        np.testing.assert_array_equal(result['pocket_coordinates'],records[1]['pocket_coordinates'])


if __name__ == '__main__': unittest.main()
