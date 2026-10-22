#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
from inference_utils import setup_inference
import numpy as np
from tqdm import tqdm
import unicore

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve


import os
from Bio.PDB import PDBParser,Chain,Model,Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import is_aa
from Bio.PDB.Residue import DisorderedResidue,Residue
from Bio.PDB.Atom import DisorderedAtom
import warnings
from Bio.PDB.StructureBuilder import PDBConstructionWarning
from tqdm import tqdm
import numpy as np
import lmdb
import numpy as np
import pickle
import re

def write_lmdb(data, lmdb_path, num):
    #resume
    
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    try:
        with env.begin(write=True) as txn:
            for d in data:
                txn.put(str(num).encode('ascii'), pickle.dumps(d))
                num += 1
    finally:
        env.close()
    
    return num

warnings.filterwarnings(
    action='ignore',
    category=PDBConstructionWarning)
def extract_lig_recpt(biopy_model,ligname):
    lig_list = []
    tmp_chain = Chain.Chain('A') 
    rid = 0
    for chain in biopy_model:
        for res in chain:
            res.detach_parent()
            if not(is_aa(res,standard=True)):
                if res.resname == ligname:
                    lig_list.append((chain.id+'_'+str(res.id[1]),res))
                continue
            if res.is_disordered():
                if isinstance(res,DisorderedResidue):
                    res = res.selected_child
                    res.id = (res.id[0],rid,res.id[2])
                    rid += 1 
                    tmp_chain.add(res.copy())      
                else:
                    new_res = Residue(res.id,res.resname,res.segid)
                    for atom in res:
                        if isinstance(atom,DisorderedAtom):  
                            atom.selected_child.disordered_flag = 0
                            new_res.add(atom.selected_child.copy())
                        else:
                            new_res.add(atom)
                    res = new_res
                    res.id = (res.id[0],rid,res.id[2])
                    rid += 1 
                    tmp_chain.add(res.copy())       
            else:
                res.id = (res.id[0],rid,res.id[2])
                rid += 1 
                tmp_chain.add(res.copy())
    # tmp_structure = Structure.Structure('tmp')
    # tmp_model = Model.Model(0)
    # tmp_structure.add(tmp_model) 
    # tmp_model.add(tmp_chain)
  
    return tmp_chain,lig_list

def get_binding_pockets(biopy_chain,liglist):
    pockets = []
    for n,lig in liglist:
            lig_coord = np.array([i.coord for i in lig.get_atoms() if i.element!='H'])
            tmp_chain = Chain.Chain('A') 
            for res in biopy_chain:
                res_coord = np.array([i.get_coord() for i in res.get_atoms()])
                dist = np.linalg.norm(res_coord[:,None,:]-lig_coord[None,:,:],axis=-1).min()
                if dist<=6:
                    tmp_chain.add(res.copy())
            pockets.append((n,tmp_chain))
    return pockets  

def pocket2lmdb(pocket_name,biopy_chain,pdb_name):
    recpt = list(biopy_chain.get_atoms())
    pocket_atom_type = [x.element for x in recpt if x.element!='H']
    pocket_coord = [x.coord for x in recpt if x.element!='H']
    print(pdb_name+'_'+pocket_name,len(pocket_atom_type))
    return {
        'pocket': pdb_name+'_'+pocket_name,
        'pocket_atoms': pocket_atom_type,
        'pocket_coordinates': pocket_coord
    }


def process_one_pdbdir(dirs, name="pocket"):
    output = os.path.join(dirs, f"{name}.lmdb")
    if os.path.exists(output):
        logger.info("Reusing existing pocket LMDB: %s", output)
        return 0
    all_pocket = []
    succeeded = failed = 0
    for filename in sorted(os.listdir(dirs)):
        if not filename.lower().endswith(".pdb") or "_clean" in filename:
            continue
        try:
            stem = os.path.splitext(filename)[0]
            if "_" not in stem or not stem.rsplit("_", 1)[1]:
                raise ValueError("Expected filename <structure>_<ligand>.pdb")
            ligand = stem.rsplit("_", 1)[1]
            model = PDBParser().get_structure("0", os.path.join(dirs, filename))[0]
            chain, ligands = extract_lig_recpt(model, ligand)
            if not ligands:
                raise ValueError(f"Ligand {ligand!r} not found in the first PDB model")
            pockets = [pocket2lmdb(n, p, stem) for n, p in get_binding_pockets(chain, ligands)]
            if not pockets or any(not record["pocket_atoms"] for record in pockets):
                raise ValueError("No nonempty protein pocket within 6 angstroms of the ligand")
            all_pocket.extend(pockets)
            succeeded += 1
        except Exception as exc:
            failed += 1
            logger.warning("Skipping PDB %s: %s", filename, exc)
    logger.info("PDB preprocessing: %d succeeded, %d failed, %d pockets",
                succeeded, failed, len(all_pocket))
    if not all_pocket:
        raise ValueError(f"No pockets extracted from {dirs}; see PDB errors above")
    write_lmdb(all_pocket, output, 0)
    return 1


def main(args):

    if not os.path.isdir(args.pocket_dir):
        raise FileNotFoundError(f"Pocket directory not found: {args.pocket_dir}")
    task, model, _ = setup_inference(args, require_cuda=True)

    if not os.path.exists(os.path.join(args.pocket_dir, "pocket.lmdb")):
        ret = process_one_pdbdir(args.pocket_dir)

    # read pocket dir 
    with torch.no_grad():
        pocket_reps, pocket_names = task.encode_pockets_multi_folds(
            model, args.pocket_dir, os.path.join(args.pocket_dir, "pocket.lmdb"))

    # print shape

    logger.info(f"pocket_reps shape: {pocket_reps.shape}")

    # print name

    logger.info(f"pocket_names shape: {pocket_names}")


    # save names and reps together to pickle file

    
    with open(os.path.join(args.pocket_dir, "pocket_reps.pkl"), "wb") as f:
        pickle.dump((pocket_names, pocket_reps), f)
   

    


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--pocket-dir", type=str, default="", help="path for pocket dir")

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
