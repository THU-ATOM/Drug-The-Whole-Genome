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



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    #state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    #model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_cuda:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    
    #names, scores = task.retrieve_mols(model, args.mol_path, args.pocket_path, args.emb_dir, 10000)

    # single conformer & multi conformer share same retrieval logic
    res_matrix, mol_names = task.drugclip_retrieval_scoring(
    model,
    pocket_path=args.pocket_path,
    mol_data_path=args.mol_path,
    fold_version=args.fold_version,
    use_cache=args.use_cache,
    use_cuda=use_cuda,
    custom_cache_dir=args.custom_cache_dir,
    multi_conf=args.multi_conf  
    )

    # single conformer & multi conformer have different aggregation logic
    if args.multi_conf:
        task.save_multiconformer_results(
            res_new=res_matrix,
            mol_names=mol_names,
            pocket_data_path=args.pocket_path,
            save_path=args.save_path,
            topk_percent=args.topk_percent
        )
    else:
        task.save_singleconformer_results(
            res_new=res_matrix,
            mol_names=mol_names,
            pocket_data_path=args.pocket_path,
            save_path=args.save_path,
            topk_percent=args.topk_percent
        )


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--pocket-path", type=str, default="", help="path for pocket data")
    parser.add_argument("--fold-version", type=str, default="6_folds", help="fold version")
    parser.add_argument("--use-cache", type=str, default="", help="whether use pre-encoded embeddings")
    parser.add_argument("--custom-cache-dir", type=str, default=None, help="optional custom cache path for ligand embeddings")
    parser.add_argument("--topk-percent", type=float, default=2.0, help="Percentage of top-ranked ligands to save in result (default: 2%%)",)
    parser.add_argument("--save-path", type=str, default="", help="path for saved result")
    parser.add_argument("--multi-conf", type=lambda x: x.lower() == "true", default=False, help="Enable multi-conformer ligand retrieval")

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
