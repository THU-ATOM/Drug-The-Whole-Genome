#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from unicore import distributed_utils, options
from inference_utils import configure_logging, setup_inference
from inference_config import FOLD_VERSIONS, str2bool, validate_retrieval_inputs

configure_logging()


def main(args):
    validate_retrieval_inputs(args.pocket_path, args.mol_path, args.fold_version,
                              args.use_cache, args.save_path)
    task, model, use_cuda = setup_inference(args, half_on_cpu=False)
    with torch.no_grad():
        task.retrieval_multi_folds(
            model, args.pocket_path, args.save_path, args.mol_path,
            fold_version=args.fold_version, use_cache=args.use_cache, use_cuda=use_cuda,
        )


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--pocket-path", type=str, default="", help="path for pocket data")
    parser.add_argument("--fold-version", type=str, default="6_folds", choices=FOLD_VERSIONS, help="fold version")
    parser.add_argument("--use-cache", type=str2bool, default=False, help="whether use pre-encoded embeddings")
    parser.add_argument("--save-path", type=str, default="", help="path for saved result")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
