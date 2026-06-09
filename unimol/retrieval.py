#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
import torch
from unicore import distributed_utils, options
from unicore import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    task = tasks.setup_task(args)
    model = task.build_model(args)

    # Move models to GPU
    if use_cuda:
        if use_fp16:
            model.half()
        model.cuda()

    logger.info(args)

    model.eval()
    task.retrieval_multi_folds(
        model,
        args.pocket_path,
        args.save_path,
        args.mol_path,
        fold_version=args.fold_version,
        use_cache=args.use_cache,
        use_cuda=use_cuda,
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--pocket-path", type=str, default="", help="path for pocket data")
    parser.add_argument("--fold-version", type=str, default="6_folds", help="fold version")
    parser.add_argument("--use-cache", type=str2bool, default=False, help="whether use pre-encoded embeddings")
    parser.add_argument("--save-path", type=str, default="", help="path for saved result")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
