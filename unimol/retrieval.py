#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from unicore import distributed_utils, options, tasks
from .inference_utils import setup_logging, load_model

setup_logging()


def main(args):
    task = tasks.setup_task(args)
    model, use_cuda = load_model(args, task)
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
    parser.add_argument("--mol-path", type=str, default="")
    parser.add_argument("--pocket-path", type=str, default="")
    parser.add_argument("--fold-version", type=str, default="6_folds")
    parser.add_argument("--use-cache", type=str2bool, default=False)
    parser.add_argument("--save-path", type=str, default="")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
