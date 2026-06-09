#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from unicore import distributed_utils, options, tasks
from .inference_utils import setup_logging, load_model

setup_logging()


def main(args):
    task = tasks.setup_task(args)
    model, use_cuda = load_model(args, task)
    model.eval()
    task.encode_mols_multi_folds(
        model, args.batch_size, args.mol_path, args.save_dir, use_cuda,
        write_npy=args.write_npy, write_h5=args.write_h5, start=args.start, end=args.end,
    )


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--write-npy", action="store_true")
    parser.add_argument("--write-h5", action="store_true")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
