#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore import distributed_utils, options
from inference_utils import configure_logging, setup_inference
from pathlib import Path

configure_logging()


def main(args):
    if not args.mol_path or not Path(args.mol_path).exists():
        raise FileNotFoundError(f"Molecule input not found: {args.mol_path!r}")
    if not args.save_dir:
        raise ValueError("--save-dir must name an output directory")
    if not (args.write_npy or args.write_h5):
        raise ValueError("Select --write-npy, --write-h5, or both")
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        raise ValueError("Require 0 <= --start < --end (or omit --end)")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    task, model, use_cuda = setup_inference(args)
    task.encode_mols_multi_folds(
        model, args.batch_size, args.mol_path, args.save_dir, use_cuda,
        write_npy=args.write_npy, write_h5=args.write_h5,
        start=args.start, end=args.end,
    )


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--save-dir", type=str, default="", help="save dir")
    parser.add_argument("--start", type=int, default=0, help="start index")
    parser.add_argument("--end", type=int, default=None, help="exclusive end index (default: all molecules)")
    parser.add_argument("--write-npy", action="store_true", help="write npy")
    parser.add_argument("--write-h5", action="store_true", help="write h5")

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
