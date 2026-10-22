#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore import distributed_utils, options
from inference_utils import configure_logging, setup_inference
from inference_config import str2bool, fold_paths
from inference_checks import sha256_file
from benchmark_utils import select_targets
import torch

configure_logging()


def main(args):
    root = "data/DUD-E" if args.test_task == "DUDE" else "data/lit_pcba"
    targets = select_targets(root, args.targets)
    weights = fold_paths("6_folds")[0] if args.use_folds else [args.path]
    metadata = {"arguments": {key: str(value) for key, value in vars(args).items()},
                "checkpoints": [{"path": path, "sha256": sha256_file(path)} for path in weights]}
    task, model, _ = setup_inference(args, load_checkpoint=not args.use_folds, require_cuda=True)
    evaluate = task.test_dude if args.test_task == "DUDE" else task.test_pcba
    with torch.no_grad():
        evaluate(model, use_folds=args.use_folds, targets=targets,
                 results_dir=args.results_path or "./results/benchmarks", metadata=metadata)


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE", help="test task", choices=["DUDE", "PCBA"])
    parser.add_argument("--use-folds", type=str2bool, default=False, help="use 6 folds weights")
    parser.add_argument("--targets", nargs="+", help="target directory names (default: all)")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
