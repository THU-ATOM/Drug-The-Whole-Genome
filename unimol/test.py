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
    model, _ = load_model(args, task)
    model.eval()
    if args.test_task == "DUDE":
        task.test_dude(model, use_folds=args.use_folds)
    elif args.test_task == "PCBA":
        task.test_pcba(model, use_folds=args.use_folds)


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE", choices=["DUDE", "PCBA"])
    parser.add_argument("--use-folds", type=str, default=True)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
