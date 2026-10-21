#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore import distributed_utils, options
from inference_utils import configure_logging, setup_inference
from inference_config import str2bool

configure_logging()


def main(args):
    task, model, _ = setup_inference(args, load_checkpoint=not args.use_folds)
    if args.test_task == "DUDE":
        task.test_dude(model, use_folds=args.use_folds)
    elif args.test_task == "PCBA":
        task.test_pcba(model, use_folds=args.use_folds)


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="DUDE", help="test task", choices=["DUDE", "PCBA"])
    parser.add_argument("--use-folds", type=str2bool, default=False, help="use 6 folds weights")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
