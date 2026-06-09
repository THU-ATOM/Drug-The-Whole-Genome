# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for DrugCLIP inference entry-points."""

import logging
import os
import sys

import torch
from unicore import checkpoint_utils

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )


def load_model(args, task):
    """Build model, optionally load weights from args.path, move to device.

    Returns (model, use_cuda).
    """
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    model = task.build_model(args)

    if getattr(args, "path", None):
        logger.info("loading model(s) from {}".format(args.path))
        state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
        model.load_state_dict(state["model"], strict=False)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    logger.info(args)
    return model, use_cuda
