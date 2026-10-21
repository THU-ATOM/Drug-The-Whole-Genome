"""Common model setup for the standalone inference entry points."""
import logging
import os
import sys

logger = logging.getLogger("unimol.inference")


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )


def setup_inference(args, load_checkpoint=False, half_on_cpu=True):
    """Build an eval model, preserving each caller's precision policy.

    Multi-fold tasks load their own checkpoints. Single-checkpoint benchmarking
    loads args.path here, as before. This helper does not alter model architecture.
    """
    import torch
    from unicore import checkpoint_utils, tasks

    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    state = None
    if load_checkpoint:
        logger.info("loading model from %s", args.path)
        state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    if state is not None:
        model.load_state_dict(state["model"], strict=False)
    if args.fp16 and (use_cuda or half_on_cpu):
        model.half()
    if use_cuda:
        model.cuda()
    model.eval()
    logger.info(args)
    return task, model, use_cuda
