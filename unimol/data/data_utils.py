# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import contextlib


def pocket_atom(atom):
    """Extract element symbol from a pocket atom label.

    Pocket atom labels sometimes start with a digit (e.g. '1CA' → 'C').
    Returns the first non-digit character.
    """
    return atom[1] if atom[0].isdigit() else atom[0]


def softmax_weights(x):
    """Numerically stable softmax probability weights (numpy array)."""
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
