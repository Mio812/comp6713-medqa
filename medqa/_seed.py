"""Single entrypoint for seeding RNGs used throughout the package.

Kept minimal and import-safe: numpy / torch / transformers imports are
optional so this can be called from scripts that haven't installed the
full dependency set.
"""

from __future__ import annotations

import random

from medqa.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Seed python-random, numpy, torch, and transformers (if available)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass
