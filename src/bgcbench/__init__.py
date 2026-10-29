"""BGC-Bench: frozen-embedding evaluation of biosynthetic gene cluster encoders.

Quick use::

    import bgcbench
    result = bgcbench.run("bigcarp")                  # a built-in encoder, end to end
    result = bgcbench.evaluate("my_embeddings.npz")   # your own vectors
    print(result.table())

Dataset revisions, model weights and evaluation seeds are pinned below.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path

__version__ = "1.1.0"

# Dataset: splits, task tables, model inputs (see the dataset card).
DATASET_REPO = "Nugkta/bgc-bench"
DATASET_REVISION = "v1.0.0"

# Encoder weights. Upstream repositories are used where they exist, pinned to commits because
# they carry no tags; the rest are mirrors of the authors' released checkpoints, pinned to
# their release tag (see each model card for the upstream source and conversions). A pin that
# changes the weights needs a new ``version`` for every encoder using it (``esm2_8m`` serves
# BGC-Prophet and the LC-PLM tokenizer), since the version keys the embedding cache.
MODELS = {
    "bigcarp": ("Nugkta/bgcbench-bigcarp", "v1.0.0"),
    "kawano": ("Nugkta/bgcbench-kawano", "v1.0.0"),
    "deepbgc": ("Nugkta/bgcbench-deepbgc", "v1.0.0"),
    "lcplm": ("Nugkta/bgcbench-lcplm", "v1.0.0"),
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D", "08e4846e537177426273712802403f7ba8261b6c"),
    "esm2_8m": ("facebook/esm2_t6_8M_UR50D", "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"),
    "esmc_600m": ("biohub/esmc-600m-2024-12", "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c"),
    "evo2_7b": ("arcinstitute/evo2_7b", "bda0089f92582d5baabf0f22d9fc85f3588f6b58"),
}

SEED = 42


def home() -> Path:
    """Cache directory for the dataset and embeddings (``$BGCBENCH_HOME``).

    Model weights go to the Hugging Face cache (``$HF_HOME``).
    """
    return Path(os.environ.get("BGCBENCH_HOME", "~/.cache/bgcbench")).expanduser()


def set_deterministic(seed: int = SEED) -> None:
    """Seed Python, NumPy and torch, and ask torch for deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@contextmanager
def limit_threads(n: int):
    """Cap BLAS/OpenMP and torch CPU threads at ``n`` for a block.

    The pipeline scores each column inside this block with the task's protocol thread count.
    """
    import torch
    from threadpoolctl import threadpool_limits

    saved = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        torch.set_num_threads(saved)


from .encoders.base import Encoder, get_encoder, list_encoders  # noqa: E402
from .pipeline import embed, evaluate, run  # noqa: E402
from .representations import Embeddings, register_representation  # noqa: E402
from .results import Result  # noqa: E402
from .tasks import Task, list_tasks, register_task  # noqa: E402

__all__ = [
    "Embeddings", "Encoder", "Result", "Task", "embed", "evaluate", "get_encoder", "home",
    "limit_threads", "list_encoders", "list_tasks", "register_representation", "register_task",
    "run", "set_deterministic",
]
