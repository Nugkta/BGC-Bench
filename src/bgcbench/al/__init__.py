"""Active learning: learning curves of frozen embeddings under a growing label budget.

Each task's AL rows are split once into an acquisition pool, validation rows and test rows.
A learning curve starts from 16 labels drawn by its seed and grows by a factor of about
sqrt(2); at every budget the task's readout is refitted on the labelled rows and scored on
the validation and test rows, and a strategy chooses the next rows to label. Curves are
summarised by AULC_low, the area under the test-score curve over 16-256 labels on a log2
axis, divided by the axis width.

Quick use::

    from bgcbench import al
    res = al.run("bag_of_pfam")                         # quick protocol: 5 seeds, <= 256 labels
    res = al.run("bigcarp", protocol="paper", workers=8)
    res = al.evaluate("my.npz", tasks=["coarse"], strategies=["random", "typiclust"])
    print(res.table())

Tasks take part through ``Task.al_rows`` and ``Task.al_readout``; strategies are registered
with :func:`register_strategy`; encoders are anything :func:`bgcbench.run` accepts.
"""

from __future__ import annotations

from .design import PROTOCOLS, Partition, Protocol, Rows, parse_seeds, partition, protocol
from .readouts import Classifier, GradEmbedding, MultiLabel, Readout, Regressor
from .strategies import Context, get_strategy, list_strategies, register_strategy

_LAZY = {"run": "runner", "evaluate": "runner", "load": "runner", "ALResult": "runner",
         "list_tasks": "runner", "summarize": "summary", "Summary": "summary"}


def __getattr__(name: str):
    # The runner imports the task registry, whose modules import this package.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ALResult", "Classifier", "Context", "GradEmbedding", "MultiLabel", "PROTOCOLS",
           "Partition", "Protocol", "Readout", "Regressor", "Rows", "evaluate", "get_strategy",
           "list_strategies", "list_tasks", "load", "parse_seeds", "partition", "protocol",
           "register_strategy", "run", "summarize", "Summary"]
