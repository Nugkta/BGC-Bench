"""Task interface and registry.

A task scores one or more representation kinds (``requires``; see
:mod:`bgcbench.representations`) on its splits; each split is one result column
``<task>_<letter>``. The built-in tasks register in ``bgcbench.tasks`` in main-table order,
and ``SUITE_V1`` names the official suite that ``tasks=None`` runs. Other registered tasks
run only when named, and result tables mark them. Each built-in task module keeps its
protocol in a module-level ``evaluate`` function and ends with a thin :class:`Task` subclass
whose ``evaluate`` method calls it.
"""

from __future__ import annotations

from ..data import MAIN, Dataset, get_dataset

SUITE_V1 = ("coarse", "fine", "chem", "retrieval", "substrate")


class Task:
    """Base class for tasks. Subclass it, set the attributes and implement ``evaluate``.

    ``evaluate`` receives ``{kind: Embeddings}`` for the kinds in ``requires`` and one split
    name, and returns a dict with ``point`` (a number) and ``ci`` (``[lo, hi]``; if left out
    it is recorded as ``[nan, nan]``), plus any counts worth keeping. Raise
    :class:`bgcbench.data.CoverageError` when the vectors do not cover the rows the split
    needs; the column is then reported as "--".
    """

    name: str = ""
    #: Bump it whenever the task's data, protocol or metric changes; results record it.
    version: str = "1"
    requires: tuple[str, ...] = ("bgc",)
    splits: dict[str, str] = {}   # column letter -> split name
    metric: str = ""
    #: BLAS / torch CPU threads the published values were computed with. Linear probes sum in
    #: a thread-dependent order, so other counts move results in the 4th decimal;
    #: ``threads_override`` maps encoder names to their counts.
    threads: int = 1
    threads_override: dict[str, int] = {}
    #: Data the task ships: ``"org/repo@revision"`` on the Hugging Face Hub or a local
    #: directory, read through :attr:`data`. ``None``: the main BGC-Bench dataset. The
    #: built-in protocols read their splits and tables from :attr:`data`.
    dataset: str | None = None

    @property
    def data(self) -> Dataset:
        """The task's dataset: :attr:`dataset`, or the main BGC-Bench dataset."""
        return MAIN if self.dataset is None else get_dataset(self.dataset)

    def columns(self):
        """``(column id, split)`` per split, in column order; the id is ``<name>_<letter>``."""
        for letter, split in self.splits.items():
            yield f"{self.name}_{letter}", split

    def threads_for(self, encoder_name: str) -> int:
        """The protocol thread count for an encoder."""
        return self.threads_override.get(encoder_name, self.threads)

    def evaluate(self, reps: dict, split: str, *, n_boot: int, device: str) -> dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Task {self.name} v{self.version}>"


REGISTRY: dict[str, Task] = {}


def register_task(task: type[Task] | Task, *, replace: bool = False) -> type[Task] | Task:
    """Register a :class:`Task` subclass or instance under its ``name``; returns ``task``.

    Works as a class decorator. Registering the same class again (a re-imported module, a
    fresh instance) replaces the entry; a name taken by another class is an error unless
    ``replace=True``. The official tasks cannot be replaced.
    """
    obj = task() if isinstance(task, type) else task
    if not isinstance(obj, Task):
        raise TypeError(f"expected a Task subclass or instance, got {task!r}")
    if not obj.name or not isinstance(obj.name, str):
        raise ValueError(f"{type(obj).__name__} has no name")
    if not obj.splits:
        raise ValueError(f"task {obj.name!r} has no splits")
    current = REGISTRY.get(obj.name)
    if current is not None and current is not obj:
        if obj.name in SUITE_V1:
            raise ValueError(f"{obj.name!r} is an official task and cannot be replaced")
        if not replace and _qualname(current) != _qualname(obj):
            raise ValueError(f"task {obj.name!r} is already registered by "
                             f"{_qualname(current)}; pass replace=True to override")
    REGISTRY[obj.name] = obj
    return task


def _qualname(task: Task) -> str:
    return f"{type(task).__module__}.{type(task).__qualname__}"


def get_task(name: str) -> Task:
    """The registered task of a name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown task {name!r}; registered: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def list_tasks() -> list[str]:
    """Names of the registered tasks, official ones first in main-table order."""
    return list(REGISTRY)


def is_official(task: Task) -> bool:
    """Whether ``task`` is the registered task of an official-suite name."""
    return task.name in SUITE_V1 and REGISTRY.get(task.name) is task


def select(tasks=None) -> list[Task]:
    """Resolve names and :class:`Task` instances (registered on the way) in registry order.

    ``None`` is the official suite.
    """
    if tasks is None:
        return [REGISTRY[n] for n in SUITE_V1]
    if isinstance(tasks, (str, Task)):
        tasks = [tasks]
    chosen, unknown = set(), []
    for t in tasks:
        if isinstance(t, Task):
            if REGISTRY.get(t.name) is not t:
                register_task(t)
            chosen.add(t.name)
        elif t in REGISTRY:
            chosen.add(t)
        else:
            unknown.append(t)
    if unknown:
        raise ValueError(f"unknown task(s) {sorted(map(str, unknown))}; "
                         f"choose from {list(REGISTRY)}")
    return [task for name, task in REGISTRY.items() if name in chosen]
