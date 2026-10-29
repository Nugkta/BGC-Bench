"""Tasks: the interface and registry (``base``) and the five tasks of the main table."""

from . import chem, coarse, fine, retrieval, substrate
from .base import REGISTRY, SUITE_V1, Task, get_task, is_official, list_tasks, register_task

for _cls in (coarse.Coarse, fine.Fine, chem.Chem, retrieval.Retrieval, substrate.Substrate):
    register_task(_cls)
del _cls

__all__ = ["REGISTRY", "SUITE_V1", "Task", "get_task", "is_official", "list_tasks",
           "register_task"]
