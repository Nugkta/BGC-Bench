"""The descriptor targets of the chem task: float64, column-major."""

import numpy as np
import pandas as pd
import pytest

from bgcbench.tasks import chem


def test_targets_are_float64_and_column_major():
    df = pd.DataFrame(np.random.default_rng(0).random((10, len(chem.TARGETS))),
                      columns=chem.TARGETS)
    mask = np.arange(10) % 3 != 0
    Y = chem._targets(df, mask)
    assert Y.dtype == np.float64 and Y.flags.f_contiguous
    np.testing.assert_array_equal(Y, df.loc[mask, chem.TARGETS].to_numpy())


def test_targets_reject_non_float_columns():
    df = pd.DataFrame({t: np.arange(4, dtype=np.int64) for t in chem.TARGETS})
    with pytest.raises(AssertionError):
        chem._targets(df, np.ones(4, dtype=bool))
