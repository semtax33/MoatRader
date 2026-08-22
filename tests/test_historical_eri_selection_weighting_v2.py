from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_historical_eri_selection_weighting_v2 import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _effective_sample_size,
    _weighted_quantile,
    cross_fitted_propensity,
    weighted_spearman,
)


def test_weighted_quantile_and_effective_sample_size() -> None:
    assert _weighted_quantile([1, 2, 3], [1, 1, 8], 0.5) == 3
    assert _effective_sample_size([1, 1, 1, 1]) == 4
    assert _effective_sample_size([1, 0, 0, 0]) == 1


def test_weighted_spearman_changes_with_selection_weights() -> None:
    unweighted = weighted_spearman(
        [1, 2, 3, 4, 5], [1, 2, 3, 5, 4], [1, 1, 1, 1, 1]
    )
    weighted = weighted_spearman(
        [1, 2, 3, 4, 5], [1, 2, 3, 5, 4], [10, 10, 10, 1, 1]
    )
    assert unweighted == pytest.approx(0.9)
    assert weighted > unweighted


def test_cross_fitted_propensity_is_group_disjoint_and_finite() -> None:
    rows = []
    for issuer in range(30):
        for observation in range(3):
            base = {
                field: float(issuer + observation + index)
                for index, field in enumerate(NUMERIC_FEATURES)
            }
            base.update({field: f"{field}_{issuer % 3}" for field in CATEGORICAL_FEATURES})
            rows.append(
                {
                    **base,
                    "issuer_id": f"I{issuer:02d}",
                    "final_common": (issuer + observation) % 4 == 0,
                }
            )
    probabilities, folds = cross_fitted_propensity(pd.DataFrame(rows), folds=5)
    assert len(probabilities) == 90
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert len(folds) == 5
    assert sum(row["test_observation_count"] for row in folds) == 90
