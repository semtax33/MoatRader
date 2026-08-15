from __future__ import annotations

from scripts.merge_kr_signal_panel import spearman


def test_spearman_handles_ties_and_detects_unstable_scores() -> None:
    assert spearman([0, 1, 1, 2], [0, 1, 1, 2]) == 1.0
    assert spearman([0, 1, 2], [2, 1, 0]) == -1.0
