from pathlib import Path

from scripts.preflight_expectation_gap_holdout import return_like


def test_preflight_rejects_return_like_inputs_before_signal_seal() -> None:
    assert return_like(Path("forward-returns.csv"))
    assert return_like(Path("evaluation.json"))
    assert return_like(Path("future_price.csv"))
    assert not return_like(Path("valuation-signals.csv"))
    assert not return_like(Path("research-inputs.json"))
