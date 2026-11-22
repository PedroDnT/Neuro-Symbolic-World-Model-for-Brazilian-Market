"""Rule arithmetic against values computed by hand."""
import pytest

from src.knowledge import rules


def test_negative_equity():
    assert rules.negative_equity(-5.0, "filed").breach is True
    assert rules.negative_equity(5.0, "filed").breach is False


def test_leverage_above_the_limit():
    result = rules.leverage(100.0, 30.0, 55.0, "filed")
    assert result.value == pytest.approx(0.85)
    assert result.breach is True


def test_leverage_below_the_limit():
    result = rules.leverage(100.0, 30.0, 40.0, "filed")
    assert result.value == pytest.approx(0.70)
    assert result.breach is False


def test_interest_coverage_uses_the_absolute_expense():
    # Financial expense is filed as a negative number.
    at_limit = rules.interest_coverage(150.0, -100.0)
    assert at_limit.value == pytest.approx(1.5)
    assert at_limit.breach is False
    assert rules.interest_coverage(149.0, -100.0).breach is True


def test_interest_coverage_with_a_missing_input_does_not_run():
    result = rules.interest_coverage(None, -100.0)
    assert result.value is None
    assert result.breach is None
    assert "not computed" in result.line()


def test_interest_coverage_under_shock():
    # Extra expense: 1000 * 0.5 * 3 / 100 = 15. Coverage: 200 / (100 + 15) = 1.7391.
    result = rules.interest_coverage_under_shock(200.0, -100.0, 1000.0, 0.5, 3.0, "extracted")
    assert result.value == pytest.approx(200 / 115)
    assert result.breach is False
    # 160 / 115 = 1.3913, below 1.5.
    assert rules.interest_coverage_under_shock(160.0, -100.0, 1000.0, 0.5, 3.0, "extracted").breach is True


def test_equity_loss_under_shock_uses_filed_equity():
    # The shock moves equity from 50 to 35: 15 % of the filed equity of 100, a breach.
    loss = rules.equity_loss(50.0, 35.0, 100.0)
    assert loss.value == pytest.approx(-15.0)
    assert loss.breach is True
    assert rules.equity_loss(50.0, 45.0, 100.0).breach is False


def test_equity_loss_does_not_run_on_non_positive_filed_equity():
    assert rules.equity_loss(-100.0, -120.0, -50.0).breach is None
    assert rules.equity_loss(10.0, 5.0, 0.0).breach is None


def test_leverage_does_not_pass_negative_liabilities():
    result = rules.leverage(100.0, -10.0, 50.0, "model")
    assert result.breach is None
    assert "out of range" in result.line()


def test_coverage_rules_do_not_run_on_a_zero_or_positive_expense():
    assert rules.interest_coverage(150.0, 0.0).breach is None
    assert rules.interest_coverage(150.0, 30.0).breach is None
    assert rules.interest_coverage_under_shock(200.0, 0.0, 1000.0, 0.5, 3.0, "extracted").breach is None


def test_rule_line_shows_the_numbers():
    line = rules.leverage(100.0, 30.0, 55.0, "model").line()
    assert "BREACH" in line and "0.85" in line and "model" in line


def test_breach_is_a_python_bool_for_numpy_inputs():
    import numpy as np

    result = rules.interest_coverage(np.float64(100.0), np.float64(-100.0))
    assert result.breach is True
