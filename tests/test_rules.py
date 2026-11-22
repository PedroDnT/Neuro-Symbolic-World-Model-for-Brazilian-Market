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


def test_equity_loss_under_shock():
    loss = rules.equity_loss(100.0, 85.0)
    assert loss.value == pytest.approx(-15.0)
    assert loss.breach is True
    assert rules.equity_loss(100.0, 95.0).breach is False


def test_equity_loss_with_negative_base_uses_the_absolute_value():
    # Base -100 and shock -120: equity falls by 20 % of its absolute size.
    assert rules.equity_loss(-100.0, -120.0).value == pytest.approx(-20.0)


def test_rule_line_shows_the_numbers():
    line = rules.leverage(100.0, 30.0, 55.0, "model").line()
    assert "BREACH" in line and "0.85" in line and "model" in line
