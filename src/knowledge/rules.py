"""Accounting rules. Each rule returns a record with its arithmetic, so a person can check it.

A rule never changes a number. It reads its inputs, computes 1 value and compares the
value with a threshold. The thresholds are conventions of this project, not regulatory
limits. Change them in THRESHOLDS.

`basis` says where the inputs come from:
    filed      numbers that the company filed
    model      numbers that the neural model predicted
    extracted  a share that the extraction read from the notes
    estimate   a share that a regression estimated

Run on the filed history of 1 company:  python -m src.knowledge.rules --cd-cvm 11312
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import pandas as pd

from src.data_ingestion.download import DATA_DIR

THRESHOLDS = {"leverage": 0.80, "interest_coverage": 1.5, "equity_loss_pct": 10.0}


@dataclass(frozen=True)
class RuleResult:
    rule: str
    formula: str
    value: float | None
    threshold: float | None
    breach: bool | None  # None when an input is missing, so the rule cannot run
    inputs: dict
    basis: str

    def line(self) -> str:
        if self.breach is None:
            return f"{self.rule}: not computed, an input is missing [{self.basis}]"
        verdict = "BREACH" if self.breach else "ok"
        numbers = ", ".join(f"{name} = {value:,.4g}" for name, value in self.inputs.items())
        return f"{self.rule}: {self.formula} -> {self.value:,.4g} (limit {self.threshold:g}) {verdict} [{self.basis}; {numbers}]"

    def to_dict(self) -> dict:
        return asdict(self)


def _missing(*values: object) -> bool:
    return any(value is None or pd.isna(value) for value in values)


def negative_equity(equity: float, basis: str) -> RuleResult:
    return RuleResult("negative_equity", "equity < 0", equity, 0.0, equity < 0, {"equity": equity}, basis)


def leverage(total_assets: float, current_liabilities: float, noncurrent_liabilities: float, basis: str) -> RuleResult:
    inputs = {
        "current_liabilities": current_liabilities,
        "noncurrent_liabilities": noncurrent_liabilities,
        "total_assets": total_assets,
    }
    formula = "(current_liabilities + noncurrent_liabilities) / total_assets"
    limit = THRESHOLDS["leverage"]
    if _missing(*inputs.values()) or total_assets <= 0:
        return RuleResult("leverage", formula, None, limit, None, inputs, basis)
    value = (current_liabilities + noncurrent_liabilities) / total_assets
    return RuleResult("leverage", formula, value, limit, value > limit, inputs, basis)


def interest_coverage(ebit_ttm: float, finexp_ttm: float, basis: str = "filed") -> RuleResult:
    inputs = {"ebit_ttm": ebit_ttm, "financial_expense_ttm": finexp_ttm}
    formula = "ebit_ttm / |financial_expense_ttm|"
    limit = THRESHOLDS["interest_coverage"]
    if _missing(ebit_ttm, finexp_ttm) or finexp_ttm == 0:
        return RuleResult("interest_coverage", formula, None, limit, None, inputs, basis)
    value = ebit_ttm / abs(finexp_ttm)
    return RuleResult("interest_coverage", formula, value, limit, value < limit, inputs, basis)


def interest_coverage_under_shock(
    ebit_ttm: float, finexp_ttm: float, gross_debt: float, floating_share: float, selic_delta_pp: float, basis: str
) -> RuleResult:
    """Interest Coverage when the floating part of the debt reprices by the SELIC change.

    Assumption: the full change passes to the floating debt for 1 year, and EBIT stays.
    """
    inputs = {
        "ebit_ttm": ebit_ttm,
        "financial_expense_ttm": finexp_ttm,
        "gross_debt": gross_debt,
        "floating_share": floating_share,
        "selic_delta_pp": selic_delta_pp,
    }
    formula = "ebit_ttm / (|financial_expense_ttm| + gross_debt * floating_share * selic_delta_pp / 100)"
    limit = THRESHOLDS["interest_coverage"]
    if _missing(*inputs.values()):
        return RuleResult("interest_coverage_under_shock", formula, None, limit, None, inputs, basis)
    expense = abs(finexp_ttm) + gross_debt * floating_share * selic_delta_pp / 100
    if expense <= 0:
        return RuleResult("interest_coverage_under_shock", formula, None, limit, None, inputs, basis)
    value = ebit_ttm / expense
    return RuleResult("interest_coverage_under_shock", formula, value, limit, value < limit, inputs, basis)


def equity_loss(equity_base: float, equity_shock: float, basis: str = "model") -> RuleResult:
    inputs = {"equity_base": equity_base, "equity_shock": equity_shock}
    formula = "(equity_shock - equity_base) / |equity_base| * 100"
    limit = -THRESHOLDS["equity_loss_pct"]
    if _missing(equity_base, equity_shock) or equity_base == 0:
        return RuleResult("equity_loss_under_shock", formula, None, limit, None, inputs, basis)
    value = (equity_shock - equity_base) / abs(equity_base) * 100
    return RuleResult("equity_loss_under_shock", formula, value, limit, value < limit, inputs, basis)


def balance_rules(state: dict, basis: str) -> list[RuleResult]:
    """The rules that need only the 4 balance accounts."""
    return [
        negative_equity(state["equity"], basis),
        leverage(state["total_assets"], state["current_liabilities"], state["noncurrent_liabilities"], basis),
    ]


def filed_history(cd_cvm: int) -> pd.DataFrame:
    """Rule results on each filed quarter of 1 company (any company of the statement table)."""
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    rows = []
    for row in statements[statements["cd_cvm"] == cd_cvm].sort_values("quarter").to_dict("records"):
        results = balance_rules(row, "filed") + [interest_coverage(row["ebit_ttm"], row["finexp_ttm"])]
        record = {"quarter": row["quarter"].date(), "company": row["company"]}
        for result in results:
            record[result.rule] = result.value
            record[f"{result.rule}_breach"] = result.breach
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule results on the filed history of 1 company")
    parser.add_argument("--cd-cvm", type=int, required=True)
    args = parser.parse_args()
    history = filed_history(args.cd_cvm)
    print(history.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
