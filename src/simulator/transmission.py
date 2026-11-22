"""Accounting transmission of a macro shock to the balance sheet of each company.

The hybrid engine does not learn the shock response. It applies these stated rules to the
debt of each company, on top of the base path of the seasonal ridge.

Inputs for each company (at the last filed quarter):
    gross_debt      loans, financing and debentures (accounts 2.01.04 + 2.02.01), in reais
    short_share     share of that debt that is current (2.01.04 / gross debt)
    floating_share  share at a floating domestic rate (CDI or SELIC, or swapped to CDI)
    ipca_share      share indexed to IPCA
    fx_share        share in a foreign currency and not swapped to CDI

The shares come from the extraction of the debt note (basis "extracted"). A company with
no usable debt profile gets the regression estimate for floating_share (basis "estimate")
and no IPCA or foreign-currency share (basis "none"): those channels then give 0.

Rules for each quarter of the horizon (TAX = 34 %, the statutory rate of income tax plus
social contribution):
    SELIC  extra interest = gross_debt * floating_share * SELIC_delta / 100 / 4, paid in cash.
           assets -= interest, current liabilities -= TAX * interest (lower tax payable),
           equity -= (1 - TAX) * interest.
    IPCA   extra accrual = gross_debt * ipca_share * IPCA_delta / 100, added to the debt.
    PTAX   revaluation = gross_debt * fx_share * PTAX_delta_pct / 100, 1 time in quarter 1.
           For IPCA and PTAX: debt rises by the amount (current and non-current by
           short_share), a deferred tax asset of TAX * amount is booked, and
           equity -= (1 - TAX) * amount.
Each rule moves assets, liabilities and equity so that Assets = Liabilities + Equity holds.
Simplifications: the debt does not change with the base path, the shock does not change
revenue or operating costs, and interest on the extra accrual is ignored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TAX = 0.34
ACCOUNTS = ["total_assets", "current_liabilities", "noncurrent_liabilities", "equity"]


@dataclass(frozen=True)
class Exposure:
    gross_debt: float
    short_share: float
    floating_share: float
    floating_basis: str
    ipca_share: float
    fx_share: float
    profile_basis: str  # "extracted" or "none"


@dataclass(frozen=True)
class QuarterEffect:
    quarter: int
    interest: float  # pre-tax extra interest from SELIC
    ipca_accrual: float  # pre-tax extra accrual from IPCA
    fx_revaluation: float  # pre-tax revaluation from PTAX
    delta: np.ndarray  # change of [assets, current liabilities, non-current liabilities, equity]


def quarter_effects(exposure: Exposure, selic_pp: float, ipca_pp: float, ptax_pct: float, horizon: int) -> list[QuarterEffect]:
    effects = []
    for quarter in range(1, horizon + 1):
        interest = exposure.gross_debt * exposure.floating_share * selic_pp / 100 / 4
        accrual = exposure.gross_debt * exposure.ipca_share * ipca_pp / 100
        revaluation = exposure.gross_debt * exposure.fx_share * ptax_pct / 100 if quarter == 1 else 0.0
        debt_rise = accrual + revaluation
        delta = np.array(
            [
                -interest + TAX * debt_rise,
                -TAX * interest + exposure.short_share * debt_rise,
                (1 - exposure.short_share) * debt_rise,
                -(1 - TAX) * (interest + debt_rise),
            ]
        )
        effects.append(QuarterEffect(quarter, interest, accrual, revaluation, delta))
    return effects


def cumulative_delta(effects: list[QuarterEffect]) -> np.ndarray:
    """Cumulative change after each quarter, shape (horizon, 4), in reais."""
    return np.cumsum(np.stack([effect.delta for effect in effects]), axis=0)


def exposures(statements_last: pd.DataFrame, profile: pd.DataFrame, floating_estimate: pd.DataFrame) -> dict[int, Exposure]:
    """Exposure of each company. `statements_last` is indexed by cd_cvm (last filed quarter).

    Some companies file their loans outside accounts 2.01.04 and 2.02.01 (for example under
    "Outras Obrigações"), so the filed debt is 0 while the notes show debt. Their gross debt
    is unknown (NaN), and their shock effect stays empty, never 0.
    """
    usable = profile.dropna(subset=["floating_share"]).set_index("cd_cvm") if len(profile) else pd.DataFrame()
    found_debt = set(profile.loc[profile.get("amount_sum", pd.Series(dtype=float)).fillna(0) > 0, "cd_cvm"]) if len(profile) else set()
    estimate = floating_estimate.set_index("cd_cvm")
    result = {}
    for cd_cvm, row in statements_last.iterrows():
        short = float(np.nan_to_num(row["debt_short"]))
        long = float(np.nan_to_num(row["debt_long"]))
        gross = short + long
        short_share = short / gross if gross > 0 else 0.0
        if gross == 0 and cd_cvm in found_debt:
            gross = float("nan")
        if cd_cvm in usable.index:
            shares = usable.loc[cd_cvm]
            result[cd_cvm] = Exposure(
                gross, short_share, float(shares["floating_share"]), "extracted",
                float(np.nan_to_num(shares["ipca_share"])), float(np.nan_to_num(shares["fx_share"])), "extracted",
            )
        else:
            floating = estimate.loc[cd_cvm, "floating_share"] if cd_cvm in estimate.index else np.nan
            known = not pd.isna(floating)
            basis = str(estimate.loc[cd_cvm, "basis"]) if known else "none"
            result[cd_cvm] = Exposure(gross, short_share, float(floating) if known else 0.0, basis, 0.0, 0.0, "none")
    return result
