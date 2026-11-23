"""Accounting transmission of a macro shock to the balance sheet of each company.

The hybrid engine does not learn the shock response. It applies these stated rules to the
debt of each company, on top of the base path of the seasonal ridge.

Inputs for each company (at the last filed quarter):
    gross_debt      loans, financing and debentures (accounts 2.01.04 + 2.02.01), in reais
    short_share     share of that debt that is current (2.01.04 / gross debt)
    floating_debt   debt at a floating domestic rate (CDI or SELIC, or swapped to CDI), in reais
    ipca_debt       debt indexed to IPCA, in reais
    fx_debt         debt in a foreign currency and not swapped to CDI, in reais

The amounts come from the lines of the debt note (basis "extracted"), not from a share of
gross_debt: the filed accounts can hold leases, and a note can list its lines before swap
lines. Leases outside the note get no shock. A company with no usable debt profile gets
gross_debt * the regression estimate of its floating share (basis "estimate") and no IPCA or
foreign-currency debt (basis "none"): those channels then give 0. When no estimate exists
either, the floating debt of a company with debt is unknown (NaN), so its SELIC effect stays
empty, never 0. A channel with a zero shock gives exactly 0.

Rules for each quarter of the horizon (TAX = 34 %, the statutory rate of income tax plus
social contribution):
    SELIC  extra interest = floating_debt * SELIC_delta / 100 / 4, paid in cash.
           assets -= interest, current liabilities -= TAX * interest (lower tax payable),
           equity -= (1 - TAX) * interest.
    IPCA   extra accrual = ipca_debt * IPCA_delta / 100, added to the debt.
    PTAX   revaluation = fx_debt * PTAX_delta_pct / 100, 1 time in quarter 1.
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
    floating_debt: float  # in reais, what the SELIC rule uses
    ipca_debt: float
    fx_debt: float
    amount_source: str  # where the amounts come from, for the audit trail
    # Cross-check against the filed split "Em Moeda Estrangeira" (see aggregate.fx_check).
    # Shown only; it does not change the shock.
    fx_lines: float = float("nan")  # foreign-currency lines of the note, swapped or not
    filed_fx: float = float("nan")
    fx_check: str = "no_profile"


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
        interest = exposure.floating_debt * selic_pp / 100 / 4 if selic_pp else 0.0
        accrual = exposure.ipca_debt * ipca_pp / 100 if ipca_pp else 0.0
        revaluation = exposure.fx_debt * ptax_pct / 100 if quarter == 1 and ptax_pct else 0.0
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
    if len(profile) and "floating_debt_brl" not in profile:
        raise ValueError("data/debt_profile.csv is older than the code: run python -m src.extraction.aggregate")
    usable = profile.dropna(subset=["floating_debt_brl"]).set_index("cd_cvm") if len(profile) else pd.DataFrame()
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
            source = f"debt note, unit x{float(shares['unit_factor']):,.0f} ({shares['unit_basis']})"
            result[cd_cvm] = Exposure(
                gross, short_share, float(shares["floating_share"]), "extracted",
                float(np.nan_to_num(shares["ipca_share"])), float(np.nan_to_num(shares["fx_share"])), "extracted",
                float(shares["floating_debt_brl"]), float(np.nan_to_num(shares["ipca_debt_brl"])),
                float(np.nan_to_num(shares["fx_debt_brl"])), source,
                float(shares.get("fx_lines_brl", float("nan"))), float(shares.get("filed_fx_brl", float("nan"))),
                str(shares.get("fx_check", "no_profile")),
            )
        else:
            floating = estimate.loc[cd_cvm, "floating_share"] if cd_cvm in estimate.index else np.nan
            known = not pd.isna(floating)
            basis = str(estimate.loc[cd_cvm, "basis"]) if known else "none"
            share = float(floating) if known else (0.0 if gross == 0 else float("nan"))
            floating_debt = 0.0 if gross == 0 else gross * share
            result[cd_cvm] = Exposure(
                gross, short_share, share, basis, 0.0, 0.0, "none", floating_debt, 0.0, 0.0, "filed debt x floating share",
            )
    return result
