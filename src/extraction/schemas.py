"""Output schemas of the extraction. Each fact carries a verbatim quote from the filing."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

QUOTE_HELP = "1 sentence or table line copied from the text, exactly as written, at most 300 characters"


class CounterpartyFact(BaseModel):
    counterparty: str = Field(description="Name of the supplier or customer, as written in the text")
    role: Literal["supplier", "customer"] = Field(
        description="supplier: it sells to the company. customer: it buys from the company"
    )
    description: str = Field(description="What it supplies or buys, in a few words")
    dependence: Literal["high", "moderate", "low", "not_stated"] = Field(
        description="How much the text says the company depends on this party"
    )
    quote: str = Field(description=QUOTE_HELP)


class CounterpartyExtraction(BaseModel):
    facts: list[CounterpartyFact]
    states_no_dependence: bool = Field(
        description="True when the text says that the company depends on no single supplier or customer"
    )


class DebtSlice(BaseModel):
    indexer: Literal["CDI", "SELIC", "IPCA", "TJLP_TLP", "USD", "EUR", "OTHER_FX", "FIXED", "OTHER"] = Field(
        description="For debt in a foreign currency use USD, EUR or OTHER_FX, also when its rate is fixed or "
        "floating (for example SOFR). FIXED and OTHER are for debt in reais only"
    )
    amount: Optional[float] = Field(
        description="Balance at the most recent date, as a plain number in the unit of the table. Null when not given"
    )
    hedged_to_cdi: bool = Field(
        description="True when the text says that this line is swapped to CDI or DI, "
        "or gives its cost after hedge as a percentage of the DI rate"
    )
    lender: Optional[str] = Field(description="Bank or creditor name when the text names it")
    instrument: Optional[str] = Field(description="Type of debt, for example debentures or bank loan")
    quote: str = Field(description=QUOTE_HELP)


class DebtExtraction(BaseModel):
    slices: list[DebtSlice]
    unit: Literal["BRL", "BRL_thousand", "BRL_million", "not_stated"]
    total_debt: Optional[float] = Field(
        description="Total loans, financing and debentures at the most recent date, in the same unit"
    )


COUNTERPARTY_INSTRUCTIONS = """You read sections of the annual reference form of a Brazilian listed company.
Extract each supplier and each customer that the text names.

Rules:
1. Use only the text. Do not use knowledge from outside the text.
2. Keep only a party that the text names: a company, a group or a state body. Do not keep a
   generic group such as "fornecedores diversos". Do not keep the company itself or its own
   subsidiaries. Give at most 20 items: first the parties with a stated dependence or share.
3. role is "supplier" when the party sells to the company, and "customer" when it buys from it.
4. dependence is "high" when the text says that the company depends on the party or that it is
   the only source. Use "not_stated" when the text does not say.
5. quote is 1 sentence copied from the text exactly as written, with the same words and the
   same broken words, at most 300 characters. The quote must name the party.
6. Return an empty list when the text names no supplier and no customer."""

DEBT_INSTRUCTIONS = """You read pages of the notes to the 2024 financial statements of a Brazilian listed company.
Extract its loans, financing and debentures by interest-rate indexer.

Rules:
1. Use only the text. Do not use knowledge from outside the text.
2. Give 1 item for each debt line that the text gives. Use the consolidated column and the
   most recent date. Each real of debt must be in 1 item only: do not give a total or a
   subtotal line together with its parts. When 2 tables show the same debt, use 1 table.
3. indexer is the contract indexer: CDI, SELIC, IPCA, TJLP_TLP, USD, EUR, OTHER_FX, FIXED or
   OTHER. hedged_to_cdi is true when the text says that the line is swapped to CDI or DI.
4. amount is the balance as written in the table, in the unit of the table, with no
   thousands separator: write 1.234.567 as 1234567 and 1.234,5 as 1234.5. Use null when
   the text gives no amount for the item.
5. lender is the bank or creditor when the text names it. Use null when it does not.
6. quote is the table line or sentence copied from the text exactly as written, at most
   300 characters.
7. Return an empty list when the pages give no debt breakdown."""
