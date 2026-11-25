"""Offline tests of the source-error rules of the statement builder and of stale checkpoints."""
import pandas as pd
import pytest
import torch

from src.data_ingestion import fetch_cvm
from src.model import dataset


def test_disagreeing_duplicates_give_an_empty_value():
    frame = pd.DataFrame({"cd_cvm": [1, 1, 2, 2], "quarter": ["q"] * 4, "column": ["x"] * 4, "value": [0.0, 646.0, 5.0, 5.0]})
    single = fetch_cvm._single(frame, ["cd_cvm", "quarter", "column"]).set_index("cd_cvm")["value"]
    assert pd.isna(single[1])
    assert single[2] == 5.0


def row(quarter: str, version: int, order: str, end: str, scale: str, value: str) -> dict:
    return {
        "CNPJ_CIA": "1", "DT_REFER": quarter, "VERSAO": str(version), "DENOM_CIA": "X", "CD_CVM": "20125",
        "MOEDA": "REAL", "ESCALA_MOEDA": scale, "ORDEM_EXERC": order, "DT_INI_EXERC": None, "DT_FIM_EXERC": end,
        "CD_CONTA": "1", "DS_CONTA": "Ativo Total", "VL_CONTA": value, "doc": "itr",
    }


def test_a_unit_label_is_fixed_only_with_proof_from_the_comparative():
    rows = [
        row("2020-12-31", 1, "ÚLTIMO", "2020-12-31", "MIL", "1921049"),
        row("2021-06-30", 1, "ÚLTIMO", "2021-06-30", "UNIDADE", "2020126"),
        row("2021-06-30", 1, "PENÚLTIMO", "2020-12-31", "UNIDADE", "1921049"),  # 1/1000 of the filed value
        row("2021-09-30", 1, "ÚLTIMO", "2021-09-30", "UNIDADE", "1900406000"),
        row("2021-09-30", 1, "PENÚLTIMO", "2020-12-31", "UNIDADE", "1921049000"),  # consistent: no fix
    ]
    fixes = fetch_cvm.unit_fixes([pd.DataFrame(rows)])
    assert fixes == {(20125, pd.Timestamp("2021-06-30"), "itr", 1)}
    cleaned = fetch_cvm._clean(pd.DataFrame(rows[1:2]), fetch_cvm.BALANCE, fixes)
    assert cleaned["value"].iloc[0] == 2020126000.0


def test_a_stale_checkpoint_raises():
    meta = {"fingerprint": "new", "tickers": ["A"]}
    with pytest.raises(ValueError, match="ERR_STALE_CHECKPOINT"):
        dataset.check_provenance({"fingerprint": "old", "arch": dataset.ARCH, "tickers": ["A"], "state_dict": {}}, meta, "x.pt")
    with pytest.raises(ValueError, match="architecture"):
        dataset.check_provenance({"fingerprint": "new", "arch": 1, "tickers": ["A"], "state_dict": {}}, meta, "x.pt")
    dataset.check_provenance({"fingerprint": "new", "arch": dataset.ARCH, "tickers": ["A"], "state_dict": {}}, meta, "x.pt")


def test_a_unit_fix_can_prove_another_unit_fix():
    rows = [
        row("2019-12-31", 1, "ÚLTIMO", "2019-12-31", "MIL", "1800000"),
        row("2020-12-31", 1, "ÚLTIMO", "2020-12-31", "UNIDADE", "1921049"),
        row("2020-12-31", 1, "PENÚLTIMO", "2019-12-31", "UNIDADE", "1800000"),  # proves 2020-12-31
        row("2021-06-30", 1, "ÚLTIMO", "2021-06-30", "UNIDADE", "2020126"),
        row("2021-06-30", 1, "PENÚLTIMO", "2020-12-31", "UNIDADE", "1921049"),  # proven only after the first fix
    ]
    fixes = fetch_cvm.unit_fixes([pd.DataFrame(rows)])
    assert {quarter for _, quarter, _, _ in fixes} == {pd.Timestamp("2020-12-31"), pd.Timestamp("2021-06-30")}


def test_a_later_comparative_proves_a_unit_fix_that_the_own_comparative_cannot():
    rows = [
        row("2018-12-31", 1, "ÚLTIMO", "2018-12-31", "UNIDADE", "1700000"),  # in thousands, unproven
        row("2019-12-31", 1, "ÚLTIMO", "2019-12-31", "UNIDADE", "1800000"),
        row("2019-12-31", 1, "PENÚLTIMO", "2018-12-31", "UNIDADE", "1700000"),  # same error: ratio 1
        row("2020-03-31", 1, "ÚLTIMO", "2020-03-31", "MIL", "1850000"),
        row("2020-03-31", 1, "PENÚLTIMO", "2019-12-31", "MIL", "1800000"),  # proves 2019-12-31
    ]
    fixes = fetch_cvm.unit_fixes([pd.DataFrame(rows)])
    assert (20125, pd.Timestamp("2019-12-31"), "itr", 1) in fixes
    assert (20125, pd.Timestamp("2018-12-31"), "itr", 1) in fixes  # proven after the first fix


def flow(quarter: str, order: str, start: str, end: str, value: str, doc: str = "itr") -> dict:
    return {
        "CNPJ_CIA": "1", "DT_REFER": quarter, "VERSAO": "1", "DENOM_CIA": "X", "CD_CVM": "20125",
        "MOEDA": "REAL", "ESCALA_MOEDA": "MIL", "ORDEM_EXERC": order, "DT_INI_EXERC": start, "DT_FIM_EXERC": end,
        "CD_CONTA": "3.06.02", "DS_CONTA": "Despesas Financeiras", "VL_CONTA": value, "doc": doc,
    }


def test_a_placeholder_comparative_leaves_the_trailing_expense_empty():
    dfp = [flow("2013-12-31", "ÚLTIMO", "2013-01-01", "2013-12-31", "-400", "dfp")]
    itr = [
        flow("2014-09-30", "ÚLTIMO", "2014-01-01", "2014-09-30", "-300"),
        flow("2014-09-30", "ÚLTIMO", "2014-07-01", "2014-09-30", "-100"),
        flow("2014-09-30", "PENÚLTIMO", "2013-01-01", "2013-09-30", "0"),  # placeholder
        flow("2014-09-30", "PENÚLTIMO", "2013-07-01", "2013-09-30", "-90"),
    ]
    wide = fetch_cvm.build_flows([pd.DataFrame(dfp)], [pd.DataFrame(itr)], set())
    assert pd.isna(wide.loc[(20125, pd.Timestamp("2014-09-30")), "finexp_ttm"])


def test_the_foreign_currency_split_keeps_only_its_own_label():
    def account(code: str, label: str, value: str) -> dict:
        return {**row("2024-12-31", 1, "ÚLTIMO", "2024-12-31", "MIL", value), "CD_CONTA": code, "DS_CONTA": label}

    frame = pd.DataFrame([
        account("2.01.04.01.02", "Em Moeda Estrangeira", "500"),
        account("2.02.01.01.02", "Debêntures", "900"),  # another label on the same code: not kept
    ])
    cleaned = fetch_cvm._clean(frame, fetch_cvm.BALANCE)
    assert list(cleaned["column"]) == ["fx_loans_short"]
    assert cleaned["value"].iloc[0] == 500_000.0
