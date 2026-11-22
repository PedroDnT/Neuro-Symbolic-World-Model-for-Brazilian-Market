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
