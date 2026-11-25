"""The quarterly flows and the comprehensive-income label rules. No download."""
import pandas as pd
import pytest

from src.data_ingestion.filed_panel import classify, oci_parts, quarter_flows


def row(quarter, doc, start, end, code, value, label="Resultado Financeiro"):
    return {"cd_cvm": 1, "quarter": pd.Timestamp(quarter), "doc": doc, "CD_CONTA": code, "DS_CONTA": label,
            "DT_INI_EXERC": start, "DT_FIM_EXERC": end, "value": value}


def test_each_quarter_is_a_difference_of_year_to_date_values():
    rows = pd.DataFrame([
        row("2024-03-31", "itr", "2024-01-01", "2024-03-31", "3.06", -20.0),
        row("2024-06-30", "itr", "2024-04-01", "2024-06-30", "3.06", -25.0),
        row("2024-06-30", "itr", "2024-01-01", "2024-06-30", "3.06", -50.0),  # Q1 restated to -25
        row("2024-09-30", "itr", "2024-07-01", "2024-09-30", "3.06", -30.0),
        row("2024-09-30", "itr", "2024-01-01", "2024-09-30", "3.06", -100.0),
        row("2024-12-31", "dfp", "2024-01-01", "2024-12-31", "3.06", -160.0),
    ])
    flows = quarter_flows(rows).set_index("quarter")["value"]
    assert list(flows.round(6)) == [-20.0, -30.0, -50.0, -60.0]
    assert flows.sum() == pytest.approx(-160.0)  # the quarters add up to the annual value


def test_a_quarter_without_the_previous_year_to_date_stays_empty():
    rows = pd.DataFrame([row("2024-12-31", "dfp", "2024-01-01", "2024-12-31", "3.06", -160.0)])
    assert quarter_flows(rows).empty


@pytest.mark.parametrize("label, kind", [
    ("Hedge de fluxo de caixa", "hedge"),
    ("Imposto de renda e contribuição social sobre hedge de fluxo de caixa", "hedge"),
    ("Ajuste ao valor justo de instrumentos financeiros derivativos", "hedge"),
    ("Ajustes acumulados de conversão", "translation"),
    ("Variação cambial de investimentos no exterior", "translation"),
    ("Diferenças cambiais de conversão de operações no exterior", "translation"),
    ("Ganhos var. camb. investimentos exterior", "translation"),
    ("Coberturas de fluxo de caixa", "hedge"),
    ("Operações de proteção a risco de variação cambial", "hedge"),
    ("Swap fluxo de caixa", "hedge"),
    ("Ganho (perda) atuarial", "other"),
    ("Efeito fiscal", "other"),
    ("Valores que serão reclassificados para o resultado", "other"),
])
def test_label_classes(label, kind):
    assert classify(label) == kind


def test_a_line_and_its_counted_parent_are_not_added_twice():
    flows = pd.DataFrame([
        {"cd_cvm": 1, "quarter": pd.Timestamp("2024-12-31"), "CD_CONTA": "4.02.01", "DS_CONTA": "Hedge accounting", "value": 10.0},
        {"cd_cvm": 1, "quarter": pd.Timestamp("2024-12-31"), "CD_CONTA": "4.02.01.01", "DS_CONTA": "IR sobre hedge", "value": -3.0},
        {"cd_cvm": 1, "quarter": pd.Timestamp("2024-12-31"), "CD_CONTA": "4.02.02", "DS_CONTA": "Valores que serão reclassificados", "value": 50.0},
        {"cd_cvm": 1, "quarter": pd.Timestamp("2024-12-31"), "CD_CONTA": "4.02.02.01", "DS_CONTA": "Ajustes de conversão", "value": 40.0},
    ])
    parts = oci_parts(flows).iloc[0]
    assert parts["oci_hedge_q"] == pytest.approx(10.0)  # the parent only
    assert parts["oci_translation_q"] == pytest.approx(40.0)  # its "other" parent is not counted
