"""The quarterly flows and the comprehensive-income label rules. No download."""
import pandas as pd
import pytest

from src.data_ingestion.filed_panel import (
    classify, combine, dre_items, oci_items, quarter_flows, restated, three_months, year_to_date,
)


def row(quarter, doc, start, end, code, value, label="Resultado Financeiro", cd_cvm=1):
    return {"cd_cvm": cd_cvm, "quarter": pd.Timestamp(quarter), "doc": doc, "CD_CONTA": code, "DS_CONTA": label,
            "DT_INI_EXERC": start, "DT_FIM_EXERC": end, "value": value}


def dre_flows(rows):
    return quarter_flows(dre_items(year_to_date(pd.DataFrame(rows)))).set_index("quarter")


def test_each_quarter_is_a_difference_of_year_to_date_values():
    rows = [
        row("2024-03-31", "itr", "2024-01-01", "2024-03-31", "3.06", -20.0),
        row("2024-06-30", "itr", "2024-04-01", "2024-06-30", "3.06", -25.0),
        row("2024-06-30", "itr", "2024-01-01", "2024-06-30", "3.06", -50.0),  # Q1 restated to -25
        row("2024-09-30", "itr", "2024-07-01", "2024-09-30", "3.06", -30.0),
        row("2024-09-30", "itr", "2024-01-01", "2024-09-30", "3.06", -100.0),
        row("2024-12-31", "dfp", "2024-01-01", "2024-12-31", "3.06", -160.0),
    ]
    flows = dre_flows(rows)
    assert list(flows["value"].round(6)) == [-20.0, -30.0, -50.0, -60.0]
    assert list(flows["fiscal_q"]) == [1, 2, 3, 4]


def test_a_quarter_without_the_previous_year_to_date_stays_empty():
    assert dre_flows([row("2024-12-31", "dfp", "2024-01-01", "2024-12-31", "3.06", -160.0)]).empty


def test_a_fiscal_year_from_april_to_march():
    rows = [
        row("2023-06-30", "itr", "2023-04-01", "2023-06-30", "3.06", -10.0),
        row("2023-09-30", "itr", "2023-04-01", "2023-09-30", "3.06", -25.0),
        row("2023-12-31", "itr", "2023-04-01", "2023-12-31", "3.06", -45.0),
        row("2024-03-31", "dfp", "2023-04-01", "2024-03-31", "3.06", -70.0),
    ]
    flows = dre_flows(rows)
    assert list(flows["fiscal_q"]) == [1, 2, 3, 4]
    assert list(flows["value"].round(6)) == [-10.0, -15.0, -20.0, -25.0]


def test_a_dre_line_with_an_unexpected_label_is_not_used():
    rows = [row("2024-03-31", "itr", "2024-01-01", "2024-03-31", "3.06", -5.0, label="Imposto de Renda e Contribuição Social sobre o Lucro")]
    assert dre_flows(rows).empty


def oci(quarter, doc, end, code, label, value):
    return row(quarter, doc, "2020-01-01", end, code, value, label=label)


def test_renumbered_lines_are_differenced_by_class_not_by_code():
    # Q1: 4.02.01 is the hedge line. Q2: the company renumbers, and 4.02.01 is now translation.
    rows = [
        oci("2020-03-31", "itr", "2020-03-31", "4.02", "Outros resultados abrangentes", -100.0),
        oci("2020-03-31", "itr", "2020-03-31", "4.02.01", "Hedge de fluxo de caixa", -100.0),
        oci("2020-06-30", "itr", "2020-06-30", "4.02", "Outros resultados abrangentes", -150.0),
        oci("2020-06-30", "itr", "2020-06-30", "4.02.01", "Ajustes de conversão", 30.0),
        oci("2020-06-30", "itr", "2020-06-30", "4.02.02", "Hedge de fluxo de caixa", -180.0),
    ]
    flows = quarter_flows(oci_items(year_to_date(pd.DataFrame(rows))))
    q2 = flows[flows["quarter"] == pd.Timestamp("2020-06-30")].set_index("item")["value"]
    assert q2["oci_hedge_q"] == pytest.approx(-80.0)
    assert q2["oci_translation_q"] == pytest.approx(30.0)  # the class was absent in Q1, so 0 there
    assert q2["oci_q"] == pytest.approx(-50.0)


def test_a_line_and_its_counted_parent_are_not_added_twice():
    rows = [
        oci("2024-12-31", "dfp", "2020-12-31", "4.02", "Outros resultados abrangentes", 97.0),
        oci("2024-12-31", "dfp", "2020-12-31", "4.02.01", "Hedge accounting", 10.0),
        oci("2024-12-31", "dfp", "2020-12-31", "4.02.01.01", "IR sobre hedge", -3.0),
        oci("2024-12-31", "dfp", "2020-12-31", "4.02.02", "Valores que serão reclassificados", 50.0),
        oci("2024-12-31", "dfp", "2020-12-31", "4.02.02.01", "Ajustes de conversão", 40.0),
        oci("2024-12-31", "dfp", "2020-12-31", "4.02.03", "Efeito fiscal", -5.0),
    ]
    frame = pd.DataFrame(rows).assign(quarter=pd.Timestamp("2020-12-31"))
    items = oci_items(year_to_date(frame)).set_index("item")["value"]
    assert items["oci_hedge_q"] == pytest.approx(10.0)  # the parent only
    assert items["oci_translation_q"] == pytest.approx(40.0)  # its "other" parent is not counted
    assert items["oci_tax_q"] == pytest.approx(-5.0)


def test_a_filing_with_the_total_only_gives_unknown_classes():
    rows = [
        oci("2020-03-31", "itr", "2020-03-31", "4.02", "Outros resultados abrangentes", -100.0),
        oci("2020-03-31", "itr", "2020-03-31", "4.02.01", "Hedge de fluxo de caixa", -100.0),
        oci("2020-06-30", "itr", "2020-06-30", "4.02", "Outros resultados abrangentes", -150.0),  # total only
        oci("2020-03-31", "itr", "2020-03-31", "4.02", "Outros resultados abrangentes", 0.0) | {"cd_cvm": 2},
    ]
    flows = quarter_flows(oci_items(year_to_date(pd.DataFrame(rows))))
    first = flows[flows["cd_cvm"] == 1]
    q1 = first[first["quarter"] == pd.Timestamp("2020-03-31")].set_index("item")["value"]
    q2 = first[first["quarter"] == pd.Timestamp("2020-06-30")].set_index("item")["value"]
    assert q1["oci_hedge_q"] == pytest.approx(-100.0) and q1["oci_translation_q"] == 0.0
    assert q2["oci_q"] == pytest.approx(-50.0)
    assert pd.isna(q2["oci_hedge_q"]) and pd.isna(q2["oci_translation_q"])  # unknown: not 0, not +100
    other = flows[flows["cd_cvm"] == 2].set_index("item")["value"]
    assert other["oci_hedge_q"] == 0.0  # a total of 0 and no lines: no comprehensive income at all


def test_the_filed_3_month_value_is_used_in_fiscal_q2_and_q3():
    rows = pd.DataFrame([
        row("2024-03-31", "itr", "2024-01-01", "2024-03-31", "3.06", -20000.0),
        row("2024-06-30", "itr", "2024-04-01", "2024-06-30", "3.06", -25000.0),
        row("2024-06-30", "itr", "2024-01-01", "2024-06-30", "3.06", -50000.0),  # Q1 restated
    ])
    ytd = year_to_date(rows)
    derived = quarter_flows(dre_items(ytd))
    months = three_months(rows).merge(ytd[["cd_cvm", "quarter", "fy_start", "fiscal_q"]].drop_duplicates(), on=["cd_cvm", "quarter"])
    flows, flag = combine(derived, dre_items(months))
    q2 = flows[flows["quarter"] == pd.Timestamp("2024-06-30")]["value"]
    assert list(q2) == [-25000.0]
    assert flag[(1, pd.Timestamp("2024-06-30"))]


def test_a_restated_quarter_is_flagged():
    rows = pd.DataFrame([
        row("2024-03-31", "itr", "2024-01-01", "2024-03-31", "3.06", -20000.0),
        row("2024-06-30", "itr", "2024-04-01", "2024-06-30", "3.06", -25000.0),  # filed 3 months
        row("2024-06-30", "itr", "2024-01-01", "2024-06-30", "3.06", -50000.0),  # so Q1 became -25000
        row("2024-03-31", "itr", "2024-01-01", "2024-03-31", "3.06", -20000.0, cd_cvm=2),
        row("2024-06-30", "itr", "2024-04-01", "2024-06-30", "3.06", -30000.0, cd_cvm=2),
        row("2024-06-30", "itr", "2024-01-01", "2024-06-30", "3.06", -50000.0, cd_cvm=2),
    ])
    flows = quarter_flows(dre_items(year_to_date(rows)))
    filed = three_months(rows).assign(item="fin_result_q")[["cd_cvm", "quarter", "item", "value"]]
    flag = restated(flows, filed)
    assert flag[(1, pd.Timestamp("2024-06-30"))]
    assert not flag[(2, pd.Timestamp("2024-06-30"))]


@pytest.mark.parametrize("label, kind", [
    ("Hedge de fluxo de caixa", "hedge"),
    ("Hedge de fluxo de caixa, líquido de impostos", "hedge"),
    ("Hedge de fluxo de caixa, líquido de IR e CS", "hedge"),
    ("Resultado de hedge (sem efeito tributário)", "hedge"),
    ("Imposto de renda e contribuição social sobre hedge de fluxo de caixa", "tax"),
    ("Ajuste ao valor justo de instrumentos financeiros derivativos", "hedge"),
    ("Variação cambial de contratos de financiamentos ACC/PPE", "hedge"),
    ("Ajustes acumulados de conversão", "translation"),
    ("Variação cambial de investimentos no exterior", "translation"),
    ("Diferenças cambiais de conversão de operações no exterior", "translation"),
    ("Ajuste na Conversão de Controlada no Exterior - Líq. dos Efeitos Tributários", "translation"),
    ("Ganhos var. camb. investimentos exterior", "translation"),
    ("Variação combial de investimento no exterior", "translation"),
    ("Diferenças de câmbio originadas no exercício", "translation"),
    ("Reclassificação do ajuste de conversão acumulado para o resultado", "other"),
    ("Coberturas de fluxo de caixa", "hedge"),
    ("Operações de proteção a risco de variação cambial", "hedge"),
    ("Swap fluxo de caixa", "hedge"),
    ("Efeito de hiperinflação", "other"),
    ("Ajustes de conversão de hiperinflação", "other"),
    ("Ganho (perda) atuarial", "other"),
    ("Ganho atuarial líquido dos efeitos tributários", "other"),
    ("Efeito fiscal", "tax"),
    ("Imposto de renda", "tax"),
    ("Valores que serão reclassificados para o resultado", "other"),
])
def test_label_classes(label, kind):
    assert classify(label) == kind
