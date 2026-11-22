# Brazilian Macro-Corporate Contagion Engine

A stress-test engine for 100 non-financial companies listed on B3. You give a macro shock (SELIC, IPCA, USD/BRL). The engine returns a ranked table of companies by the equity loss that the shock causes, and an audit trail for each company that says where each number comes from.

Each predicted balance sheet satisfies Assets = Current Liabilities + Non-Current Liabilities + Equity (residual at most 1e-5 in scaled units).

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Data (public sources; about 15 minutes, about 600 MB of downloads)
.venv/bin/python -m src.data_ingestion.fetch_macro     # SELIC, IPCA, PTAX by quarter
.venv/bin/python -m src.data_ingestion.fetch_cvm       # quarterly statements, 2010-Q4 to 2024-Q4
.venv/bin/python -m src.data_ingestion.universe        # top 100 non-financial companies by 2024 volume
.venv/bin/python -m src.data_ingestion.fetch_fre       # ownership and related-party links

# 2. Extraction of supplier links and debt indexers from filing text (needs OPENAI_API_KEY; costs money)
.venv/bin/python -m src.extraction.pipeline --all       # spending cap: EXTRACTION_CAP_USD, default 25
.venv/bin/python -m src.extraction.aggregate

# 3. Graph snapshots, training and test score
.venv/bin/python -m src.graph_builder.build_pyg
.venv/bin/python main.py                                # writes results/metrics.json
.venv/bin/python -m src.knowledge.graph                 # writes results/knowledge_graph.json

# 4. Stress test
.venv/bin/python -m src.simulator.stress_test --selic 3.0 --ptax-pct 15
```

The demo notebook is `notebooks/stress_test_demo.ipynb`. The tests run offline: `.venv/bin/pytest -q`.

## How it works

The engine has 2 parts that run side by side.

**Hybrid engine (main ranking).**

1. Base path: the "mean change" forecast of each company plus a small seasonal correction (a pooled ridge regression on last quarter's surprise and the same quarter last year). `src/model/ridge.py`.
2. Shock channel: stated accounting rules on the debt of each company (`src/simulator/transmission.py`):
   - SELIC: floating-rate debt pays more interest, in cash.
   - IPCA: IPCA-linked debt accrues more.
   - PTAX: foreign-currency debt that is not swapped to CDI is revalued.
   - Each loss is after a 34 % tax rate. Each rule keeps the balance sheet balanced.
3. The debt shares come from the notes to the 2024 statements, read by an LLM with a verbatim quote for each line (`src/extraction/`). A quote must be in the source text. An amount must be a number of its own quote, or the sum of 2 of them (current plus non-current). A fixed-rate or SOFR line counts as foreign currency when its quote names a currency code (USD, EUR and similar). Without a usable debt profile, the floating share is a regression estimate with the label "estimate", and the IPCA and foreign-currency channels give 0. Without an estimate either, the effect stays empty and the company gets no rank.
4. Contagion flag: a customer whose supplier (high dependence, from the 2021 reference form) breaches interest coverage under the shock.

**Neural engine (comparison).** The graph recurrent model of the original work order (`src/model/`): GATv2 attention over sector edges, a GRU state, a Gaussian latent, and an accounting projection layer. It sees the macro inputs, so the shock moves its forecast.

**Rules and audit trail.** `src/knowledge/rules.py` holds negative equity, leverage, interest coverage (also under the shock) and equity loss. `src/knowledge/audit.py` prints the trail of 1 company. `src/knowledge/graph.py` builds a Semantica knowledge graph (`semantica-explorer --graph results/knowledge_graph.json`).

## Results

### Test score

1-step forecast of the balance sheet, 2022-Q1 to 2024-Q4 (12 quarters, 100 companies), never used for fitting or for a choice. Each error is divided by the usual quarterly change of the company and account. Lower is better. Source: `results/metrics.json`.

| Engine | MSE | MAE | Median |
|---|---|---|---|
| No change | 7.364 | 1.269 | 0.675 |
| Mean change (naive) | 7.048 | 1.215 | 0.654 |
| **Hybrid: seasonal ridge** | **6.961** | **1.201** | **0.635** |
| Neural: sector edges | 6.976 | 1.208 | 0.648 |
| Neural: typed edges | 6.973 | 1.213 | 0.649 |
| Neural: no edges | 7.087 | 1.221 | 0.650 |

- The ridge is 1.2 % better than mean change on MSE (company bootstrap 95 % interval of the gain: +0.04 to +0.13), and better in 10 of 12 quarters.
- The neural variants with edges are about 1 % better than mean change on MSE; their MAE gain is not significant. The 3 seeds of each variant agree within 0.013.
- All gains are small. 63 % of the squared error of mean change sits in 1 % of the cells: mergers and accounting events (Azzas, Americanas, Localiza) that no model here predicts.
- The neural shock response is near zero (0.001 scaled units for a 1-standard-deviation SELIC move). The hybrid engine carries the shock ranking.

### Stress test example

`python -m src.simulator.stress_test --selic 3.0 --ptax-pct 15` writes `results/stress_selic+3_ipca+0_ptax+15.csv`.

- 4 companies have filed equity that is not positive (Braskem, Gol, Oi, Recrusul). They rank first, with a flag.
- The largest hybrid losses come from foreign-currency debt not swapped to CDI: Klabin -39 % of filed equity (84 % of its debt), Suzano -25 % (80 %), JBS -21 % (90 %), Petrobras -8 %. The transmission ignores export revenue in USD, so it overstates the loss of an exporter.
- 17 companies have no hybrid rank: they have debt, but no usable debt profile and no regression estimate (for example Minerva, Magazine Luiza, PRIO). Their effect is unknown, not 0.
- 36 companies have interest coverage below 1.5 under the shock. 6 customers get a contagion flag from a supplier with high dependence (for example Ultrapar from Braskem and Petrobras).
- Rank correlation between the 2 engines: 0.45, over the 79 companies with a result in both.

### Rules on known failures

The rules run on the filed history of each company (`python -m src.knowledge.rules --cd-cvm <code>`).

| Company | Event | First signal before the event |
|---|---|---|
| Light | judicial recovery, 2023-05 | interest coverage below 1.5 from 2021-Q3 |
| Oi | judicial recovery, 2016-06 | leverage above 0.80 from 2014-Q4 |
| Gol | Chapter 11, 2024-01 | negative equity from 2014-Q4 |

### Extraction

- 202 runs (100 companies x 2 tasks, plus 1 company that left the universe): 194 ok, 8 with no matching section. Cost: US$ 20.72 (cap US$ 25).
- Debt profiles for 75 of 100 companies. The others use a regression estimate (7) or nothing (18), with the label in each output.
- An amount must match a number of its quote exactly, or the sum of 2 of them. An earlier tolerance of 0.5 accepted some wrong-scale amounts by chance; the exact check rejects them.
- 39 supplier links between universe companies. Each quote is also in the version of the form received by 2021-12-31, so the typed neural variant may use them.
- Known error: the model read 1 dependence level wrong (Grendene's quote says it does not depend on Braskem, but the fact says "high").

## Limits

- The hybrid shock effect is an accounting pass-through with stated assumptions: no change in revenue or operating costs, debt held constant, 34 % tax. It is not learned from data.
- Without a usable debt profile, the IPCA and foreign-currency channels give 0. This understates the loss of a company with foreign-currency debt that the extraction did not read.
- The neural shock response is an extrapolation from 45 quarters of macro data.
- The universe holds survivors: companies with all 57 quarters. Statements are the latest filed version, not point-in-time.
- The sector is the current sector of the CVM register.
- The extraction covers only what a company discloses. A missing link is not proof that no link exists.
- An attention weight is a model weight, not proof of a causal link.

## Data sources

All public: CVM open data (DFP, ITR, FRE, FCA and the company register), the BCB time series system, and the B3 quotation history.
