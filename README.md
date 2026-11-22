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
3. The debt shares come from the notes to the 2024 statements, read by an LLM with a verbatim quote for each line (`src/extraction/`). A quote must be in the source text, and an amount must be a number of its own quote. Without a usable debt profile, a regression estimate is used, with the label "estimate".
4. Contagion flag: a customer whose supplier (high dependence, from the 2021 reference form) breaches interest coverage under the shock.

**Neural engine (comparison).** The graph recurrent model of the original work order (`src/model/`): GATv2 attention over sector edges, a GRU state, a Gaussian latent, and an accounting projection layer. It sees the macro inputs, so the shock moves its forecast.

**Rules and audit trail.** `src/knowledge/rules.py` holds negative equity, leverage, interest coverage (also under the shock) and equity loss. `src/knowledge/audit.py` prints the trail of 1 company. `src/knowledge/graph.py` builds a Semantica knowledge graph (`semantica-explorer --graph results/knowledge_graph.json`).

## Results

The test score is being computed. This section follows in the next commit.

## Limits

- The hybrid shock effect is an accounting pass-through with stated assumptions: no change in revenue or operating costs, debt held constant, 34 % tax. It is not learned from data.
- The neural shock response is an extrapolation from 45 quarters of macro data.
- The universe holds survivors: companies with all 57 quarters. Statements are the latest filed version, not point-in-time.
- The sector is the current sector of the CVM register.
- The extraction covers only what a company discloses. A missing link is not proof that no link exists.
- An attention weight is a model weight, not proof of a causal link.

## Data sources

All public: CVM open data (DFP, ITR, FRE, FCA and the company register), the BCB time series system, and the B3 quotation history.
