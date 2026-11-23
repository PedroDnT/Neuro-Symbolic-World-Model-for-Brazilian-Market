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
2. Shock channel: stated accounting rules on the debt of each company (`src/simulator/transmission.py`). Each channel uses the amount in reais of its lines in the debt note, not a share of the filed debt:
   - SELIC: floating-rate debt pays more interest, in cash.
   - IPCA: IPCA-linked debt accrues more.
   - PTAX: foreign-currency debt that is not swapped to CDI is revalued.
   - Each loss is after a 34 % tax rate. Each rule keeps the balance sheet balanced.
3. The debt lines come from the notes to the 2024 statements, read by an LLM with a verbatim quote for each line (`src/extraction/`).
   - A quote must be in the source text. An amount must be a number of its own quote, or the sum of 2 of them (current plus non-current).
   - A fixed-rate or SOFR line counts as foreign currency when its quote names a currency (a code such as USD, or Dólar, Euro, Iene), SOFR, LIBOR, ACC, ACE or "moeda estrangeira".
   - The unit of a table is the stated thousand or million. Else it is the factor (x1, x1,000 or x1,000,000) that puts the line sum closest to the filed debt, accepted only between 0.5 and 2 times the filed debt. The model's own "BRL" label was wrong for 3 tables in thousands.
   - A profile is usable when its line sum is within 20 % of the total that the note states, or of the filed debt, or when the note states no total.
   - Without a usable debt profile, the floating debt is the filed debt times a regression estimate of the floating share (label "estimate"), and the IPCA and foreign-currency channels give 0. Without an estimate either, the SELIC effect stays empty and the company gets no rank.
   - Reviewed corrections (`src/extraction/corrections.json`) fix a misread fact or add a debt line that the extraction missed. Each correction quotes its source verbatim. A fix states how many facts it changes; an added line must pass the same quote and amount check as an extracted line. The run stops when a check fails. The audit trail shows each correction with the label "corrected".
   - FX cross-check: the lines of the note whose type or quote names a foreign currency, whatever their indexer (swapped or not), are compared with the filed split "Em Moeda Estrangeira" (accounts 2.01.04.01.02 + 2.02.01.01.02). A gap is flagged in the stress table and the audit trail. It does not change the shock, because the filed split does not show swaps.
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
| Neural: sector edges | 6.975 | 1.208 | 0.648 |
| Neural: typed edges | 6.973 | 1.213 | 0.649 |
| Neural: no edges | 7.087 | 1.221 | 0.650 |

- The ridge is 1.2 % better than mean change on MSE (company bootstrap 95 % interval of the gain: +0.04 to +0.13), and better in 10 of 12 quarters.
- The neural variants with edges are about 1 % better than mean change on MSE; their MAE gain is not significant. The 3 seeds of each variant differ by at most 0.027.
- All gains are small. 63 % of the squared error of mean change sits in 1 % of the cells: mergers and accounting events (Azzas, Americanas, Localiza) that no model here predicts.
- The neural shock response is near zero (0.001 scaled units for a 1-standard-deviation SELIC move). The hybrid engine carries the shock ranking.

### Stress test example

`python -m src.simulator.stress_test --selic 3.0 --ptax-pct 15` writes `results/stress_selic+3_ipca+0_ptax+15.csv`.

- 4 companies have filed equity that is not positive (Braskem, Gol, Oi, Recrusul). They rank first, with a flag.
- Minerva ranks next (rank 5). Its filed equity is R$ 87 mn against R$ 14.7 bn of unswapped dollar debt in its note, so the shock takes about 22 times its equity (-2,192 % of filed equity).
- The next largest hybrid losses come from foreign-currency debt not swapped to CDI: Klabin -40 % of filed equity (84 % of its note debt), Suzano -26 % (80 %), JBS -21 % (90 %), Mahle Metal Leve -12 % (92 %), Iochpe-Maxion -8 % (57 %), BRF -8 % (60 %) and Embraer -7 % (99 %). The transmission ignores export revenue in USD, so it overstates the loss of an exporter.
- Petrobras falls to rank 21 (-3 %): its note holds R$ 143 bn of loans, and the other R$ 230 bn of its filed debt are leases, which get no shock (see Limits).
- 7 companies have no hybrid rank, because an input of their shock effect is unknown, not 0:
  - Azevedo & Travassos, Magazine Luiza, PRIO and Vulcabras: the text layer of the filing has no debt table.
  - Neoenergia and Tupy: the extracted lines miss part of the debt (the line sum is 76 % and 77 % of the stated total).
  - Allos: its filed debt accounts are 0 and its note states no unit, so its debt in reais is unknown.
- 44 companies have interest coverage below 1.5 under the shock. 5 customers get a contagion flag from a supplier with high dependence: Ultrapar (from Braskem and Petrobras), Braskem and Usiminas (from Petrobras), Unipar (from Braskem) and Cogna (from Suzano).
- Reviewed corrections move 3 companies. CPFL ranks 38th (-1.8 %) and Sabesp 59th (-0.9 %): their foreign-currency loans (CPFL: dollar, euro and yen) are swapped to CDI, so they move from the PTAX channel to the floating channel. Energisa ranks 44th (-1.6 %): its swapped foreign-currency debt (R$ 6.7 bn) is added to its floating debt.
- Rank correlation between the 2 engines: 0.47, over the 89 companies with a result in both.

### Rules on known failures

The rules run on the filed history of each company (`python -m src.knowledge.rules --cd-cvm <code>`).

| Company | Event | First signal before the event |
|---|---|---|
| Light | judicial recovery, 2023-05 | interest coverage below 1.5 from 2021-Q3 |
| Oi | judicial recovery, 2016-06 | leverage above 0.80 from 2014-Q4 |
| Gol | Chapter 11, 2024-01 | negative equity from 2014-Q4 |

### Extraction

- 202 runs (100 companies x 2 tasks, plus 1 company that left the universe): 194 ok, 8 with no matching section. Cost: US$ 21.86 (cap US$ 25).
- Debt profiles for 83 of 100 companies. The others use a regression estimate (8) or nothing (9), with the label in each output. Direcional and Taurus pass the total check through their filed debt: the model read a wrong total for them (a cash-flow table and the 2023 column). For Gol no unit fits its filed debt, so Gol has no usable profile.
- 12 debt runs ran again after the amount check moved into the extraction step, so the model got the reason for each rejected line (cost US$ 1.14). 10 of the 12 now have a usable profile: 8 from the rerun, plus Direcional and Taurus through the filed-debt check.
- An amount must match a number of its quote exactly, or the sum of 2 of them. An earlier tolerance of 0.5 accepted some wrong-scale amounts by chance; the exact check rejects them.
- 39 supplier links between universe companies. Each quote is also in the version of the form received by 2021-12-31, so the typed neural variant may use them.
- 5 reviewed corrections (15 facts). The log is `data/corrections_applied.csv`.
  - CPFL: its 3 foreign-currency loans are swapped to CDI (DFP 2024, page 140).
  - Sabesp: its SOFR line is swapped to CDI (DFP 2024, page 115).
  - Braskem: its 8 bond lines and 1 other line are in foreign currency (the summary table on DFP 2024 page 116 lists them under "Moeda estrangeira"). Its foreign-currency debt goes from R$ 20.8 bn to R$ 65.1 bn. Its rank stays 1st, because its filed equity is not positive.
  - Energisa: its foreign-currency debt line (R$ 6.7 bn, DFP 2024 page 196) is added, swapped to CDI, because the note says that 100 % of it is hedged (page 192).
  - Grendene: it does not depend on Braskem (its own quote, FRE 2021).
- FX cross-check: 26 profiles match the filed split "Em Moeda Estrangeira" within 20 %. 49 companies leave the split at 0 (for example Petrobras, Vale, BRF), so the check cannot run for them. 8 are flagged. A flag means the note's lines and the filed split disagree; the cause and its effect differ by company:
  - PTAX loss too low: Randon (0.26) and Frasle (0.41; their fixed-rate lines name no currency), Schulz (0.76; 1 dollar line of R$ 94 mn was dropped for a misread amount). Oi (0) is not checked yet.
  - SELIC loss too low, PTAX correct: Sabesp (0.56) swaps all its foreign-currency debt, and about R$ 1.5 bn of it (the yen loans) is not identified in its lines, so it is not in its floating debt.
  - No error in the shock: Eletrobras (now Axia Energia, 0) swaps its foreign-currency debt to CDI, and that debt is already in its CDI line. Braskem (1.33) includes the R$ 15.5 bn financing of Braskem Idesa, which the filing reports outside accounts 2.01.04 + 2.02.01; without it the ratio is 1.01.
  - JBS has 13 times more foreign-currency lines than its filed split, which looks incomplete.

## Limits

- The hybrid shock effect is an accounting pass-through with stated assumptions: no change in revenue or operating costs, debt held constant, 34 % tax. It is not learned from data.
- Without a usable debt profile, the IPCA and foreign-currency channels give 0. This understates the loss of a company with foreign-currency debt that the extraction did not read.
- The extraction reads only the text layer of a filing. 8 debt runs found no debt table in the text. 5 of them are companies with filed debt: Azevedo & Travassos, Magazine Luiza, PRIO (notes mostly images), Vulcabras and Positivo (notes text is unreadable glyphs). Positivo keeps a rank through the floating-share estimate, but its IPCA and foreign-currency channels give 0.
- The shock uses the debt-note amounts, so leases outside the note get no shock. For 9 companies the gap between the note and the filed debt equals their filed lease subaccounts (2.01.04.03 + 2.02.01.03) exactly: Petrobras R$ 230 bn, Vale R$ 4.4 bn, Cogna R$ 2.9 bn, Panvel, Tegma, Alupar, Mahle Metal Leve, Grendene and Unipar. For Yduqs it is close (a gap of R$ 1.63 bn against R$ 1.65 bn of leases). Leases in reais at a fixed rate are not exposed, but leases in dollars also revalue with PTAX, so the PTAX loss of a company with dollar leases is too low.
- For 3 companies the note lines miss debt that is not leases, so their shock is too low. Romi's lines cover 54 % of it. Vivo's leases are R$ 15.2 bn of a R$ 17.2 bn gap, so its lines cover 65 %. EZTC's only line is TR-indexed, and its CDI debentures (about R$ 0.3 bn) are missing, so its SELIC effect is 0.
- Known classification error: Minerva's floating debt counts its swapped lines before a R$ 7.9 bn swap offset in its note, so its SELIC loss is up to R$ 0.16 bn too high; its rank does not change. The FX cross-check finds a PTAX loss that is too low for Randon, Frasle and Schulz, and a SELIC loss that is too low for Sabesp (see Extraction).
- A debt line counts as foreign currency only when its type or its quote says so. For example, Braskem's bond lines name no currency, so its foreign-currency share (0.30) is too low. Braskem ranks first anyway, because its filed equity is not positive.
- The neural shock response is an extrapolation from 45 quarters of macro data.
- The universe holds survivors: companies with all 57 quarters. Statements are the latest filed version, not point-in-time.
- The sector is the current sector of the CVM register.
- The extraction covers only what a company discloses. A missing link is not proof that no link exists.
- An attention weight is a model weight, not proof of a causal link.

## Data sources

All public: CVM open data (DFP, ITR, FRE, FCA and the company register), the BCB time series system, and the B3 quotation history.
