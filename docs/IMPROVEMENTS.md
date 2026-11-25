# Improvement tracker

The owner's goal: track every improvement in order, and re-evaluate after every 2 steps. Each re-evaluation is done by a fresh reviewer agent that gets the code and the results, not the author's conclusions. Its verdict is copied here, with what changed because of it.

Background: `docs/network_panel.md`.

| Step | Improvement | Status | Evidence |
|---|---|---|---|
| 1 | State the panel verdict in the README and the pages | done | README section "What the network panel found"; `docs/network_panel.md` |
| 2 | Build the filed-outcome panel (financial result, translation and hedge lines, FX loans, equity) for all non-financial filers | done | `src/data_ingestion/filed_panel.py` writes `data/filed_panel.csv`: 18,637 company-quarters, 637 non-financial companies, 2011 Q1 to 2024 Q4. Each quarter is the year to date minus the previous year to date, so the 4 quarters add up to the annual value for 99.7 % to 100 % of companies (checked in 2015, 2020, 2023). The first method (3-month rows) failed this check for 20 % of companies, because restated quarters did not add up. Label map `data/dra_label_map.csv`: 609 hedge labels, 398 translation labels; 6 missed hedge labels and 1 missed translation label were found by a review and added. Translation lines are filled for 27.5 % of company-quarters, hedge lines for 17.2 %. 16 tests in `tests/test_filed_panel.py`. |
| RE1 | Re-evaluation of steps 1-2 | pending | |
| 3 | Signed first-hop event test (kill rule below, fixed before the run) | pending | |
| 4 | Pass-through regression with firm fixed effects (kill rule below) | pending | |
| RE2 | Re-evaluation of steps 3-4 | pending | |
| 5 | If step 3 passes: the stress engine uses the fitted signed layer; score the 2025 filings frozen | pending | |
| 6 | Contagion capacity bound for 4 historical shock sizes | pending | |
| RE3 | Re-evaluation of steps 5-6 | pending | |

## Kill rules, fixed before the runs

- **Step 3 passes** only if the signed predictor (pooled pass-through plus each firm's translation and hedge slopes) beats both the shipped rule and the sector mean in at least 7 of the 9 quarters since 2015 with a PTAX move above 10 %, with a mean rank-correlation gain of at least 0.15 and t ≥ 2 across those quarters. Each quarter is fitted only on earlier quarters.
- **Step 4 fails** if the out-of-sample skill against the firm mean is below 10 %, or it wins fewer than 9 of 12 test quarters (2022-2024), or the 95 % interval of the dollar pass-through contains 0 or 1.
- If step 3 fails, the README says that the firm-level data add nothing beyond the sector average, and the product is the audited exposure data set.

## Re-evaluations

(none yet)
