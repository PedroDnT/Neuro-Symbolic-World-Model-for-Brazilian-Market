# Improvement tracker

The owner's goal: track every improvement in order, and re-evaluate after every 2 steps. Each re-evaluation is done by a fresh reviewer agent that gets the code and the results, not the author's conclusions. Its verdict is copied here, with what changed because of it.

Background: `docs/network_panel.md`.

| Step | Improvement | Status | Evidence |
|---|---|---|---|
| 1 | State the panel verdict in the README and the pages | done | README section "What the network panel found"; `docs/network_panel.md` |
| 2 | Build the filed-outcome panel (financial result, translation and hedge lines, FX loans, equity) for all non-financial filers | done, fixed after RE1 | `src/data_ingestion/filed_panel.py` writes `data/filed_panel.csv`: 27,823 company-quarters, 948 non-financial companies (630 consolidated, 318 that file only individual statements), 2011 Q1 to 2024 Q4. Flows are differences of year-to-date values inside the fiscal year, so April-to-March companies (sugar and ethanol) are included. Comprehensive-income lines are classified inside each filing, then the class sums are differenced. `restated_q` flags 1,219 of 13,948 fiscal Q2 and Q3 rows (8.7 %) where the derived flow of 3.06 or 4.02 differs by more than 1 % from the filed 3-month row. Lagged stocks (`<stock>_lag`) come from the previous calendar quarter. 24 tests in `tests/test_filed_panel.py`. |
| RE1 | Re-evaluation of steps 1-2 | done | Step 1: pass with fixes. Step 2: fail (1 blocker), then fixed. Verdict and changes below. |
| 3 | Signed first-hop event test (kill rule below, fixed before the run) | pending | |
| 4 | Pass-through regression with firm fixed effects (kill rule below) | pending | |
| RE2 | Re-evaluation of steps 3-4 | pending | |
| 5 | If step 3 passes: the stress engine uses the fitted signed layer; score the 2025 filings frozen | pending | |
| 6 | Contagion capacity bound for 4 historical shock sizes | pending | |
| RE3 | Re-evaluation of steps 5-6 | pending | |

## Kill rules, fixed before the runs

Sharpened after RE1, before any step-3 or step-4 result exists.

**Common definitions**
- **Event quarters:** quarters from 2015 Q1 to 2024 Q4 with |PTAX(q) / PTAX(q-1) - 1| > 10 %, where PTAX is the last SGS-1 value of the quarter (`data/macro_quarterly.csv`). These are 9 quarters: 2015 Q1, 2015 Q3, 2018 Q2, 2020 Q1, 2021 Q2, 2022 Q1, 2022 Q2, 2024 Q2, 2024 Q4. The shock s(q) is that signed change.
- **Firms:** all rows of `data/filed_panel.csv` with an outcome, `equity_lag` > 0, and `restated_q` false. A quarter needs at least 50 firms.
- **Target (step 3):** (fin_result_q + oci_translation_q + oci_hedge_q) / equity_lag, before tax (`oci_tax_q` is not included).

**Step 3 passes** only if all of these are true:
1. The signed predictor (pooled pass-through plus each firm's translation and hedge slopes, fitted only on quarters before q) beats both the shipped rule and the sector mean in at least 7 of the 9 event quarters.
2. Its mean gain over the better of the 2 baselines, per quarter, is at least 0.15.
3. A one-sample t statistic over the 9 gains is at least 2.
- Score: the Spearman rank correlation with the target in each event quarter.
- **Shipped rule:** (debt_short_lag + debt_long_lag) × s(q) × 0.66 / equity_lag. The 2024 debt notes are not used, because they would look ahead.
- **Sector mean:** for each CVM sector, the mean of target / s over the event quarters before q, times s(q).
- The firm's own 4-quarter mean is reported as a 3rd comparison, but it is not part of the rule.

**Step 4 fails** if any of these is true:
1. The out-of-sample skill is below 10 %. Skill = 1 - MSE(model) / MSE(firm mean), on fin_result_q / total_assets_lag.
2. The model wins fewer than 9 of the 12 test quarters (2022 to 2024).
3. The 95 % interval of the dollar pass-through contains 0 or 1.
- Exposure: (fx_loans_short_lag + fx_loans_long_lag) × s(q) / total_assets_lag, for firms with a nonzero filed split only. A 0 in the split does not prove that a firm has no foreign-currency debt.
- The model has firm fixed effects, with an expanding window up to q-1. Winsor limits come from the training quarters only.
- Standard errors are clustered by quarter, with a firm bootstrap as a check.

If step 3 fails, the README says that the firm-level data add nothing beyond the sector average, and the product is the audited exposure data set.

## Re-evaluations

### RE1 (steps 1-2), 2026-11-25

**Verdict, word for word** (fresh reviewer agent; it got the code, the data and the tracker, not the author's conclusions):

> ## Verdicts
> - **Step 1: PASS WITH FIXES.** Two wording fixes are needed.
> - **Step 2: FAIL.** One blocker: the hedge and translation split of OCI is wrong for large FX firms. The financial result (3.06) and the OCI total (4.02) are correct.
>
> The tests pass (16/16, full suite 96). I edited no repository file; my scratch scripts are in the session scratchpad.
>
> ## Step 1 findings
> - **should-fix, README.md:128-130.** The README states 0.54 (standard error 0.05), 0.29–0.39, −0.11 to +0.11 and 6–7 % as findings. No code or output in the repo reproduces them (I grepped). The only warning is the lead-in on line 126.
>   - "Clearly identified" overstates it: `docs/network_panel.md:20` gives an out-of-sample skill interval of −12 % to +40 %, and the pass-through is 0.46 in large shocks but 0.76 in calm quarters.
>   - Fix: add "(panel rerun, not yet reproduced here)" to each bullet and drop "clearly".
> - **note, README.md:3.** The opening still sells the ranking that line 129 says has no rank information. Add a pointer to that finding.
>
> ## Step 2 findings
> - **BLOCKER, filed_panel.py:72-77 with 87-98.** Quarters are computed as differences by account code (CD_CONTA), but companies renumber their 4.02.x lines between filings.
>   - Among same-code year-to-date pairs, 33 % change label and 9.4 % change class.
>   - CSN 2020 Q2: the Q2 line "Ajustes de conversão" (513,453) minus the Q1 line "hedge" (−5,390,043) becomes a spurious **translation of +R$5.90 bn**. The real hedge line (−6.66 bn) has no Q1 twin, so it is dropped.
>   - Braskem 2020 Q4: the DFP hedge line at 4.02.10 (471,728) minus the Q3 translation line at 4.02.10 (4,668,466) becomes a "hedge" of −R$4.20 bn. The real translation (2,658,042 − 4,668,466 = −2.01 bn) is lost.
>   - Same problem at Vale, Gerdau and Embraer. In 2015-16 and 2020-22, 190 (translation) and 198 (hedge) of 2,559 firm-quarters differ from a class-sum method by more than 5 % and R$1 M.
>   - Fix: classify and apply the top-most rule inside each year-to-date filing, then difference the class sums. A class missing from an existing previous filing counts as 0. Add a test with renumbered codes.
> - **should-fix, tax lines (classify).** A tax line counts as hedge or translation only when its label names it. Braskem's Q3 ITR labels it "IR … hedge", but its DFP says only "Imposto de renda". So the parts swing between net and gross of tax. Fix: give tax lines their own class.
> - **should-fix, hyperinflation lines are counted as translation.** 16 IAS 29 labels (about 130 rows) follow the Argentine peso, not PTAX. Move them to other.
> - **should-fix, restatements.** In Q2/Q3, 3.3 % of 3.06 flows and 6.6 % of 4.02 flows differ by more than 1 % from the filed 3-month row. Example: Embraer 2020 Q2 is −499.9 M against −261.5 M filed (the Q1 re-presentation). Fix: add a `restated_q` flag so the tests can drop or check these quarters.
> - **should-fix, tracker evidence.** "The 4 quarters add up to the annual value" is true by construction, since Q4 = DFP minus the 9-month year to date. Report agreement with the filed 3-month rows instead (96.7 % for 3.06, 93.4 % for 4.02, within 1 %).
> - **should-fix, non-December fiscal years.** 16 firms are absent or have March quarters only: Raízen, São Martinho, Jalles, Biosev, Tereos, Cerradinho, BrasilAgro. Sugar exporters are the most FX-exposed group. Fix: derive the fiscal quarter from DT_INI_EXERC.
> - **should-fix, universe.**
>   - 17 foreign issuers (cd_cvm ≥ 80000, no sector) are kept, among them Inter & Co (a bank), GP Investments, Telefónica S.A. and JBS N.V. REAG Capital is kept as a holding.
>   - `filed_panel.py:110` checks no labels, so CELGPAR's 3.06 is income tax.
>   - Fix: drop cd_cvm ≥ 80000 and add label checks for 3.06, 3.11 and 4.02.
>   - Sanepar has 0 rows.
> - **should-fix, FX loans.** `fx_loans_*` is 0 in 70 % of rows and never empty. The split is optional, so 0 does not mean no FX debt. Only about 100 firms per event quarter have a nonzero split.
> - **should-fix, lags.** The panel has no lagged stocks, and 71 rows follow a gap. Tests must merge on quarter−1, not use shift(). Stocks are the latest filed version, not point-in-time.
> - **note, filed_panel.py:120.** Duplicate year-to-date rows (cd_cvm 21601, 2016) are resolved silently by `first`.
>
> ## Kill rules: fix before step 3
> - **Event quarters.** "|PTAX(q)/PTAX(q−1) − 1| > 10 %, using the last SGS-1 value of each quarter, 2015Q1–2024Q4." That gives 9 quarters: 2015Q1, 2015Q3, 2018Q2, 2020Q1, 2021Q2, 2022Q1, 2022Q2, 2024Q2, 2024Q4. With a log change it is 10 quarters, and with depreciation only it is 7. The predictor is signed by the shock.
> - **Target.** "(fin_result_q + oci_translation_q + oci_hedge_q) / equity at q−1". Also state whether the hedge and translation parts are gross of tax. A rank of raw reais only ranks firm size.
> - **Baselines.**
>   - Shipped rule: debt × shock × 0.66, where debt is debt_short + debt_long at q−1 from the panel. The 2024 debt notes would be look-ahead.
>   - Sector mean: the mean of the target per CVM sector over earlier event quarters, times the signed shock.
> - **Test.** Spearman correlation per quarter. Gain = predictor minus the better of the 2 baselines. Mean gain ≥ 0.15 and a one-sample t over 9 gains ≥ 2. Firms are all panel firms with a lagged stock and an outcome, at least 50 per quarter, with foreign issuers dropped.
> - **Step 4.**
>   - Skill = 1 − MSE(model) / MSE(firm mean) on fin_result_q / total_assets at q−1.
>   - Fit with an expanding window up to q−1.
>   - The exposure is fx_loans at q−1, firms with a nonzero split only.
>   - Standard errors clustered by quarter.

**What changed because of it**

| Finding | Change |
|---|---|
| Blocker: renumbered 4.02.x codes | Lines are classified inside each filing (`oci_items`), then class sums are differenced (`quarter_flows`). A class absent from a filing counts as 0. Test with renumbered codes. CSN 2020 Q2 translation is now +R$0.13 bn (was +R$5.90 bn); Braskem 2020 Q4 translation is now -R$2.01 bn. |
| Tax lines | New class `tax` and column `oci_tax_q`. "Líquido de impostos" (net of tax) does not make a line a tax line. A tax line under a counted hedge or translation line stays inside that line. |
| Hyperinflation | IAS 29 lines are class `other`. |
| Restatements | New column `restated_q`; the step 3 and step 4 rules drop flagged rows. |
| Tracker evidence | Replaced by the agreement with the filed 3-month rows. |
| Non-December fiscal years | The fiscal quarter comes from the year-to-date length; the previous filing is matched on the fiscal year start. Test with an April-to-March year. 8 sugar and agriculture companies now have all 4 fiscal quarters. |
| Universe | Foreign issuers (cd_cvm 80000 and above) are dropped. 3.06, 3.11 and 4.02 need their expected label. Companies that never file consolidated statements use their individual statements (Sanepar and 317 others). REAG Capital stays: it is a holding by its CVM sector, and no rule in the code can tell it apart; noted here. |
| FX loans | Documented in the module; the step 4 rule uses firms with a nonzero split only. |
| Lags | New `<stock>_lag` columns, merged on the previous calendar quarter (empty after a gap). Stocks stay the latest filed version; noted in the module. |
| Duplicate rows | Resolved with `fetch_cvm._single`: the value is empty when duplicates disagree. |
| Step 1 wording | "Panel rerun, not yet reproduced here" on each number; "clearly" removed; a warning in the README opening. |
| Kill rules | Rewritten with the proposed definitions, before any run. |
