# Improvement tracker

The owner's goal: track every improvement in order, and re-evaluate after every 2 steps. Each re-evaluation is done by a fresh reviewer agent that gets the code and the results, not the author's conclusions. Its verdict is copied here, with what changed because of it.

Background: `docs/network_panel.md`.

| Step | Improvement | Status | Evidence |
|---|---|---|---|
| 1 | State the panel verdict in the README and the pages | done | README section "What the network panel found"; `docs/network_panel.md` |
| 2 | Build the filed-outcome panel (financial result, translation and hedge lines, FX loans, equity) for all non-financial filers | done, fixed after RE1 and its check | `src/data_ingestion/filed_panel.py` writes `data/filed_panel.csv`: 27,820 company-quarters, 945 non-financial companies (628 consolidated, 317 that file only individual statements), 2011 Q1 to 2024 Q4. Fiscal Q2 and Q3 use the filed 3-month values; fiscal Q1 is the year to date; fiscal Q4 is the annual value minus 9 months. April-to-March companies are included. The financial result matches the raw filings in 107 rows checked by hand, and an independent rebuild matches all 27,822 rows. Translation and hedge are unknown (empty) for 2,769 rows whose filing shows only a nonzero total. `restated_q` marks 1,412 rows where the year-to-date difference disagrees with the filed 3-month value; `suspect_q` marks 58 rows. The step 3 rule leaves 325 to 470 companies in each event quarter. 34 tests in `tests/test_filed_panel.py`. |
| RE1 | Re-evaluation of steps 1-2 | done | Step 1: pass with fixes. Step 2: fail (1 blocker), then fixed. Verdict and changes below. |
| 3 | Signed first-hop event test (kill rule below, fixed before the run) | **failed** | `python -m src.evaluation.event_test` (`data/event_test.csv`). The signed predictor beats both baselines in 2 of 9 quarters; mean gain -0.27, t = -3.0. The pass rule needs 7, 0.15 and 2. Result and a design problem of the target: see "Step 3 result" below. |
| 4 | Pass-through regression with firm fixed effects (kill rule below) | pending | |
| RE2 | Re-evaluation of steps 3-4 | pending | |
| 5 | If step 3 passes: the stress engine uses the fitted signed layer; score the 2025 filings frozen | pending | |
| 6 | Contagion capacity bound for 4 historical shock sizes | pending | |
| RE3 | Re-evaluation of steps 5-6 | pending | |

## Kill rules, fixed before the runs

Sharpened after RE1 and after its check, before any step-3 or step-4 result exists.

**Common definitions**
- **Event quarters:** quarters from 2015 Q1 to 2024 Q4 with |PTAX(q) / PTAX(q-1) - 1| > 10 %, where PTAX is the last SGS-1 value of the quarter (`data/macro_quarterly.csv`). These are 9 quarters: 2015 Q1, 2015 Q3, 2018 Q2, 2020 Q1, 2021 Q2, 2022 Q1, 2022 Q2, 2024 Q2, 2024 Q4. The shock s(q) is that signed change.
- **Firms:** all rows of `data/filed_panel.csv` with fin_result_q, oci_translation_q and oci_hedge_q known, `suspect_q` false, `equity_lag` at least 5 % of `total_assets_lag` and at least R$ 10 million. A quarter needs at least 50 firms. (`restated_q` rows stay: they use the filed 3-month value.)
- **Target (step 3):** (fin_result_q + oci_translation_q + oci_hedge_q) / equity_lag, before tax (`oci_tax_q` is not included).

**Step 3 passes** only if all of these are true:
1. The signed predictor (pooled pass-through plus each firm's translation and hedge slopes, fitted only on quarters before q) beats both the shipped rule and the sector mean in at least 7 of the 9 event quarters.
2. Its mean gain over the better of the 2 baselines, per quarter, is at least 0.15.
3. A one-sample t statistic over the 9 gains is at least 2.
- Score: the Spearman rank correlation with the target in each event quarter.
- **Shipped rule:** (debt_short_lag + debt_long_lag) × s(q) × 0.66 / equity_lag. The 2024 debt notes are not used, because they would look ahead.
- **Sector mean:** for each CVM sector, the median of target / s over the event quarters before q, times s(q). (A median, so that a few companies with almost no equity do not decide it.)
- The firm's own 4-quarter mean is reported as a 3rd comparison, but it is not part of the rule.

**Step 4 fails** if any of these is true:
1. The out-of-sample skill is below 10 %. Skill = 1 - MSE(model) / MSE(firm mean), on fin_result_q / total_assets_lag.
2. The model wins fewer than 9 of the 12 test quarters (2022 to 2024).
3. The 95 % interval of the dollar pass-through contains 0 or 1.
- Exposure: (fx_loans_short_lag + fx_loans_long_lag) × s(q) / total_assets_lag, for firms with a nonzero filed split only. A 0 in the split does not prove that a firm has no foreign-currency debt. Rows with `suspect_q` true are left out.
- The model has firm fixed effects, with an expanding window up to q-1. Winsor limits come from the training quarters only.
- Standard errors are clustered by quarter, with a firm bootstrap as a check.

**Step 3 method, fixed before the first run** (`src/evaluation/event_test.py`)
- For each event quarter q, the training rows are all panel rows before q that meet the firm rule. Training variables are winsorized at their 1st and 99th percentiles in the training rows.
- (a) Shipped rule, as above, with the sign of a loss: -0.66 × (debt_short_lag + debt_long_lag) × s(q) / equity_lag.
- (b) Pooled pass-through: the slope θ of fin_result_q / equity_lag on x = -(fx_loans_short_lag + fx_loans_long_lag) × s / equity_lag, with quarter fixed effects (both demeaned by quarter), by least squares on the training rows. Prediction: θ × x(q). A firm with a 0 split gets 0.
- (c) Signed predictor: (b) plus β_i × s(q). β_i is the slope through the origin of (oci_translation_q + oci_hedge_q) / equity_lag on s over the firm's training quarters. A firm with fewer than 8 training quarters gets β_i = 0.
- Sector median: for each sector, the median of target / s over the training quarters with |s| > 10 % (from 2011 on, so 2015 Q1 has 4 earlier events), times s(q). A sector without such rows gets the median of all firms.
- Firm 4-quarter mean (reported only): the mean target of the firm's last 4 training quarters.
- Score: Spearman correlation with the target, on the firms that have a target in q. Ties get average ranks.

**Step 4 method, fixed before the first run** (`src/evaluation/pass_through.py`)
- Rows: all panel rows with fin_result_q, `total_assets_lag` > 0, `suspect_q` false, and a nonzero filed split at q-1 (fx_loans_short_lag + fx_loans_long_lag > 0).
- y = fin_result_q / total_assets_lag; x = -(fx_loans_short_lag + fx_loans_long_lag) × s / total_assets_lag, in every quarter (not only event quarters).
- θ: least squares of y on x after removing firm and quarter means (two-way fixed effects, by alternating demeaning until the change is below 1e-12).
- For each test quarter q from 2022 Q1 to 2024 Q4: fit on the rows before q, with y and x winsorized at the 1st and 99th percentiles of those rows; the same limits are applied to the rows of q. Prediction: the firm's training mean of y + θ × (x - the firm's training mean of x). Baseline: the firm's training mean of y. Only firms with at least 4 training quarters are scored.
- Skill = 1 - (sum of squared errors of the model) / (sum of squared errors of the baseline), over all rows of the 12 test quarters. A quarter is a win when the model's squared error in that quarter is lower.
- Interval of θ: θ fitted on all rows through 2024 Q4, with a standard error clustered by quarter; 95 % interval = θ ± 1.96 standard errors. A firm bootstrap (200 draws, percentile interval) is reported as a check.

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

### Check of the RE1 fixes, 2026-11-25

4 independent checkers (raw recompute, edge cases, labels, fitness for the step 3 test) tried to break the fixed panel. They found the financial result correct: 107 rows recomputed by hand from the raw filings, and an independent rebuild of all 27,822 rows. Their findings, and what changed:

| Severity | Finding | Change |
|---|---|---|
| blocker (3 checkers) | A filing with only the 4.02 total got translation = hedge = 0; differencing it against a detailed filing invented class flows (Natura &Co 2020 Q1: OCI 134 % of equity, counted as 0) | Classes are empty (unknown) when a filing shows only a nonzero total; 0 when the total is 0. Rule drops unknown rows (36 to 76 per event quarter). Test added. |
| blocker | Filings declared in units but in thousands, when the filing before has the same error: equity_lag 1000 times too small (NTS 2020 Q1 target -18.6 instead of -0.019) | `fetch_cvm.unit_fixes` also accepts proof from a later filing's comparative. `data/cvm_quarterly.csv` rebuilt: 276 rows of 52 companies fixed, none in the 100-company universe. Test added. |
| should-fix (2 checkers) | A line that changes class between filings (gross to net of tax, renamed lines) corrupts the class flows (Braskem 2022 Q2 hedge error 18 % of equity) | Fiscal Q2 and Q3 use the filed 3-month values, classified inside 1 filing. Translation moved into profit on a disposal is class other (Vale 2021 Q2 and 2024 Q4 now match). |
| should-fix | The net-of-tax rule left fragments that matched the tax rule | The whole net-of-tax clause is removed before the tax test. Tests added. |
| should-fix | Missed labels: FX on debt under hedge accounting (São Martinho), misspellings of "cambial", "diferenças de câmbio" | Added to the rules. Tests added. |
| should-fix | Equity close to 0 gives targets up to -1,816 | Equity floor in the rule; sector baseline is a median. Largest absolute target is now 3.9. |
| should-fix | `restated_q` dropped correct rows, and fiscal Q1 and Q4 were never checked | `restated_q` is now only information (the filed value is used). New `suspect_q` for Q4 placeholders and jumps of total assets by 100 times or more. Fiscal Q4 class flows still cannot be checked (stated in the module). |
| note | Financial holdings in a non-financial sector (BNDESPAR, REAG, YBYRA) | Removed by a short list in the module. |
| note | Gross and net of tax hedge lines are mixed; hedge includes interest-rate and commodity hedges | Stated in the module. No change: the effect is a scale of at most 1/0.66 on the hedge part. |
| note | Listed holding and listed subsidiary both in the panel (Suzano Holding and Suzano); small look-ahead from latest-version stocks; ITRs with a 3-month column only get fiscal Q1; companies that file consolidated statements lose their individual-only quarters | Stated in the module or here. No change. |
| note | Iochpe 2015 Q3 files its translation under a financial-instrument label | The panel follows the filed label. No change. |

## Step 3 result, 2026-11-25

Pre-registered rule: **failed** (2 wins of 9, mean gain -0.27, t = -3.0).

| Quarter | s | Firms | Shipped | Pass-through | Signed | Sector median | Firm 4-quarter mean |
|---|---|---|---|---|---|---|---|
| 2015 Q1 | +0.208 | 331 | 0.671 | 0.151 | 0.257 | 0.426 | 0.794 |
| 2015 Q3 | +0.281 | 326 | 0.621 | 0.204 | 0.356 | 0.384 | 0.817 |
| 2018 Q2 | +0.160 | 325 | 0.656 | 0.148 | 0.281 | 0.367 | 0.758 |
| 2020 Q1 | +0.290 | 357 | 0.543 | 0.064 | 0.243 | 0.337 | 0.565 |
| 2021 Q2 | -0.122 | 427 | -0.427 | 0.063 | 0.139 | -0.061 | 0.336 |
| 2022 Q1 | -0.151 | 440 | -0.581 | -0.014 | 0.135 | -0.035 | 0.506 |
| 2022 Q2 | +0.106 | 436 | 0.700 | 0.079 | 0.202 | 0.290 | 0.607 |
| 2024 Q2 | +0.113 | 470 | 0.694 | 0.077 | 0.209 | 0.262 | 0.733 |
| 2024 Q4 | +0.137 | 434 | 0.654 | -0.006 | 0.162 | 0.275 | 0.767 |

(Spearman correlation with the target. Pooled pass-through θ: 0.60 to 0.85.)

**What the result says, and what it does not say.** The target is the level of the effect. That level is mostly each firm's normal interest cost, not its response to the dollar:
- The firm's own 4-quarter mean, which has no FX information, scores 0.34 to 0.82.
- The shipped rule scores +0.54 to +0.70 when the dollar rises and -0.43 to -0.58 when it falls. Its sign follows the shock, so it ranks debt level, not the FX response.

So this test cannot tell whether the signed layer captures the FX response; it mostly measures leverage. The result stands as pre-registered: the signed predictor did not beat the baselines on this target. The author chose the level target (a recorded decision before RE1), and RE1 proposed the same target. Whether to pre-register a new test on the response (the target minus the firm's own normal level) is the owner's decision; nothing about it was run.

