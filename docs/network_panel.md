# Network design panel (2026-11-23)

The owner asked how to model the engine as a network, as first planned. A panel answered it: 3 independent designs, 3 judges with different questions, and a synthesis written by hand (the synthesis agent hit a usage limit).

The numbers below come from the panel's read-only reruns on the data in this repository. They are **provisional**: steps 2 to 4 of `docs/IMPROVEMENTS.md` rebuild them in the repository with tests.

## Designs

| Design | Idea | Judges' scores (identification, feasibility, contribution) |
|---|---|---|
| Structural | Signed exposure ledger in reais; losses propagate by clearing (Eisenberg-Noe, DebtRank); unknown link amounts are worst-case bounds. Nothing is fitted. | 6, 7, 6 |
| Econometric | Weekly share returns measure each firm's net exposure to the dollar and rates; a link type stays only if linked firms co-move more than in reshuffled networks. | 5, 5, 4 |
| Structured graph model | A 2-layer linear graph: macro → firm edges weighted by filed amounts (fixed), about 7 global parameters fitted on filed result lines; firm → firm edges only as a prior. | 7, 8, 7 |

## Where the designs agree

1. **The firm-to-firm network does not matter with this data.** 3 independent checks agree: the worst-case second-round loss is about 6 to 7 % of the direct losses; supplier links add 0.00 to 0.05 against placebo networks; the graph adds +0.0 points over plain shrinkage.
2. **The shipped ranking has no rank information.** Debt × shock × 0.66 has a rank correlation of -0.11 to +0.11 with each firm's filed equity effect in each of 7 shock years (annual data).
3. **A signed first layer has information.** The debt effect plus the translation of foreign subsidiaries plus hedge results, all from filed accounts, reaches a rank correlation of 0.29 to 0.39 in 4 past shock years.
4. **1 parameter is clearly identified:** the pass-through of the dollar to the financial result, 0.54 (standard error 0.05). Out of sample it has +30 % skill against the firm mean (+24 % on the 100 firms, with an interval of -12 % to +40 %). The gain sits in a few large shocks and a few firms, and the pass-through is lower in large shocks (0.46) than in calm quarters (0.76).
5. **The honest network is bipartite:** macro factors → firms with signed edges, and firms → lenders. Named firm pairs are reported as bounds, not as propagation.

## The open question

Is the corrected first layer more than "exporters differ"? The test in step 3 must beat the sector average, which is the obvious answer.

## Out of scope

A graph neural network; networks of market betas (51 % of the spread of dollar betas is sector, and prices need adjustment work); paid extraction; the currency of leases.
