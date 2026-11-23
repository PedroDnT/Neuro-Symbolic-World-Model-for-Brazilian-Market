"""Macro stress test with 2 engines side by side.

Shock units (natural units):
    SELIC_delta     percentage points of the SELIC target (3.0 = +300 basis points)
    IPCA_delta      percentage points of the quarterly IPCA change
    PTAX_delta_pct  percent change of the USD/BRL rate (15 = +15 %)
The shock starts after the last filed quarter and holds for the horizon.

Hybrid engine (main ranking):
    base path   the seasonal ridge forecast (src/model/ridge.py), no macro input
    shock path  base path + the accounting transmission of the shock to the debt of each
                company (src/simulator/transmission.py)
Neural engine (comparison): the graph recurrent model of main.py. The base path holds the
macro inputs at their last value; the shock path moves them. History is replayed first, so
the recurrent state is not zero at the start.

Ranking: the equity change caused by the shock (shock path minus base path at the last
quarter of the horizon), as a percent of the last filed equity. A company whose filed equity
is not positive is already insolvent by book value: it comes first, with a flag.

Run: python -m src.simulator.stress_test --selic 3.0 --ptax-pct 15
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from src.data_ingestion.download import DATA_DIR, RESULTS_DIR
from src.knowledge import rules
from src.knowledge.sensitivity import floating_shares
from src.model.dataset import N_BALANCE, balance_levels, edges_of, load_data, load_model, load_ridge
from src.protocol import check_balance
from src.simulator import transmission

ACCOUNTS = transmission.ACCOUNTS


class Shock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    SELIC_delta: float = 0.0
    IPCA_delta: float = 0.0
    PTAX_delta_pct: float = 0.0


@dataclass
class Scenario:
    shock: Shock
    horizon: int
    meta: dict
    last: np.ndarray  # last filed balance in reais, shape (N, 4)
    hybrid_base: np.ndarray  # (horizon, N, 4) in reais
    hybrid_shock: np.ndarray
    neural_base: np.ndarray
    neural_shock: np.ndarray
    exposures: dict  # cd_cvm -> transmission.Exposure
    effects: dict  # cd_cvm -> list of transmission.QuarterEffect
    flows: pd.DataFrame  # last filed quarter, indexed by cd_cvm
    attention: tuple  # neural, first shock step: (edge_index with self-loops, mean weight for each edge)
    neural_variant: str

    def coverage_under_shock(self) -> list[rules.RuleResult]:
        results = []
        for cd_cvm in self.meta["cd_cvm"]:
            exposure = self.exposures[cd_cvm]
            row = self.flows.loc[cd_cvm] if cd_cvm in self.flows.index else None
            results.append(
                rules.interest_coverage_under_shock(
                    row["ebit_ttm"] if row is not None else np.nan,
                    row["finexp_ttm"] if row is not None else np.nan,
                    exposure.floating_debt,
                    self.shock.SELIC_delta,
                    exposure.floating_basis,
                )
            )
        return results

    def contagion(self, coverage: list[rules.RuleResult]) -> list[str]:
        """For each company: its suppliers with "high" dependence whose coverage breaches under the shock."""
        path = DATA_DIR / "extracted_edges.csv"
        if not path.exists():
            return [""] * len(self.meta["cd_cvm"])
        edges = pd.read_csv(path)
        edges = edges[edges["dependence"] == "high"]
        breach = {code: result.breach is True for code, result in zip(self.meta["cd_cvm"], coverage)}
        ticker = dict(zip(self.meta["cd_cvm"], self.meta["tickers"]))
        found = []
        for cd_cvm in self.meta["cd_cvm"]:
            suppliers = edges.loc[edges["dst_cd_cvm"] == cd_cvm, "src_cd_cvm"]
            found.append(", ".join(sorted(ticker[s] for s in suppliers if breach.get(s))))
        return found

    def table(self) -> pd.DataFrame:
        """All companies, ranked by the hybrid engine from highest distress to lowest."""
        equity_last = self.last[:, 3]
        positive = equity_last > 0

        def change_pct(base: np.ndarray, shocked: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(positive, (shocked[-1, :, 3] - base[-1, :, 3]) / equity_last * 100, np.nan)

        def residual(path: np.ndarray) -> np.ndarray:
            return (np.abs(path[..., 0] - path[..., 1:].sum(axis=-1)) / np.abs(path[..., 0])).max(axis=0)

        coverage = self.coverage_under_shock()
        frame = pd.DataFrame(
            {
                "ticker": self.meta["tickers"],
                "company": self.meta["companies"],
                "sector": self.meta["sectors"],
                "equity_filed": equity_last,
                "equity_not_positive": ~positive,
                "hybrid_equity_base": self.hybrid_base[-1, :, 3],
                "hybrid_equity_shock": self.hybrid_shock[-1, :, 3],
                "hybrid_change_pct": change_pct(self.hybrid_base, self.hybrid_shock),
                "neural_equity_base": self.neural_base[-1, :, 3],
                "neural_equity_shock": self.neural_shock[-1, :, 3],
                "neural_change_pct": change_pct(self.neural_base, self.neural_shock),
                "gross_debt": [self.exposures[c].gross_debt for c in self.meta["cd_cvm"]],
                "floating_share": [self.exposures[c].floating_share for c in self.meta["cd_cvm"]],
                "floating_basis": [self.exposures[c].floating_basis for c in self.meta["cd_cvm"]],
                "ipca_share": [self.exposures[c].ipca_share for c in self.meta["cd_cvm"]],
                "fx_share": [self.exposures[c].fx_share for c in self.meta["cd_cvm"]],
                "floating_debt": [self.exposures[c].floating_debt for c in self.meta["cd_cvm"]],
                "ipca_debt": [self.exposures[c].ipca_debt for c in self.meta["cd_cvm"]],
                "fx_debt": [self.exposures[c].fx_debt for c in self.meta["cd_cvm"]],
                "fx_lines": [self.exposures[c].fx_lines for c in self.meta["cd_cvm"]],
                "filed_fx": [self.exposures[c].filed_fx for c in self.meta["cd_cvm"]],
                "fx_check": [self.exposures[c].fx_check for c in self.meta["cd_cvm"]],
                "profile_basis": [self.exposures[c].profile_basis for c in self.meta["cd_cvm"]],
                "coverage_under_shock": [result.value for result in coverage],
                "coverage_breach": [result.breach for result in coverage],
                "breaching_suppliers": self.contagion(coverage),
                "hybrid_residual": residual(self.hybrid_shock),
                "neural_residual": residual(self.neural_shock),
            }
        )
        # Companies with filed equity that is not positive take the first ranks in both engines.
        # A company with no result (an unknown input) gets no rank.
        first = int((~positive).sum())
        frame["neural_rank"] = (frame["neural_change_pct"].rank(method="first") + first).astype("Int64")
        frame = frame.sort_values(["equity_not_positive", "hybrid_change_pct"], ascending=[False, True], na_position="last")
        frame = frame.reset_index(drop=True)
        ranked = frame["equity_not_positive"] | frame["hybrid_change_pct"].notna()
        frame.insert(0, "rank", pd.Series(range(1, len(frame) + 1), dtype="Int64").where(ranked))
        return frame

    def rank_correlation(self) -> float:
        """Spearman correlation of the 2 engines' equity changes, over companies with positive filed equity."""
        table = self.table()
        both = table.dropna(subset=["hybrid_change_pct", "neural_change_pct"])
        return float(both["hybrid_change_pct"].rank().corr(both["neural_change_pct"].rank()))


def _neural_paths(snapshots: list, meta: dict, shock: Shock, horizon: int, variant: str) -> tuple:
    model = load_model(variant, meta)
    last = snapshots[-1]
    edge_index, edge_attr = edges_of(last, variant)
    with torch.no_grad():
        state = model.initial_state(last.x.shape[0])
        for snapshot in snapshots[:-1]:
            past_edges, past_attr = edges_of(snapshot, variant)
            state = model(snapshot.x, past_edges, state, past_attr).h

        std = torch.tensor(meta["macro_std"], dtype=torch.float32)
        mean = torch.tensor(meta["macro_mean"], dtype=torch.float32)
        last_macro = last.x[0, N_BALANCE:] * std + mean
        natural = torch.tensor([shock.SELIC_delta, shock.IPCA_delta, float(last_macro[2]) * shock.PTAX_delta_pct / 100])
        scale = torch.tensor(meta["scale"], dtype=torch.float64).unsqueeze(1)

        def path(macro_shift: torch.Tensor) -> tuple[np.ndarray, tuple]:
            x, h, quarters, attention = last.x.clone(), state, [], None
            x[:, N_BALANCE:] += macro_shift
            for quarter in range(horizon):
                step = model(x, edge_index, h, edge_attr)
                # Project the level again, so float32 rounding does not grow along the rollout.
                level = model.accounting_projection(step.prediction)
                check_balance(level)
                if quarter == 0:
                    attention = (step.attention[0], step.attention[1].mean(dim=1))
                quarters.append(level.double() * scale)
                x = torch.cat([level, x[:, N_BALANCE:]], dim=1)
                h = step.h
            return torch.stack(quarters).numpy(), attention

        base, _ = path(torch.zeros(3))
        shocked, attention = path(natural / std)
    return base, shocked, attention


def simulate(shocks: dict | None = None, horizon: int = 4, variant: str = "sector") -> "Scenario":
    shock = Shock(**(shocks or {}))
    snapshots, meta = load_data()
    scale = np.asarray(meta["scale"], dtype=np.float64)[None, :, None]

    ridge = load_ridge(meta)
    levels = balance_levels(snapshots)
    hybrid_base = ridge.rollout(levels, horizon).numpy() * scale
    last = levels[-1].double().numpy() * scale[0]

    universe = pd.DataFrame({"cd_cvm": meta["cd_cvm"], "ticker": meta["tickers"]})
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    flows = statements[statements["quarter"] == pd.Timestamp(meta["quarters"][-1])].set_index("cd_cvm")
    profile_path = DATA_DIR / "debt_profile.csv"
    profile = pd.read_csv(profile_path) if profile_path.exists() else pd.DataFrame(columns=["cd_cvm", "floating_share"])
    exposures = transmission.exposures(flows.loc[meta["cd_cvm"]], profile, floating_shares(universe, meta["train_end"]))
    effects = {
        code: transmission.quarter_effects(exposures[code], shock.SELIC_delta, shock.IPCA_delta, shock.PTAX_delta_pct, horizon)
        for code in meta["cd_cvm"]
    }
    cumulative = np.stack([transmission.cumulative_delta(effects[code]) for code in meta["cd_cvm"]], axis=1)
    hybrid_shock = hybrid_base + cumulative

    neural_base, neural_shock, attention = _neural_paths(snapshots, meta, shock, horizon, variant)
    return Scenario(
        shock, horizon, meta, last, hybrid_base, hybrid_shock, neural_base, neural_shock,
        exposures, effects, flows, attention, variant,
    )


def run_stress_test(shocks: dict | None = None, horizon: int = 4, variant: str = "sector") -> pd.DataFrame:
    """The ranked table of all companies, from highest distress to lowest (hybrid engine),
    with the neural engine's result next to it."""
    return simulate(shocks, horizon, variant).table()


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--selic", type=float, default=0.0, help="percentage points")
    parser.add_argument("--ipca", type=float, default=0.0, help="percentage points of the quarterly change")
    parser.add_argument("--ptax-pct", type=float, default=0.0, help="percent")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--variant", default="sector", choices=["sector", "typed"], help="neural variant to compare")
    args = parser.parse_args()
    shocks = {"SELIC_delta": args.selic, "IPCA_delta": args.ipca, "PTAX_delta_pct": args.ptax_pct}
    scenario = simulate(shocks, args.horizon, args.variant)
    table = scenario.table()
    RESULTS_DIR.mkdir(exist_ok=True)
    label = f"selic{args.selic:+g}_ipca{args.ipca:+g}_ptax{args.ptax_pct:+g}"
    if args.variant != "sector":
        label += f"_{args.variant}"
    table.to_csv(RESULTS_DIR / f"stress_{label}.csv", index=False)
    shown = ["rank", "ticker", "sector", "equity_filed", "hybrid_change_pct", "neural_change_pct", "neural_rank",
             "floating_share", "floating_basis", "coverage_under_shock", "breaching_suppliers"]
    print(f"stress test {shocks}, horizon {args.horizon} quarters, neural variant {args.variant}")
    print(table[shown].head(15).to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"rank correlation hybrid vs neural: {scenario.rank_correlation():.2f}")
    print(f"largest balance residual / assets: hybrid {table['hybrid_residual'].max():.2e}, neural {table['neural_residual'].max():.2e}")
    return table


if __name__ == "__main__":
    main()
