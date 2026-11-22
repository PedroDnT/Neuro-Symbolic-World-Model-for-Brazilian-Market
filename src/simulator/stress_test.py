"""Macro stress test: shock the macro inputs, roll the model 4 quarters, rank by equity loss.

Shock units (natural units):
    SELIC_delta     percentage points of the SELIC target (3.0 = +300 basis points)
    IPCA_delta      percentage points of the quarterly IPCA change
    PTAX_delta_pct  percent change of the USD/BRL rate (15 = +15 %)

The shock is applied 1 time to the last filed quarter and held for the horizon.
2 paths run from the same state: base (macro held at its last value) and shock. The
difference between the 2 paths is the effect of the shock in the model. History is
replayed first, so the recurrent state is not zero at the start.

Run: python -m src.simulator.stress_test --selic 3.0 --ptax-pct 15
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from src.data_ingestion.download import RESULTS_DIR
from src.model.dataset import N_BALANCE, edges_of, load_data, load_model
from src.protocol import check_balance

ACCOUNTS = ["total_assets", "current_liabilities", "noncurrent_liabilities", "equity"]


class Shock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    SELIC_delta: float = 0.0
    IPCA_delta: float = 0.0
    PTAX_delta_pct: float = 0.0


@dataclass
class Scenario:
    shock: Shock
    horizon: int
    variant: str
    meta: dict
    last: torch.Tensor  # last filed balance in reais, shape (N, 4)
    base: torch.Tensor  # base path in reais, shape (horizon, N, 4)
    shocked: torch.Tensor  # shock path in reais, shape (horizon, N, 4)
    attention: tuple  # first step of the shock path: (edge_index with self-loops, mean weight of each edge)

    def table(self) -> pd.DataFrame:
        """All companies, from the largest equity loss against the base path to the smallest."""
        equity_last, equity_base, equity_shock = self.last[:, 3], self.base[-1, :, 3], self.shocked[-1, :, 3]
        final = self.shocked[-1]
        residual = (self.shocked[..., 0] - self.shocked[..., 1:].sum(dim=-1)).abs() / self.shocked[..., 0].abs()
        frame = pd.DataFrame(
            {
                "ticker": self.meta["tickers"],
                "company": self.meta["companies"],
                "sector": self.meta["sectors"],
                "equity_last": equity_last.numpy(),
                "equity_base": equity_base.numpy(),
                "equity_shock": equity_shock.numpy(),
                "equity_change_pct": ((equity_shock - equity_base) / equity_base.abs() * 100).numpy(),
                "equity_drop_vs_last_pct": ((equity_last - equity_shock) / equity_last.abs() * 100).numpy(),
                **{f"{name}_shock": final[:, index].numpy() for index, name in enumerate(ACCOUNTS[:3])},
                "balance_residual": residual.max(dim=0).values.numpy(),
            }
        )
        frame = frame.sort_values("equity_change_pct").reset_index(drop=True)
        frame.insert(0, "rank", range(1, len(frame) + 1))
        return frame


@torch.no_grad()
def simulate(shocks: dict | None = None, horizon: int = 4, variant: str = "sector") -> Scenario:
    shock = Shock(**(shocks or {}))
    snapshots, meta = load_data()
    model = load_model(variant, meta)
    last = snapshots[-1]
    edge_index, edge_attr = edges_of(last, variant)

    state = model.initial_state(last.x.shape[0])
    for snapshot in snapshots[:-1]:
        past_edges, past_attr = edges_of(snapshot, variant)
        state = model(snapshot.x, past_edges, state, past_attr).h

    std = torch.tensor(meta["macro_std"], dtype=torch.float32)
    mean = torch.tensor(meta["macro_mean"], dtype=torch.float32)
    last_macro = last.x[0, N_BALANCE:] * std + mean
    natural = torch.tensor([shock.SELIC_delta, shock.IPCA_delta, float(last_macro[2]) * shock.PTAX_delta_pct / 100])
    scale = torch.tensor(meta["scale"], dtype=torch.float64).unsqueeze(1)

    def path(macro_shift: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        x, h, quarters, attention = last.x.clone(), state, [], None
        x[:, N_BALANCE:] += macro_shift
        for quarter in range(horizon):
            step = model(x, edge_index, h, edge_attr)
            check_balance(step.prediction)
            if quarter == 0:
                attention = (step.attention[0], step.attention[1].mean(dim=1))
            quarters.append(step.prediction.double() * scale)
            x = torch.cat([step.prediction, x[:, N_BALANCE:]], dim=1)
            h = step.h
        return torch.stack(quarters), attention

    base, _ = path(torch.zeros(3))
    shocked, attention = path(natural / std)
    filed = last.x[:, :N_BALANCE].double() * scale
    return Scenario(shock, horizon, variant, meta, filed, base, shocked, attention)


def run_stress_test(shocks: dict | None = None, horizon: int = 4, variant: str = "sector") -> pd.DataFrame:
    """The ranked table of all companies, from highest distress to lowest."""
    return simulate(shocks, horizon, variant).table()


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--selic", type=float, default=0.0, help="percentage points")
    parser.add_argument("--ipca", type=float, default=0.0, help="percentage points of the quarterly change")
    parser.add_argument("--ptax-pct", type=float, default=0.0, help="percent")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--variant", default="sector", choices=["sector", "typed"])
    args = parser.parse_args()
    shocks = {"SELIC_delta": args.selic, "IPCA_delta": args.ipca, "PTAX_delta_pct": args.ptax_pct}
    table = run_stress_test(shocks, args.horizon, args.variant)
    RESULTS_DIR.mkdir(exist_ok=True)
    label = f"selic{args.selic:+g}_ipca{args.ipca:+g}_ptax{args.ptax_pct:+g}_{args.variant}"
    table.to_csv(RESULTS_DIR / f"stress_{label}.csv", index=False)
    shown = ["rank", "ticker", "sector", "equity_last", "equity_base", "equity_shock", "equity_change_pct"]
    print(f"stress test {shocks}, horizon {args.horizon} quarters, model {args.variant}")
    print(table[shown].head(15).to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"largest balance residual / assets: {table['balance_residual'].max():.2e}")
    return table


if __name__ == "__main__":
    main()
