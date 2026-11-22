"""Build the quarterly graph snapshots.

Node features, shape (N, 7):
    0..3  Assets, Current Liabilities, Non-Current Liabilities, Equity. Each one is
          divided by the mean total assets of the company in the train split. One divisor
          for the 4 columns keeps Assets = Liabilities + Equity true in scaled units.
    4..6  SELIC, IPCA, PTAX as z-scores from the train split, the same for all nodes.
Target `y`: the 4 balance columns of the next quarter. The last quarter has no target.

Edges (`edge_index`): both directions between 2 companies of the same sector, no self-loops.
Typed edges (`typed_edge_index`, `typed_edge_attr`): sector, ownership, related party and
supplier links, with the relation as a one-hot vector.

Output: data/pyg_snapshots.pt (list of Data), data/meta.json
Run:    python -m src.graph_builder.build_pyg
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_cvm import MODEL_COLUMNS
from src.data_ingestion.fetch_fre import load_universe
from src.data_ingestion.universe import quarter_ends
from src.protocol import check_balance, check_shape

MODULE = "src/graph_builder/build_pyg.py"
TRAIN_END = "2021-12-31"
MACRO_COLUMNS = ["SELIC", "IPCA", "PTAX"]
RELATIONS = ["sector", "ownership", "related_party", "supplier"]
N_FEATURES = len(MODEL_COLUMNS) + len(MACRO_COLUMNS)


def balance_tensor(universe: pd.DataFrame, quarters: pd.DatetimeIndex) -> np.ndarray:
    """Filed balance values in reais, shape (T, N, 4), nodes in universe order."""
    statements = pd.read_csv(DATA_DIR / "cvm_quarterly.csv", parse_dates=["quarter"])
    statements = statements[statements["cd_cvm"].isin(universe["cd_cvm"]) & statements["quarter"].isin(quarters)]
    values = np.empty((len(quarters), len(universe), len(MODEL_COLUMNS)))
    for column_index, column in enumerate(MODEL_COLUMNS):
        wide = statements.pivot(index="quarter", columns="cd_cvm", values=column)
        values[:, :, column_index] = wide.reindex(index=quarters, columns=universe["cd_cvm"]).to_numpy()
    if np.isnan(values).any():
        raise ValueError("a universe company has an empty quarter in data/cvm_quarterly.csv")
    return values


def sector_edges(sectors: list[str]) -> list[tuple[int, int]]:
    return [(i, j) for i, a in enumerate(sectors) for j, b in enumerate(sectors) if i != j and a == b]


def link_edges(universe: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    """Ownership, related-party and supplier links as node pairs, in both directions."""
    position = {code: index for index, code in enumerate(universe["cd_cvm"])}
    pairs: dict[str, set[tuple[int, int]]] = {relation: set() for relation in RELATIONS[1:]}
    sources = [DATA_DIR / "fre_edges.csv", DATA_DIR / "extracted_edges.csv"]
    for path in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for relation, src, dst in zip(frame["relation"], frame["src_cd_cvm"], frame["dst_cd_cvm"]):
            if relation in pairs and src in position and dst in position and src != dst:
                pairs[relation].add((position[src], position[dst]))
                pairs[relation].add((position[dst], position[src]))
    return {relation: sorted(found) for relation, found in pairs.items()}


def main() -> list[Data]:
    universe = load_universe().sort_values("ticker").reset_index(drop=True)
    quarters = quarter_ends()
    n_nodes = len(universe)
    train = quarters <= pd.Timestamp(TRAIN_END)

    filed = balance_tensor(universe, quarters)
    scale = filed[train, :, 0].mean(axis=0)
    balance = filed / scale[None, :, None]

    macro = pd.read_csv(DATA_DIR / "macro_quarterly.csv", parse_dates=["quarter"], index_col="quarter")
    macro = macro.loc[quarters, MACRO_COLUMNS]
    macro_mean, macro_std = macro[train].mean(), macro[train].std(ddof=0)
    macro_z = ((macro - macro_mean) / macro_std).to_numpy()

    by_relation = {"sector": sector_edges(universe["sector"].tolist()), **link_edges(universe)}
    edge_index = torch.tensor(by_relation["sector"], dtype=torch.long).t().contiguous()
    typed_pairs = [(pair, kind) for kind, relation in enumerate(RELATIONS) for pair in by_relation[relation]]
    typed_edge_index = torch.tensor([pair for pair, _ in typed_pairs], dtype=torch.long).t().contiguous()
    typed_edge_attr = torch.nn.functional.one_hot(
        torch.tensor([kind for _, kind in typed_pairs]), num_classes=len(RELATIONS)
    ).float()

    snapshots = []
    for step, quarter in enumerate(quarters):
        features = np.concatenate([balance[step], np.tile(macro_z[step], (n_nodes, 1))], axis=1)
        x = torch.tensor(features, dtype=torch.float32)
        check_shape(MODULE, (n_nodes, N_FEATURES), tuple(x.shape))
        check_balance(x[:, : len(MODEL_COLUMNS)])
        snapshot = Data(
            x=x,
            edge_index=edge_index,
            typed_edge_index=typed_edge_index,
            typed_edge_attr=typed_edge_attr,
            quarter=quarter.strftime("%Y-%m-%d"),
        )
        if step + 1 < len(quarters):
            snapshot.y = torch.tensor(balance[step + 1], dtype=torch.float32)
        snapshots.append(snapshot)

    torch.save(snapshots, DATA_DIR / "pyg_snapshots.pt")
    meta = {
        "quarters": [quarter.strftime("%Y-%m-%d") for quarter in quarters],
        "train_end": TRAIN_END,
        "tickers": universe["ticker"].tolist(),
        "cd_cvm": universe["cd_cvm"].tolist(),
        "companies": universe["company"].tolist(),
        "sectors": universe["sector"].tolist(),
        "balance_columns": MODEL_COLUMNS,
        "macro_columns": MACRO_COLUMNS,
        "scale": scale.tolist(),
        "macro_mean": macro_mean.tolist(),
        "macro_std": macro_std.tolist(),
        "relations": RELATIONS,
        "directed_edges": {relation: len(pairs) for relation, pairs in by_relation.items()},
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=1))

    residual = np.abs(filed[..., 0] - filed[..., 1:].sum(axis=-1)) / filed[..., 0]
    isolated = n_nodes - len(set(edge_index[0].tolist()))
    print(f"pyg_snapshots.pt: {len(snapshots)} snapshots, x shape {tuple(snapshots[0].x.shape)}, "
          f"train quarters {int(train.sum())}, test targets {int((~train).sum())}")
    print(f"filed identity residual / assets: max {residual.max():.2e}")
    print(check_balance(snapshots[-1].x[:, : len(MODEL_COLUMNS)]))
    print(f"directed edges: {meta['directed_edges']} | nodes with no sector peer: {isolated}")
    print(f"scaled balance range: {balance.min():.2f} to {balance.max():.2f} | macro z range: {macro_z.min():.2f} to {macro_z.max():.2f}")
    return snapshots


if __name__ == "__main__":
    main()
