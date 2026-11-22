"""Build the quarterly graph snapshots.

Node features, shape (N, 7):
    0..3  Assets, Current Liabilities, Non-Current Liabilities, Equity. Each one is
          divided by the mean total assets of the company in the train split. One divisor
          for the 4 columns keeps Assets = Liabilities + Equity true in scaled units.
    4..6  SELIC, IPCA, PTAX as z-scores from the train split, the same for all nodes.
Target `y`: the 4 balance columns of the next quarter. The last quarter has no target.

Edges (`edge_index`): both directions between 2 companies of the same sector, no self-loops.
Typed edges (`typed_edge_index`, `typed_edge_attr`): sector, ownership, related party and
supplier links, with the relation as a one-hot vector. Only links known by the end of the
train split enter the model: ownership and related party from FRE 2020
(data/fre_edges_train.csv), and supplier links whose quote is in the form version received
by 2021-12-31 (`by_cutoff` in data/extracted_edges.csv).

meta.json holds a fingerprint of the snapshots. Each checkpoint and each metrics entry
stores it, so a stale checkpoint raises an error when it is loaded.

Output: data/pyg_snapshots.pt (list of Data), data/meta.json
Run:    python -m src.graph_builder.build_pyg
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data_ingestion.download import DATA_DIR
from src.data_ingestion.fetch_cvm import MODEL_COLUMNS
from src.data_ingestion.fetch_fre import load_universe
from src.data_ingestion.universe import quarter_ends
from src.protocol import check_balance, check_shape, status_line

MODULE = "src/graph_builder/build_pyg.py"
TRAIN_END = "2021-12-31"
MACRO_COLUMNS = ["SELIC", "IPCA", "PTAX"]
RELATIONS = ["sector", "ownership", "related_party", "supplier"]
N_FEATURES = len(MODEL_COLUMNS) + len(MACRO_COLUMNS)
MAX_JUMP = 20.0


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
    """Ownership, related-party and supplier links known by the train end, in both directions."""
    position = {code: index for index, code in enumerate(universe["cd_cvm"])}
    pairs: dict[str, set[tuple[int, int]]] = {relation: set() for relation in RELATIONS[1:]}
    for name in ("fre_edges_train.csv", "extracted_edges.csv"):
        path = DATA_DIR / name
        if not path.exists():
            print(status_line("WARN", "WARN_MISSING_EDGES", MODULE, expected=f"data/{name}", actual="missing",
                              action="Run the link builder and the extraction aggregate before this script."))
            continue
        frame = pd.read_csv(path)
        if "by_cutoff" in frame.columns:
            frame = frame[frame["by_cutoff"].astype(bool)]
        for relation, src, dst in zip(frame["relation"], frame["src_cd_cvm"], frame["dst_cd_cvm"]):
            if relation in pairs and src in position and dst in position and src != dst:
                pairs[relation].add((position[src], position[dst]))
                pairs[relation].add((position[dst], position[src]))
    return {relation: sorted(found) for relation, found in pairs.items()}


def check_continuity(filed: np.ndarray, tickers: list[str], quarters: pd.DatetimeIndex) -> None:
    """Raise when total assets of a company change more than MAX_JUMP times between 2 quarters."""
    ratio = filed[1:, :, 0] / filed[:-1, :, 0]
    bad = np.argwhere((ratio > MAX_JUMP) | (ratio < 1 / MAX_JUMP))
    if len(bad):
        cases = [f"{tickers[n]} {quarters[t + 1].date()} x{ratio[t, n]:.4g}" for t, n in bad[:10]]
        raise ValueError(status_line("ERROR", "ERR_DATA_CONTINUITY", MODULE, expected=f"asset ratio between 1/{MAX_JUMP:g} and {MAX_JUMP:g}",
                                     actual="; ".join(cases), action="Inspect these filings in data/cvm_quarterly.csv."))


def fingerprint(snapshots: list, tickers: list[str], quarters: list[str]) -> str:
    digest = hashlib.sha256(json.dumps([tickers, quarters]).encode())
    for snapshot in snapshots:
        for tensor in (snapshot.x, snapshot.edge_index, snapshot.typed_edge_index, snapshot.typed_edge_attr):
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:16]


def main() -> list[Data]:
    universe = load_universe().sort_values("ticker").reset_index(drop=True)
    quarters = quarter_ends()
    n_nodes = len(universe)
    train = quarters <= pd.Timestamp(TRAIN_END)

    filed = balance_tensor(universe, quarters)
    check_continuity(filed, universe["ticker"].tolist(), quarters)
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
        "fingerprint": fingerprint(snapshots, universe["ticker"].tolist(), [q.strftime("%Y-%m-%d") for q in quarters]),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=1))

    residual = np.abs(filed[..., 0] - filed[..., 1:].sum(axis=-1)) / filed[..., 0]
    isolated = n_nodes - len(set(edge_index[0].tolist()))
    print(f"pyg_snapshots.pt: {len(snapshots)} snapshots, x shape {tuple(snapshots[0].x.shape)}, "
          f"train quarters {int(train.sum())}, test targets {int((~train).sum())}")
    print(f"filed identity residual / assets: max {residual.max():.2e}")
    print(check_balance(snapshots[-1].x[:, : len(MODEL_COLUMNS)]))
    print(f"directed edges: {meta['directed_edges']} | nodes with no sector peer: {isolated} | fingerprint {meta['fingerprint']}")
    print(f"scaled balance range: {balance.min():.2f} to {balance.max():.2f} | macro z range: {macro_z.min():.2f} to {macro_z.max():.2f}")
    return snapshots


if __name__ == "__main__":
    main()
