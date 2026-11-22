"""Load the graph snapshots and choose the edge set of a model variant."""
from __future__ import annotations

import json

import torch

from src.data_ingestion.download import DATA_DIR
from src.model.engine import EnterpriseContagionEngine
from src.protocol import check_shape

MODULE = "src/model/dataset.py"
VARIANTS = ["sector", "no_edges", "typed"]
N_BALANCE = 4
CONFIG = {"input_features": 7, "hidden_dim": 64, "latent_dim": 32, "output_features": N_BALANCE, "gat_heads": 4}
CHECKPOINTS = {"sector": "contagion_engine.pt", "typed": "contagion_engine_typed.pt"}


def load_data() -> tuple[list, dict]:
    snapshots = torch.load(DATA_DIR / "pyg_snapshots.pt", weights_only=False)
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    check_shape(MODULE, (len(meta["tickers"]), CONFIG["input_features"]), tuple(snapshots[0].x.shape))
    return snapshots, meta


def split_steps(meta: dict) -> tuple[list[int], list[int]]:
    """Step t uses quarter t as input and quarter t + 1 as target."""
    quarters = meta["quarters"]
    steps = range(len(quarters) - 1)
    train = [step for step in steps if quarters[step + 1] <= meta["train_end"]]
    test = [step for step in steps if quarters[step + 1] > meta["train_end"]]
    return train, test


def edges_of(snapshot, variant: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    if variant == "sector":
        return snapshot.edge_index, None
    if variant == "no_edges":
        return torch.empty((2, 0), dtype=torch.long), None
    return snapshot.typed_edge_index, snapshot.typed_edge_attr


def build_model(variant: str, n_relations: int, n_nodes: int) -> EnterpriseContagionEngine:
    edge_dim = n_relations if variant == "typed" else None
    return EnterpriseContagionEngine(**CONFIG, edge_dim=edge_dim, n_nodes=n_nodes)


def load_model(variant: str, meta: dict) -> EnterpriseContagionEngine:
    """The trained model of a variant, in evaluation mode. The checkpoint holds the trend."""
    checkpoint = torch.load(DATA_DIR / CHECKPOINTS[variant], weights_only=False)
    model = build_model(variant, len(meta["relations"]), len(meta["tickers"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
