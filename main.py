"""Train and test the contagion engine.

Train: targets through 2021-Q4 (44 steps for each epoch). Test: targets 2022-Q1 to 2024-Q4.

Variants:
    sector    edges between companies of the same sector (the main model)
    no_edges  the same model with no edge, to measure what the graph adds
    typed     sector, ownership, related-party and supplier edges with the relation type

Baselines on the same test targets:
    no_change    next quarter = this quarter
    mean_change  next quarter = this quarter + mean change of the company in the train split

The model starts from the mean_change forecast and learns a correction to it. So the
mean_change baseline is the model with a zero correction: the model must beat it.

Outputs: data/contagion_engine.pt (main model), data/contagion_engine_typed.pt,
         results/metrics.json
Run:     python main.py [--epochs 300] [--seeds 0 1 2] [--variants sector no_edges typed]
"""
from __future__ import annotations

import argparse
import json

import torch

from src.data_ingestion.download import DATA_DIR, RESULTS_DIR
from src.model.dataset import CHECKPOINTS, CONFIG, N_BALANCE, VARIANTS, build_model, edges_of, load_data, split_steps
from src.model.loss import BETA_KL, elbo_loss
from src.protocol import BALANCE_THRESHOLD, check_balance, check_loss, status_line

LEARNING_RATE = 0.005
MAX_GRAD_NORM = 1.0
# Floor of the change scale, in scaled balance units (1 = mean total assets in the train split).
MIN_CHANGE_SCALE = 0.01


def change_scale(snapshots: list, train_steps: list[int]) -> torch.Tensor:
    """Usual size of a quarterly change of each company and account, shape (N, 4), from the train split."""
    changes = torch.stack([snapshots[step].y - snapshots[step].x[:, :N_BALANCE] for step in train_steps])
    return changes.std(dim=0).clamp_min(MIN_CHANGE_SCALE)


def train(
    snapshots: list,
    steps: list[int],
    variant: str,
    epochs: int,
    seed: int,
    n_relations: int,
    scale: torch.Tensor,
    trend: torch.Tensor,
    log: bool,
):
    torch.manual_seed(seed)
    model = build_model(variant, n_relations, snapshots[0].x.shape[0])
    model.set_trend(trend)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_nodes = snapshots[0].x.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        h = model.initial_state(n_nodes)
        worst, total = 0.0, 0.0
        for step_index in steps:
            snapshot = snapshots[step_index]
            edge_index, edge_attr = edges_of(snapshot, variant)
            optimizer.zero_grad()
            step = model(snapshot.x, edge_index, h, edge_attr)
            loss, _, _ = elbo_loss(step.prediction, snapshot.y, step.posterior, step.prior, BETA_KL, scale)
            check_loss(loss.item())
            prediction = step.prediction.detach()
            check_balance(prediction)  # raises ERR_ACCOUNTING_BALANCE on a violation
            worst = max(worst, float((prediction[:, 0] - prediction[:, 1:].sum(dim=1)).abs().max()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            h = step.h.detach()
            total += loss.item()
        if log:
            print(
                status_line(
                    "OK",
                    "PASS_ACCOUNTING_BALANCE",
                    "src/model/components.py",
                    max_residual=f"{worst:.8e}",
                    threshold=f"{BALANCE_THRESHOLD:.8e}",
                    epoch=epoch,
                    loss=f"{total / len(steps):.6f}",
                ),
                flush=True,
            )
    return model


@torch.no_grad()
def predict_test(model, snapshots: list, train_steps: list[int], test_steps: list[int], variant: str) -> torch.Tensor:
    """1-step forecasts for the test targets, shape (test steps, N, 4). History warms the state."""
    model.eval()
    h = model.initial_state(snapshots[0].x.shape[0])
    predictions = []
    for step_index in train_steps + test_steps:
        snapshot = snapshots[step_index]
        edge_index, edge_attr = edges_of(snapshot, variant)
        step = model(snapshot.x, edge_index, h, edge_attr)
        h = step.h
        if step_index in test_steps:
            check_balance(step.prediction)
            predictions.append(step.prediction)
    return torch.stack(predictions)


@torch.no_grad()
def macro_response(model, snapshot, variant: str) -> float:
    """Largest change of the prediction when SELIC moves by 1 standard deviation. 0 means no reaction."""
    model.eval()
    edge_index, edge_attr = edges_of(snapshot, variant)
    h = model.initial_state(snapshot.x.shape[0])
    moved = snapshot.x.clone()
    moved[:, N_BALANCE] += 1.0
    base = model(snapshot.x, edge_index, h, edge_attr).prediction
    return float((model(moved, edge_index, h, edge_attr).prediction - base).abs().max())


def errors(prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> dict:
    """`mse` is in scaled balance units. `mse_std` divides each error by the usual change of
    the company and account first, so 1 large company in distress cannot own the mean."""
    error = prediction - target
    return {
        "mse": float((error**2).mean()),
        "mse_std": float(((error / scale) ** 2).mean()),
        "mae": float(error.abs().mean()),
        "mae_std": float((error / scale).abs().mean()),
        "median_abs_std": float((error / scale).abs().median()),
        "mse_std_by_quarter": [float(value) for value in ((error / scale) ** 2).mean(dim=(1, 2))],
    }


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()

    snapshots, meta = load_data()
    train_steps, test_steps = split_steps(meta)
    n_relations = len(meta["relations"])
    target = torch.stack([snapshots[step].y for step in test_steps])
    current = torch.stack([snapshots[step].x[:, :N_BALANCE] for step in test_steps])
    mean_change = torch.stack([snapshots[s].y - snapshots[s].x[:, :N_BALANCE] for s in train_steps]).mean(dim=0)
    scale = change_scale(snapshots, train_steps)

    metrics = {
        "epochs": args.epochs,
        "seeds": args.seeds,
        "config": CONFIG,
        "train_steps": len(train_steps),
        "test_steps": len(test_steps),
        "test_quarters": [meta["quarters"][step + 1] for step in test_steps],
        "directed_edges": meta["directed_edges"],
        "baselines": {
            "no_change": errors(current, target, scale),
            "mean_change": errors(current + mean_change, target, scale),
        },
        "models": {},
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():  # keep the variants that this run does not train again
        metrics["models"] = json.loads(metrics_path.read_text()).get("models", {})
    for variant in args.variants:
        runs, responses = [], []
        for seed in args.seeds:
            log = variant == "sector" and seed == args.seeds[0]
            model = train(snapshots, train_steps, variant, args.epochs, seed, n_relations, scale, mean_change, log)
            runs.append(errors(predict_test(model, snapshots, train_steps, test_steps, variant), target, scale))
            responses.append(macro_response(model, snapshots[-1], variant))
            if responses[-1] == 0.0:
                print(
                    status_line(
                        "WARN",
                        "WARN_NO_MACRO_RESPONSE",
                        "main.py",
                        expected="prediction changes when SELIC changes",
                        actual="0.0",
                        action="Inspect the decoder inputs: the model ignores the macro columns.",
                    )
                )
            if seed == args.seeds[0] and variant in CHECKPOINTS:
                checkpoint = {"state_dict": model.state_dict(), "config": CONFIG, "variant": variant, "seed": seed}
                torch.save(checkpoint, DATA_DIR / CHECKPOINTS[variant])
            print(
                f"trained {variant} seed {seed}: test MSE {runs[-1]['mse']:.6f}, "
                f"standardized {runs[-1]['mse_std']:.4f}, macro response {responses[-1]:.4f}",
                flush=True,
            )

        def mean_of(key: str) -> float:
            return float(torch.tensor([run[key] for run in runs]).mean())

        metrics["models"][variant] = {
            "mse": mean_of("mse"),
            "mse_std": mean_of("mse_std"),
            "mae": mean_of("mae"),
            "mae_std": mean_of("mae_std"),
            "median_abs_std": mean_of("median_abs_std"),
            "mse_std_by_seed": [run["mse_std"] for run in runs],
            "mse_std_spread": float(torch.tensor([run["mse_std"] for run in runs]).std(unbiased=False)),
            "mse_std_by_quarter_first_seed": runs[0]["mse_std_by_quarter"],
            "macro_response_by_seed": responses,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=1))
    print("\ntest error (lower is better). standardized: error divided by the usual change of the company")
    for kind, group in (("baseline", metrics["baselines"]), ("model", metrics["models"])):
        for name, result in group.items():
            print(
                f"  {kind:8s} {name:12s} standardized: MSE {result['mse_std']:.4f}, MAE {result['mae_std']:.4f}, "
                f"median abs {result['median_abs_std']:.4f} | scaled units: MSE {result['mse']:.6f}"
            )
    return metrics


if __name__ == "__main__":
    main()
