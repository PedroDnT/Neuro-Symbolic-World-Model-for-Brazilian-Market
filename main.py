"""Train and test the 2 engines of the contagion model.

Train: targets through 2021-Q4 (44 steps for each epoch). Test: targets 2022-Q1 to 2024-Q4.

Neural engine (the graph recurrent model of the work order), 3 variants:
    sector    edges between companies of the same sector
    no_edges  the same model with no edge, to measure what the graph adds
    typed     sector, ownership, related-party and supplier links known by the train end

Hybrid engine, base path: the seasonal ridge correction (src/model/ridge.py).

Baselines on the same test targets:
    no_change    next quarter = this quarter
    mean_change  next quarter = this quarter + mean change of the company in the train split

Both engines start from the mean_change forecast and learn a correction to it.

Training recipe of the neural engine. It comes from experiments on a validation split
(fit to 2019-Q4, validate on 2020-2021), never from the test split. The last decoder layer
starts at zero, so training starts at the mean_change forecast. The learning rate is low
(0.00005): with 0.005 the weights oscillated and the test error changed by 40 % between 2
consecutive epochs. 143 epochs = 6,300 optimizer steps, the step count that the validation
chose; the result was flat from 40 to 200 epochs.

Outputs: data/contagion_engine.pt (sector), data/contagion_engine_typed.pt, data/ridge.pt,
         results/metrics.json. Each output stores the data fingerprint of data/meta.json.
Run:     python main.py [--epochs 143] [--seeds 0 1 2] [--variants sector no_edges typed]
"""
from __future__ import annotations

import argparse
import json

import torch

from src.data_ingestion.download import DATA_DIR, RESULTS_DIR
from src.model.dataset import (
    ARCH,
    CHECKPOINTS,
    CONFIG,
    N_BALANCE,
    RIDGE_CHECKPOINT,
    VARIANTS,
    balance_levels,
    build_model,
    edges_of,
    load_data,
    split_steps,
)
from src.model.loss import BETA_KL, elbo_loss
from src.model.ridge import SeasonalRidge
from src.protocol import BALANCE_THRESHOLD, check_balance, check_loss, status_line

LEARNING_RATE = 0.00005
EPOCHS = 143
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
    # Start at the mean_change forecast: the correction is exactly 0 before training.
    torch.nn.init.zeros_(model.decoder[-1].weight)
    torch.nn.init.zeros_(model.decoder[-1].bias)
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
def predict_steps(model, snapshots: list, train_steps: list[int], test_steps: list[int], variant: str) -> tuple:
    """1-step forecasts with frozen weights, for the train and the test targets. The state
    runs over the steps in order, so history warms it before the test steps."""
    model.eval()
    h = model.initial_state(snapshots[0].x.shape[0])
    train, test = [], []
    for step_index in train_steps + test_steps:
        snapshot = snapshots[step_index]
        edge_index, edge_attr = edges_of(snapshot, variant)
        step = model(snapshot.x, edge_index, h, edge_attr)
        h = step.h
        check_balance(step.prediction)
        (test if step_index in test_steps else train).append(step.prediction)
    return torch.stack(train), torch.stack(test)


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


def provenance(meta: dict, args, **extra) -> dict:
    return {
        "fingerprint": meta["fingerprint"], "arch": ARCH, "epochs": args.epochs, "seeds": args.seeds,
        "learning_rate": LEARNING_RATE, **extra,
    }


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()

    snapshots, meta = load_data()
    train_steps, test_steps = split_steps(meta)
    n_relations = len(meta["relations"])

    def targets(steps: list[int]) -> torch.Tensor:
        return torch.stack([snapshots[step].y for step in steps])

    def inputs(steps: list[int]) -> torch.Tensor:
        return torch.stack([snapshots[step].x[:, :N_BALANCE] for step in steps])

    target, target_train = targets(test_steps), targets(train_steps)
    current, current_train = inputs(test_steps), inputs(train_steps)
    mean_change = (target_train - current_train).mean(dim=0)
    scale = change_scale(snapshots, train_steps)

    ridge = SeasonalRidge().fit(balance_levels(snapshots), train_steps)
    levels = balance_levels(snapshots)
    ridge_test = torch.stack([ridge.predict(levels[: t + 1], t) for t in test_steps]).float()
    ridge_train = torch.stack([ridge.predict(levels[: t + 1], t) for t in train_steps]).float()
    check_balance(ridge_test.reshape(-1, N_BALANCE))
    torch.save({**ridge.state_dict(), "fingerprint": meta["fingerprint"], "tickers": meta["tickers"]}, DATA_DIR / RIDGE_CHECKPOINT)

    metrics = {
        "fingerprint": meta["fingerprint"],
        "config": CONFIG,
        "train_steps": len(train_steps),
        "test_steps": len(test_steps),
        "test_quarters": [meta["quarters"][step + 1] for step in test_steps],
        "directed_edges": meta["directed_edges"],
        "baselines": {
            "no_change": {**errors(current, target, scale), "train_mse_std": errors(current_train, target_train, scale)["mse_std"]},
            "mean_change": {
                **errors(current + mean_change, target, scale),
                "train_mse_std": errors(current_train + mean_change, target_train, scale)["mse_std"],
            },
        },
        "ridge": {
            **errors(ridge_test, target, scale),
            "train_mse_std": errors(ridge_train, target_train, scale)["mse_std"],
            "alphas": ridge.alphas,
            "weights": ridge.weights.tolist(),
            "inside_train_validation_ratio": ridge.cv_ratio,
        },
        "models": {},
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        # Keep a variant that this run does not train again only when it was measured on the
        # same data with the same architecture. Its own epochs, seeds and learning rate are
        # stored in its entry.
        old = json.loads(metrics_path.read_text()).get("models", {})
        metrics["models"] = {
            name: entry for name, entry in old.items()
            if entry.get("fingerprint") == meta["fingerprint"] and entry.get("arch") == ARCH
        }
    for variant in args.variants:
        runs, responses = [], []
        for seed in args.seeds:
            log = variant == "sector" and seed == args.seeds[0]
            model = train(snapshots, train_steps, variant, args.epochs, seed, n_relations, scale, mean_change, log)
            predicted_train, predicted_test = predict_steps(model, snapshots, train_steps, test_steps, variant)
            runs.append({**errors(predicted_test, target, scale), "train_mse_std": errors(predicted_train, target_train, scale)["mse_std"]})
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
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "config": CONFIG,
                    "arch": ARCH,
                    "variant": variant,
                    "seed": seed,
                    "epochs": args.epochs,
                    "learning_rate": LEARNING_RATE,
                    "fingerprint": meta["fingerprint"],
                    "tickers": meta["tickers"],
                }
                torch.save(checkpoint, DATA_DIR / CHECKPOINTS[variant])
            print(
                f"trained {variant} seed {seed}: test standardized MSE {runs[-1]['mse_std']:.4f}, "
                f"train (frozen) {runs[-1]['train_mse_std']:.4f}, macro response {responses[-1]:.4f}",
                flush=True,
            )

        def mean_of(key: str) -> float:
            return float(torch.tensor([run[key] for run in runs]).mean())

        metrics["models"][variant] = {
            **provenance(meta, args),
            "mse": mean_of("mse"),
            "mse_std": mean_of("mse_std"),
            "mae": mean_of("mae"),
            "mae_std": mean_of("mae_std"),
            "median_abs_std": mean_of("median_abs_std"),
            "train_mse_std": mean_of("train_mse_std"),
            "mse_std_by_seed": [run["mse_std"] for run in runs],
            "mse_std_spread": float(torch.tensor([run["mse_std"] for run in runs]).std(unbiased=False)),
            "mse_std_by_quarter_first_seed": runs[0]["mse_std_by_quarter"],
            "macro_response_by_seed": responses,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=1))
    print("\ntest error, 2022-Q1 to 2024-Q4 (lower is better). Standardized: error / usual change of the company")
    rows = [("baseline", name, result) for name, result in metrics["baselines"].items()]
    rows += [("hybrid", "ridge", metrics["ridge"])]
    rows += [("neural", name, result) for name, result in metrics["models"].items()]
    for kind, name, result in rows:
        print(
            f"  {kind:8s} {name:12s} MSE {result['mse_std']:.4f}  MAE {result['mae_std']:.4f}  "
            f"median {result['median_abs_std']:.4f}  | train MSE {result['train_mse_std']:.4f}"
        )
    return metrics


if __name__ == "__main__":
    main()
