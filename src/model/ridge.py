"""Seasonal ridge correction to the "mean change" forecast.

prediction(t + 1) = balance(t) + P(trend) + P(change_scale * correction(t))

    trend         mean quarterly change of each company and account in the train steps
    change_scale  standard deviation of the same changes (floor 0.01)
    correction    for each account a, a pooled ridge regression over all companies:
                  correction = w_a[0] * lag1 + w_a[1] * lag4
    lag1          surprise of the last change: (balance(t) - balance(t-1) - trend) / change_scale
    lag4          surprise of the change into the same season 1 year back:
                  (balance(t-3) - balance(t-4) - trend) / change_scale
    P             the accounting projection, so each prediction balances

Each feature and the regression target are clipped at +-3 standard units, so 1 merger or
capital raise cannot own the weights, and the correction stays bounded. The ridge
penalty of each account comes from an expanding-window validation inside the train steps.
The model is deterministic and has no macro input: it gives the base path, not the shock.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


CLIP = 3.0
ALPHAS = (0.0, 10.0, 100.0, 1e3, 1e4, 1e5)
FIRST_STEP = 4  # lag4 needs 4 earlier quarters
FOLD_START = 16
FOLD_SIZE = 4
MIN_CHANGE_SCALE = 0.01
# Raise it when the features or the fit change, so an old ridge.pt is refused.
VERSION = 1


@dataclass
class SeasonalRidge:
    trend: torch.Tensor | None = None  # (N, 4)
    scale: torch.Tensor | None = None  # (N, 4)
    weights: torch.Tensor | None = None  # (4, 2): [lag1, lag4] for each account
    alphas: list = field(default_factory=list)
    cv_ratio: float | None = None  # validation error inside train / error of a zero correction

    def __post_init__(self):
        # The accounting projection, built in float64 (a float32 build leaves 3e-7 in C @ P).
        constraint = torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float64)
        self.P = torch.eye(4, dtype=torch.float64) - torch.linalg.pinv(constraint) @ constraint

    def _standardized(self, change: torch.Tensor) -> torch.Tensor:
        return ((change - self.trend) / self.scale).clamp(-CLIP, CLIP)

    def features(self, levels: torch.Tensor, t: int) -> torch.Tensor:
        """Features of step t from levels[0 .. t] only, shape (N, 4, 2)."""
        zero = torch.zeros_like(levels[0])
        lag1 = self._standardized(levels[t] - levels[t - 1]) if t >= 1 else zero
        lag4 = self._standardized(levels[t - 3] - levels[t - 4]) if t >= 4 else zero
        return torch.stack([lag1, lag4], dim=-1)

    def fit(self, levels: torch.Tensor, train_steps: list[int]) -> "SeasonalRidge":
        """levels: balance of each quarter, shape (Q, N, 4), scaled units. Step t: levels[t] -> levels[t + 1]."""
        levels = levels.double()
        changes = levels[1:] - levels[:-1]
        train = changes[train_steps]
        self.trend = train.mean(dim=0)
        self.scale = train.std(dim=0).clamp_min(MIN_CHANGE_SCALE)
        residual = (changes - self.trend) / self.scale
        steps = [t for t in train_steps if t >= FIRST_STEP]
        X = torch.stack([self.features(levels, t) for t in steps])  # (S, N, 4, 2)
        y_fit = residual[steps].clamp(-CLIP, CLIP)  # robust target for the weights
        y_raw = residual[steps]  # the validation inside train scores the true residual
        position = {t: i for i, t in enumerate(steps)}
        last = steps[-1]
        folds = [s for s in range(FOLD_START, last - FOLD_SIZE + 2, FOLD_SIZE) if s in position]
        weights, alphas, cv_model, cv_zero = [], [], 0.0, 0.0
        for a in range(4):
            Xa, ya, ra = X[:, :, a, :], y_fit[:, :, a], y_raw[:, :, a]
            errors = {alpha: 0.0 for alpha in ALPHAS}
            zero = 0.0
            for s in folds:
                tr = [position[t] for t in steps if t < s]
                te = [position[t] for t in steps if s <= t < s + FOLD_SIZE]
                Xtr, ytr = Xa[tr].reshape(-1, 2), ya[tr].reshape(-1)
                Xte, yte = Xa[te].reshape(-1, 2), ra[te].reshape(-1)
                zero += float((yte**2).sum())
                for alpha in ALPHAS:
                    errors[alpha] += float(((Xte @ _ridge(Xtr, ytr, alpha) - yte) ** 2).sum())
            alpha = min(errors, key=errors.get)
            weights.append(_ridge(Xa.reshape(-1, 2), ya.reshape(-1), alpha))
            alphas.append(alpha)
            cv_model += errors[alpha]
            cv_zero += zero
        self.weights = torch.stack(weights)
        self.alphas = alphas
        self.cv_ratio = cv_model / cv_zero if cv_zero else None
        return self

    def correction(self, levels: torch.Tensor, t: int) -> torch.Tensor:
        """Correction in standardized units, shape (N, 4)."""
        return (self.features(levels.double(), t) * self.weights[None, :, :]).sum(dim=-1)

    def predict(self, levels: torch.Tensor, t: int) -> torch.Tensor:
        """Balance of quarter t + 1 from levels[0 .. t], shape (N, 4), float64. It balances."""
        levels = levels.double()
        change = self.trend + self.scale * self.correction(levels, t)
        return levels[t] + change @ self.P.T

    def rollout(self, levels: torch.Tensor, horizon: int) -> torch.Tensor:
        """Forecast `horizon` quarters after the last row of `levels`, feeding each forecast back."""
        path = list(levels.double())
        for _ in range(horizon):
            path.append(self.predict(torch.stack(path), len(path) - 1))
        return torch.stack(path[len(levels):])

    def state_dict(self) -> dict:
        return {
            "version": VERSION, "trend": self.trend, "scale": self.scale, "weights": self.weights,
            "alphas": self.alphas, "cv_ratio": self.cv_ratio,
        }

    @classmethod
    def from_state(cls, state: dict) -> "SeasonalRidge":
        return cls(state["trend"], state["scale"], state["weights"], state["alphas"], state["cv_ratio"])


def _ridge(X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.linalg.solve(X.T @ X + alpha * torch.eye(X.shape[1], dtype=X.dtype), X.T @ y)
