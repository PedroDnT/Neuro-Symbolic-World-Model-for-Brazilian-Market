"""Status lines and validation helpers.

Each check and each raised error is one pipe-delimited line:

    [STATUS] | CODE: <CODE> | MODULE: <PATH> | <KEY>: <VALUE> | ...

Float checks use a threshold, never equality, because float32 matrix
operations leave residuals near 1e-7.
"""
from __future__ import annotations

import math
import platform

BALANCE_THRESHOLD = 1e-5


def status_line(status: str, code: str, module: str, **fields: object) -> str:
    parts = [f"[{status}]", f"CODE: {code}", f"MODULE: {module}"]
    parts += [f"{key.upper()}: {value}" for key, value in fields.items()]
    return " | ".join(parts)


def env_check() -> str:
    """Return the PASS_ENV_SETUP line. Raise RuntimeError with ERR_ENV_SETUP on any failure."""
    try:
        major, minor = (int(part) for part in platform.python_version_tuple()[:2])
        if (major, minor) not in {(3, 11), (3, 12)}:
            raise RuntimeError(f"Python {platform.python_version()}")
        import bcb  # noqa: F401
        import torch
        import torch_geometric  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            status_line(
                "ERROR",
                "ERR_ENV_SETUP",
                ".venv",
                expected="Python 3.11-3.12 and valid torch, torch_geometric, bcb imports",
                actual=type(exc).__name__,
                action="Recreate .venv and reinstall dependencies from requirements.txt "
                "without C++ PyG extensions.",
            )
        ) from exc
    return status_line(
        "OK",
        "PASS_ENV_SETUP",
        ".venv",
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.cuda.is_available(),
    )


def check_balance(balance) -> str:
    """Check Assets = Current Liabilities + Non-Current Liabilities + Equity.

    `balance` is a tensor of shape (N, 4) in that column order. Return the PASS line.
    Raise ValueError with ERR_ACCOUNTING_BALANCE when the largest residual is above
    the threshold, or is not a number.
    """
    residual = balance[:, 0] - balance[:, 1:].sum(dim=1)
    max_err = float(residual.abs().max())
    if not max_err <= BALANCE_THRESHOLD:
        raise ValueError(
            status_line(
                "ERROR",
                "ERR_ACCOUNTING_BALANCE",
                "src/model/components.py",
                expected=f"residual <= {BALANCE_THRESHOLD:.8e}",
                actual=f"{max_err:.8e}",
                action="Halt training and inspect projection matrix P in AccountingProjectionLayer.",
            )
        )
    return status_line(
        "OK",
        "PASS_ACCOUNTING_BALANCE",
        "src/model/components.py",
        max_residual=f"{max_err:.8e}",
        threshold=f"{BALANCE_THRESHOLD:.8e}",
    )


def check_shape(module: str, expected: tuple, actual: tuple) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError(
            status_line(
                "ERROR",
                "ERR_SHAPE_MISMATCH",
                module,
                expected=tuple(expected),
                actual=tuple(actual),
                action="Verify node feature dimensions in build_pyg.py.",
            )
        )


def data_fetch_error(module: str, exc: BaseException) -> RuntimeError:
    """Build the ERR_DATA_FETCH error for a failed download. The caller raises it."""
    return RuntimeError(
        status_line(
            "ERROR",
            "ERR_DATA_FETCH",
            module,
            expected="HTTP 200 and non-empty DataFrame",
            actual=type(exc).__name__,
            action="Retry request with exponential backoff or load cached CSV from data/.",
        )
    )


def check_loss(loss_value: float) -> None:
    if not math.isfinite(loss_value):
        raise FloatingPointError(
            status_line(
                "ERROR",
                "ERR_NAN_LOSS",
                "main.py",
                expected="finite float32 loss",
                actual=loss_value,
                action="Halt training, clip gradients at max_norm=1.0, and reduce learning rate.",
            )
        )
