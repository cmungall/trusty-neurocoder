"""Reusable symbolic fitting utilities for Trusty Neurocoder demos.

The current prototype uses a lightweight form of symbolic regression:
sample a learned function densely, fit a small family of candidate
expressions by grid search, and keep the lowest-error explanation.

This module turns that pattern into a reusable package API so the logic
does not live only inside notebooks and example scripts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

type ParametricFn1D = Callable[[torch.Tensor, float], torch.Tensor]
type ParametricFn2D = Callable[[torch.Tensor, float, float], torch.Tensor]


@dataclass(frozen=True)
class Candidate1D:
    """A one-parameter symbolic family to fit against sampled data."""

    name: str
    func: ParametricFn1D
    expression: str | None = None


@dataclass(frozen=True)
class Candidate2D:
    """A two-parameter symbolic family to fit against sampled data."""

    name: str
    func: ParametricFn2D
    expression: str | None = None


@dataclass(frozen=True)
class SymbolicFit:
    """Best-fit parameters and reconstruction error for one candidate."""

    name: str
    loss: float
    params: tuple[float, ...]
    expression: str | None = None


def fit_candidate_1d(
    func: ParametricFn1D,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    param_grid: Iterable[float | torch.Tensor],
) -> SymbolicFit:
    """Fit a one-parameter candidate family by grid search."""

    best_loss = float("inf")
    best_param: float | None = None

    with torch.no_grad():
        for raw_param in param_grid:
            param = _as_float(raw_param)
            try:
                predicted = func(x_data, param)
                loss = _mse(predicted, y_data)
            except Exception:
                continue
            if loss < best_loss:
                best_loss = loss
                best_param = param

    if best_param is None:
        raise ValueError("No valid fit found for one-parameter candidate")

    return SymbolicFit(name="", loss=best_loss, params=(best_param,))


def fit_candidate_2d(
    func: ParametricFn2D,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    grid_a: Iterable[float | torch.Tensor],
    grid_b: Iterable[float | torch.Tensor],
) -> SymbolicFit:
    """Fit a two-parameter candidate family by grid search."""

    best_loss = float("inf")
    best_params: tuple[float, float] | None = None
    grid_b_values = tuple(_as_float(value) for value in grid_b)

    with torch.no_grad():
        for raw_a in grid_a:
            param_a = _as_float(raw_a)
            for param_b in grid_b_values:
                try:
                    predicted = func(x_data, param_a, param_b)
                    loss = _mse(predicted, y_data)
                except Exception:
                    continue
                if loss < best_loss:
                    best_loss = loss
                    best_params = (param_a, param_b)

    if best_params is None:
        raise ValueError("No valid fit found for two-parameter candidate")

    return SymbolicFit(name="", loss=best_loss, params=best_params)


def rank_candidates_1d(
    candidates: Sequence[Candidate1D],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    param_grid: Iterable[float | torch.Tensor],
) -> list[SymbolicFit]:
    """Return candidate fits sorted from lowest to highest loss."""

    grid = tuple(_as_float(value) for value in param_grid)
    fits = []
    for candidate in candidates:
        fit = fit_candidate_1d(candidate.func, x_data, y_data, grid)
        fits.append(
            SymbolicFit(
                name=candidate.name,
                loss=fit.loss,
                params=fit.params,
                expression=candidate.expression,
            )
        )
    return sorted(fits, key=lambda fit: fit.loss)


def rank_candidates_2d(
    candidates: Sequence[Candidate2D],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    grid_a: Iterable[float | torch.Tensor],
    grid_b: Iterable[float | torch.Tensor],
) -> list[SymbolicFit]:
    """Return candidate fits sorted from lowest to highest loss."""

    values_a = tuple(_as_float(value) for value in grid_a)
    values_b = tuple(_as_float(value) for value in grid_b)
    fits = []
    for candidate in candidates:
        fit = fit_candidate_2d(candidate.func, x_data, y_data, values_a, values_b)
        fits.append(
            SymbolicFit(
                name=candidate.name,
                loss=fit.loss,
                params=fit.params,
                expression=candidate.expression,
            )
        )
    return sorted(fits, key=lambda fit: fit.loss)


def best_candidate_1d(
    candidates: Sequence[Candidate1D],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    param_grid: Iterable[float | torch.Tensor],
) -> SymbolicFit:
    """Convenience wrapper for the lowest-loss one-parameter fit."""

    return rank_candidates_1d(candidates, x_data, y_data, param_grid)[0]


def best_candidate_2d(
    candidates: Sequence[Candidate2D],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    grid_a: Iterable[float | torch.Tensor],
    grid_b: Iterable[float | torch.Tensor],
) -> SymbolicFit:
    """Convenience wrapper for the lowest-loss two-parameter fit."""

    return rank_candidates_2d(candidates, x_data, y_data, grid_a, grid_b)[0]


def moisture_response_candidates(
    exponents: Sequence[float] = (0.5, 0.7, 1.0, 1.5, 2.0),
) -> list[Candidate1D]:
    """Candidate families used by the soil-moisture demos."""

    candidates = [
        Candidate1D(
            name="linear: a*x",
            func=lambda x, a: a * x,
            expression="a*x",
        ),
        Candidate1D(
            name="Michaelis-Menten: x/(K+x)",
            func=lambda x, k: x / (k + x),
            expression="x/(K+x)",
        ),
    ]
    for exponent in exponents:
        candidates.append(
            Candidate1D(
                name=f"Hill (n={exponent:.1f}): x^{exponent:.1f}/(K+x^{exponent:.1f})",
                func=lambda x, k, exponent=exponent: x**exponent / (k + x**exponent),
                expression=f"x^{exponent:.1f}/(K+x^{exponent:.1f})",
            )
        )
    return candidates


def temperature_response_candidates() -> list[Candidate2D]:
    """Candidate families used by the temperature-response demos."""

    return [
        Candidate2D(
            name="Q10 family: Q10^((T-T_ref)/10)",
            func=lambda temperature, q10, t_ref: q10 ** ((temperature - t_ref) / 10.0),
            expression="Q10^((T-T_ref)/10)",
        ),
        Candidate2D(
            name="Arrhenius: exp(a*(T-b))",
            func=lambda temperature, a, b: torch.exp(a * (temperature - b)),
            expression="exp(a*(T-b))",
        ),
    ]


def _as_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _mse(predicted: torch.Tensor, observed: torch.Tensor) -> float:
    if predicted.shape != observed.shape:
        raise ValueError(
            f"Shape mismatch while fitting symbolic candidate: "
            f"{predicted.shape} vs {observed.shape}"
        )
    if not torch.isfinite(predicted).all():
        raise ValueError("Candidate produced non-finite values")
    return float(((predicted - observed) ** 2).mean().item())
