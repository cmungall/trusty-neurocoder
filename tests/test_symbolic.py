import pytest
import torch

from trusty_neurocoder import (
    best_candidate_1d,
    best_candidate_2d,
    moisture_response_candidates,
    temperature_response_candidates,
)


def test_moisture_candidates_recover_hill_family():
    x_data = torch.linspace(0.05, 1.2, 200)
    y_data = x_data**0.7 / (0.3 + x_data**0.7)
    param_grid = torch.linspace(0.05, 1.0, 191)

    best = best_candidate_1d(
        moisture_response_candidates(exponents=(0.5, 0.7, 1.5)),
        x_data,
        y_data,
        param_grid,
    )

    assert "n=0.7" in best.name
    assert best.params[0] == pytest.approx(0.3, abs=0.02)
    assert best.loss < 1e-6


def test_temperature_candidates_recover_q10_family():
    temperatures = torch.linspace(0.0, 35.0, 200)
    y_data = 2.0 ** ((temperatures - 15.0) / 10.0)
    q10_grid = torch.tensor([1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0])
    tref_grid = torch.tensor([5.0, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0])

    best = best_candidate_2d(
        temperature_response_candidates(),
        temperatures,
        y_data,
        q10_grid,
        tref_grid,
    )

    assert best.name.startswith("Q10 family")
    assert best.params[0] == pytest.approx(2.0, abs=0.05)
    assert best.params[1] == pytest.approx(15.0, abs=0.3)
    assert best.loss < 1e-6
