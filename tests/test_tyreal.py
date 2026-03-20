"""Tests for the TyReal extension to the Cajal type system."""

import torch
import pytest
from cajal.syntax import TyReal, TyBool, TyNat, TmIter, TmVar, TmApp
from cajal.compiling import compile, TypedTensor, dim, bases


def test_tyreal_dim():
    assert dim(TyReal(1)) == 1
    assert dim(TyReal(3)) == 3
    assert dim(TyReal(5)) == 5
    assert dim(TyReal(10)) == 10


def test_tyreal_bases():
    b = bases(TyReal(3))
    assert len(b) == 3
    for i, basis in enumerate(b):
        assert basis.ty == TyReal(3)
        assert basis.data[i] == 1.0
        for j in range(3):
            if j != i:
                assert basis.data[j] == 0.0


def test_tyreal_equality():
    assert TyReal(3) == TyReal(3)
    assert TyReal(1) != TyReal(2)
    assert TyReal(3) != TyBool()


def test_tyreal_iteration():
    """Test that Cajal iteration works with TyReal state."""
    program = TmIter(TmVar("s0"), "s", TmApp(TmVar("f"), TmVar("s")), TmVar("n"))
    compiled = compile(program)

    # Simple 0.9x scaling step
    def scale_step(state):
        return TypedTensor(0.9 * state.data, state.ty)

    s0 = TypedTensor(torch.tensor([1.0, 2.0, 3.0]), TyReal(3))

    for n_steps in range(10):
        n_onehot = torch.zeros(10)
        n_onehot[n_steps] = 1.0
        n_val = TypedTensor(n_onehot, TyNat())
        result = compiled({"s0": s0, "f": scale_step, "n": n_val})
        expected = (0.9 ** n_steps) * torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result.data, expected, atol=1e-6)
        assert result.ty == TyReal(3)


def test_tyreal_scalar():
    """TyReal(1) works as a scalar."""
    program = TmIter(TmVar("s0"), "s", TmApp(TmVar("f"), TmVar("s")), TmVar("n"))
    compiled = compile(program)

    def halve(state):
        return TypedTensor(0.5 * state.data, state.ty)

    s0 = TypedTensor(torch.tensor([8.0]), TyReal(1))
    n_onehot = torch.zeros(10)
    n_onehot[3] = 1.0  # 3 steps
    result = compiled({"s0": s0, "f": halve, "n": TypedTensor(n_onehot, TyNat())})
    assert torch.allclose(result.data, torch.tensor([1.0]), atol=1e-6)
