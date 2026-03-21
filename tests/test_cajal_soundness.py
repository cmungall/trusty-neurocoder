"""
Regression tests for Cajal soundness bugs.

See docs/issues/cajal-soundness-and-runtime-semantics.md for details.
"""

import torch
import pytest
from cajal.syntax import *
from cajal.typing import check
from cajal.evaluating import evaluate
from cajal.compiling import TypedTensor, compile, device


class TestBug1_LambdaShadowing:
    """Lambda parameter shadowing must not silently discard outer linear bindings."""

    def test_shadowing_rejected(self):
        """A lambda whose parameter shadows an outer linear binding should fail
        if the outer binding is not otherwise consumed."""
        term = TmFun('x', TyBool(), TmVar('x'))
        with pytest.raises(TypeError, match="context remains"):
            check(term, {'x': TyNat()})

    def test_no_shadowing_still_works(self):
        """Lambda with no shadowing should still type-check normally."""
        term = TmFun('x', TyBool(), TmVar('x'))
        ty = check(term, {})
        assert ty == TyFun(TyBool(), TyBool())

    def test_different_names_ok(self):
        """Lambda with a different parameter name doesn't affect outer bindings."""
        # λy. x  (consumes outer x, introduces and consumes y... wait, y is unused)
        # Actually: λy. y should work with outer x only if x is consumed elsewhere
        # Let's test: λx. λy. y with no outer context
        inner = TmFun('y', TyBool(), TmVar('y'))
        outer = TmFun('x', TyNat(), inner)
        ty = check(outer, {})
        assert ty == TyFun(TyNat(), TyFun(TyBool(), TyBool()))


class TestBug2_IterTypeSafety:
    """TmIter must reject programs where base and step types disagree."""

    def test_mismatched_types_rejected(self):
        """Iterator with Bool base case and Nat step should be rejected."""
        bad = TmIter(
            TmTrue(),                                           # base: Bool
            'y',
            TmIf(TmVar('y'), TmZero(), TmSucc(TmZero())),     # step: Nat
            TmZero(),                                           # 0 iterations
        )
        with pytest.raises(TypeError, match="must match"):
            check(bad, {})

    def test_matching_types_accepted(self):
        """Iterator with matching base and step types should work."""
        good = TmIter(
            TmTrue(),
            'y',
            TmIf(TmVar('y'), TmFalse(), TmTrue()),  # Bool → Bool
            TmZero(),
        )
        ty = check(good, {})
        assert ty == TyBool()


class TestBug3_ClosureMutation:
    """Applying a closure must not mutate previously returned closures."""

    def test_closure_stability(self):
        """Reusing an outer closure must not mutate an earlier returned inner closure."""
        # outer = λx. λy. iter{x | z ↪ z}(y)
        outer_tm = TmFun(
            'x', TyBool(),
            TmFun('y', TyNat(), TmIter(TmVar('x'), 'z', TmVar('z'), TmVar('y')))
        )

        check(outer_tm, {})
        outer_v = evaluate(outer_tm, {})

        nat1 = TmSucc(TmZero())

        # Apply outer to True → get inner closure c_true
        c_true = evaluate(TmApp(TmVar('f'), TmTrue()), {'f': outer_v})
        res1_before = evaluate(TmApp(TmVar('g'), nat1), {'g': c_true})

        # Apply outer to False → get inner closure c_false
        c_false = evaluate(TmApp(TmVar('f'), TmFalse()), {'f': outer_v})

        # c_true should still produce True, not False
        res1_after = evaluate(TmApp(TmVar('g'), nat1), {'g': c_true})
        res2 = evaluate(TmApp(TmVar('g'), nat1), {'g': c_false})

        assert isinstance(res1_before, VTrue)
        assert isinstance(res1_after, VTrue), \
            "Closure was mutated: c_true should still return VTrue"
        assert isinstance(res2, VFalse)


class TestBug4_MatrixEquality:
    """TypedTensor equality must work for matrix-valued tensors."""

    def test_matrix_equality(self):
        """Equality on matrix TypedTensors should return bool, not raise."""
        m1 = TypedTensor(torch.eye(2, device=device), TyFun(TyBool(), TyBool()))
        m2 = TypedTensor(torch.eye(2, device=device), TyFun(TyBool(), TyBool()))
        assert m1 == m2

    def test_matrix_inequality(self):
        m1 = TypedTensor(torch.eye(2, device=device), TyFun(TyBool(), TyBool()))
        m2 = TypedTensor(torch.zeros(2, 2, device=device), TyFun(TyBool(), TyBool()))
        assert m1 != m2

    def test_type_mismatch(self):
        """Different types should not be equal even if data matches."""
        t1 = TypedTensor(torch.tensor([1.0, 0.0], device=device), TyBool())
        t2 = TypedTensor(torch.tensor([1.0, 0.0], device=device), TyReal(2))
        assert t1 != t2

    def test_vector_equality(self):
        """Existing vector equality should still work."""
        t1 = TypedTensor(torch.tensor([1.0, 0.0], device=device), TyBool())
        t2 = TypedTensor(torch.tensor([1.0, 0.0], device=device), TyBool())
        assert t1 == t2
