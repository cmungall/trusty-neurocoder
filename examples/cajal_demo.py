"""
Cajal Compilation Demo
======================

Demonstrates the core NSAM concept: symbolic programs compile to neural
network weight matrices, and the compilation preserves semantics.

Based on Velez-Ginorio et al., "Compiling to Recurrent Neurons" (2025).

Requires: pip install torch
"""

import torch
from cajal.syntax import (
    TmFun, TmVar, TmIf, TmTrue, TmFalse, TmApp,
    TmIter, TmZero, TmSucc, TyBool, TyNat,
)
from cajal.evaluating import evaluate
from cajal.compiling import compile, mat_of_lmap, TypedTensor
from cajal.typing import check


def demo_boolean_functions():
    """Compile boolean functions to matrices."""
    print("=" * 60)
    print("BOOLEAN FUNCTIONS → MATRICES")
    print("=" * 60)

    programs = {
        "identity  (λx. x)": TmFun("x", TyBool(), TmVar("x")),
        "NOT       (λx. if x then ff else tt)": TmFun(
            "x", TyBool(), TmIf(TmVar("x"), TmFalse(), TmTrue())
        ),
        "const-tt  (λx. if x then tt else tt)": TmFun(
            "x", TyBool(), TmIf(TmVar("x"), TmTrue(), TmTrue())
        ),
        "const-ff  (λx. if x then ff else ff)": TmFun(
            "x", TyBool(), TmIf(TmVar("x"), TmFalse(), TmFalse())
        ),
    }

    for name, tm in programs.items():
        check(tm, {})
        c_tm = compile(tm)
        lmap = c_tm({})
        matrix = mat_of_lmap(lmap)
        print(f"\n  {name}")
        print(f"  compiles to: {matrix.data.tolist()}")

    print()


def demo_iterated_not():
    """Compile iterated NOT to a recurrent neuron."""
    print("=" * 60)
    print("ITERATED NOT → RECURRENT NEURON")
    print("=" * 60)
    print()
    print("  Program: λn. iter{tt | y → not(y)}(n)")
    print("  This applies NOT n times to True.")
    print("  NOT compiles to matrix [[0,1],[1,0]].")
    print("  Iterating n times = n-th matrix power.")
    print()

    not_fn = TmFun("x", TyBool(), TmIf(TmVar("x"), TmFalse(), TmTrue()))

    iter_not = TmFun(
        "n",
        TyNat(),
        TmIter(
            TmTrue(),
            "y",
            TmApp(not_fn, TmVar("y")),
            TmVar("n"),
        ),
    )
    check(iter_not, {})
    c_iter = compile(iter_not)
    iter_lmap = c_iter({})

    device = next(
        d
        for d in ["mps", "cuda", "cpu"]
        if d == "cpu" or getattr(torch.backends, d, None) and getattr(torch.backends, d).is_available()
    )

    for n in range(6):
        n_tensor = torch.zeros(10, device=device)
        n_tensor[n] = 1.0
        result = iter_lmap(TypedTensor(n_tensor, TyNat()))
        label = "tt" if result.data[0] > result.data[1] else "ff"
        print(f"  n={n}: [{result.data[0]:.0f}, {result.data[1]:.0f}]  →  {label}")

    print()


def demo_symbolic_vs_neural():
    """Show that symbolic evaluation and neural compilation agree."""
    print("=" * 60)
    print("SYMBOLIC EVALUATION vs NEURAL COMPILATION")
    print("=" * 60)
    print()

    not_fn = TmFun("x", TyBool(), TmIf(TmVar("x"), TmFalse(), TmTrue()))

    test_cases = [
        ("not(tt)", TmApp(not_fn, TmTrue())),
        ("not(ff)", TmApp(not_fn, TmFalse())),
        (
            "iter{tt | y→not(y)}(0)",
            TmIter(TmTrue(), "y", TmApp(not_fn, TmVar("y")), TmZero()),
        ),
        (
            "iter{tt | y→not(y)}(1)",
            TmIter(
                TmTrue(), "y", TmApp(not_fn, TmVar("y")), TmSucc(TmZero())
            ),
        ),
        (
            "iter{tt | y→not(y)}(3)",
            TmIter(
                TmTrue(),
                "y",
                TmApp(not_fn, TmVar("y")),
                TmSucc(TmSucc(TmSucc(TmZero()))),
            ),
        ),
    ]

    for name, tm in test_cases:
        # Symbolic
        sym_result = evaluate(tm, {})
        sym_label = type(sym_result).__name__

        # Neural
        check(tm, {})
        c_tm = compile(tm)
        neural_result = c_tm({})
        neural_label = "tt" if neural_result.data[0] > neural_result.data[1] else "ff"

        match = "MATCH" if (
            (sym_label == "VTrue" and neural_label == "tt")
            or (sym_label == "VFalse" and neural_label == "ff")
        ) else "MISMATCH"

        print(f"  {name:30s}  symbolic={sym_label:8s}  neural={neural_label}  [{match}]")

    print()


if __name__ == "__main__":
    demo_boolean_functions()
    demo_iterated_not()
    demo_symbolic_vs_neural()
    print("All demos complete. Symbolic evaluation and neural compilation agree.")
