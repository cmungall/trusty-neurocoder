# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trusty Neurocoder combines LLM agents with Neuro-Symbolic Abstract Machines (NSAMs) for verified scientific code generation. The core implemented component is **Cajal**, a minimal typed, higher-order, linear programming language whose programs compile to neural network weight matrices (PyTorch tensors).

## Development Commands

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests (no test suite yet — tests are inline in compiling.py)
pytest

# Run the demo
python examples/cajal_demo.py
```

## Architecture

Two packages under `src/`:

- **`cajal/`** — The Cajal language implementation (the working core):
  - `syntax.py` — AST types (`Tm` union for terms, `Ty` union for types, `Val` union for values). Uses Python 3.12+ `type` aliases and dataclasses.
  - `typing.py` — Linear type checker. Enforces that each variable is used exactly once (linear types). `check(tm, ctx)` is the main entry point; returns the type or raises `TypeError`.
  - `evaluating.py` — Symbolic interpreter. `evaluate(tm, env)` reduces terms to values.
  - `compiling.py` — Compiles Cajal terms to PyTorch tensor operations. `compile(tm)` returns a `MultilinearMap` (callable). `mat_of_lmap()` materializes a `LinearMap` into a weight matrix. Key types: `TypedTensor` (tensor + type tag), `LinearMap` (lazy linear function).

- **`trusty_neurocoder/`** — Top-level package (stub, not yet implemented).

## Key Design Patterns

- **Structural pattern matching** is used throughout (`match`/`case` on dataclass AST nodes). Follow this style.
- **Linear type system**: the type checker consumes variables from context on use and checks that no variables remain unused. This invariant is critical — it ensures compilation to linear algebra is semantics-preserving.
- **Compilation produces closures**: `compile()` returns nested lambdas/closures that capture the compiled subterms. These closures operate on `VectorEnv` (dict of string → tensor/linear map).
- Natural numbers are encoded as one-hot vectors of dimension 10 (so max representable nat is 9).
- Booleans are 2D vectors: `[1,0]` = true, `[0,1]` = false.
- Device selection is automatic (MPS > CUDA > CPU).
