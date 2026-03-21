# Issue: Cajal Soundness and Runtime Semantics Bugs

Status: Open  
Reported: 2026-03-21  
Severity: High  
Area: `src/cajal/`

## Summary

The current Cajal core has multiple soundness and runtime-semantics bugs that are not covered by the test suite:

1. Lambda shadowing can silently discard an outer linear binding.
2. `TmIter` does not enforce that the base case and recursive step produce the same type.
3. The evaluator mutates closure environments in place, so reusing a higher-order closure can retroactively change earlier results.
4. `TypedTensor.__eq__` crashes on matrix-valued tensors, which breaks equality for compiled higher-order values.

These defects matter because the project’s main claim is that Cajal programs are linearly typed and semantics-preserving when compiled to tensors. At least two of the bugs violate that directly.

## Affected Files

- `src/cajal/typing.py`
- `src/cajal/evaluating.py`
- `src/cajal/compiling.py`

## Verification Context

- `pytest -q` under the system interpreter is misleading in this repo because the package requires Python 3.12 and `torch`, while the system shell in this environment did not provide that.
- Verified test status with the project environment:

```bash
.venv/bin/python --version
# Python 3.12.9

.venv/bin/python -m pytest -q
# 9 passed
```

The shipped tests pass, but they do not cover `cajal.evaluating` or the type-checker edge cases below.

## Bug 1: Lambda Shadowing Breaks Linearity

### Code Reference

- `src/cajal/typing.py`, `TmFun` branch

The checker currently extends the context with:

```python
ctx_extend = ctx | {x: ty1}
```

and returns the body's remaining context directly.

### Problem

If a lambda parameter shadows an existing linear binding in the outer context, the old binding is overwritten and never restored. That allows an outer linear resource to disappear without being consumed.

### Reproduction

```python
from cajal.syntax import *
from cajal.typing import check

term = TmFun('x', TyBool(), TmVar('x'))
print(check(term, {'x': TyNat()}))
```

### Actual Behavior

This succeeds and returns:

```python
TyFun(ty1=TyBool(), ty2=TyBool())
```

### Expected Behavior

This should fail, because the outer `x: TyNat()` remains unused after checking the lambda body.

### Why This Is Serious

The linear type checker is supposed to enforce exact resource usage. Silent loss of a binding is a direct soundness violation.

## Bug 2: `TmIter` Is Type-Unsafe

### Code Reference

- `src/cajal/typing.py`, `TmIter` branch

The checker computes:

- `ty1` for the base case `tm1`
- `ty2` for the recursive step `tm2`

but never checks `ty1 == ty2`.

### Problem

`TmIter` can type-check even when the zero case and successor case produce different types. That lets the term claim one type statically and produce another at runtime.

### Reproduction

```python
from cajal.syntax import *
from cajal.typing import check
from cajal.evaluating import evaluate

bad = TmIter(
    TmTrue(),
    'y',
    TmIf(TmVar('y'), TmZero(), TmSucc(TmZero())),
    TmZero(),
)

print(check(bad, {}))
print(type(evaluate(bad, {})).__name__)
```

### Actual Behavior

The term type-checks as:

```python
TyNat()
```

but evaluates to:

```python
VTrue
```

### Expected Behavior

The checker should reject this term because the base case is `TyBool()` while the recursive step returns `TyNat()`.

### Why This Is Serious

This is a direct preservation failure: the static type and runtime value disagree.

## Bug 3: Closure Application Mutates Captured Environments

### Code Reference

- `src/cajal/evaluating.py`
- `TmFun` returns `VClosure(x, ty, tm, env)`
- `TmApp` applies closures via `c_env |= {x: v2}`

### Problem

The evaluator stores the original environment object inside closures and then mutates it during application. Reusing an outer closure can therefore mutate previously returned closures that share the same captured dict.

### Reproduction

```python
from cajal.syntax import *
from cajal.typing import check
from cajal.evaluating import evaluate

outer_tm = TmFun(
    'x', TyBool(),
    TmFun('y', TyNat(), TmIter(TmVar('x'), 'z', TmVar('z'), TmVar('y')))
)

check(outer_tm, {})
outer_v = evaluate(outer_tm, {})

nat1 = TmSucc(TmZero())

c_true = evaluate(TmApp(TmVar('f'), TmTrue()), {'f': outer_v})
res1_before = evaluate(TmApp(TmVar('g'), nat1), {'g': c_true})

c_false = evaluate(TmApp(TmVar('f'), TmFalse()), {'f': outer_v})
res1_after = evaluate(TmApp(TmVar('g'), nat1), {'g': c_true})
res2 = evaluate(TmApp(TmVar('g'), nat1), {'g': c_false})

print(type(res1_before).__name__, type(res1_after).__name__, type(res2).__name__)
print(c_true.env)
print(c_false.env)
```

### Actual Behavior

Observed result:

```python
VTrue VFalse VFalse
```

and both returned closures end up sharing the mutated environment, effectively capturing the latest application rather than the original one.

### Expected Behavior

- Applying the outer closure to `true` should produce a stable inner closure that keeps `x = true`.
- A later application of the same outer closure to `false` must not mutate the first returned closure.

### Why This Is Serious

This breaks referential transparency for higher-order evaluation and makes runtime behavior dependent on evaluation history.

## Bug 4: Equality for Compiled Function Values Crashes

### Code Reference

- `src/cajal/compiling.py`, `TypedTensor.__eq__`

Current implementation:

```python
def __eq__(self, y):
    return all(self.data == y.data)
```

### Problem

For matrix-valued tensors, `all(self.data == y.data)` attempts to convert tensor rows into Python booleans and raises an exception.

### Reproduction

```python
import torch
from cajal.compiling import TypedTensor, device
from cajal.syntax import TyFun, TyBool

m1 = TypedTensor(torch.eye(2, device=device), TyFun(TyBool(), TyBool()))
m2 = TypedTensor(torch.eye(2, device=device), TyFun(TyBool(), TyBool()))
print(m1 == m2)
```

### Actual Behavior

Raises:

```python
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

### Expected Behavior

Equality should return a boolean result for equal tensors and should also account for the stored type tag.

### Why This Matters

Compiled higher-order values are represented as matrices. Basic equality on those values should not crash.

## Impact

- The linear type system is not currently sound under shadowing.
- The iterator typing rule does not preserve runtime type safety.
- Higher-order evaluation is stateful in a way that user code would not expect.
- Some basic operations on compiled values are unreliable.

Together, these defects weaken the repo's core claim that the symbolic and tensor semantics line up cleanly.

## Proposed Fixes

### `typing.py`

- In `TmFun`, treat the bound variable as scoped:
  - save any outer binding for the same name
  - check the body with the parameter binding
  - remove the parameter from the remaining context
  - restore the outer binding if one existed
- In `TmIter`, require `ty1 == ty2` before accepting the term.

### `evaluating.py`

- Make closure capture immutable from the evaluator's point of view:
  - either copy `env` when building `VClosure`
  - or build a fresh environment at application time instead of mutating the captured one

### `compiling.py`

- Replace `TypedTensor.__eq__` with a tensor-safe comparison such as `torch.equal(...)`
- Compare both `data` and `ty`

## Regression Tests Needed

- Shadowing test:
  - a lambda whose parameter shadows an outer linear binding must be rejected if the outer binding is not otherwise consumed
- Iterator preservation test:
  - an iterator with mismatched base and step result types must fail type-checking
- Closure stability test:
  - reusing an outer closure must not mutate an earlier returned inner closure
- Matrix equality test:
  - equality on `TypedTensor` matrices should return `True` or `False`, not raise

## Acceptance Criteria

- All four repro cases above behave as expected.
- New regression tests are added and pass in `.venv/bin/python -m pytest -q`.
- The fixes preserve current passing example-smoke and `TyReal` tests.
