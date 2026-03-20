"""
Exponential Decay via Cajal Compilation
========================================

Demonstrates that a Cajal program with a learnable sub-expression can
recover the rate constant of exponential decay from data.

The scientific model:
    dC/dt = -k * C
    C(t) = C₀ * exp(-k * t)

Discretized with timestep dt:
    C(n+1) = (1 - k*dt) * C(n)
    C(n) = ((1 - k*dt))^n * C₀

This IS a linear recurrent neuron:
    iter{C₀ | c ↪ f(c)}(n)

where f(c) = w * c is a learnable scalar multiplication.
After training, we recover k from the learned weight: k = (1 - w) / dt.

The point: the *structure* of the computation (iteration / ODE stepping)
is fixed by the Cajal program. Only the *rate* is learned. This is the
"verified surrogate" idea from the proposal in miniature.
"""

import torch
import torch.nn as nn
from cajal.syntax import TmIter, TmVar, TmApp, TyNat, TyReal, TyBool, TmTrue
from cajal.compiling import compile, TypedTensor

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ── Ground truth ──────────────────────────────────────────────

TRUE_K = 0.3          # true decay rate
DT = 0.1              # timestep
C0 = 1.0              # initial concentration
N_STEPS = 10          # number of timesteps (Cajal TyNat supports 0-9)
TRUE_DECAY = 1.0 - TRUE_K * DT   # = 0.97 per step


def generate_data():
    """Generate ground-truth decay curve: C(n) = C₀ * (1 - k*dt)^n."""
    ns = torch.arange(N_STEPS, dtype=torch.float32)
    cs = C0 * (TRUE_DECAY ** ns)
    return ns.to(device), cs.to(device)


# ── Learnable decay module ────────────────────────────────────

class LearnableDecay(nn.Module):
    """A single learnable scalar: f(c) = w * c.

    After training, the learned rate constant is k = (1 - w) / dt.
    """

    def __init__(self):
        super().__init__()
        # Initialize w near 1.0 (slow decay) -- deliberately wrong
        self.w = nn.Parameter(torch.tensor([0.5], device=device))

    def forward(self, c):
        return TypedTensor(self.w * c.data, c.ty)


# ── Build the Cajal program ──────────────────────────────────

def build_decay_program():
    """
    Build:  iter{C₀ | c ↪ f(c)}(n)

    where f is injected from the environment as a learnable module,
    and n selects which timestep to observe.
    """
    program = TmIter(
        TmVar("c0"),           # base case: initial concentration
        "c",                   # iterator variable
        TmApp(TmVar("f"), TmVar("c")),   # step: f(c)
        TmVar("n"),            # number of iterations
    )
    return program


# ── Training loop ─────────────────────────────────────────────

def train():
    ns, cs_true = generate_data()

    # Build the Cajal program (no type-checking needed for this PoC --
    # we're injecting learnable modules directly as environment values)
    program = build_decay_program()
    compiled = compile(program)

    # The learnable module
    decay_fn = LearnableDecay()
    optimizer = torch.optim.Adam(decay_fn.parameters(), lr=0.01)

    # Initial concentration as a 1-element tensor
    c0 = TypedTensor(torch.tensor([C0], device=device), TyReal(1))

    print("=" * 60)
    print("TRAINING: Learning decay rate from data")
    print("=" * 60)
    print(f"True k = {TRUE_K},  true w = {TRUE_DECAY:.4f}")
    print(f"Initial w = {decay_fn.w.item():.4f}")
    print()

    for epoch in range(300):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for i in range(N_STEPS):
            # One-hot encode the timestep
            n_onehot = torch.zeros(N_STEPS, device=device)
            n_onehot[i] = 1.0
            n_val = TypedTensor(n_onehot, TyNat())

            # Run the compiled Cajal program
            result = compiled({
                "c0": c0,
                "f": lambda c, _fn=decay_fn: _fn(c),
                "n": n_val,
            })

            # Loss: predicted vs true concentration
            predicted = result.data[0]
            target = cs_true[i]
            total_loss = total_loss + (predicted - target) ** 2

        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 299:
            w_learned = decay_fn.w.item()
            k_learned = (1.0 - w_learned) / DT
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.6f}  "
                  f"w={w_learned:.4f}  k_recovered={k_learned:.4f}")

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    w_final = decay_fn.w.item()
    k_final = (1.0 - w_final) / DT
    print(f"  True k:       {TRUE_K}")
    print(f"  Recovered k:  {k_final:.4f}")
    print(f"  Error:        {abs(k_final - TRUE_K):.6f}")
    print()

    # Show the decompiled result
    print("DECOMPILED SYMBOLIC EXPRESSION:")
    print(f"  C(n) = {C0} * ({w_final:.4f})^n")
    print(f"       = {C0} * (1 - {k_final:.4f} * {DT})^n")
    print(f"  which corresponds to:  dC/dt = -{k_final:.4f} * C")
    print()

    # Verify invariant: concentration must be non-negative and non-increasing
    print("VERIFICATION:")
    all_positive = True
    monotone = True
    prev = C0
    for i in range(N_STEPS):
        n_onehot = torch.zeros(N_STEPS, device=device)
        n_onehot[i] = 1.0
        n_val = TypedTensor(n_onehot, TyNat())
        with torch.no_grad():
            result = compiled({
                "c0": c0,
                "f": lambda c, _fn=decay_fn: _fn(c),
                "n": n_val,
            })
        val = result.data[0].item()
        if val < -1e-10:
            all_positive = False
        if val > prev + 1e-10:
            monotone = False
        prev = val

    w_val = decay_fn.w.item()
    print(f"  w = {w_val:.4f}, 0 < w < 1: {'VERIFIED' if 0 < w_val < 1 else 'FAILED'}")
    print(f"  C(n) >= 0 for all n:       {'VERIFIED' if all_positive else 'FAILED'}")
    print(f"  C(n+1) <= C(n) (monotone):  {'VERIFIED' if monotone else 'FAILED'}")


if __name__ == "__main__":
    train()
