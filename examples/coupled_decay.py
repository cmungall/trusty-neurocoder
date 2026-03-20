"""
Coupled Decomposition via Cajal: Learning Unknown Transfer Functions
=====================================================================

A more realistic scientific example: two coupled carbon pools where
the *structure* (mass conservation, pool coupling) is known but the
*transfer function* between pools is unknown and learned from data.

The scientific model (soil carbon decomposition):

    dC_labile/dt  = -k₁ * C_labile
    dC_stable/dt  =  α * k₁ * C_labile - k₂ * C_stable

where:
    C_labile  = fast-decomposing carbon pool
    C_stable  = slow-decomposing carbon pool
    k₁, k₂    = decomposition rates (KNOWN)
    α          = transfer coefficient: fraction of decomposed labile C
                 that becomes stable C (UNKNOWN -- this is what we learn)

Mass conservation: total carbon lost = respired CO₂
    d(C_labile + C_stable)/dt = -(1-α)*k₁*C_labile - k₂*C_stable

The Cajal program fixes the two-pool iteration structure and the known
rates, but makes the transfer coefficient α learnable. After training,
we decompile α and verify mass balance.

This demonstrates:
  - Multi-dimensional state (2D vector, not scalar)
  - Known structure constraining a learnable parameter
  - Mass conservation as a verifiable invariant
  - A scientifically meaningful result (recovering α)
"""

import torch
import torch.nn as nn
from cajal.syntax import TmIter, TmVar, TmApp, TyNat, TyReal, TyBool
from cajal.compiling import compile, TypedTensor

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── Ground truth parameters ──────────────────────────────────

TRUE_K1 = 0.3       # labile decomposition rate (known)
TRUE_K2 = 0.05      # stable decomposition rate (known)
TRUE_ALPHA = 0.4     # transfer coefficient labile→stable (UNKNOWN)
DT = 0.1
C0_LABILE = 1.0
C0_STABLE = 0.5
N_STEPS = 10


def true_update(state):
    """Ground truth: one timestep of the coupled system."""
    c_lab, c_stab = state
    c_lab_new = c_lab - TRUE_K1 * c_lab * DT
    c_stab_new = c_stab + TRUE_ALPHA * TRUE_K1 * c_lab * DT - TRUE_K2 * c_stab * DT
    return torch.stack([c_lab_new, c_stab_new])


def generate_data():
    """Generate ground-truth trajectories."""
    states = []
    state = torch.tensor([C0_LABILE, C0_STABLE], device=device)
    for i in range(N_STEPS):
        states.append(state.clone())
        state = true_update(state)
    return torch.stack(states)  # (N_STEPS, 2)


# ── Learnable update module ──────────────────────────────────

class CoupledUpdate(nn.Module):
    """
    One timestep of the coupled system with learnable transfer coefficient.

    The structure is FIXED by domain knowledge:
        C_lab'  = C_lab  - k1 * C_lab * dt
        C_stab' = C_stab + α * k1 * C_lab * dt - k2 * C_stab * dt

    k1, k2, dt are known constants.
    α is the ONLY learnable parameter.
    """

    def __init__(self, k1, k2, dt):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.dt = dt
        # Initialize α wrong on purpose
        self.alpha = nn.Parameter(torch.tensor(0.1, device=device))

    def forward(self, state):
        c_lab = state.data[0]
        c_stab = state.data[1]

        decomposed = self.k1 * c_lab * self.dt
        c_lab_new = c_lab - decomposed
        c_stab_new = c_stab + self.alpha * decomposed - self.k2 * c_stab * self.dt

        return TypedTensor(
            torch.stack([c_lab_new, c_stab_new]),
            state.ty,
        )


# ── Build and run ─────────────────────────────────────────────

def train():
    data = generate_data()

    # Cajal program: iter{C₀ | state ↪ f(state)}(n)
    program = TmIter(
        TmVar("c0"),
        "state",
        TmApp(TmVar("f"), TmVar("state")),
        TmVar("n"),
    )
    compiled = compile(program)

    update_fn = CoupledUpdate(TRUE_K1, TRUE_K2, DT)
    optimizer = torch.optim.Adam(update_fn.parameters(), lr=0.05)

    c0 = TypedTensor(
        torch.tensor([C0_LABILE, C0_STABLE], device=device), TyReal(2)
    )

    print("=" * 60)
    print("COUPLED CARBON POOLS: Learning transfer coefficient α")
    print("=" * 60)
    print(f"  Known: k₁={TRUE_K1}, k₂={TRUE_K2}, dt={DT}")
    print(f"  Unknown: α (true value = {TRUE_ALPHA})")
    print(f"  Initial α = {update_fn.alpha.item():.4f}")
    print()

    for epoch in range(500):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for i in range(N_STEPS):
            n_onehot = torch.zeros(N_STEPS, device=device)
            n_onehot[i] = 1.0
            n_val = TypedTensor(n_onehot, TyNat())

            result = compiled({
                "c0": c0,
                "f": lambda s, _fn=update_fn: _fn(s),
                "n": n_val,
            })

            predicted = result.data
            target = data[i]
            total_loss = total_loss + ((predicted - target) ** 2).sum()

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 499:
            alpha_val = update_fn.alpha.item()
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.8f}  "
                  f"α={alpha_val:.4f}")

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    alpha_final = update_fn.alpha.item()
    print(f"  True α:       {TRUE_ALPHA}")
    print(f"  Recovered α:  {alpha_final:.4f}")
    print(f"  Error:        {abs(alpha_final - TRUE_ALPHA):.6f}")
    print()

    print("DECOMPILED SYSTEM:")
    print(f"  dC_labile/dt  = -{TRUE_K1} * C_labile")
    print(f"  dC_stable/dt  = {alpha_final:.4f} * {TRUE_K1} * C_labile - {TRUE_K2} * C_stable")
    print()
    print(f"  Interpretation: {alpha_final*100:.1f}% of decomposed labile carbon")
    print(f"  transfers to stable pool; {(1-alpha_final)*100:.1f}% is respired as CO₂")
    print()

    # ── Verification ──────────────────────────────────────────
    print("VERIFICATION:")

    # 1. α must be between 0 and 1
    alpha_valid = 0 <= alpha_final <= 1
    print(f"  0 ≤ α ≤ 1:                    {'VERIFIED' if alpha_valid else 'FAILED'}")

    # 2. Both pools must remain non-negative
    all_positive = True
    total_carbon = []
    with torch.no_grad():
        for i in range(N_STEPS):
            n_onehot = torch.zeros(N_STEPS, device=device)
            n_onehot[i] = 1.0
            n_val = TypedTensor(n_onehot, TyNat())
            result = compiled({
                "c0": c0,
                "f": lambda s, _fn=update_fn: _fn(s),
                "n": n_val,
            })
            c_lab, c_stab = result.data[0].item(), result.data[1].item()
            total_carbon.append(c_lab + c_stab)
            if c_lab < -1e-10 or c_stab < -1e-10:
                all_positive = False

    print(f"  C_labile ≥ 0, C_stable ≥ 0:   {'VERIFIED' if all_positive else 'FAILED'}")

    # 3. Total carbon must be non-increasing (no carbon creation)
    monotone = all(
        total_carbon[i] >= total_carbon[i + 1] - 1e-8
        for i in range(len(total_carbon) - 1)
    )
    print(f"  Total carbon non-increasing:   {'VERIFIED' if monotone else 'FAILED'}")

    # 4. Show the trajectory
    print()
    print("TRAJECTORY:")
    print(f"  {'step':>4s}  {'C_labile':>10s}  {'C_stable':>10s}  {'Total':>10s}  {'True Total':>10s}")
    with torch.no_grad():
        for i in range(N_STEPS):
            n_onehot = torch.zeros(N_STEPS, device=device)
            n_onehot[i] = 1.0
            n_val = TypedTensor(n_onehot, TyNat())
            result = compiled({
                "c0": c0,
                "f": lambda s, _fn=update_fn: _fn(s),
                "n": n_val,
            })
            c_lab, c_stab = result.data[0].item(), result.data[1].item()
            true_lab, true_stab = data[i][0].item(), data[i][1].item()
            print(f"  {i:4d}  {c_lab:10.6f}  {c_stab:10.6f}  "
                  f"{c_lab+c_stab:10.6f}  {true_lab+true_stab:10.6f}")


if __name__ == "__main__":
    train()
