"""
Radioactive Decay Chain via Cajal: Learning Unknown Branching Ratios
====================================================================

A 4-isotope radioactive decay chain A -> B -> C -> D (stable) where the
decay constants are known but the branching ratios are unknown and learned
from trajectory data.

The scientific model (discretized with timestep dt):

    A(n+1) = A(n) - lambda_A * A(n) * dt
    B(n+1) = B(n) + lambda_A * f_branch * A(n) * dt - lambda_B * B(n) * dt
    C(n+1) = C(n) + lambda_A * (1 - f_branch) * A(n) * dt
                   + lambda_B * g_branch * B(n) * dt - lambda_C * C(n) * dt
    D(n+1) = D(n) + lambda_B * (1 - g_branch) * B(n) * dt + lambda_C * C(n) * dt

where:
    lambda_A, lambda_B, lambda_C = known decay constants
    f_branch = fraction of A decays that produce B (rest produce C)
    g_branch = fraction of B decays that produce C (rest produce D)

Known:   lambda_A=0.3, lambda_B=0.1, lambda_C=0.05, dt=0.5
Unknown: f_branch (true=0.7), g_branch (true=0.85)

Key invariant: total mass A + B + C + D is conserved (no creation or
destruction of nuclei). This is guaranteed by the structure of the
update equations -- every term that leaves one pool enters another.

This demonstrates:
  - 4-dimensional state with mass conservation
  - Learnable scalar parameters with sigmoid constraint (branching ratios in [0,1])
  - Physically meaningful decompilation of learned parameters
"""

import torch
import torch.nn as nn
from cajal.syntax import TmIter, TmVar, TmApp, TyNat, TyReal, TyBool
from cajal.compiling import compile, TypedTensor

# CPU is faster than MPS for small-tensor serial workloads (low kernel-launch overhead).
device = torch.device("cpu")

# -- Known parameters (given to the model) ------------------------------------

LAMBDA_A = 0.3      # decay constant for isotope A (/step)
LAMBDA_B = 0.1      # decay constant for isotope B (/step)
LAMBDA_C = 0.05     # decay constant for isotope C (/step)
DT = 0.5            # timestep

# Initial abundances
A0 = 1.0
B0 = 0.0
C0 = 0.0
D0 = 0.0

N_STEPS = 10

# -- Ground truth (hidden from learner) ---------------------------------------

TRUE_F_BRANCH = 0.7    # 70% of A decays go to B, 30% go to C
TRUE_G_BRANCH = 0.85   # 85% of B decays go to C, 15% go to D


def true_update(state):
    """Ground truth: one timestep of the decay chain."""
    a, b, c, d = state
    decay_a = LAMBDA_A * a * DT
    decay_b = LAMBDA_B * b * DT
    decay_c = LAMBDA_C * c * DT

    a_new = a - decay_a
    b_new = b + TRUE_F_BRANCH * decay_a - decay_b
    c_new = c + (1 - TRUE_F_BRANCH) * decay_a + TRUE_G_BRANCH * decay_b - decay_c
    d_new = d + (1 - TRUE_G_BRANCH) * decay_b + decay_c

    return torch.stack([a_new, b_new, c_new, d_new])


def generate_data():
    """Generate ground-truth trajectory."""
    states = []
    state = torch.tensor([A0, B0, C0, D0], device=device)
    for _ in range(N_STEPS):
        states.append(state.clone())
        state = true_update(state)
    return torch.stack(states)  # (N_STEPS, 4)


# -- Learnable update module --------------------------------------------------

class DecayChainUpdate(nn.Module):
    """
    One timestep of the 4-isotope decay chain with learnable branching ratios.

    The structure is FIXED by domain knowledge:
        A' = A - lambda_A * A * dt
        B' = B + f * lambda_A * A * dt - lambda_B * B * dt
        C' = C + (1-f) * lambda_A * A * dt + g * lambda_B * B * dt - lambda_C * C * dt
        D' = D + (1-g) * lambda_B * B * dt + lambda_C * C * dt

    lambda_A, lambda_B, lambda_C, dt are known constants.
    f_branch and g_branch are the ONLY learnable parameters (sigmoid-constrained).
    """

    def __init__(self, lam_a, lam_b, lam_c, dt):
        super().__init__()
        self.lam_a = lam_a
        self.lam_b = lam_b
        self.lam_c = lam_c
        self.dt = dt
        # Raw (pre-sigmoid) parameters, initialized away from true values
        self.f_branch_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.g_branch_raw = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, state):
        a = state.data[0]
        b = state.data[1]
        c = state.data[2]
        d = state.data[3]

        f = torch.sigmoid(self.f_branch_raw)
        g = torch.sigmoid(self.g_branch_raw)

        decay_a = self.lam_a * a * self.dt
        decay_b = self.lam_b * b * self.dt
        decay_c = self.lam_c * c * self.dt

        a_new = a - decay_a
        b_new = b + f * decay_a - decay_b
        c_new = c + (1 - f) * decay_a + g * decay_b - decay_c
        d_new = d + (1 - g) * decay_b + decay_c

        return TypedTensor(
            torch.stack([a_new, b_new, c_new, d_new]),
            state.ty,
        )


# -- Training -----------------------------------------------------------------

def train():
    data = generate_data()

    # Cajal program: iter{s0 | s -> f(s)}(n)
    program = TmIter(
        TmVar("s0"),
        "s",
        TmApp(TmVar("f"), TmVar("s")),
        TmVar("n"),
    )
    compiled = compile(program)

    update_fn = DecayChainUpdate(LAMBDA_A, LAMBDA_B, LAMBDA_C, DT)
    optimizer = torch.optim.Adam(update_fn.parameters(), lr=0.05)

    s0 = TypedTensor(
        torch.tensor([A0, B0, C0, D0], device=device), TyReal(4)
    )

    f_init = torch.sigmoid(update_fn.f_branch_raw).item()
    g_init = torch.sigmoid(update_fn.g_branch_raw).item()

    print("=" * 65)
    print("RADIOACTIVE DECAY CHAIN: Learning unknown branching ratios")
    print("=" * 65)
    print(f"  Chain: A -> B -> C -> D (stable)")
    print(f"  Known: lambda_A={LAMBDA_A}, lambda_B={LAMBDA_B}, "
          f"lambda_C={LAMBDA_C}, dt={DT}")
    print(f"  Unknown: f_branch (true={TRUE_F_BRANCH}), "
          f"g_branch (true={TRUE_G_BRANCH})")
    print(f"  Initial: f_branch={f_init:.4f}, g_branch={g_init:.4f}")
    print()

    for epoch in range(400):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for i in range(N_STEPS):
            n_onehot = torch.zeros(N_STEPS, device=device)
            n_onehot[i] = 1.0
            n_val = TypedTensor(n_onehot, TyNat())

            result = compiled({
                "s0": s0,
                "f": lambda s, _fn=update_fn: _fn(s),
                "n": n_val,
            })

            predicted = result.data
            target = data[i]
            total_loss = total_loss + ((predicted - target) ** 2).sum()

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 399:
            f_val = torch.sigmoid(update_fn.f_branch_raw).item()
            g_val = torch.sigmoid(update_fn.g_branch_raw).item()
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.8f}  "
                  f"f_branch={f_val:.4f}  g_branch={g_val:.4f}")

    # -- Results ---------------------------------------------------------------
    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    f_final = torch.sigmoid(update_fn.f_branch_raw).item()
    g_final = torch.sigmoid(update_fn.g_branch_raw).item()
    print(f"  True f_branch:       {TRUE_F_BRANCH}")
    print(f"  Recovered f_branch:  {f_final:.4f}")
    print(f"  Error:               {abs(f_final - TRUE_F_BRANCH):.6f}")
    print()
    print(f"  True g_branch:       {TRUE_G_BRANCH}")
    print(f"  Recovered g_branch:  {g_final:.4f}")
    print(f"  Error:               {abs(g_final - TRUE_G_BRANCH):.6f}")
    print()

    print("DECOMPILED SYSTEM:")
    print(f"  dA/dt = -{LAMBDA_A} * A")
    print(f"  dB/dt = {f_final:.4f} * {LAMBDA_A} * A - {LAMBDA_B} * B")
    print(f"  dC/dt = {1-f_final:.4f} * {LAMBDA_A} * A + "
          f"{g_final:.4f} * {LAMBDA_B} * B - {LAMBDA_C} * C")
    print(f"  dD/dt = {1-g_final:.4f} * {LAMBDA_B} * B + {LAMBDA_C} * C")
    print()
    print(f"  Interpretation:")
    print(f"    {f_final*100:.1f}% of A decays produce B; "
          f"{(1-f_final)*100:.1f}% produce C directly")
    print(f"    {g_final*100:.1f}% of B decays produce C; "
          f"{(1-g_final)*100:.1f}% produce D directly")
    print()

    # -- Verification ----------------------------------------------------------
    print("=" * 65)
    print("VERIFICATION")
    print("=" * 65)

    # 1. Branching ratios in [0, 1] (guaranteed by sigmoid)
    f_valid = 0 <= f_final <= 1
    g_valid = 0 <= g_final <= 1
    print(f"  0 <= f_branch <= 1:               "
          f"{'VERIFIED' if f_valid else 'FAILED'} (sigmoid constraint)")
    print(f"  0 <= g_branch <= 1:               "
          f"{'VERIFIED' if g_valid else 'FAILED'} (sigmoid constraint)")

    # 2. All pools non-negative
    all_positive = True
    total_masses = []
    with torch.no_grad():
        for i in range(N_STEPS):
            n_onehot = torch.zeros(N_STEPS, device=device)
            n_onehot[i] = 1.0
            n_val = TypedTensor(n_onehot, TyNat())
            result = compiled({
                "s0": s0,
                "f": lambda s, _fn=update_fn: _fn(s),
                "n": n_val,
            })
            vals = [result.data[j].item() for j in range(4)]
            total_masses.append(sum(vals))
            if any(v < -1e-10 for v in vals):
                all_positive = False

    print(f"  All pools >= 0:                   "
          f"{'VERIFIED' if all_positive else 'FAILED'}")

    # 3. Total mass conserved (A+B+C+D should be constant)
    initial_mass = A0 + B0 + C0 + D0
    mass_conserved = all(
        abs(m - initial_mass) < 1e-4 for m in total_masses
    )
    print(f"  Mass conservation (A+B+C+D={initial_mass:.1f}): "
          f"{'VERIFIED' if mass_conserved else 'FAILED'}")

    # -- Trajectory table ------------------------------------------------------
    print()
    print("TRAJECTORY:")
    print(f"  {'step':>4s}  {'A':>8s}  {'B':>8s}  {'C':>8s}  "
          f"{'D':>8s}  {'Total':>8s}  {'True Tot':>8s}")
    with torch.no_grad():
        for i in range(N_STEPS):
            n_onehot = torch.zeros(N_STEPS, device=device)
            n_onehot[i] = 1.0
            n_val = TypedTensor(n_onehot, TyNat())
            result = compiled({
                "s0": s0,
                "f": lambda s, _fn=update_fn: _fn(s),
                "n": n_val,
            })
            a, b, c, d = [result.data[j].item() for j in range(4)]
            true_tot = data[i].sum().item()
            print(f"  {i:4d}  {a:8.5f}  {b:8.5f}  {c:8.5f}  "
                  f"{d:8.5f}  {a+b+c+d:8.5f}  {true_tot:8.5f}")


if __name__ == "__main__":
    train()
