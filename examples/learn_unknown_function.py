"""
Learning an Unknown Process Function via Neural Sub-expression
==============================================================

The most ambitious demo: instead of learning a single scalar parameter,
we learn an *entire unknown function* using a neural network, then
attempt to decompile it back to a symbolic expression.

The scientific model (soil carbon with nonlinear moisture response):

    C(n+1) = C(n) - k * f_moisture(moisture(n)) * C(n) * dt

where:
    k = base decomposition rate (KNOWN)
    f_moisture = some function of soil moisture (UNKNOWN)

Ground truth (hidden from the learner):
    f_moisture(m) = m^0.7 / (0.3 + m^0.7)    (Hill equation)

We give the model:
    - The iteration structure (Cajal program)
    - The known rate k
    - Training data: (moisture, C) trajectories
    - A small MLP as the learnable f_moisture

After training, we:
    1. Evaluate the learned MLP on a grid to see what function it learned
    2. Attempt symbolic regression to recover the Hill equation
    3. Verify invariants (C >= 0, monotone decay)

This demonstrates:
    - Neural network as a learnable sub-expression in a Cajal program
    - Gradient-based learning of an arbitrary nonlinear function
    - Decompilation from neural weights to symbolic form
    - Structure (iteration + known physics) constraining the neural search
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

# ── Ground truth ──────────────────────────────────────────────

K_BASE = 0.5         # known base decomposition rate
DT = 0.1
C0 = 1.0
N_STEPS = 10
N_TRAJECTORIES = 20  # train on multiple moisture levels


def true_f_moisture(m):
    """Hill equation -- the function we're trying to recover."""
    return m ** 0.7 / (0.3 + m ** 0.7)


def generate_trajectories():
    """Generate training data: decay curves at different moisture levels."""
    moistures = torch.linspace(0.05, 1.0, N_TRAJECTORIES, device=device)
    all_curves = []

    for m in moistures:
        f_m = true_f_moisture(m)
        decay_factor = 1.0 - K_BASE * f_m * DT
        curve = []
        c = C0
        for _ in range(N_STEPS):
            curve.append(c)
            c = c * decay_factor
        all_curves.append(torch.tensor(curve, device=device))

    return moistures, torch.stack(all_curves)  # (N_TRAJ, N_STEPS)


# ── Learnable moisture response (MLP) ─────────────────────────

class MoistureResponseMLP(nn.Module):
    """
    A small neural network that learns f_moisture: ℝ → ℝ.

    Architecture: 1 → 32 → 32 → 1 with tanh activations.
    Output is sigmoid-clamped to [0, 1] since f_moisture is a
    response function (this is a structural constraint!).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # structural constraint: output in [0, 1]
        )
        self.net.to(device)

    def forward(self, m):
        """m is a scalar moisture value."""
        return self.net(m.view(1, 1)).squeeze()


class DecayWithMoisture(nn.Module):
    """
    One timestep: C' = C - k * f_moisture(moisture) * C * dt

    The iteration structure and k are fixed.
    f_moisture is a learnable MLP.
    moisture is provided as a constant for each trajectory.
    """

    def __init__(self, k, dt, f_moisture_mlp):
        super().__init__()
        self.k = k
        self.dt = dt
        self.f_moisture = f_moisture_mlp

    def forward(self, state):
        c = state.data[0]
        moisture = state.data[1]  # carried along as part of state

        f_m = self.f_moisture(moisture)
        c_new = c - self.k * f_m * c * self.dt

        return TypedTensor(
            torch.stack([c_new, moisture]),  # moisture unchanged
            state.ty,
        )


# ── Training ──────────────────────────────────────────────────

def train():
    moistures, true_curves = generate_trajectories()

    # Cajal program: iter{state₀ | s ↪ f(s)}(n)
    program = TmIter(
        TmVar("s0"),
        "s",
        TmApp(TmVar("f"), TmVar("s")),
        TmVar("n"),
    )
    compiled = compile(program)

    mlp = MoistureResponseMLP()
    update_fn = DecayWithMoisture(K_BASE, DT, mlp)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.005)

    print("=" * 60)
    print("LEARNING AN UNKNOWN FUNCTION: f_moisture(m)")
    print("=" * 60)
    print(f"  Known: k={K_BASE}, dt={DT}")
    print(f"  Unknown: f_moisture (true: Hill equation m^0.7 / (0.3 + m^0.7))")
    print(f"  Learner: MLP with {sum(p.numel() for p in mlp.parameters())} parameters")
    print(f"  Training on {N_TRAJECTORIES} trajectories at different moisture levels")
    print()

    for epoch in range(800):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for traj_idx in range(N_TRAJECTORIES):
            m = moistures[traj_idx]
            s0 = TypedTensor(
                torch.stack([torch.tensor(C0, device=device), m]),
                TyReal(2),
            )

            for step in range(N_STEPS):
                n_onehot = torch.zeros(N_STEPS, device=device)
                n_onehot[step] = 1.0
                n_val = TypedTensor(n_onehot, TyNat())

                result = compiled({
                    "s0": s0,
                    "f": lambda s, _fn=update_fn: _fn(s),
                    "n": n_val,
                })

                predicted_c = result.data[0]
                true_c = true_curves[traj_idx, step]
                total_loss = total_loss + (predicted_c - true_c) ** 2

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 799:
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.8f}")

    print()

    # ── Evaluate learned function ─────────────────────────────
    print("=" * 60)
    print("LEARNED vs TRUE f_moisture(m)")
    print("=" * 60)
    print()
    print(f"  {'m':>6s}  {'True f(m)':>10s}  {'Learned f(m)':>12s}  {'Error':>8s}")

    test_moistures = torch.linspace(0.05, 1.0, 15, device=device)
    max_err = 0.0
    with torch.no_grad():
        for m in test_moistures:
            true_val = true_f_moisture(m).item()
            learned_val = mlp(m).item()
            err = abs(true_val - learned_val)
            max_err = max(max_err, err)
            print(f"  {m.item():6.3f}  {true_val:10.4f}  {learned_val:12.4f}  {err:8.4f}")

    print(f"\n  Max absolute error: {max_err:.4f}")

    # ── Symbolic regression attempt ───────────────────────────
    print()
    print("=" * 60)
    print("SYMBOLIC REGRESSION (decompilation)")
    print("=" * 60)
    print()

    # Sample the learned function densely and try to fit candidate forms
    m_grid = torch.linspace(0.01, 1.5, 200, device=device)
    with torch.no_grad():
        learned_values = torch.tensor(
            [mlp(m).item() for m in m_grid], device="cpu"
        )
    m_cpu = m_grid.cpu()

    # Candidate symbolic forms to try
    candidates = {
        "linear: a*m":
            lambda m, a: a * m,
        "Michaelis-Menten: m/(K+m)":
            lambda m, K: m / (K + m),
        "Hill (n=0.5): m^0.5/(K+m^0.5)":
            lambda m, K: m**0.5 / (K + m**0.5),
        "Hill (n=0.7): m^0.7/(K+m^0.7)":
            lambda m, K: m**0.7 / (K + m**0.7),
        "Hill (n=1.5): m^1.5/(K+m^1.5)":
            lambda m, K: m**1.5 / (K + m**1.5),
        "quadratic: a*m^2/(K+m^2)":
            lambda m, K: m**2 / (K + m**2),
    }

    best_name = None
    best_loss = float("inf")
    best_param = None

    for name, func in candidates.items():
        # Grid search over the parameter
        min_loss = float("inf")
        min_param = None
        for param_val in torch.linspace(0.01, 2.0, 200):
            try:
                predicted = func(m_cpu, param_val)
                loss = ((predicted - learned_values) ** 2).mean().item()
                if loss < min_loss:
                    min_loss = loss
                    min_param = param_val.item()
            except Exception:
                continue

        marker = ""
        if min_loss < best_loss:
            best_loss = min_loss
            best_name = name
            best_param = min_param
            marker = "  <-- best so far"
        print(f"  {name:40s}  MSE={min_loss:.6f}  param={min_param:.3f}{marker}")

    print()
    print(f"  BEST FIT: {best_name}")
    print(f"  Parameter: {best_param:.3f}")
    print(f"  MSE: {best_loss:.6f}")
    print()

    if "0.7" in best_name:
        print("  The symbolic regression correctly identified the Hill equation")
        print("  with exponent 0.7 -- matching the ground truth!")
    else:
        print(f"  Selected {best_name} (ground truth was Hill n=0.7)")

    # ── Verification ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # f_moisture output in [0,1] -- guaranteed by sigmoid
    with torch.no_grad():
        vals = torch.tensor([mlp(m).item() for m in m_grid])
        f_range_ok = (vals >= 0).all() and (vals <= 1).all()
    print(f"  f_moisture(m) ∈ [0,1]:         {'VERIFIED' if f_range_ok else 'FAILED'} (by architecture)")

    # C(n) >= 0 for all trajectories
    all_positive = True
    all_monotone = True
    with torch.no_grad():
        for traj_idx in range(N_TRAJECTORIES):
            m = moistures[traj_idx]
            s0 = TypedTensor(
                torch.stack([torch.tensor(C0, device=device), m]),
                TyReal(2),
            )
            prev_c = C0 + 1
            for step in range(N_STEPS):
                n_onehot = torch.zeros(N_STEPS, device=device)
                n_onehot[step] = 1.0
                n_val = TypedTensor(n_onehot, TyNat())
                result = compiled({
                    "s0": s0,
                    "f": lambda s, _fn=update_fn: _fn(s),
                    "n": n_val,
                })
                c_val = result.data[0].item()
                if c_val < -1e-10:
                    all_positive = False
                if c_val > prev_c + 1e-8:
                    all_monotone = False
                prev_c = c_val

    print(f"  C(n) ≥ 0 for all trajectories: {'VERIFIED' if all_positive else 'FAILED'}")
    print(f"  C monotonically decreasing:    {'VERIFIED' if all_monotone else 'FAILED'}")


if __name__ == "__main__":
    train()
