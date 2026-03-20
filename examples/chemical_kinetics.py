"""
Chemical Kinetics: Learning Temperature-Dependent Reaction Rates
================================================================

A DOE combustion/chemistry-inspired example: a simple reversible
first-order reaction A <=> B where the forward rate constant depends
on temperature via the Arrhenius equation.

    dA/dt = -k_fwd(T) * A + k_rev * B
    dB/dt =  k_fwd(T) * A - k_rev * B

Known structure:
    - Reversible first-order kinetics
    - Mass conservation: A + B = const
    - k_rev = 0.1 (reverse rate constant)
    - dt = 0.3

Unknown: k_fwd(T) -- the forward rate as a function of temperature.

Ground truth (hidden from the learner):
    k_fwd(T) = A_prefactor * exp(-E_a / T)
    where A_prefactor = 2.0, E_a = 5.0 (simplified Arrhenius, T in
    arbitrary energy units).  This gives k_fwd ~ 0.1 at low T up to
    ~1.3 at high T.

We train on 10 trajectories at temperatures T = linspace(3, 15, 10),
each with initial conditions A=1.0, B=0.0.  The learnable k_fwd is
an MLP (1 -> 32 -> 32 -> 1) with Softplus output (rate must be > 0).

After training we:
    1. Compare learned k_fwd(T) vs true on a grid of temperatures
    2. Symbolic regression: try Arrhenius, linear, power, quadratic
    3. Verify: A >= 0, B >= 0, A + B = constant (mass conservation)
    4. Show equilibrium: at long times A/B -> k_rev / k_fwd(T)
"""

import torch
import torch.nn as nn
from cajal.syntax import TmIter, TmVar, TmApp, TyNat, TyReal, TyBool
from cajal.compiling import compile, TypedTensor

device = torch.device("cpu")

# ── Ground truth parameters ──────────────────────────────────

A_PREFACTOR = 2.0    # Arrhenius pre-exponential factor
E_A = 5.0            # activation energy (arbitrary units)
K_REV = 0.1          # reverse rate constant (known)
DT = 0.3
N_STEPS = 10
N_TRAJ = 10          # number of temperature trajectories
TEMPERATURES = torch.linspace(3.0, 15.0, N_TRAJ, device=device)


def true_k_fwd(T):
    """Arrhenius forward rate constant."""
    return A_PREFACTOR * torch.exp(-E_A / T)


def generate_trajectories():
    """Generate training data: A,B trajectories at each temperature."""
    all_curves = []
    for T in TEMPERATURES:
        k_f = true_k_fwd(T).item()
        A, B = 1.0, 0.0
        curve = []
        for _ in range(N_STEPS):
            curve.append([A, B, T.item()])
            dA = (-k_f * A + K_REV * B) * DT
            A, B = A + dA, B - dA
        all_curves.append(torch.tensor(curve, device=device))
    return torch.stack(all_curves)  # (N_TRAJ, N_STEPS, 3)


# ── Learnable forward rate (MLP) ─────────────────────────────

class ForwardRateMLP(nn.Module):
    """
    A small neural network that learns k_fwd: T -> R+.

    Architecture: 1 -> 32 -> 32 -> 1 with Softplus output
    to enforce positivity of the rate constant.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus(),  # rate must be positive
        )

    def forward(self, T):
        """T is a scalar temperature value."""
        return self.net(T.view(1, 1)).squeeze()


class ReversibleKineticsUpdate(nn.Module):
    """
    One timestep of A <=> B with learned forward rate.

    State = [A, B, T] where T is carried as a constant.
        A' = A + (-k_fwd(T)*A + k_rev*B) * dt
        B' = B + ( k_fwd(T)*A - k_rev*B) * dt
        T' = T  (unchanged)
    """

    def __init__(self, k_rev, dt, k_fwd_mlp):
        super().__init__()
        self.k_rev = k_rev
        self.dt = dt
        self.k_fwd_mlp = k_fwd_mlp

    def forward(self, state):
        A = state.data[0]
        B = state.data[1]
        T = state.data[2]

        k_f = self.k_fwd_mlp(T)
        flux = k_f * A - self.k_rev * B
        A_new = A - flux * self.dt
        B_new = B + flux * self.dt

        return TypedTensor(
            torch.stack([A_new, B_new, T]),
            state.ty,
        )


# ── Training ─────────────────────────────────────────────────

def train():
    data = generate_trajectories()  # (N_TRAJ, N_STEPS, 3)

    # Cajal program: iter{s0 | s -> f(s)}(n)
    program = TmIter(
        TmVar("s0"),
        "s",
        TmApp(TmVar("f"), TmVar("s")),
        TmVar("n"),
    )
    compiled = compile(program)

    mlp = ForwardRateMLP()
    update_fn = ReversibleKineticsUpdate(K_REV, DT, mlp)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.005)

    print("=" * 60)
    print("CHEMICAL KINETICS: Learning k_fwd(T) for A <=> B")
    print("=" * 60)
    print(f"  Known: k_rev={K_REV}, dt={DT}")
    print(f"  Unknown: k_fwd(T)  (true: {A_PREFACTOR}*exp(-{E_A}/T))")
    print(f"  Learner: MLP with {sum(p.numel() for p in mlp.parameters())} parameters")
    print(f"  Training on {N_TRAJ} trajectories, T in [{TEMPERATURES[0]:.1f}, {TEMPERATURES[-1]:.1f}]")
    print()

    for epoch in range(500):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for traj_idx in range(N_TRAJ):
            T = TEMPERATURES[traj_idx]
            s0 = TypedTensor(
                torch.tensor([1.0, 0.0, T.item()], device=device),
                TyReal(3),
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

                pred_A = result.data[0]
                pred_B = result.data[1]
                true_A = data[traj_idx, step, 0]
                true_B = data[traj_idx, step, 1]
                total_loss = total_loss + (pred_A - true_A) ** 2 + (pred_B - true_B) ** 2

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 499:
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.8f}")

    print()

    # ── Evaluate learned k_fwd(T) ────────────────────────────
    print("=" * 60)
    print("LEARNED vs TRUE k_fwd(T)")
    print("=" * 60)
    print()
    print(f"  {'T':>6s}  {'True k_fwd':>10s}  {'Learned k_fwd':>13s}  {'Error':>8s}")

    T_grid = torch.linspace(3.0, 15.0, 15, device=device)
    max_err = 0.0
    with torch.no_grad():
        for T in T_grid:
            true_val = true_k_fwd(T).item()
            learned_val = mlp(T).item()
            err = abs(true_val - learned_val)
            max_err = max(max_err, err)
            print(f"  {T.item():6.2f}  {true_val:10.4f}  {learned_val:13.4f}  {err:8.4f}")

    print(f"\n  Max absolute error: {max_err:.4f}")

    # ── Symbolic regression ──────────────────────────────────
    print()
    print("=" * 60)
    print("SYMBOLIC REGRESSION (decompilation)")
    print("=" * 60)
    print()

    T_dense = torch.linspace(3.0, 15.0, 200, device=device)
    with torch.no_grad():
        learned_vals = torch.tensor([mlp(T).item() for T in T_dense])

    best_name = None
    best_loss = float("inf")
    best_params = None

    # 1. Arrhenius: A_pre * exp(-E_a / T) -- 2D grid search
    print("  Candidate: Arrhenius  A*exp(-E/T)")
    arr_best_loss = float("inf")
    arr_best_A, arr_best_E = None, None
    for A_try in torch.linspace(0.5, 4.0, 80):
        for E_try in torch.linspace(1.0, 10.0, 80):
            pred = A_try * torch.exp(-E_try / T_dense)
            loss = ((pred - learned_vals) ** 2).mean().item()
            if loss < arr_best_loss:
                arr_best_loss = loss
                arr_best_A = A_try.item()
                arr_best_E = E_try.item()
    print(f"    Best: A={arr_best_A:.3f}, E={arr_best_E:.3f}  MSE={arr_best_loss:.6f}")
    if arr_best_loss < best_loss:
        best_loss = arr_best_loss
        best_name = "Arrhenius: A*exp(-E/T)"
        best_params = f"A={arr_best_A:.3f}, E={arr_best_E:.3f}"

    # 2. Linear: a * T
    print("  Candidate: Linear  a*T")
    lin_best_loss = float("inf")
    lin_best_a = None
    for a_try in torch.linspace(0.01, 0.5, 200):
        pred = a_try * T_dense
        loss = ((pred - learned_vals) ** 2).mean().item()
        if loss < lin_best_loss:
            lin_best_loss = loss
            lin_best_a = a_try.item()
    print(f"    Best: a={lin_best_a:.3f}  MSE={lin_best_loss:.6f}")
    if lin_best_loss < best_loss:
        best_loss = lin_best_loss
        best_name = "Linear: a*T"
        best_params = f"a={lin_best_a:.3f}"

    # 3. Power law: a * T^b
    print("  Candidate: Power  a*T^b")
    pow_best_loss = float("inf")
    pow_best_a, pow_best_b = None, None
    for a_try in torch.linspace(0.01, 2.0, 80):
        for b_try in torch.linspace(0.1, 3.0, 80):
            pred = a_try * T_dense ** b_try
            loss = ((pred - learned_vals) ** 2).mean().item()
            if loss < pow_best_loss:
                pow_best_loss = loss
                pow_best_a = a_try.item()
                pow_best_b = b_try.item()
    print(f"    Best: a={pow_best_a:.3f}, b={pow_best_b:.3f}  MSE={pow_best_loss:.6f}")
    if pow_best_loss < best_loss:
        best_loss = pow_best_loss
        best_name = "Power: a*T^b"
        best_params = f"a={pow_best_a:.3f}, b={pow_best_b:.3f}"

    # 4. Quadratic: a*T^2 + b*T + c
    print("  Candidate: Quadratic  a*T^2 + b*T + c")
    quad_best_loss = float("inf")
    quad_best = None
    for a_try in torch.linspace(-0.02, 0.02, 30):
        for b_try in torch.linspace(-0.1, 0.3, 30):
            for c_try in torch.linspace(-1.0, 1.0, 30):
                pred = a_try * T_dense ** 2 + b_try * T_dense + c_try
                loss = ((pred - learned_vals) ** 2).mean().item()
                if loss < quad_best_loss:
                    quad_best_loss = loss
                    quad_best = (a_try.item(), b_try.item(), c_try.item())
    print(f"    Best: a={quad_best[0]:.4f}, b={quad_best[1]:.4f}, c={quad_best[2]:.4f}  "
          f"MSE={quad_best_loss:.6f}")
    if quad_best_loss < best_loss:
        best_loss = quad_best_loss
        best_name = "Quadratic: a*T^2 + b*T + c"
        best_params = f"a={quad_best[0]:.4f}, b={quad_best[1]:.4f}, c={quad_best[2]:.4f}"

    print()
    print(f"  BEST FIT: {best_name}")
    print(f"  Parameters: {best_params}")
    print(f"  MSE: {best_loss:.6f}")
    print()

    if "Arrhenius" in best_name:
        print(f"  Symbolic regression recovered Arrhenius form!")
        print(f"  True:     A={A_PREFACTOR:.3f}, E_a={E_A:.3f}")
        print(f"  Learned:  {best_params}")
    else:
        print(f"  Selected {best_name} (ground truth was Arrhenius)")

    # ── Verification ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    all_positive = True
    mass_conserved = True
    with torch.no_grad():
        for traj_idx in range(N_TRAJ):
            T = TEMPERATURES[traj_idx]
            s0 = TypedTensor(
                torch.tensor([1.0, 0.0, T.item()], device=device),
                TyReal(3),
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
                A_val = result.data[0].item()
                B_val = result.data[1].item()
                if A_val < -1e-8 or B_val < -1e-8:
                    all_positive = False
                if abs((A_val + B_val) - 1.0) > 1e-4:
                    mass_conserved = False

    print(f"  A >= 0, B >= 0 (all trajectories):  {'VERIFIED' if all_positive else 'FAILED'}")
    print(f"  A + B = 1.0 (mass conservation):    {'VERIFIED' if mass_conserved else 'FAILED'}")

    # ── Equilibrium check ────────────────────────────────────
    print()
    print("=" * 60)
    print("EQUILIBRIUM ANALYSIS")
    print("=" * 60)
    print()
    print("  At equilibrium: k_fwd(T)*A_eq = k_rev*B_eq")
    print("  So A_eq/B_eq = k_rev/k_fwd(T)")
    print()
    print(f"  {'T':>6s}  {'k_fwd(T)':>8s}  {'Expected A/B':>12s}  {'Simulated A/B':>13s}")

    with torch.no_grad():
        for T in TEMPERATURES:
            k_f = mlp(T).item()
            expected_ratio = K_REV / k_f

            # Run many steps to approach equilibrium
            s = TypedTensor(
                torch.tensor([1.0, 0.0, T.item()], device=device),
                TyReal(3),
            )
            for _ in range(200):
                s = update_fn(s)
            A_eq = s.data[0].item()
            B_eq = s.data[1].item()
            sim_ratio = A_eq / B_eq if B_eq > 1e-10 else float("inf")
            print(f"  {T.item():6.2f}  {k_f:8.4f}  {expected_ratio:12.4f}  {sim_ratio:13.4f}")


if __name__ == "__main__":
    train()
