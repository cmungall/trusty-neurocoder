"""
Battery Capacity Degradation via Cajal: Learning Unknown Fade Functions
=======================================================================

A lithium-ion battery cycling model where the *structure* (SEI growth causing
capacity fade) is known but two *mechanistic functions* are unknown and
learned from cycling data.

The scientific model (per-cycle update with timestep dt):

    SEI(n+1) = SEI(n) + k_sei * f_growth(SEI(n)) * dt
    Q(n+1)   = Q(n)   - k_cap * SEI(n) * g_fade(Q(n)) * dt

where:
    Q         = remaining capacity (starts at 1.0, decays toward 0)
    SEI       = solid-electrolyte interphase thickness (grows with cycling)
    k_sei     = SEI growth rate constant (KNOWN = 0.1)
    k_cap     = capacity fade rate constant (KNOWN = 0.05)
    dt        = timestep (KNOWN = 0.5)

Unknown (to learn):
    f_growth(SEI) = how SEI growth rate depends on current thickness
        Ground truth: f_growth(s) = 1 / sqrt(0.1 + s)  (parabolic growth law)
    g_fade(Q) = how capacity fade rate depends on remaining capacity
        Ground truth: g_fade(Q) = Q^0.5  (square root dependence)

Both are learned as small MLPs (1 -> 16 -> 16 -> 1).

Training data:
    8 trajectories at different initial SEI thicknesses (simulating
    batteries that started cycling at different calendar ages).

After training, we:
    1. Compare learned vs true functions on a grid
    2. Attempt symbolic regression to recover the functional forms
    3. Verify physical invariants: Q >= 0, Q decreasing, SEI increasing

This demonstrates:
    - Two learnable sub-expressions in a single Cajal program
    - 3-dimensional state (Q, SEI, constant initial condition tag)
    - Scientifically motivated structural constraints on the neural outputs
    - Battery degradation modeling relevant to DOE energy storage research
"""

import torch
import torch.nn as nn
from cajal.syntax import TmIter, TmVar, TmApp, TyNat, TyReal, TyBool
from cajal.compiling import compile, TypedTensor

device = torch.device("cpu")

# -- Known parameters (given to the model) ----------------------------

K_SEI = 0.1       # SEI growth rate constant
K_CAP = 0.05      # capacity fade rate constant
DT = 0.5          # timestep
Q0 = 1.0          # initial capacity (fully charged, normalized)
N_STEPS = 10      # timesteps per trajectory
N_TRAJECTORIES = 8  # different initial SEI thicknesses


# -- Ground truth (hidden from the learner) ---------------------------

def true_f_growth(sei):
    """Parabolic growth law: growth rate slows as SEI thickens."""
    return 1.0 / torch.sqrt(0.1 + sei)


def true_g_fade(q):
    """Square root dependence: fade rate depends on remaining capacity."""
    return q ** 0.5


def generate_trajectories():
    """
    Generate ground-truth trajectories at different initial SEI thicknesses.

    Returns:
        sei_inits  : tensor of initial SEI values
        curves     : list of (N_STEPS, 2) tensors [Q, SEI]
    """
    sei_inits = torch.linspace(0.05, 0.6, N_TRAJECTORIES, device=device)
    curves = []

    for sei0 in sei_inits:
        curve = []
        q = Q0
        sei = sei0.item()
        for _ in range(N_STEPS):
            curve.append(torch.tensor([q, sei], dtype=torch.float32))
            sei_new = sei + K_SEI * true_f_growth(torch.tensor(sei)).item() * DT
            q_new = q - K_CAP * sei * true_g_fade(torch.tensor(q)).item() * DT
            sei = sei_new
            q = q_new
        curves.append(torch.stack(curve))  # (N_STEPS, 2)

    return sei_inits, curves


# -- Learnable sub-expression MLPs -----------------------------------

class SEIGrowthMLP(nn.Module):
    """
    f_growth: R -> R+.

    Architecture: 1 -> 16 -> 16 -> 1 with tanh + softplus output.
    Softplus ensures output is strictly positive (SEI always grows,
    never shrinks -- this is a structural constraint from electrochemistry).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )
        self.net.to(device)

    def forward(self, sei):
        return self.net(sei.view(1, 1)).squeeze()


class CapacityFadeMLP(nn.Module):
    """
    g_fade: R -> [0, 1].

    Architecture: 1 -> 16 -> 16 -> 1 with tanh + sigmoid output.
    Sigmoid ensures output is bounded in [0, 1] (fade rate is a
    fraction of capacity -- structural constraint).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.net.to(device)

    def forward(self, q):
        return self.net(q.view(1, 1)).squeeze()


class BatteryUpdate(nn.Module):
    """
    One timestep of the battery degradation model.

    State layout: [Q, SEI, sei_init_normalized]  (3-dimensional)

    The structure is fixed by domain knowledge:
        SEI' = SEI + k_sei * f_growth(SEI) * dt
        Q'   = Q   - k_cap * SEI * g_fade(Q) * dt

    sei_init_normalized is carried unchanged (identifies the trajectory).
    The ONLY learnable components are f_growth and g_fade.
    """

    def __init__(self, k_sei, k_cap, dt, f_growth_mlp, g_fade_mlp):
        super().__init__()
        self.k_sei = k_sei
        self.k_cap = k_cap
        self.dt = dt
        self.f_growth = f_growth_mlp
        self.g_fade = g_fade_mlp

    def forward(self, state):
        q = state.data[0]
        sei = state.data[1]
        sei_init = state.data[2]  # constant per trajectory

        f_g = self.f_growth(sei)
        g_f = self.g_fade(q)

        sei_new = sei + self.k_sei * f_g * self.dt
        q_new = q - self.k_cap * sei * g_f * self.dt

        return TypedTensor(
            torch.stack([q_new, sei_new, sei_init]),
            state.ty,
        )


# -- Symbolic regression helper --------------------------------------

def fit_candidate_1d(func, x_data, y_data, param_grid):
    """Grid search a 1-parameter symbolic candidate against (x, y) data."""
    best_loss = float("inf")
    best_val = None
    for pv in param_grid:
        pred = func(x_data, pv)
        loss = ((pred - y_data) ** 2).mean().item()
        if loss < best_loss:
            best_loss = loss
            best_val = pv.item() if hasattr(pv, "item") else pv
    return best_loss, best_val


# -- Training ---------------------------------------------------------

def train():
    sei_inits, true_curves = generate_trajectories()

    # Cajal iteration program: iter{s0 | s -> f(s)}(n)
    program = TmIter(
        TmVar("s0"),
        "s",
        TmApp(TmVar("f"), TmVar("s")),
        TmVar("n"),
    )
    compiled = compile(program)

    # Two independent learnable MLPs
    f_growth_mlp = SEIGrowthMLP()
    g_fade_mlp = CapacityFadeMLP()
    update_fn = BatteryUpdate(K_SEI, K_CAP, DT, f_growth_mlp, g_fade_mlp)

    # Single optimizer over both MLPs
    optimizer = torch.optim.Adam(
        list(f_growth_mlp.parameters()) + list(g_fade_mlp.parameters()),
        lr=0.005,
    )

    n_growth_params = sum(p.numel() for p in f_growth_mlp.parameters())
    n_fade_params = sum(p.numel() for p in g_fade_mlp.parameters())

    print("=" * 70)
    print("BATTERY DEGRADATION: Learning f_growth(SEI) and g_fade(Q)")
    print("=" * 70)
    print(f"  Known:   k_sei={K_SEI}, k_cap={K_CAP}, dt={DT}")
    print(f"  Unknown: f_growth(SEI) -- true: 1/sqrt(0.1 + SEI)")
    print(f"           g_fade(Q)     -- true: Q^0.5")
    print(f"  MLPs:    f_growth has {n_growth_params} params, "
          f"g_fade has {n_fade_params} params")
    print(f"  Training: {N_TRAJECTORIES} trajectories at different initial "
          f"SEI thicknesses, {N_STEPS} steps each")
    print()

    for epoch in range(500):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for traj_idx in range(N_TRAJECTORIES):
            sei0 = sei_inits[traj_idx]
            s0 = TypedTensor(
                torch.stack([
                    torch.tensor(Q0, device=device),
                    sei0,
                    sei0,  # carry initial SEI as trajectory identifier
                ]),
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

                predicted = result.data[:2]
                target = true_curves[traj_idx][step].to(device)
                total_loss = total_loss + ((predicted - target) ** 2).sum()

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 499:
            print(f"  epoch {epoch:3d}  loss={total_loss.item():.8f}")

    print()

    # -- Evaluate learned functions -----------------------------------
    print("=" * 70)
    print("LEARNED vs TRUE: f_growth(SEI)")
    print("=" * 70)
    test_sei = torch.linspace(0.05, 1.0, 12, device=device)
    print(f"  {'SEI':>6s}  {'True':>10s}  {'Learned':>10s}  {'Error':>8s}")
    with torch.no_grad():
        for s in test_sei:
            true_val = true_f_growth(s).item()
            learned_val = f_growth_mlp(s).item()
            err = abs(true_val - learned_val)
            print(f"  {s.item():6.3f}  {true_val:10.4f}  {learned_val:10.4f}  {err:8.4f}")

    print()
    print("=" * 70)
    print("LEARNED vs TRUE: g_fade(Q)")
    print("=" * 70)
    test_q = torch.linspace(0.1, 1.0, 10, device=device)
    print(f"  {'Q':>6s}  {'True':>10s}  {'Learned':>10s}  {'Error':>8s}")
    with torch.no_grad():
        for q in test_q:
            true_val = true_g_fade(q).item()
            learned_val = g_fade_mlp(q).item()
            err = abs(true_val - learned_val)
            print(f"  {q.item():6.3f}  {true_val:10.4f}  {learned_val:10.4f}  {err:8.4f}")

    # -- Symbolic regression: f_growth --------------------------------
    print()
    print("=" * 70)
    print("SYMBOLIC REGRESSION: f_growth(SEI)")
    print("=" * 70)
    print()

    sei_grid = torch.linspace(0.01, 1.5, 200)
    with torch.no_grad():
        y_growth = torch.tensor(
            [f_growth_mlp(s.to(device)).item() for s in sei_grid]
        )

    param_grid = torch.linspace(0.01, 2.0, 200)

    growth_candidates = {
        "1/sqrt(a + s)":
            lambda s, a: 1.0 / torch.sqrt(a + s),
        "1/(a + s)":
            lambda s, a: 1.0 / (a + s),
        "exp(-a*s)":
            lambda s, a: torch.exp(-a * s),
        "1/s^a":
            lambda s, a: 1.0 / (s ** a + 1e-8),
        "a / (1 + s)":
            lambda s, a: a / (1.0 + s),
    }

    best_name_growth = None
    best_loss_growth = float("inf")
    best_param_growth = None

    for name, func in growth_candidates.items():
        loss, pv = fit_candidate_1d(func, sei_grid, y_growth, param_grid)
        marker = ""
        if loss < best_loss_growth:
            best_loss_growth = loss
            best_name_growth = name
            best_param_growth = pv
            marker = "  <-- best so far"
        print(f"  {name:30s}  MSE={loss:.6f}  param={pv:.3f}{marker}")

    print()
    print(f"  BEST: {best_name_growth}")
    print(f"  Param: {best_param_growth:.3f}   (true a=0.100)")
    print(f"  MSE:   {best_loss_growth:.6f}")
    if "sqrt" in best_name_growth:
        print("  Symbolic regression correctly identified the parabolic growth law!")

    # -- Symbolic regression: g_fade ----------------------------------
    print()
    print("=" * 70)
    print("SYMBOLIC REGRESSION: g_fade(Q)")
    print("=" * 70)
    print()

    q_grid = torch.linspace(0.01, 1.0, 200)
    with torch.no_grad():
        y_fade = torch.tensor(
            [g_fade_mlp(q.to(device)).item() for q in q_grid]
        )

    exponent_grid = torch.linspace(0.1, 2.0, 200)

    fade_candidates = {
        "Q^a (power law)":
            lambda Q, a: Q ** a,
        "Q / (a + Q)":
            lambda Q, a: Q / (a + Q),
        "linear: a*Q":
            lambda Q, a: a * Q,
        "1 - exp(-a*Q)":
            lambda Q, a: 1.0 - torch.exp(-a * Q),
        "sigmoid(a*(Q-0.5))":
            lambda Q, a: torch.sigmoid(a * (Q - 0.5)),
    }

    best_name_fade = None
    best_loss_fade = float("inf")
    best_param_fade = None

    for name, func in fade_candidates.items():
        loss, pv = fit_candidate_1d(func, q_grid, y_fade, exponent_grid)
        marker = ""
        if loss < best_loss_fade:
            best_loss_fade = loss
            best_name_fade = name
            best_param_fade = pv
            marker = "  <-- best so far"
        print(f"  {name:30s}  MSE={loss:.6f}  param={pv:.3f}{marker}")

    print()
    print(f"  BEST: {best_name_fade}")
    print(f"  Param: {best_param_fade:.3f}   (true exponent=0.500)")
    print(f"  MSE:   {best_loss_fade:.6f}")
    if "power" in best_name_fade and abs(best_param_fade - 0.5) < 0.15:
        print("  Symbolic regression correctly identified Q^0.5 -- matches ground truth!")

    # -- Decompiled expression ----------------------------------------
    print()
    print("=" * 70)
    print("DECOMPILED BATTERY MODEL")
    print("=" * 70)
    print()
    print(f"  SEI(n+1) = SEI(n) + {K_SEI} * f_growth(SEI(n)) * {DT}")
    print(f"  Q(n+1)   = Q(n)   - {K_CAP} * SEI(n) * g_fade(Q(n)) * {DT}")
    print()
    if "sqrt" in best_name_growth:
        print(f"  f_growth(s) ~ 1/sqrt({best_param_growth:.3f} + s)   "
              f"[parabolic growth law]")
    else:
        print(f"  f_growth(s) ~ {best_name_growth}  (param={best_param_growth:.3f})")
    if "power" in best_name_fade:
        print(f"  g_fade(Q)   ~ Q^{best_param_fade:.3f}               "
              f"[power-law fade]")
    else:
        print(f"  g_fade(Q)   ~ {best_name_fade}  (param={best_param_fade:.3f})")

    # -- Verification -------------------------------------------------
    print()
    print("=" * 70)
    print(f"VERIFICATION (across all {N_TRAJECTORIES} trajectories)")
    print("=" * 70)

    # Architecture guarantees
    with torch.no_grad():
        sei_test = torch.linspace(0.0, 2.0, 100, device=device)
        growth_vals = torch.tensor([f_growth_mlp(s).item() for s in sei_test])
        f_growth_positive = (growth_vals > 0).all()

        q_test = torch.linspace(0.0, 1.0, 100, device=device)
        fade_vals = torch.tensor([g_fade_mlp(q).item() for q in q_test])
        g_fade_bounded = (fade_vals >= 0).all() and (fade_vals <= 1).all()

    print(f"  f_growth(SEI) > 0:               {'VERIFIED' if f_growth_positive else 'FAILED'} "
          f"(by softplus architecture)")
    print(f"  g_fade(Q) in [0,1]:              {'VERIFIED' if g_fade_bounded else 'FAILED'} "
          f"(by sigmoid architecture)")

    # Physical invariants across all trajectories
    all_q_positive = True
    q_monotone_dec = True
    sei_monotone_inc = True

    with torch.no_grad():
        for traj_idx in range(N_TRAJECTORIES):
            sei0 = sei_inits[traj_idx]
            s0 = TypedTensor(
                torch.stack([
                    torch.tensor(Q0, device=device),
                    sei0,
                    sei0,
                ]),
                TyReal(3),
            )

            prev_q = Q0 + 1.0
            prev_sei = -1.0

            for step in range(N_STEPS):
                n_onehot = torch.zeros(N_STEPS, device=device)
                n_onehot[step] = 1.0
                n_val = TypedTensor(n_onehot, TyNat())
                result = compiled({
                    "s0": s0,
                    "f": lambda s, _fn=update_fn: _fn(s),
                    "n": n_val,
                })
                q_val = result.data[0].item()
                sei_val = result.data[1].item()

                if q_val < -1e-10:
                    all_q_positive = False
                if q_val > prev_q + 1e-8:
                    q_monotone_dec = False
                if sei_val < prev_sei - 1e-8:
                    sei_monotone_inc = False
                prev_q = q_val
                prev_sei = sei_val

    print(f"  Q >= 0 for all trajectories:     {'VERIFIED' if all_q_positive else 'FAILED'}")
    print(f"  Q monotonically decreasing:      {'VERIFIED' if q_monotone_dec else 'FAILED'}")
    print(f"  SEI monotonically increasing:    {'VERIFIED' if sei_monotone_inc else 'FAILED'}")

    # Illustrative trajectory
    print()
    print("ILLUSTRATIVE TRAJECTORY (initial SEI = 0.05):")
    print(f"  {'step':>4s}  {'Q_learned':>10s}  {'SEI_learned':>12s}  "
          f"{'Q_true':>10s}  {'SEI_true':>10s}")

    with torch.no_grad():
        sei0 = sei_inits[0]
        s0 = TypedTensor(
            torch.stack([
                torch.tensor(Q0, device=device),
                sei0,
                sei0,
            ]),
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
            q_l = result.data[0].item()
            sei_l = result.data[1].item()
            q_t = true_curves[0][step][0].item()
            sei_t = true_curves[0][step][1].item()
            print(f"  {step:4d}  {q_l:10.6f}  {sei_l:12.6f}  "
                  f"{q_t:10.6f}  {sei_t:10.6f}")


if __name__ == "__main__":
    train()
