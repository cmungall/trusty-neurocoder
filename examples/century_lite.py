"""
CENTURY-Lite: Simultaneously Learning Two Unknown Environmental Response Functions
==================================================================================

A three-pool soil carbon model (simplified CENTURY/RothC) where two environmental
response functions are simultaneously learned from trajectory data.

The scientific model (discretized with timestep dt):

    C_fast(n+1)    = C_fast(n)    - k_f  * f(T,M) * C_fast(n) * dt
    C_slow(n+1)    = C_slow(n)    + α_fs * k_f * f(T,M) * C_fast(n) * dt
                                  - k_s  * f(T,M) * C_slow(n) * dt
    C_passive(n+1) = C_passive(n) + α_sp * k_s * f(T,M) * C_slow(n) * dt
                                  - k_p  * f(T,M) * C_passive(n) * dt

where the environmental modifier f(T, M) is separable:

    f(T, M) = f_temp(T) * f_moist(M)

Known (fixed structure):  k_f, k_s, k_p, α_fs, α_sp, dt
Unknown (to learn):       f_temp(T) and f_moist(M) -- two nonlinear functions

Ground truth (hidden from the learner):
    f_temp(T)  = Q10^((T - T_ref)/10)       Q10=2.0, T_ref=15°C   (CENTURY standard)
    f_moist(M) = M^0.7 / (0.3 + M^0.7)                            (Hill equation)

Training data:
    9 trajectories at all combinations of 3 temperatures × 3 moisture levels.
    Each trajectory has 10 timesteps and 3-dimensional state.

After training, we:
    1. Evaluate both MLPs on a grid and compare to ground truth
    2. Attempt symbolic regression for each function independently
    3. Verify mass-conservation invariants across all 25 trajectories

This demonstrates:
    - Two learnable sub-expressions in a single Cajal program
    - 5-dimensional state (3 carbon pools + 2 environmental drivers)
    - Simultaneous learning of two unknown nonlinear functions from trajectory data
    - Scientific realism: CENTURY is the canonical global soil carbon model
    - Structure-constrained neural learning: the physics skeleton is exact
"""

import torch
import torch.nn as nn
from cajal.syntax import TmIter, TmVar, TmApp, TyNat, TyBool
from cajal.compiling import compile, TypedTensor

# CPU is faster than MPS for small-tensor serial workloads (low kernel-launch overhead).
# The Cajal compilation model runs many tiny sequential operations per step,
# making MPS kernel-launch latency the bottleneck. CPU is ~10x faster here.
device = torch.device("cpu")

# ── Known parameters (given to the model) ────────────────────

K_FAST    = 0.40    # fast pool decomposition rate (/step)
K_SLOW    = 0.02    # slow pool decomposition rate (/step)
K_PASSIVE = 0.001   # passive pool decomposition rate (/step)
ALPHA_FS  = 0.20    # fraction of fast decomposition transferred to slow pool
ALPHA_SP  = 0.10    # fraction of slow decomposition transferred to passive pool
DT        = 0.5     # timestep

C0_FAST    = 0.5    # initial fast pool carbon
C0_SLOW    = 1.5    # initial slow pool carbon
C0_PASSIVE = 5.0    # initial passive pool carbon

N_STEPS = 10        # timesteps per trajectory

# Environmental conditions (training grid: 3×3 = 9 trajectories)
N_TEMP  = 3
N_MOIST = 3
TEMPERATURES = torch.linspace(5.0, 25.0, N_TEMP)    # degrees C
MOISTURES    = torch.linspace(0.2, 0.8, N_MOIST)    # volumetric fraction


# ── Ground truth (hidden from learner) ───────────────────────

Q10_TRUE   = 2.0    # Q10 temperature sensitivity
T_REF_TRUE = 15.0   # reference temperature (°C)


def true_f_temp(T):
    """CENTURY Q10 temperature response: doubles per 10°C above reference."""
    return Q10_TRUE ** ((T - T_REF_TRUE) / 10.0)


def true_f_moist(M):
    """Hill equation moisture response (same as learn_unknown_function.py)."""
    return M ** 0.7 / (0.3 + M ** 0.7)


def generate_trajectories():
    """
    Generate ground-truth 3-pool trajectories at 25 (T, M) conditions.

    Returns:
        conditions : list of (T, M) tuples
        curves     : list of (N_STEPS, 3) tensors  [C_fast, C_slow, C_passive]
    """
    conditions = []
    curves = []

    for T in TEMPERATURES:
        for M in MOISTURES:
            f_env = true_f_temp(T) * true_f_moist(M)

            curve = []
            c_f, c_s, c_p = float(C0_FAST), float(C0_SLOW), float(C0_PASSIVE)
            for _ in range(N_STEPS):
                curve.append(torch.tensor([c_f, c_s, c_p], dtype=torch.float32))
                decomp_f = K_FAST * f_env * c_f * DT
                decomp_s = K_SLOW * f_env * c_s * DT
                decomp_p = K_PASSIVE * f_env * c_p * DT
                c_f = c_f - decomp_f
                c_s = c_s + ALPHA_FS * decomp_f - decomp_s
                c_p = c_p + ALPHA_SP * decomp_s - decomp_p

            conditions.append((T.item(), M.item()))
            curves.append(torch.stack(curve))  # (N_STEPS, 3)

    return conditions, curves


# ── Learnable response function MLPs ─────────────────────────

class TemperatureResponseMLP(nn.Module):
    """
    f_temp: ℝ → ℝ⁺.

    Architecture: 1 → 32 → 32 → 1 with tanh + softplus output.
    Softplus ensures output is strictly positive (temperature always
    accelerates decomposition; it never goes negative).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus(),  # structural constraint: f_temp > 0
        )
        self.net.to(device)

    def forward(self, T):
        return self.net(T.view(1, 1)).squeeze()


class MoistureResponseMLP(nn.Module):
    """
    f_moist: ℝ → [0, 1].

    Architecture: 1 → 32 → 32 → 1 with tanh + sigmoid output.
    Sigmoid ensures output is in [0, 1] (moisture response is a
    fraction that saturates; this is a structural constraint).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # structural constraint: f_moist ∈ [0, 1]
        )
        self.net.to(device)

    def forward(self, M):
        return self.net(M.view(1, 1)).squeeze()


class CenturyUpdate(nn.Module):
    """
    One timestep of the CENTURY-lite 3-pool system.

    State layout: [C_fast, C_slow, C_passive, T, M]  (5-dimensional)

    The structure is fixed by domain knowledge:
        C_fast'    = C_fast    - k_f  * f_env * C_fast * dt
        C_slow'    = C_slow    + α_fs * k_f * f_env * C_fast * dt
                               - k_s  * f_env * C_slow * dt
        C_passive' = C_passive + α_sp * k_s * f_env * C_slow * dt
                               - k_p  * f_env * C_passive * dt

    where f_env = f_temp(T) * f_moist(M).
    T and M are environmental drivers carried in state (unchanged each step).

    The ONLY learnable components are f_temp and f_moist.
    """

    def __init__(self, k_f, k_s, k_p, alpha_fs, alpha_sp, dt, f_temp_mlp, f_moist_mlp):
        super().__init__()
        self.k_f      = k_f
        self.k_s      = k_s
        self.k_p      = k_p
        self.alpha_fs = alpha_fs
        self.alpha_sp = alpha_sp
        self.dt       = dt
        self.f_temp   = f_temp_mlp
        self.f_moist  = f_moist_mlp

    def forward(self, state):
        c_f = state.data[0]
        c_s = state.data[1]
        c_p = state.data[2]
        T   = state.data[3]   # temperature — constant per trajectory
        M   = state.data[4]   # moisture   — constant per trajectory

        f_env = self.f_temp(T) * self.f_moist(M)

        decomp_f = self.k_f * f_env * c_f * self.dt
        decomp_s = self.k_s * f_env * c_s * self.dt
        decomp_p = self.k_p * f_env * c_p * self.dt

        c_f_new = c_f - decomp_f
        c_s_new = c_s + self.alpha_fs * decomp_f - decomp_s
        c_p_new = c_p + self.alpha_sp * decomp_s - decomp_p

        return TypedTensor(
            torch.stack([c_f_new, c_s_new, c_p_new, T, M]),
            state.ty,
        )


# ── Symbolic regression helpers ───────────────────────────────

def fit_candidate_1d(func, x_data, y_data, param_grid, param_name="param"):
    """Grid search a 1-parameter symbolic candidate against (x, y) data."""
    best_loss = float("inf")
    best_val  = None
    for pv in param_grid:
        try:
            pred = func(x_data, pv)
            loss = ((pred - y_data) ** 2).mean().item()
            if loss < best_loss:
                best_loss = loss
                best_val  = pv.item() if hasattr(pv, "item") else pv
        except Exception:
            continue
    return best_loss, best_val


def fit_candidate_2d(func, x_data, y_data, grid_a, grid_b):
    """Grid search a 2-parameter symbolic candidate against (x, y) data."""
    best_loss = float("inf")
    best_a = best_b = None
    for a in grid_a:
        for b in grid_b:
            try:
                pred = func(x_data, a, b)
                loss = ((pred - y_data) ** 2).mean().item()
                if loss < best_loss:
                    best_loss = loss
                    best_a = a.item() if hasattr(a, "item") else a
                    best_b = b.item() if hasattr(b, "item") else b
            except Exception:
                continue
    return best_loss, best_a, best_b


# ── Training ──────────────────────────────────────────────────

def train():
    conditions, true_curves = generate_trajectories()
    N_TRAJ = len(conditions)

    # Cajal iteration program: iter{s0 | s ↪ f(s)}(n)
    program = TmIter(
        TmVar("s0"),
        "s",
        TmApp(TmVar("f"), TmVar("s")),
        TmVar("n"),
    )
    compiled = compile(program)

    # Two independent learnable MLPs
    f_temp_mlp  = TemperatureResponseMLP()
    f_moist_mlp = MoistureResponseMLP()
    update_fn   = CenturyUpdate(K_FAST, K_SLOW, K_PASSIVE, ALPHA_FS, ALPHA_SP, DT,
                                f_temp_mlp, f_moist_mlp)

    # Single optimizer over both MLPs
    optimizer = torch.optim.Adam(
        list(f_temp_mlp.parameters()) + list(f_moist_mlp.parameters()),
        lr=0.005,
    )

    n_temp_params  = sum(p.numel() for p in f_temp_mlp.parameters())
    n_moist_params = sum(p.numel() for p in f_moist_mlp.parameters())

    print("=" * 70)
    print("CENTURY-LITE: Simultaneously Learning f_temp(T) and f_moist(M)")
    print("=" * 70)
    print(f"  Known:   k_f={K_FAST}, k_s={K_SLOW}, k_p={K_PASSIVE}")
    print(f"           α_fs={ALPHA_FS}, α_sp={ALPHA_SP}, dt={DT}")
    print(f"  Unknown: f_temp(T) -- true: Q10=2.0, T_ref=15°C")
    print(f"           f_moist(M) -- true: Hill m^0.7/(0.3+m^0.7)")
    print(f"  MLPs:    f_temp has {n_temp_params} params, f_moist has {n_moist_params} params")
    print(f"  Training: {N_TRAJ} trajectories ({N_TEMP}T × {N_MOIST}M conditions), "
          f"{N_STEPS} steps each, 3 pools")
    print()

    for epoch in range(600):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for traj_idx in range(N_TRAJ):
            T_val, M_val = conditions[traj_idx]
            T_t = torch.tensor(T_val, device=device)
            M_t = torch.tensor(M_val, device=device)

            s0 = TypedTensor(
                torch.stack([
                    torch.tensor(C0_FAST,    device=device),
                    torch.tensor(C0_SLOW,    device=device),
                    torch.tensor(C0_PASSIVE, device=device),
                    T_t,
                    M_t,
                ]),
                TyBool(),
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

                predicted_pools = result.data[:3]
                true_pools = true_curves[traj_idx][step].to(device)
                total_loss = total_loss + ((predicted_pools - true_pools) ** 2).sum()

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 599:
            print(f"  epoch {epoch:4d}  loss={total_loss.item():.6f}")

    print()

    # ── Evaluate learned functions ────────────────────────────
    print("=" * 70)
    print("LEARNED vs TRUE: f_temp(T)")
    print("=" * 70)
    test_temps = torch.linspace(5.0, 25.0, 11, device=device)
    print(f"  {'T (°C)':>8s}  {'True f_temp':>12s}  {'Learned':>12s}  {'Error':>8s}")
    with torch.no_grad():
        for T in test_temps:
            true_val   = true_f_temp(T).item() if hasattr(true_f_temp(T), "item") else true_f_temp(T)
            learned_val = f_temp_mlp(T).item()
            err = abs(true_val - learned_val)
            print(f"  {T.item():8.1f}  {true_val:12.4f}  {learned_val:12.4f}  {err:8.4f}")

    print()
    print("=" * 70)
    print("LEARNED vs TRUE: f_moist(M)")
    print("=" * 70)
    test_moist = torch.linspace(0.1, 0.9, 9, device=device)
    print(f"  {'M':>6s}  {'True f_moist':>12s}  {'Learned':>12s}  {'Error':>8s}")
    max_err_moist = 0.0
    with torch.no_grad():
        for M in test_moist:
            true_val   = true_f_moist(M).item() if hasattr(true_f_moist(M), "item") else true_f_moist(M)
            learned_val = f_moist_mlp(M).item()
            err = abs(true_val - learned_val)
            max_err_moist = max(max_err_moist, err)
            print(f"  {M.item():6.3f}  {true_val:12.4f}  {learned_val:12.4f}  {err:8.4f}")

    print()

    # ── Symbolic regression: f_temp ───────────────────────────
    print("=" * 70)
    print("SYMBOLIC REGRESSION: f_temp(T)")
    print("=" * 70)
    print()

    T_grid = torch.linspace(0.0, 35.0, 200)
    with torch.no_grad():
        y_temp = torch.tensor([f_temp_mlp(t.to(device)).item() for t in T_grid])

    # Candidates for f_temp
    temp_candidates = {
        "Q10=1.5: 1.5^((T-T0)/10)":
            lambda T, Q10, T0: Q10 ** ((T - T0) / 10.0),
        "Q10=2.0: 2.0^((T-T0)/10)":
            lambda T, Q10, T0: Q10 ** ((T - T0) / 10.0),
        "Q10=3.0: 3.0^((T-T0)/10)":
            lambda T, Q10, T0: Q10 ** ((T - T0) / 10.0),
        "Arrhenius: exp(a*(T-b))":
            lambda T, a, b: torch.exp(a * (T - b)),
        "linear: a*T":
            None,  # 1-parameter
        "power: a*T^b":
            None,  # 1-parameter (treated specially)
    }

    # Q10-family: 2D grid over Q10 ∈ [1.2, 4.0] and T_ref ∈ [5, 25]
    q10_grid  = torch.linspace(1.2, 4.0, 50)
    tref_grid = torch.linspace(5.0, 25.0, 50)
    # Arrhenius: 2D grid over a ∈ [0.01, 0.2] and b ∈ [0, 20]
    a_arr_grid = torch.linspace(0.01, 0.20, 50)
    b_arr_grid = torch.linspace(0.0, 20.0, 50)

    candidates_temp = [
        ("Q10 family: Q10^((T-T_ref)/10)",
         lambda T, a, b: a ** ((T - b) / 10.0),
         q10_grid, tref_grid, True),
        ("Arrhenius: exp(a*(T-b))",
         lambda T, a, b: torch.exp(a * (T - b)),
         a_arr_grid, b_arr_grid, True),
    ]
    # 1-param candidates
    candidates_temp_1p = [
        ("linear: a*T",
         lambda T, a: a * T,
         torch.linspace(0.01, 0.5, 200)),
    ]

    best_name_temp = None
    best_loss_temp = float("inf")
    best_params_temp = None

    for name, func, ga, gb, _ in candidates_temp:
        loss, pa, pb = fit_candidate_2d(func, T_grid, y_temp, ga, gb)
        marker = ""
        if loss < best_loss_temp:
            best_loss_temp   = loss
            best_name_temp   = name
            best_params_temp = (pa, pb)
            marker = "  <-- best so far"
        print(f"  {name:45s}  MSE={loss:.6f}  params=({pa:.3f}, {pb:.3f}){marker}")

    for name, func, pgrid in candidates_temp_1p:
        loss, pv = fit_candidate_1d(func, T_grid, y_temp, pgrid)
        marker = ""
        if loss < best_loss_temp:
            best_loss_temp   = loss
            best_name_temp   = name
            best_params_temp = (pv,)
            marker = "  <-- best so far"
        print(f"  {name:45s}  MSE={loss:.6f}  param={pv:.3f}{marker}")

    print()
    print(f"  BEST: {best_name_temp}")
    if len(best_params_temp) == 2:
        pa, pb = best_params_temp
        print(f"  Q10 ≈ {pa:.3f},  T_ref ≈ {pb:.3f}°C   (true: Q10={Q10_TRUE}, T_ref={T_REF_TRUE}°C)")
    print(f"  MSE: {best_loss_temp:.6f}")
    if best_params_temp and len(best_params_temp) == 2:
        pa, pb = best_params_temp
        if abs(pa - Q10_TRUE) < 0.3 and abs(pb - T_REF_TRUE) < 3.0:
            print(f"  The symbolic regression correctly identified the Q10 form")
            print(f"  and recovered Q10 ≈ {pa:.2f} (true=2.0), T_ref ≈ {pb:.1f}°C (true=15°C)")

    # ── Symbolic regression: f_moist ─────────────────────────
    print()
    print("=" * 70)
    print("SYMBOLIC REGRESSION: f_moist(M)")
    print("=" * 70)
    print()

    M_grid = torch.linspace(0.01, 1.5, 200)
    with torch.no_grad():
        y_moist = torch.tensor([f_moist_mlp(m.to(device)).item() for m in M_grid])

    K_grid = torch.linspace(0.01, 2.0, 200)

    moist_candidates = {
        "linear: a*M":
            lambda M, a: a * M,
        "Michaelis-Menten: M/(K+M)":
            lambda M, K: M / (K + M),
        "Hill (n=0.5): M^0.5/(K+M^0.5)":
            lambda M, K: M ** 0.5 / (K + M ** 0.5),
        "Hill (n=0.7): M^0.7/(K+M^0.7)":
            lambda M, K: M ** 0.7 / (K + M ** 0.7),
        "Hill (n=1.0): M/(K+M)":
            lambda M, K: M / (K + M),
        "Hill (n=1.5): M^1.5/(K+M^1.5)":
            lambda M, K: M ** 1.5 / (K + M ** 1.5),
        "quadratic: M^2/(K+M^2)":
            lambda M, K: M ** 2 / (K + M ** 2),
    }

    best_name_moist = None
    best_loss_moist = float("inf")
    best_param_moist = None

    for name, func in moist_candidates.items():
        loss, pv = fit_candidate_1d(func, M_grid, y_moist, K_grid)
        marker = ""
        if loss < best_loss_moist:
            best_loss_moist  = loss
            best_name_moist  = name
            best_param_moist = pv
            marker = "  <-- best so far"
        print(f"  {name:40s}  MSE={loss:.6f}  param={pv:.3f}{marker}")

    print()
    print(f"  BEST: {best_name_moist}")
    print(f"  Param: {best_param_moist:.3f}   (true K=0.300)")
    print(f"  MSE:   {best_loss_moist:.6f}")
    if "0.7" in best_name_moist:
        print(f"  Symbolic regression correctly identified Hill n=0.7 -- matches ground truth!")

    # ── Verification ─────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"VERIFICATION (across all {N_TRAJ} trajectories)")
    print("=" * 70)

    # 1. Architecture guarantees
    with torch.no_grad():
        M_test = torch.linspace(0.0, 1.5, 100, device=device)
        moist_vals = torch.tensor([f_moist_mlp(m).item() for m in M_test])
        f_moist_bounded = (moist_vals >= 0).all() and (moist_vals <= 1).all()

        T_test = torch.linspace(0.0, 40.0, 100, device=device)
        temp_vals = torch.tensor([f_temp_mlp(t).item() for t in T_test])
        f_temp_positive = (temp_vals > 0).all()

    print(f"  f_moist(M) ∈ [0,1]:              {'VERIFIED' if f_moist_bounded else 'FAILED'} (by sigmoid architecture)")
    print(f"  f_temp(T)  > 0:                  {'VERIFIED' if f_temp_positive else 'FAILED'} (by softplus architecture)")

    # 2. Physical invariants across all trajectories
    all_positive = True
    all_monotone_fast = True
    mass_conserved = True

    with torch.no_grad():
        for traj_idx in range(N_TRAJ):
            T_val, M_val = conditions[traj_idx]
            T_t = torch.tensor(T_val, device=device)
            M_t = torch.tensor(M_val, device=device)
            s0 = TypedTensor(
                torch.stack([
                    torch.tensor(C0_FAST,    device=device),
                    torch.tensor(C0_SLOW,    device=device),
                    torch.tensor(C0_PASSIVE, device=device),
                    T_t, M_t,
                ]),
                TyBool(),
            )

            total_prev = C0_FAST + C0_SLOW + C0_PASSIVE
            prev_fast  = C0_FAST + 1.0

            for step in range(N_STEPS):
                n_onehot = torch.zeros(N_STEPS, device=device)
                n_onehot[step] = 1.0
                n_val = TypedTensor(n_onehot, TyNat())
                result = compiled({
                    "s0": s0,
                    "f": lambda s, _fn=update_fn: _fn(s),
                    "n": n_val,
                })
                c_f = result.data[0].item()
                c_s = result.data[1].item()
                c_p = result.data[2].item()
                total = c_f + c_s + c_p

                if c_f < -1e-9 or c_s < -1e-9 or c_p < -1e-9:
                    all_positive = False
                if c_f > prev_fast + 1e-8:
                    all_monotone_fast = False
                if total > total_prev + 1e-6:
                    mass_conserved = False
                total_prev = total
                prev_fast  = c_f

    print(f"  All pools ≥ 0:                   {'VERIFIED' if all_positive else 'FAILED'}")
    print(f"  C_fast monotonically decreasing: {'VERIFIED' if all_monotone_fast else 'FAILED'}")
    print(f"  Total carbon non-increasing:     {'VERIFIED' if mass_conserved else 'FAILED'} (no carbon created)")

    # 3. Trajectory table for one illustrative condition
    print()
    print("ILLUSTRATIVE TRAJECTORY (T=15°C, M=0.5 -- reference condition):")
    print(f"  {'step':>4s}  {'C_fast':>8s}  {'C_slow':>8s}  {'C_pass':>8s}  "
          f"{'Total':>8s}  {'True_Total':>10s}")

    # Find the (T=15, M=0.5) trajectory
    ref_idx = None
    for i, (T, M) in enumerate(conditions):
        if abs(T - 15.0) < 0.5 and abs(M - 0.5) < 0.1:
            ref_idx = i
            break

    if ref_idx is not None:
        T_val, M_val = conditions[ref_idx]
        T_t = torch.tensor(T_val, device=device)
        M_t = torch.tensor(M_val, device=device)
        s0_ref = TypedTensor(
            torch.stack([
                torch.tensor(C0_FAST,    device=device),
                torch.tensor(C0_SLOW,    device=device),
                torch.tensor(C0_PASSIVE, device=device),
                T_t, M_t,
            ]),
            TyBool(),
        )

        with torch.no_grad():
            for step in range(N_STEPS):
                n_onehot = torch.zeros(N_STEPS, device=device)
                n_onehot[step] = 1.0
                n_val = TypedTensor(n_onehot, TyNat())
                result = compiled({
                    "s0": s0_ref,
                    "f": lambda s, _fn=update_fn: _fn(s),
                    "n": n_val,
                })
                c_f = result.data[0].item()
                c_s = result.data[1].item()
                c_p = result.data[2].item()
                true_row = true_curves[ref_idx][step]
                true_tot = true_row.sum().item()
                print(f"  {step:4d}  {c_f:8.5f}  {c_s:8.5f}  {c_p:8.5f}  "
                      f"{c_f+c_s+c_p:8.5f}  {true_tot:10.5f}")


if __name__ == "__main__":
    train()
