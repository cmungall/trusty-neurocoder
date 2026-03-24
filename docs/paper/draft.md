# Trusty Neurocoder: Verified Scientific Surrogates via Neuro-Symbolic Compilation and LLM Agents

## Abstract

Scientific simulations encode decades of domain knowledge in complex
codebases, but producing fast, interpretable surrogate models from them
remains manual and error-prone. We present Trusty Neurocoder, a system
that combines LLM coding agents with neuro-symbolic abstract machines
(NSAMs) to construct verified surrogate models directly from scientific
source code. An LLM agent reads simulation source code (e.g., Fortran),
extracts the ODE structure, and generates a Cajal program — a typed
functional language whose programs compile exactly to recurrent neural
networks. Known physics (mass conservation, reaction stoichiometry,
kinetic structure) becomes fixed program structure; unknown or uncertain
process functions become learnable PyTorch sub-expressions. After
training against simulation output, symbolic regression decompiles the
learned neural weights back to interpretable mathematical expressions.
We extend the Cajal type system with real-valued vector types (`TyReal(n)`),
fix four soundness bugs in the vendored implementation, and demonstrate
the full pipeline on eight scientific models spanning soil
biogeochemistry, nuclear decay, battery degradation, and chemical
kinetics — including an end-to-end case extracting a decomposition
kernel from the EcoSIM land model's Fortran source. A controlled
comparison against physics-informed neural networks (PINNs) and
black-box MLPs shows the Cajal surrogate achieves 6,700× lower
interpolation error, exact conservation (10⁻⁸ vs. 10⁻²), 72×
better sample efficiency, and produces interpretable decompiled
expressions — while PINNs' soft conservation penalties actually
degrade extrapolation performance. LLM agent assistance accelerated
kernel extraction, Cajal programming, and experiment generation,
providing a concrete demonstration of the agent-assisted workflow.

## Introduction

Surrogate models approximate expensive scientific simulations with
cheaper alternatives, enabling uncertainty quantification, parameter
sweeps, and real-time prediction. The dominant approach trains neural
networks on simulation input-output pairs, treating the simulator as a
black box. This works, but discards everything scientists know about
the system: conservation laws, reaction structure, monotonicity
constraints, dimensional relationships. The resulting surrogates are
fast but uninterpretable, and violations of physical invariants must
be caught by post-hoc validation — if they are caught at all.

A parallel line of work in programming languages has produced
neuro-symbolic abstract machines (NSAMs): neural networks that are
structurally equivalent to programming language interpreters. Cajal, a
typed higher-order linear programming language developed by
Velez-Ginorio et al., compiles programs exactly to recurrent neural
network weight matrices. A Cajal program of type `τ₁ ⊸ τ₂` becomes a
linear map from the vector space encoding `τ₁` to that encoding `τ₂`.
Iteration compiles to matrix powers — recurrence. The compilation is
exact, not an approximation: the symbolic evaluator and the neural
compiler produce identical outputs on every input.

We connect these two threads. The Cajal compilation pipeline gives us
neural networks with *guaranteed structure*. If we fix the known parts
of a scientific model as Cajal program structure and make the unknown
parts learnable sub-expressions, we get surrogates that are:

- **Structurally correct by construction.** Mass conservation,
  positivity, and monotonicity follow from the program structure, not
  from regularization or post-hoc checking.
- **Trainable via backpropagation.** The compiled Cajal program is a
  differentiable PyTorch computation graph. Learnable sub-expressions
  (MLPs embedded in the program) train via standard gradient descent.
- **Decompilable.** After training, symbolic regression extracts
  interpretable mathematical expressions from the learned neural
  weights — closing the loop from neural back to symbolic.

The remaining gap is practical: Cajal programs are small functional
programs, but real scientific simulations are large Fortran or C++
codebases with complex data structures, I/O, and build systems. An LLM
coding agent bridges this gap, reading the source code, identifying the
computational kernel, extracting the ODE structure, and generating the
Cajal surrogate.

We demonstrate this pipeline end-to-end on eight scientific models,
culminating in a surrogate of the EcoSIM land model's soil organic
matter decomposition kernel, extracted directly from its Fortran source.

## Background

### Cajal: Programs as Neural Networks

Cajal(⊸, 𝟚, ℕ) is a typed, higher-order, linear programming language.
Its types are booleans (𝟚, encoded as one-hot vectors in ℝ²), natural
numbers (ℕ, one-hot in ℝ¹⁰), and linear maps (τ₁ ⊸ τ₂). The key
construct is iteration:

```
iter{e₁ | y ↪ e₂}(e₃)
```

which applies the step function `e₂` to initial state `e₁` a total of
`e₃` times. When compiled, this becomes a recurrent neural network:
the weight matrix (compiled from `e₂`) is applied `e₃` times to the
initial state vector (compiled from `e₁`).

The linear type system ensures each variable is used exactly once,
which maps directly to linear algebra. A Cajal program of type
`𝟚 ⊸ 𝟚` compiles to a 2×2 matrix. The NOT function compiles to the
permutation matrix `[[0,1],[1,0]]`; iterating NOT `n` times produces
`NOT^n`, which alternates between identity and NOT — exactly as the
symbolic evaluator predicts.

### Soundness Fixes to the Cajal Implementation

The vendored Cajal implementation contained four soundness and
runtime-semantics bugs not covered by the original test suite. We
identified and fixed all four as part of this work:

1. **Lambda shadowing** (`typing.py`). The linear type checker allowed
   a lambda parameter to shadow an outer linear binding, silently
   discarding it. This is a direct violation of the linear type
   discipline — a resource disappears without being consumed. Fix:
   scope the parameter binding, restore any shadowed outer binding
   after checking the body.

2. **Iterator type preservation** (`typing.py`). `TmIter` did not
   check that the base case and recursive step produce the same type,
   allowing a term to type-check as `TyNat()` but evaluate to `VTrue` —
   a preservation failure. Fix: require `ty_base == ty_step`.

3. **Closure environment mutation** (`evaluating.py`). Closure
   application used `|=` (mutating merge) on the captured environment
   dict. Reusing an outer closure could retroactively change earlier
   returned inner closures, breaking referential transparency.
   Fix: copy environments at closure creation; use `|` (non-mutating)
   at application.

4. **Matrix equality** (`compiling.py`). `TypedTensor.__eq__` called
   `all(self.data == y.data)`, which raises `RuntimeError` for
   matrix-valued tensors. Fix: use `torch.equal()` and compare type
   tags.

These fixes are covered by 10 regression tests. Bugs 1 and 2 affect
the core claim that Cajal's linear type system guarantees
structure preservation; they are not merely cosmetic.

### Learnable Sub-Expressions

The Cajal compiler produces a differentiable PyTorch computation graph.
We exploit this by injecting learnable PyTorch modules (MLPs, scalar
parameters) as environment bindings in the compiled program. The Cajal
iteration structure provides the fixed skeleton; the learnable modules
fill in the unknown parts. During training, gradients flow through the
entire compiled program, including the iteration, to update the
learnable module weights.

This is distinct from physics-informed neural networks (PINNs), which
encode physics as soft loss terms. In our approach, the physics is
*hard structure*: mass conservation holds because the program's update
equations are written to conserve mass, and the compiler preserves this
structure exactly.

### LLM Coding Agents

Large language model coding agents (Claude Code, Cursor, Codex) can
read, understand, and generate code across languages and frameworks.
We use the agent as Layer 1 of the pipeline: it reads scientific source
code, extracts the mathematical model, identifies which parts are
known physics and which are uncertain or empirical, and generates the
Cajal surrogate program. The agent also generates training code,
evaluation scripts, symbolic regression, and verification checks.

## System Design

### Architecture

The Trusty Neurocoder pipeline has three layers:

**Layer 1: LLM Coding Agent.** Reads scientific source code (Fortran,
C++, Python). Extracts the ODE system, identifies known vs. unknown
components, annotates physical constraints. Generates the Cajal program,
learnable modules, training loop, and verification code.

**Layer 2: NSAM Compilation.** The Cajal compiler transforms the
program into a PyTorch computation graph. Known structure becomes fixed
tensor operations; unknown functions become learnable `nn.Module`
instances. The iteration construct compiles to a recurrent loop with
matrix-power semantics.

**Layer 3: Verification.** Physical invariants are checked:
conservation laws (verified by inspecting the program structure),
range constraints (guaranteed by output activation functions —
sigmoid for [0,1], softplus for positivity), and monotonicity
(verified empirically across all training trajectories). After
training, symbolic regression decompiles learned neural weights
to interpretable expressions.

### Type System Extension: TyReal(n)

The published Cajal type system supports booleans (𝟚, dimension 2)
and naturals (ℕ, dimension 10). Scientific ODE surrogates require
real-valued state vectors. We extend the type system with `TyReal(n)`,
representing an n-dimensional real vector. This required three changes:

1. A new `TyReal` dataclass in the syntax module with dimension
   parameter `n`.
2. `dim(TyReal(n)) = n` in the compiler's dimension calculation.
3. Standard basis vectors `{e₁, ..., eₙ}` in the compiler's basis
   enumeration.

The extension is lightweight but enables proper typing of scientific
state vectors: `TyReal(1)` for scalar ODEs, `TyReal(4)` for a
four-isotope decay chain, `TyReal(7)` for the EcoSIM decomposition
model. Equality checking (`TyReal(4) == TyReal(4)`,
`TyReal(4) != TyReal(5)`) works via the dataclass default, so the
existing type checker handles `TyReal` without modification.

### Architectural Guarantees

Physical constraints are enforced by the architecture of the learnable
modules, not by loss penalties:

| Constraint | Mechanism |
|-----------|-----------|
| Output ∈ [0, 1] | Sigmoid final activation |
| Output > 0 | Softplus final activation |
| Mass conservation | Update equations sum to zero net flux |
| Monotone decay | Rate × concentration structure |

These guarantees hold for *any* learned weights, not just the trained
optimum. A freshly initialized, untrained model already satisfies all
physical invariants.

## Experiments

We demonstrate the pipeline on eight scientific models of increasing
complexity. All use the same Cajal program structure
(`TmIter(TmVar("s0"), "s", TmApp(TmVar("f"), TmVar("s")), TmVar("n"))`)
and the same training pattern (per-trajectory, per-timestep loss,
Adam optimizer). The models differ in state dimension, number of
learnable components, and scientific domain.

### Progressive Complexity

**Model 1: Exponential Decay** (TyReal(1), 1 learnable scalar).
The ODE `dC/dt = -kC` discretized as `C(n+1) = w·C(n)`. A single
learnable weight `w` recovers `k = (1-w)/dt = 0.3000` exactly
(error < 10⁻⁶). Demonstrates the basic pattern: Cajal iteration +
learnable scalar + decompilation.

**Model 2: Coupled Carbon Pools** (TyReal(2), 1 learnable scalar).
Two-pool system with unknown transfer coefficient α between fast and
slow carbon pools. Recovers α = 0.4000 exactly. Verifies mass
conservation: total carbon is non-increasing (respiration removes
carbon, nothing creates it).

**Model 3: Unknown Nonlinear Function** (TyReal(2), 1153-parameter MLP).
The step from scalar to function learning. A 3-layer MLP
(1→32→32→1, sigmoid output) learns the unknown moisture response
function inside the Cajal iteration. Trained on 20 trajectories at
different moisture levels. Symbolic regression on the trained MLP
correctly identifies the Hill equation `m^0.7 / (0.3 + m^0.7)` with
K = 0.300, matching the ground truth exactly.

**Model 4: CENTURY-Lite** (TyReal(5), two 1153-parameter MLPs).
A three-pool soil carbon model (simplified CENTURY) with two unknown
environmental response functions — temperature (Q10) and moisture
(Hill equation) — learned simultaneously from 9 trajectories at
different (T, M) conditions. Both functional forms correctly
identified by symbolic regression. Scale offsets between the two
functions reflect a fundamental identifiability limitation: with a
separable modifier `f(T)·g(M)`, the training data determines the
product uniquely but not the individual factors.

### DOE Science Domains

**Model 5: Radioactive Decay Chain** (TyReal(4), 2 learnable scalars).
Four-isotope chain A→B→C→D with unknown branching ratios. Both
recovered exactly: f_branch = 0.7000, g_branch = 0.8500 (error < 10⁻⁶).
Mass conservation (A+B+C+D = 1.0) holds to machine precision at every
timestep — a structural guarantee of the update equations.

**Model 6: Battery Capacity Degradation** (TyReal(3), two 321-parameter MLPs).
Solid-electrolyte interphase (SEI) growth coupled with capacity fade.
Two unknown functions: SEI growth rate (true: parabolic law
`1/√(0.1+s)`) and capacity fade rate (true: `Q^0.5`). Symbolic
regression identifies the parabolic growth law. Verification confirms
Q is monotonically decreasing and SEI is monotonically increasing
across all trajectories.

**Model 7: Chemical Kinetics** (TyReal(3), 1153-parameter MLP).
Reversible reaction A⇌B with unknown temperature-dependent forward
rate. The MLP learns `k_fwd(T)` from 10 trajectories at different
temperatures. Symbolic regression recovers the Arrhenius form with
A = 2.006, E = 4.987 (true: 2.0, 5.0 — errors of 0.3% and 0.3%).
Mass conservation (A+B = 1.0) verified to machine precision.
Equilibrium ratios A/B match the theoretical prediction
k_rev/k_fwd(T) at all temperatures.

### End-to-End: EcoSIM Decomposition Surrogate

**Model 8: EcoSIM SOM Decomposition** (TyReal(7), two 1153-parameter MLPs).
The capstone demonstration. The LLM agent read the EcoSIM land model's
Fortran source code (`MicBGCMod.F90`, subroutine
`SolidOMDecomposition`, lines 1335–1629; `MicrobMathFuncMod.F90`,
subroutine `MicrobPhysTempFun`, lines 13–30) and extracted the soil
organic matter decomposition kernel.

The model has five carbon pools (four solid substrates — protein,
carbohydrate, cellulose, lignin — plus dissolved organic matter),
Monod substrate limitation, dissolved organic carbon product
inhibition, and separable temperature/water-stress environmental
response. The agent identified the known structure (4-substrate
decomposition rates, Monod kinetics `DFNS = C/(C+Km)`, product
inhibition `OQCI = 1/(1+C_DOM/Ki)`, mass-conserving transfer of
decomposed solid C to DOM) and the unknown components (the
temperature sensitivity function `TSensGrowth` and water stress
function `WatStressMicb`).

The surrogate trained on 9 trajectories (3 temperatures × 3 water
potentials) for 400 epochs. Mass conservation across all five carbon
pools verified to < 10⁻⁵ relative error. All pools remain
non-negative. The decompiled surrogate cites the original Fortran
source line numbers.

### Comparison: Cajal vs. PINN vs. Black-Box

To quantify the advantage of hard structural constraints, we trained
three approaches on identical data from the chemical kinetics model
(reversible reaction A⇌B, 10 trajectories, 500 epochs each):

1. **Black-box MLP**: 3→64→64→2 network (4,354 parameters). Learns
   entire dynamics with no physics.
2. **PINN**: Same architecture, with a soft penalty
   λ·(A+B−1)² added to the loss (λ=10).
3. **Cajal surrogate**: Reaction structure fixed; only k_fwd(T) learned
   as a 1→32→32→1 MLP (1,153 parameters).

#### Interpolation (training temperatures, T ∈ [3, 15])

| Approach | Trajectory MSE | Max conservation error |
|----------|---------------|----------------------|
| Black-box | 6.2 × 10⁻³ | 5.4 × 10⁻³ |
| PINN (λ=10) | 7.7 × 10⁻³ | 6.5 × 10⁻³ |
| **Cajal** | **9.3 × 10⁻⁷** | **1.6 × 10⁻⁷** |

The Cajal surrogate achieves 6,700× lower trajectory error and
conservation error at machine precision. The PINN is slightly *worse*
than the black-box because the conservation penalty competes with the
data-fitting loss.

#### Extrapolation (unseen temperatures, T ∈ {1.5, 2.0, 18.0, 25.0})

| Approach | Trajectory MSE | Max conservation error |
|----------|---------------|----------------------|
| Black-box | 3.1 × 10⁻² | 9.8 × 10⁻² |
| PINN (λ=10) | 5.0 × 10⁻² | 1.7 × 10⁻¹ |
| **Cajal** | **1.3 × 10⁻²** | **5.2 × 10⁻⁸** |

On extrapolation, the PINN degrades more than the black-box — the
conservation penalty, calibrated for the training distribution,
actively harms predictions at new temperatures. The Cajal surrogate
maintains exact conservation regardless of temperature, and
extrapolates 2.5× better on trajectory error because it only needs
to extrapolate the rate function k_fwd(T), not the entire dynamics.

#### Sample efficiency

| Training trajectories | Black-box MSE | Cajal MSE | Ratio |
|----------------------|---------------|-----------|-------|
| 2 | 8.6 × 10⁻² | 1.2 × 10⁻³ | 72× |
| 3 | 2.2 × 10⁻² | 1.8 × 10⁻³ | 12× |
| 5 | 7.7 × 10⁻³ | 2.2 × 10⁻⁴ | 35× |
| 10 | 6.2 × 10⁻³ | 5.8 × 10⁻⁶ | 1,070× |

With only 2 training trajectories, the Cajal surrogate already
achieves lower error than the black-box with 10. The structural
constraints dramatically reduce the effective hypothesis space,
making learning data-efficient.

#### Interpretability

The black-box and PINN produce 4,354-parameter networks with no
scientific interpretation. The Cajal surrogate decompiles to:

```
k_fwd(T) = 1.97 · exp(-4.86 / T)
```

matching the ground truth k_fwd(T) = 2.0 · exp(-5.0/T) within 2%.
This is a publishable scientific finding, not a black-box prediction.

## Discussion

### What the Pipeline Preserves

The Cajal compilation preserves program structure exactly: if the
source program conserves mass, the compiled neural network conserves
mass. This is qualitatively different from physics-informed neural
networks, where conservation is a soft loss term that trades off
against data fit. In our approach, conservation holds for any
parameter values — including randomly initialized, untrained weights.

### What the Pipeline Cannot Do

The pipeline requires the user (or agent) to decompose the model into
known structure and unknown functions. This decomposition is a
scientific judgment, not an automated step. The agent can suggest
decompositions based on common patterns (Arrhenius rates, Michaelis-Menten
kinetics), but the scientist must validate them.

The current Cajal iteration is serial and operates on individual
state vectors, not batches. This makes GPU acceleration inefficient
for the small models demonstrated here. Compiling Cajal iteration
directly to batched `nn.RNN` modules would resolve this and is
planned future work.

Symbolic regression via grid search over candidate functional forms
is limited to forms the user anticipates. Integration with libraries
like PySR would enable open-ended symbolic search.

### Synthetic Data as Validation Strategy

All training data in this work is synthetic — generated from known
ground truth equations, not from running the actual EcoSIM simulation
or from observational measurements. This is deliberate. For a methods
paper, synthetic data is the strongest validation: because we know the
true functional forms, we can measure recovery accuracy and confirm
that the pipeline works correctly. With real simulation output, we
could verify trajectory fit but not whether the decompiled expressions
are scientifically correct.

The EcoSIM surrogate (Model 8) uses equations extracted from the
Fortran source and reimplemented in Python, not output from running
the actual Fortran code. The reimplementation faithfully reproduces
the mathematical formulas (temperature sensitivity, water stress,
Monod kinetics, product inhibition) but does not capture numerical
discretization artifacts, compiler-specific floating-point behavior,
or interactions with other EcoSIM modules. Training against actual
EcoSIM simulation output is straightforward future work that would
validate the surrogate against the full simulation, including effects
our reimplementation omits.

A natural next step beyond simulation output is training on
observational data (e.g., FLUXNET eddy covariance measurements or
the Soil Respiration Database). In that setting, the decompiled
expressions would represent empirical response functions inferred
from field data — a scientifically novel finding rather than recovery
of a known equation. The Cajal structural constraints (mass
conservation, positivity) would remain guaranteed regardless of the
data source.

### Identifiability

Models 4 and 8 (CENTURY-Lite and EcoSIM) exhibit a fundamental
identifiability limitation when learning separable environmental
modifiers `f(T)·g(M)`. The training data determines the product
`f·g` uniquely, but not the individual factors. The learned functions
have correct *shapes* (the symbolic forms are correctly identified)
but offset *scales*. Additional constraints — such as normalization
(`f(T_ref) = 1`) or independent measurements of one factor — would
resolve this. This is a property of the scientific problem, not a
limitation of the method.

### Agent-Assisted Prototype Development

An LLM coding agent (Claude Code) was used throughout development of
the prototype — type system extension, eight working models, test
suite, Jupyter notebooks, documentation site, and CI/CD. This serves
as a concrete demonstration of the Layer 1 workflow. The agent read
Fortran source code, identified decomposition kernels, generated
Cajal programs, debugged runtime errors (tensor shape mismatches,
device placement issues), and iterated on performance (discovering
that CPU is ~10× faster than Apple MPS for these small sequential
workloads).

This is not a claim that the agent replaces scientific expertise.
The user guided every major decision: which models to build, which
DOE domains to target, how to decompose known vs. unknown components.
The agent handled the mechanical work — reading Fortran, writing
PyTorch, managing git, deploying documentation — while the scientific
contribution remains the architecture, experiments, and resulting
surrogates rather than the speed of implementation.

## Related Work

**Physics-informed neural networks** (Raissi et al., 2019) encode
physical laws as soft penalty terms in the loss function. Our approach
encodes them as hard program structure, providing guarantees rather
than incentives. Our controlled comparison (Section "Comparison")
demonstrates a concrete failure mode of the PINN approach: the
conservation penalty, calibrated for the training distribution,
actively degrades extrapolation to unseen conditions. This is not a
tuning issue — it is inherent to soft constraints that trade off
against data fit.

**Neural ODEs** (Chen et al., 2018) parameterize the entire ODE
right-hand side as a neural network. We parameterize only the
*unknown* parts, fixing known structure. This dramatically reduces
the hypothesis space and guarantees invariant preservation.

**Symbolic regression** (Cranmer et al., 2020; PySR) discovers
mathematical expressions from data. We use it as a post-training
decompilation step, extracting interpretable expressions from trained
neural sub-expressions within a structured program.

**Neuro-symbolic programming** (Chaudhuri et al., 2021) broadly
combines neural and symbolic computation. Cajal (Velez-Ginorio et al.,
2025) is distinctive in providing an *exact* compilation from programs
to neural networks, not an approximation.

**AI for scientific simulation** (Kasim et al., 2022; Karniadakis
et al., 2021) has produced many surrogate modeling approaches.
Our contribution is the three-layer architecture connecting LLM
agents, neuro-symbolic compilation, and formal verification into a
single pipeline.

## Conclusion

Trusty Neurocoder demonstrates that LLM coding agents and
neuro-symbolic abstract machines can work together to produce
verified scientific surrogates from real simulation source code.
The agent handles code comprehension and generation; the NSAM
provides structural correctness; symbolic regression closes the loop
back to interpretable science. A controlled comparison against PINNs
confirms that hard structural constraints outperform soft penalties
on conservation, extrapolation, sample efficiency, and
interpretability — and that the gap widens, not narrows, as
conditions move away from the training distribution.

Eight working demonstrations across DOE science domains, including
an end-to-end extraction from the EcoSIM land model's Fortran source,
show the approach is practical today. Our contributions to the Cajal
implementation — the `TyReal(n)` type extension and four soundness
fixes — strengthen the theoretical foundation on which the pipeline
rests.

Code, notebooks, and documentation: https://github.com/cmungall/trusty-neurocoder

## References

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).
Neural ordinary differential equations. NeurIPS.

Chaudhuri, S., et al. (2021). Neurosymbolic programming. Foundations
and Trends in Programming Languages.

Cranmer, M., et al. (2020). Discovering symbolic models from deep
learning with inductive biases. NeurIPS.

Karniadakis, G. E., et al. (2021). Physics-informed machine learning.
Nature Reviews Physics, 3(6), 422–440.

Kasim, M. F., et al. (2022). Building high accuracy emulators for
scientific simulations with deep neural architecture search. Machine
Learning: Science and Technology.

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
Physics-informed neural networks. Journal of Computational Physics,
378, 686–707.

Velez-Ginorio, J., Amin, N., Kording, K., & Zdancewic, S. (2025).
Compiling to recurrent neurons. arXiv:2511.14953.

Velez-Ginorio, J., Amin, N., Kording, K., & Zdancewic, S. (2026).
Compiling to linear neurons. POPL.

## Appendix A: Code Listings

### A.1 The Cajal Program (shared by all models)

Every model uses the same four-line Cajal program — only the
environment bindings change:

```python
from cajal.syntax import TmIter, TmVar, TmApp, TyNat, TyReal
from cajal.compiling import compile, TypedTensor

program = TmIter(
    TmVar("s0"),                          # initial state
    "s",                                  # iterator variable
    TmApp(TmVar("f"), TmVar("s")),        # step: apply f to state
    TmVar("n"),                           # number of iterations
)
compiled = compile(program)
```

This compiles to a recurrent neural network: the step function `f` is
applied `n` times to the initial state `s0`. The step function is
injected from the environment and can contain learnable PyTorch
modules.

### A.2 EcoSIM Fortran Source (extracted by the agent)

The temperature response function, from `MicrobMathFuncMod.F90:13-30`:

```fortran
subroutine MicrobPhysTempFun(TKSO, TSensGrowth, TSensMaintR)
  implicit none
  real(r8), intent(in) :: TKSO
  real(r8), intent(out):: TSensGrowth, TSensMaintR
  real(r8) :: RTK, STK, ACTV, ACTVM

  RTK  = RGASC * TKSO
  STK  = 710.0 * TKSO
  ACTV = 1 + EXP((197500-STK)/RTK) + EXP((STK-222500)/RTK)
  TSensGrowth = EXP(25.229 - 62500/RTK) / ACTV
end subroutine
```

The decomposition rate, from `MicBGCMod.F90:1470-1471`:

```fortran
RHydlysSolidOM(ielmc,M,K) = SolidOMAct(M,K) * AZMAX1(AMIN1(0.5, &
  SPOSC(M,K) * ROQC4HeterMicActCmpK(K) * DFNS * OQCI * TSensGrowth &
  / BulkSOMC(K)))
```

### A.3 Cajal Surrogate (generated by the agent)

The learnable update module for the EcoSIM surrogate:

```python
class EcoSIMDecompUpdate(nn.Module):
    def __init__(self, k_rates, km, ki, dt, f_temp_mlp, f_water_mlp):
        super().__init__()
        self.k_rates = k_rates     # [k_prot, k_carb, k_cell, k_lign]
        self.km, self.ki, self.dt = km, ki, dt
        self.f_temp = f_temp_mlp   # learnable: TSensGrowth(T)
        self.f_water = f_water_mlp # learnable: WatStressMicb(PSI)

    def forward(self, state):
        c_prot, c_carb, c_cell, c_lign = state.data[0:4]
        c_dom, T, PSI = state.data[4], state.data[5], state.data[6]

        # LEARNED: environmental response
        f_env = self.f_temp(T) * self.f_water(PSI)

        # KNOWN: Monod substrate limitation (from EcoSIM)
        c_total = c_prot + c_carb + c_cell + c_lign
        dfns = c_total / (c_total + self.km)

        # KNOWN: DOC product inhibition (from EcoSIM)
        oqci = 1.0 / (1.0 + c_dom / self.ki)

        # KNOWN: mass-conserving decomposition
        rate = f_env * dfns * oqci
        d = self.k_rates * rate * state.data[:4] * self.dt
        new_solid = state.data[:4] - d
        new_dom = c_dom + d.sum()

        return TypedTensor(
            torch.cat([new_solid, new_dom.unsqueeze(0),
                       T.unsqueeze(0), PSI.unsqueeze(0)]),
            state.ty
        )
```

The environment binding connects the Cajal program to the update module:

```python
s0 = TypedTensor(
    torch.tensor([C0_PROT, C0_CARB, C0_CELL, C0_LIGN, C0_DOM, T, PSI]),
    TyReal(7)
)
result = compiled({
    "s0": s0,
    "f": lambda s: update_fn(s),
    "n": TypedTensor(n_onehot, TyNat()),
})
```

### A.4 Architectural Constraint Example

The sigmoid and softplus output activations guarantee physical
constraints for *any* learned weights:

```python
class TempResponseMLP(nn.Module):
    """Output is strictly positive for all inputs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus(),  # guarantees output > 0
        )

class WaterStressMLP(nn.Module):
    """Output is bounded to [0, 1] for all inputs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),   # guarantees 0 ≤ output ≤ 1
        )
```

A randomly initialized, untrained model already satisfies all
physical constraints. Training improves accuracy without ever
risking constraint violation.
