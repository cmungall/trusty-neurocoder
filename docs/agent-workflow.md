# Agent Workflow: From Natural Language to Verified Surrogate

This document demonstrates the full Trusty Neurocoder pipeline, where an
LLM coding agent (Claude Code) takes a natural language description of a
scientific model and produces a working, verified neuro-symbolic surrogate.

**This is not a hypothetical workflow.** The repository documents a real
agent-assisted build, starting from a DOE RFA and a pointer to the Cajal
papers.

## The Three-Layer Architecture in Practice

```
User prompt (natural language)
        ↓
┌─────────────────────────────────────┐
│  Layer 1: LLM Coding Agent          │
│  (Claude Code)                      │
│                                     │
│  • Reads RFA, papers, source code   │
│  • Designs Cajal program structure  │
│  • Writes PyTorch learnable modules │
│  • Generates training + eval code   │
├─────────────────────────────────────┤
│  Layer 2: NSAM Compilation          │
│  (Cajal → PyTorch)                  │
│                                     │
│  • Cajal iteration → recurrent net  │
│  • Learnable sub-expressions (MLPs) │
│  • Backprop through compiled prog   │
├─────────────────────────────────────┤
│  Layer 3: Verification              │
│  (structural + empirical)           │
│                                     │
│  • Architecture guarantees (>0, ∈[0,1]) │
│  • Mass/energy conservation         │
│  • Symbolic regression → decompile  │
└─────────────────────────────────────┘
        ↓
Verified surrogate with symbolic interpretation
```

## Session Transcript

### Step 1: Agent reads the RFA and background literature

**User:** "This repo is for brainstorming ideas around Genesis proposals.
Please read the PDF in assets."

The agent read DE-FOA-0003612 (DOE Genesis Mission RFA, $293.76M),
identified Topic 18C (Neuro-Symbolic Agents for Code Development), and
summarized the requirements.

### Step 2: Agent researches the foundational work

**User:** "What are some foundational works in this area? Look for papers
by Nada Amin."

The agent found the Cajal language (Velez-Ginorio, Amin, Kording, Zdancewic),
understood the compilation pipeline (programs → recurrent neural networks),
and identified the key repos on GitHub.

### Step 3: Agent drafts the proposal

**User:** "I need a proposal where we can deliver on a proof of concept
in 6 months."

The agent drafted a full proposal ("Trusty Neurocoder"), identifying three
DOE science use cases and the three-layer architecture.

### Step 4: Agent builds the repository

**User:** "Let's make a fresh repo, copy across relevant info, and push
to GitHub."

The agent:

- Created the repo structure
- Vendored the Cajal source (MIT license)
- Installed dependencies via `uv`
- Pushed to github.com/cmungall/trusty-neurocoder

### Step 5: Agent builds progressively complex examples

Each example was produced by a natural language prompt. The agent wrote the
code, debugged errors, ran the training, and verified the results.

#### Example 1: Exponential Decay

**User:** "What can we do next to make this less of a toy application?"

**Agent produced:** `exponential_decay.py` — a 1D ODE `dC/dt = -kC` where
the rate constant k is learned from trajectory data via a Cajal iteration
program. The agent:

- Designed the Cajal program: `iter{C₀ | c ↪ f(c)}(n)`
- Implemented `f(c) = w·c` as a learnable scalar
- Recovered k=0.3000 exactly
- Decompiled: `C(n) = 1.0 × (0.9700)^n → dC/dt = -0.3000·C`

#### Example 2: Coupled Carbon Pools

**Agent produced:** `coupled_decay.py` — a 2-pool coupled ODE system with
unknown transfer coefficient α. Extended the pattern to multi-dimensional
state. Recovered α=0.4000 exactly. Verified mass conservation.

#### Example 3: Unknown Nonlinear Function

**User:** "Can we expand this out to an even bigger example?"

**Agent produced:** `learn_unknown_function.py` — embedded a 1153-parameter
MLP as a learnable sub-expression inside a Cajal iteration. The MLP learned
the unknown function `f_moisture(m) = m^0.7/(0.3+m^0.7)` from 20 trajectory
examples. Symbolic regression correctly recovered the Hill equation.

#### Example 4: CENTURY-Lite Multi-Pool Model

**Agent produced:** `century_lite.py` — a 3-pool soil carbon model
(simplified CENTURY) with **two** unknown environmental response functions
(temperature Q10 and moisture Hill equation) learned simultaneously from 9
trajectories. Both functional forms correctly identified by symbolic
regression. All physical invariants verified.

#### Examples 5–7: DOE Science Domains

**User:** "Let's keep going, some other DOE/LBNL use cases!"

The agent produced three more examples in parallel:

| Example | DOE Domain | What's Learned | Result |
|---------|-----------|----------------|--------|
| `decay_chain.py` | Nuclear Science | Branching ratios in A→B→C→D | 0.70, 0.85 recovered exactly |
| `battery_degradation.py` | Energy Storage | SEI growth + capacity fade laws | Parabolic growth law recovered |
| `chemical_kinetics.py` | Combustion | Arrhenius rate k=A·exp(-E/T) | A=2.006, E=4.987 (true: 2.0, 5.0) |

### Step 6: Agent extends the Cajal type system

**User:** "I thought we extended to work with reals?"

The agent realized the examples were all hacking `TyBool()` for real-valued
state vectors, and implemented a proper `TyReal(n)` type:

- Added `TyReal` dataclass to `cajal/syntax.py`
- Added `dim()` and `bases()` support in `cajal/compiling.py`
- Updated all 7 examples and 7 notebooks

This is a genuine extension to Cajal beyond the original boolean+natural
type system.

### Step 7: Agent builds documentation site

**User:** "I like to see my notebooks in mkdocs sites."

The agent set up mkdocs with Material theme, mkdocs-jupyter for notebook
rendering, GitHub Actions for auto-deployment, and configured GitHub Pages.

## What the Agent Did NOT Do

The agent's role was to bridge between:

- **Natural language intent** ("build a soil carbon model with unknown
  moisture response")
- **Cajal program structure** (iteration, state layout, learnable
  sub-expressions)
- **PyTorch implementation** (MLPs, training loops, optimizers)
- **Verification code** (mass conservation, positivity, symbolic regression)

The agent did NOT:

- Prove correctness theorems (that's Cajal's job via its type system)
- Guarantee the learned functions are globally optimal
- Replace domain expertise (the user guided the choice of scientific models)
- Invent new mathematics (it applied existing Cajal compilation theory)

## Key Insight

The agent and the NSAM compensate for each other's weaknesses:

| Capability | Agent | NSAM |
|-----------|-------|------|
| Read messy real-world code | ✓ | ✗ |
| Understand natural language specs | ✓ | ✗ |
| Generate boilerplate + training code | ✓ | ✗ |
| Structural correctness guarantees | ✗ | ✓ |
| Mass/energy conservation by design | ✗ | ✓ |
| Decompilation to symbolic form | ✗ | ✓ |

Neither alone is sufficient. Together, they enable verified scientific
surrogate construction from natural language descriptions.

## Reproducing This Workflow

The full session transcript is available. Every example was produced by:

1. A natural language prompt from the user
2. The agent reading existing code for patterns
3. The agent writing new code following those patterns
4. The agent running the code and verifying results
5. The agent committing and pushing to GitHub

The entire repository — 7 examples, 7 notebooks, type system extension,
documentation site, CI/CD — was produced in a single session.
