# Trusty Neurocoder

Neuro-Symbolic Agents for Verified Scientific Code Generation.

## Overview

Trusty Neurocoder combines LLM-based agentic workflows with
[Neuro-Symbolic Abstract Machines (NSAMs)](https://metareflection.seas.harvard.edu/research/neuro/)
to enable verified scientific code generation, optimization, and surrogate
construction.

**NSAMs** are neural networks structurally equivalent to programming language
interpreters. They enable principled compilation of symbolic programs into
neural architectures and decompilation back to interpretable code.
**LLM agents** bridge the gap between real-world scientific codebases and the
declarative representations NSAMs require.

The key insight is that agents and NSAMs compensate for each other's weaknesses:

- LLM agents handle messy real-world code comprehension that NSAMs cannot
- NSAMs provide formal correctness guarantees that LLM agents cannot

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: LLM Agent                                 │
│  - Parses real scientific code (Fortran, C++, etc.) │
│  - Extracts algorithmic kernels and intent          │
│  - Translates to declarative representation         │
│  - Orchestrates the pipeline                        │
│  - Uses domain ontologies for semantic grounding    │
├─────────────────────────────────────────────────────┤
│  Layer 2: NSAM Compilation/Decompilation            │
│  - Compiles declarative programs → neural networks  │
│  - Fixed structure for known physics                │
│  - Learnable weights for unknown sub-expressions    │
│  - Decompiles trained networks → symbolic programs  │
│  - Based on collapsing towers / staged interpreters │
├─────────────────────────────────────────────────────┤
│  Layer 3: Formal Verification                       │
│  - Checks invariants (conservation, symmetry, etc.) │
│  - Verifies equivalence pre/post optimization       │
│  - Validates against reference outputs              │
│  - Coq/Dafny proof generation via LLM guidance      │
└─────────────────────────────────────────────────────┘
```

## The Cajal Language

The NSAM layer is built on [Cajal](https://arxiv.org/abs/2511.14953), a minimal
typed, higher-order, linear programming language whose programs compile
correctly to linear (recurrent) neurons.

```
Expressions:
  e ::= x                          -- variable
      | tt | ff                    -- booleans
      | 0 | succ(e)               -- natural numbers
      | iter{e₁ | y ↪ e₂}(e₃)    -- iterator
      | λx.e                      -- linear map
      | e₁ e₂                     -- application

Types:
  τ ::= 𝟚              -- boolean
      | ℕ              -- natural number
      | τ₁ ⊸ τ₂        -- linear map
```

Programs compile to weight matrices. For example:

| Cajal program | Compiles to |
|---------------|-------------|
| `λx. x` (identity) | `[[1,0],[0,1]]` |
| `λx. if x then ff else tt` (NOT) | `[[0,1],[1,0]]` |
| `λx. if x then tt else tt` (const-true) | `[[1,1],[0,0]]` |
| `iter{tt \| y → not(y)}(n)` | n-th power of the NOT matrix |

The linear type system ensures each variable is used exactly once, which maps
directly to linear algebra: compilation preserves semantics by construction.

## Demonstration Use Cases

### 1. Verified Surrogate Models (EcoSIM, Fortran)

Extract a soil carbon decomposition kernel from the
[EcoSIM](https://github.com/jinyun1tang/EcoSIM) biogeochemistry model. Known
physics (mass balance, Arrhenius temperature response) becomes fixed neural
structure; uncertain process terms (moisture response function) become
learnable weights. After training, decompile the learned sub-expression back
to an interpretable symbolic formula and verify conservation invariants.

### 2. Algorithm Selection (ECP Proxy App, C/C++)

Analyze [XSBench](https://github.com/ANL-CESAR/XSBench) or
[CoMD](https://github.com/ECP-copa/CoMD) to extract algorithm dispatch logic.
Compile the discrete algorithm choice (unionized grid vs. hash vs. nuclide
lookup) into a continuous, differentiable selection via NSAM. Optimize via
gradients, decompile to an interpretable decision rule, verify correctness
against built-in reference outputs.

### 3. Program Synthesis (NAS Parallel Benchmarks, Fortran)

Given a mathematical specification (e.g., Conjugate Gradient for Ax=b),
synthesize a correct implementation using NSAM-guided relational search via
staged miniKanren. Verify the synthesized code against
[NPB](https://www.nas.nasa.gov/software/npb.html) reference outputs at
multiple problem sizes.

## Running the Examples

```bash
# Clone and set up
git clone https://github.com/cmungall/trusty-neurocoder.git
cd trusty-neurocoder
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run the Cajal demo
python examples/cajal_demo.py
```

## Prior Work

This project builds on:

- [C3PO](https://github.com/chemkg/c3p) (Mungall et al., J. Cheminformatics
  2025) -- LLM-based synthesis of deterministic, verifiable chemical classifier
  programs from ontology definitions. Demonstrates the neuro→symbolic code
  generation pattern that Trusty Neurocoder formalizes with NSAM guarantees.

## References

- Velez-Ginorio, Amin, Kording, Zdancewic. "Compiling to Linear Neurons" (POPL 2026)
- Velez-Ginorio, Amin, Kording, Zdancewic. "Compiling to Recurrent Neurons" (arXiv:2511.14953, 2025)
- Amin & Rompf. "Collapsing Towers of Interpreters" (POPL 2018)
- Ballantyne, Sanna, Hemann, Byrd, Amin. "Multi-stage Relational Programming" (PLDI 2025)
- Prasad & Amin. "Guided Proof Search Using LLMs and Lemma Extraction in Coq" (ICLR VerifAI 2025)
- Mungall et al. "Chemical classification program synthesis using generative artificial intelligence" (J. Cheminformatics 2025)

## License

BSD-3-Clause
