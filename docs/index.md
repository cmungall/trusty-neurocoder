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

## Notebooks

The notebooks demonstrate a progression from basic Cajal compilation to
learning unknown functions in a multi-pool soil carbon model:

| Notebook | Description |
|----------|-------------|
| [01 - Cajal Intro](notebooks/01_cajal_intro.ipynb) | Boolean functions compile to matrices; iteration as recurrent neuron |
| [02 - Exponential Decay](notebooks/02_exponential_decay.ipynb) | Learn scalar ODE rate constant from data |
| [03 - Unknown Function](notebooks/03_learn_unknown_function.ipynb) | MLP learns unknown moisture response; symbolic regression recovers Hill equation |
| [04 - CENTURY-Lite](notebooks/04_century_lite.ipynb) | 3-pool model, 2 unknown functions learned simultaneously, mass conservation verified |

## Quick Start

```bash
# Install
uv pip install -e ".[notebooks,docs]"

# Run all notebooks
just notebooks

# Serve docs locally
just docs
```

## Architecture

```
┌──────────────────────────────────────┐
│  Layer 1: LLM Agent                  │
│  - Parses real scientific code       │
│  - Extracts algorithmic kernels      │
│  - Translates to Cajal programs      │
├──────────────────────────────────────┤
│  Layer 2: NSAM Compilation           │
│  - Cajal program → PyTorch RNN       │
│  - Learnable sub-expressions (MLPs)  │
│  - Backprop through compiled program │
├──────────────────────────────────────┤
│  Layer 3: Verification               │
│  - Structural invariants by design   │
│  - Symbolic regression (decompile)   │
│  - Mass conservation, positivity     │
└──────────────────────────────────────┘
```
