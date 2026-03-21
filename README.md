# Trusty Neurocoder

Verified scientific surrogates via neuro-symbolic compilation and LLM agents.

**Documentation**: [cmungall.github.io/trusty-neurocoder](https://cmungall.github.io/trusty-neurocoder/)

## What This Does

Takes a scientific simulation kernel (e.g., a Fortran soil decomposition
subroutine), preserves the known physics as fixed program structure, makes
uncertain parts learnable via neural networks, trains against data, and
decompiles the learned weights back to interpretable math. Physical
invariants (mass conservation, positivity) hold by construction.

```
Scientific source code (Fortran, C++, Python)
    ↓  LLM agent extracts kernel
Cajal program (typed functional language)
    ↓  compiler
PyTorch computation graph (differentiable)
    ↓  train against data
Learned neural weights
    ↓  symbolic regression
Interpretable mathematical expression + verified invariants
```

## Results

Eight working demonstrations across DOE science domains:

| Model | Domain | Learned | Result |
|-------|--------|---------|--------|
| Exponential decay | Foundation | rate k | k=0.3000 exact |
| Coupled pools | Earth science | transfer α | α=0.4000 exact |
| Unknown function | Earth science | moisture response | Hill equation recovered |
| CENTURY-Lite | Earth science | temp + moisture | both forms recovered |
| Decay chain | Nuclear | branching ratios | 0.70, 0.85 exact |
| Battery fade | Energy storage | SEI growth law | parabolic law recovered |
| Chemical kinetics | Combustion | Arrhenius rate | A=2.01, E=4.99 |
| EcoSIM decomp | Earth science | T + water stress | extracted from Fortran |

### Comparison: Cajal vs PINN vs Black-Box

On identical data (reversible reaction A⇌B):

| | Black-box | PINN | **Cajal** |
|---|---|---|---|
| Trajectory MSE | 6.2×10⁻³ | 7.7×10⁻³ | **9.3×10⁻⁷** |
| Conservation error | 5.4×10⁻³ | 6.5×10⁻³ | **1.6×10⁻⁷** |
| Extrapolation | 3.1×10⁻² | 5.0×10⁻² | **1.3×10⁻²** |
| Sample efficiency (2 traj) | 8.6×10⁻² | — | **1.2×10⁻³** |
| Interpretable | No | No | k=1.97·exp(-4.86/T) |

## Quick Start

```bash
git clone https://github.com/cmungall/trusty-neurocoder.git
cd trusty-neurocoder
uv pip install -e ".[dev,notebooks,docs]"

# Run examples
just examples

# Run notebooks
just notebooks

# Serve docs
just docs
```

## Notebooks

Interactive Jupyter notebooks with embedded output and plots:

- [01 - Cajal Intro](https://cmungall.github.io/trusty-neurocoder/notebooks/01_cajal_intro/)
- [02 - Exponential Decay](https://cmungall.github.io/trusty-neurocoder/notebooks/02_exponential_decay/)
- [03 - Learning Unknown Functions](https://cmungall.github.io/trusty-neurocoder/notebooks/03_learn_unknown_function/)
- [04 - CENTURY-Lite](https://cmungall.github.io/trusty-neurocoder/notebooks/04_century_lite/)
- [05 - Decay Chain](https://cmungall.github.io/trusty-neurocoder/notebooks/05_decay_chain/)
- [06 - Battery Degradation](https://cmungall.github.io/trusty-neurocoder/notebooks/06_battery_degradation/)
- [07 - Chemical Kinetics](https://cmungall.github.io/trusty-neurocoder/notebooks/07_chemical_kinetics/)

## Cajal Type System

Built on [Cajal](https://arxiv.org/abs/2511.14953) (Velez-Ginorio, Amin,
Kording, Zdancewic), a typed linear programming language whose programs
compile exactly to recurrent neural networks. We extend the type system
with `TyReal(n)` for real-valued state vectors and fix four soundness
bugs in the vendored implementation.

## References

- Velez-Ginorio et al. "Compiling to Recurrent Neurons" (arXiv:2511.14953, 2025)
- Velez-Ginorio et al. "Compiling to Linear Neurons" (POPL 2026)
- Amin & Rompf. "Collapsing Towers of Interpreters" (POPL 2018)

## License

BSD-3-Clause
