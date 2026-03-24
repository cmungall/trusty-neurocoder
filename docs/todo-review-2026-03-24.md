# Review — 2026-03-24

Internal review of the trusty-neurocoder prototype.

## Summary

Strong prototype with a genuinely novel angle. The Cajal compilation pipeline + LLM agent workflow fills a real gap between black-box surrogates and physics-informed methods. The PINN comparison results are compelling. Main gap: the EcoSIM end-to-end case and symbolic regression need to close the loop from "works on synthetics" to "useful for real DOE simulations."

## Strengths

- **Core idea is novel and well-positioned.** Structural correctness by construction (not regularization) differentiates cleanly from PINNs.
- **Results are strong for a prototype.** 6,700× lower interpolation error vs PINNs, exact conservation (10⁻⁸ vs 10⁻²), 8 working demos.
- **Code is solid.** 19 tests all pass. Clean PL implementation (syntax/typing/evaluating/compiling). TyReal extension is a real contribution to Cajal.
- **Paper draft is well-written.** Clear problem statement, honest about limitations, good comparison framing.

## Concerns

### The `trusty_neurocoder` package is empty
All code lives in `cajal/`. The top-level package that ties agent workflow + Cajal + symbolic regression into a coherent tool doesn't exist. Should be Phase 1 deliverable.

### No symbolic regression implementation
The paper describes the full pipeline (compile → train → decompile to math), but the decompilation step isn't implemented. The "recover the Hill equation" claim needs actual symbolic regression code. Consider PySR or a simple approach — just close the loop.

### EcoSIM integration is incomplete
The other demos are clean synthetic cases. EcoSIM is the real-world differentiator but is still TODO (see `docs/todo-ecosim-integration.md`, `docs/todo-ecosim-nersc.md`). This is where credibility lives for DOE reviewers.

### Lean spec is a scaffold
`lean-spec/` has empty files. If formal verification is part of the story, needs more substance — or defer it explicitly to Phase 2.

### "Single session" framing
"The entire codebase was produced by the LLM agent in a single interactive session" demonstrates the agent workflow but could read as "weekend hack" to skeptical reviewers. Lead with methodology and results, mention agent workflow as a bonus.

### Dev dependencies
`pytest` not in `[dev]` extras — had to install manually. CI workflow exists but won't run cleanly from a fresh clone.

## Recommendations

1. **Fix dev dependencies** — add pytest to `[dev]` in pyproject.toml
2. **Prioritize EcoSIM** — it's the differentiator over synthetic demos
3. **Implement symbolic regression end-to-end** — even one example closing the loop
4. **De-emphasize "single session" in the paper** — lead with results
5. **Consider naming for DOE audience** — "Trusty Neurocoder" is catchy but the Cajal extensions and NSAM pipeline are serious PL/ML work
