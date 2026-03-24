# Review Follow-Up — 2026-03-24

Internal status tracker for the March 24 review of the trusty-neurocoder
prototype.

## Summary

The original review surfaced one real repo gap, two framing issues, and
three stale observations. This follow-up distinguishes what is now fixed,
what was already outdated in the current tree, and what remains the main
Phase 1 credibility gap for DOE audiences.

## Status Snapshot

- **Addressed:** `trusty_neurocoder` now exposes reusable symbolic-fitting utilities instead of being an empty package.
- **Addressed:** candidate-based symbolic fitting is now implemented as a package API rather than only as inline example code.
- **Addressed:** paper/proposal language now presents the agent workflow as methodology support, not as the headline scientific claim.
- **Already fixed when reviewed:** `pytest` is present in `[project.optional-dependencies].dev`.
- **Not applicable in current tree:** there is no `lean-spec/` directory to substantiate or defer.
- **Still open:** EcoSIM remains the key end-to-end credibility milestone.

## Concern Status

### The `trusty_neurocoder` package is empty
**Status:** Addressed.

The repo now contains a reusable top-level API in `src/trusty_neurocoder/`
for symbolic fitting and decompilation helpers. This does not yet constitute
the full Phase 1 workflow package, but it moves the project from
example-only logic toward an actual library surface.

### No symbolic regression implementation
**Status:** Addressed for the prototype; still open for richer search.

The examples already contained candidate-family fitting logic. That logic is
now consolidated into `trusty_neurocoder.symbolic` with reusable functions
for one- and two-parameter candidate search plus common moisture and
temperature response families. This closes the loop for the current claims.

What remains open is broader symbolic search beyond curated candidate
families, for example integrating a tool such as PySR.

### EcoSIM integration is incomplete
**Status:** Still open.

This remains the main substantive gap. The synthetic demonstrations and
small extracted kernels are useful, but the strongest DOE-facing story is
still a more complete EcoSIM case that closes the loop from extracted kernel
to training data to learned expression to validation.

### Lean spec is a scaffold
**Status:** Stale review item.

The current tree does not contain a `lean-spec/` scaffold. If formal
verification is reintroduced into the roadmap, it should be described as
Phase 2 work unless a substantive artifact is added.

### "Single session" framing
**Status:** Addressed.

The paper and proposal now lead with method and results. Agent assistance is
still documented, but it is framed as implementation methodology rather than
the primary evidence of value.

### Dev dependencies
**Status:** Stale review item.

`pytest` is already present in the `dev` extra in `pyproject.toml`, so this
issue was resolved before this follow-up pass.

## Updated Priorities

1. **Prioritize EcoSIM** by turning the extracted kernel into a fuller end-to-end case with calibration and validation.
2. **Expand symbolic fitting carefully** if the scientific story requires discovery outside curated candidate families.
3. **Keep claims proportional to evidence** by separating implemented prototype results from Phase 2 aspirations.
4. **Continue packaging workflow utilities** in `trusty_neurocoder/` as example code stabilizes into reusable components.
