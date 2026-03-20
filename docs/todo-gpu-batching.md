# TODO: GPU/Batching Support

## Problem

Training loops make serial, tiny calls to the compiled Cajal program — one per (trajectory, timestep). Each call runs a small MLP on a single scalar. GPU kernel launch overhead dominates, making CPU ~10x faster than MPS/CUDA for current workloads.

## Approaches

### A. Batch the outer loop (hours)

Stack trajectory states into `(batch, state_dim)` tensors. One forward pass per timestep instead of per (trajectory × timestep).

- [ ] `TypedTensor` supports batch dimension
- [ ] Update modules operate on `(B, state_dim)` not `(state_dim,)`
- [ ] MLPs take `(B, 1)` input
- [ ] Training loop: one `compiled()` call per timestep

Cuts calls from `N_traj × N_steps` to `N_steps` per epoch. For CENTURY-lite: 90 → 10.

### B. `torch.vmap` auto-vectorization (quick experiment)

The compiled Cajal program is a pure function. `torch.vmap` can auto-vectorize across batch dimensions. `test20()` in vendored `compiling.py` already uses vmap — Joey was thinking about this.

- [ ] Try vmap on a simple example (exponential_decay)
- [ ] Check compatibility with TypedTensor (NamedTuple) and dynamic control flow
- [ ] If it works, apply to all examples

Risk: vmap can be finicky with custom types and closures.

### C. Compile iteration to `nn.RNN` (Phase 1 deliverable)

Cajal iteration IS a recurrent neural network by construction. Emit an actual PyTorch RNN module instead of interpreting step-by-step in Python.

- [ ] Design the compilation from `TmIter` → `nn.Module` with batched forward
- [ ] Handle learnable sub-expressions as RNN cell components
- [ ] Standard PyTorch batching, GPU, `torch.compile` all work automatically
- [ ] Benchmark against interpreted version

This is the architecturally correct solution and a strong Phase 1 result.

## Priority

Try B first (quick). Fall back to A if vmap doesn't work. C is the real research contribution for the proposal.
