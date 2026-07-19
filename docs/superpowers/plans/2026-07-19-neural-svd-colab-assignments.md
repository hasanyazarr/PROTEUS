# NeuralSVD Colab Assignments Implementation Plan

> **For agentic workers:** This plan produces learner-facing exercises. Do not implement the scientific exercise blocks for the learner. Provide progressively stronger hints only when requested.

**Goal:** Deliver a seven-cell, CS231n-style Colab sequence that teaches and tests a single-matrix NeuralSVD feasibility pilot on `casorati_500.npy`.

**Architecture:** Each assignment is exactly one independently runnable Colab cell. Boilerplate, diagnostics, assertions, timing, and reporting are provided; the learner implements only the scientific core. Each cell exports a small, named state contract consumed by the next cell.

**Tech Stack:** Google Colab, Python, NumPy, PyTorch, CUDA, Matplotlib, pandas.

## Global Constraints

- Keep each assignment to exactly one Colab cell.
- Use `/content/drive/MyDrive/casorati_500.npy` as the default input.
- Use `float32` until correctness is established.
- Use target rank `L = 8`.
- Never reveal an exercise solution unless the learner explicitly requests it.
- Assistance order: conceptual hint, pseudocode, localized implementation hint, full solution.
- Every cell must contain local assertions and must pass before continuing.
- Report training time separately from post-training inference time.

---

### Assignment 1: Casorati Validation and Scale

**Status:** Completed interactively.

**Produces:**

- `data: np.ndarray` with shape `(1_826_304, 500)` and dtype `float32`;
- `data_scale: float`, equal to the recorded global standard deviation;
- validation results for finite values, shape, distribution, and Frobenius norm.

**Acceptance checks:**

- Shape and dtype match expectations.
- NaN and Inf counts are zero.
- Median is near zero.
- `||A||_F` agrees with `std(A) * sqrt(A.size)` to numerical tolerance.
- Normalization is defined as division by `data_scale`, without centering or clipping.

### Assignment 2: Exact and Truncated SVD References

**Consumes:** `data`, `data_scale`.

**Produces:**

- `A: torch.Tensor` on CUDA;
- `U_gold`, `s_gold`, `V_gold` for the exact economy reference;
- a corrected stable randomized/truncated SVD baseline;
- runtime and peak-memory measurements for both methods.

**Learner exercise:** Identify which outputs contain the leading `L` components and extract them without claiming that slicing reduces exact-SVD computation.

**Acceptance checks:**

- Reconstruction shapes are valid.
- Singular values are finite, non-negative, and non-increasing.
- The randomized implementation orthonormalizes between power iterations.
- Rank-2 and rank-3 reference reconstructions are finite.

### Assignment 3: Sampled Casorati Operator

**Consumes:** CPU-backed `data`, `data_scale`, CUDA device.

**Produces interface:**

```python
sample_casorati_block(num_rows: int, num_frames: int, generator: torch.Generator)
    -> tuple[row_coord, frame_coord, target, row_idx, frame_idx]
```

**Learner exercise:** Implement unique index sampling, block extraction, scalar normalization, coordinate normalization to `[-1, 1]`, and CUDA transfer.

**Acceptance checks:**

- Shapes are `(R, 1)`, `(F, 1)`, and `(R, F)`.
- Indices are in range and unique within each sampled axis.
- Returned target equals the indexed NumPy block divided by `data_scale`.
- Same seed reproduces a block; a different seed changes it.

### Assignment 4: Neural Singular Functions

**Consumes:** normalized row and frame coordinates.

**Produces interface:**

```python
left_model(row_coord: torch.Tensor) -> torch.Tensor   # (R, L)
right_model(frame_coord: torch.Tensor) -> torch.Tensor # (F, L)
```

**Learner exercise:** Define two small MLPs with `L=8` outputs and choose activations suitable for continuous coordinates.

**Acceptance checks:**

- Output shapes match the contracts for multiple batch sizes.
- Forward and backward passes are finite.
- Models do not share parameters accidentally.
- Parameter count and one-forward latency are reported.

### Assignment 5: Nested Low-Rank Objective

**Consumes:** `left_model`, `right_model`, sampled `target` blocks.

**Produces interface:**

```python
nested_low_rank_loss(left: torch.Tensor, right: torch.Tensor, target: torch.Tensor)
    -> tuple[loss: torch.Tensor, diagnostics: dict[str, torch.Tensor]]
```

**Learner exercise:** Translate the paper's nested low-rank approximation into an ordered rank-prefix objective so prefixes `1..L` contribute to training.

**Acceptance checks:**

- Scalar loss is finite and differentiable.
- Permuting components changes the ordered objective when component quality differs.
- A synthetic known rank-3 matrix obtains lower loss from its correct factors than from random factors.
- Gradients reach both models.

### Assignment 6: Training and Checkpointing

**Consumes:** sampler, models, nested objective.

**Produces:**

- deterministic training history;
- best validation checkpoint;
- training wall time and peak GPU memory;
- explicit-factor low-rank control trained under the same sampling budget.

**Learner exercise:** Implement the optimizer step, validation boundary, best-checkpoint selection, and early divergence detection.

**Acceptance checks:**

- A tiny synthetic operator overfits before real-data training.
- Real-data training loss decreases from its initial value.
- NaN/Inf aborts with the failing step recorded.
- Reloading the best checkpoint reproduces its validation result.

### Assignment 7: Fidelity, Runtime, and Go/No-Go

**Consumes:** exact reference, randomized baseline, NeuralSVD checkpoint, explicit-factor control.

**Produces:**

- singular-value relative error;
- subspace distance at ranks 2, 3, and 8;
- clutter reconstruction error;
- filtered-output relative error for cutoffs 2 and 3;
- training, inference, filtering, and peak-memory table;
- saved CSV/JSON metrics and plots;
- final pass/fail verdict.

**Learner exercise:** Implement the metric aggregation and explain whether errors come from singular values, singular subspaces, or reconstruction.

**Acceptance checks:**

- Metrics are invariant to paired singular-vector sign flips.
- Identical inputs yield zero error within tolerance.
- Deliberately corrupted rank-2 factors fail the subspace check.
- Feasibility passes only if rank-2/rank-3 subspace distance is below `0.05`, mean leading singular-value relative error is below `5%`, filtered-output relative error is below `1%`, and convergence remains finite.

## Delivery Order

1. Treat Assignment 1 as completed.
2. Deliver Assignment 2 as one learner-facing cell.
3. Review its output before authoring or delivering Assignment 3.
4. Repeat this review gate through Assignment 7.
5. Do not integrate with MATLAB or begin multi-case training during this pilot.
