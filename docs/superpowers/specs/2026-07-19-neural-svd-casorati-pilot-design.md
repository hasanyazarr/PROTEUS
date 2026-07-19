# NeuralSVD Casorati Pilot Design

## Objective

Evaluate whether the ICML 2024 NeuralSVD approach can learn the leading singular components of one real PROTEUS Casorati matrix before investing in a conditional, cross-sequence real-time model.

This pilot is a feasibility test, not a production real-time implementation. Training may take minutes. Training time and post-training inference time will be reported separately because only inference is eventually required to satisfy the `<0.3 s` SVD-stage target.

## Input and Reference

- Input: `/content/drive/MyDrive/casorati_500.npy`.
- Expected shape: approximately `1,826,304 x 500`.
- Working precision: `float32` after safe normalization.
- Target rank: `L = 8`, allowing inspection of a possible two- or three-component tissue subspace.
- Gold reference: exact economy SVD, computed once and used only for evaluation. It is not exposed to NeuralSVD during training.

## Compared Methods

### Coordinate NeuralSVD

Adapt NeuralSVD's nested low-rank objective to the finite Casorati operator. Two neural functions represent the leading left and right singular functions:

- `u_theta(row_coordinate) -> L values`
- `v_phi(frame_coordinate) -> L values`

Training samples row and frame indices and evaluates only the corresponding matrix block. The model learns an ordered rank-`L` approximation without materializing a training batch covering the entire Casorati matrix.

### Direct Low-Rank Factorization Control

Optimize explicit rank-`L` factors against the same sampled blocks. This control distinguishes failure of the NeuralSVD parameterization from failure of the sampling, normalization, or loss formulation. It is not presented as the research contribution.

### Exact SVD Reference

Use the existing exact SVD path to supply reference singular values, subspaces, and clutter-filtered output. Runtime is measured but the method is not part of training.

## Data Flow

1. Load the memory-mapped NumPy matrix and transfer or sample it without modifying the read-only backing array.
2. Validate shape, dtype, finite values, Frobenius norm, and dynamic range.
3. Normalize the matrix by one recorded scalar so training remains numerically stable.
4. Compute and retain the leading eight exact singular triplets for evaluation.
5. Train Coordinate NeuralSVD from randomly sampled row/frame blocks.
6. Train the direct low-rank control with the same sampling budget.
7. Evaluate both methods against exact SVD.
8. Save metrics, checkpoints, configuration, timing, and GPU-memory results to Google Drive.

## Sampling and Model Defaults

- Rank: `L = 8`.
- Sampled rows per step: begin at `4096`.
- Sampled frames per step: begin at `32`.
- Coordinate inputs: normalized continuous row and frame coordinates.
- Model: small MLPs for left and right singular functions, with identical output rank.
- Optimization: Adam with deterministic seeds and validation checkpoints.
- The first run uses conservative `float32`; mixed precision is deferred until correctness is established.

Defaults are configuration values, not claims of optimality. A smoke test on a smaller row subset must pass before full-matrix sampling begins.

## Evaluation

The pilot records:

- training and validation loss;
- per-component singular-value relative error;
- subspace distance for ranks 2, 3, and 8;
- leading-rank clutter reconstruction error;
- relative error of the SVD-filtered Casorati output for cutoffs 2 and 3;
- training wall time;
- post-training inference and filtering time;
- peak GPU memory;
- convergence stability across at least three random seeds for the final candidate.

Sign and rotations within nearly degenerate singular subspaces are handled by subspace and reconstruction metrics rather than element-wise comparison of singular vectors.

## Go/No-Go Criteria

The method passes the single-matrix feasibility gate when the final candidate achieves all of the following on held-out sampled entries or blocks:

- rank-2 and rank-3 subspace distance below `0.05`;
- mean leading singular-value relative error below `5%`;
- filtered-output relative error below `1%` for cutoff 2 and reported separately for cutoff 3;
- stable convergence without NaN, Inf, or singular-component collapse;
- materially lower post-training inference cost than exact SVD, with the eventual `<0.3 s` target reported but not required for this feasibility gate.

Failure is informative. If direct low-rank factorization succeeds but Coordinate NeuralSVD fails, the neural coordinate representation is the likely limitation. If both fail, sampling, normalization, or the objective must be investigated before increasing model capacity.

## Outputs

The Colab experiment will produce:

- one self-contained sequence of copy-pasteable notebook cells;
- a CSV or JSON metrics report;
- loss and fidelity plots;
- saved model checkpoints and run configuration;
- a concise pass/fail summary comparing NeuralSVD, direct factorization, and exact SVD.

## Out of Scope

- Integration into MATLAB `process_run.m`;
- automatic SVD cutoff selection;
- training across multiple PROTEUS cases;
- direct neural clutter filtering;
- BF16/FP16 optimization;
- claims of real-time generalization to unseen sequences.

These become follow-up work only if the single-matrix pilot passes.
