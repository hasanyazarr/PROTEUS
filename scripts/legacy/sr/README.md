# Legacy Super-Resolution Tools

This directory preserves the Python analysis tools used by the former
CNN-style super-resolution dataset workflow. The MATLAB exporter is preserved
as the `legacy.sr` package under `scripts/+legacy/+sr/`.

These tools are not part of the active ULM research pipeline. Active work uses
simulation settings, ground truth, RF data, and Casorati matrices directly.
Use `dataset_export` only when reproducing a historical `dataset_sr` artifact;
it emits a deprecation warning and delegates to `legacy.sr.process_run`.

Existing `dataset_sr` outputs are research evidence and must not be deleted or
rewritten as part of this migration.
