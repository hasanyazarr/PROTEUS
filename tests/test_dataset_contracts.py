from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_load_rf_data_returns_frame_identity_and_pulse_metadata():
    src = read("delay-and-sum/load_RF_data.m")

    assert "function [RF_matrix, FrameNumbers, RFFileNames, PulseInfo]" in src
    assert "RFFileNames" in src
    assert "PulseInfo.PulsingScheme" in src
    assert "PulseInfo.PulseIDsUsed" in src
    assert "PulseInfo.CombinationFormula" in src


def test_process_run_uses_source_frame_numbers_for_gt_export():
    src = read("scripts/process_run.m")

    assert "[RF, sourceFrameNumbers, sourceRFFileNames, pulseInfo] = load_RF_data" in src
    assert "source_frame = sourceFrameNumbers(iframe);" in src
    assert "source_frame, npad_gt" in src
    assert "sample_metadata.source_frame_number" in src
    assert "metadata.source_frame_numbers" in src


def test_process_run_records_split_aware_preprocessing_state():
    src = read("scripts/process_run.m")

    assert "fit_frame_mask" in src
    assert "PreprocessingState.SVDFitFrameNumbers" in src
    assert "PreprocessingState.NormalizationMode" in src
    assert "PreprocessingState.NormalizationReference" in src
    assert "metadata.preprocessing = PreprocessingState" in src


def test_process_run_exports_pulse_aware_labels():
    src = read("scripts/process_run.m")

    assert "pulseInfo.PulseIDsUsed" in src
    assert "PulseLabels" in src
    assert "combined_gt_coords_px" in src
    assert "LabelPulsePolicy" in src


def test_process_run_preserves_label_validity_and_overlap_instances():
    src = read("scripts/process_run.m")

    assert "all_gt_coords_mm" in src
    assert "label_valid" in src
    assert "drop_reason" in src
    assert "instance_targets" in src
    assert "hr_frame_sum" in src
    assert "DroppedLabelCountsByReason" in src


def test_generate_streamlines_saves_reproducible_velocity_metadata():
    src = read("streamline-module/generate_streamlines.m")

    assert "FlowSimulationParameters.Velocity.Scale" in src
    assert "FlowSimulationParameters.Velocity.RawUnits" in src
    assert "FlowSimulationParameters.Velocity.EffectiveUnits" in src
    assert "FlowSimulationParameters.Velocity.LabelFieldDefinition" in src
    assert "FlowSimulationParameters.VelocityScale = VELOCITY_SCALE" in src


def test_acoustic_simulation_rejects_unmatched_tiled_trajectories():
    src = read("acoustic-module/main_RF.m")

    assert "main_RF:TilingMetadataMismatch" in src
    assert "define_medium(Grid, Medium, Geometry, FlowSimulationParameters)" in src
    assert "assert_tiling_metadata_matches" in src


def test_legacy_dataset_export_delegates_to_process_run():
    src = read("scripts/dataset_export.m")

    assert "function dataset_export" in src
    assert "process_run(RESULTS_FOLDER, SETTINGS_PATH, GT_FOLDER" in src
    assert "PREPROCESSING_OPTIONS" in src


def test_process_run_requires_explicit_preprocessing_for_dataset_export():
    src = read("scripts/process_run.m")

    assert "process_run:MissingPreprocessingOptions" in src
    assert "SplitMode" in src
    assert "frame_level" in src
    assert "case_level" in src
    assert "process_run:FrameLevelRequiresFitFrames" in src


def test_process_run_uses_explicit_svd_policy_not_hidden_constant():
    src = read("scripts/process_run.m")

    assert "SVD_CUTOFF       = 2" not in src
    assert "PreprocessingState.SVD.Mode" in src
    assert "PreprocessingState.SVD.SelectedCutoff" in src
    assert "adaptive_energy" in src
    assert "select_svd_cutoff" in src


def test_process_run_records_reproducibility_hashes_and_pipeline_state():
    src = read("scripts/process_run.m")

    assert "metadata.Hashes.SettingsFile" in src
    assert "metadata.Hashes.GTFlowSimulationParameters" in src
    assert "metadata.Hashes.RFSourceFiles" in src
    assert "metadata.Hashes.STLFile" in src
    assert "metadata.Hashes.VTUFile" in src
    assert "metadata.Hashes.GeometryPropertiesFile" in src
    assert "metadata.Pipeline.GitCommit" in src
    assert "metadata.Pipeline.GitDirty" in src
    assert "metadata.Pipeline.ExportTimestamp" in src


def test_file_hash_helper_exists():
    src = read("scripts/private/file_hash.m")

    assert "function hash = file_hash(filename)" in src
    assert "java.security.MessageDigest" in src or "DataHash" in src


def test_tiling_contract_saves_transforms_and_tile_ids():
    gt_src = read("streamline-module/generate_streamlines.m")
    medium_src = read("acoustic-module/define_medium.m")
    mb_src = read("acoustic-module/load_microbubbles.m")

    assert "Acquisition.Tiling" in gt_src
    assert "TileCfg.TransformFrame = 'vessel_to_image_consistent'" in gt_src
    assert "TileCfg.Transforms" in gt_src
    assert "tileIDs" in gt_src
    assert "Frame.(pulse).TileID" in gt_src
    assert "FlowSimulationParameters.Tiling.Transforms" in gt_src
    assert "apply_tile_transform_to_stl" in medium_src
    assert "MediumMetadata.Tiling" in medium_src
    assert "MB.tile_ids" in mb_src


def test_define_medium_closes_rotate_helper_before_tiling_helpers():
    src = read("acoustic-module/define_medium.m")

    rotate_idx = src.index("function meshXYZ = rotate_stl")
    apply_idx = src.index("function V = apply_tile_transform_to_stl")
    between_helpers = src[rotate_idx:apply_idx]

    assert between_helpers.splitlines().count("end") >= 2


def test_label_policy_is_configurable_and_frame_counts_are_saved():
    src = read("scripts/process_run.m")

    assert "LabelPolicy.VisibilityThreshold" in src
    assert "PREPROCESSING_OPTIONS.LabelPolicy" in src
    assert "LabelCountsByPulseAndReason" in src
    assert "sample_metadata.LabelCountsByPulseAndReason" in src


def test_define_sensor_mb_all_uses_current_acquisition_window():
    src = read("acoustic-module/define_sensor_MB_all.m")

    assert "frame_start = Acquisition.StartFrame;" in src
    assert "frame_end   = Acquisition.EndFrame;" in src
    assert "frame_start = 1;" not in src
    assert "frame_end   = Nframes;" not in src


def test_main_rf_splits_transducer_and_mb_transmit_batches():
    src = read("acoustic-module/main_RF.m")

    assert "get_transmit_batch_size(SimulationParameters, Acquisition)" in src
    assert "make_frame_batches(Acquisition.StartFrame, Acquisition.EndFrame" in src
    assert "sensor_data_transducer_1iter" in src
    assert "sensor_data_MB_1iter" in src
    assert "Simulating transducer-only transmit wave." in src
    assert "Simulating MB-only transmit wave." in src
    assert "kgrid.Nt = floor(run_param.tr(1) / kgrid.dt) + 1;" in src
    assert "extract_sensor_subset" in src


def test_main_rf_preserves_global_frame_numbering_after_internal_batching():
    src = read("acoustic-module/main_RF.m")

    assert "num_padding=num2str(length(num2str(Acquisition.NumberOfFrames)))" in src
    assert "file_name = ['Frame_', num2str(frame,['%0',num_padding,'i']),'.mat'];" in src


def test_v4_notebook_normalizes_geometry_rotation_before_persisting_settings():
    notebook = (ROOT.parent / "notebooks/proteus_data_generation_v4.ipynb").read_text()

    assert "Geometry.Rotation = double(Geometry.Rotation);" in notebook
