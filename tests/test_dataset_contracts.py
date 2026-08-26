from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_load_rf_data_returns_frame_identity_and_pulse_metadata():
    src = read("delay-and-sum/load_RF_data.m")

    assert ("function [RF_matrix, FrameNumbers, RFFileNames, PulseInfo, "
            "SampleRange] = ...") in src
    assert "RFFileNames" in src
    assert "PulseInfo.PulsingScheme" in src
    assert "PulseInfo.PulseIDsUsed" in src
    assert "PulseInfo.CombinationFormula" in src


def test_process_run_records_split_aware_preprocessing_state():
    src = read("scripts/process_run.m")

    assert ("[RF, sourceFrameNumbers, sourceRFFileNames, pulseInfo, "
            "sampleRange] = ...") in src
    assert "fit_frame_mask" in src
    assert "PreprocessingState.SVDFitFrameNumbers" in src
    assert "PreprocessingState.NormalizationMode" in src
    assert "PreprocessingState.NormalizationReference" in src


def test_generate_streamlines_saves_reproducible_velocity_metadata():
    src = read("streamline-module/generate_streamlines.m")

    assert "FlowSimulationParameters.Velocity.Scale" in src
    assert "FlowSimulationParameters.Velocity.RawUnits" in src
    assert "FlowSimulationParameters.Velocity.EffectiveUnits" in src
    assert "FlowSimulationParameters.Velocity.LabelFieldDefinition" in src
    assert "FlowSimulationParameters.VelocityScale = VELOCITY_SCALE" in src


def test_generate_streamlines_writes_a_per_bubble_identity():
    """Every bubble slot is tracked independently and reseeded when it leaves
    the vessel, so (slot, StreamNumber) already identifies a track. The slot
    was never written, which is why every motion metric in the audit came back
    unmeasurable on (StreamNumber, TileID) alone."""
    src = read("streamline-module/generate_streamlines.m")

    assert "Frame.(pulse).BubbleIndex" in src
    assert "Frame.(pulse).TrackID" in src
    assert "bubbleIndexes" in src
    assert "trackIDs" in src


def test_track_ids_are_unique_without_a_shared_counter():
    """Slots are tracked under parfor, so the identity has to be a formula
    over values each worker already has, not a counter."""
    src = read("streamline-module/generate_streamlines.m")

    assert "bubbleIndex + NBubbles*(streamCount - 1)" in src
    assert "FlowSimulationParameters.Identity.TrackIDFormula" in src
    assert "FlowSimulationParameters.Identity.Definition" in src


def test_acoustic_simulation_rejects_unmatched_tiled_trajectories():
    src = read("acoustic-module/main_RF.m")

    assert "main_RF:TilingMetadataMismatch" in src
    assert "define_medium(Grid, Medium, Geometry, FlowSimulationParameters)" in src
    assert "assert_tiling_metadata_matches" in src


def test_process_run_derives_the_image_grid_from_the_vessel_box():
    """The grid used to be Geometry.Domain, which is sized by the transducer
    and the medium, not by the vessel. On 9L-D renal_tree that put ~90% of the
    pixels where no vessel can be."""
    src = read("scripts/process_run.m")

    assert "select_image_roi(" in src
    assert "Geom.BoundingBox.Diagonal" in src
    assert "abs(double(Geom.Rotation))" in src
    assert "x_lat = roi_x(1) : pixelSize : roi_x(2);" in src
    assert "z_ax  = roi_z(1) : pixelSize : roi_z(2);" in src
    # The old rule, gone: a domain-wide grid anchored at the transducer face.
    assert "width = D.Ymax - D.Ymin;" not in src
    assert "z_ax  = 0        : pixelSize : depth;" not in src


def test_process_run_clamps_the_image_roi_to_the_simulated_domain():
    """Nothing outside Geometry.Domain was simulated, so the vessel box may
    not reach past it."""
    src = read("scripts/process_run.m")

    assert "min(roi_z(2), double(D.Xmax))" in src
    assert "max(roi_x(1), double(D.Ymin))" in src
    assert "process_run:EmptyImageROI" in src


def test_process_run_records_the_image_roi_it_used():
    src = read("scripts/process_run.m")

    assert "PreprocessingState.ImageROI" in src
    assert "ROIState.Mode = 'vessel_bounding_box';" in src
    assert "ROIState.Mode = 'explicit';" in src
    assert "ROIState.MarginWavelengths" in src
    assert "ROIState.VesselBoxDepth" in src
    assert "process_run:InvalidImageROI" in src


def test_process_run_loads_only_the_samples_the_beamformer_reads():
    """DAS reads a window of each trace: the shallowest pixel sets the first
    sample, the deepest corner pixel and the f-number aperture set the last.
    On v7 that is samples 3452..6936 of 9512, so nearly two thirds of every
    trace never enters the SVD, the Casorati matrix, or the DAS sum."""
    src = read("scripts/process_run.m")

    assert "select_sample_range(" in src
    assert "SAMPLE_RANGE_MARGIN" in src
    assert "F_NUMBER = 0.8;" in src   # must match compute_das_matrix
    # The window is derived before the RF is read, so the crop saves the
    # memory rather than just the arithmetic.
    assert (src.index("wantedRange = select_sample_range(")
            < src.index("load_RF_data(RESULTS_FOLDER"))


def test_process_run_time_axis_follows_the_loaded_sample_window():
    """A cropped trace starts at sample n0, so t must start there too or every
    delay is wrong by the offset."""
    src = read("scripts/process_run.m")

    assert "t  = ((sampleRange(1)-1):(sampleRange(2)-1)) / Fs;" in src
    assert "t = (0:(Nt-1)) / Fs;" not in src


def test_process_run_records_the_sample_window_it_used():
    src = read("scripts/process_run.m")

    assert "PreprocessingState.SampleRange" in src
    assert "PreprocessingState.SampleRangeSource" in src


def test_load_RF_data_takes_a_sample_range():
    src = read("delay-and-sum/load_RF_data.m")

    assert ("function [RF_matrix, FrameNumbers, RFFileNames, PulseInfo, "
            "SampleRange] = ...") in src
    assert "sampleRange" in src
    assert "load_RF_data:InvalidSampleRange" in src


def test_process_run_validates_the_preprocessing_split_policy():
    src = read("scripts/process_run.m")

    assert "SplitMode" in src
    assert "frame_level" in src
    assert "case_level" in src
    assert "process_run:InvalidSplitMode" in src
    assert "process_run:FrameLevelRequiresFitFrames" in src


def test_process_run_uses_explicit_svd_policy_not_hidden_constant():
    src = read("scripts/process_run.m")

    assert "SVD_CUTOFF       = 2" not in src
    assert "PreprocessingState.SVD.Mode" in src
    assert "[n_remove, PreprocessingState.SVD] = select_svd_cutoff(" in src
    assert "SVDState.SelectedCutoff = cutoff;" in src
    assert "adaptive_energy" in src
    assert "select_svd_cutoff" in src


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


def test_v4_notebook_normalizes_every_settings_struct_before_persisting():
    """The notebook used to widen only Geometry.Rotation. It now passes every
    loaded settings struct through normalize_settings_types, so no other
    integer/single field can reach k-Wave arithmetic either.

    The notebook lives in the surrounding workspace, not in this repo, so this
    test is skipped wherever the repo is checked out on its own (e.g. Colab).
    """
    notebook_path = ROOT.parent / "notebooks/proteus_data_generation_v4.ipynb"
    if not notebook_path.is_file():
        pytest.skip("workspace notebook not present next to the repo")

    notebook = notebook_path.read_text()

    for name in ("Acquisition", "Geometry", "Medium", "Microbubble",
                 "SimulationParameters", "Transducer", "Transmit"):
        assert f"{name} = normalize_settings_types({name});" in notebook

    # The notebook must not carry its own copy of the function any more; the
    # tracked implementation is the one that has to run.
    assert "%%writefile /content/PROTEUS/normalize_settings_types.m" not in notebook


def test_process_run_no_longer_exports_a_super_resolution_dataset():
    """The SR dataset export was removed; runs write visualizations only."""
    src = read("scripts/process_run.m")

    assert "DATASET_OUT" not in src
    assert "hr_frame" not in src
    assert "instance_targets" not in src
    assert "gauss_point" not in src
    assert not (ROOT / "scripts/dataset_export.m").exists()
