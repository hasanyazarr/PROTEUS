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


def test_velocity_scale_defaults_to_the_unscaled_cfd_field():
    """A settings file that says nothing about velocity gets the CFD field as
    the CFD solved it. The scale was hardcoded at 5 until 2026-08-27, so every
    dataset carried a 5x flow that no settings file recorded and no reader
    could have inferred from the config."""
    src = read("streamline-module/generate_streamlines.m")

    assert "VELOCITY_SCALE = 1;" in src
    assert "VELOCITY_SCALE = 5;" not in src


def test_velocity_scale_reports_whether_it_came_from_settings():
    """The scale changes what the dataset means, so a run has to say out loud
    which value it used and whether a settings file asked for it."""
    src = read("streamline-module/generate_streamlines.m")

    assert "velocityScaleSource" in src
    assert "'Acquisition.VelocityScale'" in src
    assert "MB velocity scale" in src


def test_velocity_scale_is_validated_before_it_reaches_the_ode():
    """A zero, negative, or non-finite scale integrates to a silently wrong
    trajectory rather than an error."""
    src = read("streamline-module/generate_streamlines.m")

    assert "generate_streamlines:InvalidVelocityScale" in src


def test_a_bubble_that_leaves_the_vessel_is_reseeded_at_the_inlet():
    """Upstream put a bubble that exited back at the inlet, so it had to
    traverse the tree. The tiling rewrite routed the reseed through
    build_tile_problem, which draws from the vessel bulk instead -- measured on
    run_20260827_082616, reseed positions have the same spatial distribution as
    the frame-1 bulk seeds (spans 11.35 vs 11.27 mm, centroids 0.39 mm apart)
    rather than clustering at one vessel end."""
    src = read("streamline-module/generate_streamlines.m")

    assert "RESEED_FROM = 'inlet';" in src
    assert "'Acquisition.ReseedFrom'" in src
    assert "generate_streamlines:InvalidReseedFrom" in src


def test_the_reseed_source_is_a_parameter_not_a_literal():
    """build_tile_problem drew from vtuStruct whichever call reached it, so the
    first seed and the reseed could not differ. They have to: upstream seeds
    the first bubble in the bulk and every later one at the inlet."""
    src = read("streamline-module/generate_streamlines.m")

    assert "seedStruct" in src
    assert "draw_start_position(1, seedStruct)" in src
    assert "draw_start_position(1, vtuStruct)" not in src


def test_the_reseed_policy_reaches_the_ground_truth():
    src = read("streamline-module/generate_streamlines.m")

    assert "FlowSimulationParameters.Seeding.ReseedFrom" in src


def test_main_rf_records_both_sensors_in_one_run_when_there_is_one_batch():
    """Two transmit runs are a memory trade, not a free structural split.

    The transducer-only run and the MB-only run propagate the same pulse
    through the same medium; only the recorded points and the record length
    differ. Measured 2026-08-27 on run_20260827_082616: 816.58 s for the
    transducer run plus 449.50 s for the MB run, where one combined run at
    the round-trip length is 816.58 s. Split cost is 817 + n*450 against a
    combined n*817, so the split only pays from three batches on. With a
    single batch - the production setting, TransmitBatchSize = NumberOfFrames
    - it is 450 s per pulse spent for nothing.
    """
    src = read("acoustic-module/main_RF.m")

    assert "combine_transmit_sensors = num_batches == 1;" in src
    assert "Simulating combined transducer and MB transmit wave." in src
    # The combined sensor is the union of the two masks, recorded for the
    # round trip the transducer needs.
    assert "sensor_combined.mask = logical(" in src
    assert "sensor_combined.record = sensor_MB_batch.record;" in src
    # Both sensor sets come out of that one run.
    assert "sensor_data_transducer_1iter{pulse_seq_idx} = ..." in src
    assert "mask_idx_combined, mask_idx_trans, n_transducer_time);" in src
    # The MB rows are kept only for the one-way window, so the block held
    # through the frame loop is the size the split path held.
    assert "mask_idx_combined, mask_idx_MB_batch, n_mb_time);" in src
    # The split path stays for the multi-batch case.
    assert "Simulating transducer-only transmit wave." in src
    assert "Simulating MB-only transmit wave." in src

