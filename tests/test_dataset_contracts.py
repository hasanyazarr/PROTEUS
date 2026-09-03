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
    """The SVD cutoff is a stated policy that the run records, not a constant.

    The cutoff selection moved inside apply_clutter_filter when the clutter
    filter grew modes, so select_svd_cutoff now hands its record back through
    that function rather than writing PreprocessingState directly. What the
    test protects is unchanged: no hidden constant, an adaptive option, and the
    chosen cutoff recorded in PreprocessingState.SVD.
    """
    src = read("scripts/process_run.m")

    assert "SVD_CUTOFF       = 2" not in src
    assert "[n_remove, SVDState] = select_svd_cutoff(" in src
    assert "SVDState.SelectedCutoff = cutoff;" in src
    assert "adaptive_energy" in src
    assert ("[RF_cas, PreprocessingState.Clutter, PreprocessingState.SVD] = ..."
            in src)


def test_process_run_clutter_filter_offers_modes_and_validates_them():
    """svd is one clutter mode among three, and each records what it removed.

    A static scatterer survives an SVD that keeps rank 1, which is what put the
    stationary artifact in the 2026-08-31 run; the frame mean removes it
    outright because it is a fixed direction rather than a fitted one.
    """
    src = read("scripts/process_run.m")

    assert ("function [RF_cas, ClutterState, SVDState] = apply_clutter_filter("
            in src)
    assert "case 'svd'" in src
    assert "case 'mean'" in src
    assert "case 'highpass'" in src
    assert "process_run:InvalidClutterMode" in src
    # highpass needs a cutoff, and it has to be one usable number: NaN passes
    # every range comparison and would zero the whole RF stack in silence.
    assert "process_run:MissingClutterCutoff" in src
    assert "~isscalar(fc) || ~isfinite(fc)" in src
    assert "process_run:InvalidClutterCutoff" in src
    # What each mode took out is provenance, not a print.
    assert "ClutterState.RemovedRank" in src
    assert "ClutterState.CutoffHz = fc;" in src


def test_define_sensor_MB_prunes_every_per_bubble_field():
    """A bubble that falls off the grid must leave every field, not two of them.

    Only radii and velocities were pruned until 2026-09-03, so TileID and
    RawVelocity kept their pre-exclusion length and, from the first dropped
    bubble on, described a different bubble than Points did. main_RF saves this
    struct into the RF frame, so the mislabelling is in shipped data: 10 of 40
    pulses checked in run_20260827_120645 and 5 of 120 in run_20260831_142721.
    Pruning by row count rather than by field name is what stops the next field
    added to load_microbubbles from being forgotten.
    """
    src = read("acoustic-module/define_sensor_MB.m")

    assert "MB = exclude_bubbles(MB, idx_exclude, n_bubbles);" in src
    assert "function MB = exclude_bubbles(MB, idx_exclude, n_bubbles)" in src
    assert "if size(value, 1) == n_bubbles" in src
    # The old named-field pruning must be gone, not merely supplemented.
    assert "MB.radii     (idx_exclude,:) = [];" not in src
    assert "MB.velocities(idx_exclude,:) = [];" not in src


def test_microbubble_transmit_is_streamed_not_cached():
    """The record the bubbles read is never held whole.

    It is the union of every bubble position in the batch. Without tiling that
    union deduplicates hard; with 200 tiles it does not, and at v11's grid it
    is 1.08e7 points against a 201795-point transducer mask -- 281 GB over the
    one-way window, against 83 GB of host memory. run_simulation_to_disk
    leaves it in the binary's output file and project_transmit_to_bubbles
    blocks over time, so the peak is one block.
    """
    run_to_disk = read("acoustic-module/run_simulation_to_disk.m")
    project = read("acoustic-module/project_transmit_to_bubbles.m")

    # The point of the function: return the file, do not read the record.
    # Checked against code only -- the comments explain the h5read this
    # exists to avoid, and would otherwise match.
    code = "\n".join(line for line in run_to_disk.splitlines()
                     if not line.lstrip().startswith("%"))
    assert "function output_filename = run_simulation_to_disk(" in run_to_disk
    assert "'SaveToDisk', input_filename" in run_to_disk
    assert "h5read" not in code
    # A failed binary must say so here. The 2026-09-03 run reported a full
    # disk as "object 'Nx' doesn't exist" from h5read, three frames of stack
    # away from what went wrong.
    assert "run_simulation_to_disk:BinaryFailed" in run_to_disk
    assert "run_simulation_to_disk:NoOutput" in run_to_disk
    # The two paths must set up the same simulation.
    assert "kwave_input_args(run_param)" in run_to_disk
    assert "kwave_input_args(run_param)" in read("acoustic-module/run_simulation.m")

    # Blocked over time, and the layout that makes a time block contiguous is
    # checked rather than assumed.
    assert "block_cols = max(1, floor(budget_bytes / (n_rows * 8)));" in project
    assert "project_transmit_to_bubbles:UnexpectedLayout" in project
    assert "h5info(record, '/p')" in project


def test_the_bubble_projection_is_built_once_per_batch():
    """One matrix per pulse, not one intersect and one product per frame.

    Each frame's weights, relocated onto the batch mask's columns and stacked,
    give a matrix whose product with the record is what the frame loop used to
    compute frame by frame -- the same nonzeros in the same column order, so
    the same numbers. The bubbles come along because this pass has already
    loaded and voxelised them.
    """
    src = read("acoustic-module/build_bubble_projection.m")

    assert "cols = locate_in_sorted(mask_idx_batch, mask_idx_frame);" in src
    assert "Projection(pulse_seq_idx).RowFirst = row_first;" in src
    assert "Projection(pulse_seq_idx).MB       = MB_all;" in src
    assert "Projection(pulse_seq_idx).MaxDist  = max_dist;" in src
    # Sized explicitly: one frame does not reach the last union column.
    assert "next_row - 1, n_union);" in src


def test_the_delta_truncation_radius_is_a_setting():
    """th sets the (2*th+1)^3 stencil each bubble occupies, and the union of
    those stencils is what sizes the recorded transmit. It was fixed at 4
    inside update_sensor, where no config could reach it. The transducer keeps
    the default whatever the microbubbles are set to.
    """
    setup = read("acoustic-module/sim_setup.m")
    sensor = read("acoustic-module/update_sensor.m")
    main = read("acoustic-module/main_RF.m")
    transducer = read("acoustic-module/define_sensor_transducer.m")

    assert "run_param.MicrobubbleDeltaTruncation = 4;" in setup
    assert "sim_setup:InvalidDeltaTruncation" in setup
    assert "if nargin < 6 || isempty(th)" in sensor
    assert "th = 4;" in sensor
    assert "get_truncated_grid(...\n    point, point_idx, Grid, th)" in sensor
    assert "run_param.MicrobubbleDeltaTruncation);" in main
    # The transducer's own sensor is built without it.
    assert "MicrobubbleDeltaTruncation" not in transducer


def test_the_transmit_record_is_sized_before_it_is_run():
    """Everything needed to refuse a 550 GB write is known before the first
    transmit; the 2026-09-03 run found out 36 minutes in, at 25% of pulse one.
    """
    src = read("acoustic-module/preflight_transmit_record.m")
    main = read("acoustic-module/main_RF.m")

    assert "preflight_transmit_record:RecordTooLarge" in src
    assert "getUsableSpace()" in src
    for knob in ("MicrobubbleDeltaTruncation", "CombineTransmitSensors",
                 "TransmitBatchSize", "Tiling.NumTiles"):
        assert knob in src, knob
    # Sized from the bubble counts, so it runs between the union mask and
    # build_bubble_projection -- which walks every frame of the batch, ~25 min
    # at v11's scale -- rather than after it.
    assert "combined, bubble_counts, run_param)" in src
    assert (main.index("preflight_transmit_record(")
            < main.index("Projection = build_bubble_projection(")
            < main.index("Simulating combined transducer and MB transmit wave."))


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
    """The transducer record is cached across the batch; the MB record is not.

    sensor_data_MB_1iter used to hold the microbubble transmit for every pulse
    at once. That record is the union of every bubble position in the batch,
    which tiling makes 54x the transducer mask -- 281 GB at v11's grid against
    83 GB of host memory -- so it is streamed from the binary's own output
    file and projected onto the bubbles instead of being held.
    """
    src = read("acoustic-module/main_RF.m")

    assert "get_transmit_batch_size(SimulationParameters, Acquisition)" in src
    assert "make_frame_batches(Acquisition.StartFrame, Acquisition.EndFrame" in src
    assert "sensor_data_transducer_1iter" in src
    assert "sensor_data_MB_1iter" not in src
    assert "mb_record_file = run_simulation_to_disk(" in src
    assert "Simulating transducer-only transmit wave." in src
    assert "Simulating MB-only transmit wave." in src
    assert "kgrid.Nt = floor(run_param.tr(1) / kgrid.dt) + 1;" in src
    assert "extract_sensor_subset" in src
    # The record is hundreds of gigabytes on the disk this path exists for,
    # so it must not survive a projection that throws.
    assert "onCleanup(@() delete_if_present(mb_record_file))" in src


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

    That trade is no longer free, so it is no longer automatic. The combined
    run records the microbubble mask over the round trip rather than the
    one-way window, and with tiling that mask is 54x the transducer's: the
    doubling is the difference between a transmit that fits the disk and one
    that does not. Still the default, now overridable.
    """
    src = read("acoustic-module/main_RF.m")

    assert ("combine_transmit_sensors = num_batches == 1 && ...\n"
            "        run_param.CombineTransmitSensors;") in src
    setup = read("acoustic-module/sim_setup.m")
    assert "run_param.CombineTransmitSensors = true;" in setup
    assert "Simulating combined transducer and MB transmit wave." in src
    # The combined sensor is the union of the two masks, recorded for the
    # round trip the transducer needs. OR, not sum: both masks are logical
    # and '+' would promote the whole grid to double for the temporary.
    assert "sensor_MB_batch.mask | sensor_transducer.mask;" in src
    assert "sensor_combined.record = sensor_MB_batch.record;" in src
    # Both sensor sets come out of that one run.
    assert "sensor_data_transducer_1iter{pulse_seq_idx} = ..." in src
    assert "mask_idx_combined, mask_idx_trans, n_transducer_time);" in src
    # The MB rows are kept only for the one-way window, and are projected
    # onto the bubbles here rather than carried into the frame loop.
    assert "mask_idx_combined, mask_idx_MB_batch, n_mb_time);" in src
    assert "sensed_all{pulse_seq_idx} = project_transmit_to_bubbles( ..." in src
    # The split path stays for the multi-batch case.
    assert "Simulating transducer-only transmit wave." in src
    assert "Simulating MB-only transmit wave." in src



def test_the_elevation_slab_crop_runs_before_tiling():
    """Cropping in elevation commutes with a tile transform; cropping in
    depth or lateral does not.

    A tile rotates about the elevation axis, so a cell's elevation survives
    the transform and a slab crop applied to the canonical vessel stays
    correct afterwards. Depth and lateral mix under that rotation, which is
    why crop_vessel_to_domain cannot be composed with tiling the same way -
    see the guard below. Measured 2026-09-03: clipping the renal tree to the
    imaged slab is worth 21x the in-plane bubble count, against 6x for
    raising Microbubble.Number from 200 to 1200."""
    src = read("streamline-module/generate_streamlines.m")

    assert "crop_vessel_to_slab(vtuStruct, Geometry, SlabHalfThickness)" in src
    assert "Acquisition.ElevationSlab" in src
    assert (src.index("crop_vessel_to_slab(vtuStruct, Geometry, SlabHalfThickness)")
            < src.index("TileCfg.Transforms = build_tile_transforms(TileCfg);"))


def test_domain_cropping_and_tiling_are_refused_together():
    """crop_vessel_to_domain zeroes seeding weight by image-frame position,
    and it runs before the tile transforms exist. Offsetting a cropped vessel
    by up to 25 mm afterwards leaves the crop meaningless, so the pair is an
    error rather than a silently wrong run."""
    src = read("streamline-module/generate_streamlines.m")

    assert "generate_streamlines:DomainCropWithTiling" in src


def test_tile_placement_is_validated_against_the_domain():
    """run_20260708_043218 put 0.30% of its labelled bubbles outside the
    simulated domain and 0.8% inside the transmit ringdown, because nothing
    checked where the sampled tile offsets landed. The transforms are known
    before the first frame, so the check is free."""
    src = read("streamline-module/generate_streamlines.m")

    assert "validate_tile_placement(TileCfg, vtuStruct, Geometry)" in src
    assert (src.index("validate_tile_placement(TileCfg, vtuStruct, Geometry)")
            > src.index("TileCfg.Transforms = build_tile_transforms(TileCfg);"))


def test_the_elevation_slab_reaches_the_ground_truth():
    """A knob that changes where bubbles are seeded has to be recoverable from
    the run's own data, the way Tiling, Seeding and VelocityScale already are."""
    src = read("streamline-module/generate_streamlines.m")

    assert "FlowSimulationParameters.Seeding.ElevationSlab" in src
