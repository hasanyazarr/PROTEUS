"""Contract over run reproducibility.

Two runs of the same commit with the same settings produced different RF and
different ground truth: the tissue speckle in define_medium and the bubble
start positions and radii in the streamline module were all drawn with no rng
call anywhere. The manifest gave attribution, not reproducibility, and a
dropped run could not be resumed because its second half would sit in
different tissue.

Kept apart from test_dataset_contracts.py so the two files can be edited
independently.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_the_tissue_speckle_is_drawn_from_a_recorded_seed():
    """define_medium drew the inhomogeneity with no rng at all, so two runs of
    the same commit and the same settings produced different RF. The manifest
    gave attribution, not reproducibility, and a dropped run could not be
    resumed because its second half would sit in different tissue."""
    src = read("acoustic-module/define_medium.m")

    assert "MediumMetadata.RandomSeed" in src
    # Ordering is the contract, not the call: a seed resolved after the draw
    # would record a number the speckle was never generated from.
    assert src.index("resolve_random_seed(") < src.index("random(pd,")


def test_the_ground_truth_is_drawn_from_a_recorded_seed():
    """Bubble start positions and radii are as unseeded as the speckle was --
    draw_start_position and draw_random_radii both call rand. Seeding only the
    medium would still leave a run unreproducible."""
    src = read("streamline-module/generate_streamlines.m")

    assert "resolve_random_seed" in src
    assert "FlowSimulationParameters.RandomSeed" in src


def test_each_bubble_slot_seeds_its_own_stream():
    """Tracking runs under parfor when Acquisition.ParallelTracking is set, and
    a single rng call before a parfor does not make the workers deterministic.
    Deriving each slot's stream from the base seed and its own index is what
    makes the ground truth independent of execution order."""
    src = read("streamline-module/generate_streamlines.m")

    assert "mod(RANDOM_SEED + bubbleIndex, 2^32)" in src


def test_a_resolved_seed_is_always_a_number_even_when_shuffled():
    """'shuffle' has to resolve to the integer it picked, or a run that did not
    pin a seed stays unreproducible -- which is the whole problem."""
    src = read("acoustic-module/resolve_random_seed.m")

    assert "'shuffle'" in src
    assert "rng('shuffle')" in src
    assert "resolve_random_seed:InvalidSeed" in src


def test_velocity_weighted_seeding_is_a_settings_field():
    """SeedCfg was three literals with no Acquisition hook at all -- not even a
    read path, unlike the velocity scale. It drops the slow half of the CFD
    cells and weights the rest by speed, which is why only 1.9% of visited
    samples in run_20260827_082616 sat below the cut."""
    src = read("streamline-module/generate_streamlines.m")

    assert "SeedCfg.Enabled = false;" in src
    assert "isfield(Acquisition, 'Seeding')" in src


def test_tiling_defaults_to_off():
    """TileCfg.Enabled was true, and tiling was off in production only because
    the driver notebook wrote Acquisition.Tiling = struct('Enabled', false).
    Upstream has one vessel."""
    src = read("streamline-module/generate_streamlines.m")

    assert "TileCfg.Enabled              = false;" in src


def test_a_tiling_struct_without_enabled_is_an_error():
    """v9b, v9c and v9d set tiling ranges and no Enabled field, so they ran
    tiled purely on the old default of true. Flipping that default would have
    turned all three into single-vessel runs with nothing said. Refusing the
    ambiguous struct turns a silent change into a loud one."""
    src = read("streamline-module/generate_streamlines.m")

    assert "generate_streamlines:TilingEnabledMissing" in src


def test_self_sensing_is_masked_on_the_array_it_indexes():
    """d holds one entry per distance-grid point when run_param.gridded is set
    and one per sensor otherwise, while d0 always holds one per sensor. The
    inlined Green's function masked d with d0's mask, which is the wrong length
    on the gridded path -- harmless only while no bubble sits exactly on a
    transducer point. Upstream's calc_scatter_attenuated masked r on r == 0,
    the array it was about to divide by."""
    src = read("acoustic-module/run_simulation_homogeneous.m")

    assert "d_safe(d == 0) = Inf;" in src
    assert "d_safe(d0 == 0)" not in src


def test_self_sensing_is_zeroed_after_the_grid_is_expanded_back():
    """Zeroing p on d0's mask before the i_sampled expansion indexes
    distance-grid rows with a sensor-length mask. After the expansion the rows
    are sensors again and the mask fits."""
    src = read("acoustic-module/run_simulation_homogeneous.m")

    assert "p_sensor(d0 == 0,:) = 0;" in src
    assert "p(d0 == 0,:) = 0;" not in src
