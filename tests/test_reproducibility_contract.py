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
