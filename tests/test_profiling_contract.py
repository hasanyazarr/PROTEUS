"""Contract over the per-frame timing breakdown.

Measured on run_20260827_082616 at 200 bubbles: load 1.0, sense 2.4, MB 3.4
(ODE 3.2), prop 5.7, RF 1.1, save 0.2, against a 14.8 s frame. The two largest
remaining costs are the two that were never broken down -- prop at 38% and
sense at 16% -- and guessing which half of each is expensive is what this
exists to prevent.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_sense_is_split_into_the_two_things_it_does():
    """sense both selects the frame's rows out of the batch sensor and applies
    the per-bubble interpolation weights. The row selection re-intersects a
    batch-wide index on every frame; the weighting is a sparse product through
    a double cast. One timer over both cannot say which to attack."""
    src = read("acoustic-module/main_RF.m")

    assert "run_log('stage', 'idx'," in src
    assert "run_log('stage', 'weights'," in src


def test_the_sense_children_are_registered_as_nested():
    """Otherwise they print as top-level stages and the frame line stops
    adding up."""
    src = read("acoustic-module/run_log.m")

    assert "'idx', 'sense'" in src
    assert "'weights', 'sense'" in src


def test_propagation_profiling_is_reachable_from_the_driver():
    """run_simulation_homogeneous has recorded dist/field/accum since
    2026-08-27, but nothing outside MATLAB could turn it on, so prop's 5.7 s
    stayed a single number."""
    import json
    nb = json.loads(
        (ROOT.parent / "notebooks" / "proteus_data_generation_v7.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code")

    assert "PROFILE_PROPAGATION" in src
    assert ("SimulationParameters.ProfilePropagation = "
            "{mlbool(PROFILE_PROPAGATION)}") in src
