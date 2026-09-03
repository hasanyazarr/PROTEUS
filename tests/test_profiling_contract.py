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


def test_sense_no_longer_does_the_two_things_it_was_split_for():
    """The 2.4 s sense stage measured a row selection and a sparse product,
    and idx/weights existed to say which half was expensive. The answer turned
    out to be neither: both are done once for the whole batch now, inside the
    product the recorded transmit is streamed through, so sense is a row slice
    out of an array that is already computed. Timers over work that no longer
    happens would report zero and mislead the next reader."""
    src = read("acoustic-module/main_RF.m")
    log = read("acoustic-module/run_log.m")

    assert "run_log('stage', 'idx'," not in src
    assert "run_log('stage', 'weights'," not in src
    assert "'idx', 'sense'" not in log
    assert "'weights', 'sense'" not in log
    # sense still exists and still brackets what the frame does to get there.
    assert "run_log('stage', 'sense', toc(t_sense));" in src
    assert "sensed_p = sensed_all{pulse_seq_idx}( ..." in src


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
