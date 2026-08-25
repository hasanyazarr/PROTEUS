"""Contract tests for the acquisition run log.

A 500-frame run used to emit nine lines per frame, most of them fixed strings
that carried no per-frame information, plus provenance banners that reprinted
every frame. These tests hold the log at one line per frame and keep the
banner state out of reach of MATLAB's function clearing.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


# Files on the per-frame path, with the fixed progress strings each one used to
# print. None of them said anything the frame line does not.
PER_FRAME_PRINTS = {
    "acoustic-module/compute_RF.m": [
        "Computing pressure at transducer integration points",
        "Applying lens delays",
        "Computing RF data",
    ],
    "acoustic-module/run_simulation_homogeneous.m": [
        "Propagating %d sources",
        "' done.\\n'",
    ],
    "acoustic-module/hybrid_simulator.m": [
        "Simulating receive data.",
    ],
    "acoustic-module/full_simulator.m": [
        "Simulating receive data.",
        "Simulating bubble-bubble interaction",
    ],
}


def test_per_frame_progress_strings_are_gone():
    for relpath, strings in PER_FRAME_PRINTS.items():
        src = read(relpath)
        for text in strings:
            assert text not in src, f"{relpath} still prints {text!r} per frame"


def test_run_log_state_survives_function_clearing():
    """Banners reprinted every frame because run_log kept its state in a
    persistent variable, and the per-frame addpath/rmpath dropped functions
    from memory. Root appdata is not cleared with the function.
    """
    src = read("acoustic-module/run_log.m")

    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
    assert "persistent " not in code
    assert "getappdata(0, STATE_KEY)" in src
    assert "setappdata(0, STATE_KEY, state)" in src


def test_microbubble_path_is_managed_once_per_acquisition():
    """The path used to change twice per frame inside the simulator call."""
    for relpath in ("acoustic-module/hybrid_simulator.m",
                    "acoustic-module/full_simulator.m"):
        src = read(relpath)
        assert "MicrobubblePath" not in src, f"{relpath} still edits the path"

    src = read("acoustic-module/main_RF.m")
    assert "addpath(run_param.MicrobubblePath)" in src
    # Restored on every exit, including the capture path's early return.
    assert "onCleanup" in src
    assert "function remove_microbubble_path(run_param)" in src
    assert "rmpath(run_param.MicrobubblePath)" in src


def test_frame_line_reports_progress_stages_and_eta():
    src = read("acoustic-module/run_log.m")

    # The per-frame bubble count moved out of the once-per-run banner.
    assert "case 'count'" in src
    # ODE measures part of MB; printing them as sibling columns invites
    # summing a row whose entries overlap.
    assert "NESTED = {'ODE', 'MB'};" in src
    assert "ORDER = {'MB', 'prop', 'RF'};" in src
    # An 8-hour run needs to say when it will finish.
    assert "sprintf('ETA %s'" in src
    assert "function text = format_duration(seconds)" in src


def test_solver_banner_carries_policy_not_per_frame_counts():
    src = read("acoustic-module/compute_bubble_mass_source.m")

    assert "run_log('count', 'MB', N_MB);" in src
    # A once-per-run banner cannot honestly report a count that changes per
    # frame, so the bubble count is no longer part of it.
    assert "N_MB=%d" not in src
    assert "stride=%d, n_sub=%d, precision=%s, pressure=%s" in src


def test_run_log_has_matlab_behavior_test():
    src = read("tests/matlab/test_run_log.m")

    assert "MB 47.8 (ODE 47.6)" in src
    assert "clear functions" in src
    assert "ETA 1m30s" in src
    assert "run_log:UnknownAction" in src
