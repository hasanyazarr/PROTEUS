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
    assert "addedMicrobubblePaths = add_microbubble_path(run_param)" in src
    assert "function addedPaths = add_microbubble_path(run_param)" in src
    # Restored on every exit, including the capture path's early return.
    assert "onCleanup" in src
    assert "remove_microbubble_path(addedMicrobubblePaths)" in src
    assert "function remove_microbubble_path(addedPaths)" in src


def test_microbubble_path_cleanup_only_removes_what_it_added():
    """A caller that already had the module on the path keeps it.

    The GPU solver evaluation adds the microbubble module, calls main_RF to
    capture pressure, then replays that pressure through
    compute_bubble_mass_source. An unconditional rmpath left it without
    resolve_gpu_rk4_max_phase_step after the capture returned.
    """
    src = read("acoustic-module/main_RF.m")

    # Nothing is added, or removed, without checking the search path first.
    assert "if ~is_on_search_path(candidatePaths{i})" in src
    assert "if is_on_search_path(addedPaths{i})" in src
    assert "function tf = is_on_search_path(candidatePath)" in src

    # The old unconditional form must not come back.
    assert "rmpath(run_param.MicrobubblePath)" not in src
    assert "addpath(run_param.MicrobubblePath)" not in src


def test_frame_line_reports_progress_stages_and_eta():
    src = read("acoustic-module/run_log.m")

    # The per-frame bubble count moved out of the once-per-run banner.
    assert "case 'count'" in src
    # ODE measures part of MB; printing them as sibling columns invites
    # summing a row whose entries overlap.
    assert "NESTED = {'ODE', 'MB'; 'dist', 'prop'; 'field', 'prop'; " in src
    assert "ORDER = {'load', 'sense', 'MB', 'prop', 'RF', 'save'};" in src
    # An 8-hour run needs to say when it will finish.
    assert "sprintf('ETA %s'" in src
    assert "function text = format_duration(seconds)" in src


def test_every_part_of_the_frame_is_accounted_for():
    """The frame line used to explain only two thirds of the frame.

    Measured 2026-08-25 on a 1000-frame run: MB 3.4 s, prop 5.8 s, RF 1.1 s
    against a 14.8 s frame, leaving 4.5 s - 30% of every frame - under no
    stage at all. That is more than the whole bubble solver, and nothing
    could be aimed at it while it was invisible. The work between the timed
    stages is loading the frame's bubbles and building its sensor, taking
    the batch's recorded pressure down to this frame's points, and writing
    the result.
    """
    src = read("acoustic-module/main_RF.m")

    assert "run_log('stage', 'load', toc(t_load));" in src
    assert "run_log('stage', 'sense', toc(t_sense));" in src
    assert "run_log('stage', 'save', toc(t_save));" in src
    # The stages have to bracket the real work, not just exist.
    for timer, call in (("t_load", "load_microbubbles("),
                        ("t_sense", "extract_sensor_subset("),
                        ("t_save", "save([savedir filesep file_name]")):
        opened = src.index(timer + " = tic;")
        closed = src.index("toc(" + timer + ")")
        assert opened < src.index(call) < closed, timer


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


def test_propagation_substages_are_opt_in_and_synchronised():
    """prop is 39% of a frame; its 5.8 s has no breakdown.

    Measured 2026-08-25: prop 5.8 s against a 14.8 s frame, the largest
    single stage. The loop mixes CPU distance work, a per-source transfer,
    a large complex exp and ifft, and an expansion back to sensor points -
    and nothing says which of those it is.

    gpuArray work is queued asynchronously, so a bare toc around a GPU
    stage measures the queueing, not the work. Honest per-stage numbers
    need a device synchronisation, and that serialises a loop which is
    otherwise free to overlap. So the breakdown is off unless asked for.
    """
    src = read("acoustic-module/run_simulation_homogeneous.m")
    setup = read("acoustic-module/sim_setup.m")
    log = read("acoustic-module/run_log.m")

    # Off by default; a production run must not pay for the synchronisation.
    assert "run_param.ProfilePropagation = false;" in setup
    assert "isfield(SimulationParameters, 'ProfilePropagation')" in setup

    # The three candidates the 5.8 s could be hiding in.
    for stage in ("dist", "field", "accum"):
        assert f"stage_toc(profilePropagation, '{stage}'" in src, stage
        assert f"'{stage}', 'prop'" in log, stage

    # Without this the GPU stages would report queueing time.
    assert "wait(gpuDevice)" in src
    assert "function stage_toc(" in src
