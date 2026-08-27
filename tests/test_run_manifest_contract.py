"""Contract over the run provenance manifest.

A run is identified by three things that live in three different places: the
simulator commit (only the Colab clone knows it), the effective settings (only
the overridden .mat knows them), and the driver's own knobs (only the notebook
knows those). Losing any one of them makes a run irreproducible -- issue 4.
This file pins the two that MATLAB can write.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_write_run_manifest_pins_the_simulator_commit():
    """Colab clones the fork, so the commit is what says which code ran.
    v7's two runs disagreed on bubble placement while their settings matched,
    which is exactly the question a commit answers and settings do not."""
    src = read("scripts/write_run_manifest.m")

    assert "rev-parse HEAD" in src
    assert "rev-parse --abbrev-ref HEAD" in src
    assert "status --porcelain" in src
    assert "Simulator.Commit" in src
    assert "Simulator.Branch" in src
    assert "Simulator.Dirty" in src
    assert "Simulator.DirtyFiles" in src


def test_write_run_manifest_records_a_missing_git_explicitly():
    """A silently absent field reads the same as 'clean', which is the failure
    mode that produced issue 4 in the first place."""
    src = read("scripts/write_run_manifest.m")

    assert "'unavailable'" in src


def test_write_run_manifest_embeds_the_effective_settings():
    """The manifest has to answer 'which settings ran' without a .mat reader,
    because the .mat is the thing that goes missing."""
    src = read("scripts/write_run_manifest.m")

    assert "Settings.SHA256" in src
    assert "Settings.Values" in src
    assert "jsonencode" in src


def test_write_run_manifest_copies_the_settings_beside_the_frames():
    """The RESULTS folder must be self-sufficient before anything copies it to
    Drive; the notebook's copy step then becomes a redundancy, not the only
    path."""
    src = read("scripts/write_run_manifest.m")

    assert "settings_used.mat" in src
    assert "copyfile" in src


def test_write_run_manifest_appends_one_segment_per_invocation():
    """Batched and resumed runs call main_RF more than once. Overwriting would
    leave only the last segment, hiding that the run was resumed at all."""
    src = read("scripts/write_run_manifest.m")

    assert "Segments" in src
    assert "Segment.StartFrame" in src
    assert "Segment.EndFrame" in src
    assert "jsondecode" in src


def test_write_run_manifest_records_the_environment():
    src = read("scripts/write_run_manifest.m")

    assert "Environment.MatlabVersion" in src
    assert "Environment.Computer" in src
    assert "Environment.GPU" in src
    assert "Environment.Solver" in src


def test_main_RF_writes_the_manifest_before_it_simulates():
    """Written after the frame loop, the manifest is lost for every run that
    crashes or is cut short -- and those are the runs whose provenance is
    hardest to reconstruct afterwards."""
    src = read("acoustic-module/main_RF.m")

    assert "write_run_manifest(" in src

    manifest_at = src.index("write_run_manifest(")
    grid_at = src.index("Creating k-Wave grid")
    assert manifest_at < grid_at, (
        "the manifest must be written before the simulation starts"
    )
