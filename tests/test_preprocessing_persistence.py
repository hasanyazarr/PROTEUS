"""Contract over what process_run leaves behind about its own choices.

The image ROI, the RF sample window, the SVD fit frames and the normalization
reference were all built and then dropped: the only trace was stdout, so
asserting on the crop of a finished run meant scraping a notebook log. A
20-case dataset makes that the difference between a provenance record and a
guess.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_process_run_writes_its_preprocessing_state():
    src = read("scripts/process_run.m")

    assert "write_preprocessing_state(VIZ_OUT, PreprocessingState)" in src


def test_the_state_lands_beside_the_viz_it_describes_not_with_the_rf():
    """process_run can be re-run over one RESULTS_FOLDER with different
    options, each into its own VIZ_OUT. The state belongs to that pass, so
    writing it next to the RF data would have each run overwrite the last."""
    src = read("scripts/write_preprocessing_state.m")

    assert "fullfile(vizOut, 'preprocessing_state.json')" in src
    assert "RESULTS_FOLDER" not in src


def test_the_state_is_json_so_the_python_analysis_can_read_it():
    src = read("scripts/write_preprocessing_state.m")

    assert "jsonencode(" in src


def test_the_recorded_state_covers_the_choices_that_were_only_on_stdout():
    src = read("scripts/write_preprocessing_state.m")

    for field in ("ImageROI", "SampleRange", "SVDFitFrameNumbers",
                  "NormalizationMode"):
        assert field in src, "{} is not recorded".format(field)
