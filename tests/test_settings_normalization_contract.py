from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_normalize_settings_types_recurses_into_containers():
    src = read("normalize_settings_types.m")

    # Struct arrays: every element must be visited, not only the first.
    assert "for index = 1:numel(value)" in src
    assert "names = fieldnames(value(index));" in src
    assert "value(index).(name) = normalize_settings_types(value(index).(name));" in src

    # Cell arrays: settings carry cells (e.g. tissue/label lists).
    assert "elseif iscell(value)" in src
    assert "value{index} = normalize_settings_types(value{index});" in src

    # Numeric leaves are widened; non-numeric leaves are left alone.
    assert "elseif isnumeric(value)" in src
    assert "value = double(value);" in src


def test_normalize_settings_types_lives_in_the_repo_not_the_notebook():
    """The Colab notebook used to %%writefile its own copy over the cloned repo,
    so the tracked implementation never ran. The notebook now relies on this
    file, which means it must stay at the repo root where addpath(pwd) finds it.
    """
    assert (ROOT / "normalize_settings_types.m").is_file()


def test_normalize_settings_types_has_matlab_behavior_test():
    src = read("tests/matlab/test_normalize_settings_types.m")

    assert "normalize_settings_types(arr)" in src
    assert "normalize_settings_types(c)" in src
    assert "isa(out.Geometry.Rotation, 'double')" in src
