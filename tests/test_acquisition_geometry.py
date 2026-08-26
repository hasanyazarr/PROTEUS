"""Tests for the settings-driven ground-truth geometry.

The notebook used to hard-code the transform and the field of view. Those
constants were correct for one L22 config and silently wrong for every other,
so these tests pin that the derivation moves when the settings move.

The beamforming grid follows the vessel bounding box, not the simulation
domain. The domain is sized by the transducer surface and the far edge of the
medium, so a domain-sized grid put ~90 % of every pixel outside any vessel on
the 9L-D renal_tree configs. These tests pin the vessel-box rule, the domain
clamp that still bounds it, and the explicit override.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.acquisition_geometry import (  # noqa: E402
    AcquisitionGeometry, _colon, _index_on_grid)


# The renal_tree bounding box, shared by every config that seeds in it.
BBOX_CENTER = [0.00808885, 0.00695069, 0.01109843]
BBOX_DIAGONAL = [0.01335366, 0.00950394, 0.01841407]
ROTATION = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]


def settings(*, center_x, y_half, x_max, f0, pitch, n_elements,
             element_height, elevation_focus, probe="probe", c=1540.0):
    """A settings dict with the fields AcquisitionGeometry reads."""
    return {
        "Geometry": SimpleNamespace(
            Rotation=np.array(ROTATION),
            BoundingBox=SimpleNamespace(
                Center=np.array(BBOX_CENTER),
                Diagonal=np.array(BBOX_DIAGONAL)),
            Center=np.array([center_x, 0.0, 0.0]),
            Domain=SimpleNamespace(
                Xmin=-0.0005811320754716982, Xmax=x_max,
                Ymin=-y_half, Ymax=y_half,
                Zmin=-0.0035811320754716982, Zmax=0.0035811320754716982),
        ),
        "Transducer": SimpleNamespace(
            Type=probe, NumberOfElements=n_elements, Pitch=pitch,
            ElementHeight=element_height, ElevationFocus=elevation_focus),
        "Transmit": SimpleNamespace(CenterFrequency=f0),
        "Medium": SimpleNamespace(SpeedOfSound=c),
    }


def l22_settings():
    return settings(center_x=0.015207, y_half=0.006975, x_max=0.011995,
                    f0=15e6, pitch=100e-6, n_elements=128,
                    element_height=1.6e-3, elevation_focus=8e-3,
                    probe="L22-14v")


def nine_l_settings():
    return settings(center_x=0.03420704, y_half=0.022651132075471703,
                    x_max=0.0439952033645342, f0=5.3e6, pitch=230e-6,
                    n_elements=192, element_height=6e-3,
                    elevation_focus=28e-3, probe="9L-D")


def equivalent_translation_mm(geometry):
    """The single translation the notebook used to hard-code."""
    return (geometry.center - geometry.rotation @ geometry.bbox_center) * 1e3


def test_reproduces_the_hardcoded_l22_translation():
    """The transform constant the notebook carried as a literal."""
    geometry = AcquisitionGeometry(l22_settings())

    assert equivalent_translation_mm(geometry) == pytest.approx(
        [26.305, 8.089, -6.951], abs=5e-3)


def test_the_same_constants_are_wrong_for_a_different_probe():
    """The failure the hard-coded version could not see."""
    geometry = AcquisitionGeometry(nine_l_settings())

    # 19 mm of axial offset, the difference in Geometry.Center.
    assert equivalent_translation_mm(geometry)[0] == pytest.approx(
        45.305, abs=5e-3)


def test_the_grid_follows_the_vessel_not_the_domain():
    """The 9L-D domain is 44 x 45.3 mm; the vessel occupies a tenth of it."""
    geometry = AcquisitionGeometry(nine_l_settings())

    # Vessel box, imaging frame: axial 25.000..43.414, lateral +-6.677.
    assert geometry.vessel_box_mm["axial"] == pytest.approx(
        (25.000, 43.414), abs=5e-3)
    assert geometry.vessel_box_mm["lateral"] == pytest.approx(
        (-6.677, 6.677), abs=5e-3)

    # Grid: that box plus a 5 lambda margin, clamped by the domain.
    assert geometry.axial_range_mm == pytest.approx(
        (23.5472, 43.9952), abs=5e-4)
    assert geometry.lateral_range_mm == pytest.approx(
        (-8.1297, 8.1297), abs=5e-4)
    assert geometry.image_shape == (352, 280)


def test_the_grid_no_longer_starts_at_the_transducer_face():
    """The old rule always began at 0, which is where the transmit transient
    dominates and where no vessel can be."""
    geometry = AcquisitionGeometry(nine_l_settings())

    assert geometry.z_ax_mm[0] > 20.0


def test_the_domain_still_clamps_the_grid():
    """The L22 vessel box runs to 24.4 mm but only 12 mm was simulated."""
    geometry = AcquisitionGeometry(l22_settings())

    assert geometry.vessel_box_mm["axial"][1] == pytest.approx(24.414, abs=5e-3)
    assert geometry.axial_range_mm[1] == pytest.approx(11.995, abs=5e-4)
    # Lateral is clamped by the domain on both sides here.
    assert geometry.lateral_range_mm == pytest.approx(
        (-6.975, 6.975), abs=5e-4)


def test_an_explicit_roi_overrides_the_vessel_box():
    geometry = AcquisitionGeometry(
        nine_l_settings(), image_roi={"Depth": (0.030, 0.035),
                                      "Lateral": (-0.002, 0.002)})

    assert geometry.axial_range_mm == pytest.approx((30.0, 35.0), abs=5e-4)
    assert geometry.lateral_range_mm == pytest.approx((-2.0, 2.0), abs=5e-4)
    assert geometry.image_roi_mode == "explicit"


def test_an_explicit_roi_is_still_clamped_to_the_domain():
    geometry = AcquisitionGeometry(
        nine_l_settings(), image_roi={"Depth": (-0.010, 0.900),
                                      "Lateral": (-0.900, 0.900)})

    assert geometry.axial_range_mm == pytest.approx((0.0, 43.9952), abs=5e-4)
    assert geometry.lateral_range_mm[0] == pytest.approx(-22.6511, abs=5e-4)


def test_an_empty_roi_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        AcquisitionGeometry(
            nine_l_settings(), image_roi={"Depth": (0.100, 0.200),
                                          "Lateral": (-0.002, 0.002)})


def test_pixel_size_follows_the_centre_frequency():
    l22 = AcquisitionGeometry(l22_settings())
    nine_l = AcquisitionGeometry(nine_l_settings())

    # lambda / 5, as process_run.m sets it.
    assert nine_l.pixel_size_mm == pytest.approx(1540 / 5.3e6 * 1e3 / 5)
    assert l22.pixel_size_mm < nine_l.pixel_size_mm


def test_transform_matches_the_matlab_composition():
    """pts - bbox_center, rotate, + center -- the order load_gt_frame uses."""
    geometry = AcquisitionGeometry(nine_l_settings())
    world = np.array([[0.01, 0.02, 0.03], [0.0, 0.0, 0.0]])

    expected = np.vstack([
        geometry.rotation @ (p - geometry.bbox_center) + geometry.center
        for p in world]) * 1e3

    assert geometry.to_image_mm(world) == pytest.approx(expected)


def test_a_single_point_is_accepted():
    geometry = AcquisitionGeometry(nine_l_settings())
    assert geometry.to_image_mm([0.01, 0.02, 0.03]).shape == (1, 3)


def test_pixels_are_one_based_and_nan_outside_the_grid():
    geometry = AcquisitionGeometry(nine_l_settings())
    lateral = geometry.x_lat_mm
    axial = geometry.z_ax_mm

    at_origin = geometry.to_pixels([[axial[0], lateral[0], 0.0]])
    assert at_origin[0] == pytest.approx([1.0, 1.0])

    at_end = geometry.to_pixels([[axial[-1], lateral[-1], 0.0]])
    assert at_end[0] == pytest.approx(
        [len(lateral), len(axial)])

    outside = geometry.to_pixels([[axial[-1] + 1.0, lateral[0] - 1.0, 0.0]])
    assert np.all(np.isnan(outside))


def test_classification_separates_the_two_geometric_reasons():
    geometry = AcquisitionGeometry(nine_l_settings())

    # Build world points from image points so the case under test is exact.
    def world_of(axial_mm, lateral_mm, elevation_mm):
        image = np.array([axial_mm, lateral_mm, elevation_mm]) / 1e3
        return np.linalg.inv(geometry.rotation) @ (
            image - geometry.center) + geometry.bbox_center

    # 33 mm is inside the vessel-box grid; 20 mm no longer is.
    points = np.vstack([
        world_of(33.0, 0.0, 0.0),      # in frame
        world_of(33.0, 0.0, 5.0),      # in the image, off the plane
        world_of(200.0, 0.0, 0.0),     # past the far edge
        world_of(33.0, 100.0, 0.0),    # past the lateral edge
        world_of(20.0, 0.0, 0.0),      # shallower than the grid now starts
    ])

    reasons = geometry.classify(points, elevation_filter_mm=1.0)
    assert list(reasons) == [
        "in_frame", "out_of_plane", "out_of_fov", "out_of_fov", "out_of_fov"]


def test_out_of_fov_wins_over_out_of_plane():
    """A point outside the image is out of the field of view whatever its
    elevation, the order the geometric label rules apply."""
    geometry = AcquisitionGeometry(nine_l_settings())
    far = np.linalg.inv(geometry.rotation) @ (
        np.array([0.2, 0.0, 0.05]) - geometry.center) + geometry.bbox_center

    assert geometry.classify(far[None, :], 1.0)[0] == "out_of_fov"


def test_suggested_elevation_scales_with_the_elevation_aperture():
    l22 = AcquisitionGeometry(l22_settings())
    nine_l = AcquisitionGeometry(nine_l_settings())

    assert nine_l.suggested_elevation_mm() == pytest.approx(1.36, abs=0.01)
    assert l22.suggested_elevation_mm() == pytest.approx(0.51, abs=0.01)
    # The 1.0 mm the notebook applied to both is neither probe's estimate.
    assert not (0.9 < nine_l.suggested_elevation_mm() < 1.1)


def test_colon_matches_matlab_endpoint_handling():
    # Exact fit keeps the endpoint.
    assert _colon(0.0, 0.5, 2.0) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    # Short of the endpoint, the last point is dropped rather than clamped.
    assert _colon(0.0, 0.6, 2.0) == pytest.approx([0.0, 0.6, 1.2, 1.8])
    # A step that only just overshoots by rounding still lands on the end.
    grid = _colon(0.0, 0.1, 0.3)
    assert len(grid) == 4


def test_index_on_grid_is_affine_and_bounded():
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    index = _index_on_grid(grid, np.array([0.0, 0.5, 3.0, -0.1, 3.1]))

    assert index[:3] == pytest.approx([1.0, 1.5, 4.0])
    assert np.isnan(index[3]) and np.isnan(index[4])


def test_describe_names_the_probe_and_the_image():
    text = AcquisitionGeometry(nine_l_settings()).describe()

    assert "9L-D" in text
    assert "192 elements" in text
    assert "5.30 MHz" in text


# -- against the real settings files, when the workspace is present --------

CONFIGS = ROOT.parent / "configs" / "gui_parameters"

real_settings = pytest.mark.skipif(
    not CONFIGS.is_dir(),
    reason="settings snapshots live in the workspace, not in the clone")


@real_settings
@pytest.mark.parametrize("name, axial, lateral, shape", [
    ("GUI_output_parameters_v7.mat",
     (23.5472, 43.9952), (-8.1297, 8.1297), (352, 280)),
    ("GUI_output_parameters_v9e_L22.mat",
     (5.4867, 12.0000), (-6.9711, 6.9711), (318, 680)),
])
def test_real_settings_give_the_expected_field_of_view(
        name, axial, lateral, shape):
    geometry = AcquisitionGeometry.from_settings(str(CONFIGS / name))

    assert geometry.axial_range_mm == pytest.approx(axial, abs=5e-4)
    assert geometry.lateral_range_mm == pytest.approx(lateral, abs=5e-4)
    assert geometry.image_shape == shape


def test_from_settings_rejects_a_file_that_is_not_settings(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "not_settings.mat"
    savemat(str(path), {"Something": np.zeros(3)})

    with pytest.raises(KeyError, match="Geometry"):
        AcquisitionGeometry.from_settings(str(path))
