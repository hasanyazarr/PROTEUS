"""Read the acquisition geometry out of a settings .mat.

Ground-truth diagnostics need the same world-to-image transform and the same
beamforming grid that ``scripts/process_run.m`` builds, or they answer a
question about a different acquisition than the one that ran. Hard-coding
those numbers in a notebook means every probe or domain change silently
invalidates the diagnostic: the constants for an L22 run applied to a 9L-D run
report a near-zero yield on a perfectly healthy geometry.

Everything here is derived from the settings file, so a different transducer,
domain, or centre frequency needs no edit.

The MATLAB source of each rule:

    transform    process_run.m, load_gt_frame
    grid         process_run.m, select_image_roi and the x_lat / z_ax colons

The label rules these diagnostics apply (in field of view, in plane) are the
geometry-only ones; they used to mirror the dataset export's label
classification, which was removed with the SR export on 2026-08-26.

Kept compatible with Python 3.9 so it runs on this workstation as well as on
Colab.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat


# MATLAB's colon operator keeps a point that lands within a rounding error of
# the endpoint. Mirror that so the grid lengths match element for element.
_COLON_TOLERANCE = 1e-9

# Margin added around the vessel bounding box, in wavelengths. Wide enough for
# the point spread function tails of a bubble sitting on the edge of the
# vessel; the lateral PSF is about 0.76 lambda at the 9L-D imaging depth.
# Mirrors ROI_MARGIN_LAMBDA in process_run.m.
IMAGE_ROI_MARGIN_LAMBDA = 5.0


class AcquisitionGeometry:
    """The geometry of one acquisition, as the MATLAB pipeline sees it."""

    def __init__(self, settings: Dict, image_roi: Optional[Dict] = None):
        geom = settings["Geometry"]
        transducer = settings["Transducer"]
        transmit = settings["Transmit"]
        medium = settings["Medium"]

        self.rotation = np.asarray(geom.Rotation, dtype=float).reshape(3, 3)
        self.bbox_center = np.asarray(
            geom.BoundingBox.Center, dtype=float).reshape(3)
        self.bbox_diagonal = np.asarray(
            geom.BoundingBox.Diagonal, dtype=float).reshape(3)
        self.center = np.asarray(geom.Center, dtype=float).reshape(3)

        self.speed_of_sound = float(medium.SpeedOfSound)
        self.center_frequency = float(transmit.CenterFrequency)
        self.wavelength_mm = 1e3 * self.speed_of_sound / self.center_frequency
        self.pixel_size_mm = self.wavelength_mm / 5.0

        domain = geom.Domain
        self.domain_mm = {
            "x_min": 1e3 * float(domain.Xmin), "x_max": 1e3 * float(domain.Xmax),
            "y_min": 1e3 * float(domain.Ymin), "y_max": 1e3 * float(domain.Ymax),
            "z_min": 1e3 * float(domain.Zmin), "z_max": 1e3 * float(domain.Zmax),
        }

        # The vessel bounding box in the imaging frame. |R| turns the box's
        # half-extents into the half-extents of its axis-aligned envelope,
        # which is exact for the signed permutations the configs use and
        # correct for any rotation.
        half_mm = 1e3 * (np.abs(self.rotation) @ (self.bbox_diagonal / 2))
        centre_mm = 1e3 * self.center
        self.vessel_box_mm = {
            "axial": (centre_mm[0] - half_mm[0], centre_mm[0] + half_mm[0]),
            "lateral": (centre_mm[1] - half_mm[1], centre_mm[1] + half_mm[1]),
            "elevation": (centre_mm[2] - half_mm[2], centre_mm[2] + half_mm[2]),
        }

        # The beamforming grid follows that box, not the simulation domain.
        # The domain is sized by the transducer surface and the far edge of the
        # medium, so a domain-sized grid spends most of its pixels where no
        # vessel can be. The domain survives only as a clamp: nothing outside
        # it was simulated.
        (self.axial_range_mm, self.lateral_range_mm,
         self.image_roi_mode) = self._select_image_roi(image_roi)
        self.x_lat_mm = _colon(self.lateral_range_mm[0], self.pixel_size_mm,
                               self.lateral_range_mm[1])
        self.z_ax_mm = _colon(self.axial_range_mm[0], self.pixel_size_mm,
                              self.axial_range_mm[1])

        self.probe_type = str(transducer.Type)
        self.n_elements = int(transducer.NumberOfElements)
        self.pitch_mm = 1e3 * float(transducer.Pitch)
        self.aperture_mm = self.n_elements * self.pitch_mm
        self.element_height_mm = 1e3 * float(transducer.ElementHeight)
        self.elevation_focus_mm = 1e3 * float(transducer.ElevationFocus)

    def _select_image_roi(
            self, image_roi: Optional[Dict]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], str]:
        """Axial and lateral extent of the image, in millimetres.

        Mirrors ``select_image_roi`` in process_run.m: the vessel box widened
        by a fixed margin, or an explicit override, either way clamped to the
        simulation domain.
        """
        if image_roi:
            missing = [k for k in ("Depth", "Lateral") if k not in image_roi]
            if missing:
                raise ValueError(
                    "image_roi needs Depth and Lateral in metres, missing "
                    "{}".format(", ".join(missing)))
            axial = sorted(1e3 * float(v) for v in image_roi["Depth"])
            lateral = sorted(1e3 * float(v) for v in image_roi["Lateral"])
            mode = "explicit"
        else:
            margin = IMAGE_ROI_MARGIN_LAMBDA * self.wavelength_mm
            axial = [self.vessel_box_mm["axial"][0] - margin,
                     self.vessel_box_mm["axial"][1] + margin]
            lateral = [self.vessel_box_mm["lateral"][0] - margin,
                       self.vessel_box_mm["lateral"][1] + margin]
            mode = "vessel_bounding_box"

        axial = (max(axial[0], 0.0), min(axial[1], self.domain_mm["x_max"]))
        lateral = (max(lateral[0], self.domain_mm["y_min"]),
                   min(lateral[1], self.domain_mm["y_max"]))
        if axial[1] <= axial[0] or lateral[1] <= lateral[0]:
            raise ValueError(
                "image ROI is empty after clamping to the simulation domain: "
                "axial {}, lateral {}".format(axial, lateral))
        return axial, lateral, mode

    @classmethod
    def from_settings(cls, settings_path: str,
                      image_roi: Optional[Dict] = None) -> "AcquisitionGeometry":
        settings = loadmat(
            settings_path, squeeze_me=True, struct_as_record=False)
        missing = [k for k in ("Geometry", "Transducer", "Transmit", "Medium")
                   if k not in settings]
        if missing:
            raise KeyError(
                "{} is missing the settings structs {}".format(
                    settings_path, ", ".join(missing)))
        return cls(settings, image_roi=image_roi)

    # -- geometry ---------------------------------------------------------

    @property
    def lateral_limit_mm(self) -> float:
        """Rightmost column of the beamformed image.

        This was the half-width back when the grid was centred on zero. The
        grid follows the vessel box now, so read it as an edge, not a radius,
        and use ``lateral_range_mm`` when both edges matter.
        """
        return float(self.x_lat_mm[-1])

    @property
    def axial_limit_mm(self) -> float:
        """Deepest row of the beamformed image."""
        return float(self.z_ax_mm[-1])

    @property
    def image_shape(self) -> Tuple[int, int]:
        """(rows, columns) of the beamformed frame."""
        return (len(self.z_ax_mm), len(self.x_lat_mm))

    def suggested_elevation_mm(self) -> float:
        """A first estimate of the half slice thickness, lambda * F-number.

        The elevation filter is a modelling choice, not something the settings
        fix, but its scale follows the elevation aperture. Use this to notice
        that a value carried over from another probe is off by a factor, not as
        a derived truth.
        """
        f_number = self.elevation_focus_mm / self.element_height_mm
        return self.wavelength_mm * f_number

    def to_image_mm(self, points_world: np.ndarray) -> np.ndarray:
        """World points (N x 3, metres) to image millimetres.

        Returns an N x 3 array of (axial, lateral, elevation), the same three
        components ``load_gt_frame`` takes out of the rotated points.
        """
        points_world = np.atleast_2d(np.asarray(points_world, dtype=float))
        if points_world.shape[1] != 3:
            raise ValueError(
                "expected N x 3 world points, got shape {}".format(
                    points_world.shape))
        centred = points_world - self.bbox_center
        rotated = centred @ self.rotation.T + self.center
        return rotated * 1e3

    def to_pixels(self, image_mm: np.ndarray) -> np.ndarray:
        """Image millimetres to 1-based (column, row) pixels, NaN outside.

        Mirrors ``interp1(x_lat_mm, 1:N, ..., 'linear', NaN)``: the grid is
        uniform, so the index is affine in the coordinate, and a point beyond
        either end of the grid has no index at all.
        """
        image_mm = np.atleast_2d(np.asarray(image_mm, dtype=float))
        col = _index_on_grid(self.x_lat_mm, image_mm[:, 1])
        row = _index_on_grid(self.z_ax_mm, image_mm[:, 0])
        return np.column_stack([col, row])

    def classify(self, points_world: np.ndarray,
                 elevation_filter_mm: float) -> np.ndarray:
        """Label each world point 'in_frame', 'out_of_fov', or 'out_of_plane'."""
        return self.classify_image_mm(
            self.to_image_mm(points_world), elevation_filter_mm)

    def classify_image_mm(self, image_mm: np.ndarray,
                          elevation_filter_mm: float) -> np.ndarray:
        """Label points already in image millimetres.

        These are the geometry-only label outcomes.
        'in_frame' is not the same as MATLAB's 'valid': a point in frame can
        still be dropped as a weak response once the frame is beamformed.
        """
        image_mm = np.atleast_2d(np.asarray(image_mm, dtype=float))
        if image_mm.size == 0:
            return np.zeros(0, dtype=object)

        pixels = self.to_pixels(image_mm)
        rows, cols = self.image_shape

        col, row = pixels[:, 0], pixels[:, 1]
        in_fov = (~np.isnan(col) & ~np.isnan(row)
                  & (row >= 1) & (row <= rows)
                  & (col >= 1) & (col <= cols))
        in_plane = np.abs(image_mm[:, 2]) <= elevation_filter_mm

        reasons = np.full(len(image_mm), "in_frame", dtype=object)
        reasons[~in_fov] = "out_of_fov"
        reasons[in_fov & ~in_plane] = "out_of_plane"
        return reasons

    # -- reporting --------------------------------------------------------

    def describe(self) -> str:
        return "\n".join([
            "probe      {} | {} elements | pitch {:.3f} mm | "
            "aperture {:.1f} mm".format(
                self.probe_type, self.n_elements, self.pitch_mm,
                self.aperture_mm),
            "transmit   {:.2f} MHz | c {:.0f} m/s | lambda {:.3f} mm | "
            "pixel {:.3f} mm".format(
                self.center_frequency / 1e6, self.speed_of_sound,
                self.wavelength_mm, self.pixel_size_mm),
            "elevation  height {:.1f} mm | focus {:.1f} mm | "
            "suggested filter +-{:.2f} mm".format(
                self.element_height_mm, self.elevation_focus_mm,
                self.suggested_elevation_mm()),
            # The grid follows the vessel box, so neither extent is centred
            # on zero and the axial one does not start at the probe face.
            # Printing it as +-half-width and 0..depth described a grid this
            # code stopped building when the ROI moved to the vessel box.
            "image      lateral {:.2f}..{:.2f} mm | axial {:.2f}..{:.2f} mm | "
            "{} x {} px ({})".format(
                self.lateral_range_mm[0], self.lateral_range_mm[1],
                self.axial_range_mm[0], self.axial_range_mm[1],
                self.image_shape[0], self.image_shape[1],
                self.image_roi_mode),
        ])


def _colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB's ``start:step:stop``."""
    if step <= 0:
        raise ValueError("step must be positive")
    count = int(np.floor((stop - start) / step + _COLON_TOLERANCE)) + 1
    return start + step * np.arange(max(count, 0))


def _index_on_grid(grid_mm: np.ndarray, values_mm: np.ndarray) -> np.ndarray:
    """1-based fractional index of VALUES_MM on a uniform grid, NaN outside."""
    values_mm = np.asarray(values_mm, dtype=float)
    step = grid_mm[1] - grid_mm[0]
    index = 1.0 + (values_mm - grid_mm[0]) / step
    outside = (values_mm < grid_mm[0]) | (values_mm > grid_mm[-1])
    return np.where(outside, np.nan, index)


# -- ground truth ---------------------------------------------------------

def frame_files(gt_folder: str) -> List[str]:
    """The ground-truth frame files, in frame order rather than name order."""
    import glob
    import os
    import re

    paths = glob.glob(os.path.join(gt_folder, "Frame_*.mat"))

    def frame_number(path: str) -> int:
        match = re.search(r"Frame_(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    return sorted(paths, key=frame_number)


def frame_points(path: str,
                 pulses: Optional[Sequence[str]] = None) -> np.ndarray:
    """World points (N x 3, metres) from one ground-truth frame.

    PULSES defaults to every pulse the frame holds, which is what the dataset
    export labels; pass ``['Pulse1']`` for a single-pulse view of a multi-pulse
    scheme.
    """
    frame = loadmat(path, squeeze_me=True, struct_as_record=False)["Frame"]
    names = pulses if pulses is not None else [
        name for name in frame._fieldnames if name.startswith("Pulse")]

    blocks = []
    for name in names:
        if name not in frame._fieldnames:
            continue
        blocks.append(np.atleast_2d(
            np.asarray(getattr(frame, name).Points, dtype=float)))
    if not blocks:
        return np.zeros((0, 3))
    return np.vstack(blocks)


def survey(gt_folder: str, geometry: AcquisitionGeometry,
           elevation_filter_mm: float,
           pulses: Optional[Sequence[str]] = None) -> Dict:
    """Count how many seeded bubbles land in frame, over every GT frame.

    Returns the totals, the per-frame in-frame count, and the pooled image
    coordinates, so a caller can print a yield line, a distribution, or sweep
    the elevation filter without reading the frames again.
    """
    paths = frame_files(gt_folder)
    if not paths:
        raise FileNotFoundError("no Frame_*.mat in {}".format(gt_folder))

    blocks = []
    per_frame = []
    counts = {"in_frame": 0, "out_of_fov": 0, "out_of_plane": 0}
    for path in paths:
        points = frame_points(path, pulses)
        if len(points) == 0:
            per_frame.append(0)
            continue
        image_mm = geometry.to_image_mm(points)
        reasons = geometry.classify_image_mm(image_mm, elevation_filter_mm)
        blocks.append(image_mm)
        per_frame.append(int(np.sum(reasons == "in_frame")))
        for reason in counts:
            counts[reason] += int(np.sum(reasons == reason))

    image_mm = np.vstack(blocks) if blocks else np.zeros((0, 3))
    result = {
        "n_frames": len(paths),
        "n_points": len(image_mm),
        "image_mm": image_mm,
        "per_frame_in_frame": np.asarray(per_frame),
    }
    result.update(counts)
    return result
