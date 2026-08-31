"""Contract over the element blocking in compute_RF and its preflight.

Commit a06965a bounded the receive propagation's [N_sensor x Nt] accumulator
and stopped there. The same record then reached compute_RF whole, and the v10
run died at its gpuArray upload after two hours of transmit simulation, with
zero frames written. Four of compute_RF's arrays are over the intmax('int32')
element cap at that grid, not one:

    gpuArray(sensor_data.p)   201795 x 12444 = 2.51e9   1.17x the limit
    sensor_weights*p          190080 x 12444 = 2.37e9   1.10x
    fft(p,N,2)                190080 x 12500 = 2.38e9   1.11x
    exp(-2i*pi*delays*f)      190080 x 12500 = 2.38e9   1.11x

The sensor axis is reduced away by the first product, so splitting it bounds
the upload and nothing after it -- chunking that axis alone would have failed
again four lines further down. The element axis bounds all four, and these
tests hold that shape in place, together with the two things that turn a
repeat of this into a cheap failure instead of an expensive one: a preflight
that runs before the transmit, and a CPU retry that is not keyed to
out-of-memory alone.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def test_the_record_is_never_uploaded_or_multiplied_whole():
    """The unconditional upload and the full-width product are the bug."""
    src = read("acoustic-module/compute_RF.m")

    assert "sensor_data.p  = gpuArray(sensor_data.p);" not in src
    assert "p = sensor_weights*double(sensor_data.p);" not in src
    assert "rf_sensor_chunks(" in src
    assert "rf_element_blocks(" in src


def test_the_work_that_is_sized_by_integration_points_is_inside_the_block_loop():
    """The transform, the delay and the apodization are the three arrays the
    element axis exists to bound. Any of them left outside the loop is the
    whole [N_el*N_int x N] array again."""
    src = read("acoustic-module/compute_RF.m")

    loop = src.index("for b = 1:N_block")
    assert src.index("p = fft(p,N,2);", loop) > loop
    assert src.index("exp(-2*pi*1i*delays_b*f)", loop) > loop
    assert src.index("reshape(p,n_el_b,N_int,N)", loop) > loop


def test_the_impulse_response_convolution_stays_outside_the_block_loop():
    """It acts along time on the assembled [N_el x N] result, which no grid
    takes over the limit. Per block it would pad every block separately and
    change the answer at the seams."""
    src = read("acoustic-module/compute_RF.m")

    assert "convn(p_all,IR)" in src
    assert src.index("convn(p_all,IR)") > src.index("for b = 1:N_block")


def test_both_axes_partition_through_the_one_helper():
    """sensor_chunk_bounds is balanced, tested, and already carries the
    reasoning about the limit. A second partitioner would drift from it."""
    assert "sensor_chunk_bounds(" in read("acoustic-module/rf_element_blocks.m")
    assert "sensor_chunk_bounds(" in read("acoustic-module/rf_sensor_chunks.m")


def test_the_padded_length_has_one_definition():
    """The preflight has to size the arrays compute_RF will build. A second
    copy of the padding arithmetic would let the check drift from the run."""
    src = read("acoustic-module/compute_RF.m")

    assert "rf_signal_length(" in src
    assert "optimize_grid_size(" not in src
    assert "optimize_grid_size(" in read("acoustic-module/rf_signal_length.m")


def test_the_cpu_retry_is_not_keyed_to_out_of_memory_alone():
    """The size-limit failure is not an OOM, so the old identifier match
    rethrew it. Every parallel:gpu:* failure describes a device that cannot do
    the work, and the host can."""
    src = read("acoustic-module/compute_RF_data.m")

    assert "contains(ME.identifier, 'parallel:gpu:array:OOM')" not in src
    assert "startsWith(ME.identifier, 'parallel:gpu:')" in src


def test_the_preflight_refuses_rather_than_warns():
    """A warning would still cost the transmit. Once a single element's own
    work is over the limit no blocking helps, and the run must not start."""
    src = read("acoustic-module/preflight_array_limits.m")

    assert "error('PROTEUS:preflight:arrayOverLimit'" in src
    assert "rf_element_blocks(" in src
    assert "rf_sensor_chunks(" in src


def test_the_preflight_runs_before_any_transmit():
    """Its whole point is to fail in seconds rather than after two hours of
    k-Wave. Both acquisition paths have to reach it first."""
    src = read("acoustic-module/main_RF.m")

    first_check = src.index("preflight_array_limits(")
    first_transmit = src.index("run_simulation(")
    assert first_check < first_transmit

    # Once per path: the hybrid one and the full_simulator one.
    assert src.count("preflight_array_limits(") == 2


def test_the_transducer_record_is_accumulated_in_place():
    """Both operands are the full [N_sensor x Nt] record -- 9.4 GB each at
    v10's grid -- so summing into a third array put the host peak at three
    copies of it."""
    src = read("acoustic-module/main_RF.m")

    assert "sensor_data.p = sensor_data_trans.p + sensor_data.p;" not in src
    assert "sensor_data.p(:,cols) = sensor_data.p(:,cols) + ..." in src


def test_the_blocking_is_forceable_without_a_gpu():
    """Neither split can be exercised on a GPU in CI, so the MATLAB
    equivalence test forces the same partitions on the CPU."""
    assert "RFBlockElements" in read("acoustic-module/rf_element_blocks.m")
    assert "RFSensorChunkElements" in read("acoustic-module/rf_sensor_chunks.m")
    assert (ROOT / "tests" / "matlab" / "test_compute_RF_blocking.m").is_file()
