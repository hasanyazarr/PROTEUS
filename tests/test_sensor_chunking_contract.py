"""Contract over the sensor-axis chunking in run_simulation_homogeneous.

The receive propagation accumulates into an [N_sensor x Nt] array, and on the
GPU that array is a single gpuArray -- which MATLAB caps at intmax('int32')
elements. v7 (lambda/6, 192 elements) needed 1.06e9 and fitted; v10 (lambda/8)
needs 2.52e9 and died at the allocation on line 37 with "Maximum variable size
allowed on the device is exceeded", before frame 1. It is not an out-of-memory:
10 GB of single fits an A100 several times over.

These tests hold the fix in place: the accumulator is chunked along the sensor
axis, and the expensive per-source work -- the transfer function and the
inverse transform, both over the distance grid -- stays outside the chunk loop,
because the distance grid does not depend on how the sensors are split.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def source() -> str:
    return read("acoustic-module/run_simulation_homogeneous.m")


def test_the_accumulator_is_never_allocated_whole_on_the_device():
    """The single [N_sensor x Nt] gpuArray is the bug. Allocation goes through
    the chunk bounds instead -- via prop_sensor_chunks, which adds the byte
    budget on top of sensor_chunk_bounds' element cap."""
    src = source()

    assert "gpuArray(zeros(N_sensor,kgrid.Nt, dataType))" not in src
    assert "prop_sensor_chunks(" in src


def test_the_chunk_size_is_a_byte_budget_not_only_the_element_cap():
    """The element cap bounds one array; it does not bound the chunk-sized
    temporary the accumulate rebuilds once per source. Sized by intmax alone
    that temporary is 4.69 GiB at v11's grid. The propagation path budgets it
    the way rf_sensor_chunks and rf_element_blocks already do."""
    src = read("acoustic-module/prop_sensor_chunks.m")

    assert "CHUNK_BYTES" in src
    assert "sensor_chunk_bounds(" in src
    # The tests-only override has to keep working: it is how the split is
    # exercised without a GPU.
    assert "SensorChunkElements" in src


def test_the_preflight_reports_device_residency():
    """Chunking bounds each array, never the resident total -- every chunk is
    allocated before the source loop. The banner has to say what that total is,
    or a config that cannot fit still looks fine in the preflight."""
    src = read("acoustic-module/preflight_array_limits.m")

    assert "prop_sensor_chunks(" in src
    assert "propagation accumulator" in src
    assert "AvailableMemory" in src


def test_chunk_bounds_live_in_their_own_file():
    """So the partition can be tested in MATLAB without a GPU or a frame."""
    assert (ROOT / "acoustic-module" / "sensor_chunk_bounds.m").is_file()
    assert (ROOT / "acoustic-module" / "prop_sensor_chunks.m").is_file()
    assert (ROOT / "tests" / "matlab" / "test_sensor_chunk_bounds.m").is_file()


def test_the_expensive_per_source_work_stays_out_of_the_chunk_loop():
    """W and the inverse transform are computed over the distance grid, whose
    size does not depend on the sensor split. Recomputing them per chunk would
    multiply prop -- the largest stage of the frame -- by the chunk count."""
    src = source()

    per_source = src.index("for m = 1:N_source")
    field = src.index("W = exp(-d * alpha_f", per_source)
    # The chunk loop that accumulates -- not the one that preallocates.
    chunk_loop = src.index("for c = 1:N_chunk", per_source)
    assert field < chunk_loop, "the transfer function is inside the chunk loop"
    assert src.index("p = p(:, 1:N_ext/2);", per_source) < chunk_loop


def test_self_sensing_is_still_masked_on_sensor_rows_within_the_chunk():
    """d0 is sensor-length and the chunk holds a slice of it, so the mask has
    to be sliced with it -- masking the chunk with the whole d0 is the same
    length mismatch the gridded path had, one level down."""
    src = source()

    assert "p_sensor(d0(rows) == 0,:) = 0;" in src
    assert "p_sensor(d0 == 0,:) = 0;" not in src


def test_the_chunks_are_only_joined_on_the_host():
    """One [N_sensor x Nt] array is over the device limit by construction
    whenever there is more than one chunk, so the join gathers."""
    src = source()

    assert "gather(sensor_p{c})" in src


def test_the_split_is_announced_once():
    """A run whose grid crossed the limit changed its memory shape; the log
    should say so rather than leave it to be inferred from timings."""
    src = source()

    assert "run_log('banner', 'sensorchunks'" in src
