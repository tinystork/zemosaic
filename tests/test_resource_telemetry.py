import zemosaic_resource_telemetry as rt
from zemosaic_align_stack_gpu import _is_cupy_runtime_unavailable_error


def test_gpu_defaults_are_stable():
    info = rt._gpu_info_defaults()
    assert info["gpu_used_mb"] is None
    assert info["gpu_total_mb"] is None
    assert info["gpu_free_mb"] is None
    assert info["gpu_util_percent"] is None
    assert info["gpu_mem_util_percent"] is None


def test_runtime_sampler_returns_legacy_and_new_gpu_keys():
    info = rt._sample_runtime_resources_for_telemetry()
    for key in (
        "cpu_percent",
        "ram_used_mb",
        "ram_total_mb",
        "ram_available_mb",
        "gpu_used_mb",
        "gpu_total_mb",
        "gpu_free_mb",
        "gpu_util_percent",
        "gpu_mem_util_percent",
        "gpu_name",
        "gpu_backend",
        "gpu_sample_source",
    ):
        assert key in info


def test_detects_missing_nvrtc_error_variants():
    assert _is_cupy_runtime_unavailable_error(RuntimeError('Failure finding "libnvrtc.so": No such file'))
    assert _is_cupy_runtime_unavailable_error(RuntimeError("nvrtc64_120_0.dll not found"))
    assert not _is_cupy_runtime_unavailable_error(RuntimeError("plain arithmetic failure"))
