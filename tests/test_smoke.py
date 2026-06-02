"""Import smoke test: the package and its public API resolve."""


def test_package_imports():
    import murmur

    for name in murmur.__all__:
        assert hasattr(murmur, name), f"murmur.__all__ exports missing {name!r}"


def test_submodules_import():
    # Each subpackage imports cleanly (catches broken top-level imports/syntax).
    import murmur.inference.kv_cache  # noqa: F401
    import murmur.inference.local_engine  # noqa: F401
    import murmur.chunking.vad_chunker  # noqa: F401
    import murmur.metrics.asr_metrics  # noqa: F401


def test_benchmark_script_imports():
    import benchmark  # noqa: F401  (benchmarks/benchmark.py, via conftest path)
