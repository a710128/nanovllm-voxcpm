import importlib.util
from pathlib import Path

import pytest


def _load_benchmark():
    path = Path(__file__).resolve().parents[2] / "benchmark" / "bench_streaming_latency.py"
    spec = importlib.util.spec_from_file_location("bench_streaming_latency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streaming_latency_metrics_separate_first_chunk_and_subsequent_gaps(monkeypatch):
    module = _load_benchmark()
    monkeypatch.setattr(module.time, "perf_counter", lambda: 10.5)
    result = module.summarize_request(10.0, [10.1, 10.15, 10.2], [160, 160, 80], 1000)
    assert result["ttfb_ms"] == pytest.approx(100)
    assert result["inter_chunk_ms"] == pytest.approx([50, 50])
    assert result["chunk_arrival_ms"] == pytest.approx([100, 150, 200])
    assert result["first_chunk_audio_ms"] == 160
    assert result["audio_s"] == 0.4
    assert result["wall_s"] == 0.5
    assert result["num_chunks"] == 3


def test_streaming_latency_accepts_a_single_chunk():
    module = _load_benchmark()
    result = module.summarize_request(0.0, [0.1], [160], 1000)
    assert result["inter_chunk_ms"] == []
    assert result["num_chunks"] == 1


def test_streaming_latency_rejects_empty_generation():
    module = _load_benchmark()
    with pytest.raises(RuntimeError, match="no nonempty audio chunks"):
        module.summarize_request(0.0, [], [], 1000)
