# Parity tests

This directory previously contained GPU parity tests for VoxCPM.

What they are:
- GPU integration tests that compare engine output against a reference run.
- They depend on model weights and CUDA, so they are not suitable for default CPU CI.

Why there are no `.py` sources here now:
- Git history lookups for `tests/parity/` and the expected test filenames did not surface any recoverable tracked sources in this repo snapshot.
- The remaining `__pycache__` files are stale bytecode and should not be treated as canonical test sources.

How to run parity tests locally, if the sources are restored:
```bash
uv run pytest tests/parity -m gpu --model /path/to/VoxCPM1.5
```

Runtime code under test lives in `nanovllm_voxcpm.models.voxcpm2`.
