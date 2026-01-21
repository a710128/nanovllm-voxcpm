# AGENTS.md (nano-vllm-voxcpm)

This repository is a Python package (`nano-vllm-voxcpm`) plus a small FastAPI demo.
It is GPU-centric (PyTorch + Triton + flash-attn); many runtime paths assume CUDA.

- Python: >=3.10,<3.13 (see `pyproject.toml`)
- Package manager: `uv` (lockfile: `uv.lock`)

## Repo Layout

- `nanovllm_voxcpm/`: core library
- `nanovllm_voxcpm/engine/`: scheduler, KV cache, runner lifecycle
- `nanovllm_voxcpm/models/voxcpm/`: VoxCPM model integration (engine/server/runner/model)
- `fastapi/`: demo server (`fastapi/app.py`)
- `tests/`: pytest tests (mostly `tests/unit/`)

## Docs Worth Reading First

- `ARCHITECTURE.md`: end-to-end runtime architecture and module relationships

## Setup (uv)

Use the lockfile (recommended):
```bash
uv sync --frozen
```

FastAPI demo deps:
```bash
uv pip install -r fastapi/requirements.txt
```

Notes:
- Run commands inside the managed env: `uv run <cmd...>`
- Prefer `uv sync --frozen` to keep `uv.lock` stable

## Run

Core example:
```bash
uv run python example.py
```

FastAPI demo:
```bash
uv run fastapi run fastapi/app.py
```

## Build / Lint / Test

### Quick Sanity (syntax)
```bash
uv run python -m compileall nanovllm_voxcpm fastapi tests
```

### Build (sdist/wheel)
`build` may not be installed by default:
```bash
uv pip install -U build
uv run python -m build
```

### Lint / Format (optional tooling)
Black config exists in `pyproject.toml` (line length 120). Ruff is not configured
in-repo but can be used ad-hoc.

```bash
uv pip install -U black ruff
uv run black .
uv run ruff check .
```

### Tests (pytest)
Install (if needed):
```bash
uv pip install -U pytest
```

Run all tests:
```bash
uv run pytest
```

Run a single file:
```bash
uv run pytest tests/unit/test_scheduler.py
```

Run a single test (most useful for agents):
```bash
uv run pytest tests/unit/test_scheduler.py::test_scheduler_prefill_then_decode_round_robin -q
```

Run a single test by keyword (handy when test names are long):
```bash
uv run pytest -k "prefill_then_decode" -q
```

Run by keyword:
```bash
uv run pytest -k "scheduler and not slow"
```

Re-run last failures:
```bash
uv run pytest --lf
```

Test environment notes:
- Unit tests provide import shims for some GPU-only deps so CPU imports work.
- TorchDynamo/Inductor may be disabled in unit tests to avoid compilation/toolchain
  requirements; do not assume `torch.compile` paths execute under pytest.

## Code Style Guidelines (Project Conventions)

**Formatting**
- 4 spaces; keep diffs small; Black-compatible formatting preferred
- Line length: 120 (see `[tool.black]` in `pyproject.toml`)

**Imports**
- Order: stdlib, third-party, then `nanovllm_voxcpm...`
- Avoid `import *`
- Prefer explicit imports over heavy aliasing

**Typing**
- Prefer modern typing: `X | None`, `list[int]`, `dict[str, T]`
- Use `typing_extensions` only when needed
- Avoid `Any` unless crossing a truly dynamic boundary (framework hooks, dict payloads)
- Prefer precise iterators/generators for streaming APIs (`AsyncIterator[bytes]`, etc.)

**Naming**
- `snake_case` for functions/vars/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Use descriptive tensor/shape names (`num_seqs`, `seq_len`, `max_num_batched_tokens`)

**Docstrings / API Docs**
- Use Google-style docstrings (Args/Returns/Raises/Yields) consistently
- Docstrings should be self-contained (avoid referencing local-only files/paths)
- For FastAPI endpoints, prefer explicit `response_model=` and `Field(..., description=...)`

**Mutable defaults**
- Do NOT use mutable defaults like `devices: list[int] = []`
- Prefer `devices: list[int] | None = None` and normalize inside `__init__`

**Error handling**
- `ValueError` for invalid user input / inconsistent args
- `FileNotFoundError` for missing paths
- `RuntimeError` for unexpected state/runtime conditions
- Avoid swallowing exceptions; add context and preserve the original error (`raise ... from e`)

**Torch / CUDA performance**
- Use `torch.inference_mode()` for inference paths
- Minimize sync points (`.item()`, `.cpu()`, implicit device sync) in hot loops
- Be explicit about device/dtype when converting buffers (e.g. float32 waveform bytes)

**Multiprocessing / async**
- Multiprocessing uses `spawn`; keep args picklable
- Guard scripts with `if __name__ == "__main__":`
- For async streaming endpoints, keep a clear contract: yielded chunks are bytes-ready

**Data / artifacts**
- Do not commit large binaries (`*.wav`, `*.pt`, `*.safetensors`); they are ignored by `.gitignore`

## Cursor / Copilot Rules

No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.
