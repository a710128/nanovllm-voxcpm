# Contributing: Writing CPU-Testable Tests

This guide explains how to write and run tests for a GPU-centric codebase where CI has no CUDA device.
Read `AGENTS.md` for code style, build, and project layout. This doc focuses entirely on **testing**.

---

## Prerequisites

```bash
uv sync --all-packages --frozen
uv run pytest -m "not gpu" -q   # must pass on a CPU-only machine
```

---

## How tests work

Two test roots, both discovered by a plain `uv run pytest`:

| Root | What it covers |
|---|---|
| `tests/unit/` | Core library (`nanovllm_voxcpm/`) |
| `deployment/tests/` | FastAPI server (`deployment/app/`) |

`conftest.py` in `tests/unit/` registers the `gpu` marker and installs import shims before any test
module is collected, so GPU-absent imports don't fail at collection time.

---

## GPU vs CPU tests

Mark a test `@pytest.mark.gpu` when it **must** run real CUDA kernels and cannot be faked:

```python
@pytest.mark.gpu
def test_flash_attn_varlen_output():
    ...
```

`conftest.py:pytest_collection_modifyitems` automatically skips every `@pytest.mark.gpu` test when
`torch.cuda.is_available()` is `False` or `CUDA_VISIBLE_DEVICES` is empty. You don't need any
`pytest.importorskip("torch.cuda")` boilerplate.

Use `# pragma: no cover` only for branches that are physically unreachable on CPU (e.g. the interior
of a `_unavailable` shim function). Don't use it to hide testable logic.

---

## CPU-test idioms

### 1. conftest.py shims (lines 57-127)

`tests/unit/conftest.py` installs lightweight stub modules for `flash_attn`, `triton`,
`huggingface_hub`, `pydantic`, and `transformers` when those packages aren't importable. Each stub
exposes just enough surface area for imports to resolve:

```python
# conftest.py (excerpt, lines 77-86)
if not _module_available("flash_attn"):
    flash_attn = _ensure_module("flash_attn")
    def _unavailable(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("flash_attn is not available in unit tests")
    setattr(flash_attn, "flash_attn_varlen_func", _unavailable)
```

Don't add new per-test stubs for these packages. The conftest shim is the single source of truth.

### 2. `__new__` shell + attribute seeding

When `__init__` loads tokenizer weights or starts GPU workers, bypass it with `__new__` and seed
only the attributes your test exercises. From `tests/unit/test_voxcpm_engine_max_model_len.py:16-30`:

```python
from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMEngine

e = VoxCPMEngine.__new__(VoxCPMEngine)
e.max_model_len = 4
e.tokenizer = lambda _s: list(range(token_count))
e.add_sequence = lambda seq: setattr(e, "_captured_seq", seq)
e.resolve_lora = lambda name: None if name is None else 7
```

Only seed the attributes the method under test actually reads.

### 3. `_DummyRunner` fake worker

`LLMEngineBase` spawns multiprocessing runners. Replace the runner class before constructing the
engine so no GPU process is ever spawned. From `tests/unit/test_llm_engine.py:17-36`:

```python
class _DummyRunner:
    fail_register = False
    instances: list["_DummyRunner"] = []

    def __init__(self, config, rank, device_idx, distributed_port, event):
        self.calls: list[tuple[str, tuple]] = []
        type(self).instances.append(self)

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "run":
            return [{"token": f"token-{i}".encode(), "stop": True} for i, _ in enumerate(args[0])]
        return None
```

### 4. `set_backend_for_testing()` for LoRA

Swap out LoRA backend availability without touching real GPU state (from `test_llm_engine.py`):

```python
from nanovllm_voxcpm.lora import set_backend_for_testing
set_backend_for_testing(_AvailableBackend())
```

Reset after the test or use a `finally` block so other tests aren't affected.

### 5. Monkeypatching the CUDA edge

For scheduler logic gated on `block_manager.can_allocate`, patch the method directly rather than
mocking CUDA. From `tests/unit/test_scheduler.py`:

```python
monkeypatch.setattr(sched.block_manager, "can_allocate", lambda seq: seq.seq_id != "blocked")
```

pytest's `monkeypatch` fixture rolls back the patch automatically after the test.

---

## Running coverage locally

```bash
bash scripts/coverage.sh
```

This runs `pytest -m "not gpu"` across both test roots, writes `coverage.xml` and `htmlcov/`. The
script uses `[tool.coverage.*]` config from `pyproject.toml` so you don't need extra `--cov=` flags.

---

## Before pushing: check diff-cover locally

New code added to CPU-testable paths (scheduler, engine logic, LoRA management) must be covered. Check
what your branch adds before opening a PR:

```bash
uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=80 \
  --include 'nanovllm_voxcpm/engine/**' \
  --include 'nanovllm_voxcpm/models/voxcpm/**' \
  --include 'nanovllm_voxcpm/lora/**'
```

`diff-cover` reports coverage only on lines changed relative to `origin/main`. The CI job runs the
same command and will fail the build if new CPU-testable lines are uncovered.

---

## Style checklist

- Black, 120-char line length (see `[tool.black]` in `pyproject.toml`)
- Google-style docstrings (Args / Returns / Raises)
- `snake_case` for functions and variables, `PascalCase` for classes
- No mutable defaults; use `None` and normalize in `__init__`
- See `AGENTS.md` for the full style reference
