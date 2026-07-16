#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/gpu_smoke.sh MODE

Manual-only CUDA smoke evaluation. This script is not intended for CI.

Modes:
  --single  Run the curated single-GPU CUDA smoke suites. Defaults an unset
            CUDA_VISIBLE_DEVICES to 0; an explicitly empty value remains hidden.
  --tp      Run the two-rank NCCL tensor-parallel smoke suite. Exactly two CUDA
            devices must be visible to PyTorch.
  --help, -h
            Show this help text.

Examples:
  CUDA_VISIBLE_DEVICES=0 bash scripts/gpu_smoke.sh --single
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/gpu_smoke.sh --tp
  CUDA_VISIBLE_DEVICES="" bash scripts/gpu_smoke.sh --single
  CUDA_VISIBLE_DEVICES=0 bash scripts/gpu_smoke.sh --tp
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

mode="$1"
case "$mode" in
    --help|-h)
        usage
        exit 0
        ;;
    --single|--tp)
        ;;
    *)
        echo "Unknown mode: $mode" >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$mode" == "--single" && ! -v CUDA_VISIBLE_DEVICES ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "=== Phase: prerequisite checks ==="
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Linux required for GPU smoke" >&2
    exit 1
fi

if ! uv run python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "PyTorch reports no CUDA available" >&2
    exit 1
fi

if ! uv run python -c "import flash_attn"; then
    echo "flash-attn import failed" >&2
    exit 1
fi

if ! uv run python -c "import triton"; then
    echo "Triton import failed" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for GPU smoke diagnostics" >&2
    exit 1
fi

if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    echo "CUDA_VISIBLE_DEVICES is unset"
fi
echo "System GPUs:"
nvidia-smi --query-gpu=index,name --format=csv,noheader

visible_device_count="$(uv run python -c "import torch; print(torch.cuda.device_count())")"
echo "PyTorch visible CUDA device count: ${visible_device_count}"
echo "PyTorch visible GPU names:"
uv run python -c "import torch; [print(f'{index},{torch.cuda.get_device_name(index)}') for index in range(torch.cuda.device_count())]"

case "$mode" in
    --single)
        if [[ "$visible_device_count" -lt 1 ]]; then
            echo "GPU smoke requires at least 1 visible CUDA device" >&2
            exit 1
        fi

        echo "=== Phase: single-GPU smoke ==="
        uv run pytest -m gpu tests/unit/test_lora_cuda.py tests/unit/test_attention_layers.py -q
        echo "GPU_SMOKE_SINGLE_OK"
        ;;
    --tp)
        if [[ "$visible_device_count" -ne 2 ]]; then
            echo "GPU smoke requires 2 visible CUDA devices" >&2
            exit 1
        fi

        echo "=== Phase: two-rank TP smoke ==="
        timeout 60s uv run torchrun --standalone --nproc-per-node=2 -m pytest tests/unit/test_tp_gpu_smoke.py -q -s
        echo "GPU_SMOKE_TP_OK"
        ;;
esac
