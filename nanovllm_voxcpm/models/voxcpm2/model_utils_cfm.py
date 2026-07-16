from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class EulerSolverInputs:
    x: torch.Tensor
    t_span: torch.Tensor
    mu: torch.Tensor
    cond: torch.Tensor
    cfg_value: torch.Tensor


@dataclass(frozen=True, slots=True)
class EulerSolverConfig:
    in_channels: int
    mean_mode: bool


class EulerEstimator(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor: ...


class EulerSolverOps(Protocol):
    estimator: EulerEstimator

    def optimized_scale(self, positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor: ...


def compute_optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor:
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def build_cfm_t_span(inference_timesteps: int, device: torch.device | None = None) -> torch.Tensor:
    t_span = torch.linspace(1, 0, inference_timesteps + 1, device=device, dtype=torch.float32)
    return t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)


def compute_zero_init_steps(t_span_len: int) -> int:
    return max(1, int(t_span_len * 0.04))


def solve_euler(inputs: EulerSolverInputs, config: EulerSolverConfig, ops: EulerSolverOps) -> torch.Tensor:
    """Integrate the CFM trajectory while reusing fully overwritten buffers."""
    x = inputs.x
    t_span = inputs.t_span
    t, dt = t_span[0], t_span[0] - t_span[1]
    zero_init_steps = compute_zero_init_steps(len(t_span))
    bsz = x.size(0)
    x_in = torch.empty([2 * bsz, config.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
    mu_in = torch.empty([2 * bsz, inputs.mu.size(1)], device=x.device, dtype=x.dtype)
    t_in = torch.empty([2 * bsz], device=x.device, dtype=x.dtype)
    dt_in = torch.empty([2 * bsz], device=x.device, dtype=x.dtype)
    cond_in = torch.empty([2 * bsz, config.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
    for step in range(1, len(t_span)):
        if step <= zero_init_steps:
            dphi_dt = 0.0
        else:
            x_in[:bsz], x_in[bsz:] = x, x
            mu_in[:bsz] = inputs.mu
            mu_in[bsz:].zero_()
            t_in[:bsz] = t.unsqueeze(0)
            t_in[bsz:] = t.unsqueeze(0)
            dt_in[:bsz] = dt.unsqueeze(0)
            dt_in[bsz:] = dt.unsqueeze(0)
            if not config.mean_mode:
                dt_in.zero_()
            cond_in[:bsz], cond_in[bsz:] = inputs.cond, inputs.cond
            dphi_dt = ops.estimator(x_in, mu_in, t_in, cond_in, dt_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            st_star = ops.optimized_scale(dphi_dt.view(bsz, -1), cfg_dphi_dt.view(bsz, -1))
            st_star = st_star.view(bsz, *([1] * (len(dphi_dt.shape) - 1)))
            dphi_dt = cfg_dphi_dt * st_star + inputs.cfg_value[:, None, None] * (dphi_dt - cfg_dphi_dt * st_star)
        x = x - dt * dphi_dt
        t = t - dt
        sol = x
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]
    return sol
