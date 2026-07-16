import pytest

torch = pytest.importorskip("torch")


class _CapturingEstimator:
    """Fake CFM estimator that records the inputs it receives each Euler step."""

    def __init__(self, in_channels: int, patch: int):
        self.in_channels = in_channels
        self.patch = patch
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, x_in, mu_in, t_in, cond_in, dt_in):
        self.calls.append(
            {
                "x_in": x_in.clone(),
                "mu_in": mu_in.clone(),
                "t_in": t_in.clone(),
                "cond_in": cond_in.clone(),
                "dt_in": dt_in.clone(),
            }
        )
        return torch.zeros(x_in.size(0), self.in_channels, self.patch, dtype=x_in.dtype)


def _run_solver(mean_mode: bool):
    from nanovllm_voxcpm.models.voxcpm2.model import UnifiedCFM

    bsz, in_channels, patch = 2, 4, 3
    n_steps = 6

    cfm = UnifiedCFM.__new__(UnifiedCFM)
    cfm.in_channels = in_channels
    cfm.mean_mode = mean_mode
    cfm.estimator = _CapturingEstimator(in_channels, patch)

    x = torch.randn(bsz, in_channels, patch)
    mu = torch.randn(bsz, in_channels * patch)
    cond = torch.randn(bsz, in_channels, patch)
    cfg_value = torch.ones(bsz)
    t_span = torch.linspace(1, 0, n_steps + 1)

    cfm.solve_euler(x, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value)
    return cfm.estimator, bsz, mu, cond


def test_euler_unconditioned_mu_half_is_zeroed():
    estimator, bsz, mu, _cond = _run_solver(mean_mode=False)

    assert estimator.calls, "estimator was never invoked"
    for call in estimator.calls:
        mu_in = call["mu_in"]
        # Conditioned half carries mu; unconditioned half MUST be zero even though
        # the buffer is pre-allocated once with torch.empty (uninitialized memory).
        torch.testing.assert_close(mu_in[:bsz], mu, rtol=0, atol=0)
        torch.testing.assert_close(mu_in[bsz:], torch.zeros_like(mu_in[bsz:]), rtol=0, atol=0)


def test_euler_buffers_fully_overwritten_each_step():
    estimator, bsz, _mu, cond = _run_solver(mean_mode=False)

    for call in estimator.calls:
        x_in, cond_in = call["x_in"], call["cond_in"]
        # Both halves of x_in / cond_in are written every step, so no stale/garbage
        # values leak from the reused empty buffers.
        torch.testing.assert_close(x_in[:bsz], x_in[bsz:], rtol=0, atol=0)
        torch.testing.assert_close(cond_in[:bsz], cond, rtol=0, atol=0)
        torch.testing.assert_close(cond_in[bsz:], cond, rtol=0, atol=0)


def test_euler_dt_zeroed_when_not_mean_mode():
    estimator, _bsz, _mu, _cond = _run_solver(mean_mode=False)

    for call in estimator.calls:
        dt_in = call["dt_in"]
        torch.testing.assert_close(dt_in, torch.zeros_like(dt_in), rtol=0, atol=0)
