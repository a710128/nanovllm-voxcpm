from nanovllm_voxcpm.utils.seed import derive_step_seed


def test_derive_step_seed_is_stable_for_same_request_step():
    assert derive_step_seed(123, 4) == derive_step_seed(123, 4)


def test_derive_step_seed_advances_for_different_steps():
    seeds = {derive_step_seed(123, step) for step in range(8)}
    assert len(seeds) == 8
