import pytest


@pytest.mark.parametrize(
    ("module_path", "config_name"),
    [
        ("nanovllm_voxcpm.models.voxcpm.config", "LoRAConfig"),
        ("nanovllm_voxcpm.models.voxcpm2.config", "LoRAConfig"),
    ],
)
def test_lora_config_forbids_removed_rank_and_alpha_fields(module_path, config_name):
    module = __import__(module_path, fromlist=[config_name])
    config_cls = getattr(module, config_name)

    with pytest.raises(Exception, match="r|alpha|extra"):
        config_cls(r=8, alpha=16.0)
