from nanovllm_voxcpm.models.voxcpm2.model_utils import derive_decoder_config_fields, derive_encoder_config_fields


class TestDeriveEncoderConfigFields:
    def test_correct_fields_returned(self):
        fields = derive_encoder_config_fields(
            lm_config_hidden_size=2048,
            lm_config_intermediate_size=8192,
            lm_config_num_attention_heads=32,
            encoder_hidden_dim=512,
            encoder_ffn_dim=2048,
            encoder_num_heads=8,
            encoder_num_layers=4,
            encoder_kv_channels=64,
        )
        assert fields == {
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "kv_channels": 64,
            "vocab_size": 0,
        }

    def test_none_kv_channels_preserved(self):
        fields = derive_encoder_config_fields(
            lm_config_hidden_size=64,
            lm_config_intermediate_size=128,
            lm_config_num_attention_heads=4,
            encoder_hidden_dim=8,
            encoder_ffn_dim=16,
            encoder_num_heads=2,
            encoder_num_layers=1,
            encoder_kv_channels=None,
        )
        assert fields["kv_channels"] is None


class TestDeriveDecoderConfigFields:
    def test_correct_fields_returned(self):
        fields = derive_decoder_config_fields(
            dit_hidden_dim=256,
            dit_ffn_dim=1024,
            dit_num_heads=4,
            dit_num_layers=2,
            dit_kv_channels=32,
        )
        assert fields == {
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "kv_channels": 32,
            "vocab_size": 0,
        }

    def test_none_kv_channels_preserved(self):
        fields = derive_decoder_config_fields(
            dit_hidden_dim=8,
            dit_ffn_dim=16,
            dit_num_heads=2,
            dit_num_layers=1,
            dit_kv_channels=None,
        )
        assert fields["kv_channels"] is None
