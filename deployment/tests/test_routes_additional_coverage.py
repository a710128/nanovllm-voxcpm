import base64

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("prometheus_client")
pytest.importorskip("lameenc")


from starlette.testclient import TestClient


def test_ready_returns_503_when_not_ready(app):
    with TestClient(app) as client:
        client.app.state.ready = False
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json() == {"detail": "not ready"}


def test_metrics_endpoint_exposes_prometheus_text(app):
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("text/plain")
        assert "nanovllm_http_requests_total" in r.text


def test_info_returns_503_when_server_missing(app):
    with TestClient(app) as client:
        delattr(client.app.state, "server")
        r = client.get("/info")
        assert r.status_code == 503
        assert r.json() == {"detail": "Model server not ready"}


def test_loras_returns_503_when_server_missing(app):
    with TestClient(app) as client:
        delattr(client.app.state, "server")
        r = client.get("/loras")
        assert r.status_code == 503
        assert r.json() == {"detail": "Model server not ready"}


def test_lora_endpoints_validate_registration_errors(app):
    with TestClient(app) as client:
        r = client.delete("/loras/missing")
        assert r.status_code == 400
        assert "not registered" in r.json()["detail"]

        r = client.post("/loras", json={"name": "demo", "path": "/tmp/demo"})
        assert r.status_code == 200

        r = client.post("/loras", json={"name": "demo", "path": "/tmp/demo"})
        assert r.status_code == 400
        assert "already registered" in r.json()["detail"]


def test_generate_rejects_mutually_exclusive_prompt_wav_and_latents_400(app):
    wav_b64 = base64.b64encode(b"FAKEWAV").decode("utf-8")
    latents_b64 = base64.b64encode(b"LATENTS").decode("utf-8")
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={
                "target_text": "hi",
                "prompt_wav_base64": wav_b64,
                "prompt_wav_format": "wav",
                "prompt_latents_base64": latents_b64,
            },
        )
        assert r.status_code == 400
        assert "mutually exclusive" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload, expected_detail",
    [
        (
            {
                "target_text": "hi",
                "prompt_wav_base64": base64.b64encode(b"x").decode("utf-8"),
                "prompt_text": "p",
            },
            "wav prompt requires prompt_wav_base64 + prompt_wav_format",
        ),
        (
            {
                "target_text": "hi",
                "prompt_wav_base64": base64.b64encode(b"x").decode("utf-8"),
                "prompt_wav_format": "wav",
                "prompt_text": "",
            },
            "wav prompt requires non-empty prompt_text",
        ),
        (
            {
                "target_text": "hi",
                "prompt_latents_base64": base64.b64encode(b"x").decode("utf-8"),
            },
            "latents prompt requires non-empty prompt_text",
        ),
        (
            {"target_text": "hi", "prompt_text": "should-fail"},
            "prompt_text is not allowed for zero-shot",
        ),
    ],
)
def test_generate_prompt_validation_matrix_400(app, payload, expected_detail):
    with TestClient(app) as client:
        r = client.post("/generate", json=payload)
        assert r.status_code == 400
        assert r.json() == {"detail": expected_detail}


def test_generate_invalid_base64_prompt_wav_returns_400(app):
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={
                "target_text": "hi",
                "prompt_wav_base64": "a",  # incorrect padding -> base64 decode error
                "prompt_wav_format": "wav",
                "prompt_text": "p",
            },
        )
        assert r.status_code == 400
        assert "Invalid base64 in prompt_wav_base64" in r.json()["detail"]


def test_generate_invalid_base64_prompt_latents_returns_400(app):
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={
                "target_text": "hi",
                "prompt_latents_base64": "a",  # incorrect padding -> base64 decode error
                "prompt_text": "p",
            },
        )
        assert r.status_code == 400
        assert "Invalid base64 in prompt_latents_base64" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload, expected_detail",
    [
        (
            {
                "target_text": "hi",
                "ref_audio_wav_base64": base64.b64encode(b"x").decode("utf-8"),
                "ref_audio_latents_base64": base64.b64encode(b"y").decode("utf-8"),
            },
            "ref_audio_wav_* and ref_audio_latents_base64 are mutually exclusive",
        ),
        (
            {
                "target_text": "hi",
                "ref_audio_wav_base64": base64.b64encode(b"x").decode("utf-8"),
            },
            "reference wav requires ref_audio_wav_base64 + ref_audio_wav_format",
        ),
    ],
)
def test_generate_ref_audio_validation_matrix_400(app, payload, expected_detail):
    with TestClient(app) as client:
        r = client.post("/generate", json=payload)
        assert r.status_code == 400
        assert r.json() == {"detail": expected_detail}


def test_generate_invalid_prompt_latent_shape_returns_400(app):
    with TestClient(app) as client:
        invalid_latents = np.arange(63, dtype=np.float32).tobytes()
        r = client.post(
            "/generate",
            json={
                "target_text": "hi",
                "prompt_latents_base64": base64.b64encode(invalid_latents).decode("utf-8"),
                "prompt_text": "p",
            },
        )
        assert r.status_code == 400
        assert "Invalid latent payload in prompt_latents_base64" in r.json()["detail"]


def test_generate_returns_500_if_cfg_missing(app):
    with TestClient(app) as client:
        delattr(client.app.state, "cfg")
        r = client.post("/generate", json={"target_text": "hi"})
        assert r.status_code == 500
        assert "missing app.state.cfg" in r.json()["detail"]


def test_generate_rejects_unknown_lora_name_400(app):
    with TestClient(app) as client:
        r = client.post("/generate", json={"target_text": "hi", "lora_name": "missing"})
        assert r.status_code == 400
        assert "not registered" in r.json()["detail"]


def test_generate_returns_500_if_channels_not_mono(app, monkeypatch):
    async def get_model_info_stereo():
        return {
            "sample_rate": 16000,
            "channels": 2,
            "feat_dim": 64,
            "patch_size": 2,
            "model_path": "/fake/model",
        }

    with TestClient(app) as client:
        monkeypatch.setattr(client.app.state.server, "get_model_info", get_model_info_stereo)
        r = client.post("/generate", json={"target_text": "hi"})
        assert r.status_code == 500
        assert "Only mono is supported" in r.json()["detail"]


def test_generate_records_ttfb_even_if_stream_is_empty(app, monkeypatch):
    # Force the streaming layer to yield nothing so the endpoint hits the
    # "not ttfb_recorded" fallback after the stream finishes.
    import app.api.routes.generate as generate_route

    async def empty_stream_mp3(*args, **kwargs):
        if False:  # pragma: no cover
            yield b""

    monkeypatch.setattr(generate_route, "stream_mp3", empty_stream_mp3)

    with TestClient(app) as client:
        with client.stream("POST", "/generate", json={"target_text": "hi"}) as resp:
            assert resp.status_code == 200
            data = resp.read()
            assert data == b""


def test_generate_wav_prompt_hits_encode_latents_branch(app, monkeypatch):
    import app.api.routes.generate as generate_route

    called = {"ok": False}

    async def record_encode_latents(wav: bytes, wav_format: str):
        called["ok"] = True
        assert wav == b"FAKEWAV"
        assert wav_format == "wav"
        return b"LATENTS"

    async def empty_stream_mp3(*args, **kwargs):
        # Don't actually encode MP3; we only want to cover the prompt branches.
        if False:  # pragma: no cover
            yield b""

    monkeypatch.setattr(generate_route, "stream_mp3", empty_stream_mp3)

    with TestClient(app) as client:
        monkeypatch.setattr(client.app.state.server, "encode_latents", record_encode_latents)
        wav_b64 = base64.b64encode(b"FAKEWAV").decode("utf-8")
        with client.stream(
            "POST",
            "/generate",
            json={
                "target_text": "hi",
                "prompt_wav_base64": wav_b64,
                "prompt_wav_format": "wav",
                "prompt_text": "p",
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.read() == b""

    assert called["ok"] is True


def test_generate_invalid_base64_ref_audio_wav_returns_400(app):
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={
                "target_text": "hi",
                "ref_audio_wav_base64": "a",
                "ref_audio_wav_format": "wav",
            },
        )
        assert r.status_code == 400
        assert "Invalid base64 in ref_audio_wav_base64" in r.json()["detail"]


def test_generate_latents_prompt_hits_latents_decode_branch(app, monkeypatch):
    import app.api.routes.generate as generate_route

    async def consume_then_empty_stream_mp3(*, wav_chunks, **kwargs):
        # Force the server.generate path to run at least once.
        async for _chunk in wav_chunks:
            break
        if False:  # pragma: no cover
            yield b""

    monkeypatch.setattr(generate_route, "stream_mp3", consume_then_empty_stream_mp3)

    with TestClient(app) as client:
        latents = np.arange(64, dtype=np.float32).tobytes()
        latents_b64 = base64.b64encode(latents).decode("utf-8")
        with client.stream(
            "POST",
            "/generate",
            json={
                "target_text": "hi",
                "prompt_latents_base64": latents_b64,
                "prompt_text": "p",
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.read() == b""


def test_generate_rejects_ref_audio_for_models_without_support(app, monkeypatch):
    import app.api.routes.generate as generate_route

    def unsupported_generate(
        *,
        target_text,
        prompt_latents=None,
        prompt_text="",
        max_generate_length=2000,
        temperature=1.0,
        cfg_value=1.5,
        lora_name=None,
    ):
        async def _stream():
            if False:  # pragma: no cover
                yield None

        return _stream()

    async def consume_stream_once(*, wav_chunks, **kwargs):
        async for _chunk in wav_chunks:
            break
        if False:  # pragma: no cover
            yield b""

    monkeypatch.setattr(generate_route, "stream_mp3", consume_stream_once)

    with TestClient(app) as client:
        monkeypatch.setattr(client.app.state.server, "generate", unsupported_generate)
        ref_latents = np.arange(64, dtype=np.float32).tobytes()
        r = client.post(
            "/generate",
            json={
                "target_text": "hi",
                "ref_audio_latents_base64": base64.b64encode(ref_latents).decode("utf-8"),
            },
        )
        assert r.status_code == 400
        assert "Reference audio is not supported" in r.json()["detail"]


def test_generate_converts_server_value_error_to_400(app, monkeypatch):
    import app.api.routes.generate as generate_route

    async def boom_generate(**kwargs):
        raise ValueError("bad input")
        yield  # pragma: no cover

    async def consume_stream_once(*, wav_chunks, **kwargs):
        async for _chunk in wav_chunks:
            break
        if False:  # pragma: no cover
            yield b""

    monkeypatch.setattr(generate_route, "stream_mp3", consume_stream_once)

    with TestClient(app) as client:
        monkeypatch.setattr(client.app.state.server, "generate", boom_generate)
        r = client.post("/generate", json={"target_text": "hi"})
        assert r.status_code == 400
        assert r.json() == {"detail": "bad input"}


def test_encode_latents_invalid_base64_returns_400(app):
    with TestClient(app) as client:
        r = client.post("/encode_latents", json={"wav_base64": "a", "wav_format": "wav"})
        assert r.status_code == 400
        assert "Invalid base64 in wav_base64" in r.json()["detail"]


def test_encode_latents_server_exception_returns_500(app, monkeypatch):
    async def boom_encode_latents(wav: bytes, wav_format: str):
        raise RuntimeError("boom")

    with TestClient(app) as client:
        monkeypatch.setattr(client.app.state.server, "encode_latents", boom_encode_latents)
        wav_b64 = base64.b64encode(b"FAKEWAV").decode("utf-8")
        r = client.post("/encode_latents", json={"wav_base64": wav_b64, "wav_format": "wav"})
        assert r.status_code == 500
        assert r.json() == {"detail": "boom"}


def test_encode_latents_propagates_http_exception(app, monkeypatch):
    from fastapi import HTTPException

    async def encode_latents_503(wav: bytes, wav_format: str):
        raise HTTPException(status_code=503, detail="nope")

    with TestClient(app) as client:
        monkeypatch.setattr(client.app.state.server, "encode_latents", encode_latents_503)
        wav_b64 = base64.b64encode(b"FAKEWAV").decode("utf-8")
        r = client.post("/encode_latents", json={"wav_base64": wav_b64, "wav_format": "wav"})
        assert r.status_code == 503
        assert r.json() == {"detail": "nope"}
