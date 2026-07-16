"""nanovllm_voxcpm.models.voxcpm.runner_utils

Pure CPU helpers extracted from VoxCPMRunner so they can be unit-tested without
a CUDA device.  runner.py delegates to these functions; tests import them directly.
"""

from __future__ import annotations

import numpy as np


def collect_seeded_rows(
    seqs: list,
    *,
    payload_attr: str = "custom_payload",
) -> list[tuple[int, int, int]]:
    """Return ``(index, seed_val, seed_step)`` for every sequence with a valid seed.

    A seed is considered valid when it is not ``None`` and ``>= 0``.

    Args:
        seqs: Sequence of :class:`~nanovllm_voxcpm.engine.model_runner.RunnerTask`
            objects (or any object whose *payload_attr* has ``seed`` and
            ``seed_step`` attributes).
        payload_attr: Attribute name on each item that holds the payload.

    Returns:
        List of ``(i, seed_val, seed_step)`` tuples in the order the rows
        appear in *seqs*.
    """
    result = []
    for i, seq in enumerate(seqs):
        payload = getattr(seq, payload_attr)
        seed = payload.seed
        if seed is not None and seed >= 0:
            result.append((i, int(seed), payload.seed_step))
    return result


def compute_pad_lengths(seqs: list, *, payload_attr: str = "custom_payload") -> list[int]:
    """Return the padding-decode length for each sequence.

    For sequences whose payload has no ``padding_decode`` (i.e. it is ``None``),
    the length is 0.

    Args:
        seqs: List of runner tasks.
        payload_attr: Attribute name on each task holding the payload.

    Returns:
        List of integer pad lengths, one per sequence.
    """
    result = []
    for seq in seqs:
        payload = getattr(seq, payload_attr)
        pd = payload.padding_decode
        result.append(pd.shape[0] if pd is not None else 0)
    return result


def assemble_batch_inputs(
    seqs: list,
    *,
    payload_attr: str = "custom_payload",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    """Concatenate per-sequence payload arrays into flat batch arrays.

    All shape validation (same first dimension across ``text_tokens``,
    ``feats``, ``feat_masks``) is delegated to the caller (runner) via
    ``assert`` statements; this helper just concatenates.

    Args:
        seqs: List of runner tasks with VoxCPMPayload as ``custom_payload``.
        payload_attr: Attribute name on each task holding the payload.

    Returns:
        Tuple of ``(text_tokens, feats, feat_masks, temperatures, cfg_values)``
        where the numpy arrays are concatenated along axis 0 and the scalar
        lists have one entry per sequence.
    """
    text_tokens_list: list[np.ndarray] = []
    feats_list: list[np.ndarray] = []
    feat_masks_list: list[np.ndarray] = []
    temperatures: list[float] = []
    cfg_values: list[float] = []

    for seq in seqs:
        payload = getattr(seq, payload_attr)
        text_tokens_list.append(payload.text_tokens)
        feats_list.append(payload.feats)
        feat_masks_list.append(payload.feat_masks)
        temperatures.append(payload.temperature)
        cfg_values.append(payload.cfg_value)

    return (
        np.concatenate(text_tokens_list, axis=0),
        np.concatenate(feats_list, axis=0),
        np.concatenate(feat_masks_list, axis=0),
        temperatures,
        cfg_values,
    )


def slice_waveforms(
    vae_outputs: np.ndarray,
    pad_lengths: list[int],
    patch_size: int,
    chunk_size: int,
) -> list[np.ndarray]:
    """Slice per-sequence waveform chunks from VAE decoder output.

    The VAE decoder processes a padded input of shape
    ``(bsz, max_pad_decode, feat_dim)`` and returns a waveform array of shape
    ``(bsz, total_samples)``.  This helper extracts the waveform slice that
    corresponds to the *current* step's latent patch (excluding any historical
    padding).

    Args:
        vae_outputs: ``(bsz, total_samples)`` numpy array from the VAE decoder.
        pad_lengths: Number of historical latent frames used as padding for each
            sequence (0 when no padding is applied).
        patch_size: Number of latent frames produced per decode step.
        chunk_size: Number of waveform samples per latent frame (VAE property).

    Returns:
        List of 1-D numpy arrays, one per sequence.
    """
    ret = []
    for i, pad_len in enumerate(pad_lengths):
        start = pad_len * chunk_size
        end = (pad_len + patch_size) * chunk_size
        ret.append(vae_outputs[i, start:end])
    return ret


def assemble_run_outputs(
    np_latents: np.ndarray,
    stop_flags: list[int],
    waveforms: list[np.ndarray],
) -> list[dict]:
    """Combine per-sequence latents, stop flags and waveforms into output dicts.

    Args:
        np_latents: ``(bsz, patch_size, feat_dim)`` float32 numpy array.
        stop_flags: Per-sequence stop flag integers (0 or 1).
        waveforms: Per-sequence waveform numpy arrays.

    Returns:
        List of dicts, each with keys ``"latents"``, ``"stop_flag"``,
        ``"waveforms"``.
    """
    return [
        {"latents": np_latents[i], "stop_flag": stop_flags[i], "waveforms": waveforms[i]}
        for i in range(len(stop_flags))
    ]
