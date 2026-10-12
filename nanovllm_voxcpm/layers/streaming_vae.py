from collections.abc import Hashable, Sequence

import torch
from torch import nn

from nanovllm_voxcpm.utils.batch_buckets import make_batch_size_buckets


class BatchedStreamingVAEDecoder:
    """Stateful causal VAE decoder for dynamically batched request streams."""

    def __init__(
        self,
        vae: nn.Module,
        causal_conv_type: type[nn.Conv1d],
        causal_transpose_conv_type: type[nn.ConvTranspose1d],
        max_batch_size: int | None = None,
    ):
        self._vae = vae
        self._causal_conv_type = causal_conv_type
        self._causal_transpose_conv_type = causal_transpose_conv_type
        self._batch_size_buckets = make_batch_size_buckets(max_batch_size)
        self._states: dict[int, dict[Hashable, torch.Tensor]] = {}
        self._stream_ids: Sequence[Hashable] | None = None
        self._initialized_streams: set[Hashable] = set()
        self._install()

    @torch.inference_mode()
    def decode_chunks(
        self,
        z_chunks: torch.Tensor,
        stream_ids: Sequence[Hashable],
        initial_contexts: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Decode one new latent chunk for each stream in a dynamic batch."""
        if z_chunks.ndim != 3:
            raise ValueError(f"z_chunks must have shape [B, D, T], got {tuple(z_chunks.shape)}")
        if len(stream_ids) != z_chunks.size(0):
            raise ValueError("stream_ids length must match z_chunks batch size")
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream_ids must be unique within a decode batch")
        if initial_contexts is None:
            initial_contexts = [None] * len(stream_ids)
        if len(initial_contexts) != len(stream_ids):
            raise ValueError("initial_contexts length must match stream_ids length")

        missing = object()
        state_snapshot = {
            layer_id: {stream_id: layer_states.get(stream_id, missing) for stream_id in stream_ids}
            for layer_id, layer_states in self._states.items()
        }
        initialized_snapshot = {stream_id for stream_id in stream_ids if stream_id in self._initialized_streams}
        try:
            for stream_id, context in zip(stream_ids, initial_contexts):
                if stream_id in self._initialized_streams:
                    continue
                if context is not None and context.numel() > 0:
                    if context.ndim != 3 or context.size(0) != 1:
                        raise ValueError("each initial context must have shape [1, D, T]")
                    self._decode(context, [stream_id])
                self._initialized_streams.add(stream_id)

            return self._decode_bucketed(z_chunks, stream_ids)
        except Exception:
            for layer_id, layer_states in self._states.items():
                for stream_id, state in state_snapshot[layer_id].items():
                    if state is missing:
                        layer_states.pop(stream_id, None)
                    else:
                        layer_states[stream_id] = state
            self._initialized_streams.difference_update(stream_ids)
            self._initialized_streams.update(initialized_snapshot)
            raise

    @torch.inference_mode()
    def warmup(self, latent_channels: int, chunk_size: int) -> None:
        """Initialize decoder kernels for every configured batch-size bucket."""
        if not self._batch_size_buckets:
            return
        if self._initialized_streams or any(self._states.values()):
            raise RuntimeError("streaming VAE warmup requires empty stream state")

        parameter = next(self._vae.decoder.parameters(), None)
        if parameter is None:
            raise RuntimeError("streaming VAE decoder has no parameters")
        for batch_size in self._batch_size_buckets:
            z_chunks = torch.zeros(
                batch_size,
                latent_channels,
                chunk_size,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            stream_ids = [object() for _ in range(batch_size)]
            self.decode_chunks(z_chunks, stream_ids)
            self.clear()
        if parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)

    def release(self, stream_id: Hashable) -> None:
        """Release all cached convolution state for a completed request."""
        self._initialized_streams.discard(stream_id)
        for layer_states in self._states.values():
            layer_states.pop(stream_id, None)

    def clear(self) -> None:
        self._initialized_streams.clear()
        for layer_states in self._states.values():
            layer_states.clear()

    def _decode(self, z_chunks: torch.Tensor, stream_ids: Sequence[Hashable]) -> torch.Tensor:
        if self._stream_ids is not None:
            raise RuntimeError("Streaming VAE decode is not reentrant")
        self._stream_ids = stream_ids
        try:
            return self._vae.decode(z_chunks)
        finally:
            self._stream_ids = None

    def _decode_bucketed(self, z_chunks: torch.Tensor, stream_ids: Sequence[Hashable]) -> torch.Tensor:
        batch_size = z_chunks.size(0)
        padded_batch_size = next(
            (bucket for bucket in self._batch_size_buckets if bucket >= batch_size),
            batch_size,
        )
        if padded_batch_size == batch_size:
            return self._decode(z_chunks, stream_ids)

        num_padding = padded_batch_size - batch_size
        padding = z_chunks.new_zeros((num_padding, *z_chunks.shape[1:]))
        dummy_stream_ids = [object() for _ in range(num_padding)]
        try:
            output = self._decode(
                torch.cat([z_chunks, padding], dim=0),
                [*stream_ids, *dummy_stream_ids],
            )
            return output[:batch_size]
        finally:
            for stream_id in dummy_stream_ids:
                self.release(stream_id)

    def _install(self) -> None:
        for module in self._vae.decoder.modules():
            if isinstance(module, self._causal_conv_type):
                padding = module._CausalConv1d__padding * 2 - getattr(
                    module,
                    "_CausalConv1d__output_padding",
                    0,
                )
                if padding > 0:
                    self._patch_causal_conv(module, padding)
            elif isinstance(module, self._causal_transpose_conv_type):
                trim = module._CausalTransposeConv1d__padding * 2 - module._CausalTransposeConv1d__output_padding
                context = module.kernel_size[0] // module.stride[0] - 1
                if context > 0:
                    self._patch_transpose_conv(module, context, trim)

    def _current_stream_ids(self, batch_size: int) -> Sequence[Hashable]:
        if self._stream_ids is None or len(self._stream_ids) != batch_size:
            raise RuntimeError("Causal VAE layer executed outside decode_chunks")
        return self._stream_ids

    def _patch_causal_conv(self, module: nn.Conv1d, padding: int) -> None:
        layer_states = self._states.setdefault(id(module), {})

        def forward(x: torch.Tensor, *, _module=module, _padding=padding, _states=layer_states):
            if self._stream_ids is None:
                return type(_module).forward(_module, x)
            stream_ids = self._current_stream_ids(x.size(0))
            contexts = []
            for index, stream_id in enumerate(stream_ids):
                state = _states.get(stream_id)
                if state is None:
                    state = torch.zeros(
                        x.size(1),
                        _padding,
                        device=x.device,
                        dtype=x.dtype,
                    )
                contexts.append(state)
                combined = torch.cat([state, x[index]], dim=-1)
                _states[stream_id] = combined[:, -_padding:].detach()
            padded = torch.cat([torch.stack(contexts, dim=0), x], dim=-1)
            return nn.Conv1d.forward(_module, padded)

        module.forward = forward

    def _patch_transpose_conv(self, module: nn.ConvTranspose1d, context: int, trim: int) -> None:
        layer_states = self._states.setdefault(id(module), {})

        def forward(x: torch.Tensor, *, _module=module, _context=context, _trim=trim, _states=layer_states):
            if self._stream_ids is None:
                return type(_module).forward(_module, x)
            stream_ids = self._current_stream_ids(x.size(0))
            contexts = []
            for index, stream_id in enumerate(stream_ids):
                state = _states.get(stream_id)
                if state is None:
                    state = torch.zeros(
                        x.size(1),
                        _context,
                        device=x.device,
                        dtype=x.dtype,
                    )
                contexts.append(state)
                combined = torch.cat([state, x[index]], dim=-1)
                _states[stream_id] = combined[:, -_context:].detach()
            full_input = torch.cat([torch.stack(contexts, dim=0), x], dim=-1)
            output = nn.ConvTranspose1d.forward(_module, full_input)
            left = _context * _module.stride[0]
            return output[..., left:-_trim] if _trim > 0 else output[..., left:]

        module.forward = forward


__all__ = ["BatchedStreamingVAEDecoder"]
