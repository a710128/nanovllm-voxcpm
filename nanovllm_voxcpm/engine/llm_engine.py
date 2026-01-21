"""nanovllm_voxcpm.engine.llm_engine

This module orchestrates the end-to-end inference runtime loop.

The core class :class:`LLMEngineBase` wires together three subsystems:

- Request state: :class:`~nanovllm_voxcpm.engine.sequence.Sequence`
  (prompt, generated tokens, KV bookkeeping).
- Scheduling + KV allocation: :class:`~nanovllm_voxcpm.engine.scheduler.Scheduler`
  (batching policy) and :class:`~nanovllm_voxcpm.engine.block_manager.BlockManager`
  (KV block pool + prefix caching).
- GPU execution: :class:`~nanovllm_voxcpm.engine.model_runner.BaseModelRunner`
  (build tensors, set attention context, run the model).

Lifecycle (one engine step)
---------------------------
1) :meth:`Scheduler.schedule` selects a list of sequences to run and whether the
   step is prefill or decode.
2) The model-specific engine converts each selected :class:`Sequence` into a
   :class:`~nanovllm_voxcpm.engine.model_runner.RunnerTask` via
   :meth:`preprocess_seq`.
3) :meth:`BaseModelRunner.call` executes :meth:`BaseModelRunner.run` on rank 0
   (and on other ranks if tensor parallelism is enabled).
4) Outputs are merged back into each :class:`Sequence` via
   :meth:`postprocess_seq` (typically appending a token and checking stop/EOS).
5) Sequences that set ``seq.stoped`` are finalized with :meth:`Scheduler.finish`
   (KV blocks are deallocated and the request is removed).

Multi-process / tensor parallel
-------------------------------
If ``tensor_parallel_size > 1``, this engine spawns additional worker processes
using the "spawn" start method. Rank 0 is created in-process; ranks 1..N-1 are
separate processes. Communication uses NCCL for model collectives (inside the
runner) plus a small shared-memory RPC mechanism to broadcast method calls.

Extending the engine
--------------------
:class:`LLMEngineBase` is intentionally model-agnostic. Model families implement
the interface by subclassing and providing:
- :meth:`preprocess_seq` to build model inputs from a :class:`Sequence`
- :meth:`postprocess_seq` to update the :class:`Sequence` from runner outputs

See the model packages under ``nanovllm_voxcpm/models/*`` for concrete
implementations.

Concrete example: VoxCPM
------------------------
``nanovllm_voxcpm/models/voxcpm/engine.py`` implements an audio generation loop
on top of this base engine:

- ``Sequence.token_ids`` is used as a hashable prefix stream (ints for text token
  ids, bytes for latent patches) so KV prefix caching can work across requests.
- ``Sequence.custom_payload`` carries the real model inputs:
  ``text_tokens`` (list[int]), ``feats`` (list[np.ndarray]), ``feat_masks`` and
  sampling controls (temperature / cfg_value), plus streaming buffers.
- ``postprocess_seq`` appends the predicted latent patch, updates a decode-pad
  window used by the VAE decoder, and sets ``seq.stoped`` when ``stop_flag`` is
  emitted or a max length is reached.
"""

import atexit
import torch.multiprocessing as mp

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.sequence import Sequence
from nanovllm_voxcpm.engine.scheduler import Scheduler
from nanovllm_voxcpm.engine.model_runner import RunnerTask, BaseModelRunner
import socket
import torch


def get_distributed_port():
    # find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class LLMEngineBase:
    model_runner: BaseModelRunner
    scheduler: Scheduler

    def __init__(
        self,
        RunnerType: type[BaseModelRunner],
        config: Config,
        tensor_parallel_size: int,
    ):

        self.distributed_port = get_distributed_port()

        if config.devices is None or len(config.devices) == 0:
            n_devices = torch.cuda.device_count()
            if tensor_parallel_size > n_devices:
                raise ValueError(
                    f"Tensor parallel size {tensor_parallel_size} is greater than the number of available devices {n_devices}"
                )
            config.devices = list(range(tensor_parallel_size))

        if len(config.devices) != tensor_parallel_size:
            raise ValueError(
                f"Number of devices {len(config.devices)} is not equal to tensor parallel size {tensor_parallel_size}"
            )

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=RunnerType,
                args=(config, i, config.devices[i], self.distributed_port, event),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = RunnerType(config, 0, config.devices[0], self.distributed_port, self.events)
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_sequence(self, seq: Sequence):
        self.scheduler.add(seq)

    def cancel_sequence(self, seq_id: str):
        self.scheduler.cancel(seq_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        runner_tasks = [self.preprocess_seq(seq, is_prefill) for seq in seqs]
        outputs = self.model_runner.call("run", runner_tasks, is_prefill)

        for seq, output in zip(seqs, outputs):
            self.postprocess_seq(seq, output, is_prefill)

        for seq in seqs:
            if seq.stoped:
                self.scheduler.finish(seq)

        return seqs

    def is_finished(self):
        return self.scheduler.is_finished()

    def preprocess_seq(self, seq: Sequence, is_prefill: bool) -> RunnerTask:
        """Convert a scheduled :class:`~nanovllm_voxcpm.engine.sequence.Sequence` into a runner input.

        The engine loop is model-agnostic, so each model family must implement
        how to build a :class:`~nanovllm_voxcpm.engine.model_runner.RunnerTask`
        (plus any model-specific payload) from a scheduled ``Sequence``.

        What you typically need to do
        -----------------------------
        - **Pass through KV mapping**:
          The returned :class:`RunnerTask` must carry ``seq.block_table`` and
          ``seq.block_size`` so the runner can map logical token positions to
          physical KV-cache slots.
        - **Set lengths correctly**:
          ``RunnerTask.seq_length`` should reflect the logical sequence length
          the model should consider for this step.
        - **Set cached-prefix semantics**:
          ``RunnerTask.num_cached_tokens`` tells the runner how many tokens are
          already present in KV (for prefill) or equivalently where the "query"
          begins (for decode). The runner uses this to build attention context:
          prefill may have ``Q < K`` when prefix caching is active; decode is
          typically ``Q = 1``.
        - **Build model-specific tensors**:
          Convert ``seq.custom_payload`` (Python/numpy objects) into a payload
          structure that your runner knows how to turn into GPU tensors.
          The base engine never interprets this payload.

        Concrete example: VoxCPM
        ------------------------
        See ``nanovllm_voxcpm/models/voxcpm/engine.py``:

        - Prefill (``is_prefill=True``):
          ``RunnerTask.num_cached_tokens = seq.num_cached_tokens`` and the engine
          slices prompt inputs to *only* send the uncached tail:
          ``text_tokens[seq.num_cached_tokens:]``, ``feats[seq.num_cached_tokens:]``,
          ``feat_masks[seq.num_cached_tokens:]``.
          This avoids recomputing KV for the cached prefix blocks.
        - Decode (``is_prefill=False``):
          VoxCPM sends only the last step (length 1) as inputs, and sets
          ``RunnerTask.num_cached_tokens = len(seq) - 1`` so the runner builds a
          decode context (keys over full context, query over the last token).

        Returns
        -------
        A :class:`RunnerTask` whose ``custom_payload`` is specific to the model
        runner implementation.
        """
        raise NotImplementedError()

    def postprocess_seq(self, seq: Sequence, outputs: dict, is_prefill: bool):
        """Merge runner outputs back into the :class:`~nanovllm_voxcpm.engine.sequence.Sequence`.

        This method is the model-specific "state update" step. The base engine
        calls it once per scheduled sequence after ``BaseModelRunner.run``
        returns.

        What you typically need to do
        -----------------------------
        - **Append new token(s)**:
          Update ``seq.token_ids`` via :meth:`Sequence.append_token` so the
          scheduler/block manager sees the sequence length grow. This is crucial
          for correct KV block allocation on the next decode step.
        - **Update model-side payload state**:
          Extend/refresh fields inside ``seq.custom_payload`` (e.g. decoded text,
          generated audio chunks, streaming buffers, sampling state, etc.). Keep
          any parallel arrays in sync with the logical sequence length.
        - **Stop condition**:
          Set ``seq.stoped = True`` when the request should finish (EOS token,
          stop_flag, max tokens, user-defined stopping criteria, etc.).
          The base engine will call ``scheduler.finish(seq)`` in the same step.

        What you should NOT do
        ----------------------
        - Do not allocate/free KV blocks here. KV lifecycle is owned by
          :class:`~nanovllm_voxcpm.engine.scheduler.Scheduler` /
          :class:`~nanovllm_voxcpm.engine.block_manager.BlockManager`.

        Concrete example: VoxCPM
        ------------------------
        In ``nanovllm_voxcpm/models/voxcpm/engine.py``:

        - The runner returns ``latents`` (one patch), ``stop_flag``, and a decoded
          waveform chunk ``waveforms``.
        - The engine appends ``latents.tobytes()`` into ``seq.token_ids`` so prefix
          caching and KV block accounting see the new token.
        - It appends the same latent patch into ``custom_payload.feats`` and
          updates ``text_tokens``/``feat_masks`` so the next step has consistent
          aligned inputs.
        - It updates a small ``decode_pad`` window used by the AudioVAE decoder.
        - It sets ``seq.stoped`` when ``stop_flag == 1`` or when
          ``generated_waveforms`` reaches ``max_generate_length``.
        """
        raise NotImplementedError()
