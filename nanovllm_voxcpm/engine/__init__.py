"""nanovllm_voxcpm.engine

Model-agnostic inference runtime core.

This package contains the components that make inference work end-to-end:

- :mod:`nanovllm_voxcpm.engine.sequence`: per-request state machine + KV mapping.
- :mod:`nanovllm_voxcpm.engine.block_manager`: KV-cache block pool + prefix cache.
- :mod:`nanovllm_voxcpm.engine.scheduler`: batching policy + preemption.
- :mod:`nanovllm_voxcpm.engine.model_runner`: GPU execution abstraction.
- :mod:`nanovllm_voxcpm.engine.llm_engine`: orchestrates the engine step loop.

The intent is that model implementations only need to provide a thin adapter
layer ("preprocess" and "postprocess") while the runtime handles scheduling,
memory management, and execution.

Reference implementation
------------------------
For a complete, working example of how to plug a model family into this runtime,
see ``nanovllm_voxcpm/models/voxcpm``.
"""
