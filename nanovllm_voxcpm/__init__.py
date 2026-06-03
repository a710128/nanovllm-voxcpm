from nanovllm_voxcpm.llm import VoxCPM

try:
    from nanovllm_voxcpm._version import version as __version__
except Exception:
    try:        
        import importlib.metadata
        __version__ = importlib.metadata.version("nano-vllm-voxcpm")
    except Exception:        
        __version__ = "0.0.0"

__all__ = [
    "VoxCPM",
    "__version__",
]
