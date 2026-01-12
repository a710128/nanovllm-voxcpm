import os
from glob import glob
from pathlib import Path
from typing import Tuple, List
import torch
from torch import nn
from safetensors import safe_open

try:
    from safetensors.torch import load_file as safetensors_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    visited_param_names = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        visited_param_names.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    visited_param_names.add(weight_name)
    
    missing_param_names = []
    for name, _ in model.named_parameters():
        if name not in visited_param_names:
            # Skip LoRA parameters (they are optional)
            if "lora_" in name:
                continue
            missing_param_names.append(name)
    
    if missing_param_names:
        raise ValueError(f"Missing parameters: {missing_param_names}")


# ============================================================================
# LoRA Weight Loading
# ============================================================================

# Mapping from VoxCPM's separate projections to nanovllm's fused projections
LORA_NAME_MAPPING = {
    # QKV projections: q_proj/k_proj/v_proj -> qkv_proj
    "q_proj.lora_A": ("qkv_proj.lora_A", "q"),
    "q_proj.lora_B": ("qkv_proj.lora_B_q", None),
    "k_proj.lora_A": ("qkv_proj.lora_A", "k"),
    "k_proj.lora_B": ("qkv_proj.lora_B_k", None),
    "v_proj.lora_A": ("qkv_proj.lora_A", "v"),
    "v_proj.lora_B": ("qkv_proj.lora_B_v", None),
    # MLP projections: gate_proj/up_proj -> gate_up_proj
    "gate_proj.lora_A": ("gate_up_proj.lora_A", 0),
    "gate_proj.lora_B": ("gate_up_proj.lora_B_0", None),
    "up_proj.lora_A": ("gate_up_proj.lora_A", 1),
    "up_proj.lora_B": ("gate_up_proj.lora_B_1", None),
    # o_proj and down_proj remain the same
    "o_proj.lora_A": ("o_proj.lora_A", None),
    "o_proj.lora_B": ("o_proj.lora_B", None),
    "down_proj.lora_A": ("down_proj.lora_A", None),
    "down_proj.lora_B": ("down_proj.lora_B", None),
}


def _map_lora_weight_name(orig_name: str) -> Tuple[str, str]:
    """
    Map VoxCPM LoRA weight name to nanovllm format.
    
    Returns:
        Tuple of (new_name, shard_id) where shard_id is used for fused lora_A
    """
    for pattern, (replacement, shard_id) in LORA_NAME_MAPPING.items():
        if pattern in orig_name:
            new_name = orig_name.replace(pattern.split(".")[0], replacement.split(".")[0])
            if ".lora_A" in pattern:
                if shard_id is not None:
                    # Fused lora_A (q/k/v/gate/up): return shard_id for special loading
                    return new_name, shard_id
                else:
                    # Non-fused lora_A (o_proj/down_proj): no special handling needed
                    return new_name, None
            elif ".lora_B" in pattern:
                # For lora_B, use the full replacement name
                new_name = orig_name.replace(pattern, replacement)
                return new_name, None
    return orig_name, None


def load_lora_weights(
    model: nn.Module, 
    lora_path: str,
    device: str = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Load LoRA weights from VoxCPM checkpoint into nanovllm model.
    
    Handles the name mapping between VoxCPM's separate projections 
    (q_proj, k_proj, v_proj) and nanovllm's fused projections (qkv_proj).
    
    Args:
        model: The model to load weights into
        lora_path: Path to LoRA weights (directory or file)
        device: Device to load weights to
        
    Returns:
        Tuple of (loaded_keys, skipped_keys)
    """
    lora_path = Path(os.path.expanduser(lora_path))
    
    # Find the weights file
    if lora_path.is_dir():
        safetensors_file = lora_path / "lora_weights.safetensors"
        ckpt_file = lora_path / "lora_weights.ckpt"
    else:
        safetensors_file = lora_path if lora_path.suffix == ".safetensors" else None
        ckpt_file = lora_path if lora_path.suffix in [".ckpt", ".pth"] else None
    
    # Load state dict
    if safetensors_file and safetensors_file.exists() and SAFETENSORS_AVAILABLE:
        state_dict = safetensors_load_file(str(safetensors_file), device=device)
    elif ckpt_file and ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
    else:
        raise FileNotFoundError(
            f"LoRA checkpoint not found. Expected either {safetensors_file} or {ckpt_file}"
        )
    
    # Build parameter mapping
    model_params = dict(model.named_parameters())
    
    # Track which lora_A have been loaded (for fused loading)
    lora_A_loaded = {}  # key: param_name, value: {shard_id: tensor}
    
    loaded_keys = []
    skipped_keys = []
    
    for orig_name, tensor in state_dict.items():
        # Skip non-lora parameters
        if "lora_" not in orig_name:
            skipped_keys.append(orig_name)
            continue
        
        # Map the name
        new_name, shard_id = _map_lora_weight_name(orig_name)
        
        # Handle fused lora_A (need to accumulate all shards)
        if shard_id is not None and ".lora_A" in new_name:
            if new_name not in lora_A_loaded:
                lora_A_loaded[new_name] = {}
            lora_A_loaded[new_name][shard_id] = tensor
            loaded_keys.append(orig_name)
            continue
        
        # Try to find the parameter in model
        if new_name in model_params:
            param = model_params[new_name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader:
                weight_loader(param, tensor.to(device))
            else:
                param.data.copy_(tensor.to(device))
            loaded_keys.append(orig_name)
        else:
            skipped_keys.append(orig_name)
    
    # Load accumulated fused lora_A
    for param_name, shards in lora_A_loaded.items():
        if param_name not in model_params:
            continue
        
        param = model_params[param_name]
        module_name = ".".join(param_name.split(".")[:-1])
        
        # Find the module to get lora_targets order
        try:
            module = model.get_submodule(module_name)
            lora_targets = getattr(module, 'lora_targets', None)
            lora_r = getattr(module, 'lora_r', 0)
            
            if lora_targets and lora_r > 0:
                # Load each shard to the correct position
                for shard_id, tensor in shards.items():
                    if hasattr(module, 'load_lora_A'):
                        module.load_lora_A(tensor.to(device), shard_id)
        except Exception as e:
            print(f"Warning: Failed to load fused lora_A for {param_name}: {e}")
    
    return loaded_keys, skipped_keys
