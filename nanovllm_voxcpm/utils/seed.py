_UINT64_MASK = (1 << 64) - 1
_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MUL2 = 0x94D049BB133111EB


def derive_step_seed(seed: int, seed_step: int) -> int:
    """Derive a deterministic per-step seed from a request seed."""

    value = (int(seed) & _UINT64_MASK) ^ ((int(seed_step) + _SPLITMIX64_INCREMENT) & _UINT64_MASK)
    value = (value + _SPLITMIX64_INCREMENT) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * _SPLITMIX64_MUL1) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * _SPLITMIX64_MUL2) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK
