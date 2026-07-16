import torch.distributed as dist


def get_tp_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_tp_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()
