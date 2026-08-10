_CANONICAL_BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512)


def make_batch_size_buckets(max_batch_size: int | None) -> tuple[int, ...]:
    """Build the shared CUDA Graph and streaming VAE batch-size buckets."""
    if max_batch_size is None:
        return ()
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")

    buckets = [size for size in _CANONICAL_BATCH_SIZE_BUCKETS if size < max_batch_size]
    buckets.append(max_batch_size)
    return tuple(buckets)


__all__ = ["make_batch_size_buckets"]
