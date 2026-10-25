"""ZM-ZEGRID-R52-OOC Lot A — bounded counting rejection sink (product helper).

The streaming executor's *dense* rejection diagnostic is an in-RAM
``(N, H, W, C)`` bool plane. That is fine for the default API, but a bounded
storage path must be able to collect the SAME scalar diagnostics (exact
rejection counts) without ever materialising that plane.

:class:`BoundedCountingRejectionSink` is the drop-in sink for that path:

* it accepts the executor's interior-block writes through the exact same
  ``sink[:, y0:y1, x0:x1, :] = block`` surface a dense ``ndarray`` would expose;
* it accumulates an exact ``true_count`` over bounded bool views of at most
  ``SINK_BLOCK_BYTES`` (2 MiB) using ``np.count_nonzero`` — no ``int64`` upcast,
  no full ``reshape``/copy of a large tile;
* it supports GPU-size write blocks LARGER than 4 MiB by splitting each written
  block into bounded views (the write itself is never copied wholesale);
* it validates bool dtype and the ``(N, H, W, C)`` shape, and fails loudly after
  ``close()``.

Scope / non-goals (Lot A): storage/counting primitives only. This sink is not
wired into any product dispatch path, and it is NOT a rejection *mask* — the
executor returns ``rejection_mask=None`` when a sink is supplied, because a
count-only sink cannot honestly reproduce the full mask.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BoundedCountingRejectionSink", "SINK_BLOCK_BYTES"]

#: Maximum bytes of any single ``np.count_nonzero`` view. Matches the R52
#: counting block budget (2 MiB) and keeps the counting working set bounded
#: independently of the written block size.
SINK_BLOCK_BYTES = 2 * 1024 * 1024


def _iter_bounded_blocks(value: np.ndarray, max_bytes: int):
    """Yield views of ``value`` whose ``nbytes`` is ``<= max_bytes``.

    Views only: no element is copied. The split walks leading axes so the
    yielded blocks stay as contiguous as the input allows; when a leading-axis
    chunk is still too large, the next axis is split. A final 1-D strided view is
    yielded as-is (bounded in bytes), and ``np.count_nonzero`` walks it without
    materialising a copy.
    """
    if value.size == 0:
        return
    if value.nbytes <= max_bytes:
        yield value
        return
    if value.ndim == 1:
        step = max(1, max_bytes // max(1, value.itemsize))
        for i in range(0, value.shape[0], step):
            yield value[i:i + step]
        return
    if value.shape[0] == 0:
        return
    if value.shape[0] == 1:
        # Singleton leading axis: dropping it strictly shrinks the block, so the
        # recursion terminates even when one slice alone exceeds the budget.
        yield from _iter_bounded_blocks(value[0], max_bytes)
        return
    per_leading = int(value[0].nbytes)
    step = max(1, max_bytes // max(1, per_leading))
    for i in range(0, value.shape[0], step):
        yield from _iter_bounded_blocks(value[i:i + step], max_bytes)


def _axis_length(item, dim: int):
    """Length selected on one axis, or ``None`` when an integer drops the axis.

    Only the ``slice``/integer forms the executor uses are accepted; anything
    else (ellipsis, fancy indexing, ...) raises, because a counting sink cannot
    honestly claim it knows which cells were addressed.
    """
    if isinstance(item, slice):
        start, stop, step = item.indices(dim)
        return len(range(start, stop, step))
    if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
        idx = int(item)
        if idx < 0:
            idx += dim
        if not (0 <= idx < dim):
            raise IndexError(f"index {item} is out of bounds for axis of size {dim}")
        return None
    raise TypeError(
        "BoundedCountingRejectionSink keys must be integers or slices "
        f"(got {type(item).__name__})"
    )


class BoundedCountingRejectionSink:
    """Exact bool-count sink with a bounded working set.

    Attributes
    ----------
    shape / dtype:
        Declared ``(N, H, W, C)`` domain, dtype ``bool`` (the executor validates
        both before writing).
    true_count:
        Exact number of ``True`` cells written so far.
    written_cells:
        Total number of cells written (for partition/duplicate checks).
    writes / flush_count / blocks:
        Write calls, flush calls, and bounded count blocks processed.
    max_write_bytes / max_block_bytes:
        Largest written block and largest ``np.count_nonzero`` view. The latter
        never exceeds ``block_bytes``; the former may exceed 4 MiB.
    storage_domain:
        Honest storage-model label (``counting-bool-bounded``).
    """

    storage_domain = "counting-bool-bounded"

    def __init__(self, shape, *, block_bytes: int = SINK_BLOCK_BYTES) -> None:
        if isinstance(shape, (str, bytes)) or not hasattr(shape, "__len__"):
            raise ValueError("rejection sink shape must be a 4-tuple (N, H, W, C)")
        dims = tuple(shape)
        if len(dims) != 4:
            raise ValueError(
                f"rejection sink shape must be 4-D (N, H, W, C), got {len(dims)}-D"
            )
        normalized = []
        for value in dims:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"rejection sink shape entries must be integers, got {value!r}")
            value = int(value)
            if value < 0:
                raise ValueError(f"rejection sink shape entries must be >= 0, got {value}")
            normalized.append(value)
        if int(block_bytes) <= 0:
            raise ValueError("block_bytes must be a positive integer")
        self.shape = tuple(normalized)
        self.dtype = np.dtype(bool)
        self.ndim = 4
        self.block_bytes = int(block_bytes)
        self.closed = False
        self.true_count = 0
        self.written_cells = 0
        self.writes = 0
        self.flush_count = 0
        self.blocks = 0
        self.max_write_bytes = 0
        self.max_block_bytes = 0

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Close the sink (idempotent); later writes/flushes fail loudly."""
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("BoundedCountingRejectionSink is closed")

    # -- counting -----------------------------------------------------------
    def _count(self, value: np.ndarray) -> int:
        total = 0
        for block in _iter_bounded_blocks(value, self.block_bytes):
            total += int(np.count_nonzero(block))
            self.blocks += 1
            self.max_block_bytes = max(self.max_block_bytes, int(block.nbytes))
            self.written_cells += int(block.size)
        return total

    # -- dense-ndarray write surface ---------------------------------------
    def __setitem__(self, key, value) -> None:
        """Count the bool block written at ``key`` (same surface as ndarray)."""
        self._check_open()
        if not isinstance(key, tuple) or len(key) != 4:
            raise TypeError(
                "BoundedCountingRejectionSink expects a 4-tuple key "
                "(N, H, W, C), got {!r}".format(key)
            )
        block = np.asarray(value)
        if block.dtype != np.dtype(bool):
            raise TypeError(
                f"BoundedCountingRejectionSink only counts bool blocks, got {block.dtype}"
            )
        expected = []
        for item, dim in zip(key, self.shape):
            length = _axis_length(item, dim)
            if length is not None:
                expected.append(length)
        if tuple(expected) != tuple(block.shape):
            raise ValueError(
                f"written block shape {tuple(block.shape)} does not match key "
                f"selection {tuple(expected)}"
            )
        self.true_count += self._count(block)
        self.writes += 1
        self.max_write_bytes = max(self.max_write_bytes, int(block.nbytes))

    def flush_batch(self) -> None:
        """Record one boundary flush (no eviction is implied or performed)."""
        self._check_open()
        self.flush_count += 1

    def flush(self) -> None:
        self.flush_batch()
