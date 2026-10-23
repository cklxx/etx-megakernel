"""CPU bookkeeping for a fixed-size paged KV cache.

Block tables contain physical block IDs in logical token order. A token at
position p maps to table[p // block_size] * block_size + p % block_size,
matching vLLM's paged attention / reshape-and-cache convention. Consumers may
convert tables to int32 tensors and slot mappings to int64 tensors themselves;
this module has no tensor or GPU dependency. Block zero is usable (there is no
reserved null block). Tables have no padding; batch consumers must pad as needed.

Requests exclusively own their blocks: no prefix sharing, eviction, sliding
window, or copy-on-write is performed. This manager is single-threaded.
"""
from collections import deque
from dataclasses import dataclass
from typing import Hashable


class OutOfBlocksError(RuntimeError):
    """The pool cannot satisfy an allocation; existing state is unchanged."""


def _nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class RequestUsage:
    """Immutable accounting snapshot, including unused slots in the final block."""

    num_tokens: int
    num_blocks: int
    capacity: int

    @property
    def unused_slots(self) -> int:
        return self.capacity - self.num_tokens


class KVBlockManager:
    """Manage ``num_blocks`` physical blocks of ``block_size`` token slots.

    ``allocate(id, num_tokens)`` admits a new request, and
    ``extend(id, num_tokens)`` appends that many tokens (not a total length).
    Both return the complete immutable block table. Capacity failures are atomic.
    Unknown IDs raise KeyError, including double free; duplicate admission raises
    ValueError. Zero-token requests are valid and own no blocks.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self._num_blocks = _nonnegative(num_blocks, "num_blocks")
        self._block_size = _nonnegative(block_size, "block_size")
        if not block_size:
            raise ValueError("block_size must be positive")
        self._free = deque(range(num_blocks))
        self._requests: dict[Hashable, tuple[int, tuple[int, ...]]] = {}

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def num_requests(self) -> int:
        return len(self._requests)

    def _take(self, count: int) -> tuple[int, ...]:
        if count > len(self._free):
            raise OutOfBlocksError(f"need {count} blocks, only {len(self._free)} free")
        return tuple(self._free.popleft() for _ in range(count))

    def allocate(self, request_id: Hashable, num_tokens: int) -> tuple[int, ...]:
        num_tokens = _nonnegative(num_tokens, "num_tokens")
        if request_id in self._requests:
            raise ValueError(f"request already allocated: {request_id!r}")
        blocks = self._take((num_tokens + self.block_size - 1) // self.block_size)
        self._requests[request_id] = (num_tokens, blocks)
        return blocks

    def extend(self, request_id: Hashable, num_tokens: int = 1) -> tuple[int, ...]:
        num_tokens = _nonnegative(num_tokens, "num_tokens")
        old_tokens, blocks = self._requests[request_id]
        total = old_tokens + num_tokens
        needed = (total + self.block_size - 1) // self.block_size - len(blocks)
        blocks = blocks + self._take(needed)
        self._requests[request_id] = (total, blocks)
        return blocks

    def free(self, request_id: Hashable) -> None:
        _, blocks = self._requests.pop(request_id)
        self._free.extend(blocks)

    def block_table(self, request_id: Hashable) -> tuple[int, ...]:
        return self._requests[request_id][1]

    def usage(self, request_id: Hashable) -> RequestUsage:
        tokens, blocks = self._requests[request_id]
        return RequestUsage(tokens, len(blocks), len(blocks) * self.block_size)

    def slot_mapping(
        self, request_id: Hashable, start: int = 0, end: int | None = None
    ) -> tuple[int, ...]:
        """Physical slots for the half-open token range [start, end).

        The default covers all admitted tokens. To cache newly appended tokens,
        pass the previous token count as start. Unallocated positions are errors.
        """
        tokens, blocks = self._requests[request_id]
        start = _nonnegative(start, "start")
        end = tokens if end is None else _nonnegative(end, "end")
        if not start <= end <= tokens:
            raise ValueError("slot range must satisfy 0 <= start <= end <= num_tokens")
        return tuple(
            blocks[p // self.block_size] * self.block_size + p % self.block_size
            for p in range(start, end)
        )
