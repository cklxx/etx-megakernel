"""CPU-only KV allocation, vLLM layout, and randomized ownership checks."""
import random

import pytest

from etx.serve.kv import KVBlockManager, OutOfBlocksError, RequestUsage


def test_layout_extend_and_reuse():
    manager = KVBlockManager(4, 4)
    assert manager.allocate("a", 3) == (0,)
    assert manager.allocate("b", 4) == (1,)
    assert manager.extend("a", 3) == (0, 2)
    assert manager.slot_mapping("a") == (0, 1, 2, 3, 8, 9)
    assert manager.slot_mapping("a", 3) == (3, 8, 9)
    assert manager.usage("a") == RequestUsage(6, 2, 8)
    assert manager.usage("a").unused_slots == 2
    snapshot = manager.block_table("a")
    manager.free("a")
    assert snapshot == (0, 2)
    assert manager.allocate("c", 12) == (3, 0, 2)
    assert manager.slot_mapping("c", 3, 6) == (15, 0, 1)
    manager.free("b")
    manager.free("c")
    assert manager.num_free_blocks == 4
    assert manager.num_requests == manager.num_used_blocks == 0


def test_capacity_failure_is_atomic():
    manager = KVBlockManager(2, 4)
    manager.allocate("a", 3)
    with pytest.raises(OutOfBlocksError):
        manager.allocate("b", 9)
    with pytest.raises(KeyError):
        manager.usage("b")
    with pytest.raises(OutOfBlocksError):
        manager.extend("a", 6)
    assert manager.usage("a") == RequestUsage(3, 1, 4)
    assert manager.num_free_blocks == 1
    assert manager.extend("a", 1) == (0,)
    assert manager.extend("a") == (0, 1)


def test_zero_tokens_and_empty_pool():
    manager = KVBlockManager(0, 16)
    assert manager.allocate("a", 0) == ()
    assert manager.extend("a", 0) == ()
    assert manager.slot_mapping("a") == ()
    assert manager.usage("a") == RequestUsage(0, 0, 0)
    with pytest.raises(OutOfBlocksError):
        manager.extend("a")
    manager.free("a")
    with pytest.raises(KeyError):
        manager.free("a")


@pytest.mark.parametrize("value,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_counts(value, error):
    with pytest.raises(error):
        KVBlockManager(value, 4)
    with pytest.raises(error):
        KVBlockManager(4, value)
    manager = KVBlockManager(4, 4)
    manager.allocate("a", 1)
    for operation in (lambda: manager.allocate("b", value),
                      lambda: manager.extend("a", value),
                      lambda: manager.slot_mapping("a", value),
                      lambda: manager.slot_mapping("a", end=value)):
        with pytest.raises(error):
            operation()
    assert manager.num_free_blocks == 3
    assert manager.usage("a").num_tokens == 1


def test_invalid_requests_and_ranges():
    with pytest.raises(ValueError):
        KVBlockManager(2, 0)
    manager = KVBlockManager(2, 4)
    manager.allocate("a", 2)
    with pytest.raises(ValueError):
        manager.allocate("a", 0)
    for operation in (manager.extend, manager.free, manager.usage,
                      manager.block_table, manager.slot_mapping):
        with pytest.raises(KeyError):
            operation("missing")
    for start, end in [(3, None), (2, 1), (0, 3)]:
        with pytest.raises(ValueError):
            manager.slot_mapping("a", start, end)
    assert manager.slot_mapping("a", 2, 2) == ()


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_random_10000_operations(block_size):
    """State-machine property test with an independent token-count model."""
    rng = random.Random(3000 + block_size)
    manager = KVBlockManager(31, block_size)
    model = {}
    next_id = 0
    for _ in range(10_000):
        action = rng.choice(("admit", "extend", "free")) if model else "admit"
        before = {key: (manager.usage(key), manager.block_table(key)) for key in model}
        free_before = manager.num_free_blocks
        if action == "free":
            key = rng.choice(list(model))
            manager.free(key)
            del model[key]
        else:
            key = next_id if action == "admit" else rng.choice(list(model))
            amount = rng.randrange(4 * block_size + 1)
            old = model.get(key, 0)
            blocks_before = (old + block_size - 1) // block_size
            needed = (old + amount + block_size - 1) // block_size - blocks_before
            operation = manager.allocate if action == "admit" else manager.extend
            if needed > free_before:
                with pytest.raises(OutOfBlocksError):
                    operation(key, amount)
                assert manager.num_free_blocks == free_before
                assert {k: (manager.usage(k), manager.block_table(k)) for k in model} == before
            else:
                operation(key, amount)
                model[key] = old + amount
            if action == "admit":
                next_id += 1
        owned = []
        for key, tokens in model.items():
            table = manager.block_table(key)
            count = (tokens + block_size - 1) // block_size
            assert manager.usage(key) == RequestUsage(tokens, count, count * block_size)
            assert len(table) == count
            assert all(0 <= block < 31 for block in table)
            slots = manager.slot_mapping(key)
            expected = tuple(slot for block in table
                             for slot in range(block * block_size, (block + 1) * block_size))[:tokens]
            assert slots == expected
            owned.extend(table)
        assert len(owned) == len(set(owned))  # no double ownership
        assert len(owned) + manager.num_free_blocks == 31  # no leak
        assert manager.num_used_blocks == len(owned)
        assert manager.num_requests == len(model)
    for key in model:
        manager.free(key)
    assert manager.num_free_blocks == 31
    # Exhaust the recovered pool: detects duplicate free-list entries as well.
    table = manager.allocate("all", 31 * block_size)
    assert set(table) == set(range(31))
    assert len(table) == 31
