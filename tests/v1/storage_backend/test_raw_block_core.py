# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import RawBlockCore, encode_object_key
from tests.v1.storage_backend.raw_block_test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
    memory_obj_bytes,
)

pytest.importorskip("lmcache_rust_raw_block_io")


def test_raw_block_core_store_load_and_exists(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        keys = [make_object_key(i) for i in range(3)]
        specs = [encode_object_key(key) for key in keys]
        payloads = [
            bytes([1]) * 1024,
            bytes([2]) * 2048,
            bytes([3]) * 3072,
        ]
        objects = [make_memory_obj(payload) for payload in payloads]

        put_result = core.put_many(specs, objects)

        assert put_result.results == [True, True, True]
        assert put_result.stored_keys == [spec.encoded for spec in specs]
        assert core.exists_many([spec.encoded for spec in specs]) == [
            True,
            True,
            True,
        ]

        loaded = [make_empty_memory_obj(len(payload)) for payload in payloads]
        load_result = core.load_many_into([spec.encoded for spec in specs], loaded)

        assert load_result == [True, True, True]
        assert [memory_obj_bytes(obj) for obj in loaded] == payloads
    finally:
        core.close()


def test_raw_block_core_duplicate_put_keeps_original_payload(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        spec = encode_object_key(make_object_key(11))
        original = b"original"
        duplicate = b"mutated!"

        first_result = core.put_many([spec], [make_memory_obj(original)])
        duplicate_result = core.put_many([spec], [make_memory_obj(duplicate)])

        assert first_result.results == [True]
        assert first_result.stored_keys == [spec.encoded]
        assert duplicate_result.results == [True]
        assert duplicate_result.stored_keys == []

        loaded = make_empty_memory_obj(len(original))
        assert core.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == original
    finally:
        core.close()


def test_raw_block_core_delete_and_missing_load(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        existing = encode_object_key(make_object_key(21))
        missing = encode_object_key(make_object_key(22))

        put_result = core.put_many([existing], [make_memory_obj(b"delete-me")])
        assert put_result.results == [True]
        assert core.contains_key(existing.encoded) is True

        assert core.delete_many([existing.encoded, missing.encoded]) == [True, False]
        assert core.exists_many([existing.encoded, missing.encoded]) == [False, False]

        loaded = make_empty_memory_obj(len(b"delete-me"))
        assert core.load_many_into([existing.encoded], [loaded]) == [False]
    finally:
        core.close()


def test_raw_block_core_recovers_checkpoint_from_temp_file(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    spec = encode_object_key(make_object_key(31))
    payload = b"recoverable-raw-block-payload"

    core = RawBlockCore(config, key_namespace="object")
    try:
        put_result = core.put_many([spec], [make_memory_obj(payload)])
        assert put_result.results == [True]
        core.checkpoint_now()
    finally:
        core.close()

    recovered = RawBlockCore(config, key_namespace="object")
    try:
        assert recovered.contains_key(spec.encoded) is True
        loaded = make_empty_memory_obj(len(payload))
        assert recovered.load_many_into([spec.encoded], [loaded]) == [True]
        assert memory_obj_bytes(loaded) == payload
    finally:
        recovered.close()


# ---------------------------------------------------------------------------
# _put_many_batch_io correctness tests
# These call _put_many_batch_io directly on a posix core to verify
# slot allocation / commit correctness independent of io_uring availability.
# ---------------------------------------------------------------------------


def test_put_many_batch_io_round_trip(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        keys = [make_object_key(i) for i in range(10)]
        specs = [encode_object_key(k) for k in keys]
        payloads = [bytes([i % 256]) * (512 * (i + 1)) for i in range(10)]
        objects = [make_memory_obj(p) for p in payloads]

        result = core._put_many_batch_io(specs, objects)

        assert result.results == [True] * 10
        assert result.stored_keys == [s.encoded for s in specs]

        loaded = [make_empty_memory_obj(len(p)) for p in payloads]
        assert core.load_many_into([s.encoded for s in specs], loaded) == [True] * 10
        assert [memory_obj_bytes(o) for o in loaded] == payloads
    finally:
        core.close()


def test_put_many_batch_io_skips_already_indexed(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        existing_keys = [make_object_key(i) for i in range(5)]
        existing_specs = [encode_object_key(k) for k in existing_keys]
        existing_payloads = [bytes([i]) * 512 for i in range(5)]
        existing_objs = [make_memory_obj(p) for p in existing_payloads]

        first = core.put_many(existing_specs, existing_objs)
        assert first.results == [True] * 5

        new_keys = [make_object_key(i) for i in range(5, 10)]
        new_specs = [encode_object_key(k) for k in new_keys]
        new_payloads = [bytes([i + 5]) * 512 for i in range(5)]
        new_objs = [make_memory_obj(p) for p in new_payloads]

        all_specs = existing_specs + new_specs
        all_objs = existing_objs + new_objs
        result = core._put_many_batch_io(all_specs, all_objs)

        assert result.results == [True] * 10
        assert len(result.stored_keys) == 5
        assert set(result.stored_keys) == {s.encoded for s in new_specs}
    finally:
        core.close()


def test_put_many_batch_io_write_failure_rolls_back(tmp_path):
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path)
    core = RawBlockCore(config, key_namespace="object")

    try:
        specs = [encode_object_key(make_object_key(i)) for i in range(5)]
        objects = [make_memory_obj(bytes([i]) * 512) for i in range(5)]

        free_slots_before = len(core._free_slots) + (
            core._max_slots - core._next_slot
        )

        with patch.object(core, "_write_buffers", side_effect=RuntimeError("disk error")):
            result = core._put_many_batch_io(specs, objects)

        assert result.results == [False] * 5
        assert result.stored_keys == []
        assert not core._inflight

        free_slots_after = len(core._free_slots) + (
            core._max_slots - core._next_slot
        )
        assert free_slots_after == free_slots_before
    finally:
        core.close()
