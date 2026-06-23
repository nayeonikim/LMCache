# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
import ctypes

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


class _RecordingRawDevice:
    def __init__(self) -> None:
        self.offsets: list[int] = []
        self.buffers: list[memoryview] = []
        self.lengths: list[int] = []
        self.read_buffers: list[memoryview] = []
        self.read_data = b""
        self.read_cursor = 0
        self.waited_batch_id: int | None = None

    def batched_write(
        self,
        offsets: list[int],
        buffers: list[memoryview],
        lengths: list[int],
    ) -> int:
        self.offsets = offsets
        self.buffers = buffers
        self.lengths = lengths
        return 17

    def wait_iouring(self, batch_id: int) -> None:
        self.waited_batch_id = batch_id

    def read_uring(
        self,
        offset: int,
        target: memoryview,
        payload_len: int,
        total_len: int,
    ) -> None:
        del offset, payload_len
        self.read_buffers.append(target)
        end = self.read_cursor + total_len
        target[:total_len] = self.read_data[self.read_cursor : end]
        self.read_cursor = end


def _buffer_address(buf: memoryview) -> int:
    return ctypes.addressof((ctypes.c_byte * 1).from_buffer(buf))


def test_raw_block_core_uring_cmd_write_padding_uses_aligned_chunks(monkeypatch):
    core = RawBlockCore.__new__(RawBlockCore)
    core.block_align = 4096
    core.max_data_transfer_size = 4096
    raw_dev = _RecordingRawDevice()
    monkeypatch.setattr(core, "_rawdev", lambda: raw_dev)

    payload = bytes([3]) * 5000

    core._write_uring_cmd_buffers(
        offsets=[4096],
        buffers=[bytearray(payload)],
        payload_lens=[len(payload)],
        total_lens=[8192],
    )

    assert raw_dev.offsets == [4096, 8192]
    assert raw_dev.lengths == [4096, 4096]
    assert raw_dev.waited_batch_id == 17
    assert all(_buffer_address(buf) % core.block_align == 0 for buf in raw_dev.buffers)
    assert b"".join(bytes(buf) for buf in raw_dev.buffers) == payload + bytes(3192)


def test_raw_block_core_uring_cmd_read_copyback_uses_aligned_chunks(monkeypatch):
    core = RawBlockCore.__new__(RawBlockCore)
    core.block_align = 4096
    core.max_data_transfer_size = 4096
    raw_dev = _RecordingRawDevice()
    monkeypatch.setattr(core, "_rawdev", lambda: raw_dev)

    payload = bytes([5]) * 5000
    raw_dev.read_data = payload + bytes(3192)
    dst = bytearray(len(payload))

    core._read_uring_cmd_buffers(
        offsets=[4096],
        buffers=[dst],
        payload_lens=[len(payload)],
        total_lens=[8192],
    )

    assert dst == payload
    assert all(
        _buffer_address(buf) % core.block_align == 0 for buf in raw_dev.read_buffers
    )


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
