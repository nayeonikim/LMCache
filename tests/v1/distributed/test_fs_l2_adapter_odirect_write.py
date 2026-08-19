# SPDX-License-Identifier: Apache-2.0
"""Tests that the FS adapter's O_DIRECT store path writes every byte.

``os.write`` is allowed to accept fewer bytes than it was handed. The
adapter's read path already loops for exactly this reason (see
``_readinto_full``); the store path must not stop at a short write, or
the object is silently persisted truncated.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
from typing import cast
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import MemoryObj


class _Buf:
    """Minimal MemoryObj stand-in exposing the ``byte_array`` the
    adapter's store path reads."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._data)


def _key(h: bytes = b"\xde\xad\xbe\xef") -> ObjectKey:
    return ObjectKey(
        chunk_hash=h,
        model_name="llama",
        kv_rank=7,
        cache_salt="alice",
    )


@pytest.fixture
def odirect_adapter(tmp_path: Path) -> Iterator[FSL2Adapter]:
    adp = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path), use_odirect=True))
    try:
        yield adp
    finally:
        adp.close()


def _store_and_wait(adp: FSL2Adapter, key: ObjectKey, payload: bytes) -> None:
    """Submit a store and poll until it reports success.

    ``_execute_store`` swallows write failures (it logs, unlinks the tmp
    file and reports ``success=False``) while still publishing a
    completed result, so the result itself must be asserted.
    """
    task_id = adp.submit_store_task([key], cast("list[MemoryObj]", [_Buf(payload)]))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adp.pop_completed_store_tasks()
        if task_id in completed:
            assert int(completed[task_id]) >= 0, "store task reported failure"
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


class TestODirectShortWrite:
    def test_store_persists_all_bytes_on_short_write(
        self,
        odirect_adapter: FSL2Adapter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A kernel accepting only part of the buffer must not truncate."""
        real_write = os.write
        calls: list[int] = []

        # NOTE: ``lmcache.utils.os`` is the ``os`` module itself, so this
        # patches os.write process-wide for the duration of the test --
        # including the adapter's executor threads. Capping at 4096 keeps
        # unrelated one-byte eventfd notifies intact.
        def short_write(fd: int, buf) -> int:  # type: ignore[no-untyped-def]
            chunk = bytes(buf)[:4096]
            calls.append(len(chunk))
            return real_write(fd, chunk)

        monkeypatch.setattr("lmcache.utils.os.write", short_write)

        payload = bytes(range(256)) * 128  # 32 KiB, several blocks
        _store_and_wait(odirect_adapter, _key(), payload)

        # The store path only uses os.write on its O_DIRECT branch; it
        # falls back to buffered aiofiles when the payload is not block
        # aligned. More than one call therefore pins both "the O_DIRECT
        # branch ran" and "the write loop ran" -- without this the test
        # would silently pass while exercising the buffered path.
        assert len(calls) > 1, f"expected a looping O_DIRECT write, got {calls}"

        written = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert len(written) == 1, f"expected one stored file, got {written}"
        assert written[0].read_bytes() == payload
