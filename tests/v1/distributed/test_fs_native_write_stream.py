# SPDX-License-Identifier: Apache-2.0
"""Tests for fs_native ``FS_IOC_WRITE_STREAM`` placement hints."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import fcntl
import os
import struct

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    _object_key_to_string,
)

lmcache_fs = pytest.importorskip(
    "lmcache.lmcache_fs",
    reason="lmcache_fs native extension required for write stream tests",
)


# Mirror of the ioctl ABI used by the C++ connector, so the tests can tell
# "the filesystem has no write stream support" apart from "the connector never
# checked".  struct fs_write_stream_arg is {u32 op_flags; u32 stream_id; u64
# rsvd;} and the request is _IOWR('f', 135, ...) in the asm-generic encoding.
_WRITE_STREAM_ARG = struct.Struct("=IIQ")
_WRITE_STREAM_OP_GET_MAX = 1 << 0
_FS_IOC_WRITE_STREAM = (
    (3 << 30) | (_WRITE_STREAM_ARG.size << 16) | (ord("f") << 8) | 135
)


def filesystem_supports_write_streams(directory: Path) -> bool:
    """Report whether ``directory``'s filesystem implements write streams.

    Args:
        directory: Existing directory to probe.

    Returns:
        True if ``FS_IOC_WRITE_STREAM`` GET_MAX succeeds on a file there.
    """
    probe = directory / ".write_stream_support_probe"
    fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    try:
        fcntl.ioctl(
            fd,
            _FS_IOC_WRITE_STREAM,
            _WRITE_STREAM_ARG.pack(_WRITE_STREAM_OP_GET_MAX, 0, 0),
        )
    except OSError:
        return False
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)
    return True


def kv_rank_for_worker(global_rank: int) -> int:
    """Return a ``kv_rank`` that encodes ``global_rank``.

    Args:
        global_rank: Global worker rank packed into ``kv_rank`` bits 16-23.

    Returns:
        The packed ``ObjectKey.kv_rank`` value.
    """
    return ObjectKey.ComputeKVRank(
        world_size=8,
        global_rank=global_rank,
        local_world_size=4,
        local_rank=global_rank % 4,
    )


@pytest.mark.parametrize(
    "global_rank,expected_stream",
    [(0, 1), (1, 2), (2, 3), (3, 4)],
)
def test_select_write_stream_id_maps_worker_to_stream(
    global_rank: int, expected_stream: int
) -> None:
    """Each worker rank maps to its own 1-based stream ID."""
    stream_id = lmcache_fs.LMCacheFSClient.select_write_stream_id(
        kv_rank_for_worker(global_rank), 4
    )
    assert stream_id == expected_stream


def test_select_write_stream_id_wraps_around_stream_count() -> None:
    """Worker ranks beyond the stream count wrap with modulo."""
    assert (
        lmcache_fs.LMCacheFSClient.select_write_stream_id(kv_rank_for_worker(5), 4) == 2
    )


def test_select_write_stream_id_ignores_non_worker_kv_rank_bits() -> None:
    """Only bits 16-23 of kv_rank select the stream."""
    same_worker_different_topology = ObjectKey.ComputeKVRank(
        world_size=64,
        global_rank=2,
        local_world_size=8,
        local_rank=7,
    )
    assert lmcache_fs.LMCacheFSClient.select_write_stream_id(
        same_worker_different_topology, 4
    ) == lmcache_fs.LMCacheFSClient.select_write_stream_id(kv_rank_for_worker(2), 4)


def test_select_write_stream_id_rejects_zero_stream_count() -> None:
    """A zero stream count has no valid mapping."""
    with pytest.raises(RuntimeError):
        lmcache_fs.LMCacheFSClient.select_write_stream_id(kv_rank_for_worker(0), 0)


@pytest.mark.parametrize("kv_rank", [-1, 2**32])
def test_select_write_stream_id_rejects_out_of_range_kv_rank(kv_rank: int) -> None:
    """The diagnostic API accepts only packed unsigned 32-bit ranks."""
    with pytest.raises(TypeError):
        lmcache_fs.LMCacheFSClient.select_write_stream_id(kv_rank, 4)


def serialized_key(kv_rank: int, cache_salt: str = "") -> str:
    """Serialize an ObjectKey with the given kv_rank to the native wire format.

    Args:
        kv_rank: Packed ``ObjectKey.kv_rank`` value.
        cache_salt: Optional per-user isolation salt.

    Returns:
        The native-connector serialized key string.
    """
    return _object_key_to_string(
        ObjectKey(
            chunk_hash=b"\x00\x01\x02\x03",
            model_name="llama",
            kv_rank=kv_rank,
            cache_salt=cache_salt,
        )
    )


def test_parse_kv_rank_round_trips_serialized_key() -> None:
    """parse_kv_rank recovers the kv_rank the connector actually writes."""
    kv_rank = kv_rank_for_worker(2)
    assert lmcache_fs.LMCacheFSClient.parse_kv_rank(serialized_key(kv_rank)) == kv_rank


def test_parse_kv_rank_accepts_salted_key() -> None:
    """The extra salt field does not disturb kv_rank extraction."""
    kv_rank = kv_rank_for_worker(2)
    assert (
        lmcache_fs.LMCacheFSClient.parse_kv_rank(
            serialized_key(kv_rank, cache_salt="alice")
        )
        == kv_rank
    )


def test_parse_kv_rank_handles_large_world_size() -> None:
    """world_size >= 256 makes kv_rank exceed 32 bits; the worker byte must
    still survive so placement does not fail for large deployments."""
    # world_size=256 packs bit 32, so the serialized field is 9 hex digits.
    kv_rank = ObjectKey.ComputeKVRank(
        world_size=256,
        global_rank=3,
        local_world_size=8,
        local_rank=3,
    )
    key = serialized_key(kv_rank)
    assert key.split("@")[1] == f"{kv_rank:08x}"
    assert len(key.split("@")[1]) == 9

    parsed = lmcache_fs.LMCacheFSClient.parse_kv_rank(key)
    # Low 32 bits are kept; the worker byte (bits 16-23) is global_rank == 3.
    assert (parsed >> 16) & 0xFF == 3
    assert lmcache_fs.LMCacheFSClient.select_write_stream_id(parsed, 4) == 4


def test_parse_kv_rank_rejects_malformed_key() -> None:
    """Keys without the expected field layout are rejected."""
    with pytest.raises(RuntimeError):
        lmcache_fs.LMCacheFSClient.parse_kv_rank("llama@deadbeef")


def test_parse_kv_rank_rejects_non_hex_field() -> None:
    """A non-hexadecimal kv_rank field is rejected rather than silently zeroed."""
    with pytest.raises(RuntimeError):
        lmcache_fs.LMCacheFSClient.parse_kv_rank("llama@nothex@0@abcd")


def test_unsupported_write_stream_policy_rejected(tmp_path: Path) -> None:
    """An unknown policy name fails at construction."""
    with pytest.raises(RuntimeError, match="policy"):
        lmcache_fs.LMCacheFSClient(str(tmp_path), 1, "", False, 0, "rank_mod", 0)


def test_write_stream_count_without_policy_rejected(tmp_path: Path) -> None:
    """A stream count with no policy would be silently ignored, so reject it."""
    with pytest.raises(RuntimeError, match="write_stream_count"):
        lmcache_fs.LMCacheFSClient(str(tmp_path), 1, "", False, 0, "", 4)


def test_default_config_constructs_without_write_streams(tmp_path: Path) -> None:
    """The default (disabled) configuration works on any filesystem."""
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1, "", False, 0, "", 0)
    client.close()


def test_write_stream_policy_fails_fast_on_unsupported_filesystem(
    tmp_path: Path,
) -> None:
    """Opting in on a filesystem without FS_IOC_WRITE_STREAM support fails at
    construction, rather than degrading silently or failing per write."""
    if filesystem_supports_write_streams(tmp_path):
        pytest.skip(f"filesystem backing {tmp_path} supports write streams")

    with pytest.raises(RuntimeError, match="write stream"):
        lmcache_fs.LMCacheFSClient(str(tmp_path), 1, "", False, 0, "kv_rank_worker", 0)
    assert list(tmp_path.iterdir()) == [], "probe file was not cleaned up"
