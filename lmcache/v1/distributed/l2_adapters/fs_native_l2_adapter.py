# SPDX-License-Identifier: Apache-2.0
"""
Filesystem native L2 adapter config and factory.

Backed by the native C++ filesystem connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

logger = init_logger(__name__)


class FSNativeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    filesystem connector.

    Fields:
    - base_path: directory for storing KV cache files.
    - num_workers: C++ worker threads for I/O (default 4).
    - relative_tmp_dir: relative sub-dir for temp files.
    - use_odirect: bypass page cache via O_DIRECT.
    - read_ahead_size: trigger filesystem readahead by
      reading this many bytes first (optional).
    - write_stream_policy: empty disables hints; ``kv_rank_worker`` maps the
      ObjectKey worker rank to an XFS write stream.
    - write_stream_count: number of streams in the configured pool. Zero asks
      the filesystem for the available count when the policy is enabled.
    - write_stream_offset: number of leading streams excluded from the pool.
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
        write_stream_policy: str = "",
        write_stream_count: int = 0,
        write_stream_offset: int = 0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb
        self.write_stream_policy = write_stream_policy
        self.write_stream_count = write_stream_count
        self.write_stream_offset = write_stream_offset

    @classmethod
    def from_dict(cls, d: dict) -> "FSNativeL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        relative_tmp_dir = d.get("relative_tmp_dir", "")
        if not isinstance(relative_tmp_dir, str):
            raise ValueError("relative_tmp_dir must be a string")

        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")

        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        write_stream_policy = d.get("write_stream_policy", "")
        if not isinstance(write_stream_policy, str) or write_stream_policy not in (
            "",
            "kv_rank_worker",
        ):
            raise ValueError("write_stream_policy must be empty or 'kv_rank_worker'")

        write_stream_count = d.get("write_stream_count", 0)
        if (
            isinstance(write_stream_count, bool)
            or not isinstance(write_stream_count, int)
            or write_stream_count < 0
        ):
            raise ValueError("write_stream_count must be a non-negative integer")

        write_stream_offset = d.get("write_stream_offset", 0)
        if (
            isinstance(write_stream_offset, bool)
            or not isinstance(write_stream_offset, int)
            or write_stream_offset < 0
        ):
            raise ValueError("write_stream_offset must be a non-negative integer")

        if not write_stream_policy and (write_stream_count or write_stream_offset):
            raise ValueError(
                "write_stream_policy is required when a stream pool is configured"
            )

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
            write_stream_policy=write_stream_policy,
            write_stream_count=write_stream_count,
            write_stream_offset=write_stream_offset,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS native L2 adapter config fields:\n"
            "- base_path (str): directory for KV "
            "cache files (required)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- relative_tmp_dir (str): relative "
            "sub-dir for temp files (default empty)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (default false)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes "
            "first (optional)\n"
            "- max_capacity_gb (float): max L2 capacity "
            "in GB for usage tracking / eviction "
            "(default 0 = disabled)\n"
            "- write_stream_policy (str): empty disables hints; "
            "'kv_rank_worker' maps worker ranks to streams\n"
            "- write_stream_count (int): streams in the placement pool; "
            "0 uses the filesystem maximum (default 0)\n"
            "- write_stream_offset (int): leading streams excluded from "
            "the placement pool (default 0)"
        )


def _create_fs_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ filesystem connector."""
    try:
        # First Party
        from lmcache.lmcache_fs import (
            LMCacheFSClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "FS native L2 adapter requires the C++ FS "
            "extension. Build with: pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    if not isinstance(config, FSNativeL2AdapterConfig):
        raise TypeError("config must be an FSNativeL2AdapterConfig")
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
        config.write_stream_policy,
        config.write_stream_count,
        config.write_stream_offset,
    )
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, read_ahead=%s)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
    )
    return NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="FSNativeL2Adapter",
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
            "write_stream_policy": config.write_stream_policy,
            "write_stream_count": config.write_stream_count,
            "write_stream_offset": config.write_stream_offset,
        },
    )


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)
