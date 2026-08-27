# SPDX-License-Identifier: Apache-2.0

"""Public status contract tests for L2 eviction campaign evidence."""

from __future__ import annotations

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
    L2EvictionController,
)


def test_l2_eviction_status_exposes_quiescence_and_delete_outcomes() -> None:
    adapter = MagicMock()
    adapter.get_usage.return_value = AdapterUsage(
        total_bytes_used=94,
        total_capacity_bytes=100,
    )
    state = L2AdapterEvictionState(
        adapter_id=7,
        adapter=adapter,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
            trigger_watermark=0.97,
            eviction_ratio=0.03,
        ),
    )
    state.record_trigger()
    state.record_delete_result(3, succeeded=True)
    controller = L2EvictionController([state])

    status = controller.report_status()

    assert status["pass_in_progress"] is False
    assert status["completed_passes_total"] == 0
    assert status["adapters"] == [
        {
            "adapter_id": 7,
            "eviction_policy": "LRU",
            "trigger_watermark": 0.97,
            "eviction_ratio": 0.03,
            "current_usage": 0.94,
            "total_bytes_used": 94,
            "total_capacity_bytes": 100,
            "num_cache_salt_buckets": 0,
            "trigger_count": 1,
            "delete_requested_keys_total": 3,
            "delete_succeeded_keys_total": 3,
            "delete_failed_keys_total": 0,
        }
    ]
