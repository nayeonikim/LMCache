# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for :class:`StorageReplayDriver`.

Each test records a short sequence of StorageManager operations into a
binary trace file via the real ``EventBus`` + ``StorageTraceRecorder``
stack, then constructs a fresh StorageManager and drives replay over
that trace.  These tests avoid any GPU/vLLM dependencies — the whole
flow runs in-process on CPU memory.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.cli.commands.trace._dispatch import (
    CallDispatcher,
    ReplayContext,
    build_default_dispatcher,
)
from lmcache.cli.commands.trace._driver import StorageReplayDriver
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.distributed.storage_controllers.store_controller import StoreListener
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.decorator import set_tracing_enabled
from lmcache.v1.mp_observability.trace.recorder import StorageTraceRecorder
import lmcache.v1.mp_observability.event_bus as _bus_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _should_use_lazy() -> bool:
    """Lazy allocator requires CUDA.  CPU-only hosts (our primary replay
    target) must use eager allocation."""
    return torch_dev.is_available()


def _make_sm_config() -> StorageManagerConfig:
    """Build a small StorageManagerConfig suitable for CPU testing."""
    memory = L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,
        use_lazy=_should_use_lazy(),
        init_size_in_bytes=32 * 1024 * 1024,
        align_bytes=0x1000,
    )
    l1 = L1ManagerConfig(
        memory_config=memory,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    return StorageManagerConfig(
        l1_manager_config=l1,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


def _make_slow_l2_sm_config() -> StorageManagerConfig:
    config = _make_sm_config()
    config.l2_adapter_config = L2AdaptersConfig(
        adapters=[
            MockL2AdapterConfig(
                max_size_gb=0.01,
                mock_bandwidth_gb=0.00001,
            )
        ]
    )
    return config


def _make_key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=i.to_bytes(4, "big"),
        model_name="test",
        kv_rank=0,
    )


def _make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([16, 16])],
        dtypes=[torch.float16],
    )


@pytest.fixture(autouse=True)
def restore_global_bus():
    saved = _bus_module._global_bus
    yield
    _bus_module._global_bus = saved
    set_tracing_enabled(False)


@pytest.fixture
def trace_path(tmp_path):
    return str(tmp_path / "run.lct")


def _flush(bus: EventBus) -> None:
    time.sleep(0.25)
    bus._drain_all()


# ---------------------------------------------------------------------------
# Helpers: record a scripted sequence into a trace file
# ---------------------------------------------------------------------------


def _record_sequence(
    trace_path: str,
    sm_config: StorageManagerConfig,
    script: Callable[[StorageManager], None],
) -> None:
    """Record whatever ``script(sm)`` does into *trace_path*.

    ``script`` receives a live StorageManager and should call traced
    methods on it.  This helper handles the bus / recorder lifecycle.
    """
    bus = EventBus(EventBusConfig(enabled=True))
    _bus_module._global_bus = bus
    bus.start()

    sm = StorageManager(sm_config)
    rec = StorageTraceRecorder(trace_path)
    rec.attach_storage_config(sm_config)
    bus.register_subscriber(rec)
    try:
        script(sm)
        _flush(bus)
    finally:
        bus.stop()
        sm.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecordReplayRoundtrip:
    def test_store_listener_has_no_pending_to_processing_zero_gap(self):
        listener = StoreListener()
        keys = [_make_key(i) for i in range(3)]
        try:
            listener.on_l1_keys_write_finished(keys)

            assert listener.pending_count() == 3
            assert listener.processing_count() == 0
            assert listener.pop_pending_keys() == keys
            assert listener.pending_count() == 0
            assert listener.processing_count() == 3

            listener.mark_processed(len(keys))
            assert listener.processing_count() == 0
        finally:
            listener.close()

    def test_reserve_and_finish_write_replay(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(i) for i in range(3)]

        def script(sm: StorageManager) -> None:
            reserved = sm.reserve_write(keys, layout, mode="new")
            assert len(reserved) == 3
            sm.finish_write(keys)

        _record_sequence(trace_path, sm_config, script)

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run()

        assert result.records_failed == 0
        assert result.storage_status["num_l2_adapters"] == 0
        assert result.storage_status["is_healthy"] is True
        assert result.records_replayed >= 2  # reserve_write + finish_write
        assert result.header_level == "storage"
        # Same config used on both sides → digest matches.
        assert result.header_digest == result.replay_config_digest

    def test_run_waits_for_async_l2_stores(self, trace_path):
        layout = _make_layout()
        keys = [_make_key(i) for i in range(3)]

        def script(sm: StorageManager) -> None:
            reserved = sm.reserve_write(keys, layout, mode="new")
            assert len(reserved) == len(keys)
            sm.finish_write(keys)

        _record_sequence(trace_path, _make_sm_config(), script)

        with StorageReplayDriver(_make_slow_l2_sm_config(), trace_path) as driver:
            result = driver.run()
            adapter = driver.storage_manager.l2_adapters()[0][1]

            assert all(adapter.debug_has_key(key) for key in keys)

        assert result.records_failed == 0

    def test_store_drain_rejects_async_l2_failures(self, trace_path, monkeypatch):
        _record_sequence(trace_path, _make_sm_config(), lambda _sm: None)

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            monkeypatch.setattr(
                driver.storage_manager,
                "report_status",
                lambda: {
                    "num_l2_adapters": 1,
                    "store_controller": {
                        "pending_keys_count": 0,
                        "processing_keys_count": 0,
                        "in_flight_task_count": 0,
                        "failed_task_count": 1,
                        "failed_key_count": 3,
                    },
                },
            )

            with pytest.raises(RuntimeError, match="failed L2 store tasks"):
                driver._wait_for_store_drain()

    def test_store_drain_accepts_accounted_admission_rejections(
        self, trace_path, monkeypatch
    ):
        _record_sequence(trace_path, _make_sm_config(), lambda _sm: None)
        status = {
            "num_l2_adapters": 1,
            "store_controller": {
                "pending_keys_count": 0,
                "processing_keys_count": 0,
                "in_flight_task_count": 0,
                "failed_task_count": 2,
                "failed_key_count": 3,
            },
            "l2_adapters": [
                {
                    "store_admission_mode": "reject",
                    "store_rejected_keys_total": 3,
                    "store_failed_keys_total": 0,
                }
            ],
        }

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            monkeypatch.setattr(
                driver.storage_manager,
                "report_status",
                lambda: status,
            )

            driver._wait_for_store_drain()

    def test_prefetch_drain_ignores_completed_results(self, trace_path, monkeypatch):
        _record_sequence(trace_path, _make_sm_config(), lambda _sm: None)
        status = {
            "prefetch_controller": {
                "submission_queue_size": 0,
                "pending_queue_size": 0,
                "in_flight_request_count": 0,
                "lookup_phase_count": 0,
                "load_phase_count": 0,
                "completed_results_count": 7,
            }
        }

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            monkeypatch.setattr(
                driver.storage_manager,
                "report_status",
                lambda: status,
            )

            driver._wait_for_prefetch_completion()

    @pytest.mark.parametrize(
        "timeout_seconds",
        [0, -1, float("nan"), float("inf")],
    )
    def test_prefetch_completion_timeout_must_be_finite_and_positive(
        self,
        trace_path,
        timeout_seconds,
    ):
        with pytest.raises(ValueError, match="prefetch_completion_timeout_seconds"):
            StorageReplayDriver(
                _make_sm_config(),
                trace_path,
                prefetch_completion_timeout_seconds=timeout_seconds,
            )

    def test_full_prefetch_cycle_replay(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(i) for i in range(3)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")
            sm.finish_write(keys)
            handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, {0: layout}))
            assert handle is not None
            with sm.read_prefetched_results(keys) as objs:
                assert objs is not None
                assert len(objs) == 3

        _record_sequence(trace_path, sm_config, script)

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run()

        # Everything replayed, no failures.
        assert result.records_failed == 0
        assert result.records_skipped == 0
        qns = result.stats.summary().keys()
        # Every op from the script appears in stats.
        expected_substrings = [
            "reserve_write",
            "finish_write",
            "submit_prefetch_task",
            "read_prefetched_results.__enter__",
            "read_prefetched_results.__exit__",
        ]
        for sub in expected_substrings:
            assert any(sub in qn for qn in qns), (
                f"missing qualname containing {sub!r}: saw {sorted(qns)}"
            )

    def test_on_record_callback_fires_per_record(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")
            sm.finish_write(keys)

        _record_sequence(trace_path, sm_config, script)

        seen: list[tuple[str, bool]] = []

        def on_record(qualname: str, latency_s: float, failed: bool) -> None:
            seen.append((qualname, failed))

        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run(on_record=on_record)

        assert len(seen) == result.records_replayed
        assert all(not failed for _, failed in seen)


class TestMismatchHandling:
    def test_unknown_qualname_is_skipped(self, trace_path):
        """An empty dispatcher with no matching handlers skips every
        record without raising."""
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")

        _record_sequence(trace_path, sm_config, script)

        empty = CallDispatcher()
        with StorageReplayDriver(
            _make_sm_config(), trace_path, dispatcher=empty
        ) as driver:
            result = driver.run()
        assert result.records_replayed == 0
        assert result.records_skipped >= 1

    def test_handler_failure_counted(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")

        _record_sequence(trace_path, sm_config, script)

        d = build_default_dispatcher()
        # Replace the reserve_write handler with one that raises.
        failing = CallDispatcher()
        for qn in d.registered_qualnames():
            if qn.endswith(".reserve_write"):

                def _boom(_ctx: ReplayContext, _a: dict) -> None:
                    raise RuntimeError("boom")

                failing.register(qn, _boom)
            else:
                # Reuse the default handler for anything else; none
                # is expected in this script but registering keeps
                # the test robust against future decorator additions.
                # First Party
                from lmcache.cli.commands.trace._dispatch import (
                    _call_sm_method,
                    _enter_read_prefetched,
                    _exit_read_prefetched,
                )

                if qn.endswith(".__enter__"):
                    failing.register(qn, _enter_read_prefetched)
                elif qn.endswith(".__exit__"):
                    failing.register(qn, _exit_read_prefetched)
                else:
                    failing.register(qn, _call_sm_method(qn.split(".")[-1]))

        with StorageReplayDriver(
            _make_sm_config(), trace_path, dispatcher=failing
        ) as driver:
            result = driver.run()

        assert result.records_failed >= 1

    def test_reservation_timeout_aborts_replay(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")

        _record_sequence(trace_path, sm_config, script)

        dispatcher = CallDispatcher()

        def _timeout(_ctx: ReplayContext, _args: dict) -> None:
            raise TimeoutError("reservation stalled")

        dispatcher.register(
            "lmcache.v1.distributed.storage_manager.StorageManager.reserve_write",
            _timeout,
        )

        with (
            StorageReplayDriver(
                _make_sm_config(),
                trace_path,
                dispatcher=dispatcher,
            ) as driver,
            pytest.raises(TimeoutError, match="reservation stalled"),
        ):
            driver.run()


class TestReplayCacheSaltSuffix:
    def test_rewrites_object_keys_before_dispatch(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        key = ObjectKey(
            chunk_hash=b"salted",
            model_name="test",
            kv_rank=7,
            object_group_id=3,
            cache_salt="tenant",
        )

        def script(sm: StorageManager) -> None:
            sm.reserve_write([key], layout, mode="new")

        _record_sequence(trace_path, sm_config, script)

        captured: list[ObjectKey] = []
        dispatcher = CallDispatcher()
        dispatcher.register(
            "lmcache.v1.distributed.storage_manager.StorageManager.reserve_write",
            lambda _ctx, args: captured.extend(args["keys"]),
        )

        with StorageReplayDriver(
            _make_sm_config(),
            trace_path,
            dispatcher=dispatcher,
            replay_cache_salt_suffix="iter-0001",
        ) as driver:
            result = driver.run()

        assert result.records_failed == 0
        assert captured == [
            ObjectKey(
                chunk_hash=b"salted",
                model_name="test",
                kv_rank=7,
                object_group_id=3,
                cache_salt="tenant.iter-0001",
            )
        ]

    def test_replays_multiple_rounds_with_one_storage_manager(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        key = ObjectKey(
            chunk_hash=b"repeatable",
            model_name="test",
            kv_rank=7,
            object_group_id=3,
            cache_salt="tenant",
        )

        def script(sm: StorageManager) -> None:
            sm.reserve_write([key], layout, mode="new")

        _record_sequence(trace_path, sm_config, script)

        captured: list[ObjectKey] = []
        dispatcher = CallDispatcher()
        dispatcher.register(
            "lmcache.v1.distributed.storage_manager.StorageManager.reserve_write",
            lambda _ctx, args: captured.extend(args["keys"]),
        )

        with StorageReplayDriver(
            _make_sm_config(),
            trace_path,
            dispatcher=dispatcher,
        ) as driver:
            manager_identity = id(driver.storage_manager)
            first = driver.run(replay_cache_salt_suffix="round-0001")
            second = driver.run(replay_cache_salt_suffix="round-0002")

            assert id(driver.storage_manager) == manager_identity

        assert first.records_replayed == second.records_replayed == 1
        assert first.records_failed == second.records_failed == 0
        assert [item.cache_salt for item in captured] == [
            "tenant.round-0001",
            "tenant.round-0002",
        ]


class TestPacing:
    def test_replay_does_not_regress_past_monotonic(self, trace_path):
        """Replay never runs *before* the recorded offset.

        Records have t_mono=0 and a positive value; the driver always
        honors the recorded gap — there is no as-fast-as-possible
        mode, because async read/write dependencies inside
        ``StorageManager`` make it unsafe to collapse the recorded
        inter-call intervals.
        """
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")
            time.sleep(0.05)  # force a gap
            sm.finish_write(keys)

        _record_sequence(trace_path, sm_config, script)

        start = time.monotonic()
        with StorageReplayDriver(_make_sm_config(), trace_path) as driver:
            result = driver.run()
        elapsed = time.monotonic() - start

        assert result.records_failed == 0
        # Replay should have slept ≈ 50ms at minimum.  Use a generous
        # bound to avoid flakes under load.
        assert elapsed >= 0.04

    def test_time_scale_stretches_recorded_gaps(self, trace_path):
        sm_config = _make_sm_config()
        layout = _make_layout()
        keys = [_make_key(0)]

        def script(sm: StorageManager) -> None:
            sm.reserve_write(keys, layout, mode="new")
            time.sleep(0.05)
            sm.finish_write(keys)

        _record_sequence(trace_path, sm_config, script)

        start = time.monotonic()
        with StorageReplayDriver(
            _make_sm_config(), trace_path, time_scale=2.0
        ) as driver:
            result = driver.run()
        elapsed = time.monotonic() - start

        assert result.records_failed == 0
        assert elapsed >= 0.09

    @pytest.mark.parametrize("time_scale", [0, -1, float("nan"), float("inf")])
    def test_time_scale_must_be_finite_and_positive(
        self,
        trace_path,
        time_scale,
    ):
        with pytest.raises(ValueError, match="time_scale"):
            StorageReplayDriver(
                _make_sm_config(),
                trace_path,
                time_scale=time_scale,
            )

    @pytest.mark.parametrize(
        "timeout_seconds",
        [-1, float("nan"), float("inf")],
    )
    def test_write_reservation_timeout_must_be_finite_and_non_negative(
        self,
        trace_path,
        timeout_seconds,
    ):
        with pytest.raises(ValueError, match="write_reservation_timeout_seconds"):
            StorageReplayDriver(
                _make_sm_config(),
                trace_path,
                write_reservation_timeout_seconds=timeout_seconds,
            )
