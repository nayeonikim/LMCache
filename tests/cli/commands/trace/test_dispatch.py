# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`CallDispatcher` and the default v1 registrations."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace._dispatch import (
    CallDispatcher,
    ReplayContext,
    build_default_dispatcher,
)
from lmcache.v1.distributed.api import ObjectKey, PrefetchRequestSpec

_SM_PREFIX = "lmcache.v1.distributed.storage_manager.StorageManager"


class _FakeSM:
    """Minimal StorageManager stand-in for dispatcher tests.

    Records each call into ``self.calls`` so assertions can match
    forwarded arguments exactly.  For the context-manager entry, it
    returns a tiny object whose ``__enter__``/``__exit__`` push into
    ``self.events`` so the FIFO ordering can be verified.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.events: list[str] = []
        self.prefetch_handle = object()
        self.prefetch_wait_result = True
        self.prefetch_result = _FakeBitmap()

    def reserve_write(self, **kw: Any) -> dict[Any, Any]:
        self.calls.append(("reserve_write", kw))
        return {}

    def finish_write(self, **kw: Any) -> None:
        self.calls.append(("finish_write", kw))

    def submit_prefetch_task(self, **kw: Any) -> object:
        self.calls.append(("submit_prefetch_task", kw))
        return self.prefetch_handle

    def wait_prefetch_status(self, handle: object, timeout: float) -> bool:
        self.calls.append(
            ("wait_prefetch_status", {"handle": handle, "timeout": timeout})
        )
        return self.prefetch_wait_result

    def query_prefetch_status(self, handle: object) -> "_FakeBitmap | None":
        self.calls.append(("query_prefetch_status", {"handle": handle}))
        return self.prefetch_result

    def finish_read_prefetched(self, **kw: Any) -> None:
        self.calls.append(("finish_read_prefetched", kw))

    def report_status(self) -> dict[str, Any]:
        return {
            "store_controller": {
                "pending_keys_count": 0,
                "processing_keys_count": 0,
                "in_flight_task_count": 7,
                "failed_task_count": 0,
            }
        }

    def read_prefetched_results(self, keys: list[ObjectKey]) -> "_FakeCM":
        return _FakeCM(self, keys)


class _SequencedReserveSM(_FakeSM):
    """Return a configured subset of requested keys on each reserve call."""

    def __init__(self, responses: list[set[ObjectKey]]) -> None:
        super().__init__()
        self._responses = iter(responses)

    def reserve_write(self, **kw: Any) -> dict[ObjectKey, object]:
        self.calls.append(("reserve_write", kw))
        accepted = next(self._responses, set())
        return {_key: _FakeMemoryObj() for _key in kw["keys"] if _key in accepted}


class _FakeMemoryObj:
    def __init__(self, size: int = 17) -> None:
        self.data = bytearray([0xA5] * size)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self.data)


class _MaterializingReserveSM(_FakeSM):
    def __init__(self, keys: list[ObjectKey]) -> None:
        super().__init__()
        self.objects = {key: _FakeMemoryObj() for key in keys}

    def reserve_write(self, **kw: Any) -> dict[ObjectKey, _FakeMemoryObj]:
        self.calls.append(("reserve_write", kw))
        return {key: self.objects[key] for key in kw["keys"]}


class _FakeBitmap:
    """Return the first requested key as the retained prefetch set."""

    def gather(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        return keys[:1]


class _FakeCM:
    def __init__(self, parent: _FakeSM, keys: list[ObjectKey]) -> None:
        self._parent = parent
        self._keys = keys

    def __enter__(self) -> None:
        self._parent.events.append(f"enter-{len(self._keys)}")

    def __exit__(self, *_exc: object) -> None:
        self._parent.events.append(f"exit-{len(self._keys)}")


def _key(i: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([i]), model_name="test", kv_rank=0)


class TestCallDispatcher:
    def test_register_and_has(self):
        d = CallDispatcher()
        assert not d.has("x")
        d.register("x", lambda c, a: None)
        assert d.has("x")

    def test_duplicate_registration_raises(self):
        d = CallDispatcher()
        d.register("x", lambda c, a: None)
        with pytest.raises(ValueError):
            d.register("x", lambda c, a: None)

    def test_dispatch_unknown_qualname_raises_keyerror(self):
        d = CallDispatcher()
        with pytest.raises(KeyError):
            d.dispatch("no.such.qualname", ReplayContext(sm=_FakeSM()), {})

    def test_registered_qualnames_returns_new_list(self):
        d = CallDispatcher()
        d.register("a", lambda c, a: None)
        qns = d.registered_qualnames()
        qns.append("b")
        assert "b" not in d.registered_qualnames()


class TestDefaultDispatcher:
    def test_registers_all_v1_qualnames(self):
        d = build_default_dispatcher()
        expected = {
            f"{_SM_PREFIX}.reserve_write",
            f"{_SM_PREFIX}.finish_write",
            f"{_SM_PREFIX}.submit_prefetch_task",
            f"{_SM_PREFIX}.finish_read_prefetched",
            f"{_SM_PREFIX}.read_prefetched_results.__enter__",
            f"{_SM_PREFIX}.read_prefetched_results.__exit__",
        }
        assert set(d.registered_qualnames()) == expected

    def test_simple_method_forwarded_with_kwargs(self):
        sm = _FakeSM()
        ctx = ReplayContext(sm=sm)
        d = build_default_dispatcher()
        d.dispatch(
            f"{_SM_PREFIX}.reserve_write",
            ctx,
            {"keys": [_key(1)], "layout_desc": "LAYOUT", "mode": "new"},
        )
        assert sm.calls == [
            (
                "reserve_write",
                {"keys": [_key(1)], "layout_desc": "LAYOUT", "mode": "new"},
            ),
        ]

    def test_reserve_wait_retries_only_missing_keys(self):
        keys = [_key(1), _key(2)]
        sm = _SequencedReserveSM([{keys[0]}, {keys[1]}])
        ctx = ReplayContext(sm=sm, write_reservation_timeout_seconds=1.0)

        build_default_dispatcher().dispatch(
            f"{_SM_PREFIX}.reserve_write",
            ctx,
            {"keys": keys, "layout_desc": "LAYOUT", "mode": "new"},
        )

        assert [call[1]["keys"] for call in sm.calls] == [keys, [keys[1]]]

    def test_reserve_wait_timeout_is_a_dispatch_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        key = _key(1)
        sm = _SequencedReserveSM([])
        ctx = ReplayContext(sm=sm, write_reservation_timeout_seconds=1.0)
        clock = iter((0.0, 2.0))

        monkeypatch.setattr(
            "lmcache.cli.commands.trace._dispatch.time.monotonic",
            lambda: next(clock),
        )

        with pytest.raises(
            TimeoutError,
            match=r"1 keys.*in_flight_task_count.*7",
        ):
            build_default_dispatcher().dispatch(
                f"{_SM_PREFIX}.reserve_write",
                ctx,
                {"keys": [key], "layout_desc": "LAYOUT", "mode": "new"},
            )

    def test_reserve_wait_is_disabled_by_default(self):
        key = _key(1)
        sm = _SequencedReserveSM([])

        build_default_dispatcher().dispatch(
            f"{_SM_PREFIX}.reserve_write",
            ReplayContext(sm=sm),
            {"keys": [key], "layout_desc": "LAYOUT", "mode": "new"},
        )

        assert len(sm.calls) == 1

    def test_reserve_write_zero_fills_replay_buffers(self):
        keys = [_key(1), _key(2)]
        sm = _MaterializingReserveSM(keys)

        build_default_dispatcher().dispatch(
            f"{_SM_PREFIX}.reserve_write",
            ReplayContext(sm=sm),
            {"keys": keys, "layout_desc": "LAYOUT", "mode": "new"},
        )

        assert all(not any(obj.data) for obj in sm.objects.values())

    def test_legacy_prefetch_args_are_upgraded_and_completed(self):
        keys = [_key(1), _key(2)]
        sm = _FakeSM()
        ctx = ReplayContext(sm=sm, prefetch_completion_timeout_seconds=5.0)

        build_default_dispatcher().dispatch(
            f"{_SM_PREFIX}.submit_prefetch_task",
            ctx,
            {
                "keys": keys,
                "layout_desc": "LAYOUT",
                "extra_count": 2,
                "external_request_id": "request-1",
            },
        )

        call_name, call_args = sm.calls[0]
        assert call_name == "submit_prefetch_task"
        spec = call_args["spec"]
        assert isinstance(spec, PrefetchRequestSpec)
        assert spec.keys == keys
        assert spec.group_layout_descs == {0: "LAYOUT"}
        assert spec.extra_count == 2
        assert call_args["external_request_id"] == "request-1"
        assert sm.calls[1:] == [
            (
                "wait_prefetch_status",
                {"handle": sm.prefetch_handle, "timeout": 5.0},
            ),
            ("query_prefetch_status", {"handle": sm.prefetch_handle}),
            ("finish_read_prefetched", {"keys": [keys[0]], "extra_count": 2}),
        ]

    def test_legacy_prefetch_completion_timeout_is_dispatch_failure(self):
        sm = _FakeSM()
        sm.prefetch_wait_result = False
        ctx = ReplayContext(sm=sm, prefetch_completion_timeout_seconds=5.0)

        with pytest.raises(TimeoutError, match="prefetch completion"):
            build_default_dispatcher().dispatch(
                f"{_SM_PREFIX}.submit_prefetch_task",
                ctx,
                {
                    "keys": [_key(1)],
                    "layout_desc": "LAYOUT",
                    "extra_count": 0,
                    "external_request_id": "request-1",
                },
            )

    def test_read_prefetched_enter_exit_fifo(self):
        """Two overlapping contexts with identical keys exit in FIFO order."""
        sm = _FakeSM()
        ctx = ReplayContext(sm=sm)
        d = build_default_dispatcher()

        keys = [_key(1), _key(2)]
        enter = f"{_SM_PREFIX}.read_prefetched_results.__enter__"
        exit_ = f"{_SM_PREFIX}.read_prefetched_results.__exit__"

        d.dispatch(enter, ctx, {"keys": keys})
        d.dispatch(enter, ctx, {"keys": keys})
        assert sm.events == ["enter-2", "enter-2"]
        # Both contexts share the same key tuple.
        assert tuple(keys) in ctx.open_read_contexts
        assert len(ctx.open_read_contexts[tuple(keys)]) == 2

        d.dispatch(exit_, ctx, {"keys": keys})
        assert sm.events == ["enter-2", "enter-2", "exit-2"]
        d.dispatch(exit_, ctx, {"keys": keys})
        assert sm.events == ["enter-2", "enter-2", "exit-2", "exit-2"]
        # Fully drained → dict cleaned up.
        assert ctx.open_read_contexts == {}

    def test_exit_without_matching_enter_is_warning_only(self, caplog):
        sm = _FakeSM()
        ctx = ReplayContext(sm=sm)
        d = build_default_dispatcher()

        with caplog.at_level("WARNING"):
            d.dispatch(
                f"{_SM_PREFIX}.read_prefetched_results.__exit__",
                ctx,
                {"keys": [_key(1)]},
            )
        # No crash, no effect on sm.
        assert sm.events == []
