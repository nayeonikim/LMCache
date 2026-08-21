# SPDX-License-Identifier: Apache-2.0

"""Argument parsing tests for ``lmcache trace replay``."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.trace.replay_command import add_replay_arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_replay_arguments(parser)
    return parser


_REQUIRED_ARGS = [
    "trace.lct",
    "--l1-size-gb",
    "0.1",
    "--eviction-policy",
    "LRU",
]


def test_replay_evaluation_arguments_default_to_legacy_behavior() -> None:
    args = _parser().parse_args(_REQUIRED_ARGS)

    assert args.time_scale == 1.0
    assert args.replay_cache_salt_suffix == ""
    assert args.store_drain_timeout_seconds == 60.0
    assert args.write_reservation_timeout_seconds == 0.0


def test_replay_evaluation_arguments_are_parsed() -> None:
    args = _parser().parse_args(
        _REQUIRED_ARGS
        + [
            "--time-scale",
            "2.5",
            "--replay-cache-salt-suffix",
            "iter-0007",
            "--store-drain-timeout-seconds",
            "120",
            "--write-reservation-timeout-seconds",
            "45",
        ]
    )

    assert args.time_scale == 2.5
    assert args.replay_cache_salt_suffix == "iter-0007"
    assert args.store_drain_timeout_seconds == 120.0
    assert args.write_reservation_timeout_seconds == 45.0
