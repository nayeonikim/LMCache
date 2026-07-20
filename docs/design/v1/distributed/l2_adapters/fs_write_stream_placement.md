# FS Native Write-Stream Placement

This document describes the opt-in write-stream placement path in the MP
`fs_native` L2 adapter. The feature lets a filesystem place KV chunks from
different model workers on separate write streams without changing stored
bytes, filenames, or cache lookup behavior.

## Scope

The first policy is deliberately key-derived:

```text
ObjectKey.kv_rank
        |
        | global_rank is bits 16-23
        v
worker_index = (kv_rank >> 16) & 0xff
stream_id    = (worker_index % stream_count) + 1
```

Stream identifiers are 1-based because stream 0 means that no stream is set in
the filesystem ABI. Request-derived placement, directory fan-out, and policy
sharing with the raw-block backend are outside this feature.

## Data Flow

```text
FSNativeL2AdapterConfig
        |
        v
LMCacheFSClient (pybind)
        |
        v
FSConnector constructor
  - parse policy
  - create a probe file under base_path
  - FS_IOC_WRITE_STREAM GET_MAX
  - resolve write_stream_count
        |
        v
FSConnector::do_single_set
  - parse kv_rank from the serialized ObjectKey
  - select the stream ID
  - open the temporary file
  - FS_IOC_WRITE_STREAM SET
  - write and close
  - atomically rename to the final filename
```

The hint is applied to the temporary file before its first write. Rename keeps
the inode, so the stream assignment remains attached to the final file.

## Configuration Contract

`write_stream_policy` accepts:

- `""`: disabled; no write-stream ioctl is issued
- `"kv_rank_worker"`: select a stream from `ObjectKey.kv_rank`

`write_stream_count` is an unsigned 32-bit integer:

- `0`: use the maximum returned by the filesystem
- `1..max_streams`: use the configured count
- greater than `max_streams`: reject adapter construction

A nonzero count without a policy is rejected because silently ignoring it
would make placement measurements unreliable.

## UAPI Compatibility

The connector includes `<linux/fs.h>` and uses its `FS_IOC_WRITE_STREAM`
definition when available. Older userspace headers use an isolated
compatibility definition matching the XFS write-stream v3 patch series:

```text
ioctl: _IOWR('f', 135, struct fs_write_stream)
argument size: 16 bytes
operations: GET_MAX = 1 << 0, SET = 1 << 2
```

This fallback is an experimental ABI compatibility path, not a claim that
stock kernels implement the ioctl. A runtime kernel with an earlier patch
revision or a different ioctl number is incompatible and fails the startup
probe. When Linux publishes a stable UAPI in installed headers, that definition
takes precedence over the fallback.

## Failure Semantics

Write streams are opt-in, so the disabled path must behave exactly like the
baseline connector. Once explicitly enabled, failures are not silently
ignored:

- unsupported policy: constructor error
- unsupported ioctl or zero reported streams: constructor error
- configured count above the filesystem maximum: constructor error
- SET failure after a successful probe: store task failure

Failing loudly prevents a benchmark from appearing stream-enabled while its
writes are actually unhinted.

## Diagnostics and Testing

The placement path has two pure static halves, both exposed on
`LMCacheFSClient` so they can be exercised without a write-stream-capable
filesystem:

- `select_write_stream_id(kv_rank, stream_count)`: the numeric mapping, for
  checking topology-to-stream assignment.
- `parse_kv_rank(key)`: the string half, returning the low 32 bits of the
  kv_rank field. It exists so the parse/truncation contract has CI coverage —
  the mapping alone cannot reach a serialized key whose kv_rank exceeds 32 bits
  (world_size >= 256), which is where truncation matters.

Unit tests cover configuration validation, factory argument forwarding, stream
mapping, kv_rank parsing (including the large-world_size truncation case),
unsupported-policy rejection, and startup failure on filesystems that do not
implement the ioctl. Verifying SET on a final file requires a compatible kernel
and filesystem and remains an opt-in integration test.

## References

- User configuration: `docs/source/mp/l2_storage/fs_native.rst`
- Python config: `lmcache/v1/distributed/l2_adapters/fs_native_l2_adapter.py`
- Native connector: `csrc/storage_backends/fs/connector.cpp`
- Compatibility ABI: `csrc/storage_backends/fs/write_stream_compat.h`
