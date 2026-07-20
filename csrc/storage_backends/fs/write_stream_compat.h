// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sys/ioctl.h>

#if defined(__linux__)
  #include <linux/fs.h>
#endif

namespace lmcache {
namespace connector {
namespace write_stream_compat {

#ifdef FS_IOC_WRITE_STREAM

using WriteStreamArg = ::fs_write_stream;
constexpr unsigned long kIoctl = FS_IOC_WRITE_STREAM;
constexpr uint32_t kOpGetMax = FS_WRITE_STREAM_OP_GET_MAX;
constexpr uint32_t kOpSet = FS_WRITE_STREAM_OP_SET;
constexpr const char* kAbiName = "Linux UAPI from <linux/fs.h>";

#else

// Compatibility definition for the XFS write-stream v3 patch series. This
// fallback keeps builds working with stock userspace headers while making the
// experimental ABI dependency explicit and isolated from connector logic.
struct WriteStreamArg {
  uint32_t op_flags;
  union {
    uint32_t stream_id;
    uint32_t max_streams;
  };
  uint64_t rsvd;
};

constexpr uint32_t kOpGetMax = 1U << 0;
constexpr uint32_t kOpSet = 1U << 2;
constexpr unsigned long kIoctl = _IOWR('f', 135, WriteStreamArg);
constexpr const char* kAbiName = "XFS write-stream v3 compatibility ABI";

// The fallback bakes the struct size into kIoctl via _IOWR, so guard the
// hand-rolled layout. The UAPI branch trusts <linux/fs.h>: its own
// FS_IOC_WRITE_STREAM encodes whatever size the kernel expects, and asserting
// 16 here would turn a valid future struct into a build failure on exactly the
// machines that support the ioctl.
static_assert(sizeof(WriteStreamArg) == 16,
              "XFS write-stream v3 compatibility struct must be 16 bytes");

#endif

}  // namespace write_stream_compat
}  // namespace connector
}  // namespace lmcache
