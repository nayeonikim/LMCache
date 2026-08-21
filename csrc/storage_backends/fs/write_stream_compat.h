// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <linux/fs.h>
#include <sys/ioctl.h>
#include <cstdint>

#ifndef FS_IOC_WRITE_STREAM
struct fs_write_stream {
  uint32_t op_flags;
  union {
    uint32_t stream_id;
    uint32_t max_streams;
  };
  uint64_t rsvd;
};

#define FS_WRITE_STREAM_OP_GET_MAX (1U << 0)
#define FS_WRITE_STREAM_OP_GET (1U << 1)
#define FS_WRITE_STREAM_OP_SET (1U << 2)
#define FS_IOC_WRITE_STREAM _IOWR('f', 135, struct fs_write_stream)
#endif

static_assert(sizeof(struct fs_write_stream) == 16,
              "fs_write_stream UAPI layout must remain 16 bytes");
