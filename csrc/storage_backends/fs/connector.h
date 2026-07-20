// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace lmcache {
namespace connector {

// Key encoding constants — must match fs_l2_adapter.py
static constexpr char KEY_SEP = '@';
static constexpr const char* PATH_SLASH_REPLACEMENT = "-SEP-";
static constexpr const char* FILE_EXT = ".data";
static constexpr const char* TMP_EXT = ".tmp";

// Placement policy for FS_IOC_WRITE_STREAM hints on written files.
//
// Filesystems that implement write streams (e.g. XFS on a kernel with write
// stream support) forward the hint to the block layer, which lets an NVMe FDP
// device place separately-hinted writes into different reclaim units. The
// mapping below only decides which stream a chunk belongs to; it does not
// change what is written.
enum class WriteStreamPolicy {
  // No hint is issued; writes behave exactly as without this feature.
  kDisabled,
  // Derive the stream from the global worker rank in ObjectKey.kv_rank, so
  // each worker's chunks are placed on a separate stream.
  kKvRankWorker,
};

// Policy name accepted by the constructor for kKvRankWorker. Must stay in
// sync with _WRITE_STREAM_POLICIES in fs_native_l2_adapter.py.
static constexpr const char* WRITE_STREAM_POLICY_KV_RANK_WORKER =
    "kv_rank_worker";

// Per-worker connection state for the FS connector.
// Each worker maintains its own I/O buffer for O_DIRECT.
struct WorkerFSConn {
  std::filesystem::path base_path;
  std::filesystem::path tmp_dir;  // empty if not configured
  bool use_odirect = false;
  size_t disk_block_size = 0;
  // If > 0, trigger filesystem readahead by issuing a small
  // initial read of this many bytes before reading the rest.
  size_t read_ahead_size = 0;
  WriteStreamPolicy write_stream_policy = WriteStreamPolicy::kDisabled;
  // Number of streams the policy maps onto. Resolved against the filesystem
  // maximum at construction time, so it is always > 0 once a policy is
  // enabled and unused while the policy is kDisabled.
  uint32_t write_stream_count = 0;
};

class FSConnector : public ConnectorBase<WorkerFSConn> {
 public:
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0,
              const std::string& write_stream_policy = "",
              uint32_t write_stream_count = 0);
  ~FSConnector() override;

  // Map a packed ObjectKey.kv_rank to a 1-based FS_IOC_WRITE_STREAM stream ID
  // under the kv_rank_worker policy.
  //
  // ObjectKey.ComputeKVRank() packs the global worker rank into bits 16-23 of
  // kv_rank, so that byte identifies the writing worker regardless of the
  // TP/PP topology encoded in the remaining bytes. Streams are numbered from
  // 1 because 0 means "no stream" in the kernel ABI.
  //
  // Args:
  //   kv_rank: Packed ObjectKey.kv_rank value.
  //   stream_count: Number of streams to spread workers over; must be > 0.
  //
  // Throws std::runtime_error if stream_count is 0.
  static uint32_t select_write_stream_id(uint32_t kv_rank,
                                         uint32_t stream_count);

  // Extract the low 32 bits of ObjectKey.kv_rank from a serialized key. This
  // is the string-parsing half of the placement path (the other half is
  // select_write_stream_id) and is exposed for diagnostics/testing without a
  // write-stream-capable filesystem. The field can exceed 32 bits when
  // world_size >= 256, but only bits 16-23 (the worker byte) drive placement,
  // so the low word is sufficient. Throws std::runtime_error if the key is
  // malformed or the kv_rank field is not valid hexadecimal.
  static uint32_t parse_kv_rank(const std::string& key);

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerFSConn& conn, const std::string& key) override;

 private:
  // Build the filesystem-safe filename from a serialized key string.
  //
  // Input key (from NativeConnectorL2Adapter._object_key_to_string):
  //   Unsalted:
  //   "{model}@{kv_rank:08x}@{object_group_id:x}@{hash.hex()}"
  //   Salted  :
  //   "{model}@{kv_rank:08x}@{object_group_id:x}@"
  //   "{hash.hex()}@{cache_salt}"
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  //   Unsalted:
  //   "{safe_model}@{kv_rank:#010x}@{object_group_id:x}@{hash.hex()}.data"
  //   Salted  :
  //   "{safe_model}@{kv_rank:#010x}@{object_group_id:x}@"
  //   "{hash.hex()}@{cache_salt}.data"
  //
  // Differences from the input: '/' in model becomes '-SEP-', kv_rank gains
  // a '0x' prefix, and '.data' is appended. Both model_name and cache_salt
  // are forbidden from containing '@' (enforced on the Python side), so the
  // parse is unambiguous.
  static std::string key_to_filename(const std::string& key);

  // Split a serialized key into its 4 (unsalted) or 5 (salted) '@'-separated
  // fields. Throws std::runtime_error on any other field count.
  static std::vector<std::string> split_key(const std::string& key);

  static std::string replace_all(const std::string& str,
                                 const std::string& from,
                                 const std::string& to);

  // Resolve a configured policy name. Throws std::runtime_error if unknown.
  static WriteStreamPolicy parse_write_stream_policy(const std::string& name);

  // Query how many write streams the filesystem holding base_path supports,
  // by issuing FS_IOC_WRITE_STREAM GET_MAX on a temporary file there.
  // Throws std::runtime_error if the filesystem does not support them.
  uint32_t probe_max_write_streams() const;

  std::string base_path_;
  std::string relative_tmp_dir_;
  bool use_odirect_;
  size_t disk_block_size_;
  size_t read_ahead_size_;
  WriteStreamPolicy write_stream_policy_;
  uint32_t write_stream_count_;
};

}  // namespace connector
}  // namespace lmcache
