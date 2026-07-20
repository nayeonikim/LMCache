// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include "write_stream_compat.h"
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace lmcache {
namespace connector {

namespace {

// Returns how many write streams the filesystem backing `fd` exposes.
// Throws std::runtime_error (typically ENOTTY) if it has no support.
uint32_t query_max_write_streams(int fd) {
  write_stream_compat::WriteStreamArg arg{};
  arg.op_flags = write_stream_compat::kOpGetMax;
  if (::ioctl(fd, write_stream_compat::kIoctl, &arg) != 0) {
    throw std::runtime_error("FS_IOC_WRITE_STREAM GET_MAX failed using " +
                             std::string(write_stream_compat::kAbiName) + ": " +
                             strerror(errno));
  }
  return arg.max_streams;
}

// Attach `stream_id` to every subsequent write on `fd`.
void apply_write_stream(int fd, uint32_t stream_id) {
  write_stream_compat::WriteStreamArg arg{};
  arg.op_flags = write_stream_compat::kOpSet;
  arg.stream_id = stream_id;
  if (::ioctl(fd, write_stream_compat::kIoctl, &arg) != 0) {
    throw std::runtime_error("FS_IOC_WRITE_STREAM SET stream " +
                             std::to_string(stream_id) + " failed using " +
                             write_stream_compat::kAbiName + ": " +
                             strerror(errno));
  }
}

}  // namespace

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

std::string FSConnector::replace_all(const std::string& str,
                                     const std::string& from,
                                     const std::string& to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.size(), to);
    pos += to.size();
  }
  return result;
}

WriteStreamPolicy FSConnector::parse_write_stream_policy(
    const std::string& name) {
  if (name.empty()) {
    return WriteStreamPolicy::kDisabled;
  }
  if (name == WRITE_STREAM_POLICY_KV_RANK_WORKER) {
    return WriteStreamPolicy::kKvRankWorker;
  }
  throw std::runtime_error("FSConnector: unsupported write stream policy '" +
                           name + "' (expected an empty string or '" +
                           WRITE_STREAM_POLICY_KV_RANK_WORKER + "')");
}

uint32_t FSConnector::select_write_stream_id(uint32_t kv_rank,
                                             uint32_t stream_count) {
  if (stream_count == 0) {
    throw std::runtime_error(
        "FSConnector: write stream count must be greater than zero");
  }

  const uint32_t worker_index = (kv_rank >> 16) & 0xff;
  return (worker_index % stream_count) + 1;
}

std::vector<std::string> FSConnector::split_key(const std::string& key) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 4 && parts.size() != 5) {
    throw std::runtime_error(
        "FSConnector: malformed key (expected 4 or 5 '@'-separated fields): " +
        key);
  }
  return parts;
}

uint32_t FSConnector::parse_kv_rank(const std::string& key) {
  const std::vector<std::string> parts = split_key(key);
  const std::string& kv_rank_hex = parts[1];
  // ComputeKVRank() packs world_size into bits 24+, so kv_rank exceeds 32 bits
  // once world_size >= 256. Parse the full value and keep the low 32 bits: the
  // worker byte (bits 16-23) is preserved, and parsing into a uint32_t here
  // would instead reject those keys and fail every write under the policy.
  uint64_t kv_rank = 0;
  const char* first = kv_rank_hex.data();
  const char* last = first + kv_rank_hex.size();
  const auto [end, error] = std::from_chars(first, last, kv_rank, 16);
  if (error != std::errc() || end != last) {
    throw std::runtime_error("FSConnector: invalid kv_rank in key: " + key);
  }
  return static_cast<uint32_t>(kv_rank);
}

std::string FSConnector::key_to_filename(const std::string& key) {
  // Input key format (from _object_key_to_string):
  //   Unsalted:
  //   <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>
  //   Salted  :
  //   <model_name>@<kv_rank_hex>@<object_group_id_hex>@
  //   <chunk_hash_hex>@<cache_salt>
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  //   Unsalted:
  //   <model_name_safe>@0x<kv_rank_hex>@<object_group_id_hex>@
  //   <chunk_hash_hex>.data
  //   Salted  :
  //   <model_name_safe>@0x<kv_rank_hex>@<object_group_id_hex>@
  //   <chunk_hash_hex>@<cache_salt>.data
  //
  // NOTE: both model_name and cache_salt are forbidden from containing
  // '@' (invariant enforced on the Python side), so splitting on '@'
  // is unambiguous — no marker, no rsplit.

  // Split on '@' — must yield 4 (unsalted) or 5 (salted) fields.
  const std::vector<std::string> parts = split_key(key);

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& object_group_id = parts[2];
  const std::string& chunk_hash = parts[3];
  const std::string cache_salt = parts.size() == 5 ? parts[4] : std::string();

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model_name, "/", PATH_SLASH_REPLACEMENT);

  // Emit filename. Salt is appended at the tail to match fs_l2_adapter.py.
  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() +
                 object_group_id.size() + chunk_hash.size() +
                 cache_salt.size() + 32);
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  result += KEY_SEP;
  result += object_group_id;
  result += KEY_SEP;
  result += chunk_hash;
  if (!cache_salt.empty()) {
    result += KEY_SEP;
    result += cache_salt;
  }
  result += FILE_EXT;
  return result;
}

// ---------------------------------------------------------------
// read/write helpers
// ---------------------------------------------------------------

static void write_all(int fd, const void* data, size_t len) {
  size_t written = 0;
  const char* ptr = static_cast<const char*>(data);
  while (written < len) {
    ssize_t n = ::write(fd, ptr + written, len - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write failed: " + std::string(strerror(errno)));
    }
    if (n == 0) {
      throw std::runtime_error("write returned 0");
    }
    written += static_cast<size_t>(n);
  }
}

static size_t read_all(int fd, void* buf, size_t len) {
  size_t total = 0;
  char* ptr = static_cast<char*>(buf);
  while (total < len) {
    ssize_t n = ::read(fd, ptr + total, len - total);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

// ---------------------------------------------------------------
// FSConnector
// ---------------------------------------------------------------

FSConnector::FSConnector(std::string base_path, int num_workers,
                         std::string relative_tmp_dir, bool use_odirect,
                         size_t read_ahead_size,
                         const std::string& write_stream_policy,
                         uint32_t write_stream_count)
    : ConnectorBase(num_workers),
      base_path_(std::move(base_path)),
      relative_tmp_dir_(std::move(relative_tmp_dir)),
      use_odirect_(use_odirect),
      disk_block_size_(0),
      read_ahead_size_(read_ahead_size),
      write_stream_policy_(parse_write_stream_policy(write_stream_policy)),
      write_stream_count_(write_stream_count) {
  if (write_stream_policy_ == WriteStreamPolicy::kDisabled &&
      write_stream_count_ != 0) {
    throw std::runtime_error(
        "FSConnector: write_stream_count is set but no write stream policy "
        "is configured, so it would have no effect");
  }

  // Create base directory
  std::filesystem::create_directories(base_path_);

  // Create tmp directory if configured
  if (!relative_tmp_dir_.empty()) {
    auto tmp_path = std::filesystem::path(base_path_) / relative_tmp_dir_;
    std::filesystem::create_directories(tmp_path);
  }

  // Query disk block size for O_DIRECT
  if (use_odirect_) {
    struct statvfs st;
    if (statvfs(base_path_.c_str(), &st) == 0) {
      disk_block_size_ = st.f_bsize;
    }
  }

  // Resolve the stream count once, here, so the write path never issues a
  // probing ioctl. Write streams are opt-in, so an unsupported filesystem is
  // a configuration error: fail now rather than let every write silently lose
  // its placement hint, which would quietly invalidate any measurement taken
  // with the feature "enabled".
  if (write_stream_policy_ != WriteStreamPolicy::kDisabled) {
    const uint32_t max_streams = probe_max_write_streams();
    if (max_streams == 0) {
      throw std::runtime_error(
          "FSConnector: filesystem at '" + base_path_ +
          "' reports 0 write streams; write_stream_policy cannot be used");
    }
    if (write_stream_count_ == 0) {
      write_stream_count_ = max_streams;
    } else if (write_stream_count_ > max_streams) {
      throw std::runtime_error("FSConnector: write_stream_count " +
                               std::to_string(write_stream_count_) +
                               " exceeds the write stream maximum of " +
                               std::to_string(max_streams) + " at '" +
                               base_path_ + "'");
    }
  }

  start_workers();  // IMPORTANT: call at END of constructor
}

uint32_t FSConnector::probe_max_write_streams() const {
  // FS_IOC_WRITE_STREAM is a regular-file ioctl, so support has to be probed
  // on a real file inside base_path rather than on the directory itself.
  // mkstemp() keeps the name unique even if several connectors share a
  // base_path, and leaves nothing behind once the probe is unlinked.
  std::string probe_path =
      (std::filesystem::path(base_path_) / ".lmcache_write_stream_probe.XXXXXX")
          .string();

  const int fd = ::mkstemp(probe_path.data());
  if (fd < 0) {
    throw std::runtime_error(
        "FSConnector: cannot probe write stream support, failed to create a "
        "temporary file under '" +
        base_path_ + "': " + strerror(errno));
  }

  uint32_t max_streams = 0;
  try {
    max_streams = query_max_write_streams(fd);
  } catch (const std::exception& e) {
    ::close(fd);
    ::unlink(probe_path.c_str());
    throw std::runtime_error(
        "FSConnector: write_stream_policy requires a filesystem with write "
        "stream support at '" +
        base_path_ + "': " + e.what());
  }
  ::close(fd);
  ::unlink(probe_path.c_str());
  return max_streams;
}

FSConnector::~FSConnector() { close(); }

WorkerFSConn FSConnector::create_connection() {
  WorkerFSConn conn;
  conn.base_path = base_path_;
  if (!relative_tmp_dir_.empty()) {
    conn.tmp_dir = std::filesystem::path(base_path_) / relative_tmp_dir_;
  }
  conn.use_odirect = use_odirect_;
  conn.disk_block_size = disk_block_size_;
  conn.read_ahead_size = read_ahead_size_;
  conn.write_stream_policy = write_stream_policy_;
  conn.write_stream_count = write_stream_count_;
  return conn;
}

void FSConnector::do_single_get(WorkerFSConn& conn, const std::string& key,
                                void* buf, size_t len, size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  int flags = O_RDONLY;
  bool do_odirect = conn.use_odirect;
  if (do_odirect) {
    bool aligned = conn.disk_block_size > 0 && len % conn.disk_block_size == 0;
    if (aligned) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#endif
    } else {
      do_odirect = false;
    }
  }

  int fd = ::open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + file_path.string() +
                             ": " + strerror(errno));
  }

  try {
    size_t n;
    if (conn.read_ahead_size > 0 && len > conn.read_ahead_size) {
      // Trigger filesystem readahead with a small initial
      // read, then read the remainder.
      size_t ra = conn.read_ahead_size;
      size_t n_head = read_all(fd, buf, ra);
      if (n_head < ra) {
        // Short read on the head portion — treat as
        // incomplete
        n = n_head;
      } else {
        size_t n_tail = read_all(fd, static_cast<char*>(buf) + ra, len - ra);
        n = n_head + n_tail;
      }
    } else {
      n = read_all(fd, buf, len);
    }
    if (n != len) {
      throw std::runtime_error("incomplete read for " + file_path.string() +
                               ": expected " + std::to_string(len) + ", got " +
                               std::to_string(n));
    }
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
}

void FSConnector::do_single_set(WorkerFSConn& conn, const std::string& key,
                                const void* buf, size_t len,
                                size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  // Skip if already stored on disk
  if (std::filesystem::exists(file_path)) {
    return;
  }

  // Determine temp file path
  std::filesystem::path tmp_path;
  if (!conn.tmp_dir.empty()) {
    tmp_path = conn.tmp_dir / filename;
  } else {
    tmp_path = file_path;
    tmp_path.replace_extension(TMP_EXT);
  }

  int flags = O_CREAT | O_WRONLY | O_TRUNC;
  bool do_odirect = conn.use_odirect;
  if (do_odirect) {
    bool aligned = conn.disk_block_size > 0 && len % conn.disk_block_size == 0;
    if (aligned) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#endif
    } else {
      do_odirect = false;
    }
  }

  int fd = ::open(tmp_path.c_str(), flags, 0644);
  if (fd < 0) {
    throw std::runtime_error("open for write failed: " + tmp_path.string() +
                             ": " + strerror(errno));
  }

  try {
    if (conn.write_stream_policy == WriteStreamPolicy::kKvRankWorker) {
      apply_write_stream(fd, select_write_stream_id(parse_kv_rank(key),
                                                    conn.write_stream_count));
    }
    write_all(fd, buf, len);
  } catch (...) {
    ::close(fd);
    // Clean up temp file on failure
    std::filesystem::remove(tmp_path);
    throw;
  }
  ::close(fd);

  // Atomic rename: tmp -> final
  std::error_code ec;
  std::filesystem::rename(tmp_path, file_path, ec);
  if (ec) {
    // Try to clean up, but prioritize reporting the original error.
    std::error_code remove_ec;
    std::filesystem::remove(tmp_path, remove_ec);
    throw std::runtime_error("rename failed: " + tmp_path.string() + " -> " +
                             file_path.string() + ": " + ec.message());
  }
}

bool FSConnector::do_single_exists(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  return std::filesystem::exists(file_path);
}

bool FSConnector::do_single_delete(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  std::error_code ec;
  return std::filesystem::remove(file_path, ec);
}

}  // namespace connector
}  // namespace lmcache
