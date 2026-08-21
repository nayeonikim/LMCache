// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include "write_stream_compat.h"
#include <charconv>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace lmcache {
namespace connector {

namespace {

uint32_t get_max_write_streams(int fd) {
  struct fs_write_stream stream = {};
  stream.op_flags = FS_WRITE_STREAM_OP_GET_MAX;
  if (::ioctl(fd, FS_IOC_WRITE_STREAM, &stream) != 0) {
    throw std::runtime_error("FS_IOC_WRITE_STREAM GET_MAX failed: " +
                             std::string(strerror(errno)));
  }
  return stream.max_streams;
}

uint32_t get_write_stream(int fd) {
  struct fs_write_stream stream = {};
  stream.op_flags = FS_WRITE_STREAM_OP_GET;
  if (::ioctl(fd, FS_IOC_WRITE_STREAM, &stream) != 0) {
    throw std::runtime_error("FS_IOC_WRITE_STREAM GET failed: " +
                             std::string(strerror(errno)));
  }
  return stream.stream_id;
}

void set_write_stream(int fd, uint32_t stream_id) {
  struct fs_write_stream stream = {};
  stream.op_flags = FS_WRITE_STREAM_OP_SET;
  stream.stream_id = stream_id;
  if (::ioctl(fd, FS_IOC_WRITE_STREAM, &stream) != 0) {
    throw std::runtime_error("FS_IOC_WRITE_STREAM SET failed for stream " +
                             std::to_string(stream_id) + ": " +
                             std::string(strerror(errno)));
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

uint64_t FSConnector::parse_kv_rank(const std::string& key) {
  std::vector<std::string> parts = split_key(key);
  const std::string& kv_rank_hex = parts[1];
  uint64_t kv_rank = 0;
  auto [end, error] = std::from_chars(
      kv_rank_hex.data(), kv_rank_hex.data() + kv_rank_hex.size(), kv_rank, 16);
  if (error != std::errc() || end != kv_rank_hex.data() + kv_rank_hex.size()) {
    throw std::runtime_error("FSConnector: invalid kv_rank in key: " + key);
  }
  return kv_rank;
}

uint32_t FSConnector::select_write_stream_id(uint64_t kv_rank,
                                             uint32_t stream_count,
                                             uint32_t stream_offset) {
  if (stream_count == 0) {
    throw std::runtime_error(
        "FSConnector: write stream count must be positive");
  }

  uint32_t worker_index = (static_cast<uint32_t>(kv_rank) >> 16) & 0xff;
  uint64_t stream_id = static_cast<uint64_t>(stream_offset) +
                       (worker_index % stream_count) + 1;
  if (stream_id > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "FSConnector: selected write stream overflows uint32");
  }
  return static_cast<uint32_t>(stream_id);
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

  std::vector<std::string> parts = split_key(key);

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

static bool try_enable_odirect(int& flags, const void* buf, size_t len,
                               size_t disk_block_size) {
#ifdef O_DIRECT
  if (disk_block_size == 0 || len % disk_block_size != 0) {
    return false;
  }
  auto addr = reinterpret_cast<std::uintptr_t>(buf);
  if (addr % disk_block_size != 0) {
    throw std::runtime_error(
        "O_DIRECT buffer address is not aligned to filesystem block size");
  }
  flags |= O_DIRECT;
  return true;
#else
  (void)flags;
  (void)buf;
  (void)len;
  (void)disk_block_size;
  return false;
#endif
}

// ---------------------------------------------------------------
// FSConnector
// ---------------------------------------------------------------

FSConnector::FSConnector(std::string base_path, int num_workers,
                         std::string relative_tmp_dir, bool use_odirect,
                         size_t read_ahead_size,
                         std::string write_stream_policy,
                         uint32_t write_stream_count,
                         uint32_t write_stream_offset)
    : ConnectorBase(num_workers),
      base_path_(std::move(base_path)),
      relative_tmp_dir_(std::move(relative_tmp_dir)),
      use_odirect_(use_odirect),
      disk_block_size_(0),
      read_ahead_size_(read_ahead_size) {
  if (write_stream_policy.empty()) {
    if (write_stream_count != 0 || write_stream_offset != 0) {
      throw std::runtime_error(
          "FSConnector: write stream policy is required for a stream pool");
    }
  } else if (write_stream_policy == "kv_rank_worker") {
    write_stream_policy_ = WriteStreamPolicy::kKvRankWorker;
  } else {
    throw std::runtime_error("FSConnector: unsupported write stream policy: " +
                             write_stream_policy);
  }

  // Create base directory
  std::filesystem::create_directories(base_path_);

  // Create tmp directory if configured
  if (!relative_tmp_dir_.empty()) {
    auto tmp_path = std::filesystem::path(base_path_) / relative_tmp_dir_;
    std::filesystem::create_directories(tmp_path);
  }

  if (write_stream_policy_ != WriteStreamPolicy::kDisabled) {
    configure_write_streams(write_stream_count, write_stream_offset);
  }

  // Query disk block size for O_DIRECT
  if (use_odirect_) {
    struct statvfs st;
    if (statvfs(base_path_.c_str(), &st) == 0) {
      disk_block_size_ = st.f_bsize;
    }
  }

  start_workers();  // IMPORTANT: call at END of constructor
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
  conn.write_stream_offset = write_stream_offset_;
  return conn;
}

void FSConnector::configure_write_streams(uint32_t requested_count,
                                          uint32_t requested_offset) {
  auto probe_template =
      std::filesystem::path(base_path_) / ".lmcache-write-stream-probe-XXXXXX";
  std::string probe_text = probe_template.string();
  std::vector<char> probe_path(probe_text.begin(), probe_text.end());
  probe_path.push_back('\0');

  int fd = ::mkstemp(probe_path.data());
  if (fd < 0) {
    throw std::runtime_error(
        "FSConnector: write stream probe creation failed: " +
        std::string(strerror(errno)));
  }

  std::filesystem::path created_path(probe_path.data());
  try {
    uint32_t max_streams = get_max_write_streams(fd);
    if (max_streams == 0) {
      throw std::runtime_error(
          "FSConnector: filesystem reports zero write streams");
    }

    uint64_t pool_end = static_cast<uint64_t>(requested_offset) +
                        static_cast<uint64_t>(requested_count);
    if (requested_count == 0) {
      if (requested_offset >= max_streams) {
        throw std::runtime_error(
            "FSConnector: write stream offset leaves no available streams");
      }
      write_stream_count_ = max_streams - requested_offset;
    } else {
      if (pool_end > max_streams) {
        throw std::runtime_error(
            "FSConnector: configured write stream pool exceeds filesystem "
            "maximum");
      }
      write_stream_count_ = requested_count;
    }
    write_stream_offset_ = requested_offset;

    uint32_t probe_stream = write_stream_offset_ + 1;
    set_write_stream(fd, probe_stream);
    if (get_write_stream(fd) != probe_stream) {
      throw std::runtime_error(
          "FSConnector: filesystem did not retain the probe write stream");
    }
  } catch (...) {
    ::close(fd);
    std::error_code remove_error;
    std::filesystem::remove(created_path, remove_error);
    throw;
  }

  ::close(fd);
  std::error_code remove_error;
  std::filesystem::remove(created_path, remove_error);
}

void FSConnector::do_single_get(WorkerFSConn& conn, const std::string& key,
                                void* buf, size_t len, size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  int flags = O_RDONLY;
  bool do_odirect = conn.use_odirect &&
                    try_enable_odirect(flags, buf, len, conn.disk_block_size);

  int fd = ::open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + file_path.string() +
                             ": " + strerror(errno));
  }

  try {
    size_t n;
    bool use_read_ahead =
        !do_odirect && conn.read_ahead_size > 0 && len > conn.read_ahead_size;
    if (use_read_ahead) {
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
  if (conn.use_odirect) {
    try_enable_odirect(flags, buf, len, conn.disk_block_size);
  }

  int fd = ::open(tmp_path.c_str(), flags, 0644);
  if (fd < 0) {
    throw std::runtime_error("open for write failed: " + tmp_path.string() +
                             ": " + strerror(errno));
  }

  try {
    if (conn.write_stream_policy == WriteStreamPolicy::kKvRankWorker) {
      uint32_t stream_id =
          select_write_stream_id(parse_kv_rank(key), conn.write_stream_count,
                                 conn.write_stream_offset);
      set_write_stream(fd, stream_id);
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
