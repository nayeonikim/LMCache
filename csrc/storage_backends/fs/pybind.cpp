// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <cstdint>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_fs, m) {
  py::class_<lmcache::connector::FSConnector>(m, "LMCacheFSClient")
      .def(py::init<std::string, int, std::string, bool, size_t, std::string,
                    uint32_t>(),
           py::arg("base_path"), py::arg("num_workers"),
           py::arg("relative_tmp_dir") = "", py::arg("use_odirect") = false,
           py::arg("read_ahead_size") = 0, py::arg("write_stream_policy") = "",
           py::arg("write_stream_count") = 0)
      .def_static("select_write_stream_id",
                  &lmcache::connector::FSConnector::select_write_stream_id,
                  py::arg("kv_rank"), py::arg("stream_count"),
                  "Map a packed ObjectKey.kv_rank to its 1-based write stream "
                  "ID under the kv_rank_worker policy.")
      .def_static("parse_kv_rank",
                  &lmcache::connector::FSConnector::parse_kv_rank,
                  py::arg("key"),
                  "Extract the low 32 bits of ObjectKey.kv_rank from a "
                  "serialized native-connector key (diagnostics).")
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::FSConnector);
}
