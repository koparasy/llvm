#pragma once

namespace sycl {
namespace detail {

struct host_pipe_map {
  static void add(void *, const char *) {}
};

} // namespace detail
} // namespace sycl