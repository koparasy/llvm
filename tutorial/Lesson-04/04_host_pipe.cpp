#include <sycl/sycl.hpp>

#include "host_pipe_compat.hpp"

using namespace sycl::ext::intel::experimental;

class Lesson04PipeTag;

void submit_host_pipe(sycl::queue &queue) {
  queue.submit([&](sycl::handler &handler) {
    handler.single_task<class Lesson04HostPipeKernel>([=]() {
      (void)host_pipe<Lesson04PipeTag, int>::read();
    });
  });
}