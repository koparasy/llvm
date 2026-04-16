#include <sycl/sycl.hpp>

void submit_increment(sycl::queue &queue, int *data) {
  queue.submit([&](sycl::handler &handler) {
    handler.single_task<class SimpleIncrement>([=]() {
      data[0] += 1;
    });
  });
}