#include <sycl/sycl.hpp>

inline constexpr sycl::specialization_id<int> Lesson02SpecValue{7};

void submit_specialization_id(sycl::queue &queue, int *result) {
  queue.submit([&](sycl::handler &handler) {
    handler.single_task<class Lesson02SpecKernel>(
        [=](sycl::kernel_handler kernelHandler) {
          result[0] =
              kernelHandler.get_specialization_constant<Lesson02SpecValue>();
        });
  });
}