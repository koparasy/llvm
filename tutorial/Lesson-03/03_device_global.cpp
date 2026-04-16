#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/device_global/device_global.hpp>

inline sycl::ext::oneapi::experimental::device_global<int> Lesson03Global;

void submit_device_global(sycl::queue &queue, int *result) {
  queue.submit([&](sycl::handler &handler) {
    handler.single_task<class Lesson03DeviceGlobalKernel>([=]() {
      Lesson03Global = 123;
      result[0] = Lesson03Global.get();
    });
  });
}