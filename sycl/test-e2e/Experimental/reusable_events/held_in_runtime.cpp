// REQUIRES: aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: event signals and waits held in the SYCL runtime behind a host task
// (handler::host_task) complete, and keep the dependencies captured at
// submission when the event is signaled again in the meantime.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <cassert>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;

// Submits a host task to Q which runs until Go is set.
static void blockQueue(sycl::queue &Q, std::atomic<bool> &Go) {
  Q.submit([&](sycl::handler &CGH) {
    CGH.host_task([&Go] {
      while (!Go.load())
        std::this_thread::yield();
    });
  });
}

int main() {
  sycl::queue Q1{sycl::property::queue::in_order{}};
  sycl::context Ctx = Q1.get_context();
  sycl::device Dev = Q1.get_device();
  sycl::queue Q2{Ctx, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q3{Ctx, Dev, sycl::property::queue::in_order{}};

  int *Data = sycl::malloc_host<int>(8, Q1);
  for (int I = 0; I < 8; ++I)
    Data[I] = 0;

  // The signal is held behind the host task, which only finishes once the
  // signal call has returned.
  {
    std::atomic<bool> Go{false};
    auto Event = syclex::make_event(Ctx);
    blockQueue(Q1, Go);
    Q1.single_task([=] { Data[0] = 1; });
    syclex::enqueue_signal_event(Q1, Event);
    syclex::enqueue_wait_event(Q2, Event);
    Q2.single_task([=] { Data[1] = Data[0] + 1; });
    Go = true;
    Q2.wait();
    assert(Data[1] == 2);
  }

  // A kernel held behind a host task depends on the first signal of Event.
  // Event is then signaled again behind that kernel. Had the kernel taken the
  // second signal for its dependency, the two would wait for each other.
  {
    std::atomic<bool> Go{false};
    auto Event = syclex::make_event(Ctx);
    Q2.single_task([=] { Data[2] = 5; });
    syclex::enqueue_signal_event(Q2, Event);

    blockQueue(Q1, Go);
    sycl::event Kernel = Q1.submit([&](sycl::handler &CGH) {
      CGH.depends_on(Event);
      CGH.single_task([=] { Data[3] = Data[2]; });
    });
    syclex::enqueue_wait_event(Q2, Kernel);
    syclex::enqueue_signal_event(Q2, Event);

    Go = true;
    Event.wait();
    assert(Data[3] == 5);
  }

  // The event of a kernel held behind a host task is signaled on another
  // queue: new dependencies on it follow the signal, and the kernel completing
  // later does not override it.
  {
    std::atomic<bool> Go{false};
    blockQueue(Q1, Go);
    sycl::event Kernel = Q1.single_task([=] { Data[4] = 1; });

    Q2.single_task([=] { Data[5] = 9; });
    syclex::enqueue_signal_event(Q2, Kernel);
    Q3.submit([&](sycl::handler &CGH) {
      CGH.depends_on(Kernel);
      CGH.single_task([=] { Data[6] = Data[5]; });
    });
    Q3.wait();
    assert(Data[6] == 9);

    Go = true;
    Q1.wait();
    Kernel.wait();
    assert(Data[4] == 1);
  }

  sycl::free(Data, Q1);
  return 0;
}
