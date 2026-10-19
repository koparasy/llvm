//===-- ReusableEvents.cpp --- Unit tests for reusable events extension --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/reusable_events.hpp>

#include <detail/event_impl.hpp>

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

class TestKernel;
MOCK_INTEGRATION_HEADER(TestKernel)

namespace {

const auto DummyEventHandle = reinterpret_cast<ur_event_handle_t>(1u);
const auto DummyContextHandle = reinterpret_cast<ur_context_handle_t>(2u);
const auto DummyDeviceHandle = reinterpret_cast<ur_device_handle_t>(3u);

int UrEventCreateExp_counter = 0;
int UrEventRelease_counter = 0;
int RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter = 0;
int RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter = 0;
bool LastCreateHadLowPowerSyncDesc = false;

ur_result_t redefinedUrEventCreateExp(void *pParams) {
  auto params = *static_cast<ur_event_create_exp_params_t *>(pParams);
  **params.pphEvent = DummyEventHandle;
  EXPECT_EQ(*params.phContext, DummyContextHandle);

  const ur_exp_event_desc_t *desc = *params.ppEventDesc;
  for (const auto *base = static_cast<const ur_base_desc_t *>(desc->pNext);
       base; base = static_cast<const ur_base_desc_t *>(base->pNext)) {
    if (base->stype == UR_STRUCTURE_TYPE_EXP_EVENT_SYNC_MODE_DESC) {
      const auto *sync =
          reinterpret_cast<const ur_exp_event_sync_mode_desc_t *>(base);
      if (sync->flags & UR_EXP_EVENT_SYNC_MODE_FLAG_LOW_POWER_WAIT)
        LastCreateHadLowPowerSyncDesc = true;
      break;
    }
  }

  UrEventCreateExp_counter++;

  return UR_RESULT_SUCCESS;
}

bool CheckUrEventReleaseHandle = true;

ur_result_t redefinedUrEventRelease(void *pParams) {
  auto params = *static_cast<ur_event_release_params_t *>(pParams);
  if (CheckUrEventReleaseHandle) {
    EXPECT_EQ(*params.phEvent, DummyEventHandle);
  }

  UrEventRelease_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS) {
    auto *result = reinterpret_cast<ur_event_status_t *>(*params.ppPropValue);
    *result = UR_EVENT_STATUS_COMPLETE;
  } else if (*params.ppropName == UR_EVENT_INFO_CONTEXT) {
    auto *result = reinterpret_cast<ur_context_handle_t *>(*params.ppPropValue);
    *result = DummyContextHandle;
  }
  return UR_RESULT_SUCCESS;
}

uint32_t ExpectedNumEventsInWaitList = 0;

ur_result_t redefinedUrEnqueueEventsWaitWithBarrierExt_wait(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList, ExpectedNumEventsInWaitList);
  for (uint32_t i = 0; i != *params.pnumEventsInWaitList; ++i) {
    EXPECT_EQ((*params.pphEventWaitList)[i], DummyEventHandle);
  }

  RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEnqueueEventsWaitWithBarrierExt_signal(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  EXPECT_EQ(DummyEventHandle, **params.pphEvent);
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);

  RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t
redefinedUrEnqueueEventsWaitWithBarrierExt_signal_create_event(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  **params.pphEvent = DummyEventHandle;
  EXPECT_EQ(*params.pnumEventsInWaitList, 0u);

  RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter++;

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventGetProfilingInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_profiling_info_params_t *>(pParams);
  if (*params.ppPropValue) {
    auto *result = reinterpret_cast<uint64_t *>(*params.ppPropValue);
    *result = 1000000; // 1ms in nanoseconds
  }
  return UR_RESULT_SUCCESS;
}

bool ReusableEventsSupported = true;
bool EventProfilingSupported = true;

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) =
          ur_bool_t{ReusableEventsSupported};
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_PER_EVENT_PROFILING_SUPPORT_EXP:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) =
          ur_bool_t{EventProfilingSupported};
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t redefinedUrContextCreate(void *pParams) {
  auto params = *static_cast<ur_context_create_params_t *>(pParams);
  **params.pphContext = DummyContextHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrContextRelease(void *pParams) {
  auto params = *static_cast<ur_context_release_params_t *>(pParams);
  EXPECT_EQ(*params.phContext, DummyContextHandle);
  return UR_RESULT_SUCCESS;
}

uint32_t ExpectedNumEventsInWaitListKernelLaunch = 0;

ur_result_t redefinedUrEnqueueKernelLaunchWithArgsExp(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  EXPECT_EQ(*params.pnumEventsInWaitList,
            ExpectedNumEventsInWaitListKernelLaunch);
  for (uint32_t i = 0; i != *params.pnumEventsInWaitList; ++i) {
    EXPECT_EQ((*params.pphEventWaitList)[i], DummyEventHandle);
  }
  **params.pphEvent = DummyEventHandle;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices && *params.pNumEntries > 0) {
    (*params.pphDevices)[0] = DummyDeviceHandle;
    (*params.pphDevices)[1] = DummyDeviceHandle;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrDeviceRelease(void *pParams) {
  auto params = *static_cast<ur_device_release_params_t *>(pParams);
  EXPECT_EQ(*params.phDevice, DummyDeviceHandle);
  return UR_RESULT_SUCCESS;
}

// Callbacks giving every UR event a distinct handle, so that the signals of an
// event can be told apart. They may be called from host task threads.
std::mutex RecordMutex;
uintptr_t LastUniqueHandle = 0x1000;
std::vector<ur_event_handle_t> SignaledHandles;
std::vector<std::vector<ur_event_handle_t>> KernelWaitLists;
std::vector<ur_event_handle_t> KernelHandles;

ur_event_handle_t nextUniqueHandle() {
  return reinterpret_cast<ur_event_handle_t>(++LastUniqueHandle);
}

ur_result_t redefinedUrEventCreateExpUnique(void *pParams) {
  auto params = *static_cast<ur_event_create_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(RecordMutex);
  **params.pphEvent = nextUniqueHandle();
  UrEventCreateExp_counter++;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEventSuccess(void *) { return UR_RESULT_SUCCESS; }

ur_result_t redefinedUrEnqueueEventsWaitWithBarrierExtRecord(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_events_wait_with_barrier_ext_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(RecordMutex);
  if (*params.pnumEventsInWaitList == 0) {
    if (*params.pphEvent && !**params.pphEvent)
      **params.pphEvent = nextUniqueHandle();
    SignaledHandles.push_back(*params.pphEvent ? **params.pphEvent : nullptr);
  } else {
    RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter++;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrEnqueueKernelLaunchRecord(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(pParams);
  std::lock_guard<std::mutex> Lock(RecordMutex);
  KernelWaitLists.emplace_back(*params.pphEventWaitList,
                               *params.pphEventWaitList +
                                   *params.pnumEventsInWaitList);
  ur_event_handle_t Handle = nextUniqueHandle();
  if (*params.pphEvent)
    **params.pphEvent = Handle;
  KernelHandles.push_back(Handle);
  return UR_RESULT_SUCCESS;
}

ur_event_handle_t currentHandle(const sycl::event &Event) {
  return sycl::detail::event_impl::resolveSignalTarget(
             sycl::detail::getSyclObjImpl(Event))
      ->getHandle();
}

} // namespace

class ReusableEventsTest : public ::testing::Test {
protected:
  void SetUp() override {

    UrEventCreateExp_counter = 0;
    UrEventRelease_counter = 0;
    RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter = 0;
    RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter = 0;
    ExpectedNumEventsInWaitList = 0;
    ExpectedNumEventsInWaitListKernelLaunch = 0;
    CheckUrEventReleaseHandle = true;
    ReusableEventsSupported = true;
    LastCreateHadLowPowerSyncDesc = false;

    mock::getCallbacks().set_replace_callback("urEventCreateExp",
                                              &redefinedUrEventCreateExp);
    mock::getCallbacks().set_replace_callback("urEventRelease",
                                              &redefinedUrEventRelease);
    mock::getCallbacks().set_replace_callback("urEventGetInfo",
                                              &redefinedUrEventGetInfo);
    mock::getCallbacks().set_replace_callback(
        "urEventGetProfilingInfo", &redefinedUrEventGetProfilingInfo);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &after_urDeviceGetInfo);
    mock::getCallbacks().set_replace_callback("urContextCreate",
                                              &redefinedUrContextCreate);
    mock::getCallbacks().set_replace_callback("urContextRelease",
                                              &redefinedUrContextRelease);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &redefinedUrEnqueueEventsWaitWithBarrierExt_wait);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueKernelLaunchWithArgsExp",
        &redefinedUrEnqueueKernelLaunchWithArgsExp);
  }

  sycl::unittest::UrMock<> Mock;
};

// Signals and waits held in the scheduler behind a host task, with distinct
// UR event handles.
class ReusableEventsSchedulerTest : public ReusableEventsTest {
protected:
  void SetUp() override {
    ReusableEventsTest::SetUp();
    CheckUrEventReleaseHandle = false;
    LastUniqueHandle = 0x1000;
    SignaledHandles.clear();
    KernelWaitLists.clear();
    KernelHandles.clear();

    mock::getCallbacks().set_replace_callback("urEventCreateExp",
                                              &redefinedUrEventCreateExpUnique);
    mock::getCallbacks().set_replace_callback("urEventRetain",
                                              &redefinedUrEventSuccess);
    mock::getCallbacks().set_replace_callback("urEventWait",
                                              &redefinedUrEventSuccess);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueEventsWaitWithBarrierExt",
        &redefinedUrEnqueueEventsWaitWithBarrierExtRecord);
    mock::getCallbacks().set_replace_callback(
        "urEnqueueKernelLaunchWithArgsExp",
        &redefinedUrEnqueueKernelLaunchRecord);

    // Created only now, so that the context is the mocked one.
    MaybeDev = sycl::platform().get_devices()[0];
    MaybeCtx.emplace(*MaybeDev);
  }

  // Submits a host task to Q which runs until Release is set.
  void blockQueue(sycl::queue &Q) {
    std::shared_future<void> Released = Release.get_future().share();
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([Released] { Released.wait(); });
    });
  }

  std::promise<void> Release;
  std::optional<sycl::device> MaybeDev;
  std::optional<sycl::context> MaybeCtx;
  sycl::device &Dev() { return *MaybeDev; }
  sycl::context &Ctx() { return *MaybeCtx; }
};

// make_event with no parameters
TEST_F(ReusableEventsTest, MakeEventDefault) {
  {
    auto event = syclex::make_event();

    auto status = event.get_info<sycl::info::event::command_execution_status>();
    EXPECT_EQ(status, sycl::info::event_command_status::complete);
  }

  // make_event should not call UR.
  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);
}

// enqueue_wait_event
TEST_F(ReusableEventsTest, EnqueueWaitEvent) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  {
    auto event = syclex::make_event(Ctx);

    EXPECT_NO_THROW({ syclex::enqueue_wait_event(Queue, event); });
  }

  // Waiting for an event which was not enqueued for signaling should complete
  // immediately.
  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 0);
  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);

  Queue.wait();
}

// enqueue_wait_events with vector
TEST_F(ReusableEventsTest, EnqueueWaitEvents) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  {
    auto event1 = syclex::make_event(Ctx);
    auto event2 = syclex::make_event(Ctx);
    std::vector<sycl::event> events{event1, event2};

    EXPECT_NO_THROW({ syclex::enqueue_wait_events(Queue, events); });
  }

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 0);
  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);

  Queue.wait();
}

// enqueue_signal_event
TEST_F(ReusableEventsTest, EnqueueSignalEvent) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev, sycl::property::queue::in_order{}};

  {
    auto event = syclex::make_event(Ctx);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 1);
    EXPECT_EQ(UrEventCreateExp_counter, 1);
    EXPECT_EQ(UrEventRelease_counter, 0);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 2);
    EXPECT_EQ(UrEventCreateExp_counter, 1);
    EXPECT_EQ(UrEventRelease_counter, 0);
  }

  EXPECT_EQ(UrEventCreateExp_counter, 1);
  EXPECT_EQ(UrEventRelease_counter, 1);

  Queue.wait();
}

// enqueue_signal_event
TEST_F(ReusableEventsTest, EnqueueSignalEventReusableEventsNotSupported) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal_create_event);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev, sycl::property::queue::in_order{}};

  ReusableEventsSupported = false;

  {
    auto event = syclex::make_event(Ctx);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 1);
    EXPECT_EQ(UrEventRelease_counter, 0);

    EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

    EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 2);
    EXPECT_EQ(UrEventRelease_counter, 1);
  }

  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 2);

  Queue.wait();
}

// Default event constructor
TEST_F(ReusableEventsTest, DefaultConstructor) {
  {
    // Default constructor is equivalent to make_event() per spec
    sycl::event event;

    auto status = event.get_info<sycl::info::event::command_execution_status>();
    EXPECT_EQ(status, sycl::info::event_command_status::complete);
  }

  EXPECT_EQ(UrEventCreateExp_counter, 0);
  EXPECT_EQ(UrEventRelease_counter, 0);
}

// Queue-returned event can be used with extension functions
TEST_F(ReusableEventsTest, QueueReturnedEvent) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  auto event = Queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });

  ExpectedNumEventsInWaitList = 1;

  // Queue-returned events are associated with the queue's context per spec;
  // verify the event is usable with extension wait functions
  EXPECT_NO_THROW({ syclex::enqueue_wait_event(Queue, event); });

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 1);

  Queue.wait();
}

// Event with profiling can query profiling info
TEST_F(ReusableEventsTest, ProfilingInfoQuery) {
  sycl::device Dev;
  sycl::platform Plt = Dev.get_platform();

  sycl::context Ctx{Dev};
  syclex::properties PropList{syclex::enable_profiling{true}};
  auto event = syclex::make_event(Ctx, PropList);
  sycl::queue Queue{Ctx, Dev, sycl::property::queue::in_order{}};

  syclex::enqueue_signal_event(Queue, event);
  event.wait();

  // Should be able to query profiling info
  EXPECT_NO_THROW({
    auto submit_time =
        event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)submit_time;
    (void)start_time;
    (void)end_time;
  });
}

// Event can be used in depends_on
TEST_F(ReusableEventsTest, EventInDependsOn) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  auto event = syclex::make_event(Ctx);
  syclex::enqueue_signal_event(Queue, event);

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_signal_counter, 1);

  ExpectedNumEventsInWaitListKernelLaunch = 1;

  EXPECT_NO_THROW({
    Queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(event);
      cgh.single_task<TestKernel>([]() {});
    });
  });

  Queue.wait();
}

// Cross-context event with wait: the event of another context is a regular
// dependency of the wait.
TEST_F(ReusableEventsTest, CrossContextEventWait) {
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefinedUrDeviceGet);
  mock::getCallbacks().set_replace_callback("urDeviceRelease",
                                            &redefinedUrDeviceRelease);

  sycl::platform Plt = sycl::platform();
  auto Devices = Plt.get_devices();

  if (Devices.size() < 2) {
    GTEST_SKIP() << "Need at least 2 devices for this test";
  }

  const sycl::device Dev1 = Devices[0];
  const sycl::device Dev2 = Devices[1];
  sycl::context Ctx1{Dev1};
  sycl::context Ctx2{Dev2};

  sycl::queue Queue1{Ctx1, Dev1};
  sycl::queue Queue2{Ctx2, Dev2};

  auto event = syclex::make_event(Ctx1);

  EXPECT_NO_THROW({ syclex::enqueue_wait_event(Queue2, event); });

  Queue1.wait();
  Queue2.wait();
}

// Cross-context events with wait (multiple events).
TEST_F(ReusableEventsTest, CrossContextEventsWait) {
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefinedUrDeviceGet);
  mock::getCallbacks().set_replace_callback("urDeviceRelease",
                                            &redefinedUrDeviceRelease);

  sycl::platform Plt = sycl::platform();
  auto Devices = Plt.get_devices();

  if (Devices.size() < 2) {
    GTEST_SKIP() << "Need at least 2 devices for this test";
  }

  const sycl::device Dev1 = Devices[0];
  const sycl::device Dev2 = Devices[1];
  sycl::context Ctx1{Dev1};
  sycl::context Ctx2{Dev2};

  sycl::queue Queue{Ctx1, Dev1};

  auto event1 = syclex::make_event(Ctx1);
  auto event2 = syclex::make_event(Ctx2);

  std::vector<sycl::event> events{event1, event2};

  EXPECT_NO_THROW({ syclex::enqueue_wait_events(Queue, events); });

  Queue.wait();
}

// Cross-context make_event and signal event - not allowed
TEST_F(ReusableEventsTest, CrossContextMakeEventSignalEvent) {
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefinedUrDeviceGet);
  mock::getCallbacks().set_replace_callback("urDeviceRelease",
                                            &redefinedUrDeviceRelease);

  sycl::platform Plt = sycl::platform();
  auto Devices = Plt.get_devices();

  if (Devices.size() < 2) {
    GTEST_SKIP() << "Need at least 2 devices for this test";
  }

  const sycl::device Dev1 = Devices[0];
  const sycl::device Dev2 = Devices[1];
  sycl::context Ctx1{Dev1};
  sycl::context Ctx2{Dev2};

  sycl::queue Queue1{Ctx1, Dev1};

  auto event = syclex::make_event(Ctx2);

  bool exception = false;

  try {
    syclex::enqueue_signal_event(Queue1, event);
  } catch (sycl::exception const &e) {
    exception = true;
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "Event context must match the queue context.");
  }

  EXPECT_TRUE(exception);

  Queue1.wait();
}

// Empty event vector wait
TEST_F(ReusableEventsTest, EmptyEventVectorWait) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  std::vector<sycl::event> empty_events;

  EXPECT_NO_THROW({ syclex::enqueue_wait_events(Queue, empty_events); });

  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 0);

  Queue.wait();
}

TEST_F(ReusableEventsTest, LowPowerEventModeChainsSyncModeDesc) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  syclex::properties PropList{
      syclex::event_mode{syclex::event_mode_enum::low_power}};
  auto event = syclex::make_event(Ctx, PropList);

  syclex::enqueue_signal_event(Queue, event);

  EXPECT_EQ(UrEventCreateExp_counter, 1);
  EXPECT_TRUE(LastCreateHadLowPowerSyncDesc);

  Queue.wait();
}

TEST_F(ReusableEventsTest, NoneEventModeDoesNotChainSyncModeDesc) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  syclex::properties PropList{
      syclex::event_mode{syclex::event_mode_enum::none}};
  auto event = syclex::make_event(Ctx, PropList);
  syclex::enqueue_signal_event(Queue, event);

  EXPECT_EQ(UrEventCreateExp_counter, 1);
  EXPECT_FALSE(LastCreateHadLowPowerSyncDesc);

  Queue.wait();
}

TEST_F(ReusableEventsTest, MakeEventLowPowerWithoutPropertiesList) {
  mock::getCallbacks().set_replace_callback(
      "urEnqueueEventsWaitWithBarrierExt",
      &redefinedUrEnqueueEventsWaitWithBarrierExt_signal);
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};

  auto event = syclex::make_event(
      Ctx, syclex::event_mode{syclex::event_mode_enum::low_power});
  syclex::enqueue_signal_event(Queue, event);

  EXPECT_EQ(UrEventCreateExp_counter, 1);
  EXPECT_TRUE(LastCreateHadLowPowerSyncDesc);
  Queue.wait();
}

// Queue with enable_profiling property
TEST_F(ReusableEventsTest, QueueWithProfilingProperty) {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev, sycl::property::queue::enable_profiling{}};

  auto event = syclex::make_event(Ctx);

  syclex::enqueue_signal_event(Queue, event);
  event.wait();

  // Event should capture profiling info because queue has profiling enabled
  EXPECT_NO_THROW({
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)end_time;
  });
}

// A signal behind a host task is held in the scheduler: the call returns while
// the host task still waits for the application, so no deadlock can arise.
TEST_F(ReusableEventsSchedulerTest, SignalBehindHostTask) {
  sycl::queue Queue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  auto event = syclex::make_event(Ctx());

  blockQueue(Queue);
  EXPECT_NO_THROW({ syclex::enqueue_signal_event(Queue, event); });

  EXPECT_TRUE(SignaledHandles.empty());
  EXPECT_EQ(event.get_info<sycl::info::event::command_execution_status>(),
            sycl::info::event_command_status::submitted);

  Release.set_value();
  event.wait();

  // The backend event is created with the event's properties when the signal
  // reaches the backend, and signaled by the barrier.
  ASSERT_EQ(SignaledHandles.size(), 1u);
  EXPECT_EQ(UrEventCreateExp_counter, 1);
  EXPECT_EQ(SignaledHandles[0], currentHandle(event));
}

// A wait behind a host task is held in the scheduler too.
TEST_F(ReusableEventsSchedulerTest, WaitBehindHostTask) {
  sycl::queue SignalQueue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  sycl::queue Queue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  auto event = syclex::make_event(Ctx());
  syclex::enqueue_signal_event(SignalQueue, event);

  blockQueue(Queue);
  EXPECT_NO_THROW({ syclex::enqueue_wait_event(Queue, event); });
  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 0);

  Release.set_value();
  Queue.wait();
  EXPECT_EQ(RedefinedUrEnqueueEventsWaitWithBarrierExt_wait_counter, 1);
}

// A dependency is captured at submission: signaling the event again before the
// dependent command reaches the backend does not change what it waits for.
TEST_F(ReusableEventsSchedulerTest, ResignalKeepsCapturedDependency) {
  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  sycl::queue SignalQueue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  sycl::queue Queue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  auto event = syclex::make_event(Ctx());

  syclex::enqueue_signal_event(SignalQueue, event);
  ur_event_handle_t First = currentHandle(event);

  blockQueue(Queue);
  Queue.submit([&](sycl::handler &CGH) {
    CGH.depends_on(event);
    CGH.single_task<TestKernel>([] {});
  });

  // The pending kernel refers to the first signal, so the event moves on to a
  // new backend event instead of signaling the first one again.
  syclex::enqueue_signal_event(SignalQueue, event);
  ur_event_handle_t Second = currentHandle(event);
  EXPECT_NE(First, Second);
  EXPECT_EQ(UrEventCreateExp_counter, 2);

  Release.set_value();
  Queue.wait();
  ASSERT_EQ(KernelWaitLists.size(), 1u);
  EXPECT_EQ(KernelWaitLists[0], std::vector<ur_event_handle_t>{First});

  // Once nothing refers to the second signal, the next one reuses it.
  syclex::enqueue_signal_event(SignalQueue, event);
  EXPECT_EQ(currentHandle(event), Second);
  EXPECT_EQ(UrEventCreateExp_counter, 2);
}

// Signaling the event of a command which is still held in the scheduler: the
// command still produces its own event, and the event refers to the signal.
TEST_F(ReusableEventsSchedulerTest, ResignalPendingCommandEvent) {
  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  sycl::queue SignalQueue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  sycl::queue Queue{Ctx(), Dev(), sycl::property::queue::in_order{}};

  blockQueue(Queue);
  sycl::event KernelEvent = Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([] {}); });

  syclex::enqueue_signal_event(SignalQueue, KernelEvent);
  ur_event_handle_t Signal = currentHandle(KernelEvent);
  ASSERT_NE(Signal, nullptr);

  Release.set_value();
  Queue.wait();
  ASSERT_EQ(KernelHandles.size(), 1u);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(KernelEvent)->getHandle(),
            KernelHandles[0]);
  EXPECT_EQ(currentHandle(KernelEvent), Signal);

  Queue.submit([&](sycl::handler &CGH) {
    CGH.depends_on(KernelEvent);
    CGH.single_task<TestKernel>([] {});
  });
  Queue.wait();
  ASSERT_EQ(KernelWaitLists.size(), 2u);
  EXPECT_EQ(KernelWaitLists[1], std::vector<ur_event_handle_t>{Signal});
}

// A signal held in the scheduler may depend on the event it signals (here the
// last event of the in-order queue). That must not keep the event alive.
TEST_F(ReusableEventsSchedulerTest, HeldSignalOfOwnDependencyIsReleased) {
  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  std::weak_ptr<sycl::detail::event_impl> KernelEventImpl;
  {
    sycl::queue Queue{Ctx(), Dev(), sycl::property::queue::in_order{}};
    blockQueue(Queue);
    sycl::event KernelEvent = Queue.submit(
        [&](sycl::handler &CGH) { CGH.single_task<TestKernel>([] {}); });
    KernelEventImpl = sycl::detail::getSyclObjImpl(KernelEvent);

    syclex::enqueue_signal_event(Queue, KernelEvent);

    Release.set_value();
    KernelEvent.wait();
    Queue.wait();
  }
  // The scheduler may defer the cleanup of commands to its next activity, so
  // trigger it. A cycle would keep the event alive regardless.
  sycl::queue Other{Ctx(), Dev()};
  Other.submit([&](sycl::handler &CGH) { CGH.host_task([] {}); }).wait();
  EXPECT_TRUE(KernelEventImpl.expired());
}

// An external event of an in-order queue is resolved when the next command is
// submitted, so it refers to the signal current at that point.
TEST_F(ReusableEventsSchedulerTest, ExternalEventCapturedAtSubmission) {
  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  sycl::queue SignalQueue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  sycl::queue Queue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  auto event = syclex::make_event(Ctx());

  Queue.ext_oneapi_set_external_event(event);
  syclex::enqueue_signal_event(SignalQueue, event);
  Queue.single_task<TestKernel>([] {});
  Queue.wait();

  ASSERT_EQ(KernelWaitLists.size(), 1u);
  EXPECT_EQ(KernelWaitLists[0],
            std::vector<ur_event_handle_t>{currentHandle(event)});
}

// Events handed back to the application compare equal to the event it holds
// while that event still stands for the same signal.
TEST_F(ReusableEventsSchedulerTest, WaitListReturnsSignaledEvent) {
  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"TestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  sycl::queue SignalQueue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  sycl::queue Queue{Ctx(), Dev()};
  auto event = syclex::make_event(Ctx());
  syclex::enqueue_signal_event(SignalQueue, event);
  ur_event_handle_t First = currentHandle(event);

  sycl::event Kernel = Queue.submit([&](sycl::handler &CGH) {
    CGH.depends_on(event);
    CGH.single_task<TestKernel>([] {});
  });
  std::vector<sycl::event> WaitList = Kernel.get_wait_list();
  ASSERT_EQ(WaitList.size(), 1u);
  EXPECT_EQ(WaitList[0], event);

  // Once the event moves on to another signal, the wait list still refers to
  // the first one.
  syclex::enqueue_signal_event(SignalQueue, event);
  ASSERT_NE(currentHandle(event), First);
  WaitList = Kernel.get_wait_list();
  ASSERT_EQ(WaitList.size(), 1u);
  EXPECT_NE(WaitList[0], event);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(WaitList[0])->getHandle(), First);
  Queue.wait();
}

TEST_F(ReusableEventsSchedulerTest, LastEventOfHeldSignal) {
  sycl::queue Queue{Ctx(), Dev(), sycl::property::queue::in_order{}};
  auto event = syclex::make_event(Ctx());

  blockQueue(Queue);
  syclex::enqueue_signal_event(Queue, event);
  auto LastEvent = Queue.ext_oneapi_get_last_event();
  ASSERT_TRUE(LastEvent.has_value());
  EXPECT_EQ(*LastEvent, event);

  Release.set_value();
  Queue.wait();
}
