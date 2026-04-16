#pragma once

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

template <class NameT, class DataT> class host_pipe {
public:
  struct
#ifdef __SYCL_DEVICE_ONLY__
      [[__sycl_detail__::sycl_type(host_pipe)]]
#endif
      __pipeType {
    const char Value;
  };

  static constexpr __pipeType __pipe{0};

  static DataT read() {
    (void)__pipe;
    return DataT{};
  }
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace _V1
} // namespace sycl