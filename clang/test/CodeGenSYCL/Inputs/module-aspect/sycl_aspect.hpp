#pragma once

#define __SYCL_TYPE(x) [[__sycl_detail__::sycl_type(x)]]

namespace sycl {
inline namespace _V1 {

enum class __SYCL_TYPE(aspect) aspect {
  host = 0,
  cpu = 1,
  gpu = 2,
  accelerator = 3,
  custom = 4,
  fp16 = 5,
  fp64 = 6,
};

} // namespace _V1
} // namespace sycl
