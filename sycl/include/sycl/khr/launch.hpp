//===-- launch.hpp --- KHR free function kernel checked launch -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prototype (Step 5, header-only) of the checked sycl::khr free function kernel
// launch surface. Provides:
//   * khr::kernel_function<Func>  -- the kernel-function handle the user passes
//   * khr::nd_launch(...)         -- queue and handler overloads
//   * khr::single_task(...)       -- queue and handler overloads
//
// The launchers add a compile-time gate -- kernel KIND + DIMENSIONALITY and
// argument-TYPE static_asserts -- in FRONT of the existing experimental free
// function kernel launchers, then FORWARD to them. The gate reuses the
// host-readable Step-4 traits (is_nd_kernel_v / is_single_task_kernel_v, read
// straight off the decorated FunctionDecl) and the Step-2/3 arg-validity trait
// (is_valid_kernel_arg_v), so the checks fire at host parse time WITHOUT the
// integration header. Size/work-group mismatches stay a RUNTIME concern by
// design (constant nd_ranges are uncommon, so a compile-time size check is
// unreasonable); this step only checks what is compile-time-knowable from the
// kernel's declared properties and the nd_range<Dims> TYPE.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/khr/free_kernel.hpp>
#include <sycl/khr/kernel_arg_traits.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>

#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {
namespace khr {

// The kernel-function handle the khr user passes to a launcher. It IS an
// experimental::kernel_function_s<Func> value (a single source of truth), so
// the experimental forwardee deduces Func from it cleanly.
template <auto *Func>
inline constexpr auto kernel_function =
    ext::oneapi::experimental::kernel_function<Func>;

// nd_launch -- queue form.
template <auto *Func, int Dims, typename... ArgsT>
void nd_launch(queue Q, nd_range<Dims> Range,
               ext::oneapi::experimental::kernel_function_s<Func> KF,
               ArgsT &&...Args) {
  static_assert(is_nd_kernel_v<Func, Dims>,
                "khr::nd_launch: kernel's declared dimensionality does not "
                "match the nd_range used to launch it (or the function is not "
                "an nd_range kernel)");
  static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                "khr::nd_launch: a kernel argument type is not a valid free "
                "function kernel argument");
  ext::oneapi::experimental::nd_launch<Func>(std::move(Q), Range, KF,
                                             std::forward<ArgsT>(Args)...);
}

// nd_launch -- handler form.
template <auto *Func, int Dims, typename... ArgsT>
void nd_launch(handler &CGH, nd_range<Dims> Range,
               ext::oneapi::experimental::kernel_function_s<Func> KF,
               ArgsT &&...Args) {
  static_assert(is_nd_kernel_v<Func, Dims>,
                "khr::nd_launch: kernel's declared dimensionality does not "
                "match the nd_range used to launch it (or the function is not "
                "an nd_range kernel)");
  static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                "khr::nd_launch: a kernel argument type is not a valid free "
                "function kernel argument");
  ext::oneapi::experimental::nd_launch<Func>(CGH, Range, KF,
                                             std::forward<ArgsT>(Args)...);
}

// single_task -- queue form.
template <auto *Func, typename... ArgsT>
void single_task(queue Q,
                 ext::oneapi::experimental::kernel_function_s<Func> KF,
                 ArgsT &&...Args) {
  static_assert(is_single_task_kernel_v<Func>,
                "khr::single_task: kernel is not a single_task kernel");
  static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                "khr::single_task: a kernel argument type is not a valid free "
                "function kernel argument");
  ext::oneapi::experimental::single_task<Func>(std::move(Q), KF,
                                               std::forward<ArgsT>(Args)...);
}

// single_task -- handler form.
template <auto *Func, typename... ArgsT>
void single_task(handler &CGH,
                 ext::oneapi::experimental::kernel_function_s<Func> KF,
                 ArgsT &&...Args) {
  static_assert(is_single_task_kernel_v<Func>,
                "khr::single_task: kernel is not a single_task kernel");
  static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                "khr::single_task: a kernel argument type is not a valid free "
                "function kernel argument");
  ext::oneapi::experimental::single_task<Func>(CGH, KF,
                                               std::forward<ArgsT>(Args)...);
}

} // namespace khr
} // namespace _V1
} // namespace sycl
