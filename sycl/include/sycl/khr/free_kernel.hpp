//===-- free_kernel.hpp --- KHR free function kernels extension -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prototype (Step 1, header-only) of a Khronos-style free function kernels
// extension. Provides the SYCL_KHR_KERNEL(...) decoration macro and the kernel
// kind property names in the sycl::khr namespace. The macro forwards to the
// existing [[__sycl_detail__::add_ir_attributes_function(...)]] machinery used
// by the experimental extension; the property types alias the experimental
// definitions so there is a single source of truth.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>

#include <cstddef>

namespace sycl {
inline namespace _V1 {
namespace khr {

// Kernel-kind property names re-exported into sycl::khr. These alias the
// experimental property values; do NOT duplicate the property definitions.
//
// NOTE the rename: the khr spelling of the nd-range kind is `nd_kernel`
// (shortened from the experimental `nd_range_kernel`).
template <int Dims>
inline constexpr auto nd_kernel =
    ext::oneapi::experimental::nd_range_kernel<Dims>;

inline constexpr auto single_task_kernel =
    ext::oneapi::experimental::single_task_kernel;

namespace detail {
namespace exp_detail = ::sycl::ext::oneapi::experimental::detail;

// Upper bound on the number of properties a single SYCL_KHR_KERNEL(...) may
// carry. The macro expands a fixed number of slots (see below); this constant
// must match that arity.
inline constexpr std::size_t MaxKernelProperties = 8;

// A property value list captured as a type pack. The macro forwards the whole
// __VA_ARGS__ into make_property_bundle(...) in ONE go, so the C++ parser (not
// the preprocessor) groups commas nested inside template-argument lists such as
// work_group_size<8, 8, 8>. This is why a fixed-arity slot expansion is used
// instead of a preprocessor FOR_EACH, which would split those nested commas.
template <typename... Props> struct property_bundle {
  static_assert(sizeof...(Props) <= MaxKernelProperties,
                "SYCL_KHR_KERNEL supports at most 8 properties.");
};

template <typename... Props>
property_bundle<Props...> make_property_bundle(Props...);

template <std::size_t I, typename First, typename... Rest>
struct pack_element {
  using type = typename pack_element<I - 1, Rest...>::type;
};
template <typename First, typename... Rest>
struct pack_element<0, First, Rest...> {
  using type = First;
};

// Maps slot index I to the (name, value) pair for the I-th property. Slots past
// the end of the property list yield an empty name, which
// add_ir_attributes_function drops (an empty attribute name generates no IR
// attribute). This lets the macro emit a constant number of arguments while
// supporting 0..MaxKernelProperties actual properties.
template <std::size_t I, typename Bundle, bool InRange> struct property_slot_impl;

template <std::size_t I, typename... Props>
struct property_slot_impl<I, property_bundle<Props...>, true> {
  using prop_t =
      exp_detail::remove_cvref_t<typename pack_element<I, Props...>::type>;
  static constexpr auto name =
      exp_detail::FunctionPropertyMetaInfo<prop_t>::name;
  static constexpr auto value =
      exp_detail::FunctionPropertyMetaInfo<prop_t>::value;
};

template <std::size_t I, typename... Props>
struct property_slot_impl<I, property_bundle<Props...>, false> {
  static constexpr const char *name = "";
  static constexpr int value = 0;
};

template <std::size_t I, typename Bundle> struct property_slot;
template <std::size_t I, typename... Props>
struct property_slot<I, property_bundle<Props...>>
    : property_slot_impl<I, property_bundle<Props...>,
                         (I < sizeof...(Props))> {};

} // namespace detail

// Host-evaluable, compile-time kernel-property queries (Step 4).
//
// These traits read the kernel's properties straight off the decorated
// FunctionDecl via the __builtin_sycl_*_property builtins, so they evaluate at
// host parse time WITHOUT the integration header. The template parameter is the
// kernel itself, passed as a non-type template parameter function pointer
// (`is_kernel_v<k_nd>`); for a templated kernel pass the specialization
// (`is_nd_kernel_v<k_tmpl<float>, 1>`), which reads the substituted property
// values off the instantiated decl.

// True if Func carries either kernel-kind property (i.e. it is a free function
// kernel entry point).
template <auto *Func>
inline constexpr bool is_kernel_v =
    __builtin_sycl_has_property(Func, "sycl-nd-range-kernel") ||
    __builtin_sycl_has_property(Func, "sycl-single-task-kernel");

// True if Func is an nd-range kernel whose dimensionality equals Dims.
template <auto *Func, int Dims>
inline constexpr bool is_nd_kernel_v =
    __builtin_sycl_has_property(Func, "sycl-nd-range-kernel") &&
    (__builtin_sycl_get_property(Func, "sycl-nd-range-kernel") == Dims);

// True if Func is a single-task kernel.
template <auto *Func>
inline constexpr bool is_single_task_kernel_v =
    __builtin_sycl_has_property(Func, "sycl-single-task-kernel");

} // namespace khr
} // namespace _V1
} // namespace sycl

// Bundle type for the forwarded property values. The values are captured by a
// single decltype(make_property_bundle(...)) so nested template-argument commas
// survive macro expansion.
#define __SYCL_KHR_KERNEL_BUNDLE(...)                                          \
  decltype(::sycl::khr::detail::make_property_bundle(__VA_ARGS__))

#define __SYCL_KHR_KERNEL_SLOT_NAME(I, ...)                                    \
  ::sycl::khr::detail::property_slot<I,                                        \
                                    __SYCL_KHR_KERNEL_BUNDLE(__VA_ARGS__)>::name

#define __SYCL_KHR_KERNEL_SLOT_VALUE(I, ...)                                   \
  ::sycl::khr::detail::property_slot<                                          \
      I, __SYCL_KHR_KERNEL_BUNDLE(__VA_ARGS__)>::value

// Variadic, pure-forwarding decoration macro. Properties are the arguments; the
// list may be empty. Because there is no leading fixed argument there is no
// trailing-comma problem, so SYCL_KHR_KERNEL() is C++17-safe.
//
// The expansion emits MaxKernelProperties name slots followed by
// MaxKernelProperties value slots into ONE add_ir_attributes_function, matching
// its required N-names-then-N-values shape. Unused trailing slots have empty
// names and are dropped.
//
// NOTE (Step 4 dissolution-chain change): the attribute is emitted in BOTH the
// host and device compilations. add_ir_attributes_function only GENERATES LLVM
// IR attributes in the device pass (SYCLAddIRAttributesFunction is
// SYCLIsDevice / SilentlyIgnoreSYCLIsHost, and CodeGenFunction gates IR
// emission on SYCLIsDevice), so the attribute is INERT on the host -- it
// produces no IR and no diagnostic -- yet it is still attached to the
// FunctionDecl. This makes the kernel's properties DECLARATION-LOCAL: they are
// readable at host parse time via __builtin_sycl_has_property /
// __builtin_sycl_get_property WITHOUT any device->host integration-header
// round-trip. (Step 1 used `#else /*empty*/`, mirroring the experimental
// extension; that hid the properties from the host.)
#define SYCL_KHR_KERNEL(...)                                                   \
  [[__sycl_detail__::add_ir_attributes_function(                              \
      __SYCL_KHR_KERNEL_SLOT_NAME(0, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_NAME(1, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_NAME(2, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_NAME(3, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_NAME(4, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_NAME(5, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_NAME(6, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_NAME(7, __VA_ARGS__),                            \
      __SYCL_KHR_KERNEL_SLOT_VALUE(0, __VA_ARGS__),                           \
      __SYCL_KHR_KERNEL_SLOT_VALUE(1, __VA_ARGS__),                           \
      __SYCL_KHR_KERNEL_SLOT_VALUE(2, __VA_ARGS__),                           \
      __SYCL_KHR_KERNEL_SLOT_VALUE(3, __VA_ARGS__),                           \
      __SYCL_KHR_KERNEL_SLOT_VALUE(4, __VA_ARGS__),                           \
      __SYCL_KHR_KERNEL_SLOT_VALUE(5, __VA_ARGS__),                           \
      __SYCL_KHR_KERNEL_SLOT_VALUE(6, __VA_ARGS__),                           \
      __SYCL_KHR_KERNEL_SLOT_VALUE(7, __VA_ARGS__))]]

// Checked launch surface (Step 5): khr::kernel_function, khr::nd_launch and
// khr::single_task. Included last so the Step-4 property traits above are
// already declared when launch.hpp consumes them. (launch.hpp's own include of
// this header is a pragma-once no-op in that direction.)
#include <sycl/khr/launch.hpp>
