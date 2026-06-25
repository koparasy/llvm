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

// The khr launch-property-list container. Step 6 does NOT introduce a new
// container type (per scope): it reuses the existing experimental properties<>
// list. A plain using-declaration (not an alias template) keeps CTAD working,
// so `khr::properties{...}` deduces the contained property pack the same way
// `experimental::properties{...}` does.
using ext::oneapi::experimental::properties;

namespace detail {

// --- launch-property-list introspection (Step 6) --------------------------
//
// A launch property list may carry ONLY runtime properties: properties whose
// value is not known until launch (e.g. work_group_scratch_size(bytes)).
// Compile-time / decoration properties (kind/dim, work_group_size, ...) are
// the kernel's source of truth and belong on SYCL_KHR_KERNEL(...) -- passing
// them at launch is rejected (see static_assert in the overloads below).
//
// The discriminator follows the experimental property-key base tags
// (property.hpp:236-256): a property key is RUNTIME iff it does NOT derive
// from `compile_time_property_key_base_tag`. Compile-time keys
// (compile_time_property_key<Kind>) derive from it; runtime keys
// (run_time_property_key<...>) derive only from property_key_base_tag.
//
// Each element of a `properties<properties_type_list<Ps...>>` exposes a
// `key_t` (property_base::key_t, property.hpp:212). For a compile-time
// property the element is a `property_value<Key, ...>` whose key_t is the
// (compile-time) key; for a runtime property the element IS the key. So
// querying `Ps::key_t` and testing the base tag is uniform across both.

// True iff property element P's key is a runtime (launch-time) property key.
template <typename P>
inline constexpr bool is_runtime_launch_property_v = !std::is_base_of_v<
    ext::oneapi::experimental::detail::compile_time_property_key_base_tag,
    typename P::key_t>;

// Iterate the contained property keys of a property list and decide whether
// every one is a runtime property. The empty list trivially passes (empty
// fold == true), which is the realistic v1 case.
template <typename PropsT>
struct launch_props_are_runtime_only : std::true_type {};

template <typename... Ps>
struct launch_props_are_runtime_only<
    ext::oneapi::experimental::properties<
        ext::oneapi::experimental::detail::properties_type_list<Ps...>>>
    : std::bool_constant<(is_runtime_launch_property_v<Ps> && ... && true)> {};

template <typename PropsT>
inline constexpr bool launch_props_are_runtime_only_v =
    launch_props_are_runtime_only<PropsT>::value;

// True iff PropsT is the empty property list. single_task has no experimental
// launch_config/properties forwardee, so a non-empty list is "not yet
// supported" there (Step 6 option b); an empty list forwards to the plain
// no-props single_task (option a).
template <typename PropsT>
inline constexpr bool is_empty_launch_props_v =
    std::is_same_v<PropsT, ext::oneapi::experimental::empty_properties_t>;

} // namespace detail

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

// nd_launch -- queue form WITH a launch-property list (Step 6).
// PropsT is constrained to be a property list (is_property_list_v) so this
// overload does NOT collide with the no-properties overload above (whose 3rd
// arg is the kernel_function_s<Func> handle, NOT a property list). Forwards
// to the experimental launch_config FFK overload.
template <auto *Func, int Dims, typename PropsT, typename... ArgsT,
          typename = std::enable_if_t<
              ext::oneapi::experimental::is_property_list_v<PropsT>>>
void nd_launch(queue Q, nd_range<Dims> Range, PropsT Props,
               ext::oneapi::experimental::kernel_function_s<Func> KF,
               ArgsT &&...Args) {
  static_assert(
      detail::launch_props_are_runtime_only_v<PropsT>,
      "khr::nd_launch: only runtime launch properties may be passed here; "
      "compile-time properties belong on the kernel's SYCL_KHR_KERNEL "
      "decoration");
  static_assert(is_nd_kernel_v<Func, Dims>,
                "khr::nd_launch: kernel's declared dimensionality does not "
                "match the nd_range used to launch it (or the function is not "
                "an nd_range kernel)");
  static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                "khr::nd_launch: a kernel argument type is not a valid free "
                "function kernel argument");
  // Guard the forward behind the SAME runtime-only predicate as the assert
  // above. Rationale: static_asserts in a function body do NOT short-circuit
  // -- every one is evaluated even after a prior one fails -- and the forward
  // instantiates launch_config, which carries its OWN compile-time-effect
  // static_assert (enqueue_functions.hpp:58, "launch_config does not allow
  // properties with compile-time kernel effects"). So a compile-time property
  // list would fire BOTH our assert and launch_config's, emitting two
  // diagnostics for one mistake (and breaking the -verify test's expected
  // count). `if constexpr` makes the forward a DISCARDED branch when the list
  // is not runtime-only, so launch_config is never instantiated and our
  // runtime-only assert is the single, clear diagnostic.
  if constexpr (detail::launch_props_are_runtime_only_v<PropsT>) {
    ext::oneapi::experimental::nd_launch<Func>(
        std::move(Q), ext::oneapi::experimental::launch_config{Range, Props},
        KF, std::forward<ArgsT>(Args)...);
  }
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

// nd_launch -- handler form WITH a launch-property list (Step 6).
template <auto *Func, int Dims, typename PropsT, typename... ArgsT,
          typename = std::enable_if_t<
              ext::oneapi::experimental::is_property_list_v<PropsT>>>
void nd_launch(handler &CGH, nd_range<Dims> Range, PropsT Props,
               ext::oneapi::experimental::kernel_function_s<Func> KF,
               ArgsT &&...Args) {
  static_assert(
      detail::launch_props_are_runtime_only_v<PropsT>,
      "khr::nd_launch: only runtime launch properties may be passed here; "
      "compile-time properties belong on the kernel's SYCL_KHR_KERNEL "
      "decoration");
  static_assert(is_nd_kernel_v<Func, Dims>,
                "khr::nd_launch: kernel's declared dimensionality does not "
                "match the nd_range used to launch it (or the function is not "
                "an nd_range kernel)");
  static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                "khr::nd_launch: a kernel argument type is not a valid free "
                "function kernel argument");
  // See the queue form: static_asserts don't short-circuit and the forward
  // instantiates launch_config (which has its own compile-time-effect assert),
  // so gate the forward on the same predicate -- our runtime-only assert stays
  // the single diagnostic for a rejected compile-time property list.
  if constexpr (detail::launch_props_are_runtime_only_v<PropsT>) {
    ext::oneapi::experimental::nd_launch<Func>(
        CGH, ext::oneapi::experimental::launch_config{Range, Props}, KF,
        std::forward<ArgsT>(Args)...);
  }
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

// single_task -- queue form WITH a launch-property list (Step 6).
// KNOWN GAP (Greg open-question d): the experimental single_task FFK overloads
// (enqueue_functions.hpp:189-208) have NO launch_config/properties variant, so
// there is no prop-carrying forwardee. Option (a)+(b): an EMPTY list forwards
// to the plain no-props single_task; a NON-EMPTY list is a clear "not yet"
// (consistent with v1 having no in-scope single_task launch property).
template <auto *Func, typename PropsT, typename... ArgsT,
          typename = std::enable_if_t<
              ext::oneapi::experimental::is_property_list_v<PropsT>>>
void single_task(queue Q, PropsT Props,
                 ext::oneapi::experimental::kernel_function_s<Func> KF,
                 ArgsT &&...Args) {
  static_assert(
      detail::launch_props_are_runtime_only_v<PropsT>,
      "khr::single_task: only runtime launch properties may be passed here; "
      "compile-time properties belong on the kernel's SYCL_KHR_KERNEL "
      "decoration");
  // Nest the rest behind the runtime-only predicate. static_asserts in a body
  // do NOT short-circuit (every one is evaluated even after a prior failure),
  // so without this gate a compile-time property list would fire BOTH the
  // runtime-only assert above AND the "not yet supported" assert below -- two
  // diagnostics for one mistake. Nesting makes a compile-time list yield ONLY
  // the runtime-only diagnostic; the "not yet supported" assert is then
  // reserved for a (well-formed) runtime list, which single_task has no
  // prop-carrying forwardee for today.
  if constexpr (detail::launch_props_are_runtime_only_v<PropsT>) {
    static_assert(
        detail::is_empty_launch_props_v<PropsT>,
        "khr::single_task: launch properties are not yet supported for "
        "single_task (no experimental launch_config single_task forwardee); "
        "pass an empty property list");
    static_assert(is_single_task_kernel_v<Func>,
                  "khr::single_task: kernel is not a single_task kernel");
    static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                  "khr::single_task: a kernel argument type is not a valid "
                  "free function kernel argument");
    (void)Props;
    ext::oneapi::experimental::single_task<Func>(
        std::move(Q), KF, std::forward<ArgsT>(Args)...);
  }
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

// single_task -- handler form WITH a launch-property list (Step 6).
// Same gap as the queue form: empty forwards, non-empty is "not yet".
template <auto *Func, typename PropsT, typename... ArgsT,
          typename = std::enable_if_t<
              ext::oneapi::experimental::is_property_list_v<PropsT>>>
void single_task(handler &CGH, PropsT Props,
                 ext::oneapi::experimental::kernel_function_s<Func> KF,
                 ArgsT &&...Args) {
  static_assert(
      detail::launch_props_are_runtime_only_v<PropsT>,
      "khr::single_task: only runtime launch properties may be passed here; "
      "compile-time properties belong on the kernel's SYCL_KHR_KERNEL "
      "decoration");
  // See the queue form: static_asserts don't short-circuit, so nest the rest
  // behind the runtime-only predicate -- a compile-time list then yields only
  // the runtime-only diagnostic above, not also the "not yet supported" one.
  if constexpr (detail::launch_props_are_runtime_only_v<PropsT>) {
    static_assert(
        detail::is_empty_launch_props_v<PropsT>,
        "khr::single_task: launch properties are not yet supported for "
        "single_task (no experimental launch_config single_task forwardee); "
        "pass an empty property list");
    static_assert(is_single_task_kernel_v<Func>,
                  "khr::single_task: kernel is not a single_task kernel");
    static_assert((is_valid_kernel_arg_v<std::decay_t<ArgsT>> && ...),
                  "khr::single_task: a kernel argument type is not a valid "
                  "free function kernel argument");
    (void)Props;
    ext::oneapi::experimental::single_task<Func>(
        CGH, KF, std::forward<ArgsT>(Args)...);
  }
}

} // namespace khr
} // namespace _V1
} // namespace sycl
