// RUN: %clangxx -fsycl -fsyntax-only %s -I %sycl_include -DCOMBO=1
// RUN: %clangxx -fsycl -fsyntax-only %s -I %sycl_include -DCOMBO=2
// RUN: %clangxx -fsycl -fsyntax-only %s -I %sycl_include -DCOMBO=3
// RUN: %clangxx -fsycl -fsyntax-only %s -I %sycl_include -DCOMBO=4
// RUN: %clangxx -fsycl -fsyntax-only %s -I %sycl_include -DCOMBO=5
// RUN: %clangxx -fsycl -fsyntax-only %s -I %sycl_include -DCOMBO=6
// RUN: %clangxx -fsycl -fsyntax-only %s -I %sycl_include -DCOMBO=7

#if COMBO == 1
#include <sycl/khr/includes/work_item_queries.hpp>
#define TEST_KHR_QUERIES 1
#define TEST_KHR_SUB_GROUP 0
#define TEST_EXT_QUERIES 0
#elif COMBO == 2
#include <sycl/khr/work_item_queries.hpp>
#define TEST_KHR_QUERIES 1
#define TEST_KHR_SUB_GROUP 1
#define TEST_EXT_QUERIES 0
#elif COMBO == 3
#include <sycl/ext/oneapi/free_function_queries.hpp>
#define TEST_KHR_QUERIES 0
#define TEST_KHR_SUB_GROUP 1
#define TEST_EXT_QUERIES 1
#elif COMBO == 4
#include <sycl/khr/includes/work_item_queries.hpp>
#include <sycl/khr/work_item_queries.hpp>
#define TEST_KHR_QUERIES 1
#define TEST_KHR_SUB_GROUP 1
#define TEST_EXT_QUERIES 0
#elif COMBO == 5
#include <sycl/khr/includes/work_item_queries.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#define TEST_KHR_QUERIES 1
#define TEST_KHR_SUB_GROUP 1
#define TEST_EXT_QUERIES 1
#elif COMBO == 6
#include <sycl/khr/work_item_queries.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#define TEST_KHR_QUERIES 1
#define TEST_KHR_SUB_GROUP 1
#define TEST_EXT_QUERIES 1
#elif COMBO == 7
#include <sycl/khr/includes/work_item_queries.hpp>
#include <sycl/khr/work_item_queries.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#define TEST_KHR_QUERIES 1
#define TEST_KHR_SUB_GROUP 1
#define TEST_EXT_QUERIES 1
#else
#error Unsupported COMBO value
#endif

#include <type_traits>

#if TEST_EXT_QUERIES
template <int Dims> struct get_nd_item_caller {
  auto operator()() const {
    return sycl::ext::oneapi::this_work_item::get_nd_item<Dims>();
  }
};

template <int Dims> struct get_work_group_caller {
  auto operator()() const {
    return sycl::ext::oneapi::this_work_item::get_work_group<Dims>();
  }
};
#endif

#if TEST_KHR_QUERIES
template <int Dims> struct this_nd_item_caller {
  auto operator()() const { return sycl::khr::this_nd_item<Dims>(); }
};

template <int Dims> struct this_group_caller {
  auto operator()() const { return sycl::khr::this_group<Dims>(); }
};
#endif

template <template <int> class IterationPoint, int Dims,
          template <int> class Callable>
void test(Callable<Dims> &&callable) {
  static_assert(std::is_same_v<decltype(callable()), IterationPoint<Dims>>,
                "Wrong return type of work-item query");
}

SYCL_EXTERNAL void test_all() {
#if TEST_EXT_QUERIES
  test<sycl::nd_item>(get_nd_item_caller<1>{});
  test<sycl::nd_item>(get_nd_item_caller<2>{});
  test<sycl::nd_item>(get_nd_item_caller<3>{});

  test<sycl::group>(get_work_group_caller<1>{});
  test<sycl::group>(get_work_group_caller<2>{});
  test<sycl::group>(get_work_group_caller<3>{});
#endif

#if TEST_KHR_QUERIES
  test<sycl::nd_item>(this_nd_item_caller<1>{});
  test<sycl::nd_item>(this_nd_item_caller<2>{});
  test<sycl::nd_item>(this_nd_item_caller<3>{});

  test<sycl::group>(this_group_caller<1>{});
  test<sycl::group>(this_group_caller<2>{});
  test<sycl::group>(this_group_caller<3>{});
  #endif

  #if TEST_EXT_QUERIES
  static_assert(
      std::is_same_v<
          decltype(sycl::ext::oneapi::this_work_item::get_sub_group()),
          sycl::sub_group>,
      "Wrong return type of ext sub-group query");
  #endif

  #if TEST_KHR_SUB_GROUP
  static_assert(std::is_same_v<decltype(sycl::khr::this_sub_group()),
                               sycl::sub_group>,
                "Wrong return type of KHR sub-group query");
  #endif
}