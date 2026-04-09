// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==- work_item_primitives.cpp - SYCL scalar work-item free queries test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/work_item_primitives.hpp>
#include <sycl/sub_group.hpp>

#include <array>
#include <cassert>

int main() {
  constexpr std::size_t n = 16;

  {
    constexpr int checks_number = 8;
    constexpr int global_range_index = 1;
    constexpr int local_id_index = 2;
    constexpr int local_range_index = 3;
    constexpr int group_id_index = 4;
    constexpr int group_range_index = 5;
    constexpr int global_linear_id_index = 6;
    constexpr int local_linear_id_index = 7;
    std::array<int, checks_number> results{};

    {
      sycl::buffer<int> results_buf(results.data(),
                    sycl::range<1>(checks_number));
      sycl::queue q;
      sycl::nd_range<1> ndr(sycl::range<1>{n}, sycl::range<1>{4});

      q.submit([&](sycl::handler &cgh) {
      auto results_acc =
        results_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for<class WorkItemPrimitiveNDRangeTest>(
        ndr, [=](sycl::nd_item<1> item) {
          if (item.get_global_linear_id() == 0) {
          results_acc[0] =
            sycl::ext::oneapi::this_work_item::get_global_id<1>(0) ==
            item.get_global_id(0);
          results_acc[global_range_index] =
            sycl::ext::oneapi::this_work_item::get_global_range<1>(0) ==
            item.get_global_range(0);
          results_acc[local_id_index] =
            sycl::ext::oneapi::this_work_item::get_local_id<1>(0) ==
            item.get_local_id(0);
          results_acc[local_range_index] =
            sycl::ext::oneapi::this_work_item::get_local_range<1>(0) ==
            item.get_local_range(0);
          results_acc[group_id_index] =
            sycl::ext::oneapi::this_work_item::get_group_id<1>(0) ==
            item.get_group(0);
          results_acc[group_range_index] =
            sycl::ext::oneapi::this_work_item::get_group_range<1>(0) ==
            item.get_group_range(0);
          results_acc[global_linear_id_index] =
            sycl::ext::oneapi::this_work_item::get_global_linear_id<1>() ==
            item.get_global_linear_id();
          results_acc[local_linear_id_index] =
            sycl::ext::oneapi::this_work_item::get_local_linear_id<1>() ==
            item.get_local_linear_id();
          }
        });
      });
    }

    for (int value : results) {
      assert(value == 1);
    }
  }

  {
    sycl::queue q;
    if (!q.get_device().get_info<sycl::info::device::sub_group_sizes>().empty()) {
      constexpr int checks_number = 1;
      std::array<int, checks_number> results{};

      sycl::nd_range<1> ndr(sycl::range<1>{n}, sycl::range<1>{4});

      {
        sycl::buffer<int> results_buf(results.data(),
                                      sycl::range<1>(checks_number));

        q.submit([&](sycl::handler &cgh) {
          auto results_acc =
              results_buf.get_access<sycl::access::mode::write>(cgh);

          cgh.parallel_for<class WorkItemPrimitiveSubGroupTest>(
              ndr, [=](sycl::nd_item<1> item) {
                if (item.get_global_linear_id() == 0) {
                  auto subgroup = item.get_sub_group();
                  results_acc[0] = sycl::ext::oneapi::this_work_item::get_sub_group_local_id() ==
                                   subgroup.get_local_linear_id();
                }
              });
        });
      }

      for (int value : results) {
        assert(value == 1);
      }
    }
  }

  return 0;
}