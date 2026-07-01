# SYCL + Clang `-fmodules`: Status Report

## Goal
Evaluate Clang's implicit `-fmodules` (which auto-generates module maps in the
background and caches parsed headers as Precompiled Modules) as a way to reduce
SYCL compile time. PCM files are more powerful than PCH technology. This effort
is orthogonal to OpenVino, and I was looking in reducing compilation for AOT SYCL
 cases.

## Problems found in the current state
Out of the box, `-fsycl -fmodules` does not work:

- **(a) x86 builtin PCM on the device pass.** The device compilation
  (`spir64`) pulls x86-only builtin intrinsic modules (via `CL/cl_platform.h`
  → `<xmmintrin.h>`), which carry `requires x86`. On a non-x86 device target
  this fails outright before any SYCL header is parsed.

- **(b) SYCL integration headers.** The integration header/footer are generated
  at unique temporary paths per invocation and deleted afterward. They leaked
  into every implicitly-built module's PCM as recorded input files, so the
  cached modules failed validation on the next compile and were rebuilt every
  time — eliminating any caching benefit (and adding overhead).

## Fixes applied
Both issues were patched with changes that appear reasonably correct and are
verified on small cases. I have not published these changes yet.

## Remaining blocker: ODR and the standard library
With a module map in play, `-fsycl` compilations hit ODR failures
(`'std::align' has different definitions in different modules`) whenever the
standard library itself is not modularized: unowned system headers get absorbed
into the SYCL module's PCM and conflict with a later textual include. This is
**not** fixable by tuning the SYCL module map alone.

Clang does, however, provide a mechanism to generate a module map for **libc++**
(shipped as `libcxx/include/module.modulemap.in`, produced automatically by the
build). libstdc++ has no such module map.

## libc++ approach
We built libc++ with its module map and authored a module map for `sycl.hpp`.
With both `std` and `sycl` modularized, the ODR failures disappear and the build
becomes include-order-invariant.

## Still open: `stl_wrappers`
Under `-fsycl -fmodules`, compilation still fails because SYCL injects
`stl_wrappers/` (its `<complex>`, `<cmath>`, … shadows) ahead of libc++ on the
include path. The libc++ `std` module resolves those names to the SYCL wrappers
and pulls SYCL headers into the `std` PCM, where they are compiled without SYCL
macro context. Every module-map arrangement tried hits either header absorption,
a `std ↔ wrapper` cyclic dependency (via `#include_next`), or a macro-visibility
failure. This needs further investigation.

## What works today, and the measured gain
We can compile a translation unit that includes `sycl.hpp` plus several standard
headers (`<vector> <memory> <type_traits> <iterator> <algorithm> <functional>`)
**without** `-fsycl`. This is nearly equivalent to compiling host-only code and
sidesteps the `stl_wrappers` injection that I mentioned before.

Measured on a real compile-to-object (verified by building and running an
executable that produces correct output, identical to the non-modules build):

| Configuration            | Time   |
|--------------------------|--------|
| No modules (baseline)    | ~2.2 s |
| Modules, cold PCM cache  | ~4.4 s |
| Modules, hot PCM cache   | ~0.1 s |

That is roughly a **22× speedup with a hot PCM cache** and a **~2× slowdown
when the cache is cold**.

**Clarification — PCM is not PCH.** The cold-cache penalty is paid once. A
Precompiled Module is a shared, reusable artifact keyed by the compilation
configuration, so in an AOT build across many translation units it is populated
once and then reused for every subsequent TU. This differs fundamentally from
PCH, which is a single monolithic prefix tied to one build. The economics only
make sense when the PCM cache is expected to be heavily reused. PCM could help
customers such as Py-Torch.

## Bottom line
The mechanism and the compile-time benefit are real and the correctness is
verified for the host-only path; the two upstream bugs are fixed. The remaining
work to reach a full `-fsycl -fmodules` build is the bounded `stl_wrappers`
issue.

## Limitations and adoption caveat
The measured speedups come from a small, header-parse-dominated TU; the absolute
saving is the header-parsing cost (~2 s here), so the ratio shrinks for TUs with
heavy template instantiation in their own bodies. The gains also assume a hot,
heavily-reused PCM cache; cold builds are slower.

Most importantly, even if every remaining issue (including `stl_wrappers`) is
fixed, this approach is **expected to work only with libc++, not libstdc++**.
The whole order-invariant, ODR-free result depends on the standard library
providing a Clang module map — which libc++ ships and libstdc++ does not — and
authoring/shipping a libstdc++ module map is not something this project should
own. Since the large majority of Linux SYCL users build against libstdc++ (the
system default with GCC), adoption of `-fmodules` for SYCL would be limited to
the libc++ configuration unless libstdc++ gains its own module map upstream.
