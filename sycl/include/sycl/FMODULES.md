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
verified on small cases (committed on branch `fmodules-no-syshdr-absorption`):

- **(a)** On an unmet `requires` feature, `HandleHeaderIncludeOrImport()` now
  falls back to textual inclusion instead of erroring, so the spir64 device
  pass no longer dies on the x86-only intrinsic modules.
- **(b)** `PreprocessorOptions::resetNonModularOptions()` now clears
  `IncludeHeader`/`IncludeFooter`, so the SYCL integration header/footer are no
  longer recorded as inputs of implicitly-built modules. PCMs are now stable
  and reused across compiles instead of being rebuilt every time.

## ODR and the standard library (resolved via libc++)
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

## Resolved: `stl_wrappers` vs the `std` module
SYCL injects `stl_wrappers/` (its `<complex>`, `<cmath>`, … shadows) ahead of
libc++ on the include path — deliberately, to add device `std::complex` /
`std::` math support. Under `-fmodules` this broke: libc++'s own headers
`#include <cmath>`/`<complex>` by name, which (because the wrapper dir precedes
libc++) resolve back to the SYCL wrappers. When clang builds the `std` module it
therefore compiles the wrapper and drags SYCL headers (`access.hpp`, the SPIR-V
declarations, …) into the `std` PCM, where they are compiled without SYCL macro
context, producing errors like `'mode' must be declared before it is used` and
`__SYCL_EXPORT` incomplete-type failures.

**Root cause:** the wrappers intercept even libc++'s *internal* std includes, not
just the user's top-level `#include`. This is include-search-order, so no module
map directive fixes it. Gating on `__SYCL_DEVICE_ONLY__` is also wrong — the
device pass *defines* it, and that is exactly the compilation that fails.

**Fix (committed):** guard each wrapper's SYCL-additions block with
`#if !__building_module(std) && !__building_module(std_core)`. This is the
correct axis — it is false for the user TU and for the `sycl` module build (both
still get the specializations) and true only while the `std` module itself is
being compiled, where the wrapper must stay a pure `#include_next` pass-through.
With this guard, `-fsycl -stdlib=libc++ -fmodules` compiles cleanly (device and
host), the SYCL complex specializations remain in the `sycl` module, and the
`std` PCM no longer absorbs any SYCL-internal headers.

## What works today, and the measured gain
The **full `-fsycl -stdlib=libc++ -fmodules` path** (device + host) now compiles,
links, and runs: a TU including `<sycl/sycl.hpp>` + `<complex>` + `<vector>` was
compiled to an object, linked against a libc++-built `libsycl`, and executed on
an Intel Data Center GPU Max — producing the correct result, identical to the
non-modules build.

Measured on that real compile-to-object (`-c`, both device and host passes):

| Configuration            | Time    |
|--------------------------|---------|
| No modules (baseline)    | ~3.86 s |
| Modules, cold PCM cache  | ~8.88 s |
| Modules, hot PCM cache   | ~1.36 s |

That is roughly a **2.85× speedup with a hot PCM cache** and a **~2.3× slowdown
when the cache is cold** (paid once). The absolute saving is ~2.5 s of host-side
standard/SYCL header parsing per TU.

Note this is the honest end-to-end `-fsycl` figure. An earlier host-only proxy
(`-fsyntax-only`, no `-fsycl`) showed ~22×, but that measured header parsing in
isolation. Under real `-fsycl` the ratio is lower because (1) the device pass and
host codegen still run in full every time — modules only cache the host header
*parse* — and (2) `-c` includes irreducible codegen the cache cannot elide.

**Clarification — PCM is not PCH.** The cold-cache penalty is paid once. A
Precompiled Module is a shared, reusable artifact keyed by the compilation
configuration, so in an AOT build across many translation units it is populated
once and then reused for every subsequent TU. This differs fundamentally from
PCH, which is a single monolithic prefix tied to one build. The economics only
make sense when the PCM cache is expected to be heavily reused — the many-TU AOT
scenario. Single-file builds lose (they pay only the cold cost).

## Bottom line
The mechanism, the correctness, and the compile-time benefit are all verified
end-to-end for `-fsycl -stdlib=libc++ -fmodules`: it compiles, links, and runs
correctly, with a ~2.85× hot-cache speedup on a representative TU. Three fixes
are committed (the two above plus the `stl_wrappers` `__building_module` guard).

## Limitations and adoption caveat
The measured speedup comes from a single small TU; the ~2.5 s absolute saving is
roughly the host header-parse cost and is largely fixed per TU, so the *ratio*
shrinks for TUs with heavy template instantiation in their own bodies (more
codegen the cache cannot elide) and grows across many TUs (the parse is saved N
times). The gain assumes a hot, heavily-reused PCM cache; a from-cold single
build is slower. Only the **host** header parse is cached — the device pass is
unaffected.

**Standard-library dependency (the key adoption constraint).** The whole
order-invariant, ODR-free result depends on the standard library providing a
Clang module map. **libc++ ships one; libstdc++ does not**, and authoring/shipping
a libstdc++ module map is not something this project should own. So this works
only with `-stdlib=libc++`. Since most Linux SYCL users build against libstdc++
(the GCC default), adoption is limited to the libc++ configuration unless
libstdc++ gains its own module map upstream.

**Build/packaging reality.** Making it run required a `libsycl` built against
libc++ (a libstdc++-linked `libsycl` cannot link a libc++ translation unit — ABI
clash between `std::` and `std::__1::`). intel/llvm builds `libsycl` inside the
LLVM build (as an `LLVM_EXTERNAL_PROJECT`), so there is no turnkey
"build `libsycl` against a prebuilt clang+libc++ toolchain" flow; the validation
here stitched a standalone libc++ install into the SYCL build. A clean solution
would decouple the `libsycl` build so a fixed clang+libc++ toolchain can be built
once and `libsycl` built separately against it.
