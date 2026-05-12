# SYCL + Clang Modules: Investigation Findings

Branch: `features/fmodules`
Base: `4adef62218a879d84df1a23826f6a016b76e41b1`

## Goal

Enable `-fmodules` on SYCL end-to-end tests to cache sycl headers as PCM and
cut compile time from ~7s to ~1s per TU.

## Current branch state

Two commits on top of base:

| Commit | Summary |
|--------|---------|
| `4686485f61bc` | Add support for modules (skeleton) |
| `d273d7b9fc59` | [SYCL][Modules] Add Clang modules support for faster compilation |

Seven files changed:

| File | Role |
|------|------|
| `sycl/CMakeLists.txt` | Generate `module.modulemap` into build include dir |
| `sycl/test-e2e/CMakeLists.txt` | `SYCL_E2E_ENABLE_MODULES` option |
| `clang/lib/Driver/ToolChains/Clang.cpp` | `hasFlag` → `hasFlagNoClaim` for `-fmodules` propagation |
| `clang/lib/Frontend/CompilerInstance.cpp` | **`setFileIsTransient` on int header/footer (BUG — see below)** |
| `clang/lib/Lex/PPDirectives.cpp` | Fallback to textual include when module `requires` unmet |
| `clang/test/Modules/auto-import-unavailable.cpp` | Test update |
| `sycl/include/sycl/accessor.hpp` | Drop unused `<cstddef>` |

## Baseline behavior (before deeper work)

- Fresh build with `-fmodules`: PCM created, sycl.pcm ~7s build, then reused.
- Second TU with same flags: PCM reused (confirmed via `-Rmodule-build` =
  silent; md5+mtime of sycl.pcm unchanged across TUs).
- **Runtime execution fails** on many kernels. E2E lit run: 137/221 FAIL.

## Bug 1 — missing aspect enum metadata under modules

### Symptom

Device IR missing `!sycl_aspects` named metadata + `!sycl_fixed_targets`
attached to kernel definitions. Runtime:
```
terminate called after throwing an instance of 'sycl::_V1::exception'
  what():  opencl backend failed with error: 66 (UR_RESULT_ERROR_ADAPTER_SPECIFIC)
```

### Root cause

`SYCLPropagateAspectsUsagePass` in
`llvm/lib/SYCLLowerIR/SYCLPropagateAspectsUsage.cpp:724` early-exits at
line 733 when `sycl_aspects` NamedMD missing:

```cpp
AspectValueToNameMapTy AspectValues = getAspectsFromMetadata(M);
if (AspectValues.empty()) {
  assert(TypesWithAspects.empty() && ...);
  return PreservedAnalyses::all();   // <-- early exit, no metadata emitted
}
```

The `sycl_aspects` named metadata is emitted by
`CodeGenModule::Release()` in `clang/lib/CodeGen/CodeGenModule.cpp:1679`
only when `AspectsEnumDecl` is non-null. Population path:

1. Sycl `aspect` enum parsed textually in `sycl::_V1` namespace of
   `sycl/aspects.hpp`.
2. When CodeGen lowers the enum type, `CodeGenTypes::UpdateCompletedType`
   in `clang/lib/CodeGen/CodeGenTypes.cpp:272` sets `AspectsEnumDecl`
   via `CGM.setAspectsEnumDecl`.
3. `UpdateCompletedType` only fires when the enum type is
   `ConvertType`'d — i.e. the TU contains code that references the enum.

With modules, the `aspect` enum lives in PCM. Lazily deserialized.
Unless user TU mentions `sycl::aspect`, CodeGen never lowers it,
`UpdateCompletedType` never runs, `AspectsEnumDecl` stays null, metadata
never emitted, middle-end pass bails, kernels missing `sycl_fixed_targets`.

Device binary orphaned from host's kernel registration table → adapter
error at first kernel launch.

### Fix (applied)

Added `findSYCLAspectEnumAndTypesEagerly()` in
`clang/lib/CodeGen/CodeGenModule.cpp`. Called from `Release()` before
aspect MD emission. Walks the `sycl` namespace with a
`RecursiveASTVisitor`, triggers lazy PCM deserialization, finds the
`EnumDecl` with `SYCLTypeAttr::aspect`, calls `setAspectsEnumDecl`.

Aspect-tagged record types (`SYCLUsesAspectsAttr`) intentionally NOT
pre-registered — their keys reference LLVM struct types that may not
exist in the module, and the middle-end reader asserts they do. Those
will be registered normally via `ConvertRecordDeclType` when actually
used.

### Regression test

`clang/test/CodeGenSYCL/aspect-enum-from-module.cpp` plus
`clang/test/CodeGenSYCL/Inputs/module-aspect/{sycl_aspect.hpp,module.modulemap}`.
Checks that `!sycl_aspects` named MD is emitted when sycl headers come
from a PCM even though the TU never mentions `sycl::aspect`.

## Bug 2 — `kernel_signatures[]` truncated on PCM cache reuse

### Symptom

After cache reuse across TUs, `static constexpr kernel_signatures[]`
defined in SYCL integration header gets truncated:

- Fresh cache: symbol size `0x54` bytes = 7 entries (correct)
- Poisoned cache (built after another TU): `0x18` bytes = 2 entries

`KernelInfo<K>::getParamDesc(i)` bodies still emitted correctly,
compute `&kernel_signatures[i + N]`, but `N + i > 1` reads past end of
truncated array → runtime garbage → adapter error at launch.

Failure shows up in tests like `Basic/access_to_subset.cpp` when
lit runs multiple tests sharing one module cache dir.

### Root cause

The `setFileIsTransient` block added to
`clang/lib/Frontend/CompilerInstance.cpp:483-492`:

```cpp
if (!PPOpts.IncludeHeader.empty()) {
  if (auto FE = getFileManager().getOptionalFileRef(PPOpts.IncludeHeader,
                                                     /*openFile=*/false))
    getSourceManager().setFileIsTransient(*FE);
}
if (!PPOpts.IncludeFooter.empty()) {
  if (auto FE = getFileManager().getOptionalFileRef(PPOpts.IncludeFooter,
                                                     /*openFile=*/false))
    getSourceManager().setFileIsTransient(*FE);
}
```

`setFileIsTransient(FE)` tells SourceManager the file's bytes may
disappear / change. Side effect: when PCM is reused, declarations
coming from this file interact incorrectly with PCM-imported decls.
Specifically the initializer list of a namespace-scope
`static constexpr` array gets partially populated — exact clang
internal mechanism not pinned down, but empirically the
`kernel_signatures` initializer loses entries.

### Validation

1. **With block enabled:** `access_to_subset` fails rc=134 on reused
   cache, `kernel_signatures` = 0x18 bytes.
2. **With block disabled** (via `#if 0`): `access_to_subset` passes
   rc=0 on reused cache, `kernel_signatures` = 0x54 bytes.
3. **PCM still reused** without the block: sycl.pcm md5 + mtime
   unchanged across TUs. Verified via `-Rmodule-build` (no
   `"building module 'sycl'"` on second TU).

### Commit message claim vs reality

Commit `d273d7b9fc59` justifies the block as:
> Mark integration headers as transient to prevent cache invalidation

Tested: PCM is **not** invalidated when the int header/footer file is
deleted, even without the transient flag. The flag is redundant and
actively harmful.

### Fix (recommended)

Delete the block entirely.

```cpp
// No setFileIsTransient — not needed for PCM reuse, causes
// declaration-merging bugs that truncate namespace-scope
// static constexpr array initializers.
```

### Upstream direction

Single-file patch to `clang/lib/Frontend/CompilerInstance.cpp`.
Revert the transient block. Ship with a test:

```
// RUN: compile TU-A with -fmodules -fmodules-cache-path=%t
// RUN: compile TU-B with same cache
// CHECK: kernel_signatures symbol has 7 entries in both outputs
```

## Bug 3 — TU-specific `-fsycl-unique-prefix` and builtins (partial)

### Symptom

Separate from Bug 2. `sycl-namespace.cpp` and similar tests fail with
`UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE` when cache is shared
across TUs.

### Root cause

`-fsycl-unique-prefix=uidXXXX` flag feeds `LangOpts.SYCLUniquePrefix`.
Not declared as `LANGOPT` in `clang/include/clang/Basic/LangOptions.def`,
so not hashed into PCM key. Each TU has different prefix but can reuse
same PCM.

Sycl headers use `__builtin_sycl_unique_stable_name(T)` in three places:

- `sycl/include/sycl/detail/kernel_desc.hpp:143` — `KernelInfoImpl<T>::n`
- `sycl/include/sycl/kernel_handler.hpp:65,74`
- `sycl/include/sycl/ext/oneapi/experimental/virtual_functions.hpp` (multiple)

The builtin returns a string that incorporates the unique prefix.
Evaluated at CodeGen time, but `static constexpr auto n =
__builtin_sycl_unique_stable_name(T)` at template member scope means
once a template is instantiated under TU-A's prefix and its
instantiation side effects reach PCM, TU-B reusing that PCM sees
TU-A's prefix.

### Fix (applied, partial)

Marked these files as `textual header` in the generated modulemap via
`sycl/CMakeLists.txt`:

- `kernel_handler.hpp`
- `detail/kernel_desc.hpp`
- `detail/compile_time_kernel_info.hpp`
- `detail/get_device_kernel_info.hpp`
- `detail/kernel_launch_helper.hpp`
- `ext/oneapi/experimental/virtual_functions.hpp`

Textual headers belong to the module but clang re-parses per TU instead
of baking into PCM. Eliminates cross-TU leakage of unique-prefix state.

### Remaining work

After Bug 2 fix (delete `setFileIsTransient` block), most failures
resolve without needing these textual declarations. Verify whether
the textual-header additions are still required or can be reverted.
Minimizing textual-header scope improves compile time.

## Bug 4 — PCM hash misses `SYCLUniquePrefix`

### Symptom

Same as Bug 3 — two TUs with different `-fsycl-unique-prefix` share
one PCM.

### Root cause

`SYCLUniquePrefix` is a plain `std::string` in `LangOptions.h:609`,
declared in `Options.td` via `MarshallingInfoString<LangOpts<...>>`
but NOT via any of the `LANGOPT`/`VALUE_LANGOPT`/`ENUM_LANGOPT` macros
in `LangOptions.def`. Only macros in `.def` contribute to PCM hash.

### Fix direction (not applied)

Either:

**A.** Add as `LANGOPT` → PCM hash includes it → each TU gets unique
PCM. Defeats caching goal (prefix is per-TU by design).

**B.** Decouple unique-prefix from template instantiations. Rewrite
`KernelInfoImpl<T>::n` so `__builtin_sycl_unique_stable_name(T)` is
not evaluated at template-member scope. Possible via accessor
function evaluated at callsite. Invasive.

**C.** Keep textual-header workaround from Bug 3 fix. Pragmatic but
fragile — any new header using the builtin must be added to textual
list.

Recommendation: **C short term, B long term**.

## How SYCL modules work (design reference)

### Flow

1. CMake generates `module.modulemap` at
   `build-dev/include/sycl/module.modulemap` via
   `sycl/CMakeLists.txt:302`. Default = flat umbrella:
   ```
   module sycl {
     header "accessor.hpp"
     ...
     textual header "kernel_handler.hpp"
     textual header "detail/kernel_desc.hpp"
     ...
     export *
   }
   ```
2. User compiles with `-fmodules -fmodules-cache-path=<dir>`.
3. Driver flag `-fmodules` must propagate to cc1 — achieved by
   `Args.hasFlagNoClaim(OPT_fmodules, ...)` in
   `clang/lib/Driver/ToolChains/Clang.cpp` (was `hasFlag`, which
   claimed the arg and dropped it from cc1 line).
4. First TU: clang parses sycl headers, serializes AST as
   `<hash>/sycl-<hash>.pcm`.
5. Next TU with matching flag hash: `mmap`'s the PCM instead of
   re-parsing. ~7x speedup on warm cache.
6. Flag hash depends on langopts marked `NotCompatible` in
   `LangOptions.def`, target triple, include paths, -D macros.

### SYCL-specific flags needed for modules correctness

**`-Rmodule-build`:** diagnostic flag showing cache misses vs hits.
Useful for debugging.

**Module requires fallback for target-specific intrinsics:**
x86 intrinsic modules have `requires x86` clause. Device pass
(`spir64`) doesn't match → module import fails. Our branch's change
to `clang/lib/Lex/PPDirectives.cpp` falls back to textual include
when `requires` unmet, letting the header's own `#ifdef`s handle
absence. Correct behavior.

### Per-file treatment

| File category | Modulemap decl | PCM participation |
|---------------|----------------|-------------------|
| `sycl/*.hpp` top level (safe) | `header "X.hpp"` | Parsed into PCM once |
| `sycl/kernel_handler.hpp`, `sycl/detail/kernel_desc.hpp`, etc. | `textual header "X.hpp"` | Belongs to module but re-parsed per TU |
| `sycl/ext/*`, `sycl/detail/*` (most) | Not declared | Implicitly textual (reparse per TU) |

Textual marking required for files using `__builtin_sycl_*` to avoid
baking TU-specific state into PCM.

### PCM hash key ingredients

- Module name
- Module map content hash
- Language options marked `NotCompatible` in
  `clang/include/clang/Basic/LangOptions.def`
- Target triple
- `-D` macros and `-U` macros
- `-I`/`-isystem` paths affecting module resolution
- Transitive module deps (each imported submodule's hash)

**NOT in hash:** TU filename, `#include` order inside the TU,
ad-hoc `LangOptions` string fields not declared via LANGOPT macros.

## Coverage impact

### E2E test include profile

2625 .cpp files under `sycl/test-e2e`. 1314 include any `sycl/*` header.

| Header | Usage | In modulemap? |
|--------|-------|---------------|
| `sycl/detail/core.hpp` | 1210 (92%) | No — outside top-level `sycl/*.hpp` glob |
| `sycl/usm.hpp` | 413 | Yes |
| `sycl/kernel_bundle.hpp` | 139 | Yes |
| `sycl/properties/all_properties.hpp` | 99 | No |
| `sycl/ext/intel/esimd.hpp` | 81 | No |
| `sycl/sycl.hpp` | 45 | Yes |

Current modulemap globs `sycl/*.hpp` (non-recursive) in
`sycl/CMakeLists.txt:295`. **92% of tests `#include` a header that
is NOT in the module** → textual parse → no module benefit.

### Real speedup scope (with current modulemap)

Only TUs that exclusively include top-level `sycl/*.hpp` benefit.
That's ~4% of E2E tests.

### To widen impact

Broaden glob to `sycl/**/*.hpp`, or add explicit entries for:

- `detail/core.hpp`
- `ext/intel/esimd.hpp`
- `ext/oneapi/bindless_images.hpp`
- `ext/oneapi/free_function_queries.hpp`
- `ext/oneapi/experimental/invoke_simd.hpp`
- `ext/oneapi/experimental/enqueue_functions.hpp`
- `properties/all_properties.hpp`

Captures the 92% cliff. khr migration orthogonal — decouple the two.

## Action items (prioritized)

1. **Delete `setFileIsTransient` block.** One file, ~10 lines. Upstream-safe.
   Unlocks the 137 failing tests.
2. **Verify textual-header additions still needed** after fix 1. If not,
   revert them to recover lost compile-time.
3. **Decide on `SYCLUniquePrefix` hashing.** Either add as LANGOPT
   (correctness, slower) or keep textual-header workaround (faster,
   fragile).
4. **Broaden modulemap scope** to `detail/core.hpp` + top `ext/*` headers.
   Unlocks 92% of E2E test coverage.
5. **Upstream PRs:**
   - Bug 1 fix (aspect enum eager lookup) + test
   - Bug 2 fix (delete transient block) + test
   - Fix or workaround for Bug 3/4 (unique prefix)

## Status snapshot (as of this writing)

Applied locally, not yet upstreamed:

- `clang/lib/CodeGen/CodeGenModule.cpp` — `findSYCLAspectEnumAndTypesEagerly`
- `clang/lib/CodeGen/CodeGenModule.h` — declaration for above
- `clang/lib/Frontend/CompilerInstance.cpp` — `setFileIsTransient` block
  guarded by `#if 0`
- `sycl/CMakeLists.txt` — textual-header list in modulemap generator
- `clang/test/CodeGenSYCL/aspect-enum-from-module.cpp` + Inputs — regression test

Verified working:

- `access_to_subset.cpp` passes fresh + reuse under modules
- `sycl-namespace.cpp` passes fresh + reuse under modules
- PCM (`sycl-<hash>.pcm`) reused correctly across TUs (md5+mtime stable)
- Regression test in `clang/test/CodeGenSYCL/` passes

Not yet verified:

- Full lit Basic/ pass rate (L0 device probe hanging on current
  environment blocks lit startup — environmental, not related to
  module work; use `ONEAPI_DEVICE_SELECTOR='!level_zero:*'` to
  work around).
- Other lit suites (USM, Graph, ESIMD, etc.)
- Compile-time benchmark before/after on representative TU set
