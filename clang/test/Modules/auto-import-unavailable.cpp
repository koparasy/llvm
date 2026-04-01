// RUN: rm -rf %t
// RUN: not %clang_cc1 -x c++ -Rmodule-build -DMISSING_HEADER -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/auto-import-unavailable %s 2>&1 | FileCheck %s --check-prefix=MISSING-HEADER
// RUN: %clang_cc1 -x c++ -Rmodule-build -DNONREQUIRED_MISSING_HEADER -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/auto-import-unavailable %s 2>&1 | FileCheck %s --check-prefix=NONREQUIRED-MISSING-HEADER
// RUN: %clang_cc1 -x c++ -Rmodule-build -DMISSING_REQUIREMENT -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/auto-import-unavailable %s 2>&1 | FileCheck %s --check-prefix=MISSING-REQUIREMENT --allow-empty

#ifdef MISSING_HEADER

// Even if the header we ask for is not missing, if the top-level module
// containing it has a missing header, then the whole top-level is
// unavailable and we issue an error.

// MISSING-HEADER: module.modulemap:2:27: error: header 'missing_header/missing.h' not found
// MISSING-HEADER-DAG: auto-import-unavailable.cpp:[[@LINE+1]]:10: note: submodule of top-level module 'missing_header' implicitly imported here
#include "missing_header/not_missing.h"

// We should not attempt to build the module.
// MISSING-HEADER-NOT: remark: building module

#endif // #ifdef MISSING_HEADER


#ifdef NONREQUIRED_MISSING_HEADER

// However, if the missing header is dominated by an unsatisfied
// `requires`, then that is acceptable.
// This also tests that an unsatisfied `requires` elsewhere in the
// top-level module doesn't affect an available module.

// NONREQUIRED-MISSING-HEADER: auto-import-unavailable.cpp:[[@LINE+2]]:10: remark: building module 'nonrequired_missing_header'
// NONREQUIRED-MISSING-HEADER: auto-import-unavailable.cpp:[[@LINE+1]]:10: remark: finished building module 'nonrequired_missing_header'
#include "nonrequired_missing_header/not_missing.h"

#endif // #ifdef NONREQUIRED_MISSING_HEADER


#ifdef MISSING_REQUIREMENT

// An implicit #include of a header whose module has an unmet 'requires' feature
// constraint should fall back to textual inclusion rather than emitting an
// error.  The header's own preprocessor guards take effect, and the module is
// not built.

// MISSING-REQUIREMENT-NOT: error:
// MISSING-REQUIREMENT-NOT: remark: building module
#include "missing_requirement.h"

#endif // #ifdef MISSING_REQUIREMENT
