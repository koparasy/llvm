// Known soundness gap N13 (closed-tu, for T18): TWIN_INDIRECT of t11-devirt.cpp
// under closed-tu. The Local comes from the external part, so the vptr load
// gives an External callee node and b->f() turns indirect-unresolved; but
// Local::f was built with complete formals (closed-tu), its this points to an
// empty node, and *out and *in on the PAIR line come out NoAlias
// (distinct-objects) although in == out at run time: PT-VIOLATION. When T18
// makes formals of functions whose node is External incomplete, this passes;
// then drop the XFAIL and move the RUN lines into t11-devirt.cpp.
//
// XFAIL: *
// RUN: env CXXFLAGS=-DTWIN_INDIRECT %pt_run %S/t11-devirt.cpp %t.indirect %pt_opts closed-tu=1 2>&1 | FileCheck %S/t11-devirt.cpp --check-prefix=EXEC
// RUN: FileCheck %S/t11-devirt.cpp --check-prefix=TWIN --input-file=%t.indirect.table
