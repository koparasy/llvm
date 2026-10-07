// T7 (obj.a and obj.b set by two factories in setup(), read back in run()
// through this) and its twin negatives (plan V3). Positive, under closed-tu:
// the accesses on the PAIR line in Obj::run are NoAlias (distinct-objects).
// Open, Obj::run is externally callable and the pair is not NoAlias. The twins
// run under closed-tu (but OPEN), change the program so that the pair must not
// be NoAlias, and every run (all NoAlias pairs checked) must stay clean:
//   TWIN_ASSIGN    b = a in setup()
//   TWIN_COPY      b = new Buf(*a): a shallow copy shares a->data
//   TWIN_ESCAPE    ext_alias(&o), defined in the external part, sets
//                  b->data = a->data
//   TWIN_REASSIGN  o.b->data = o.a->data between setup() and run()
//   TWIN_INDIRECT  ext_alias called through a function pointer from the
//                  external part
//   TWIN_OPEN      closed-tu off, and an external caller runs an Obj whose a
//                  and b are one Buf
//
// RUN: %pt_run %s %t.pos %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.pos.table
// RUN: env CXXFLAGS=-DTWIN_ASSIGN %pt_run %s %t.assign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.assign.table
// RUN: env CXXFLAGS=-DTWIN_COPY %pt_run %s %t.copy %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.copy.table
// RUN: env CXXFLAGS=-DTWIN_ESCAPE %pt_run %s %t.escape %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.escape.table
// RUN: env CXXFLAGS=-DTWIN_REASSIGN %pt_run %s %t.reassign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.reassign.table
// RUN: env CXXFLAGS=-DTWIN_INDIRECT %pt_run %s %t.indirect %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.indirect.table
// RUN: env CXXFLAGS=-DTWIN_OPEN %pt_run %s %t.open %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.open.table
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

struct Buf {
  float *data;
  int n;
};
struct Obj {
  Buf *a = nullptr, *b = nullptr;
  void setup();
  float run();
};
extern "C" void ext_alias(Obj *o);
using AliasFn = void (*)(Obj *);
extern "C" AliasFn ext_get_alias();
extern "C" int ext_call_run();

#ifdef PT_EXTERNAL
extern "C" void ext_alias(Obj *o) { o->b->data = o->a->data; }
extern "C" AliasFn ext_get_alias() { return ext_alias; }
#ifdef TWIN_OPEN
extern "C" int ext_call_run() {
  float d[16] = {};
  Buf one{d, 16};
  Obj o;
  o.a = o.b = &one;
  return o.run() == 1.0f ? 0 : 1;
}
#endif
#else
static Buf *make_a(int n) {
  Buf *b = new Buf;
  b->data = new float[n];
  b->n = n;
  for (int i = 0; i < n; ++i)
    b->data[i] = i;
  return b;
}
static Buf *make_b(int n) {
  Buf *b = new Buf;
  b->data = new float[n]();
  b->n = n;
  return b;
}

void Obj::setup() {
  a = make_a(16);
#if defined(TWIN_ASSIGN)
  b = a;
#elif defined(TWIN_COPY)
  // A shallow copy: b->data is a->data.
  b = new Buf(*a);
#else
  b = make_b(16);
#endif
}

float Obj::run() {
  float *x = a->data, *y = b->data;
  for (int i = 0; i < a->n; ++i)
    y[i] = x[i] + 1; // PAIR
    // POS: {{^[0-9]+}} @{{[^ ]+}} distinct-objects {{[a-z.]+}}(cir.{{[a-z_]+}}@t7-members.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t7-members.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} at
    // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t7-members.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t7-members.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return y[15];
}

int main() {
  Obj o;
  o.setup();
  float *keep = o.b->data;
#ifdef TWIN_ESCAPE
  ext_alias(&o);
#endif
#ifdef TWIN_REASSIGN
  o.b->data = o.a->data;
#endif
#ifdef TWIN_INDIRECT
  ext_get_alias()(&o);
#endif
  int r = o.run() >= 16.0f ? 0 : 1;
#ifdef TWIN_OPEN
  r += ext_call_run();
#endif
  if (keep != o.a->data)
    delete[] keep;
  delete[] o.a->data;
  if (o.b != o.a)
    delete o.b;
  delete o.a;
  return r;
}
#endif
