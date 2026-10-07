/* Runtime of -test-cir-points-to-instrument: the overlap check every
   instrumented NoAlias pair calls. Two ranges [p, p + sa) and [q, q + sb)
   that overlap mean the analysis claimed NoAlias for two accesses that touch
   the same byte: print "PT-VIOLATION <id>" and abort. At exit, print
   "PT-CHECKS <n>", the number of checks that passed. A null pointer or an
   empty range overlaps nothing. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long pt_checks;

void __cir_pt_check(const void *p, uint64_t sa, const void *q, uint64_t sb,
                    uint64_t id) {
  uintptr_t a = (uintptr_t)p, b = (uintptr_t)q;
  if (a && b && sa && sb && (a <= b ? b - a < sa : a - b < sb)) {
    fprintf(stderr, "PT-VIOLATION %llu\n", (unsigned long long)id);
    fflush(stderr);
    abort();
  }
  ++pt_checks;
}

__attribute__((destructor)) static void pt_report(void) {
  fprintf(stderr, "PT-CHECKS %llu\n", pt_checks);
}
