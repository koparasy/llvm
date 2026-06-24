// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -std=c++17 -verify %s

// Diagnostics for the __builtin_sycl_*_property builtins: the first operand
// must be a function or function pointer, the second must be a string literal,
// and exactly two arguments are required.

void f();
const char *runtime_name = "sycl-nd-range-kernel";

void test() {
  // First operand must be a function / function pointer.
  (void)__builtin_sycl_has_property(5, "sycl-nd-range-kernel");
  // expected-error@-1 {{1st argument must be a function pointer type}}
  (void)__builtin_sycl_get_property(5, "sycl-nd-range-kernel");
  // expected-error@-1 {{1st argument must be a function pointer type}}

  // Second operand must be a string literal, not a runtime value.
  (void)__builtin_sycl_has_property(f, runtime_name);
  // expected-error@-1 {{expression is not a string literal}}
  (void)__builtin_sycl_get_property(f, runtime_name);
  // expected-error@-1 {{expression is not a string literal}}

  // Exactly two arguments.
  (void)__builtin_sycl_has_property(f);
  // expected-error@-1 {{builtin requires exactly 2 arguments}}
  (void)__builtin_sycl_get_property(f, "a", "b");
  // expected-error@-1 {{builtin requires exactly 2 arguments}}
}
