# SYCL Frontend Tutorial

This tutorial is organized as small, focused lessons under numbered
subdirectories. Each lesson contains one source example and is intended to be
lowered separately so you can inspect the generated host AST, device AST,
integration header/footer, and LLVM IR.

## Lessons

- `Lesson-01`: a minimal `queue.submit` plus `single_task` example that shows
  the basic host/device split and the generated integration header.
- `Lesson-02`: a `specialization_id` example that makes the device integration
  footer non-empty by emitting specialization-constant symbol mapping code.
- `Lesson-03`: a `device_global` example that makes the footer emit device
  global registration code.
- `Lesson-04`: a host-pipe example that makes the footer emit host-pipe
  registration code. This lesson uses a tiny local compatibility shim because
  `host_pipe` is not exposed as a public header API in this build tree.

## Layout

- `Lesson-XX/*.cpp`: tutorial source for the lesson
- `Lesson-XX/README.md`: short lesson-specific notes
- `Lesson-XX/out/`: generated artifacts after running the helper script
- `compile_flags.txt`: common editor flags for the tutorial sources
- `generate_artifacts.sh`: helper script to regenerate artifacts for a lesson

## Regenerate Artifacts

Run from the repository root:

```bash
bash tutorial/generate_artifacts.sh Lesson-01
bash tutorial/generate_artifacts.sh Lesson-02
bash tutorial/generate_artifacts.sh Lesson-03
bash tutorial/generate_artifacts.sh Lesson-04
```

The script auto-detects the lesson source file and writes outputs into that
lesson's `out/` directory.# SYCL Frontend Tutorial: 01 Simple Submit

This directory starts with the smallest useful SYCL submission example in
`01_simple_submit.cpp` and splits it into:

- a device-side AST and LLVM IR view
- a host-side AST and LLVM IR view
- the generated SYCL integration header

The point of the exercise is to see what Clang's SYCL frontend keeps as normal
C++ on the host side and what it synthesizes for the device side.

## Files

- `01_simple_submit.cpp`: source example
- `out/01_simple_submit.device.filtered.ast.txt`: focused device AST dump
- `out/01_simple_submit.host.filtered.ast.txt`: focused host AST dump
- `out/01_simple_submit.device.ll`: device LLVM IR
- `out/01_simple_submit.host.ll`: host LLVM IR
- `out/01_simple_submit.hpp`: generated integration header

## Regenerate

From the repository root:

```bash
bash playground/sycl_frontend_tutorial/generate_artifacts.sh
```

The script assumes the `llvm-dev` conda environment exists and uses the local
compiler in `build-orig/bin/clang++`.

## What To Look At

### 1. The source stays small

The example is intentionally minimal:

```c++
void submit_increment(sycl::queue &queue, int *data) {
  queue.submit([&](sycl::handler &handler) {
    handler.single_task<class SimpleIncrement>([=]() {
      data[0] += 1;
    });
  });
}
```

There are two nested lambdas:

- the command-group lambda passed to `queue.submit`
- the kernel lambda passed to `handler.single_task`

That distinction drives most of the frontend lowering.

### 2. Device AST: the frontend manufactures a kernel entry

Search `out/01_simple_submit.device.filtered.ast.txt` for these two items:

- `FunctionDecl ... submit_increment`
- `FunctionDecl ... device_kernel`

What happens:

- The original nested kernel lambda is still visible as a lambda closure with a
  captured `data` field.
- Clang synthesizes a new function with `DeviceKernelAttr`, `SYCLKernelAttr`,
  and an asm label based on the stable kernel name.
- That synthetic function is the actual device entry point. It is not written
  in the source, but it is what later becomes the SPIR kernel in LLVM IR.

### 3. Integration header: host and device meet here

Open `out/01_simple_submit.hpp`.

This file contains:

- the stable kernel name string
- the parameter count
- the kernel parameter descriptor array
- the file, line, column, and kernel object size

For this example, the integration header says the kernel has one parameter:
the captured pointer `data`.

That header is exactly how the host compilation learns the device kernel's
identity and ABI without parsing the device AST again.

### 4. Device LLVM IR: the kernel entry rebuilds the lambda object

Open `out/01_simple_submit.device.ll` and find:

- `define weak_odr dso_local spir_kernel void`
- `define internal spir_func void ... ENKUlvE_clEv`

What to notice:

- The synthetic kernel entry is emitted as `spir_kernel`.
- Its argument is `_arg_data` in address space 1, because the captured pointer
  is a kernel parameter.
- The entry allocates a temporary closure object `__SYCLKernel`.
- It stores the incoming kernel argument into the closure field.
- It then calls the original kernel lambda body, now lowered as an internal
  helper function.

So the device side is effectively:

1. receive raw kernel arguments
2. reconstruct the lambda closure object
3. invoke the lambda's `operator()`

### 5. Host AST: `submit` and `single_task` are still ordinary C++ calls

Open `out/01_simple_submit.host.filtered.ast.txt`.

The host AST still shows:

- a call to `queue.submit`
- a nested lambda with `operator()(sycl::handler &)`
- a call to `handler.single_task`
- the kernel lambda with its captured `data`

That is an important split:

- the host frontend does not turn the source into a device kernel directly
- it keeps the submission path as host C++ and threads kernel metadata through
  the handler/runtime layer

### 6. Host LLVM IR: the submission path becomes runtime setup

Useful functions in `out/01_simple_submit.host.ll`:

- `_Z16submit_increment...`
- `queue::submit...`
- `handler::single_task...`
- `handler::wrap_kernel...`
- `handler::StoreLambda...`
- `handler::setDeviceKernelInfo<SimpleIncrement>...`

Read them in that order.

What they show:

- `submit_increment` builds the command-group lambda and calls `queue::submit`.
- The command-group lambda body reconstructs the kernel lambda object and calls
  `handler.single_task<SimpleIncrement>`.
- `handler::single_task` immediately forwards into `wrap_kernel` with a
  one-element `range<1>{1}`. That is the single-task execution shape.
- `wrap_kernel` copies compile-time kernel info and performs checks such as
  kernel-parameter misuse validation.
- `StoreLambda` allocates a `HostKernel<...>` object, stores the kernel lambda
  inside it, and then calls `setDeviceKernelInfo<SimpleIncrement>`.
- `setDeviceKernelInfo` copies the kernel name and hooks the handler up to the
  device-kernel metadata built from the integration header.

So the host-side frontend job is not to emit a device entry. Its job is to:

1. preserve the C++ submission flow
2. materialize host-callable wrappers for the lambdas
3. attach stable kernel metadata to the handler
4. prepare runtime state that will later launch the compiled device image

## First Takeaway

For this tiny program, the SYCL frontend splits one source construct into two
different stories:

- Device compilation creates a synthetic kernel entry around the kernel lambda.
- Host compilation keeps the submission code as ordinary C++ and threads kernel
  identity plus parameter metadata through the handler machinery.

That host/device split is the core pattern to keep in mind for the next
examples.