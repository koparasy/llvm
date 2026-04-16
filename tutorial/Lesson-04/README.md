# Lesson 04: host_pipe

This lesson exercises the frontend's host-pipe footer path. The current build
tree does not expose a public `host_pipe` header, so the lesson includes a
small compatibility shim that declares the same kind of frontend-recognized
type used by Clang's own SYCL codegen tests.

Regenerate with:

```bash
bash tutorial/generate_artifacts.sh Lesson-04
```