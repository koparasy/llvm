# Lesson 02: specialization_id

This lesson shows the smallest kernel submission that references a
namespace-scope `sycl::specialization_id` through `sycl::kernel_handler`.

When lowered in device mode, the lesson should produce a non-empty integration
footer containing specialization-constant symbolic ID support.

Regenerate with:

```bash
bash tutorial/generate_artifacts.sh Lesson-02
```