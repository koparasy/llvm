// End to end: the offload combine proves on the host that the three buffers a
// gemm-like kernel accesses are distinct, and the device kernel's accessor
// pointer parameters get noalias in the final device LLVM IR.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -emit-cir %s -o %t/host.cir
//
// ipo + closed-tu: A, B and C are noalias.
// RUN: cir-offload-merge -combine --remarks-filter="cir-offload.*" -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-facts=ipo \
// RUN:   -cir-offload-assume-closed-tu -output=%t/closed.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REMARK --implicit-check-not=kernel-name-redirect:
// RUN: cir-offload-merge -split -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/closed.cir -output=%t/closed.host.cir -output=%t/closed.dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -O1 -emit-llvm -x cir \
// RUN:   %t/closed.dev.cir -o - | FileCheck %s --check-prefix=CLOSED
//
// ipo without closed-tu: setup() and run() may have callers in other TUs, so
// nothing is proven for the kernel their callers in other TUs launch; the
// path from main is internalized (T26), and the copy of the kernel it
// launches gets A, B and C noalias.
// RUN: cir-offload-merge -combine --remarks-filter="cir-offload.*" -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-facts=ipo \
// RUN:   -output=%t/open.cir 2>&1 | FileCheck %s --check-prefix=OPEN-REMARK \
// RUN:   --implicit-check-not=kernel-name-redirect:
// RUN: cir-offload-merge -split -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/open.cir -output=%t/open.host.cir -output=%t/open.dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -O1 -emit-llvm -x cir \
// RUN:   %t/open.dev.cir -o %t/open.dev.ll
// RUN: FileCheck %s --check-prefix=NONE --input-file=%t/open.dev.ll
// RUN: FileCheck %s --check-prefix=OPEN-COPY --input-file=%t/open.dev.ll
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -emit-llvm -x cir \
// RUN:   %t/open.host.cir -o %t/open.host.ll
// RUN: FileCheck %s --check-prefix=OPEN-HOST --input-file=%t/open.host.ll
//
// -DSAME: B's buffer is a copy of A's (one storage), so A and B may alias;
// only C is noalias.
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-cir -DSAME %s -o %t/same.dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -emit-cir -DSAME %s -o %t/same.host.cir
// RUN: cir-offload-merge -combine --remarks-filter="cir-offload.*" -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/same.host.cir -input=%t/same.dev.cir --cir-offload-host-facts=ipo \
// RUN:   -cir-offload-assume-closed-tu -output=%t/same.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME-REMARK --implicit-check-not=kernel-name-redirect:
// RUN: cir-offload-merge -split -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/same.cir -output=%t/same.host2.cir -output=%t/same.dev2.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -O1 -emit-llvm -x cir \
// RUN:   %t/same.dev2.cir -o - | FileCheck %s --check-prefix=SAME
//
// The SYCL and C++ library parts are reduced to the shapes the combine
// recognizes (the CIRGen host API roles are given by declaration, see
// host-api-roles.cpp): the handler submit path of the real headers
// (queue::submit -> type_erased_cgfo_ty -> handler::parallel_for ->
// StoreLambda -> HostKernel -> setDeviceKernelInfo, with the kernel name in a
// CompileTimeKernelInfo global), a device accessor that is a SYCL special
// class, and a reference-counting std::shared_ptr / std::vector. The
// polybench gemm compile with the real headers and libstdc++ is in the T20
// report.

// REMARK-DAG: kernel-param-noalias: kernel @_ZTS4Gemm arg {{[0-9]+}} (accessor 'A'): noalias
// REMARK-DAG: kernel-param-noalias: kernel @_ZTS4Gemm arg {{[0-9]+}} (accessor 'B'): noalias
// REMARK-DAG: kernel-param-noalias: kernel @_ZTS4Gemm arg {{[0-9]+}} (accessor 'C'): noalias

// CLOSED: define {{.*}}spir_kernel void @_ZTS4Gemm(
// CLOSED-SAME: ptr addrspace(1) noalias {{[^,]*}}%{{[0-9]+}},
// CLOSED-SAME: ptr addrspace(1) noalias {{[^,]*}}%{{[0-9]+}},
// CLOSED-SAME: ptr addrspace(1) noalias {{[^,]*}}%{{[0-9]+}},

// NONE: define {{.*}}spir_kernel void @_ZTS4Gemm(
// NONE-NOT: noalias
// NONE-SAME: {{$}}

// OPEN-COPY: define {{.*}}spir_kernel void @_ZTS4Gemm.int.{{[0-9a-f]+}}(
// OPEN-COPY-SAME: ptr addrspace(1) noalias {{[^,]*}}%{{[0-9]+}},
// OPEN-COPY-SAME: ptr addrspace(1) noalias {{[^,]*}}%{{[0-9]+}},
// OPEN-COPY-SAME: ptr addrspace(1) noalias {{[^,]*}}%{{[0-9]+}},

// OPEN-REMARK: launch-path-internalization: kernel @_ZTS4Gemm -> @_ZTS4Gemm.int.{{[0-9a-f]+}}

// The handler's kernel name (handler::finalize() asserts that it is the
// name of the DeviceKernelInfo) on the internalized path is the kernel
// copy's: the internal copy of setDeviceKernelInfo<Gemm> copies it from a
// copy of its `__const` Info that names the copy's string, with its length.
// The original keeps the original's.
// OPEN-HOST-DAG: @__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4GemmEEvPv.Info = private constant {{.*}} { ptr @.str, i64 9 } }
// OPEN-HOST-DAG: @__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4GemmEEvPv.Info.int = internal constant {{.*}} { ptr @.str.int, i64 30 } }
// OPEN-HOST-DAG: @.str = private constant [10 x i8] c"_ZTS4Gemm\00"
// OPEN-HOST-DAG: @.str.int = internal constant [31 x i8] c"_ZTS4Gemm.int.{{[0-9a-f]+}}\00"
// OPEN-HOST-LABEL: define linkonce_odr void @_ZN4sycl3_V17handler19setDeviceKernelInfoI4GemmEEvPv(
// OPEN-HOST: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}@__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4GemmEEvPv.Info, i64 16
// OPEN-HOST: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4GemmEERNS1_16DeviceKernelInfoEv()
// OPEN-HOST-LABEL: define internal void @_ZN4sycl3_V17handler19setDeviceKernelInfoI4GemmEEvPv.int(
// OPEN-HOST: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}@__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4GemmEEvPv.Info.int, i64 16
// OPEN-HOST: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4GemmEERNS1_16DeviceKernelInfoEv.int()

// SAME-REMARK-DAG: kernel-param-noalias: kernel @_ZTS4Gemm arg {{[0-9]+}} (accessor 'A'): not stamped ('A' and 'B' use the same memory{{.*}})
// SAME-REMARK-DAG: kernel-param-noalias: kernel @_ZTS4Gemm arg {{[0-9]+}} (accessor 'B'): not stamped ('B' and 'A' use the same memory{{.*}})
// SAME-REMARK-DAG: kernel-param-noalias: kernel @_ZTS4Gemm arg {{[0-9]+}} (accessor 'C'): noalias

// The kernel captures C, A, B in this order (first use).
// SAME: define {{.*}}spir_kernel void @_ZTS4Gemm(
// SAME-SAME: ptr addrspace(1) noalias {{[^,]*}}%{{[0-9]+}},
// SAME-NOT: noalias
// SAME-SAME: {{$}}

#define __SYCL_TYPE(x) [[__sycl_detail__::sycl_type(x)]]

namespace std {
template <typename T> T &&move(T &t) { return static_cast<T &&>(t); }

template <typename T> class vector {
  T *_M_start = nullptr;
  T *_M_finish = nullptr;

public:
  void resize(unsigned long n) {
    T *p = new T[n];
    delete[] _M_start;
    _M_start = p;
    _M_finish = p + n;
  }
  T *data() { return _M_start; }
  ~vector() { delete[] _M_start; }
};

template <typename T> class shared_ptr {
  T *_M_ptr = nullptr;
  long *_M_count = nullptr;

  void release() {
    if (_M_count && --*_M_count == 0) {
      delete _M_ptr;
      delete _M_count;
    }
  }

public:
  shared_ptr() = default;
  explicit shared_ptr(T *p) : _M_ptr(p), _M_count(new long(1)) {}
  shared_ptr(const shared_ptr &o) : _M_ptr(o._M_ptr), _M_count(o._M_count) {
    if (_M_count)
      ++*_M_count;
  }
  shared_ptr &operator=(const shared_ptr &o) {
    if (o._M_count)
      ++*o._M_count;
    release();
    _M_ptr = o._M_ptr;
    _M_count = o._M_count;
    return *this;
  }
  ~shared_ptr() { release(); }
  T *operator->() const { return _M_ptr; }
  T &operator*() const { return *_M_ptr; }
};

template <typename T, typename A, typename B>
shared_ptr<T> make_shared(A a, B b) {
  return shared_ptr<T>(new T(a, b));
}
} // namespace std

namespace sycl {
inline namespace _V1 {
namespace access {
enum class target { device = 2014 };
enum class mode { read = 1024, write, read_write };
enum class placeholder { false_t };
} // namespace access

template <int D> struct range {
  unsigned long v[D];
  range() { v[0] = 0; }
  range(unsigned long a) { v[0] = a; }
  unsigned long operator[](int i) const { return v[i]; }
};
template <int D> struct id {
  unsigned long v[D];
  id() { v[0] = 0; }
  unsigned long operator[](int i) const { return v[i]; }
};

class handler;
class queue;
template <typename T, int D> class buffer;

template <int D> struct AccImpl {
  range<D> AccessRange;
  range<D> MemRange;
  id<D> Offset;
};

template <typename T, int D, access::mode M,
          access::target Tgt = access::target::device,
          access::placeholder P = access::placeholder::false_t>
class __attribute__((sycl_special_class)) __SYCL_TYPE(accessor) accessor {
#ifdef __SYCL_DEVICE_ONLY__
  __attribute__((opencl_global)) T *MData;
  AccImpl<D> impl;
  void __init(__attribute__((opencl_global)) T *Ptr, range<D> AccessRange,
              range<D> MemRange, id<D> Offset) {
    MData = Ptr;
    impl.AccessRange = AccessRange;
    impl.MemRange = MemRange;
    impl.Offset = Offset;
  }

public:
  accessor() = default;
  T &operator[](id<D> i) const { return MData[i[0]]; }
  T &operator[](unsigned long i) const { return MData[i]; }
#else
  void *impl;

public:
  accessor(buffer<T, D> &b, handler &h);
  accessor(const accessor &other);
  ~accessor();
  T &operator[](id<D> i) const;
  T &operator[](unsigned long i) const;
#endif
};

template <typename T, int D> class buffer {
  void *impl;
  unsigned long n;

public:
  buffer(T *hostData, const range<D> &r);
  buffer(const buffer &other);
  ~buffer();
  template <access::mode M> accessor<T, D, M> get_access(handler &h);
  void set_write_back(bool flag = true);
};

namespace detail {
struct string_view {
  const char *str;
  unsigned long len;
};
struct CompileTimeKernelInfoTy {
  string_view Name;
};
template <class K>
inline constexpr CompileTimeKernelInfoTy CompileTimeKernelInfo{
    {__builtin_sycl_unique_stable_name(K),
     __builtin_strlen(__builtin_sycl_unique_stable_name(K))}};
struct DeviceKernelInfo {};
DeviceKernelInfo &getDeviceKernelInfo(const CompileTimeKernelInfoTy &);
template <class K> DeviceKernelInfo &getDeviceKernelInfo() {
  static DeviceKernelInfo &Info = getDeviceKernelInfo(CompileTimeKernelInfo<K>);
  return Info;
}
inline namespace nd_range_view_v1 {
class nd_range_view {
public:
  template <int D> nd_range_view(range<D> &r) : g(r.v), dims(D) {}
  const unsigned long *g;
  int dims;
};
} // namespace nd_range_view_v1
class HostKernelBase {
public:
  virtual ~HostKernelBase() = default;
};
template <class K> class HostKernel : public HostKernelBase {
  K MKernel;

public:
  HostKernel(const K &k) : MKernel(k) {}
};
class type_erased_cgfo_ty {
  template <typename T> struct invoker {
    static void call(const void *o, handler &h) { (*(const T *)o)(h); }
  };
  const void *object;
  void (*invoker_f)(const void *, handler &);

public:
  template <class T>
  type_erased_cgfo_ty(T &f) : object(&f), invoker_f(&invoker<T>::call) {}
};
struct KernelWrapperHelperFuncs {
#ifdef __SYCL_DEVICE_ONLY__
  template <typename Kn, typename K>
  [[clang::sycl_kernel]] static void kernel_parallel_for(const K &k) {
    k(id<1>());
  }
#else
  template <typename Kn, typename K>
  static void kernel_parallel_for(const K &) {}
#endif
};
} // namespace detail

class handler {
  void setKernelFunc(void *p);
  void setDeviceKernelInfoPtr(detail::DeviceKernelInfo *p);
  void setNDRangeDescriptor(detail::nd_range_view rv);
  // The real headers' shape: the handler's kernel name comes from a
  // constexpr copy of the CompileTimeKernelInfo (a `__const.<f>.Info`
  // global in CIR), and handler::finalize() asserts that it is the name of
  // the DeviceKernelInfo.
  detail::string_view MKernelName;
  template <class Kn> void setDeviceKernelInfo(void *p) {
    constexpr auto Info = detail::CompileTimeKernelInfo<Kn>;
    MKernelName = Info.Name;
    setKernelFunc(p);
    setDeviceKernelInfoPtr(&detail::getDeviceKernelInfo<Kn>());
  }
  template <class Kn, class K> void StoreLambda(const K &k) {
    setDeviceKernelInfo<Kn>(new detail::HostKernel<K>(k));
  }

public:
  template <class Kn, class K> void parallel_for(range<1> r, const K &k) {
    detail::KernelWrapperHelperFuncs::kernel_parallel_for<Kn>(k);
    setNDRangeDescriptor(detail::nd_range_view(r));
    StoreLambda<Kn>(k);
  }
};

class event {};
class queue {
  event submit_with_event_impl(const detail::type_erased_cgfo_ty &cgf) const;

public:
  template <typename T> event submit(T CGF) {
    return submit_with_event_impl(detail::type_erased_cgfo_ty{CGF});
  }
};
} // namespace _V1
} // namespace sycl

class Gemm;

class Bench {
public:
  explicit Bench(sycl::queue &q, unsigned long n) : q(q), n(n) {}

  void setup() {
    A.resize(n * n);
    B.resize(n * n);
    C.resize(n * n);
    A_buffer = std::make_shared<sycl::buffer<float, 1>>(A.data(),
                                                        sycl::range<1>(n * n));
#ifdef SAME
    B_buffer = A_buffer;
#else
    B_buffer = std::make_shared<sycl::buffer<float, 1>>(B.data(),
                                                        sycl::range<1>(n * n));
#endif
    C_buffer = std::make_shared<sycl::buffer<float, 1>>(C.data(),
                                                        sycl::range<1>(n * n));
    A_buffer->set_write_back(false);
  }

  void run() {
    using namespace sycl;
    q.submit([&](handler &h) {
      auto A = A_buffer->get_access<access::mode::read>(h);
      auto B = B_buffer->get_access<access::mode::read>(h);
      auto C = C_buffer->get_access<access::mode::read_write>(h);
      unsigned long nk = n;
      h.parallel_for<Gemm>(range<1>(nk * nk), [=](id<1> it) {
        unsigned long i = it[0] / nk, j = it[0] % nk;
        for (unsigned long k = 0; k < nk; ++k)
          C[it] += A[i * nk + k] * B[k * nk + j];
      });
    });
  }

private:
  sycl::queue &q;
  unsigned long n;
  std::vector<float> A, B, C;
  std::shared_ptr<sycl::buffer<float, 1>> A_buffer, B_buffer, C_buffer;
};

int main() {
  sycl::queue q;
  Bench bench(q, 64);
  bench.setup();
  bench.run();
  return 0;
}
