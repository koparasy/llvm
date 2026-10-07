// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir -DLAUNCH %s -o %t.cir
// RUN: FileCheck %s --check-prefix=LAUNCH --input-file=%t.cir
// RUN: grep sycl_host_api %t.cir | FileCheck %s --check-prefix=UNTAGGED

// On the SYCL host, CIRGen tags the buffer and accessor API (and
// std::vector::data) with the role the offload combine traces buffer identity
// through, recognized by declaration: cir.sycl_host_api, plus the
// source-level index of the buffer/host-pointer parameter the role reads.
// With -DLAUNCH: the queue/handler kernel launch path, geometry and USM API.

#ifndef LAUNCH

namespace std {
template <typename T> struct vector {
  T *data();
  T *begin();
};
} // namespace std

namespace sycl {
inline namespace _V1 {
class handler;
template <int D> struct range { unsigned long v[D]; };
template <int D> struct id { unsigned long v[D]; };
template <typename T, int D> class accessor;
template <typename T, int D> class buffer {
public:
  buffer(const range<D> &r);
  buffer(T *hostData, const range<D> &r);
  buffer(buffer &parent, const id<D> &offset, const range<D> &r);
  buffer(const buffer &other);
  buffer(int fromSomethingElse);
  buffer &operator=(const buffer &other);
  template <typename U> buffer<U, D> reinterpret(range<D> r) const;
  accessor<T, D> get_access(handler &h);
  range<D> get_range() const;
  ~buffer();
};
template <typename T, int D> class accessor {
public:
  accessor(buffer<T, D> &b, handler &h);
  accessor(handler &h, buffer<T, D> &b, int tag);
  accessor(const accessor &other);
  accessor();
  accessor &operator=(const accessor &other);
  T &operator[](unsigned long i) const;
  ~accessor();
};
template <typename T, int D> class host_accessor : public accessor<T, D> {
public:
  host_accessor(buffer<T, D> &b);
  ~host_accessor();
};
template <typename T, int D> class local_accessor {
public:
  local_accessor(const range<D> &r, handler &h);
  ~local_accessor();
};
} // namespace _V1
} // namespace sycl

namespace other {
template <typename T, int D> class buffer {
public:
  buffer(const sycl::range<D> &r);
  ~buffer();
};
} // namespace other

// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EEC1ERKNS0_5rangeILi1EEE({{.*}} attributes {cir.sycl_host_api = "buffer.alloc", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EEC1EPfRKNS0_5rangeILi1EEE({{.*}} attributes {cir.sycl_host_api = "buffer.host_ptr", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EEC1ERS2_RKNS0_2idILi1EEERKNS0_5rangeILi1EEE({{.*}} attributes {cir.sycl_host_api = "buffer.sub", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EEC1ERKS2_({{.*}} attributes {cir.sycl_host_api = "buffer.copy", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EEC1Ei({{.*}} attributes {cir.sycl_host_api = "buffer.other", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EEaSERKS2_({{.*}} attributes {cir.sycl_host_api = "buffer.assign", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZNK4sycl3_V16bufferIfLi1EE11reinterpretIiEENS1_IT_Li1EEENS0_5rangeILi1EEE({{.*}} attributes {cir.sycl_host_api = "buffer.reinterpret", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EE10get_accessERNS0_7handlerE({{.*}} attributes {cir.sycl_host_api = "buffer.get_access", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V18accessorIfLi1EEC1ERNS0_6bufferIfLi1EEERNS0_7handlerE({{.*}} attributes {cir.sycl_host_api = "accessor.from_buffer", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V18accessorIfLi1EEC1ERNS0_7handlerERNS0_6bufferIfLi1EEEi({{.*}} attributes {cir.sycl_host_api = "accessor.from_buffer", cir.sycl_host_api_arg = 1 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V18accessorIfLi1EEC1ERKS2_({{.*}} attributes {cir.sycl_host_api = "accessor.copy", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V18accessorIfLi1EEC1Ev({{.*}} attributes {cir.sycl_host_api = "accessor.other", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V18accessorIfLi1EEaSERKS2_({{.*}} attributes {cir.sycl_host_api = "accessor.assign", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZNSt6vectorIfE4dataEv({{.*}} attributes {cir.sycl_host_api = "vector.data", cir.sycl_host_api_arg = 0 : i32

// Neither another class's methods nor a same-named class elsewhere are tagged.
// CHECK-DAG: cir.func {{.*}}@_ZNSt6vectorIfE5beginEv(
// CHECK-NOT: @_ZNSt6vectorIfE5beginEv(
// CHECK-DAG: cir.func {{.*}}@_ZNK4sycl3_V16bufferIfLi1EE9get_rangeEv(
// CHECK-NOT: @_ZNK4sycl3_V16bufferIfLi1EE9get_rangeEv(
// CHECK-DAG: cir.func {{.*}}@_ZN5other6bufferIfLi1EEC1ERKN4sycl3_V15rangeILi1EEE(
// CHECK-NOT: @_ZN5other6bufferIfLi1EEC1ERKN4sycl3_V15rangeILi1EEE(
// The destructors are declared last; another class's is not tagged.
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V16bufferIfLi1EED1Ev({{.*}} attributes {cir.sycl_host_api = "buffer.dtor", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V18accessorIfLi1EED1Ev({{.*}} attributes {cir.sycl_host_api = "accessor.dtor", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V113host_accessorIfLi1EED1Ev({{.*}} attributes {cir.sycl_host_api = "accessor.dtor", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN4sycl3_V114local_accessorIfLi1EED1Ev({{.*}} attributes {cir.sycl_host_api = "accessor.dtor", cir.sycl_host_api_arg = 0 : i32
// CHECK-DAG: cir.func {{.*}}@_ZN5other6bufferIfLi1EED1Ev(
// CHECK-NOT: @_ZN5other6bufferIfLi1EED1Ev(
void use(sycl::handler &h, std::vector<float> &v, int n) {
  sycl::range<1> r{{4}};
  sycl::id<1> o{{0}};
  sycl::buffer<float, 1> a(r);
  sycl::buffer<float, 1> b(v.data(), r);
  sycl::buffer<float, 1> s(a, o, r);
  sycl::buffer<float, 1> c(a);
  sycl::buffer<float, 1> d(n);
  d = a;
  auto ri = a.reinterpret<int>(r);
  auto x = a.get_access(h);
  sycl::accessor<float, 1> y(b, h);
  sycl::accessor<float, 1> z(h, b, 0);
  sycl::accessor<float, 1> w(x);
  sycl::accessor<float, 1> e;
  e = y;
  sycl::host_accessor<float, 1> ha(b);
  sycl::local_accessor<float, 1> la(r, h);
  (void)v.begin();
  (void)a.get_range();
  other::buffer<float, 1> ob(r);
}

#else // LAUNCH

// A mock of the launch path with the shape of the real headers: the
// queue::submit -> type_erased_cgfo_ty -> handler::parallel_for -> StoreLambda
// -> HostKernel -> setDeviceKernelInfo chain, and the queue shortcut
// queue::parallel_for -> detail::submit_kernel_direct_parallel_for ->
// HostKernelRef -> submit_kernel_direct_with_event_impl.
namespace std {
template <typename T> struct shared_ptr {
  T *p;
};
} // namespace std

namespace sycl {
inline namespace _V1 {
class handler;
class kernel {};
class event {
public:
  void wait();
};
template <typename T, int D> class accessor {
public:
  accessor(const accessor &other);
  ~accessor();
};
namespace ext::oneapi::experimental {
template <typename T> class device_global {};
} // namespace ext::oneapi::experimental
template <int D> struct range {
  unsigned long v[D];
  range(unsigned long a) { v[0] = a; }
};
template <int D> struct id {
  unsigned long v[D];
  id() { v[0] = 0; }
};
template <int D> struct nd_range {
  range<D> g, l;
  nd_range(range<D> g, range<D> l) : g(g), l(l) {}
};
class queue;
namespace detail {
struct DeviceKernelInfo {};
struct CompileTimeKernelInfoTy {};
inline namespace nd_range_view_v1 {
class nd_range_view {
public:
  template <int D> nd_range_view(range<D> &r) : g(r.v) {}
  const unsigned long *g;
};
} // namespace nd_range_view_v1
DeviceKernelInfo &getDeviceKernelInfo(const CompileTimeKernelInfoTy &);
template <class K> DeviceKernelInfo &getDeviceKernelInfo() {
  static CompileTimeKernelInfoTy i;
  return getDeviceKernelInfo(i);
}
class HostKernelBase {
public:
  virtual ~HostKernelBase() = default;
};
template <class K> class HostKernel : public HostKernelBase {
  K MKernel;

public:
  HostKernel(const K &k) : MKernel(k) {}
};
class HostKernelRefBase : public HostKernelBase {};
template <class K> class HostKernelRef : public HostKernelRefBase {
  const K &MKernel;

public:
  HostKernelRef(const K &k) : MKernel(k) {}
};
class type_erased_cgfo_ty {
  template <typename T> struct invoker {
    static void call(const void *o, handler &h) { (*(T *)o)(h); }
  };
  const void *object;
  void (*invoker_f)(const void *, handler &);

public:
  template <class T>
  type_erased_cgfo_ty(T &&f)
      : object(&f), invoker_f(&invoker<__remove_reference_t(T)>::call) {}
};
} // namespace detail
event submit_kernel_direct_with_event_impl(const queue &q,
                                           const detail::nd_range_view &rv,
                                           detail::HostKernelRefBase &hk,
                                           detail::DeviceKernelInfo *dki);
namespace detail {
struct KernelWrapperHelperFuncs {
  template <typename Kn, typename K> static void kernel_single_task(const K &k) {
    (void)k;
  }
};
template <typename K, int D>
void submit_kernel_direct_parallel_for(const queue &q, range<D> r, K &&k) {
  HostKernelRef<__remove_reference_t(K)> hk(k);
  nd_range_view rv(r);
  submit_kernel_direct_with_event_impl(q, rv, hk, nullptr);
}
} // namespace detail
class handler {
  void setKernelFunc(void *p);
  void setDeviceKernelInfoPtr(detail::DeviceKernelInfo *p);
  void setNDRangeDescriptor(detail::nd_range_view rv);
  template <class Kn> void setDeviceKernelInfo(void *p) {
    setKernelFunc(p);
    setDeviceKernelInfoPtr(&detail::getDeviceKernelInfo<Kn>());
  }
  template <class Kn, class K> void StoreLambda(const K &k) {
    setDeviceKernelInfo<Kn>(new detail::HostKernel<K>(k));
  }

  template <class Kn, class K>
  void parallel_for_lambda_impl(range<1> r, const K &k) {
    setNDRangeDescriptor(detail::nd_range_view(r));
    StoreLambda<Kn>(k);
  }

public:
  template <class Kn, class K> void parallel_for(range<1> r, const K &k) {
    parallel_for_lambda_impl<Kn>(r, k);
  }
  template <class Kn, class K, int D>
  void parallel_for(nd_range<D> r, const K &k) {
    setNDRangeDescriptor(detail::nd_range_view(r.g));
    StoreLambda<Kn>(k);
  }
  template <class Kn, class K> void single_task(const K &k) {
    detail::KernelWrapperHelperFuncs::kernel_single_task<Kn>(k);
    StoreLambda<Kn>(k);
  }
  void parallel_for(range<1> r, kernel k);

  // Explicit memory commands.
  void memcpy(void *d, const void *s, unsigned long n);
  template <typename T>
  void memcpy(ext::oneapi::experimental::device_global<T> &d, const void *s,
              unsigned long n);
  void memset(void *p, int v, unsigned long n);
  template <typename T> void fill(void *p, const T &pattern, unsigned long n);
  template <typename T> void fill(accessor<T, 1> d, const T &pattern);
  template <typename T> void copy(const T *s, T *d, unsigned long n);
  template <typename T> void copy(accessor<T, 1> s, T *d);
  template <typename T> void copy(const T *s, accessor<T, 1> d);
  template <typename T> void copy(accessor<T, 1> s, accessor<T, 1> d);
  template <typename T> void copy(accessor<T, 1> s, std::shared_ptr<T> d);
};
class queue {
  event submit_with_event_impl(const detail::type_erased_cgfo_ty &cgf) const;

public:
  template <typename T> event submit(T CGF) {
    return submit_with_event_impl(detail::type_erased_cgfo_ty{CGF});
  }
  template <class Kn, class K> event parallel_for(range<1> r, const K &k) {
    detail::submit_kernel_direct_parallel_for(*this, r, k);
    return {};
  }
  template <class Kn, class K> event single_task(const K &k) {
    detail::HostKernelRef<K> hk(k);
    range<1> r(1);
    detail::nd_range_view rv(r);
    return submit_kernel_direct_with_event_impl(
        *this, rv, hk, &detail::getDeviceKernelInfo<Kn>());
  }
  event memset(void *p, int v, unsigned long n);
  event memcpy(void *d, const void *s, unsigned long n);
  template <typename T>
  event memcpy(ext::oneapi::experimental::device_global<T> &d, const void *s,
               unsigned long n);
  template <typename T> event fill(void *p, const T &pattern, unsigned long n);
  template <typename T> event copy(const T *s, T *d, unsigned long n);
  void wait();
};
void *malloc_device(unsigned long bytes, const queue &q);
template <typename T> T *malloc_device(unsigned long count, const queue &q) {
  return (T *)malloc_device(count * sizeof(T), q);
}
void *aligned_alloc_shared(unsigned long align, unsigned long bytes,
                           const queue &q);
void free(void *p, const queue &q);
} // namespace _V1
} // namespace sycl

namespace other {
void *malloc_device(unsigned long bytes, const sycl::queue &q);
template <int D> struct range {
  range(unsigned long) {}
};
} // namespace other

// The queue API; the argument is the command-group or kernel lambda.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue6submitIZ{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "queue.submit", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue12parallel_forI2K4Z{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "queue.kernel", cir.sycl_host_api_arg = 1 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue11single_taskI2K4Z{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "queue.kernel", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZNK4sycl3_V15queue22submit_with_event_implERKNS0_6detail19type_erased_cgfo_tyE({{.*}} attributes {cir.sycl_host_api = "queue.submit_impl", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail19type_erased_cgfo_tyC2IRZ{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "cgfo.ctor", cir.sycl_host_api_arg = 0 : i32

// The handler API and its internal entry points.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler12parallel_forI2K1Z{{[^(]*}}NS0_5rangeILi1EEERKT0_({{.*}} attributes {cir.sycl_host_api = "handler.kernel", cir.sycl_host_api_arg = 1 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler12parallel_forI2K2Z{{[^(]*}}NS0_8nd_rangeIXT1_EEERKT0_({{.*}} attributes {cir.sycl_host_api = "handler.kernel", cir.sycl_host_api_arg = 1 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler11single_taskI2K3Z{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "handler.kernel", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler24parallel_for_lambda_implI2K1Z{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "handler.kernel_impl", cir.sycl_host_api_arg = 1 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail24KernelWrapperHelperFuncs18kernel_single_taskI2K3Z{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "kernel.wrapper", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler11StoreLambdaI2K1Z{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "handler.store_lambda", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler19setDeviceKernelInfoI2K1EEvPv({{.*}} attributes {cir.sycl_host_api = "handler.set_kernel_func", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler13setKernelFuncEPv({{.*}} attributes {cir.sycl_host_api = "handler.set_kernel_func", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler22setDeviceKernelInfoPtrEPNS0_6detail16DeviceKernelInfoE({{.*}} attributes {cir.sycl_host_api = "handler.set_kernel_info", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler20setNDRangeDescriptorENS0_6detail16nd_range_view_v113nd_range_viewE({{.*}} attributes {cir.sycl_host_api = "handler.set_ndrange", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail10HostKernelIZZ{{[^(]*}}EC2ERKSB_({{.*}} attributes {cir.sycl_host_api = "hostkernel.ctor", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI2K1EERNS1_16DeviceKernelInfoEv({{.*}} attributes {cir.sycl_host_api = "kernel.info", cir.sycl_host_api_arg = 0 : i32

// The direct (queue shortcut) launch path.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail33submit_kernel_direct_parallel_forI{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "launch.direct", cir.sycl_host_api_arg = 2 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail13HostKernelRefIKZ{{[^(]*}}EC2ERS9_({{.*}} attributes {cir.sycl_host_api = "hostkernel.ctor", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V136submit_kernel_direct_with_event_impl{{[^(]*}}({{.*}} attributes {cir.sycl_host_api = "launch.direct_impl", cir.sycl_host_api_arg = 2 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI2K4EERNS1_16DeviceKernelInfoEv({{.*}} attributes {cir.sycl_host_api = "kernel.info", cir.sycl_host_api_arg = 0 : i32

// Geometry.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V18nd_rangeILi1EEC2ENS0_5rangeILi1EEES4_({{.*}} attributes {cir.sycl_host_api = "ndrange.ctor", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15rangeILi1EEC2Em({{.*}} attributes {cir.sycl_host_api = "range.ctor", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V12idILi1EEC2Ev({{.*}} attributes {cir.sycl_host_api = "id.ctor", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail16nd_range_view_v113nd_range_viewC2ILi1EEERNS0_5rangeIXT_EEE({{.*}} attributes {cir.sycl_host_api = "ndrange_view.ctor", cir.sycl_host_api_arg = 0 : i32

// USM; the argument is the size or count.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V113malloc_deviceEmRKNS0_5queueE({{.*}} attributes {cir.sycl_host_api = "usm.alloc", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V113malloc_deviceIfEEPT_mRKNS0_5queueE({{.*}} attributes {cir.sycl_host_api = "usm.alloc", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V120aligned_alloc_sharedEmmRKNS0_5queueE({{.*}} attributes {cir.sycl_host_api = "usm.alloc", cir.sycl_host_api_arg = 1 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V14freeEPvRKNS0_5queueE({{.*}} attributes {cir.sycl_host_api = "usm.free", cir.sycl_host_api_arg = 0 : i32

// Explicit memory commands; arg is the destination, arg2 the source.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue6memcpyEPvPKvm({{.*}} attributes {cir.sycl_host_api = "queue.memcpy", cir.sycl_host_api_arg = 0 : i32, cir.sycl_host_api_arg2 = 1 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue6memsetEPvim({{.*}} attributes {cir.sycl_host_api = "queue.memset", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue4fillIiEENS0_5eventEPvRKT_m({{.*}} attributes {cir.sycl_host_api = "queue.fill", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue4copyIfEENS0_5eventEPKT_PS4_m({{.*}} attributes {cir.sycl_host_api = "queue.copy", cir.sycl_host_api_arg = 1 : i32, cir.sycl_host_api_arg2 = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler6memcpyEPvPKvm({{.*}} attributes {cir.sycl_host_api = "handler.memcpy", cir.sycl_host_api_arg = 0 : i32, cir.sycl_host_api_arg2 = 1 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler6memsetEPvim({{.*}} attributes {cir.sycl_host_api = "handler.memset", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler4fillIPfEEvPvRKT_m({{.*}} attributes {cir.sycl_host_api = "handler.fill", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler4fillIfEEvNS0_8accessorIT_Li1EEERKS4_({{.*}} attributes {cir.sycl_host_api = "handler.fill_accessor", cir.sycl_host_api_arg = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler4copyIfEEvPKT_PS3_m({{.*}} attributes {cir.sycl_host_api = "handler.copy", cir.sycl_host_api_arg = 1 : i32, cir.sycl_host_api_arg2 = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler4copyIfEEvNS0_8accessorIT_Li1EEEPS4_({{.*}} attributes {cir.sycl_host_api = "handler.copy_from_accessor", cir.sycl_host_api_arg = 1 : i32, cir.sycl_host_api_arg2 = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler4copyIfEEvPKT_NS0_8accessorIS3_Li1EEE({{.*}} attributes {cir.sycl_host_api = "handler.copy_to_accessor", cir.sycl_host_api_arg = 1 : i32, cir.sycl_host_api_arg2 = 0 : i32
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler4copyIfEEvNS0_8accessorIT_Li1EEES5_({{.*}} attributes {cir.sycl_host_api = "handler.copy_accessors", cir.sycl_host_api_arg = 1 : i32, cir.sycl_host_api_arg2 = 0 : i32

// Not tagged: a launch without a function object (prebuilt sycl::kernel),
// the non-template kernel-info lookup, the invoker trampoline, other queue
// API, the HostKernel bases, and same-named functions outside sycl. UNTAGGED
// reads only the tagged functions.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler12parallel_forENS0_5rangeILi1EEENS0_6kernelE(
// UNTAGGED-NOT: @_ZN4sycl3_V17handler12parallel_forENS0_5rangeILi1EEENS0_6kernelE(
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_23CompileTimeKernelInfoTyE(
// UNTAGGED-NOT: @_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_23CompileTimeKernelInfoTyE(
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerI{{[^(]*}}4callEPKvS8_(
// UNTAGGED-NOT: @_ZN4sycl3_V16detail19type_erased_cgfo_ty7invokerI{{[^(]*}}4callEPKvS8_(
// The device_global and shared_ptr memory-command overloads and the waits
// are untagged too.
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue6memcpyIiEENS0_5eventERNS0_3ext6oneapi12experimental13device_globalIT_EEPKvm(
// UNTAGGED-NOT: @_ZN4sycl3_V15queue6memcpyIiEENS0_5eventERNS0_3ext6oneapi12experimental13device_globalIT_EEPKvm(
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler6memcpyIiEEvRNS0_3ext6oneapi12experimental13device_globalIT_EEPKvm(
// UNTAGGED-NOT: @_ZN4sycl3_V17handler6memcpyIiEEvRNS0_3ext6oneapi12experimental13device_globalIT_EEPKvm(
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V17handler4copyIfEEvNS0_8accessorIT_Li1EEESt10shared_ptrIS4_E(
// UNTAGGED-NOT: @_ZN4sycl3_V17handler4copyIfEEvNS0_8accessorIT_Li1EEESt10shared_ptrIS4_E(
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15queue4waitEv(
// UNTAGGED-NOT: @_ZN4sycl3_V15queue4waitEv(
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V15event4waitEv(
// UNTAGGED-NOT: @_ZN4sycl3_V15event4waitEv(
// LAUNCH-DAG: cir.func {{.*}}@_ZN4sycl3_V16detail17HostKernelRefBaseC2Ev(
// UNTAGGED-NOT: @_ZN4sycl3_V16detail17HostKernelRefBaseC2Ev(
// LAUNCH-DAG: cir.func {{.*}}@_ZN5other13malloc_deviceEmRKN4sycl3_V15queueE(
// UNTAGGED-NOT: @_ZN5other13malloc_deviceEmRKN4sycl3_V15queueE(
// LAUNCH-DAG: cir.func {{.*}}@_ZN5other5rangeILi1EEC2Em(
// UNTAGGED-NOT: @_ZN5other5rangeILi1EEC2Em(
class K1;
class K2;
class K3;
class K4;
sycl::accessor<float, 1> &someAccessor();
sycl::ext::oneapi::experimental::device_global<int> &someDeviceGlobal();
void launch(sycl::queue &q, sycl::kernel kobj) {
  sycl::accessor<float, 1> &acc = someAccessor();
  sycl::ext::oneapi::experimental::device_global<int> &dg = someDeviceGlobal();
  float *a = sycl::malloc_device<float>(16, q);
  void *b = sycl::aligned_alloc_shared(64, 16, q);
  void *c = other::malloc_device(16, q);
  sycl::nd_range<1> ndr(sycl::range<1>(16), sycl::range<1>(4));
  sycl::id<1> o;
  other::range<1> orr(4);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<K1>(sycl::range<1>(16), [=](sycl::id<1>) { a[0] = 1; });
    h.parallel_for<K2>(ndr, [=](sycl::id<1>) { a[0] = 2; });
    h.single_task<K3>([=] { a[0] = 3; });
    h.parallel_for(sycl::range<1>(16), kobj);
    float *fp = a;
    h.memcpy(a, b, 16);
    h.memcpy(dg, b, 4);
    h.memset(a, 0, 16);
    h.fill(b, fp, 2);
    h.fill(acc, 1.0f);
    h.copy(fp, a, 4);
    h.copy(acc, a);
    h.copy(fp, acc);
    h.copy(acc, acc);
    h.copy(acc, std::shared_ptr<float>{a});
  });
  q.parallel_for<K4>(sycl::range<1>(16), [=](sycl::id<1>) { a[0] = 4; });
  q.single_task<K4>([=] { a[0] = 5; });
  q.memset(b, 0, 16);
  q.memcpy(b, a, 16);
  q.memcpy(dg, a, 4);
  q.fill(b, 7, 4);
  float *fa = a;
  q.copy(fa, a, 4).wait();
  q.wait();
  sycl::free(a, q);
  sycl::free(b, q);
  (void)c;
}

#endif // LAUNCH
