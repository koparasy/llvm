// A reduced SYCL + C++ library for the combine-canary-*.cpp end-to-end tests,
// with the shapes the offload combine recognizes (the CIRGen host API roles
// are given by declaration, see host-api-roles.cpp). It extends the mock of
// combine-gemm-noalias.cpp with sub-buffers, buffer::get_range, USM
// allocation and memcpy, nd_range launches (through a detail::nd_range_view
// with the real layout {MGlobalSize, MLocalSize, MOffset, MDims}) and
// nd_item.

#define __SYCL_TYPE(x) [[__sycl_detail__::sycl_type(x)]]

namespace std {
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
  unsigned long &operator[](int i) { return v[i]; }
};
template <int D> struct id {
  unsigned long v[D];
  id() { v[0] = 0; }
  id(unsigned long a) { v[0] = a; }
  unsigned long operator[](int i) const { return v[i]; }
  unsigned long &operator[](int i) { return v[i]; }
};
template <int D> struct nd_range {
  range<D> globalSize;
  range<D> localSize;
  id<D> offset;
  nd_range(range<D> g, range<D> l) : globalSize(g), localSize(l) {}
};
template <int D> struct nd_item {
  unsigned long get_global_id(int) const { return 0; }
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
  buffer(buffer &parent, const id<D> &offset, const range<D> &r);
  buffer(const buffer &other);
  ~buffer();
  template <access::mode M> accessor<T, D, M> get_access(handler &h);
  range<D> get_range() const;
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
     sizeof(__builtin_sycl_unique_stable_name(K)) - 1}};
struct DeviceKernelInfo {};
DeviceKernelInfo &getDeviceKernelInfo(const CompileTimeKernelInfoTy &);
template <class K> DeviceKernelInfo &getDeviceKernelInfo() {
  static DeviceKernelInfo &Info = getDeviceKernelInfo(CompileTimeKernelInfo<K>);
  return Info;
}
inline namespace nd_range_view_v1 {
class nd_range_view {
public:
  template <int D>
  nd_range_view(range<D> &r) : MGlobalSize(&(r[0])), MDims((unsigned long)D) {}
  template <int D>
  nd_range_view(nd_range<D> &r)
      : MGlobalSize(&(r.globalSize[0])), MLocalSize(&(r.localSize[0])),
        MOffset(&(r.offset[0])), MDims((unsigned long)D) {}
  const unsigned long *MGlobalSize = nullptr;
  const unsigned long *MLocalSize = nullptr;
  const unsigned long *MOffset = nullptr;
  unsigned long MDims = 0;
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
  template <typename Kn, typename Item, typename K>
  [[clang::sycl_kernel]] static void kernel_parallel_for(const K &k) {
    k(Item());
  }
#else
  template <typename Kn, typename Item, typename K>
  static void kernel_parallel_for(const K &) {}
#endif
};
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

public:
  template <class Kn, class K> void parallel_for(range<1> r, const K &k) {
    detail::KernelWrapperHelperFuncs::kernel_parallel_for<Kn, id<1>>(k);
    setNDRangeDescriptor(detail::nd_range_view(r));
    StoreLambda<Kn>(k);
  }
  template <class Kn, class K> void parallel_for(nd_range<1> r, const K &k) {
    detail::KernelWrapperHelperFuncs::kernel_parallel_for<Kn, nd_item<1>>(k);
    setNDRangeDescriptor(detail::nd_range_view(r));
    StoreLambda<Kn>(k);
  }
};

class event {
public:
  void wait();
};
class queue {
  event submit_with_event_impl(const detail::type_erased_cgfo_ty &cgf) const;

public:
  template <typename T> event submit(T CGF) {
    return submit_with_event_impl(detail::type_erased_cgfo_ty{CGF});
  }
  event memcpy(void *d, const void *s, unsigned long n);
  void wait();
};

void *malloc_shared(unsigned long bytes, const queue &q);
void *malloc_host(unsigned long bytes, const queue &q);
template <typename T> T *malloc_shared(unsigned long count, const queue &q) {
  return (T *)malloc_shared(count * sizeof(T), q);
}
template <typename T> T *malloc_host(unsigned long count, const queue &q) {
  return (T *)malloc_host(count * sizeof(T), q);
}
void free(void *p, const queue &q);
} // namespace _V1
} // namespace sycl
