// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s

// On the SYCL host, CIRGen tags the buffer and accessor API (and
// std::vector::data) with the role the offload combine traces buffer identity
// through, recognized by declaration: cir.sycl_host_api, plus the
// source-level index of the buffer/host-pointer parameter the role reads.

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
};
template <typename T, int D> class accessor {
public:
  accessor(buffer<T, D> &b, handler &h);
  accessor(handler &h, buffer<T, D> &b, int tag);
  accessor(const accessor &other);
  accessor();
  accessor &operator=(const accessor &other);
  T &operator[](unsigned long i) const;
};
} // namespace _V1
} // namespace sycl

namespace other {
template <typename T, int D> class buffer {
public:
  buffer(const sycl::range<D> &r);
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
// CHECK-NOT: @_ZNSt6vectorIfE5beginEv({{.*}}cir.sycl_host_api
// CHECK-DAG: cir.func {{.*}}@_ZNK4sycl3_V16bufferIfLi1EE9get_rangeEv(
// CHECK-NOT: @_ZNK4sycl3_V16bufferIfLi1EE9get_rangeEv({{.*}}cir.sycl_host_api
// CHECK-DAG: cir.func {{.*}}@_ZN5other6bufferIfLi1EEC1ERKN4sycl3_V15rangeILi1EEE(
// CHECK-NOT: @_ZN5other6bufferIfLi1EEC1ERKN4sycl3_V15rangeILi1EEE({{.*}}cir.sycl_host_api
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
  (void)v.begin();
  (void)a.get_range();
  other::buffer<float, 1> ob(r);
}
