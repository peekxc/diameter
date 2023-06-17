#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
#include <execution>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// From: https://stackoverflow.com/questions/63340193/c-how-to-elegantly-use-c17-parallel-execution-with-for-loop-that-counts-an-i
// template <typename NumericType>
// struct ioterable{
//   using iterator_category = std::input_iterator_tag;
//   using value_type = NumericType;
//   using difference_type = NumericType;
//   using pointer = std::add_pointer<NumericType>;
//   using reference = NumericType;
//   explicit ioterable(NumericType n) : val_(n) {}

//   ioterable() = default;
//   ioterable(ioterable&&) = default;
//   ioterable(ioterable const&) = default;
//   ioterable& operator=(ioterable&&) = default;
//   ioterable& operator=(ioterable const&) = default;

//   ioterable& operator++() { ++val_; return *this; }
//   ioterable operator++(int) { ioterable tmp(*this); ++val_; return tmp; }
//   bool operator==(ioterable const& other) const { return val_ == other.val_; }
//   bool operator!=(ioterable const& other) const { return val_ != other.val_; }

//   value_type operator*() const { return val_; }

// private:
//   NumericType val_{ std::numeric_limits<NumericType>::max() };
// };




constexpr auto sq_dist(double u, double v) noexcept -> double { 
  return (u - v) * (u - v); 
}

// Exhaustive search of (n choose 2) distances
double diam_exhaustive(py::array_t<double>& X) {
  py::buffer_info X_buffer = X.request();
  double* x = static_cast<double *>(X_buffer.ptr);
 
  const size_t n = X_buffer.shape[0];
  const size_t d = X_buffer.shape[1];
  if (n == 0 || d == 0){ return(0.0); }
  
  std::array< double, 2 > best = { 0.0, 0.0 }; // c_max, c_diff
  for(size_t i = 0; i < n; ++i){ 
    for (size_t j = i+1; j < n; ++j){
      best[1] = std::inner_product(x+(i*d), x+((i+1)*d), x+(j*d), 0.0, std::plus< double >(), sq_dist);
      best[0] = best[best[1] > best[0]];
    }
  }
  return(best[0]);
}
// Finds the largest distance d(p,q) over all pairs (p,q) \in P x Q 
#include <omp.h>
double exhaustive_search(py::array_t<double>& P,  py::array_t<double>& Q, bool do_parallel) {
  py::buffer_info P_buffer = P.request(), Q_buffer = Q.request();
  double* p = static_cast<double *>(P_buffer.ptr);
  double* q = static_cast<double *>(Q_buffer.ptr);
  const int d = P_buffer.shape[1];
  const size_t np = P_buffer.shape[0];
  const size_t nq = Q_buffer.shape[0];
  if (np == 0 || nq == 0){ return(0.0); } // After this, P, Q must be non-empty
  
  double max_diam = 0.0; 
  if (do_parallel){
    std::vector< double > diffs(np, 0.0);
    std::vector< int > PI(np, 0), QI(nq, 0);
    #pragma omp parallel for
    for(size_t i = 0; i < np; ++i){ 
      std::array< double, 2 > best = { 0.0, 0.0 }; // c_max, diff
      for (size_t j = 0; j < nq; ++j){
        best[1] = std::inner_product(p+(i*d), p+((i+1)*d), q+(j*d), 0.0, std::plus< double >(), sq_dist);
        best[0] = best[best[1] > best[0]];
      }
      diffs[i] = best[0];
    }
    max_diam = *std::max_element(diffs.begin(), diffs.end()); // guarenteed to exist 
  } else { 
    std::array< double, 2 > best = { 0.0, 0.0 }; // c_max, diff
    for (size_t i = 0; i < np; ++i){
      for (size_t j = 0; j < nq; ++j){
        best[1] = std::inner_product(p+(i*d), p+((i+1)*d), q+(j*d), 0.0, std::plus< double >(), sq_dist);
        best[0] = best[best[1] > best[0]]; // branchless way to avoid conditional assignment
      }
    }
    max_diam = best[0];
  }
  return(max_diam);
}

auto exhaustive_index(const py::array_t<double>& X, const py::array_t< int >& P, const py::array_t< int >& Q) -> py::tuple {
  py::buffer_info X_buffer = X.request(), P_buffer = P.request(), Q_buffer = Q.request();
  double* x = static_cast<double *>(X_buffer.ptr);
  auto p = P.unchecked< 1 >();
  auto q = Q.unchecked< 1 >();
  const int d = X_buffer.shape[1];
  const size_t np = P_buffer.size;
  const size_t nq = Q_buffer.size;
  double max_diam = 0.0, c_diam = 0.0;
  int best_p, best_q, pi, qj;
  for (size_t i = 0; i < np; ++i){
    for (size_t j = 0; j < nq; ++j){
      pi = p(i);
      qj = q(j);
      c_diam = std::inner_product(x+(pi*d), x+((pi+1)*d), x+(qj*d), 0.0, std::plus< double >(), sq_dist);
      if (c_diam > max_diam){
        best_p = pi; 
        best_q = qj; 
        max_diam = c_diam; 
      }
    }
  }
  return(py::make_tuple(best_p, best_q, max_diam));
}


//  Returns the indices of 'P' modulo a ball B(c, d/r) with c := center and r := diameter/2. 
// Q = P U B(c, diam / 2) if intersect = true 
// Q = P \ B(c, diam / 2) if intersect = false
auto subset_ball(py::array_t<double>& X, const py::array_t<int>& P, py::array_t<double>& center, const double diam, const bool intersect) -> py::array_t< int > {
  py::buffer_info X_buffer = X.request(), center_buffer = center.request();
  double* x = static_cast<double *>(X_buffer.ptr);
  double* c = static_cast<double *>(center_buffer.ptr);
  
  // Variables
  const size_t d = X_buffer.shape[1];
  const size_t p_sz = P.request().size;
  const double threshold = std::pow(diam/2.0, 2); // add the epsilon at call
  double dist_to_center = 0.0;
  auto p_ind = P.unchecked< 1 >();
  auto Q = std::vector< int >(); 
  
  // Record all elements of P in the ball
  if (intersect){
    for (size_t i = 0, p = 0; i < p_sz; ++i){
      p = p_ind(i);
      dist_to_center = std::inner_product(x+(p*d), x+((p+1)*d), c, 0.0, std::plus< double >(), sq_dist);
      if (dist_to_center < threshold){
        Q.push_back(p);
      }
    }
  } else {
    for (size_t i = 0, p = 0; i < p_sz; ++i){
      p = p_ind(i);
      dist_to_center = std::inner_product(x+(p*d), x+((p+1)*d), c, 0.0, std::plus< double >(), sq_dist);
      if (dist_to_center > threshold){
        Q.push_back(p);
      }
    }
  }
  auto result = py::array_t< int >(Q.size(), Q.data());
  return(result);
}

// From: https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
// Iterates over the positions of the set bits in a bitmap
// template <typename Lambda> 
// size_t bitmap_decode_ctz(uint64_t* bitmap, size_t bitmapsize, Lambda f) noexcept {
//   uint64_t bitset;
//   for (size_t k = 0; k < bitmapsize; ++k) {
//     bitset = bitmap[k];
//     while (bitset != 0) {
//       uint64_t t = bitset & -bitset;
//       int r = __builtin_ctzll(bitset);
//       f(k * 64 + r);
//       bitset ^= t;
//     }
//   }
// }

// #include <cmath>

// // Converts a given COO-sparse matrix into a uint64_t bitset
// // R := row indices, representing set membership 
// // C := col indices, representing point indices
// // The bit-array encodes the (R, C) indices in a row-oriented fashion, allowing for 
// // contiguous access to the subsets a given point participates in
// auto encode_subsets(py::array< int >& R, py::array< int >& C, const size_t ns, const size_t np) -> std::unique_ptr< uint64_t[] > {
//   auto r = R.unchecked<1>();
//   auto c = C.unchecked<1>();
//   const size_t ne = r.shape(0); // numebr of element 
//   // assert r.shape == c.shape
//   auto ni = static_cast< size_t >(std::ceil(double(ns*np)/64.0)); // number of uint64_t's
//   auto bit_array = std::make_unique< uint64_t[] >(ni);
//   uint64_t* p = bit_array.get();
//   for (size_t i = 0; i < ne; i++){
//     uint64_t k = r(i)*ns + c(i)
//     // Do d, r = k mod 64, then p[d] << r
//     // p |= (p << k);
//   }
//   return(bit_array);
// }
  

PYBIND11_MODULE(diameter_ext, m) {
  m.doc() = "pybind11 diameter plugin"; // optional module docstring
  m.def("exhaustive_search", &exhaustive_search, "Gets the max distance between P x Q");
  m.def("exhaustive_index", &exhaustive_index, "Gets the maximum segment (p, q, d(p,q)) between P x Q");
  m.def("diam_exhaustive", &diam_exhaustive, "Exhaustively checks all n choose 2 distances for the diameter");
  m.def("subset_ball", &subset_ball, "Subset ball");
}