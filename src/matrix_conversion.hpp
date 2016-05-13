#define ARMA_NO_DEBUG

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// converts an SEXP object from R which was created as a sparse
// matrix via the Matrix package) into an Armadillo sp_mat matrix
//
// NB: called as_() here as a similar method is already in the 
//     RcppArmadillo sources
//
template <typename T> 
SpMat<T> as(SEXP sx) {
  // Rcpp representation of template type
  const int RTYPE = Rcpp::traits::r_sexptype_traits<T>::rtype;

  // instantiate S4 object with the sparse matrix passed in
  S4 mat(sx);  
  IntegerVector dims = mat.slot("Dim");
  IntegerVector i = mat.slot("i");
  IntegerVector p = mat.slot("p");     
  Vector<RTYPE> x = mat.slot("x");

          // create sp_mat object of appropriate size
  SpMat<T> res(dims[0], dims[1]);
  // create space for values, and copy
  access::rw(res.values) = memory::acquire_chunked<T>(x.size() + 1);
  arrayops::copy(access::rwp(res.values), x.begin(), x.size() + 1);
  // create space for row_indices, and copy 
  access::rw(res.row_indices) = 
  memory::acquire_chunked<uword>(i.size() + 1);
  arrayops::copy(access::rwp(res.row_indices), i.begin(), i.size() + 1);

  // create space for col_ptrs, and copy 
  access::rw(res.col_ptrs) = memory::acquire<uword>(p.size() + 2);
  arrayops::copy(access::rwp(res.col_ptrs), p.begin(), p.size() + 1);
  // important: set the sentinel as well
  access::rwp(res.col_ptrs)[p.size()+1] = std::numeric_limits<uword>::max();

  // set the number of non-zero elements
  access::rw(res.n_nonzero) = x.size();
  return res;
}

// convert an Armadillo sp_mat into a corresponding R sparse matrix
// we copy to STL vectors as the Matrix package expects vectors whereas the
// default wrap in Armadillo returns matrix with one row (or col) 
//
// NB: called wrap_() here as a similar method is already in the 
//     RcppArmadillo sources
//
template <typename T> 
SEXP wrap(const arma::SpMat<T>& sm) {
  const int  RTYPE = Rcpp::traits::r_sexptype_traits<T>::rtype;
  IntegerVector dim = IntegerVector::create( sm.n_rows, sm.n_cols );
  // copy the data into R objects
  Vector<RTYPE> x(sm.values, sm.values + sm.n_nonzero ) ;
  IntegerVector i(sm.row_indices, sm.row_indices + sm.n_nonzero);
  IntegerVector p(sm.col_ptrs, sm.col_ptrs + sm.n_cols+1 ) ;
  std::string klass;
  switch (RTYPE) {
    case REALSXP: 
      klass = "dgCMatrix"; 
      break;
  // case INTSXP:   // class not exported in Matrix package
  //  klass = "igCMatrix"; 
  //  break; 
    case LGLSXP: 
      klass = "lgCMatrix"; 
      break;
    default:
      throw std::invalid_argument("RTYPE not matched in conversion to sparse matrix");
  }
  S4 s(klass);
  s.slot("i")   = i;
  s.slot("p")   = p;
  s.slot("x")   = x;
  s.slot("Dim") = dim;
  return s;
}
