// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(BH)]]

#include <boost/iterator/counting_iterator.hpp>
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <vector>

#include "coordinate_descent.hpp"

using namespace RcppParallel;
using namespace boost;
using namespace std::placeholders;

template <typename VecType>
arma::sp_mat vec_to_diag( const VecType &vec ) {
  unsigned long i;
  arma::umat locs(2, vec.size());
  arma::vec copy_vec(vec.size());
  for (i=0; i < vec.size(); ++i) {
    locs(0, i) = i;
    locs(0, i) = i;
    copy_vec(i) = vec[i];
  }
  return arma::sp_mat(locs, copy_vec, i, i);
}

template <typename VecType>
arma::sp_mat vec_to_col_sp_mat( const VecType &vec ) {
  unsigned long i,j;
  unsigned int nz=0;
  for (i=0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      ++nz;
    }
  }
  arma::vec vals(nz);
  arma::umat locs(2, nz);
  j=0;
  for (i=0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      vals[j] = vec[i];
      locs(0, j) = i;
      locs(1, j++) = 0;
    }
  }
  return arma::sp_mat(locs, vals, vec.size(), 1);
}

template<typename ModelMatrixType, typename WeightMatrixType,
         typename ResponseType, typename BetaMatrixType>
struct UpdateCoordinate : public Worker {
  // members
  const ModelMatrixType &_X;
  const WeightMatrixType &_W;
  const ResponseType &_z;
  const BetaMatrixType &_beta;
  const double &_lambda;
  const double &_alpha;
  const double _thresh;
  std::vector<double> _output;

  // constructor
  UpdateCoordinate(const ModelMatrixType &X, const WeightMatrixType &W,
                   const ResponseType &z, const BetaMatrixType &beta,
                   const double &lambda, const double &alpha,
                   const double thresh) :
    _X(X), _W(W), _z(z), _beta(beta), _lambda(lambda), _alpha(alpha),
    _thresh(thresh), _output(beta.n_rows) {
    }

  void operator()(std::size_t begin, std::size_t end) {
    transform(counting_iterator<size_t>(begin), counting_iterator<size_t>(end),
      _output.begin(),
      [&](const unsigned int i)->double 
        {return update_coordinate(i, _X, _W, _z, _lambda, _alpha, _beta,
                                  _thresh);});
  }
};

// [[Rcpp::export]]
arma::uvec c_safe_filter(arma::mat X, arma::mat z, double lambda) {
  arma::vec Xty(X.t() * z);
  arma::vec cutoffs(X.n_cols);
  double zl2 = sqrt(sum(square(z)))[0];
  double lambda_max = Xty.max();
  unsigned long i=0;
  for (i=0; i < X.n_cols; ++i) {
    cutoffs[i] = lambda * sqrt(sum(square(X.col(i)))) * zl2 * 
      (lambda_max - lambda) / lambda_max;
  }
  return find( abs(Xty) > cutoffs );
}
  
// [[Rcpp::export]]
arma::sp_mat c_update_coordinates(arma::mat X, arma::sp_mat W, 
  arma::mat z, double lambda, double alpha, arma::sp_mat beta, 
  bool parallel=false, unsigned int grain_size=100) {
    double thresh = accu(W)*lambda*alpha;
    if (!parallel) {
      arma::sp_mat beta_new(beta);
      for (unsigned i=0; i < beta.n_rows; ++i) {
        beta(i, 0) = update_coordinate(i, X, W, z, lambda, alpha, beta, thresh);
      }
      return beta;
    } else {
      UpdateCoordinate<arma::mat, arma::sp_mat, arma::mat, arma::sp_mat> 
        uc(X, W, z, beta, lambda, alpha, thresh);
      parallelFor(0, beta.n_rows, uc, grain_size);
      uc(0, beta.n_rows);
      return vec_to_col_sp_mat(uc._output);
    }
}

// [[Rcpp::export]]
double c_quadratic_loss(arma::mat X, arma::sp_mat W,
  arma::mat z, double lambda, double alpha, arma::sp_mat beta) {
  return quadratic_loss(X, W, z, lambda, alpha, beta);
}

