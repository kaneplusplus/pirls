#define ARMA_NO_DEBUG
#include "soft_threshold.hpp"

template<typename ModelMatrixType, typename WeightMatrixType, 
         typename ResponseType, typename BetaMatrixType>
double update_coordinate(const unsigned int i, 
  const ModelMatrixType &X, const WeightMatrixType &W, const ResponseType &z, 
  const double &lambda, const double &alpha, const BetaMatrixType &beta,
  const double thresh) {
  ModelMatrixType X_shed(X), X_col(X.col(i));
  BetaMatrixType beta_shed(beta);
  X_shed.shed_col(i);
  beta_shed.shed_row(i);
  double val = arma::mat((W*X_col).t() * (z - X_shed * beta_shed))[0];
  double ret = soft_thresh(val, thresh);
  // Assume W is a diagonal matrix.
  if (ret != 0) 
    ret /= accu(W * square(X_col)) + lambda*(1-alpha);
  return ret;
}

template<typename ModelMatrixType, typename WeightMatrixType,
         typename ResponseType, typename BetaMatrixType>
double quadratic_loss(const ModelMatrixType &X, const WeightMatrixType &W,
  const ResponseType &z, const double &lambda, const double &alpha, 
  const BetaMatrixType &beta) {
  return 1./X.n_rows/2. * accu(W*square(z-X*beta)) - 
    lambda * ((1-alpha) * accu(square(beta))/2 * alpha * accu(abs(beta)));
}

//template <typename ModelMatrixType, typename WeightMatrixType, 
//          typename VectorType, typename BetaMatrixType>
//BetaMatrixType generic_coordinate_descent(const ModelMatrixType &X, 
//  const WeightMatrixType &W, const VectorType &z, const double &lambda,
//  const double &alpha, const BetaMatrixType &beta, unsigned int maxit) 
//{
//  double quad_loss = quadratic_loss(X, W, z, lambda, alpha, beta, maxit);
//  double quad_loss_old;
//  BetaMatrixType beta_old(beta);
//  for (unsigned int i=0; i < maxit; ++i) {
//    beta_old = beta;
//    quad_loss_old = quad_loss;
//    beta = update_coordinates(X, W, z, lambda, alpha, beta);
//    quad_loss = quadratic_loss(X, W, z, lambda, alpha, beta, maxit);
//    if (quad_loss >= quad_loss_old) {
//      beta = beta_old;
//      break;
//    }
//  }
//}


