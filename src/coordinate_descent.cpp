#define ARMA_NO_DEBUG

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

template <typename ModelMatrixType, typename WeightMatrixType, 
          typename VectorType, typename BetaMatrixType>
BetaMatrixType generic_coordinate_descent(const MatrixType &X, 
  const WeightMatrixType &W, const VectorType &z, const double &lambda,
  const double &alpha, const BetaMatrixType &beta, unsigned int maxit) 
{
  double quad_loss = quadratic_loss(X, W, z, lambda, alpha, beta, maxit);
  double quad_loss_old;
  BetaMatrixType beta_old(beta);
  for (unsigned int i=0; i < maxit; ++i) {
    beta_old = beta;
    quad_loss_old = quad_loss;
    beta = update_coordinates(X, W, z, lambda, alpha, beta);
    quad_loss = quadratic_loss(X, W, z, lambda, alpha, beta, maxit);
    if (quad_loss >= quad_loss_old) {
      beta = beta_old;
      break;
    }
  }
  if (i == maxit && quad_loss
}
