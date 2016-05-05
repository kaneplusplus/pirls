
#' Fit a Penalized Iteratively Reweighted Least Squares (PIRLS) model
#'
#' 'pirls' is used to fit the generalized linear model using penalized
#' maximum likelihood based on the model matrix.
#' @param X the model matrix.
#' @param y the response.
#' @param lambda the penalty
#' @param alpha the elasticnet mixing parameter 0 <= alpha <= 1. Default is
#' 1 (LASSO regression).
#' @param family a description of the error distribution and link function
#' to be used in the model.
#' @param maxit the maximum number of iterations for both the weight updates
#' as well as the beta update. Default is 25.
#' @param tol the numeric tolerance. Default is 1e-8.
#' @param beta initial slope coefficients. Default is zeros.
#' @param beta_update a function for optimizing the slope coefficients
#' for the current weight matrix. Default is coordinate_descent.
#' @export
pirls = function(X, y, lambda, alpha=1, family=binomial, maxit=500, 
                      tol=1e-8, beta=spMatrix(nrow=ncol(X), ncol=1), 
                      beta_update=coordinate_descent) {
  converged = FALSE
  for(i in 1:maxit) {
    # Note that there aren't sparse family functions.
    eta      = as.vector(X %*% beta)
    g        = family()$linkinv(eta)
    gprime   = family()$mu.eta(eta)
    z        = eta + (y - g) / gprime
    W        = as.vector(gprime^2 / family()$variance(g))
    beta_old = beta
    beta = beta_update(X, W, z, lambda, alpha, beta, maxit)
    if(sqrt(as.vector(Matrix::crossprod(beta-beta_old))) < tol) {
      converged = TRUE
      break
    }
  }
  list(beta=beta, iterations=i, converged=converged)
}
