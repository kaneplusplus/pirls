
#' The R implementation of the STRONG filter.
#'
#' @param X the model matrix
#' @param z the response
#' @param lambda the penalty parameter.
#' @export
strong_filter = function(X, z, lambda) {
  Xty = t(X) %*% y / nrow(X)
  lambda_max = max(Xty)
  which( abs(Xty) >= 2*lambda - lambda_max)
}
