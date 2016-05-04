
#' The R implementation of the soft thresholding function.
#' 
#' Soft threshold x with the parameter g in R
#' @param x a vector of values to soft threshold.
#' @param g the threshold paramter
#' @export
#' @examples
#' soft_thresh_r(rnorm(10), 0.5)
soft_thresh_r = function(x, g) {
  x = as.vector(x)
  x[abs(x) < g] = 0
  x[x > 0] = x[x > 0] - g
  x[x < 0] = x[x < 0] + g
  x
}

