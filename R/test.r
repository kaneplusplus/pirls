library(glmnet)
library(foreach)
library(doMC)
registerDoMC(cores=6)

source("coordinate_descent.r")
source("soft_threshold.r")
source("glmnet_fit.r")

# lmnet sanity check
x=matrix(rnorm(100*20),100,ncol=20)
y=rnorm(100)
fit1 = glmnet(x, y, standardize=FALSE, intercept=FALSE)
lambdas = fit1$lambdas
fit2_betas = foreach(lambda=lambdas, .combine=cbind) %dopar% {
  glmnet_fit(x, y, lambda=lambda, family=gaussian)$beta
}
beta_diff = (fit1$beta - fit2_betas)^2
if (any(sqrt(colSums(beta_diff)) > 1e-3)) stop("Precision problem.")

cbind(fit1$beta, fit2$beta)

# Timing
glmnet_times = c()
glmnet_fit_times=c()
num_cols = round(seq(20, 1000, length.out=20))
for (j in num_cols) {
  x=matrix(rnorm(100*j), 100, ncol=j)
  y=rnorm(100)

  lambdas=glmnet(x, y, standardize=FALSE, intercept=FALSE)$lambda
  lambda = lambdas[round(length(lambdas)/2)]
  glmnet_times = c(glmnet_times, 
    system.time({
      glmnet(x, y, standardize=FALSE, lambda=lambda, intercept=FALSE)
    })[3])

  glmnet_fit_times = c(glmnet_fit_times,
    system.time({
      glmnet_fit(x, y, lambda=lambda, family=gaussian)
    })[3])
  print(j)
}


#binomial
g2=sample(1:2,100,replace=TRUE)
fit1 = glmnet(x,g2,family="binomial", standardize=FALSE, la)

