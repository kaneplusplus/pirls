library(glmnet)
library(foreach)
library(doMC)
registerDoMC(cores=6)

document()

# lmnet sanity check
x=matrix(rnorm(100*20),100,ncol=20)
y=rnorm(100)
fit1 = glmnet(x, y, standardize=FALSE, intercept=FALSE)
lambdas = fit1$lambda
fit2_betas = foreach(lambda=lambdas, .combine=cbind) %do% {
  pirls(x, y, lambda=lambda, family=gaussian)$beta
}
beta_diff = sqrt(crossprod(fit1$beta,fit2_betas))
if (any(sqrt(colSums(beta_diff)) > 1e-3)) stop("Precision problem.")

cbind(fit1$beta, fit2$beta)

mycd = function(X, W, z, lambda, alpha, beta, active_cols, maxit = 1000L) {
  W = as(diag(length(W)), "dgCMatrix")
  beta = as(beta, "dgCMatrix")
  z = matrix(z, nrow=length(z))
  c_coordinate_descent(X, W, z, lambda, alpha, beta,
                       as.integer(0:(ncol(X)-1)), maxit)
}

# Timing
glmnet_times = c()
glmnet_fit_times=c()
glmnet_c_fit_times=c()
num_cols = round(seq(20, 1000, length.out=20))
for (j in num_cols) {
  x=matrix(rnorm(1000*j), 1000, ncol=j)
  y=rnorm(1000)

  lambdas=glmnet(x, y, standardize=FALSE, intercept=FALSE)$lambda
  lambda = lambdas[round(length(lambdas)/2)]
  glmnet_times = c(glmnet_times, 
    system.time({
      a = glmnet(x, y, standardize=FALSE, lambda=lambda, intercept=FALSE)
    })[3])

  glmnet_fit_times = c(glmnet_fit_times,
    system.time({
      b = pirls(x, y, lambda=lambda, family=gaussian)
    })[3])

  glmnet_c_fit_times = c(glmnet_c_fit_times,
    system.time({
      b = pirls(x, y, lambda=lambda, family=gaussian, 
                beta_update=mycd)
    })[3])
  cat(paste(j, "cols.\n"))
}
library(ggplot2)
library(tidyr)

df = data.frame(list(glmnet=glmnet_times, pirls=glmnet_fit_times, 
                     Columns=num_cols))
dfg = gather(df, key=Regression, value=Time, glmnet:pirls)
p=ggplot(aes(y=Time, x=Columns, color=Regression), data=dfg) + 
  geom_line(size=1.5)
ggsave("pirls1.jpg", p, width=8, height=6)

df2 = data.frame(list(glmnet=glmnet_times, pirls=glmnet_fit_times, 
                      pirls_c=glmnet_c_fit_times, Columns=num_cols))
dfg2 = gather(df2, key=Regression, value=Time, glmnet:pirls_c)
p=ggplot(aes(y=Time, x=Columns, color=Regression), data=dfg2) + 
  geom_line(size=1.5)
ggsave("pirls2.jpg", p, width=8, height=6)

  x[,1] = x[,1] 
# Timing 2
glmnet_times = c()
glmnet_fit_times=c()
glmnet_c_fit_times=c()
num_cols = round(seq(20, 1000, length.out=20))
for (j in num_cols) {
  x=matrix(rnorm(1000*j), 1000, ncol=j)
  y=rnorm(1000)

  lambdas=glmnet(x, y, standardize=FALSE, intercept=FALSE)$lambda
  lambda = lambdas[round(length(lambdas)/10)]
  
  glmnet_times = c(glmnet_times, 
    system.time({
      a = glmnet(x, y, standardize=FALSE, lambda=lambda, intercept=FALSE)
    })[3])

  glmnet_fit_times = c(glmnet_fit_times,
    system.time({
      inds = strong_filter(x, y, lambda)
      if (length(inds) > 0) 
        xs = x[,inds]
      else
        xs = x
      b = pirls(x, y, lambda=lambda, family=gaussian)
    })[3])

  glmnet_c_fit_times = c(glmnet_c_fit_times,
    system.time({
      inds = strong_filter(x, y, lambda)
      if (length(inds) > 0) 
        xs = x[,inds]
      else
        xs = x
      b = pirls(xs, y, lambda=lambda, family=gaussian, 
                beta_update=mycd)
    })[3])
  cat(paste(j, "cols.\n"))
}
library(ggplot2)
library(tidyr)

#df = data.frame(list(glmnet=glmnet_times, pirls=glmnet_fit_times, 
#                     Columns=num_cols))
#dfg = gather(df, key=Regression, value=Time, glmnet:pirls)
#p=ggplot(aes(y=Time, x=Columns, color=Regression), data=dfg) + 
#  geom_line(size=1.5)
#ggsave("pirls1.jpg", p, width=8, height=6)

df2 = data.frame(list(glmnet=glmnet_times, pirls=glmnet_fit_times, 
                      pirls_c=glmnet_c_fit_times, Columns=num_cols))
dfg2 = gather(df2, key=Regression, value=Time, glmnet:pirls_c)
p=ggplot(aes(y=Time, x=Columns, color=Regression), data=dfg2) + 
  geom_line(size=1.5)
ggsave("pirls3.jpg", p, width=8, height=6)



Rprof(append=FALSE, interval=0.02, memory.profiling=TRUE, gc.profiling=TRUE, 
      line.profiling = TRUE)
Rprof(NULL)

system.time({b = pirls(x, y, lambda=lambda, family=gaussian)})

#binomial
x=matrix(rnorm(100*20),100,ncol=20)
g2=sample(1:2,100,replace=TRUE)
fit1 = glmnet(x,g2,family="binomial", standardize=FALSE, intercept=FALSE)
lambdas = fit1$lambda
lambda = lambdas[length(round(lambdas))/2]
fit2 = pirls(x, g2-1, family=binomial, lambda=lambda)
cbind(fit1$beta[,length(round(lambdas))/2], fit2$beta)


# Soft threshold testing
RcppParallel::setThreadOptions(numThreads = 4)
library(Matrix)
library(devtools)
document("..")
nrows = 1000
ncols = 1000
sp_mat = spMatrix(ncol=1, nrow=nrows)

X = matrix(rnorm(nrows*ncols), nrow=nrows, ncol=ncols)
W = as(diag(nrows), "dgCMatrix")
z = matrix(rnorm(nrows), ncol=1)
lambda = 0.3
alpha = 1
beta = as(matrix(0, nrow=ncols, ncol=1), "dgCMatrix")

mycd = function(X, W, z, lambda, alpha, beta, active_cols, maxit = 1000L) {
  W = as(diag(length(W)), "dgCMatrix")
  beta = as(beta, "dgCMatrix")
  z = matrix(z, nrow=length(z))
  c_coordinate_descent(X, W, z, lambda, alpha, beta, 
                       as.integer(0:(ncol(X)-1)), maxit)
}

system.time({b1 = pirls(X, z, lambda=lambda, family=gaussian, 
                       beta_update=mycd)})
system.time({b2 = pirls(X, z, lambda=lambda, family=gaussian)})


system.time({
  cuc = c_update_coordinates(X, W, z, lambda, alpha, beta, parallel=FALSE,
                             grain_size=100)})
system.time({
  uc = update_coordinates(X, diag(W), z, lambda, alpha, beta)})

c_quadratic_loss(X, W, z, lambda, alpha, beta)
quadratic_loss(X, diag(W), z, lambda, alpha, beta)

c_safe_filter(X, z, lambda)
