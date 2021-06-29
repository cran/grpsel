## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 5
)

## -----------------------------------------------------------------------------
set.seed(123)
n <- 100 # Number of observations
p <- 10 # Number of predictors
g <- 5 # Number of groups
group <- rep(1:g, each = p / g) # Group structure
beta <- numeric(p)
beta[which(group %in% 1:2)] <- 1 # First two groups are nonzero
x <- matrix(rnorm(n * p), n, p)
y <- x %*% beta + rnorm(n)

## -----------------------------------------------------------------------------
library(grpsel)
fit <- grpsel(x, y, group)

## -----------------------------------------------------------------------------
coef(fit)

## -----------------------------------------------------------------------------
plot(fit)

## -----------------------------------------------------------------------------
x.new <- matrix(rnorm(10 * p), 10, p)
predict(fit, x.new)

## -----------------------------------------------------------------------------
fit <- grpsel(x, y, group, penalty = 'grSubset+grLasso')
coef(fit, lambda = 0.05, gamma = 0.1)

fit <- grpsel(x, y, group, penalty = 'grSubset+Ridge')
coef(fit, lambda = 0.05, gamma = 0.1)

## -----------------------------------------------------------------------------
cvfit <- cv.grpsel(x, y, group, penalty = 'grSubset+Ridge', nfold = 10) # 10-fold cross-validation

## -----------------------------------------------------------------------------
plot(cvfit)

## -----------------------------------------------------------------------------
coef(cvfit)
predict(cvfit, x.new)

## -----------------------------------------------------------------------------
y <- pmax(sign(y), 0)

fit <- cv.grpsel(x, y, group, loss = 'logistic')
coef(fit)

## -----------------------------------------------------------------------------
p <- 5
x <- matrix(rnorm(n * p), n, p)
y <- rowSums(x) + rnorm(n)
group <- list(c(1, 2, 3), c(3, 4, 5))
fit <- grpsel(x, y, group)

## -----------------------------------------------------------------------------
coef(fit)

## -----------------------------------------------------------------------------
p <- 10
group <- rep(1:g, each = p / g)
Sigma <- 0.9 ^ t(sapply(1:p, function(i, j) abs(i - j), 1:p))
x <- matrix(rnorm(n * p), n, p)
x <- t(chol(Sigma) %*% t(x))
beta[which(group %in% 1:2)] <- 1 # First two groups are nonzero
y <- x %*% beta + rnorm(n)
fit <- cv.grpsel(x, y, group)
coef(fit)
fit <- cv.grpsel(x, y, group, ls = T)
coef(fit)

## -----------------------------------------------------------------------------
n <- 100
p <- 10
m <- 10 # Number of response variables
beta <- matrix(0, p, m)
beta[1:5, ] <- 1
x <- matrix(rnorm(n * p), n, p)
y <- x %*% beta + matrix(rnorm(n * m), n, m)
x <- scale(x)
y <- scale(y)

y <- matrix(y, ncol = 1)
x <- diag(m) %x% x
group <- rep(1:p, m)

cvfit <- cv.grpsel(x, y, group)
matrix(coef(cvfit)[- 1, , drop = F], ncol = m)

## -----------------------------------------------------------------------------
n <- 100
p <- 10
x <- matrix(runif(n * p), n, p)
y <- sinpi(2 * x[, 1]) + cospi(2 * x[, 2]) + rnorm(n, sd = 0.1)
df <- 5
splines <- lapply(1:p, function(j) splines::ns(x[, j], df = df))
x.s <- do.call(cbind, splines)
group <- rep(1:p, each = df)
fit <- cv.grpsel(x.s, y, group)

## -----------------------------------------------------------------------------
unique(group[coef(fit)[- 1] != 0])

## -----------------------------------------------------------------------------
library(ggplot2)

beta0 <- coef(fit)[1]
beta <- coef(fit)[- 1]
int.x1 <- (beta0 + colMeans(x.s[, - (1:df)]) %*% beta[- (1:df)])[, ]
int.x2 <- (beta0 + colMeans(x.s[, - (1:df + df)]) %*% beta[- (1:df + df)])[, ]

ggplot(x = seq(- 1, 1, length.out = 101)) +
  stat_function(fun = function(x) sinpi(2 * x), aes(linetype = 'True function')) +
  stat_function(fun = function(x) int.x1 + predict(splines[[1]], x) %*% beta[1:df], aes(linetype = 'Fitted function')) +
  xlab('x') +
  ylab('f(x)')

ggplot(x = seq(- 1, 1, length.out = 101)) +
  stat_function(fun = function(x) cospi(2 * x), aes(linetype = 'True function')) +
  stat_function(fun = function(x) int.x2 + predict(splines[[2]], x) %*% beta[1:df + df], aes(linetype = 'Fitted function')) +
  xlab('x') +
  ylab('f(x)')

## -----------------------------------------------------------------------------
n <- 100
p <- 10
x <- matrix(rnorm(n * p), n, p)
y <- x[, 1] + x[, 2] + x[, 3] + x[, 1] * x[, 2] + rnorm(n)

x.int <- model.matrix(~ - 1 + . ^ 2, data = as.data.frame(x))
group <- c(1:p, mapply(c, combn(1:p, 2, simplify = F), 1:choose(p, 2) + p, SIMPLIFY = F))
fit <- cv.grpsel(x.int, y, group)
colnames(x.int)[which(coef(fit)[- 1] != 0)]

