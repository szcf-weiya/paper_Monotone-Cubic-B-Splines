library(cobs)
library(Iso)
library(lsei)
library(splines)
library(MonoPoly)
library(Rcpp)
sourceCpp("ext/SLSE.cpp")

slse = function(x, y) {
    res = SLSE(cbind(x, y), length(x))
    #res$SLSE # two columns, grid_x and fitted value
    res$fitted 
}

mono.poly = function(x, y, newx = NULL) {
    m = monpol(y ~ x)
    yhat = fitted(m)
    if (is.null(newx))
        ynew.hat = yhat
    else
        ynew.hat = as.numeric(predict(m, newx))
    list(fitted = yhat, pred = ynew.hat)    
}

# first smoothing then isotonic
mSI = function(x, y, newx = NULL) {
    # smooth by loess
    ys = loess(y ~ x)$fitted
    # isotonic
    yhat = pava(ys[order(x)])
    if (is.null(newx))
        yhat[rank(x)]
    else
        list(fitted=yhat[rank(x)], pred=approxfun(sort(x), yhat)(newx))
}
# first isotonic then smoothing
mIS = function(x, y, newx = NULL) {
    yi = pava(y[order(x)])[rank(x)]
    fit.lo = loess(yi ~ x)
    if (is.null(newx))
        fit.lo$fitted
    else
        list(fitted=fit.lo$fitted, pred=predict(fit.lo, newx))
}

mono3spl.aic = function(x, y, J = 10, alpha = 1, tol=1e-6) {
    ord = order(x)
    H = bs(x[ord], df = J, intercept = TRUE)
    A = diag(J)
    diag(A[1:J-1, 2:J]) = -1
    b = numeric(J-1)
    beta = lsi(H, y[ord], e = -A[1:J-1,]*alpha, f = b)
    n_eq = sum(abs(beta[1:(J-1)] - beta[2:J]) < tol)
    yhat = H[rank(x),] %*% beta
    aic = length(x) * log(sum((yhat - y)^2)) + 2 * (J - n_eq)
    list(aic = aic, df = J - n_eq)
}

bspline = function(x, y, newx = NULL, J = 10, degree = 3) {
    ord = order(x)
    H = bs(x[ord], df = J, intercept = TRUE, degree = degree)
    beta = lm(y[ord] ~ 0+H)$coefficients
    if (is.null(newx))
        H[rank(x),] %*% beta
    else
        list(fitted = H[rank(x),] %*% beta, pred = predict(H, newx) %*% beta)
}

mono3spl = function(x, y, newx=NULL, J = 10, aic = FALSE, alpha = 1) {
    if (aic) {
        aics = c()
        dfs = c()
        # first local minimizer
        J.local = 4
        isfound = FALSE
        for (j in 4:length(x)) {
            res = mono3spl.aic(x, y, J = j)
            aic = res$aic
            df = res$df
#             cat("J = ", j, "; AIC = ", aic, "\n", sep = "")
            aics = c(aics, aic)
            dfs = c(dfs, df)
            if (!isfound & j > 4) {
                if (aics[j-3] < aics[j-4]) {
                    J.local = j
                }
                else
                    isfound = TRUE
            }
        }
        J.global = which.min(aics) + 3
        list(aics = aics, j1 = J.local, j2 = J.global, df1 = dfs[J.local-3], df2 = dfs[J.global-3])
    } else {
        ord = order(x)
        H = bs(x[ord], df = J, intercept = TRUE)
        A = diag(J)
        diag(A[1:J-1, 2:J]) = -1
        b = numeric(J-1)
        beta = lsi(H, y[ord], e = -A[1:J-1,]*alpha, f = b)
        if (is.null(newx))
            H[rank(x),] %*% beta
        else
            list(fitted = H[rank(x),] %*% beta, pred = predict(H, newx) %*% beta)
    }
}
