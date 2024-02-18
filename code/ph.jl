using MonotoneSplines
using DelimitedFiles
using Plots
using RCall

function compare_cv_error(; nfold = 5)
    methods = [MQB,     LOESS, ISO, mSI, mIS, MONMLP,           MCS,            MSS, cpspline, monpol, slse]
    name_methods = ["MQS", "QS", "LOESS", "Isotonic", "SI", 
                    "IS", "MONMLP (2x2)", "MONMLP", "MONMLP (32x2)", "MCS1", 
                    "MCS", "CS1", "CS", "MSS", "SS", 
                    "cpsplines", "MonoPoly", "SLSE"]

    data = readdlm("../data/ph.dat")
    x = data[:, 1]
    y = data[:, 2]
    submethods = [2, 1, 1, 1, 1, 3, 4, 2, 1, 1, 1]
    nmethod = sum(submethods)
    err = zeros(nfold, 3, nmethod)
    # calculate the cross-validation error
    n = length(x)
    folds = MonotoneSplines.div_into_folds(n, K = nfold, seed = -1)        
    for k = 1:nfold
        test_idx = folds[k]
        train_idx = setdiff(1:n, test_idx)
        for f in methods
            res = f(x[train_idx], y[train_idx], x[test_idx])
            if isa(res, Tuple)
                ni = 1
            else
                ni = length(res)
            end
            for j = 1:ni
                imethod += 1
                if ni == 1
                    yhat, y0hat = res
                else
                    yhat, y0hat = res[j]
                end
                err[k, :, imethod] .= [norm(y0hat - y[test_ids], p) for p in [1, 2, Inf]]
            end
        end
    end
    return err
end