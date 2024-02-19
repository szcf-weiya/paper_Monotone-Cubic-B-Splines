using MonotoneSplines
using DelimitedFiles
using Plots
using RCall
using Random
using Serialization

function compare_cv_error(; nfold = 5)
    # slse did not offer a prediction method
    methods = [MQB,     LOESS, ISO, mSI, mIS, MONMLP,           MCS,            MSS, cpspline, monpol]#, slse]
    name_methods = ["MQS", "QS", "LOESS", "Isotonic", "SI", 
                    "IS", "MONMLP (2x2)", "MONMLP", "MONMLP (32x2)", "MCS1", 
                    "MCS", "CS1", "CS", "MSS", "SS", 
                    "cpsplines", "MonoPoly"]#, "SLSE"]

    data = readdlm("../data/ph.dat")
    x = data[:, 1]
    y = data[:, 2]
    y = -y # since it is decreasing, let it be increasing
    submethods = [2, 1, 1, 1, 1, 3, 4, 2, 1, 1]#, 1]
    nmethod = sum(submethods)
    err = zeros(nfold, 3, nmethod)
    # calculate the cross-validation error
    n = length(x)
    folds = MonotoneSplines.div_into_folds(n, K = nfold, seed = -1)        
    for k = 1:nfold
        test_idx = folds[k]
        train_idx = setdiff(1:n, test_idx)
        imethod = 0
        for f in methods
            println(string(f))
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
                # println(y0hat)
                if sum(ismissing.(y0hat)) > 0
                    err[k, :, imethod] .= NaN    
                else
                    err[k, :, imethod] .= [try norm(y0hat - y[test_idx], p) catch e NaN end for p in [1, 2, Inf]]
                end
            end
        end
    end
    return err
end

function run_experiments()
    Random.seed!(1234)
    res10 = compare_cv_error(nfold = 10)
    serialize("../output/real/res10.sil", res10)
    res256 = compare_cv_error(nfold = 256)
end