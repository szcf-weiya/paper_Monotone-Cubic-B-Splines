using MonotoneSplines
using DelimitedFiles
using Plots
using RCall
using Random
using Serialization

# res16 = compare_cv_error(nfold = 16, seed = 1)
function compare_cv_error(; nfold = 5, seed = 1)
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
    folds = MonotoneSplines.div_into_folds(n, K = nfold, seed = seed)        
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
                if ismissing(y0hat) # a scalar when LOOCV
                    y0hat = mean(y[train_idx])
                else
                    if sum(ismissing.(y0hat)) > 0
                        y0hat[ismissing.(y0hat)] .= mean(y0hat[.!ismissing.(y0hat)])
                    end
                end
                err[k, :, imethod] .= [
                    try 
                        norm(y0hat .- y[test_idx], p) 
                    catch e 
                        norm(mean(y[train_idx]) .- y[test_idx], p) 
                    end for p in [1, 2, Inf]]
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

function write2table(resfile = "../output/real/res10.sil")
    all_method_lbl = ["MQB", "QB", "LOESS", "Isotonic", "SI", 
    "IS", "MONMLP (2x2)", "MONMLP (4x2)", "MONMLP (32x2)", "MCS1", 
    "MCS2", "CS1", "CS2", "MSS", "SS", 
    "cpsplines", "MonoPoly", "SLSE"]
    sel_idx = [13, 11, 15, 14, 2, 1, 3, 4, 5, 6, 17, 9, 16]
    all_method_lbl_full = ["\\textcite{heMonotoneBsplineSmoothing1998}: MQS", "Quadratic Spline (QS)", "LOESS", "Isotonic", "\\textcite{mammenEstimatingSmoothMonotone1991}: SI (LOESS+Isotonic)", 
                    "\\textcite{mammenEstimatingSmoothMonotone1991}: IS (Isotonic+LOESS)", "MONMLP (2x2)", "MONMLP (4x2)", "\\textcite{cannonMonmlpMultilayerPerceptron2017}: MONMLP", "Monotone CS (MCS)", 
                    "Monotone CS (MCS)", "Cubic Spline (CS)", "Cubic Spline (CS)", "Montone SS (MSS)", "Smoothing Spline (SS)", 
                    "\\textcite{navarro-garciaConstrainedSmoothingOutofrange2023}: cpsplines", "\\textcite{murrayFastFlexibleMethods2016a}: MonoPoly",
                    "\\textcite{groeneboomConfidenceIntervalsMonotone2023}: SLSE",
                    "\\textcite{groeneboomConfidenceIntervalsMonotone2023}: SLSE"]

    res = deserialize(resfile)
    μerr = mean(res, dims = 1)[1, :, sel_idx]'
    σerr = std(res, dims = 1)[1, :, sel_idx]' / sqrt(10) # 10 fold
    n, m = size(μerr)
    isbf = zeros(Bool, n, m)
    rks = zeros(Int, n, m)
    for j = 1:3
        idx = argmin(μerr[:, j])
        cutoff = μerr[idx, j] + σerr[idx, j]
        isbf[μerr[:, j] .< cutoff, j] .= 1
        isbf[idx, j] = 1
        rks[:, j] .= sortperm(sortperm(μerr[:, j]))
    end
    output_file = resfile[1:end-4] * ".tex"
    colname = [raw"$L_1$", raw"$L_2$", raw"$L_\infty$"]
    print2tex([Matrix(μerr)], [Matrix(σerr)], [""], [""], all_method_lbl_full[sel_idx], colname, 
                colnames_of_rownames = ["Method"], 
                file = output_file,
                isbf = [isbf], rank_sup = [rks])
end