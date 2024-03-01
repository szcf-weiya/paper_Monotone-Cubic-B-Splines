using LinearAlgebra
using StatsBase
using Serialization
using LaTeXTables
using LaTeXStrings
using MonotoneSplines
using Plots
using Random
using SpecialFunctions: erf
using RCall
curr_folder = @__DIR__
rfile = "$curr_folder/competitors.R"
R"source($rfile)"
using PyCall # for cpsplines (https://github.com/ManuelNavarroGarcia/cpsplines)
try
    _py_sys = pyimport("sys")
    pushfirst!(_py_sys."path", @__DIR__)
    _py_cps = pyimport("cpspline")        
catch
    @warn "failed to import cpspline python env"
end

fstep(x::Float64; xs::Vector{Float64}) = sum(x .> xs)

function slse(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    res = rcopy(R"slse($x, $y)")
    return res[:, 2], res[:, 2]
end

function cpspline(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    try
        return _py_cps."cps"(x, y, x0)
    catch e
        @warn "cpspline failed due to $e"
        return y, y
    end
end

function monpol(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where  T <: AbstractFloat
    res = R"mono.poly($x, $y, newx=$x0)"
    return rcopy(R"$res$fitted"), rcopy(R"$res$pred")
end

function mSI(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    res = R"mSI($x, $y, newx=$x0)"
    return rcopy(R"$res$fitted"), rcopy(R"$res$pred")
end

function mIS(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    res = R"mIS($x, $y, newx=$x0)"
    return rcopy(R"$res$fitted"), rcopy(R"$res$pred")
end

function ISO(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    res = R"pava($y[order($x)])"
    fitted = rcopy(R"$res[rank($x)]")
    pred = rcopy(R"approxfun(sort($x), $res)($x0)")
    return fitted, pred
end

function MQB(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    res = R"cobs($x, $y, constraint = 'increase', nknots=20, print.mesg = F)"
    fitted = rcopy(R"$res$fitted")
    pred = rcopy(R"predict($res, $x0)[, 2]")
    nknots = rcopy(R"length($res$knots)")
    res2 = R"bspline($x, $y, newx = $x0, degree = 2, J = $nknots - 2 + 3)"
    return [(fitted, pred), (rcopy(R"$res2$fitted"), rcopy(R"$res2$pred") )]
end

function LOESS(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    res = R"loess($y ~ $x)"
    fitted = rcopy(R"$res$fitted")
    pred = rcopy(R"predict($res, $x0)")
    return fitted, pred
end

function _MONMLP(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}; hidden1 = 10, hidden2 = 2) where T <: AbstractFloat
    res = R"monmlp::monmlp.fit(x=as.matrix($x), y=as.matrix($y), hidden1=$hidden1, monotone = 1, hidden2 = $hidden2, silent = TRUE)"
    fitted = rcopy(R"attr($res, 'y.pred')")[:]
    pred = rcopy(R"monmlp::monmlp.predict(x = as.matrix($x0), weights = $res)")
    if isa(pred, Matrix)
        pred = pred[:]
    end
    return fitted, pred
end

function MONMLP(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    return [_MONMLP(x, y, x0, hidden1 = 2), _MONMLP(x, y, x0, hidden1 = 4), _MONMLP(x, y, x0, hidden1 = 32)]
end

function MCS(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    Js = 4:50
    cverr = [MonotoneSplines.cv_err(x, y, J  = j, nfold=2) for j in Js]
    Jopt = Js[argmin(cverr)]
    res = R"mono3spl($x, $y, newx = $x0, J = $Jopt)"
    resb = R"bspline($x, $y, newx = $x0, degree = 3, J = $Jopt)"
    return [(rcopy(R"$res$fitted"), rcopy(R"$res$pred")),
            (rcopy(R"$res$fitted"), rcopy(R"$res$pred")), # keep to compatiable with MCB
            (rcopy(R"$resb$fitted"), rcopy(R"$resb$pred")),
            (rcopy(R"$resb$fitted"), rcopy(R"$resb$pred"))]
end

function MCB(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    res = R"mono3spl($x, $y, aic = TRUE)"
    J1 = rcopy(R"$res$j1")
    J2 = rcopy(R"$res$j2")
    res1 = R"mono3spl($x, $y, newx = $x0, J = $J1)"
    res2 = R"mono3spl($x, $y, newx = $x0, J = $J2)"
    res1b = R"bspline($x, $y, newx = $x0, degree = 3, J = $J1)"
    res2b = R"bspline($x, $y, newx = $x0, degree = 3, J = $J2)"
    return [(rcopy(R"$res1$fitted"), rcopy(R"$res1$pred")),
            (rcopy(R"$res2$fitted"), rcopy(R"$res2$pred")),
            (rcopy(R"$res1b$fitted"), rcopy(R"$res1b$pred")),
            (rcopy(R"$res2b$fitted"), rcopy(R"$res2b$pred"))]
end

function MSS(x::AbstractVector{T}, y::AbstractVector{T}, x0::AbstractVector{T}) where T <: AbstractFloat
    yhat_ss, yhatnew_ss, _, λ = MonotoneSplines.smooth_spline(x, y, x0)
    mss = mono_ss(x, y, λ)
    y0hat = predict(mss, x0)
    return [(mss.fitted, y0hat), (yhat_ss, yhatnew_ss)]
end

f_papp(x) = 5 + sum([erf(15i*(x-i/5)) for i = 1:4])

function compare(;nrep = 2, n = 30, nknot = 10, σ = 0.1, curve = "step", fig = false, k = 1, kw...)
    #          MQB, QB, LOESS, ISO, mSI, mIS, MONMLP(3), MCB1, MCB2, CB1, CB2, MSS, SS
    methods = [MQB,     LOESS, ISO, mSI, mIS, MONMLP,           MCS,            MSS, cpspline, monpol, slse]
    name_methods = ["MQS", "QS", "LOESS", "Isotonic", "SI", 
                    "IS", "MONMLP (2x2)", "MONMLP", "MONMLP (32x2)", "MCS1", 
                    "MCS", "CS1", "CS", "MSS", "SS", 
                    "cpsplines", "MonoPoly", "SLSE"]
    cols = [:blue, :blue, :orange, :purple, :purple, 
            :purple, :green, :green, :green, :red, 
            :red, :red, :red, :cyan, :cyan,
            :magenta, :gray]
    lty = [:solid, :dash, :solid, :solid, :dash, 
            :dot, :solid, :solid, :solid, :dot, 
            :solid, :dashdot, :dash, :solid, :dash,
            :solid, :solid]
    submethods = [2, 1, 1, 1, 1, 3, 4, 2, 1, 1, 1]
    nmethod = sum(submethods)
    err = zeros(nrep, ifelse(k==1, 3, 2), nmethod) #　fitting & prediction
    if nrep > 1
        fig = false
    end
    if fig
        p = plot(;legend_cell_align = "left", extra_kwargs = :subplot, kw...)
    end
    for i = 1:nrep
        if curve == "step"
            knots = rand(nknot) * 2 .- 1
            x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->fstep(z, xs=knots), k = k)
        elseif curve == "growth"
            x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->1/(1-0.42log(z)), xmin = 0, xmax = 10, k = k)
        elseif curve == "logit"
            x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->exp(z)/(1+exp(z)), xmin = -5, xmax = 5, k = k)
        elseif curve == "erf"
            x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, f_papp, xmin = 0, xmax = 1, k = k) 
        else
            x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->z^3, k = k)
        end
        if fig
            scatter!(p, x, y, label = "", color = :black, markersize=1.5)
            plot!(p, x0, y0, ls = :solid, label = "Truth", color = :black, legend_cell_align = "left", extra_kwargs = :subplot)
        end
        imethod = 0
        for f in methods
            res = f(x, y, x0) # SLSE need to be scaled to [0, 1]
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
                if fig
                    if name_methods[imethod] in ["MONMLP (2x2)", "MONMLP (32x2)", "MCS1", "CS1"]
                        continue
                    end
                    lw = ifelse(lty[imethod] == :solid, 0.8, 1.2)
                    plot!(p, x0, y0hat, label = name_methods[imethod], color = cols[imethod], ls = lty[imethod], lw=lw ;legend_cell_align = "left", extra_kwargs = :subplot)
                end
                if k == 1
                    err[i, :, imethod] .= [norm(yhat - y0, p) for p in [1, 2, Inf]]
                else
                    err[i, :, imethod] .= [norm(yhat - y)^2 / length(y), norm(y0hat - y0)^2 / length(y0)]
                end
            end
        end
    end
    if fig
        return p
    end
    # return err
    return mean(err, dims = 1)[1, :, :], std(err, dims = 1)[1, :, :] / sqrt(nrep)
end

function experiments_slse(nrep = 100)
    σs = [0.1, 1.0, 1.5]
    err = zeros(nrep, 3, 4, 3)
    err_erf = zeros(nrep, 2, 3)
    for i = 1:nrep
        for (j, σ) in enumerate(σs)
            for (k, curve) in enumerate(["poly3", "logit", "growth", "step"])
                err[i, j, k, :] .= single_experiment_slse(curve = curve, σ = σ)
            end
        end
        for (j, σ) in enumerate([0.15, 0.3])
            err_erf[i, j, :] .= single_experiment_slse(curve = "erf", σ = σ)
        end
    end
    μerr = mean(err, dims = 1)
    σerr = std(err, dims = 1) / sqrt(nrep)
    μerr_erf = mean(err_erf, dims = 1)
    σerr_erf = std(err_erf, dims = 1) / sqrt(nrep)
    return μerr, σerr, μerr_erf, σerr_erf
    #serialize(joinpath(resfolder, ""))
    serialize("../output/sim/slse_2023-09-26T21_27_06-04_00_nrep100.sil", res)
end

function single_experiment_slse(; nknot = 10, n = 100, k = 1, curve = "step", σ = 0.1)
    f_growth = z->1/(1-0.42log(z))
    f_logit = z->exp(z)/(1+exp(z))
    f_cubic = z->z^3
    if curve == "step"
        knots = rand(nknot) * 2 .- 1
        x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->fstep(z, xs=knots), k = k)
        x = (x .+ 1.0) ./ 2
    elseif curve == "growth"
        x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, f_growth, xmin = 0, xmax = 10, k = k)
        x = x ./ 10
    elseif curve == "logit"
        x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, f_logit, xmin = -5, xmax = 5, k = k)
        x = x ./ 10 .+ 0.5
    elseif curve == "erf"
        x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, f_papp, xmin = 0, xmax = 1, k = k) 
    else
        x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, f_cubic, k = k)
        x = (x .+ 1.0) ./ 2
    end
    ypred, ypred = slse(x, y, x0)
    ygrid = y0 # modified the original code to output the fit at each x; otherwise, 
    ### use the following code, but the comparison would unfair
    # if curve == "step"
    #     ygrid = fstep.(2xgrid .- 1, xs = knots)
    # elseif curve == "growth"
    #     ygrid = f_growth.(10xgrid)
    # elseif curve == "logit"
    #     ygrid = f_logit.(10xgrid .- 5)
    # elseif curve == "erf"
    #     ygrid = f_papp.(xgrid)
    # else
    #     ygrid = f_cubic.(2xgrid .- 1)
    # end
    #return [norm(ypred - ygrid, p) for p in [1, 2, Inf]]
    return [norm(ypred - ygrid, 1)/n, norm(ypred - ygrid, 2)/sqrt(n), norm(ypred - ygrid, Inf)]
end

function experiments(nrep = 100)
    timestamp = replace(strip(read(`date -Iseconds`, String)), ":" => "_")
    for σ in [0.1, 1.0, 1.5]
        for curve in ["step", "logit", "growth", "poly3"]
            @info "σ = $σ, curve = $curve"
            serialize("../output/sim/$(curve)_$(σ)_$(timestamp)_nrep$nrep.sil", compare(nrep = nrep, curve = curve, σ = σ, k = 1, n = 100))
        end
    end
    for σ in [0.15, 0.3]
        curve = "erf"
        @info "σ = $σ, curve = $curve"
        serialize("../output/sim/$(curve)_$(σ)_$(timestamp)_nrep$nrep.sil", compare(nrep = nrep, curve = curve, σ = σ, k = 1, n = 100))
    end
end

function write2tables(; include_slse = true, one_se_rule = true)
    # timestamp = "2022-07-04T11_07_25+08_00"
    # nrep = 100
    #timestamp = "2022-07-04T12_19_32+08_00"
    #nrep = 1000
    # timestamp = "2023-01-17T17_20_41-05_00"
    # timestamp = "2023-09-25T17_54_13-04_00"
    timestamp = "2023-09-27T13_14_34-04_00"
    nrep = 100
    resfolder = "../output/sim/"
    all_method_lbl = ["MQB", "QB", "LOESS", "Isotonic", "SI", 
                      "IS", "MONMLP (2x2)", "MONMLP (4x2)", "MONMLP (32x2)", "MCS1", 
                      "MCS2", "CS1", "CS2", "MSS", "SS", 
                      "cpsplines", "MonoPoly", "SLSE"]
    #sel_idx = [12, 10, 15, 14, 2, 1, 3, 4, 5, 6, 9, 16, 17]
    
    if include_slse
        # sel_idx = [12, 10, 15, 14, 2, 1, 3, 4, 5, 6, 17, 9, 16, 18]
        #sel_idx = [13, 11, 15, 14, 2, 1, 3, 4, 5, 6, 17, 9, 16, 19] # 18->19, discard the SI from experiments (due to unscaled)
        sel_idx = [11, 14, 13, 15, 2, 1, 3, 4, 5, 6, 17, 9, 16, 19] # 18->19, discard the SI from experiments (due to unscaled)
        res_slse = deserialize(joinpath(resfolder, "slse_2023-09-26T21_27_06-04_00_nrep100.sil"))
    else
        sel_idx = [13, 11, 15, 14, 2, 1, 3, 4, 5, 6, 17, 9, 16, 18]
    end
    all_method_lbl_full = ["\\textcite{heMonotoneBsplineSmoothing1998}: MQS", "Quadratic Spline (QS)", "LOESS", "Isotonic", "\\textcite{mammenEstimatingSmoothMonotone1991}: SI (LOESS+Isotonic)", 
                    "\\textcite{mammenEstimatingSmoothMonotone1991}: IS (Isotonic+LOESS)", "MONMLP (2x2)", "MONMLP (4x2)", "\\textcite{cannonMonmlpMultilayerPerceptron2017}: MONMLP", "Monotone CS (MCS)", 
                    "Monotone CS (MCS)", "Cubic Spline (CS)", "Cubic Spline (CS)", "Monotone SS (MSS)", "Smoothing Spline (SS)", 
                    "\\textcite{navarro-garciaConstrainedSmoothingOutofrange2023}: cpsplines", "\\textcite{murrayFastFlexibleMethods2016}: MonoPoly",
                    "\\textcite{groeneboomConfidenceIntervalsMonotone2023}: SLSE",
                    "\\textcite{groeneboomConfidenceIntervalsMonotone2023}: SLSE"]
    cubic = Array{Any, 1}(undef, 3)
    logit = Array{Any, 1}(undef, 3)
    growth = Array{Any, 1}(undef, 3)
    steps = Array{Any, 1}(undef, 3)
    erfs = Array{Any, 1}(undef, 2)
    for (i, σ) in enumerate([0.1, 1.0, 1.5])
        cubic[i] = deserialize(joinpath(resfolder, "poly3_$(σ)_$(timestamp)_nrep$nrep.sil"))
        logit[i] = deserialize(joinpath(resfolder, "logit_$(σ)_$(timestamp)_nrep$nrep.sil"))
        growth[i] = deserialize(joinpath(resfolder, "growth_$(σ)_$(timestamp)_nrep$nrep.sil"))
        steps[i] = deserialize(joinpath(resfolder, "step_$(σ)_$(timestamp)_nrep$nrep.sil"))
    end
    for (i, σ) in enumerate([0.15, 0.3])
        erfs[i] = deserialize(joinpath(resfolder, "erf_$(σ)_$(timestamp)_nrep$nrep.sil"))
    end

    for i = 1:2
        for j = 1:2
            erfs[i][j][1, :] ./= 100 # L1 scale by 100
            erfs[i][j][2, :] ./= 10 # L2 scale by 10
        end
    end
    if include_slse
        # μerr is res[3] & σerr is res[4]
        for i = 1:2
            erfs[i] = (hcat(erfs[i][1], res_slse[3][1, i, :]), # mean
                       hcat(erfs[i][2], res_slse[4][1, i, :])) # se
        end
    end
    μerr = [erfs[i][1][:, sel_idx]' for i = 1:2]
    σerr = [erfs[i][2][:, sel_idx]' for i = 1:2]
    n, m = size(μerr[1])
    isbf = [zeros(Bool, n, m), zeros(Bool, n, m)]
    rks = [zeros(Int, n, m) for i=1:2]
    for i = 1:2
        for j = 1:3
            idx = argmin(μerr[i][:, j])
            if one_se_rule
                cutoff = μerr[i][idx, j] + σerr[i][idx, j]
                isbf[i][μerr[i][:, j] .< cutoff, j] .= 1
            else
                isbf[i][idx, j] .= 1
            end
            rks[i][:, j] .= sortperm(sortperm(μerr[i][:, j]))
        end
    end
    texfile = joinpath(resfolder, "erf_$(timestamp)_nrep100.tex")
    colname = [raw"$\frac 1n L_1$", raw"$\frac{1}{\sqrt n}L_2$", raw"$L_\infty$"]
    print2tex(μerr, σerr, ["0.15", "0.3"], [""], all_method_lbl_full[sel_idx], colname, 
                file = texfile,
                isbf = isbf, 
                rank_sup = rks,
                colnames_of_rownames = ["Noise \$\\sigma\$", "Method"])
    tex2png(texfile)
    # other curves
    curve_names = ["poly3", "logit", "growth", "step"]
    for (k, curve) in enumerate([cubic, logit, growth, steps])
        for i = 1:3 # noise level
            for j = 1:2 # mean and sd
                curve[i][j][1, :] ./= 100 # scale L1 by n
                curve[i][j][2, :] ./= 10 # scale L2 by sqrt(n)
            end
        end
        if include_slse
            for i = 1:3
                curve[i] = (hcat(curve[i][1], res_slse[1][1, i, k, :]),
                            hcat(curve[i][2], res_slse[2][1, i, k, :]))
            end
        end
        μerr = [curve[i][1][:, sel_idx]' for i = 1:3]
        σerr = [curve[i][2][:, sel_idx]' for i = 1:3]
        n, m = size(μerr[1])
        isbf = [zeros(Bool, n, m) for i = 1:3]
        rks = [zeros(Int, n, m) for i = 1:3]
        for i = 1:3
            for j = 1:3
                idx = argmin(μerr[i][:, j])
                if one_se_rule
                    cutoff = μerr[i][idx, j] + σerr[i][idx, j]
                    isbf[i][μerr[i][:, j] .< cutoff, j] .= 1
                else
                    isbf[i][idx, j] .= 1
                end
                rks[i][:, j] = sortperm(sortperm(μerr[i][:, j]))
            end
        end
        texfile = joinpath(resfolder, "$(curve_names[k])_$(timestamp)_nrep100.tex")
        print2tex(μerr, σerr, ["0.1", "1.0", "1.5"], [""], all_method_lbl_full[sel_idx], colname,
                  file = texfile,
                  isbf = isbf, rank_sup = rks,
                  colnames_of_rownames = ["Noise \$\\sigma\$", "Method"])
        tex2png(texfile)
    end
end

# deprecated
function combind_tables()
    μ = [vcat(logit[i][1], growth[i][1])' for i = 1:3]
    σ = [vcat(logit[i][2], growth[i][2])' for i = 1:3]
    n, m = size(μ[1])
    isbf = [zeros(Bool, n, m), zeros(Bool, n, m), zeros(Bool, n, m)]
    for i = 1:3
        isbf[i][argmin(μ[i][:, 2]), 2] = 1
        isbf[i][argmin(μ[i][:, 4]), 4] = 1
    end
    print2tex(μ, σ, ["0.1", "1.0", "1.5"], ["Logistic Curve", "Growth Curve"], method_lbl, ["MSFE", "MSPE"], file = "$folder/$(timestamp)_logit_and_growth.tex", isbf = isbf, colnames_of_rownames = ["Noise \$\\sigma\$", "Method"])

    μ = [vcat(cubic[i][1], steps[i][1])' for i = 1:3]
    σ = [vcat(cubic[i][2], steps[i][2])' for i = 1:3]
    n, m = size(μ[1])
    isbf = [zeros(Bool, n, m), zeros(Bool, n, m), zeros(Bool, n, m)]
    for i = 1:3
        isbf[i][argmin(μ[i][:, 2]), 2] = 1
        isbf[i][argmin(μ[i][:, 4]), 4] = 1
    end
    print2tex(μ, σ, ["0.1", "1.0", "1.5"], ["Cubic Curve", "Step Curve"], method_lbl, ["MSFE", "MSPE"], file = "$folder/$(timestamp)_cubic_and_steps.tex", isbf = isbf, colnames_of_rownames = ["Noise \$\\sigma\$", "Method"])
end

function demo_plots()
#    timestamp = replace(strip(read(`date -Iseconds`, String)), ":" => "_")
    #timestamp = "2023-01-17T17_14_25-05_00"
    timestamp = "2023-09-25T17_54_13-04_00"
    seed = 5
    Random.seed!(seed)
    compare(;nrep=1,fig=true,curve="logit", σ = 0.3, n=100, title = latexstring("Logistic Curve (\$\\sigma = 0.3\$)"))
    savefig("../output/sim/demo_logit_seed$seed-$timestamp.pdf")
    Random.seed!(seed)
    compare(nrep=1,fig=true,curve="erf", σ = 0.3, n=100, title = latexstring("Error Function Curve (\$\\sigma = 0.3\$)") )
    savefig("../output/sim/demo_erf_seed$seed-$timestamp.pdf")
end

function compare_papp()
    f_papp(x) = 5 + sum([erf(15i*(x-i/5)) for i = 1:4])
    σs = [0.15, 0.3]
    n = 100
    nrep = 100
    errs = zeros(2, nrep, 3, 3)
    for (j, σ) in enumerate(σs)
        for i = 1:nrep
            @info "σ = $σ, i = $i"
            x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, f_papp, xmin = 0, xmax = 1, k = 1) 
            # λs = exp.(-20:0.1:2)
            # cverr, B, L, J = cv_mono_ss(x, y, λs)
            # λopt = λs[argmin(cverr)]
            # mss = mono_ss(x, y, λopt)
            yhat_ss, yhatnew_ss, _, λ = MonotoneSplines.smooth_spline(x, y, x0)
            mss = mono_ss(x, y, λ)        
            yhat, y0hat = cpspline(x, y, x0)
            errs[j, i, :, 1] .= [norm(mss.fitted - y0, p) for p in [1, 2, Inf]] / n
            errs[j, i, :, 2] .= [norm(yhat_ss - y0, p) for p in [1, 2, Inf]] / n
            errs[j, i, :, 3] .= [norm(yhat - y0, p) for p in [1, 2, Inf]] / n
        end
    end
    μerr = mean(errs, dims = 2)
    σerr = std(errs, dims = 2)

    # demo 
    Random.seed!(1234)
    x, y, x0, y0 = MonotoneSplines.gen_data(100, σ, f_papp, xmin = 0, xmax = 1, k = 10) 
    yhat_ss, yhatnew_ss = MonotoneSplines.smooth_spline(x, y, x0)
    yhat, y0hat = cpspline(x, y, x0)
    y0mss = predict(mss, x0)
    scatter(x, y, ms = 2, label = "")
    plot!(x0, y0, label = "truth", ls = :dash, lw = 1)
    #plot!(x, yhat_ss, label = "SS", lw = 1.5)
    plot!(x0, yhatnew_ss, label = "SS", lw = 1.5)
    # plot!(x, mss.fitted, label = "MSS", lw = 2)
    plot!(x0, y0mss, label = "MSS", lw = 2)
    # plot!(x, yhat, label = "cpsplines", lw = 2)
end
