using Serialization
using LaTeXTables
using Plots
using StatsBase
using LaTeXStrings
using Printf
using Random
using MonotoneSplines
using PyCall

function change_ns()
    λs = exp.(range(-10, -4, length = 100))
    ns = [100, 500, 1000, 5000, 10000, 50000, 100000]
    #ns = [100, 1000]
    ts = zeros(length(ns))
    for (i, n) in enumerate(ns)
        x, y, x0, y0 = gen_data(n, 0.2, z->z^3, xmin = -1, xmax = 1)
        ts[i] = @elapsed errs, B, L, J = MonotoneSplines.cv_mono_ss(x, y, λs)
    end        
end

function read_folder_ns(folder; σ = 0.2, f = "cubic", ns = [50, 100, 200, 500, 1000], keywords = [])
    files = readdir(folder, join = true)
    nn = length(ns)
    RES = Array{Any, 1}(undef, nn)
    for file in files
        if endswith(file, "sil")
            if !isempty(keywords)
                if !all([occursin(k, file) for k in keywords]) # attention the keyword might appear in the folder, so be specific
                    continue
                end
            end
            j = findfirst(occursin.("n" .* string.(ns) .* "-", file))
            RES[j] = deserialize(file)
        end
    end
    return RES
end

function read_folder(folder; σs = [0.01, 0.1, 0.2, 0.5], fs = ["cubic", "logit", "exp", "logit5"],
                             is_rep = false, # if true, then each (f, σ) can have multiple records
                             keywords = [])
#    files = readdir("res/ci_monofit/9f56d34/", join = true)
    files = readdir(folder, join = true)
    nf = length(fs)
    nσ = length(σs)
    RES = Array{Any, 1}(undef, nf)
    for i = 1:nf
        RES[i] = Array{Any, 1}(undef, nσ)
        if is_rep
            for j = 1:nσ
                RES[i][j] = []
            end
        end
    end
    for file in files
        if endswith(file, "sil")
            if !isempty(keywords)
                if !all([occursin(k, file) for k in keywords]) # attention the keyword might appear in the folder, so be specific
                    continue
                end
            end
            # i = findfirst([occursin("$(fname)-", file) for fname in fs])
            # j = findfirst([occursin("σ$σ", file) for σ in σs])
            # alternatively
            i = findfirst(occursin.(fs, file))
            j = findfirst(occursin.("σ" .* string.(σs), file))

            if !isnothing(i) & !isnothing(j) # allow σs be subset of σs in the folder, e.g., i is not nothing for file σ0.02, but if 0.02 is not in σs, then j is nothing
                res = deserialize(file)
                if is_rep
                    push!(RES[i][j], res)
                else
                    RES[i][j] = res
                end
            end
        end
    end
    return RES        
end

function plot_summary(RES, fs = ["logit5", "exp", "cubic"], σs = [0.1, 0.2, 0.5, 1.0])
    nf = length(fs)
    nσ = length(σs)
    txt_σs = reshape(["σ$σ" for σ in σs], 1, nσ)
    figs = Array{Plots.Plot, 1}(undef, nf)
    for i = 1:nf
        yhat_diff = hcat([mean(RES[i][j][1],dims=1)[1,:,1] for j = 1:nσ]...)

        rel_yhat_diff = hcat([mean(RES[i][j][1][:, :, 1] ./ RES[i][j][1][:, :, 2], dims=1)[:] for j = 1:nσ]...)
        # ratio of the fittness
        #                                   fitness_yGMS / fitness_monoOpt
        ratio_fitness = hcat([mean(RES[i][j][1][:,:,3] ./ RES[i][j][1][:,:,2], dims=1)[:] for j=1:nσ]...)
        figs[i] = plot(plot(log.(ratio_fitness), title="$(fs[i]) ratio fittness", xlab = "lambda", label = txt_σs),
            #  plot(yhat_diff, title="cubic yhat difference", xlab = "lambda", label = ["σ0.1" "σ0.2" "σ0.5" "σ1.0"]
            plot(log.(rel_yhat_diff), title="relative $(fs[i]) yhat difference", xlab = "lambda", label = txt_σs
            ), size=(1200, 600))        
    end
    return figs
end


function summary_in_table(resfolder = "../output/acc_monofit/c80aff2/nepoch05",
                          σs = [0.1, 0.2, 0.5], 
                          fs = ["logit5", "exp", "cubic", "sinhalfpi"];
                          simplify = false,
                          idx_acc = 3, # depends on the experiment output, if using `check_acc`, then take 1 since the first location is the error accuracy
                          fs_tex = [L"S(5x)", L"e^x", L"x^3", L"\sin(\pi x/2)"],
                          )
    RES = read_folder(resfolder, σs = σs, fs = fs)
    nσ = length(σs)
    nf = length(fs)
    μRES = Array{Matrix, 1}(undef, nσ)
    σRES = Array{Matrix, 1}(undef, nσ)
    tabname = "res.tex"
    if simplify
        tabname = "res_simplify.tex"
    end
    idx_λ = 1:10
    for j = 1:nσ
        # idx_λ = [1, 4, 7, 10]
        nλ = length(idx_λ)
        if simplify
            nλ = 3 # λ_l, λ_u, E λ
        end
        μRES[j] = zeros(nf, 2nλ) # relative gap & fitness ratio
        σRES[j] = zeros(nf, 2nλ)
        for i = 1:nf
            if simplify
                err_ij1 = mean(RES[i][j][idx_acc], dims=1)[1,[1, end],1:2]
                err_ij1_σ = std(RES[i][j][idx_acc], dims=1)[1, [1, end], 1:2]
                err_ij2 = mean(RES[i][j][idx_acc], dims=[1,2])[1, :, 1:2] # mean over all lambdas, results in 1x2 matrix due to `:` 
                err_ij2_σ = std(RES[i][j][idx_acc], dims=[1,2])[1, :, 1:2]
                err_ij = vcat(err_ij1, err_ij2)
                err_ij_σ = vcat(err_ij1_σ, err_ij2_σ)
            else
                err_ij = mean(RES[i][j][idx_acc], dims=1)[1,idx_λ,1:2]
                err_ij_σ = std(RES[i][j][idx_acc], dims=1)[1,idx_λ,1:2]
            end
            μRES[j][i, :] .= reshape(err_ij, 2nλ) # no transpose such that (λ1, λ2, ...) appears in order
            σRES[j][i, :] .= reshape(err_ij_σ, 2nλ)
        end
    end
    subcolnames = [latexstring("\\lambda_{$i}") for i in idx_λ]
    if simplify
        subcolnames = [L"\lambda_l", L"\lambda_u", "Avg."]
    end
    # print2tex(μRES, ["sigma = $σ" for σ in σs], ["Relative Gap", "Fitness Ratio"], 
    #             subcolnames = subcolnames, 
    #             subrownames = fs, 
    #             colnames_of_rownames = ["noise", "curve"],
    #             file = joinpath(resfolder, tabname))
    print2tex(μRES, σRES, [latexstring("\$\\sigma = $σ\$") for σ in σs], ["Relative Gap", "Fitness Ratio"], 
                fs_tex,
                subcolnames, 
                colnames_of_rownames = ["noise", "curve"],
                file = joinpath(resfolder, tabname))
    try
        tex2png(joinpath(resfolder, tabname))
    catch
        @warn "fail to run tex2png"
    end
end

function plot_jaccard(; resfile = "res/ci_monofit/3e487fc/demo-CI-exp-n100-σ0.2-seed175-B2000-K010-K10-nepoch550000-prop0.2-2022-11-20T12_23_44+08_00.sil")
    resfolder = dirname(resfile)
    x, y, λs, J, Yhat, Yhat0, RES_YCI, cp = deserialize(resfile)
    RES_YCI0 = deserialize(resfile[1:end-4] * "_supp.sil")
    nλ = length(λs)
    overlaps = zeros(nλ)
    for i = 1:nλ
        overlaps[i] = MonotoneSplines.jaccard_index(RES_YCI[i], RES_YCI0[i])
    end
    plot(log.(λs), overlaps, markershape = :star5, label = "", xlab = "log λ", ylab = "Jaccard Index")
end

function plot_demo_ci(; resfile = "res/ci_monofit/3e487fc/demo-CI-exp-n100-σ0.2-seed175-B2000-K010-K10-nepoch550000-prop0.2-2022-11-20T12_23_44+08_00.sil", 
                        idx_lambda = [1, 10, 13, 20])
    resfolder = dirname(resfile)
    x, y, λs, J, Yhat, Yhat0, RES_YCI, cp = deserialize(resfile)
    RES_YCI0 = deserialize(resfile[1:end-4] * "_supp.sil")
    f = identity
    if occursin("exp", resfile)
        f = exp
    elseif occursin("cubic", resfile)
        f = x->x^3
    end
    idx = sortperm(x)
    
    #for (ii, λ) in enumerate(λs[idx_lambda])
    for i in idx_lambda
        overlap = MonotoneSplines.jaccard_index(RES_YCI[i], RES_YCI0[i])
        λ = λs[i]
        lw_ci = 0.5
        colors = [:gray, :orange, :blue]
        fillalpha = 0.3
        fig = scatter(x, y, label = "", legend = :topleft, color = :black, markersize = 1, 
                      title = "λ = $(@sprintf "%.3e" λ), overlap = $(@sprintf "%.3f" overlap)",
                      xlab = L"x", ylab = L"y",
                      labelfontsize = 14, legendfontsize = 14, tickfontsize = 11)
        plot!(fig, x[idx], f.(x[idx]), label = "truth", color = colors[1], lw = 0.5)
        plot!(fig, x[idx], Yhat0[i, idx], label = "OPT", color = colors[3], ls = :dot, lw = 1)
        plot!(fig, x[idx], Yhat[i, idx], label = "MLP", color = colors[2], ls = :dash, lw = 1)
        cp1 = MonotoneSplines.coverage_prob(RES_YCI[i], f.(x))
        cp2 = MonotoneSplines.coverage_prob(RES_YCI0[i], f.(x))
        plot!(fig, x[idx], RES_YCI0[i][idx, 1], fillrange = RES_YCI0[i][idx, 2], fillcolor = colors[3], fillalpha=fillalpha, linealpha=0, label = "CI (OPT) prob = $cp2")
        plot!(fig, x[idx], RES_YCI[i][idx, 1], fillrange = RES_YCI[i][idx, 2], fillcolor = colors[2], fillalpha=fillalpha, linealpha=0, label = "CI (MLP) prob = $cp1")
        # plot!(fig, x[idx], RES_YCI0[i][idx, 1], ls = :dot, lw = lw_ci, color = colors[3], label = "")
        # plot!(fig, x[idx], RES_YCI0[i][idx, 2], ls = :dot, lw = lw_ci, color = colors[3], label = "")
        # plot!(fig, x[idx], RES_YCI[i][idx, 1], ls = :dash, lw = lw_ci, color = colors[2], label = "")
        # plot!(fig, x[idx], RES_YCI[i][idx, 2], ls = :dash, lw = lw_ci, color = colors[2], label = "")
        # savefig(fig, resfile[1:end-4] * "_$i-reverse-boundary.pdf")
        savefig(fig, resfile[1:end-4] * "_$i.pdf")
    end
end

function plot_demo_ci_draft()
    resfile = "res/ci_monofit/3e487fc/demo-CI-exp-n100-σ0.2-seed175-B100000-K010-K10-nepoch550000-prop0.2-2022-11-20T11_27_12+08_00.sil"
    resfile = "res/ci_monofit/3e487fc/demo-CI-cubic-n100-σ0.2-seed85-B2000-K010-K10-nepoch550000-prop0.2-2022-11-19T11_49_48+08_00.sil"
    
    resfile = "res/ci_monofit/3e487fc/demo-CI-cubic-n100-σ0.2-seed5-B2000-K050-K50-nepoch550000-prop0.2-2022-11-21T05_15_29+08_00.sil"
    resfile = "res/ci_monofit/3e487fc/demo-CI-cubic-n100-σ0.2-seed5-B2000-K020-K20-nepoch550000-prop0.2-2022-11-21T01_03_14+08_00.sil"
    resfile = "res/ci_monofit/3e487fc/demo-CI-cubic-n100-σ0.2-seed5-B2000-K010-K10-nepoch550000-prop0.2-2022-11-20T11_12_10+08_00.sil"
    resfile = "res/ci_monofit/3e487fc/demo-CI-exp-n100-σ0.2-seed175-B2000-K010-K10-nepoch550000-prop0.2-2022-11-20T12_23_44+08_00.sil"
    resfolder = dirname(resfile)
    x, y, λs, J, Yhat, Yhat0, RES_YCI, cp = deserialize(resfile)
    
    # have Yhat
    resfile = "res/ci_monofit/3e487fc/demo-CI-cubic-n100-σ0.2-seed5-B2000-K010-K10-nepoch550000-prop0.2-2022-11-19T12_34_20+08_00.sil"
    resfolder = dirname(resfile)
    x, y, λs, J, Yhat, Yhat0, RES_YCI, RES_Yhat, cp = deserialize(resfile)


    f = exp
    cubic(x) = x^3
    f(x) = x^3 # error of redefinition if f = exp first
    f = cubic # no error of redefinition even if f = exp first
    idx = sortperm(x)
    FIGS = Plots.Plot[]
    for (i, λ) in enumerate(λs)
        overlap = jaccard_index(RES_YCI[i], RES_YCI0[i])
        fig = scatter(x, y, title = "λ = $(@sprintf "%.3e" λ), overlap = $overlap", label = "", legend = :topleft)
        plot!(fig, x[idx], f.(x[idx]), label = "truth")
        plot!(fig, x[idx], Yhat[i, idx], label = "MLP")
        plot!(fig, x[idx], Yhat0[i, idx], label = "OPT")
        cip = MonotoneSplines.coverage_prob(RES_YCI[i], f.(x))
        plot!(fig, x[idx], RES_YCI[i][idx, 1], ls = :dash, label = "lower (prob = $cip)")
        plot!(fig, x[idx], RES_YCI[i][idx, 2], ls = :dash, label = "upper")
        # yhat, YCI = MonotoneSplines.ci_mono_ss(x, y, λ, prop_nknots=0.2)
        cp = MonotoneSplines.coverage_prob(RES_YCI0[i], f.(x))
        plot!(fig, x[idx], RES_YCI0[i][idx, 1], label="lower (cov prob = $cp)", ls = :dot)
        plot!(fig, x[idx], RES_YCI0[i][idx, 2], label="upper", ls = :dot)
        # cip2 = coverage_prob(hcat(2Yhat[i, :] - RES_YCI[i][:, 2],
        #                           2Yhat[i, :] - RES_YCI[i][:, 1]), f.(x))
        # plot!(fig, x[idx], 2Yhat[i, idx] - RES_YCI[i][idx, 2], ls = :dash, label = "lower (corrected, prob = $cip2)")
        # plot!(fig, x[idx], 2Yhat[i, idx] - RES_YCI[i][idx, 1], ls = :dash, label = "upper (corrected)")
        push!(FIGS, fig)
    end

    save_grid_plots(FIGS)

    RES_YCI0 = Array{Any, 1}(undef, length(λs))
    for (i, λ) in enumerate(λs)
        yhat, YCI = MonotoneSplines.ci_mono_ss(x, y, λ, prop_nknots=0.2)
        RES_YCI0[i] = YCI
    end
    supp_resfile = resfile[1:end-4] * "_supp.sil"
    serialize(supp_resfile, RES_YCI0)

    ## select three lambda as demo

    ## check i = 11
    i = 11
    λ = λs[i]
    fig = scatter(x, y, title = "λ = $λ", label = "", legend = :topleft)
    plot!(fig, x[idx], Yhat[i, idx], label = "MLP")
    plot!(fig, x[idx], Yhat0[i, idx], label = "OPT")
    plot!(fig, x[idx], RES_YCI[i][idx, 1], ls = :dash, label = "lower")
    plot!(fig, x[idx], RES_YCI[i][idx, 2], ls = :dash, label = "upper")
    plot!(fig, x[idx], yci11[idx,1], ls = :dot)
    plot!(fig, x[idx], yci11[idx,2], ls = :dot)
    
    for ystar in eachcol(RES_Yhat[i][:,1:500])
        plot!(fig, x[idx], ystar[idx], lw = 0.05, label = "")
    end

    # bias correct CI
end

function plot_demo(; ls = :dash, xx = 1.05, kw...)
    # []
    resfile = "res/acc_monofit/2b5310d/demo-cubic173-prop0.2-eta01e-4/demo-acc-cubic-n100-σ0.1-seed173-M100-η00.0001-niter150000-prop0.2-2022-11-15T23_31_13+08_00.sil"
    resfolder = dirname(resfile)
    x, y, λs, J, Yhat, Yhat0, LOSS = deserialize(resfile)
    fitfig = scatter(x, y; legend=:topleft, label="", 
                            xlab = "x",
                            ylab = "y",
                            right_margin = -20Plots.mm,
                            # title=latexstring("Cubic (\$n=100,\\sigma = 0.1\$)"), 
                            color = :black, markersize = 1, kw...)
    idx = sortperm(x)
    colors = palette(:darktest, length(λs))
    for (j, λ) in enumerate(λs)
        str_λ = @sprintf "%.3e" λ
        lbl1 = ""
        lbl2 = ""
        if j == 1
            lbl1 = "OPT Solution"
            lbl2 = "MLP Generator"
        end
        plot!(fitfig, x[idx], Yhat[j, idx], label = lbl1, color = colors[j], lw = 0.5)
        plot!(fitfig, x[idx], Yhat0[j, idx], label = lbl2, ls = ls, color = colors[j])
    end
    colorbar = scatter([0, 0], [0, 1], zcolor = [minimum(λs), maximum(λs)], color = :darktest, label = "", 
                        xlim = (1, 1.1), framestyle=:none, 
                        colorbar_title = L"\lambda",
                        # left_margin = -20Plots.mm
                        )
    # annotate!(colorbar, xx, -0.05, text(L"\lambda"))
    fig = plot(fitfig, colorbar, layout=@layout [a b{0.15w}])
    savefig(fig, joinpath(resfolder, "fit.pdf"))
    figloss = plot(log.(LOSS[:, 1:3]), label = [L"{\mathrm E}_\lambda {\cal L}(\lambda)" L"{\cal L}(\lambda_l)" L"{\cal L}(\lambda_u)"], xlab = "iteration", ylab = "log Loss", legend = :topright)
    savefig(figloss, joinpath(resfolder, "loss.pdf"))
end

function load_model_for_lambdas()
    
end

function plot_cp_vs_lambda(; folder = joinpath(@__DIR__, "../res/ci_monofit/8ffaabe/nepoch100-nepoch5-K032-K32-nlam20"), 
                            #    folder = "res/ci_monofit/b22db54/K32-nepoch30w"
                            #    σs = [0.1, 0.2, 0.5]
                            λs = exp.(range(-8, -2, length = 20)),
                             σs = [0.1])
    fs = ["logit5", "exp", "cubic", "sinhalfpi"]
    nf = length(fs)
    nσ = length(σs)
    nλ = length(λs)
    funcs = [x-> 1/(1+exp(-5x)), exp, x->x^3, x->sin(pi/2*x)]
    RES = read_folder(folder, σs = σs, fs = fs, is_rep = true, keywords = ["demo-CI", "K032-K32"])
    CP = zeros(nf, nσ, 5, nλ, 2)
    for j = 1:nf
        for ℓ in 1:nσ
            for i = 1:5
                x, y, λs, J, Yhat, Yhat0, RES_YCI, RES_YCI0, cp = RES[j][ℓ][i]
                cp0 = zeros(length(λs))
                for (k, λ) in enumerate(λs)
                    cp0[k] = MonotoneSplines.coverage_prob(RES_YCI0[k], funcs[j].(x))
                end
                CP[j, ℓ, i, :, 1] .= cp0
                CP[j, ℓ, i, :, 2] .= cp
            end
        end
    end
    μs = mean(CP, dims = 3)[:, :, 1, :, :]
    ys = [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    ms = [:star5 :x :dtriangle :diamond]
    
    for ℓ in 1:nσ
        fig = plot(0, xlab = L"\log\lambda", ylab = "Coverage Prob.", ylim = [0, 1], yticks = (ys, string.(ys))) # title also not work, mv to hline!
        for j in 1:nf
            plot!(log.(λs), μs[j, ℓ, :, :], legend = :bottomleft, label = ["$(fs[j]) OPT" "$(fs[j]) MLP"], color = [:blue :orange], markershape = ms[j], markeralpha = 0.4)
        end
        hline!(fig, [0.95], ls=:dash, label = "", color = :gray, title = "σ = $(σs[ℓ])")
        savefig(fig, joinpath(folder, "covprob_vs_lambda_σ$(σs[ℓ]).pdf"))
    end

end

function plot_demo_cv(folder = "res/ci_monofit/b22db54/K32-nepoch30w")
    funcs = [x-> 1/(1+exp(-5x)), exp, x->x^3, x->sin(pi/2*x)]
    fs = ["logit5", "exp", "cubic", "sinhalfpi"]
    markershapes = [:star5 :x :dtriangle :diamond]
    seeds = [118, 13, 106, 54]
    colors = palette(:auto)
    n = 100
    σ = 0.2
    λs = exp.(range(-8, -2, length = 20))
    fig = plot(0, xlab = L"\log\lambda", ylab = "10-fold CV Error")
    for i = 1:4
        seed = seeds[i]
        f = funcs[i]
        x = rand(MersenneTwister(seed), n) * 2 .- 1
        y = f.(x) + randn(MersenneTwister(seed), n) * σ
        errs, _, _, _ = MonotoneSplines.cv_mono_ss(x, y, λs, nfold = 10)
        idx = argmin(errs)
        plot!(fig, log.(λs), errs, label = "", color = colors[i])
        plot!(fig, log.(λs[idx:idx]), errs[idx:idx], markershape=markershapes[i], label = fs[i], legend = :topleft, color = colors[i])
    end
    str_seeds = replace("$seeds", ", "=>"_", "["=>"", "]"=>"")
    savefig(fig, joinpath(folder, "cv-$(str_seeds).pdf"))
end

# plot_overlap_and_covprob_vs_lambda(σs = [0.2], demo_overlap=true)
function plot_overlap_and_covprob_vs_lambda(folder = "../output/ci_monofit/c80aff2/batch_ci/", 
                                keywords = []; 
                                figsize = (1000, 400), ylim = [0, 1.15], ys = [0, 1/4, 1/2, 3/4, 0.875, 0.95, 1],
                                λs = exp.(range(-8, -2, length = 10)),
                                σs = [0.1, 0.2],
                                fs = ["logit5", "exp", "cubic", "sinhalfpi"],
                                ms = [:star5 :x :dtriangle :diamond],
                                demo_overlap = false, # whether to include the overlap by two colorbars
                                )
    RES = read_folder(folder, σs = σs, fs = fs, keywords = keywords)
    nσ = length(σs)
    nf = length(fs)
    # for sigma = 0.2, second index = 2, last index 2 indicate overlap
    overlap = [hcat([mean(RES[i][j][2], dims = 1)[1,:] for i = 1:nf]...) for j = 1:nσ]
    cp = [hcat([mean(RES[i][j][1], dims=1)[1, :, 1] for i = 1:nf]...) for j=1:nσ]
    cp0 = [hcat([mean(RES[i][j][1], dims=1)[1, :, 2] for i = 1:nf]...) for j=1:nσ]
    fig2 = plot_demo_jaccard(ylim = ylim)
    for j = 1:nσ
        fig1 = plot(log.(λs), overlap[j], 
                # yerror = μs_std / sqrt(5) * 2 # seems not proper
                xlab = L"\log\lambda",
                ylab = "Jaccard index",
                ylim = (ylim[1], ifelse(demo_overlap, ylim[2], 1.0)),
                yticks = (ys, string.(ys)),
                markershape = ms,
                label = reshape(fs, 1, 4),
                legend = :bottomright,
                title = "σ = $(σs[j])",
                labelfontsize = 14, legendfontsize = 14, tickfontsize = 11
                )
        if demo_overlap
            fig = plot(fig2, fig1, size = figsize)
            savefig(fig, joinpath(folder, "overlap_vs_lambda_σ$(σs[j]).pdf"))
        else
            savefig(fig1, joinpath(folder, "jaccard_vs_lambda_σ$(σs[j]).pdf"))
        end
    end
   
    ys = [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    
    for ℓ in 1:nσ
        fig = plot(0, xlab = L"\log\lambda", ylab = "Coverage Prob.", ylim = [0, 1], yticks = (ys, string.(ys)), labelfontsize = 14, legendfontsize = 14, tickfontsize = 11) # title also not work, mv to hline!
        for j in 1:nf
            plot!(log.(λs), hcat(cp0[ℓ][:,j], cp[ℓ][:, j]), legend = :bottomleft, label = ["$(fs[j]) OPT" "$(fs[j]) MLP"], color = [:blue :orange], markershape = ms[j], markeralpha = 0.4)
        end
        hline!(fig, [0.95], ls=:dash, label = "", color = :gray, title = "σ = $(σs[ℓ])")
        savefig(fig, joinpath(folder, "covprob_vs_lambda_σ$(σs[ℓ]).pdf"))
    end
end



function plot_overlap_vs_lambda(folder = "res/ci_monofit/b22db54/K32-nepoch30w", keywords = ["K32", "nepoch300000", "nrep5"]; 
                                figsize = (1000, 400), ylim = [0, 1.15], ys = [0, 1/4, 1/2, 3/4, 0.875, 0.95, 1],
                                λs = exp.(range(-8, -2, length = 20)),
                                σs = [0.1, 0.2],
                                fs = ["logit5", "exp", "cubic", "sinhalfpi"],
                                demo_overlap = false, # whether to include the overlap by two colorbars
                                )
    RES = read_folder(folder, σs = σs, fs = fs, keywords = keywords)
    nσ = length(σs)
    # for sigma = 0.2, second index = 2, last index 2 indicate overlap
    μs = [hcat([mean(RES[i][j][2], dims = 1)[1,:] for i = 1:4]...) for j = 1:nσ]
    μs_std = [hcat([std(RES[i][j][2], dims = 1)[1,:] for i = 1:4]...) for j = 1:nσ]
    fig2 = plot_demo_jaccard(ylim = ylim)
    for j = 1:nσ
        fig1 = plot(log.(λs), μs[j], 
                # yerror = μs_std / sqrt(5) * 2 # seems not proper
                xlab = L"\log\lambda",
                ylab = "Jaccard index",
                ylim = (ylim[1], ifelse(demo_overlap, ylim[2], 1.0)),
                yticks = (ys, string.(ys)),
                markershape = [:star5 :x :dtriangle :diamond],
                label = reshape(fs, 1, 4),
                legend = :bottomright,
                title = "σ = $(σs[j])"
                )
        if demo_overlap
            fig = plot(fig2, fig1, size = figsize)
            savefig(fig, joinpath(folder, "overlap_vs_lambda_σ$(σs[j]).pdf"))
        else
            savefig(fig1, joinpath(folder, "jaccard_vs_lambda_σ$(σs[j]).pdf"))
        end
    end
end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0,0], y .+ [0,0,h,h,0]) # if 4 dim, pgfplotsx is not closed, although gr is ok


function plot_demo_jaccard(;h = 0.05, α = 0.4, ϵ = 0.0, ylim = [0, 1.15], ys = [0, 1/4, 1/2, 3/4, 0.875, 0.95, 1])
    fig = plot(0, ylim = ylim, ylab = "Jaccard index", 
                legend = :outertopleft,# legend not work, why??? need to specify in the below plot! command
                yticks = (ys, string.(ys))
                ) 
    for y in ys[2:end-1]
        if y == ys[2]
            lbl1 = latexstring("interval \$[0, 1]\$")
            lbl2 = latexstring("interval \$[a-1, a]\$")
        else
            lbl1 = ""
            lbl2 = ""
        end
        plot!(fig, rectangle(1, h, 0.0, y-h/2-ϵ), color = :orange, alpha = α, label = lbl1, legend = :topleft, xlab = "value of the toy intervals")
        x = 2y / (1 + y)
        plot!(fig, rectangle(1, h, x - 1, y-h/2+ϵ), color = :blue, alpha = α, label = lbl2)
    end
    return fig        
end

function run_scripts()
    RES = read_folder("res/ci_monofit/9f56d34/")

    RES = read_folder("res/acc_monofit/35b6513", σs = [0.1, 0.2, 0.5, 1.0],
                                           fs = ["logit", "logit5", "exp", "cubic"])
    
    
    RES = read_folder("res/acc_monofit/35b6513/eta0-1e-4-niter1w", σs = [0.1, 0.2, 0.5, 1.0],
                                           fs = ["logit5", "exp", "cubic"])
    
    
    RES = read_folder("res/acc_monofit/7943be8/", σs = [0.1, 0.2, 0.5],
                                           fs = ["logit5", "exp", "cubic"])
    
    RES = read_folder("res/acc_monofit/7943be8/3x10w-eta01e-3", σs = [0.1, 0.2, 0.5],
                                           fs = ["logit5", "exp", "cubic"])
    
    # xx
    RES = read_folder("res/acc_monofit/7943be8/3x30w-eta01e-4", σs = [0.1, 0.2, 0.5],
                                           fs = ["logit5", "exp", "cubic"])
    
    
    RES = read_folder("res/acc_monofit/7943be8/10x40w-eta01e-4", σs = [0.1, 0.2, 0.5],
                                           fs = ["logit5", "exp", "cubic"])
    
    
    RES = read_folder("res/acc_monofit/35b6513/eta0-1e-4-niter2w", σs = [0.1, 0.2],
                                           fs = ["logit5", "exp", "cubic"])
    
    RES = read_folder("res/acc_monofit/2b5310d/5x15w-prop0.2", σs = [0.1, 0.2],
                                           fs = ["logit5", "exp", "cubic"])
    
    
    plot_summary(RES, ["logit5", "exp", "cubic"], [0.1, 0.2])
    
    plot_summary(RES, ["logit5", "exp", "cubic"], [0.1, 0.2, 0.5])
    
    #print2tex(μRES, [L"\sigma = 0.1", L"\sigma = 0.2"], ["Relative Gap", "Fittness Ratio"], subcolnames = [L"\lambda_1", L"\lambda_2", L"\lambda_3", L"\lambda_4"], subrownames = fs, colnames_of_rownames = ["noise", "curve"])
    
    summary_in_table("res/acc_monofit/2b5310d/5x15w-prop0.2-eta01e-4")
    
    summary_in_table("res/acc_monofit/2b5310d/5x15w-prop0.5-eta01e-4")
    
    summary_in_table("res/acc_monofit/2b5310d/5x30w-prop0.5-eta01e-4", [0.1])
    
    summary_in_table("res/acc_monofit/2b5310d/10x15w-prop0.2-eta01e-4", [0.02, 0.1, 0.2, 0.5], ["logit5", "exp", "cubic", "sinhalfpi"])        


    print2tex(μRES, ["sigma = 0.1", "sigma = 0.2", "sigma = 0.5"], ["Relative Gap", "Fittness Ratio"], subcolnames = [latexstring("\\lambda_{$i}") for i in idx_λ], subrownames = fs, colnames_of_rownames = ["noise", "curve"])
    ## accuracy
    [mean(RES[i][j][2], dims=1)[:] for i=1:4, j=1:4]

    μRES = Array{Matrix, 1}(undef, 4)
    σRES = Array{Matrix, 1}(undef, 4)
    for j = 1:4 # sigma 
        μRES[j] = hcat([vcat(mean(RES[i][j][1]), mean(RES[i][j][2], dims=1)[:]) for i=1:4]...)'
        σRES[j] = hcat([vcat(std(RES[i][j][1]), std(RES[i][j][2], dims=1)[:]) for i=1:4]...)'
    end



    print2tex(μRES, σRES, ["σ0.01", "σ0.1", "σ0.2", "σ0.5"], ["Measurements"], ["cubic", "logit5", "exp", "logit"], ["Cov. Prob", "Acc1", "Acc2", "Acc3"], file = "ex1.tex", colnames_of_rownames = ["sigma", "curve"])


    ## cv
    RES = read_folder("res/ci_monofit/ef63000")
end

function plot_ci_runtime()
    ns = [20, 50, 100, 200, 500, 1000, 2000, 5000] 
    folder = "../output/ci_monofit/470af41/nepoch100-nepoch010/"
    RES = read_folder_ns(folder, ns = ns)
    nn = length(ns)
    acc_ratio = zeros(nn)
    overlap = zeros(nn)
    run_time = zeros(nn, 5)
    for i = 1:nn
        overlap[i] = mean(RES[i][2])
        acc_ratio[i] = mean(RES[i][3][:,:,2])
        run_time[i, :] .= mean(RES[i][4], dims=1)[:]
    end
    # to avoid the compilation time of the first run
    yticks = [0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1]
    plot(ns[2:end], acc_ratio[2:end], xticks = (ns[4:end], string.(ns[4:end])), xlab = L"n", label = "Fitness Ratio", markershape = :square, ylim = (0.0, 1.11), yticks = (yticks, string.(yticks)), legend=:bottomright, labelfontsize = 14, legendfontsize = 14, tickfontsize = 11)
    plot!(ns[2:end], overlap[2:end], markershape = :pentagon, label = "Jaccard Index")
    hline!([1.0, 0.9], ls = :dash, label="")
    savefig(joinpath(folder, "acc_and_overlap.pdf"))

    ms = [:star5, :dtriangle, :circle]
    yticks2 = [250, 1000, 3000, 5000, 10000, 15000, 20000]
    plot(ns[2:end], run_time[2:end, 3] * 10, markershape=ms[1], label = latexstring("OPT CI band via 2000 \$\\mathbf{y^\\star}\$ for 100 \$\\lambda\$s"), legend=:topleft, xlab = L"n", ylab="seconds", xticks = (ns[4:end], string.(ns[4:end])), yticks = (yticks2, string.(yticks2)), labelfontsize = 14, legendfontsize = 11, tickfontsize = 11)
    plot!(ns[2:end], run_time[2:end, 2], markershape=ms[2], label = latexstring("MLP trains CI band for any \$\\mathbf{y^\\star}\$ and continuous (\$\\infty\$) \$\\lambda\$s"))
    plot!(ns[2:end], run_time[2:end, 5] * 10, markershape=ms[3], label = latexstring("MLP evaluates CI band via 2000 \$\\mathbf{y^\\star}\$ for 100 \$\\lambda\$s"))
    savefig(joinpath(folder, "run_time.pdf"))
end

function plot_runtime()
    ns = [20, 50, 100, 200, 500, 1000, 2000, 5000] 
    folder = "../output/acc_monofit/c80aff2/nepoch05_cubic_time/"
    RES = read_folder_ns(folder, ns = ns)
    # skip ns = 20 for compilation time
    ns = ns[2:end]
    RES = RES[2:end]
    nn = length(ns)
    acc_ratio = zeros(nn)
    run_time = zeros(nn, 5)
    for i = 1:nn
        # make sure accuracy is ok
        acc_ratio[i] = mean(RES[i][3][:,:,2])
        run_time[i, :] .= mean(RES[i][4], dims = 1)[:] 
    end
    # fn = log
    fn = identity
    ms = [:star5, :dtriangle, :circle]
    #plot(fn.(ns), 200*run_time[:, 1], markershape = ms[1], label = "OPT (2000 lambda)", xticks = (log.(ns), "log " .* string.(ns)), xlab = "log n", ylab = "seconds")
    plot(fn.(ns), acc_ratio, xticks = (ns[3:end], string.(ns[3:end])), xlab = L"n", ylab = "Fitness Ratio", markershape = :square, ylim = (0.89, 1.11), label = "", labelfontsize = 14, legendfontsize = 14, tickfontsize = 11)
    hline!([1.0], ls = :dash, label = "")
    savefig(joinpath(folder, "acc_ratio.pdf"))
    plot(fn.(ns), 200*run_time[:, 1], markershape = ms[1], label = latexstring("OPT solves 2000 \$\\lambda\$s"), xticks = (ns[3:end], string.(ns[3:end])), xlab = L"n", ylab = "seconds", legend = :topleft, labelfontsize = 14, legendfontsize = 14, tickfontsize = 11)
    plot!(fn.(ns), run_time[:, 2], markershape = ms[2], label = latexstring("MLP trains continuous (i.e., \$\\infty\$) \$\\lambda\$s") )
    plot!(fn.(ns), 200*run_time[:, 4], markershape = ms[3], label = latexstring("MLP evaluates 2000 \$\\lambda\$s"))
    savefig(joinpath(folder, "run_time.pdf"))
end

function demo_overfit()

    py"""
    import sys
    sys.path.insert(0, ".")
    import mlp
    import importlib
    importlib.reload(mlp)
    """

    N = 100
    # https://github.com/szcf-weiya/MonotoneSplines.jl/blob/2b5310d6d8583f9c898fb9c9e5583eb842493f13/src/boot.jl#L105
    seed = 1
    n = 100
    σ = 0.2
    f = x -> x^3
    x = rand(MersenneTwister(seed), n) * 2 .- 1
    y = f.(x) + randn(MersenneTwister(seed+1), n) * σ

    ## 
    #λs = exp.(range(-10, -1, length = 10))
    λs = 10 .^ (range(-6, 0, length = 10))
    λmin = minimum(λs)
    λmax = maximum(λs)
    prop_nknots = 0.2
    arr_λs = rand(N) * (λmax - λmin) .+ λmin
    #arr_λs = 10 .^ (rand(B) * (log10(λmax) - log10(λmin)) .+ log10(λmin))
    βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in arr_λs]
    #βs = [mono_ss(x, y, λ, prop_nknots = 1.0).β for λ in arr_λs]
    ngrid = 100
    grid_λs = range(λmin, λmax, length=ngrid)
    #grid_λs = 10 .^ range(log10(λmin), log10(λmax), length = ngrid)
    grid_βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in grid_λs]
    βs_mat = vcat(βs'...)'
    λs_mat = vcat(MonotoneSplines.aug.(arr_λs)'...)'
    J = length(βs[1])

    βs_grid_mat = vcat(grid_βs'...)'
    λs_grid_mat = vcat(MonotoneSplines.aug.(grid_λs)'...)'


    tloss, vloss = py"mlp.train_MLP"(Float32.(λs_mat'), Float32.(βs_mat'), Float32.(λs_grid_mat'), Float32.(βs_grid_mat'), -1, nepoch = 10000)

end

function demo_mlp_flux()
    device = :cpu
    #    device = :gpu
    device = eval(device)
    
    nhidden = 100
    activation = gelu
    dim_λ = 8

    model_g = Chain(Dense(dim_λ => nhidden, activation),
                    Dense(nhidden => nhidden, activation),
                    Dense(nhidden => nhidden, activation),
                    Dense(nhidden => J)) |> device
    opt = Flux.setup(AMSGrad(0.001), model_g)
    data = [(MonotoneSplines.aug(arr_λs[i]), βs[i]) for i in eachindex(βs)]
    loader = Flux.DataLoader((λs_mat, βs_mat) |> device, batchsize = B, shuffle = false)
    losses = []
    val_losses = []
    for _ in 1:100
        _losses = []
        # for d in data
        #     λi, βi = d
        for (λi, βi) in loader
            loss, grads = Flux.withgradient(model_g) do m
                βi_hat = m(λi)
                Flux.mse(βi_hat, βi)
            end
            Flux.update!(opt, model_g, grads[1])
            push!(_losses, loss)
        end
        push!(losses, mean(cpu(_losses)))
        push!(val_losses, mean([Flux.mse(cpu(model_g(device(MonotoneSplines.aug(grid_λs[i])))), grid_βs[i]) for i in eachindex(grid_λs)]))
    end
    plot(log10.(losses))

end