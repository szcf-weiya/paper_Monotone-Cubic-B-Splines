using Random
using MonotoneSplines
using PyCall
using DelimitedFiles

py"""
import sys
sys.path.insert(0, ".")
import mlp
import importlib
importlib.reload(mlp)
"""

function cpr_classic_mlp(; N = 100, nepoch = 10000, gpu_id = -1, ngrid = 100, nhidden = 100, eta = 1e-4, λmin = 1e-6, λmax = 1.0)
    # https://github.com/szcf-weiya/MonotoneSplines.jl/blob/2b5310d6d8583f9c898fb9c9e5583eb842493f13/src/boot.jl#L105
    seed = 1
    n = 100
    σ = 0.2
    prop_nknots = 0.2
    f = x -> x^3
    x = rand(MersenneTwister(seed), n) * 2 .- 1
    y = f.(x) + randn(MersenneTwister(seed+1), n) * σ
    B, L, J = MonotoneSplines.build_model(x, prop_nknots = prop_nknots)


    #λs = rand(N) * (λmax - λmin) .+ λmin
    λs = range(λmin, λmax, length=N)
    βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in λs]
    grid_λs = range(λmin, λmax, length=ngrid)
    grid_βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in grid_λs]
    βs_mat = vcat(βs'...)'
    λs_mat = vcat(MonotoneSplines.aug.(λs)'...)'
    βs_grid_mat = vcat(grid_βs'...)'
    λs_grid_mat = vcat(MonotoneSplines.aug.(grid_λs)'...)'

    tloss, vloss = py"mlp.train_MLP"(Float32.(λs_mat'), Float32.(βs_mat'), Float32.(λs_grid_mat'), Float32.(βs_grid_mat'), gpu_id, nepoch = nepoch, nhidden = nhidden, eta = eta)
    #writedlm("loss-N$N-nepoch$nepoch-ngrid$ngrid-nhidden$nhidden-eta$eta.txt", hcat(tloss, vloss))

    __init_pytorch__()
    G, Loss, Loss1 = MonotoneSplines.py_train_G_lambda(y, B, L, nepoch = 0, nepoch0 = 1, K = N, λl = λmin, λu = λmax, gpu_id = gpu_id, η0 = eta,
                nhidden = nhidden, λs_opt_train = λs, λs_opt_val = grid_λs, βs_opt_train = βs_mat', βs_opt_val = βs_grid_mat', niter_per_epoch = nepoch)
    writedlm("loss-N$N-nepoch$nepoch-ngrid$ngrid-nhidden$nhidden-eta$eta-lammin$λmin-lammax$λmax.txt", hcat(tloss, vloss, Loss1))
end

function demo_plot_loss()
    res100 = readdlm("../output/sim/trace-loss/loss-N100-nepoch20000-ngrid100-nhidden1000-eta0.0001-lammin1.0e-6-lammax0.1.txt")
    res1000 = readdlm("../output/sim/trace-loss/loss-N1000-nepoch20000-ngrid100-nhidden1000-eta0.0001-lammin1.0e-6-lammax0.1.txt")
    colors = [:orange, :blue, :black]
    plot(log10.(res100[:, 1]), label = "Classic MLP (N=100)", xlab = "iterations", ylab = "Log Training L2 Loss", color = colors[1], legend = :topright)
    plot!(log10.(res1000[:, 1]), label = "Classic MLP (N=1000)", color = colors[2])
    plot!(log10.(res100[:, 3]), label = "MLP Generator", color = colors[3])
    savefig("../output/sim/trace-loss/train-loss.pdf")

    plot(log10.(res100[:, 2]), label = "Classic MLP (N=100)", xlab = "iterations", ylab = "Log Validation L2 Loss", color = colors[1], legend = :topright)
    plot!(log10.(res1000[:, 2]), label = "Classic MLP (N=1000)", color = colors[2])
    plot!(log10.(res100[:, 4]), label = "MLP Generator", color = colors[3])
    savefig("../output/sim/trace-loss/val-loss.pdf")
end