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

function cpr_classic_mlp(; N = 100, nepoch = 10000, gpu_id = -1, ngrid = 100, nhidden = 100, eta = 1e-4)
    # https://github.com/szcf-weiya/MonotoneSplines.jl/blob/2b5310d6d8583f9c898fb9c9e5583eb842493f13/src/boot.jl#L105
    seed = 1
    n = 100
    σ = 0.2
    prop_nknots = 0.2
    f = x -> x^3
    x = rand(MersenneTwister(seed), n) * 2 .- 1
    y = f.(x) + randn(MersenneTwister(seed+1), n) * σ
    B, L, J = MonotoneSplines.build_model(x, prop_nknots = prop_nknots)


    λs = 10 .^ (range(-6, 0, length = 10))
    λmin = minimum(λs)
    λmax = maximum(λs)
    arr_λs = rand(N) * (λmax - λmin) .+ λmin
    βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in arr_λs]
    grid_λs = range(λmin, λmax, length=ngrid)
    grid_βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in grid_λs]
    βs_mat = vcat(βs'...)'
    λs_mat = vcat(MonotoneSplines.aug.(arr_λs)'...)'
    βs_grid_mat = vcat(grid_βs'...)'
    λs_grid_mat = vcat(MonotoneSplines.aug.(grid_λs)'...)'

    tloss, vloss = py"mlp.train_MLP"(Float32.(λs_mat'), Float32.(βs_mat'), Float32.(λs_grid_mat'), Float32.(βs_grid_mat'), gpu_id, nepoch = nepoch, nhidden = nhidden, eta = eta)
    #writedlm("loss-N$N-nepoch$nepoch-ngrid$ngrid-nhidden$nhidden-eta$eta.txt", hcat(tloss, vloss))

    __init_pytorch__()
    G, Loss, Loss1 = MonotoneSplines.py_train_G_lambda(y, B, L, nepoch = 0, nepoch0 = 1, K = N, λl = λmin, λu = λmax, gpu_id = gpu_id,
                nhidden = nhidden, λs_opt_train = λs, λs_opt_val = grid_λs, βs_opt_train = βs_mat', βs_opt_val = βs_grid_mat', niter_per_epoch = nepoch)
    writedlm("loss-N$N-nepoch$nepoch-ngrid$ngrid-nhidden$nhidden-eta$eta.txt", hcat(tloss, vloss, Loss1))
end