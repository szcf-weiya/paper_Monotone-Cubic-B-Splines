# paper_Monotone-Cubic-B-Splines

Repository for reproducing figures and tables in

> Lijun Wang, Xiaodan Fan, Huabai Li, and Jun S. Liu. (2024). “Monotone Cubic B-Splines with a Neural-Network Generator.” Journal of Computational and Graphical Statistics *(accepted)*. [![](https://img.shields.io/badge/arXiv-2307.01748-red)](https://doi.org/10.48550/arXiv.2307.01748)

The proposed method is wrapped into a standalone Julia package: https://github.com/szcf-weiya/MonotoneSplines.jl

## Reproduce Results

### Table 1

```bash
cd code
julia --project=. -L GpBS.jl -e 'summary_in_table(simplify=true)'
```

### Tables 2 & 3

```bash
cd code
julia --project=. -L monofit.jl -e 'write2tables()'
```

### Table 4

```bash
cd code
julia --project=. -L ph.jl -e 'write2table("../output/real/res16.sil")'
```

### Figure 3

```bash
cd code
julia --project=. -L loss_ml.jl -e 'demo_plot_loss()'
```

### Figure 4

```bash
cd code
julia --project=. -L GpBS.jl -e 'plot_runtime()'
```

### Figure 5

```bash
cd code
julia --project=. -L GpBS.jl -e 'plot_demo_ci(resfile="../output/ci_monofit/3e487fc/demo-CI-cubic-n100-σ0.2-seed5-B2000-K050-K50-nepoch550000-prop0.2-2022-11-21T05_15_29+08_00.sil", idx_lambda=[1, 10, 20])'
```

### Figure 6

```bash
cd code
julia --project=. -L GpBS.jl -e 'plot_overlap_and_covprob_vs_lambda()'
```

### Figure 7

```bash
cd code
julia --project=. -L GpBS.jl -e 'plot_ci_runtime()'
```
