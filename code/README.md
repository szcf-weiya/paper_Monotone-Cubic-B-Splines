## Competitors

For ease of comparisons, we would call competitors in Julia, although their original implementation might been in R or Python.

### Navarro-García et al. (2023): cpsplines

1. install the package following the instruction in their github repo <https://github.com/ManuelNavarroGarcia/cpsplines>
2. install PyCall in Julia with `ENV["PYTHON"] = "THE_PYTHON_PATH"` (for me, it is `/media/weiya/PSSD/Programs/anaconda3/envs/cpsplines/bin/python`)

Now we can call the python program from Julia.

### Papp and Alizadeh (2014): AMPL 

1. install AMPL community edition: <https://ampl.com/ce/>
    - download
    - activate license
    - manage solver trials: select the `CPLEX`, which is used in the original paper
2. install AMPL Python API: `amplpy` <https://amplpy.ampl.com/en/latest/index.html>

```bash
python -m pip install amplpy
python -m amplpy.modules install cplex
python -m amplpy.modules activate <license-uuid>
```

3. the AMPL model file `ext/1D-regression.increasing.smooth.model` is downloaded from the supplementary of [Papp and Alizadeh (2014)](https://www.tandfonline.com/doi/full/10.1080/10618600.2012.707343)

### Murray et al. (2016): MonoPoly

1. Install the R package `MonoPoly`, which is accessible on CRAN: <https://cran.r-project.org/web/packages/MonoPoly/index.html>
2. Then we can call it from Julia via `RCall`