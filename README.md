# Pumas.jl

[![pipeline status](https://gitlab.com/PumasAI/Pumas-jl/badges/master/pipeline.svg)](https://gitlab.com/PumasAI/Pumas-jl/commits/master)
[![codecov](https://codecov.io/gh/PumasAI/Pumas.jl/branch/master/graph/badge.svg?token=O3F3YVonX8)](https://codecov.io/gh/PumasAI/Pumas.jl)

Pumas: A Pharmaceutical Modeling and Simulation toolkit

## Resources
  * [Downloads & Install Instructions](https://pumas.ai/download)
  * [Documentation](https://docs.pumas.ai/dev/)
  * [Tutorials](https://tutorials.pumas.ai/)
  * [Blog](https://pumas.ai/blog)
  * [Discourse Forum](https://discourse.pumas.ai/)

## Demo: A Simple PK model

```julia
using Pumas, Plots
```

For reproducibility, we will set a random seed:

```julia
using Random
Random.seed!(1)
```

A simple one compartment oral absorption model using an analytical solution

```julia
model = @model begin
  @param   begin
    tvcl ∈ RealDomain(lower=0)
    tvv ∈ RealDomain(lower=0)
    pmoncl ∈ RealDomain(lower = -0.99)
    Ω ∈ PDiagDomain(2)
    σ_prop ∈ RealDomain(lower=0)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates wt isPM

  @pre begin
    CL = tvcl * (1 + pmoncl*isPM) * (wt/70)^0.75 * exp(η[1])
    Vc  = tvv * (wt/70) * exp(η[2])
  end

  @dynamics Central1
    #@dynamics begin
    #    Central' =  - (CL/Vc)*Central
    #end

  @derived begin
      cp = @. 1000*(Central / Vc)
      dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
    end
end
```

Develop a simple dosing regimen for a subject

```julia
ev = DosageRegimen(100, time=0, addl=4, ii=24)
s1 = Subject(id=1, events=ev, covariates=(isPM=1, wt=70))
```

Simulate a plasma concentration time profile

```julia
param = (
  tvcl = 4.0,
  tvv  = 70,
  pmoncl = -0.7,
  Ω = Diagonal([0.09,0.09]),
  σ_prop = 0.04
  )
obs = simobs(model, s1, param, obstimes=0:1:120)
plot(obs)
```

![single_sub](https://user-images.githubusercontent.com/1814174/62414914-e37de280-b5ef-11e9-959a-419805577ba2.png)

Generate a population of subjects

```julia
choose_covariates() = (isPM = rand([1, 0]),
              wt = rand(55:80))
pop_with_covariates = Population(map(i -> Subject(id=i, events=ev, covariates=choose_covariates()),1:10))
```

Simulate into the population

```julia
obs = simobs(model, pop_with_covariates, param, obstimes=0:1:120)
```
and visualize the output

```julia
plot(obs)
```
![pop_sim](https://user-images.githubusercontent.com/1814174/62414924-03ada180-b5f0-11e9-8613-3b696a335f5d.png)

Let's roundtrip this simulation to test our estimation routines

```julia
simdf = DataFrame(obs)
simdf.cmt .= 1
first(simdf, 6)
```
Read the data in to Pumas

```julia
data = read_pumas(simdf, time=:time, covariates=[:isPM, :wt])
```

Evaluating the results of a model fit goes through an `fit` --> `infer` --> `inspect` --> `validate` cycle

### `fit`

```julia
julia> res = fit(model,data,param,Pumas.FOCEI())
FittedPumasModel

Successful minimization:                true

Likelihood approximation:        Pumas.FOCEI
Objective function value:            8363.13
Total number of observation records:    1210
Number of active observation records:   1210
Number of subjects:                       10

-----------------
       Estimate
-----------------
tvcl    4.7374
tvv    68.749
pmoncl -0.76408
Ω₁,₁    0.046677
Ω₂,₂    0.024126
σ_prop  0.041206
-----------------
```

### `infer`

`infer` provides the model inference


```julia
julia> infer(res)
Calculating: variance-covariance matrix. Done.
FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:        Pumas.FOCEI
Objective function value:            8363.13
Total number of observation records:    1210
Number of active observation records:   1210
Number of subjects:                       10

---------------------------------------------------
       Estimate       RSE           95.0% C.I.
---------------------------------------------------
tvcl    4.7374     10.057  [ 3.8036   ;  5.6713  ]
tvv    68.749       5.013  [61.994    ; 75.503   ]
pmoncl -0.76408    -3.9629 [-0.82342  ; -0.70473 ]
Ω₁,₁    0.046677   34.518  [ 0.015098 ;  0.078256]
Ω₂,₂    0.024126   31.967  [ 0.0090104;  0.039243]
σ_prop  0.041206    3.0853 [ 0.038714 ;  0.043698]
---------------------------------------------------
```

### `inspect`

`inspect` gives you the model predictions, residuals and Empirical Bayes estimates

```julia
resout = DataFrame(inspect(res))
```

```julia
julia> first(resout, 6)
6×13 DataFrame
│ Row │ id     │ time    │ isPM  │ wt    │ pred    │ ipred   │ pred_approx │ wres     │ iwres     │ wres_approx │ ebe_1     │ ebe_2     │ ebes_approx │
│     │ String │ Float64 │ Int64 │ Int64 │ Float64 │ Float64 │ Pumas.FOCEI │ Float64  │ Float64   │ Pumas.FOCEI │ Float64   │ Float64   │ Pumas.FOCEI │
├─────┼────────┼─────────┼───────┼───────┼─────────┼─────────┼─────────────┼──────────┼───────────┼─────────────┼───────────┼───────────┼─────────────┤
│ 1   │ 1      │ 0.0     │ 1     │ 74    │ 1344.63 │ 1679.77 │ FOCEI()     │ 0.273454 │ -0.638544 │ FOCEI()     │ -0.189025 │ -0.199515 │ FOCEI()     │
│ 2   │ 1      │ 0.0     │ 1     │ 74    │ 1344.63 │ 1679.77 │ FOCEI()     │ 0.273454 │ -0.638544 │ FOCEI()     │ -0.189025 │ -0.199515 │ FOCEI()     │
│ 3   │ 1      │ 0.0     │ 1     │ 74    │ 1344.63 │ 1679.77 │ FOCEI()     │ 0.273454 │ -0.638544 │ FOCEI()     │ -0.189025 │ -0.199515 │ FOCEI()     │
│ 4   │ 1      │ 0.0     │ 1     │ 74    │ 1344.63 │ 1679.77 │ FOCEI()     │ 0.273454 │ -0.638544 │ FOCEI()     │ -0.189025 │ -0.199515 │ FOCEI()     │
│ 5   │ 1      │ 0.0     │ 1     │ 74    │ 1344.63 │ 1679.77 │ FOCEI()     │ 0.273454 │ -0.638544 │ FOCEI()     │ -0.189025 │ -0.199515 │ FOCEI()     │
│ 6   │ 1      │ 0.0     │ 1     │ 74    │ 1344.63 │ 1679.77 │ FOCEI()     │ 0.273454 │ -0.638544 │ FOCEI()     │ -0.189025 │ -0.199515 │ FOCEI()     │
```

### Simulate from fitted model

In order to simulate from a fitted model `simobs` can be used. The final parameters of the fitted models are available in the `coef(res)`

```julia
fitparam = coef(res)
```

You can then pass these optimized parameters into a `simobs` call and pass the same dataset or simulate into a different design

```julia
ev_sd_high_dose = DosageRegimen(200, time=0, addl=4, ii=48)
s2 = Subject(id=1, events=ev_sd_high_dose, covariates=(isPM=1, wt=70))
```

```julia
obs = simobs(model, s2, fitparam, obstimes=0:1:160)
plot(obs)
```
![highdose](https://user-images.githubusercontent.com/1814174/62414975-a8c87a00-b5f0-11e9-9176-10fe37aef986.png)

## Visualization

There are several plot recipes you can use to visualize your model fit.  These are mainly provided by the [`PumasPlots.jl`](https://github.com/PumasAI/PumasPlots.jl) package (currently unregistered, add via URL).

PumasPlots provides recipes to visualize `FittedPumasModel`s, and also has powerful grouping functionality.  While some plot types are specialized for fitted models, you can use all of Plots' features by converting your result to a DataFrame (using `DataFrame(inspect(res))`).

- `convergence(res::FittedPumasModel)` - plots the optimization trajectory of all variables.
    ```julia
    convergence(res)
    ```
    ![convergence](https://user-images.githubusercontent.com/32143268/64719839-1ccf2b00-d497-11e9-9b3f-98730295bc36.png)

- `etacov(res::FittedPumasModel)` - plots the covariates of the model against the etas.  Keyword arguments are described in the docstring - essentially, you can use any column in `DataFrame(inspect(res))` to plot.  
    ```julia
    etacov(res; catmap = (isPM = true,))
    ```
    ![etacov](https://user-images.githubusercontent.com/32143268/64719841-1ccf2b00-d497-11e9-84fe-e4efc35f7c0c.png)


- `resplot(res::FittedPumasModel)` - plots the covariates of the model against their residuals, determined by the approximation type.  Accepts many of the same kwargs as `etacov`.
    ```julia
    resplot(res; catmap = (isPM = true,))
    ```
    ![resplot](https://user-images.githubusercontent.com/32143268/64719842-1ccf2b00-d497-11e9-9f94-43fcd170f5b5.png)    

- `corrplot(empirical_bayes(res); labels)` - overload of the `StatsPlots` corrplot, for etas.
    ```julia
    corrplot(empirical_bayes(res); labels = ["eta_1", "eta_2"])
    ```
    ![corrplot](https://user-images.githubusercontent.com/32143268/64719840-1ccf2b00-d497-11e9-8bc5-e961a60d628a.png)

Most of these plotting functions have docstrings, which can be accessed in the REPL help mode by `>? resplot` (for example).

In addition to these specialized plots, PumasPlots offers powerful grouping functionality.  By working with the DataFrame of your results (`DataFrame(inspect(res))`), you can create arbitrary plots, and by using the `plot_grouped` function, you can create panelled (a.k.a. grouped) plots.

```julia
df = DataFrame(inspect(res))
gdf = groupby(df, :isPM) # group by categorical covariate

plot_grouped(gdf) do subdf # `plot_grouped` will iterate through the groups of `gdf`
    boxplot(subdf.wt, subdf.wres; xlabel = :wt, ylabel = :wres, legend = :none) # you can use any arbitrary plotting function here
end
```
![groupedboxplot](https://user-images.githubusercontent.com/32143268/64720345-4177d280-d498-11e9-8bea-6270e8d227b9.png)
