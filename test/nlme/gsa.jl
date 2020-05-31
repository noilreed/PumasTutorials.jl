using Test
using Pumas
using LinearAlgebra, DiffEqSensitivity, Random

@testset "GSA Tests" begin
choose_covariates() = (isPM = rand([1, 0]),
                       Wt = rand(55:80))

function generate_population(events,nsubs=4)
  pop = Population(map(i -> Subject(id=i,evs=events,cvs=choose_covariates()),1:nsubs))
  return pop
end

ev = DosageRegimen(100, cmt = 2)
ev2 = generate_population(ev)

m_diffeq = @model begin
  @param   begin
    θ1 ∈ RealDomain(lower=0.1,  upper=3)
    θ2 ∈ RealDomain(lower=0.5,  upper=10)
    θ3 ∈ RealDomain(lower=10,  upper=30)
  end

  @pre begin
    Ka = θ1
    CL = θ2
    Vc = θ3
  end

  @covariates isPM Wt

  @dynamics begin
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot - (CL/Vc)*Central
  end

  @derived begin
    cp = @. 1000*(Central / Vc)
    nca := @nca cp
    auc =  NCA.auc(nca)
    thalf =  NCA.thalf(nca)
    cmax = NCA.cmax(nca)
  end
end

p = (  θ1 = 1.5,  #Ka
       θ2  =  1.1,  #CL
       θ3  =   20.0  #Vc
           ,
    )

sobol = gsa(m_diffeq,
            ev2,
            p,
            DiffEqSensitivity.Sobol(order=[0,1,2]),
            [:auc], (θ1 = 0.1, θ2 = 0.5); N=1000)

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), sobol) ==
"""Sobol Sensitivity Analysis

First Order Indices
1×3 DataFrame
│ Row │ dv_name │ θ1      │ θ2      │
│     │ Any     │ Float64 │ Float64 │
├─────┼─────────┼─────────┼─────────┤
│ 1   │ auc     │ 0.0     │ 1.00167 │

Total Order Indices
1×3 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │
│     │ Any     │ Float64 │ Float64  │
├─────┼─────────┼─────────┼──────────┤
│ 1   │ auc     │ 0.0     │ 0.983581 │

Second Order Indices
1×2 DataFrame
│ Row │ dv_name │ θ1*θ2    │
│     │ Any     │ Float64  │
├─────┼─────────┼──────────┤
│ 1   │ auc     │ 0.992625 │

"""

sobol_sub = gsa(m_diffeq,
            ev2[1],
            p,
            DiffEqSensitivity.Sobol(order=[0,1,2]),
            [:auc], (θ1 = 0.1, θ2 = 0.5); N=1000)

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), sobol_sub) ==
"""Sobol Sensitivity Analysis

First Order Indices
1×3 DataFrame
│ Row │ dv_name │ θ1      │ θ2      │
│     │ Any     │ Float64 │ Float64 │
├─────┼─────────┼─────────┼─────────┤
│ 1   │ auc     │ 0.0     │ 1.00167 │

Total Order Indices
1×3 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │
│     │ Any     │ Float64 │ Float64  │
├─────┼─────────┼─────────┼──────────┤
│ 1   │ auc     │ 0.0     │ 0.983581 │

Second Order Indices
1×2 DataFrame
│ Row │ dv_name │ θ1*θ2    │
│     │ Any     │ Float64  │
├─────┼─────────┼──────────┤
│ 1   │ auc     │ 0.992625 │

"""

sobol_ci = gsa(m_diffeq,
                   ev2,
                   p,
                   DiffEqSensitivity.Sobol(order=[0,1,2],nboot=10),
                   [:auc], (θ1 = 0.1, θ2 = 0.5); N=1000)
@test sprint((io, t) -> show(io, MIME"text/plain"(), t), sobol_ci) ==
"""Sobol Sensitivity Analysis

First Order Indices
1×5 DataFrame
│ Row │ dv_name │ θ1 Min CI │ θ1 Max CI │ θ2 Min CI │ θ2 Max CI │
│     │ Any     │ Float64   │ Float64   │ Float64   │ Float64   │
├─────┼─────────┼───────────┼───────────┼───────────┼───────────┤
│ 1   │ auc     │ 0.0       │ 0.0       │ 0.993729  │ 0.999553  │

Total Order Indices
1×5 DataFrame
│ Row │ dv_name │ θ1 Min CI │ θ1 Max CI │ θ2 Min CI │ θ2 Max CI │
│     │ Any     │ Float64   │ Float64   │ Float64   │ Float64   │
├─────┼─────────┼───────────┼───────────┼───────────┼───────────┤
│ 1   │ auc     │ 0.0       │ 0.0       │ 0.998093  │ 1.00503   │

Second Order Indices
1×3 DataFrame
│ Row │ dv_name │ θ1*θ2 Min CI │ θ1*θ2 Max CI │
│     │ Any     │ Float64      │ Float64      │
├─────┼─────────┼──────────────┼──────────────┤
│ 1   │ auc     │ 0.997199     │ 1.00101      │

"""
Random.seed!(123)
morris = gsa(m_diffeq,
                   ev2,
                   p,
                   DiffEqSensitivity.Morris(relative_scale = true, num_trajectory=5000),
                   [:auc],(θ1 = 0.1, θ2 = 0.5))

@test morris.means[!, :θ1][1] ≈ 0.0 rtol = 1e-12
@test morris.means[!, :θ2][1] ≈ -0.877494 atol = 5e-2
@test [morris.means_star[!, :θ1][1], morris.means_star[!, :θ2][1]] ≈ abs.([morris.means[!, :θ1][1], morris.means[!, :θ2][1]]) rtol = 1e-12
@test morris.variances[!, :θ1][1] ≈ 0.0 rtol = 1e-12
@test morris.variances[!, :θ2][1] ≈ 0.14922 atol = 5e-2

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), morris) ==
"""Morris Sensitivity Analysis

Means (μ)
1×3 DataFrame
│ Row │ dv_name │ θ1      │ θ2        │
│     │ Any     │ Float64 │ Float64   │
├─────┼─────────┼─────────┼───────────┤
│ 1   │ auc     │ 0.0     │ -0.889083 │

Means star (μ*)
1×3 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │
│     │ Any     │ Float64 │ Float64  │
├─────┼─────────┼─────────┼──────────┤
│ 1   │ auc     │ 0.0     │ 0.889083 │

Variances
1×3 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │
│     │ Any     │ Float64 │ Float64  │
├─────┼─────────┼─────────┼──────────┤
│ 1   │ auc     │ 0.0     │ 0.149501 │

"""

Random.seed!(123)
efast = gsa(m_diffeq,
                   ev2,
                   p,
                   DiffEqSensitivity.eFAST(),
                   [:auc], (θ2 = 0.5,),n=500)

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), efast) ==
"""eFAST Sensitivity Analysis

First Order Indices
1×2 DataFrame
│ Row │ dv_name │ θ2       │
│     │ Any     │ Float64  │
├─────┼─────────┼──────────┤
│ 1   │ auc     │ 0.970995 │

Total Order Indices
1×2 DataFrame
│ Row │ dv_name │ θ2      │
│     │ Any     │ Float64 │
├─────┼─────────┼─────────┤
│ 1   │ auc     │ 0.99769 │

"""
end
