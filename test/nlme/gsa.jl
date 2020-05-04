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
            [:auc], (θ1 = 0.1, θ2 = 0.5, θ3 = 10); N=1000)

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), sobol) == 
"""Sobol Sensitivity Analysis

First Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1      │ θ2      │ θ3          │
│     │ Any     │ Float64 │ Float64 │ Float64     │
├─────┼─────────┼─────────┼─────────┼─────────────┤
│ 1   │ auc     │ 0.0     │ 1.01175 │ -8.67356e-6 │

Total Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │ θ3         │
│     │ Any     │ Float64 │ Float64  │ Float64    │
├─────┼─────────┼─────────┼──────────┼────────────┤
│ 1   │ auc     │ 0.0     │ 0.994045 │ 5.43116e-6 │

Second Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1*θ2   │ θ1*θ3      │ θ2*θ3   │
│     │ Any     │ Float64 │ Float64    │ Float64 │
├─────┼─────────┼─────────┼────────────┼─────────┤
│ 1   │ auc     │ 1.0029  │ 4.28388e-7 │ 1.0029  │

"""

sobol_sub = gsa(m_diffeq,
            ev2[1],
            p,
            DiffEqSensitivity.Sobol(order=[0,1,2]),
            [:auc], (θ1 = 0.1, θ2 = 0.5, θ3 = 10); N=1000)

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), sobol_sub) ==
"""Sobol Sensitivity Analysis

First Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1      │ θ2      │ θ3          │
│     │ Any     │ Float64 │ Float64 │ Float64     │
├─────┼─────────┼─────────┼─────────┼─────────────┤
│ 1   │ auc     │ 0.0     │ 1.01175 │ -8.67356e-6 │

Total Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │ θ3         │
│     │ Any     │ Float64 │ Float64  │ Float64    │
├─────┼─────────┼─────────┼──────────┼────────────┤
│ 1   │ auc     │ 0.0     │ 0.994045 │ 5.43116e-6 │

Second Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1*θ2   │ θ1*θ3      │ θ2*θ3   │
│     │ Any     │ Float64 │ Float64    │ Float64 │
├─────┼─────────┼─────────┼────────────┼─────────┤
│ 1   │ auc     │ 1.0029  │ 4.28388e-7 │ 1.0029  │

"""

Random.seed!(123)
morris = gsa(m_diffeq,
                   ev2,
                   p,
                   DiffEqSensitivity.Morris(relative_scale = true, num_trajectory=5000),
                   [:auc],(θ1 = 0.1, θ2 = 0.5, θ3 = 10))

@test morris.means[!, :θ1][1] ≈ 0.0 rtol = 1e-12
@test morris.means[!, :θ2][1] ≈ -0.877494 atol = 5e-2
@test morris.means[!, :θ3][1] ≈ -5.18611e-5 atol = 5e-2
@test [morris.means_star[!, :θ1][1], morris.means_star[!, :θ2][1], morris.means_star[!, :θ3][1]] ≈ abs.([morris.means[!, :θ1][1], morris.means[!, :θ2][1], morris.means[!, :θ3][1]]) rtol = 1e-12
@test morris.variances[!, :θ1][1] ≈ 0.0 rtol = 1e-12
@test morris.variances[!, :θ2][1] ≈ 0.14922 atol = 5e-2
@test morris.variances[!, :θ3][1] ≈ 8.56573e-9 atol = 5e-2

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), morris) ==
"""Morris Sensitivity Analysis

Means (μ)
1×4 DataFrame
│ Row │ dv_name │ θ1      │ θ2        │ θ3          │
│     │ Any     │ Float64 │ Float64   │ Float64     │
├─────┼─────────┼─────────┼───────────┼─────────────┤
│ 1   │ auc     │ 0.0     │ -0.883422 │ -5.07704e-5 │

Means star (μ*)
1×4 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │ θ3         │
│     │ Any     │ Float64 │ Float64  │ Float64    │
├─────┼─────────┼─────────┼──────────┼────────────┤
│ 1   │ auc     │ 0.0     │ 0.883422 │ 5.07704e-5 │

Variances
1×4 DataFrame
│ Row │ dv_name │ θ1      │ θ2       │ θ3         │
│     │ Any     │ Float64 │ Float64  │ Float64    │
├─────┼─────────┼─────────┼──────────┼────────────┤
│ 1   │ auc     │ 0.0     │ 0.148674 │ 8.09321e-9 │

"""

Random.seed!(123)
efast = gsa(m_diffeq,
            ev2,
            p,
            DiffEqSensitivity.eFAST(),
            [:auc], (θ1 = 0.1, θ2 = 0.5, θ3 = 10))

@test sprint((io, t) -> show(io, MIME"text/plain"(), t), efast) ==
"""eFAST Sensitivity Analysis

First Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1         │ θ2       │ θ3         │
│     │ Any     │ Float64    │ Float64  │ Float64    │
├─────┼─────────┼────────────┼──────────┼────────────┤
│ 1   │ auc     │ 2.26511e-8 │ 0.982978 │ 3.48522e-7 │

Total Order Indices
1×4 DataFrame
│ Row │ dv_name │ θ1        │ θ2       │ θ3        │
│     │ Any     │ Float64   │ Float64  │ Float64   │
├─────┼─────────┼───────────┼──────────┼───────────┤
│ 1   │ auc     │ 7.8894e-6 │ 0.998808 │ 0.0170702 │

"""
end