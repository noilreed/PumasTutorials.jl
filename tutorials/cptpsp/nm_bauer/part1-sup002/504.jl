using Pumas
using CSV
#using Plots
using StaticArrays
#using StatsPlots
#using Dates
#using Pipe


cd("/Users/beatrizguglieri/Documents/Pumas-AI/PumasTutorials.jl/")

data = CSV.read("tutorials/cptpsp/nm_bauer/part1-sup002/501pumas.csv", DataFrame, missingstring="NA",type=Float64)

data.AMT= ifelse.(data.TIME .!= 0, 0, data.AMT)
data.DV= ifelse.(data.TIME .== 0, missing, data.DV)

data.CMT= 1


data_pop = read_pumas(data,
                   id = :ID,
                   time = :TIME,
                   observations = [:DV],
                   amt = :AMT,
                   rate = :RATE,
                   cmt = :CMT,
                   covariates = [:WT, :AGE, :SEX])



model504 = @model begin
 @param begin
     tvCL ∈ RealDomain(lower = 0.01, init = 4)
     tvV ∈ RealDomain(lower =0.01, init = 30)
     WTonCL ∈ RealDomain(lower = 0.01, init = 0.8)
     WTonV ∈ RealDomain(lower = 0.01, init = 0.8)
     AGEonCL ∈ RealDomain(init = -0.1)
     AGEonV ∈ RealDomain(lower = 0.01, init = 0.1)
     SEXonCL ∈ RealDomain(lower = 0.01, init = 0.7)
     SEXonV ∈ RealDomain(lower = 0.01, init = 0.7)
     Ω ∈ PDiagDomain(init=[0.02,0.02])
     σ₁ ∈ RealDomain(lower=0.001, init = 0.02)

 end
 @random begin
     η ~ MvNormal(Ω)
 end
 @covariates WT AGE SEX

 @pre begin
     CL = tvCL * ((WT/70)^WTonCL * (AGE/50)^AGEonCL * SEXonCL^SEX) * exp(η[1])
     V = tvV * ((WT/70)^WTonV * (AGE/50)^AGEonV * SEXonV^SEX) * exp(η[2])


 end

 @init begin
     Cent  = 0
 end

 @vars begin
     Conc=Cent/V

 end

 @dynamics begin
     Cent' = -(CL/V)*Cent
 end

 @derived begin
     DV ~ @. Normal(Conc, sqrt((Conc*σ₁)^2))


 end
end

@time model504_fit = fit(
  model504,
  data_pop,
  init_param(model504),
  Pumas.FOCEI())

model504_fit # Parameters are the same. Objective function value is not the same.

#generate CI
model504_fit_infer = infer(model504_fit) |> coeftable

CSV.write("tutorials/cptpsp/nm_bauer/part1-sup002/model504_finalparam.csv", model504_fit_infer)
