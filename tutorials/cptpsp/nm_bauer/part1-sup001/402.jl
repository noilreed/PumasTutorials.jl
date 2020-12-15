using Pumas
using CSV
#using Plots
using StaticArrays
#using StatsPlots
#using Dates
#using Pipe


cd("/Users/beatrizguglieri/Documents/Pumas-AI/PumasTutorials.jl/")

data = CSV.read("tutorials/cptpsp/nm_bauer/part1-sup001/402pumas.csv", DataFrame, missingstring="NA",type=Float64)

data.AMT= ifelse.(data.TIME .!= 0, 0, data.AMT)
data.DV= ifelse.(data.TIME .== 0, missing, data.DV)

data.CMT= 1


data_pop = read_pumas(data,
                   id = :ID,
                   time = :TIME,
                   observations = [:DV],
                   amt = :AMT,
                   cmt = :CMT)



model402 = @model begin
  @param begin
      tvV ∈ RealDomain(lower =0.01, init = 9.8)
      tvCL ∈ RealDomain(lower = 0.01, init = 3.7)
      tvV2 ∈ RealDomain(lower = 0.01, init = 8.6)
      tvQ ∈ RealDomain(lower = 0.01, init = 31)
      Ω ∈ PDiagDomain(init=[0.02,0.02,0.02,0.02])
      σ₁ ∈ RealDomain(lower=0.001, init = 0.02)

  end
  @random begin
      η ~ MvNormal(Ω)
  end
  @covariates

  @pre begin
      V = tvV * exp(η[1])
      CL = tvCL * exp(η[2])
      V2 = tvV2 * exp(η[3])
      Q = tvQ * exp(η[4])

  end

  @init begin
      Cent  = 0
      Periph = 0
  end

  @vars begin
      Conc=Cent/V

  end

  @dynamics begin
      Cent' = -(CL/V)*Cent - (Q/V)*Cent + (Q/V2)*Periph
      Periph' =              (Q/V)*Cent - (Q/V2)*Periph
  end

  @derived begin
      DV ~ @. Normal(Conc, sqrt((Conc*σ₁)^2))


  end
end

@time model402_fit = fit(
  model402,
  data_pop,
  init_param(model402),
  Pumas.FOCEI())

model402_fit # Parameters are the same. Objective function value is not the same.

#generate CI
model402_fit_infer = infer(model402_fit) |> coeftable

CSV.write("tutorials/cptpsp/nm_bauer/part1-sup001/model402_finalparam.csv", model402_fit_infer)
