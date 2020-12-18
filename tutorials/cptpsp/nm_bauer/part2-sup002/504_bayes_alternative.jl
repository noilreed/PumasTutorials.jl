using Pumas
using CSV
#using Plots
using StaticArrays
#using StatsPlots
#using Dates
#using Pipe


cd("/Users/beatrizguglieri/Documents/Pumas-AI/PumasTutorials.jl/")

data = CSV.read("tutorials/cptpsp/nm_bauer/part2-sup002/501pumas.csv", DataFrame, missingstring="NA",type=Float64)

data.AMT= ifelse.(data.TIME .!= 0, 0, data.AMT)
data.DV= ifelse.(data.TIME .== 0, missing, data.DV)

data.CMT= 1


data_pop = read_pumas(data,
                   id = :ID,
                   time = :TIME,
                   observations = [:DV],
                   amt = :AMT,
                   cmt = :CMT,
                   covariates = [:WT,:AGE,:SEX])



model504bayes_alternative = @model begin
 @param begin
     θ ~ Constrained(
      MvNormal(
        fill(0.01,8),
        Diagonal(fill(100000.0,8))
      ),
      #lower = zeros(8),
      upper = fill(10.0, 8),
      init  = [0.7, 3, 0.8, 0.8,-0.1,0.1,0.7,0.7])

     Ω ~ InverseWishart(4, diagm(fill(0.04, 2)) .* (4 + 2 + 1))

     #σ²  ~ InverseGamma(3/2, 0.05*(3 + 1 + 1)/2)
     σ² ~ Gamma(1.0, 0.05)

 end
 @random begin
     η ~ MvNormal(Ω)
 end
 @covariates WT AGE SEX

 @pre begin
     LCL = θ[1] + log((WT/70))*θ[3] + log(AGE/50)*θ[5] + θ[7]*SEX
     LV = θ[2] + log((WT/70))*θ[4] + log(AGE/50)*θ[6] + θ[8]*SEX
     MU_1 = LCL
     MU_2 = LV
     CL = exp(MU_1+η[1])
     Vc = exp(MU_2+η[2])

 end

 @init begin
     Central  = 0
 end

 @vars begin
     Conc=Central/Vc

 end

 #@dynamics begin
 #     Cent' = -(CL/V)*Cent
 #end

 @dynamics Central1

 @derived begin
     μ := @. Conc
     DV ~ @. Normal(μ, σ²)

 end
end

_initpars = (
  θ = [0.7, 3.0, 0.8, 0.8, -0.1, 0.1, 0.7, 0.7],
  Ω = [0.1 0.001; 0.001 0.1],
  σ² = 0.5)

@time model504bayes_alternative_fit = fit(
  model504bayes_alternative,
  data_pop,
  _initpars,
  Pumas.BayesMCMC();
  nsamples=10000, nadapts=1000)

model504bayes_alternative_fit

chains = plot(Pumas.Chains(model504bayes_alternative_fit))


@time model504bayes_alternative_fit_noinit = fit(
  model504bayes_alternative,
  data_pop,
  init_param(model504bayes_alternative),
  Pumas.BayesMCMC();
  nsamples=10000, nadapts=1000)

model504bayes_alternative_fit_noinit

chains = plot(Pumas.Chains(model504bayes_alternative_fit_noinit))
