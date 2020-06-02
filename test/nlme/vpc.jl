using Pumas, Test

@testset "VPC" begin
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
      #    Central' =  - (CL/V)*Central
      #end
  
    @derived begin
        cp = @. 1000*(Central / Vc)
        dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
    end
end

ev = DosageRegimen(100, time=0, addl=2, ii=24)
s1 = Subject(id=1,  evs=ev, cvs=(isPM=1, wt=70))

param = (
  tvcl = 4.0,
  tvv  = 70,
  pmoncl = -0.7,
  Ω = Diagonal([0.09,0.09]),
  σ_prop = 0.04
  )

choose_covariates() = (isPM = rand([1, 0]),
wt = rand(55:80))
pop_with_covariates = Population(map(i -> Subject(id=i, evs=ev, cvs=choose_covariates()),1:10))
obs = simobs(model, pop_with_covariates, param, obstimes=0:1:60)
simdf = DataFrame(obs)
simdf[rand(1:length(simdf.dv),5),:dv] .= missing
data = read_pumas(simdf, time=:time,cvs=[:isPM, :wt])

vpc_data = vpc(data)
@test typeof(vpc_data) <: Pumas.PopVPC
vpc_data_stratwt = vpc(data,100,stratify_by=[:wt])
@test typeof(vpc_data) <: Pumas.PopVPC
vpc_data_stratispm = vpc(data,100,stratify_by=[:isPM])
@test typeof(vpc_data) <: Pumas.PopVPC

vpc_model = vpc(model, data, param, 100)
@test typeof(vpc_model) <: Pumas.VPC
vpc_model_stratwt = vpc(model, data, param, 100, stratify_by=[:wt])
@test typeof(vpc_model_stratwt) <: Pumas.VPC
vpc_model_stratispm = vpc(model, data, param, 100, stratify_by=[:isPM])
@test typeof(vpc_model_stratispm) <: Pumas.VPC

res = fit(model,data,param,Pumas.FOCEI())
vpc_fpm = vpc(res, 100)
@test typeof(vpc_fpm) <: Pumas.VPC
vpc_fpm_stratwt = vpc(res, 100, stratify_by=[:wt])
@test typeof(vpc_fpm_stratwt) <: Pumas.VPC
vpc_fpm_stratispm = vpc(res, 100, stratify_by=[:isPM])
@test typeof(vpc_fpm_stratispm) <: Pumas.VPC

end