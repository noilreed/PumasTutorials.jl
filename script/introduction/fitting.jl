
using Pumas, Plots, Random
Random.seed!(0);


repeated_dose_regimen = DosageRegimen(100, time=0, ii=4, addl=2)


choose_covariates() = (Wt = rand(55:80),)


pop = Population(map(i -> Subject(id = i,
                                  events = repeated_dose_regimen,
                                  observations =(dv=Float64[],),
                                  covariates = choose_covariates()),
                                  1:24))


mymodel = @model begin
  @param   begin
    cl ∈ RealDomain(lower = 0.0, init = 1.0)
    tv ∈ RealDomain(lower = 0.0, init = 10.0)
    ka ∈ RealDomain(lower = 0.0, init = 1.0)
    q  ∈ RealDomain(lower = 0.0, init = 0.5)
    Ω  ∈ PDiagDomain(init = [0.9,0.07, 0.05])
    σ_prop ∈ RealDomain(lower = 0,init = 0.03)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates Wt

  @pre begin
    CL = cl * (Wt/70)^0.75 * exp(η[1])
    Vc = tv * (Wt/70) * exp(η[2])
    Ka = ka * exp(η[3])
    Vp = 30.0
    Q  = q
  end

  @dynamics Depots1Central1Periph1

  @derived begin
      cp := @. 1000*(Central / Vc) # We use := because we don't want simobs to store the variable
      dv ~ @. Normal(cp, abs(cp)*σ_prop)
    end
end


param = init_param(mymodel)
obs = simobs(mymodel, pop, param, obstimes=1:1:72)
plot(obs)


result = fit(mymodel, Subject.(obs), param, Pumas.FOCEI())


alternative_param = (
    cl = 0.5,
    tv = 9.0,
    ka = 1.3,
    q  = 0.3,
    Ω  = Diagonal([0.18,0.04, 0.03]),
    σ_prop = 0.04)

fit(mymodel, Subject.(obs), alternative_param, Pumas.FOCEI())


infer(result)


pop_big = Population(map(i -> Subject(id = i,
                                  events = repeated_dose_regimen,
                                  observations =(dv=Float64[],),
                                  covariates = choose_covariates()),
                                  1:100))
obs_big = simobs(mymodel, pop_big, param, obstimes=1:1:72)
result_big = fit(mymodel, Subject.(obs_big), param, Pumas.FOCEI())
infer(result_big)


mymodel_misspec = @model begin
  @param   begin
    cl ∈ RealDomain(lower = 0.0, init = 1.0)
    tv ∈ RealDomain(lower = 0.0, init = 20.0)
    ka ∈ RealDomain(lower = 0.0, init = 1.0)
    Ω  ∈ PDiagDomain(init = [0.12, 0.05, 0.08])
    σ_prop ∈ RealDomain(lower = 0, init = 0.03)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = cl * (Wt/70)^0.75 * exp(η[1])
    Vc = tv * (Wt/70) * exp(η[2])
    Ka = ka * exp(η[3])
  end
  @covariates Wt

  @dynamics Depots1Central1

  @derived begin
      cp = @. 1000*(Central / Vc)
      dv ~ @. Normal(cp, abs(cp)*σ_prop)
    end
end


alternative_param_no_q = (
    cl = 0.5,
    tv = 9.0,
    ka = 1.3,
    Ω  = Diagonal([0.18,0.04, 0.03]),
    σ_prop = 0.04)

result_misspec = fit(mymodel_misspec, Subject.(obs), alternative_param_no_q, Pumas.FOCEI())


wres = wresiduals(result)
wres_misspec = wresiduals(result_misspec)
p1 = plot([w.wres.dv for w in wres], title="Correctly specified", legend=false)
p2 = plot([w.wres.dv for w in wres_misspec], title = "Misspecified", legend=false)
plot(p1, p2)

