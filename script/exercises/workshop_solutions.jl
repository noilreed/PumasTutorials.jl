
using Pumas, Plots, CSV, Random
Random.seed!(0)


single_dose_regimen = DosageRegimen(100, time=0)
first(single_dose_regimen.data)


s1 = Subject(id=1, evs=single_dose_regimen,cvs=(Wt=70,))


choose_covariates() = (Wt = rand(55:80),)


pop = Population(map(i -> Subject(id = i,evs = single_dose_regimen, cvs =  choose_covariates()),1:24))


pop[1].covariates


mymodel = @model begin
  @param   begin
    tvcl ∈ RealDomain(lower=0, init = 1.0)
    tvv ∈ RealDomain(lower=0, init = 20)
    tvka ∈ RealDomain(lower = 0, init= 1)
    Ω ∈ PDiagDomain(init=[0.09,0.09, 0.09])
    σ_prop ∈ RealDomain(lower=0,init=0.04)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = tvcl * (Wt/70)^0.75 * exp(η[1])
    V  = tvv * (Wt/70) * exp(η[2])
    Ka = tvka * exp(η[3])
  end
  @covariates Wt

  @dynamics Depots1Central1
    #@dynamics begin
    #    Depot' =  -Ka*Depot
    #    Central' =  Ka*Depot - (CL/V)*Central
    #end

  @derived begin
      cp = @. 1000*(Central / V)
      dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
    end
end


param = init_param(mymodel)


obs = simobs(mymodel, pop, param, obstimes=0:1:72)
plot(obs)


simdf = DataFrame(obs)
first(simdf, 6)


simdf[!, :route] .= "ev"


timeu = u"hr"
concu = u"mg/L"
amtu  = u"mg"


ncadf = read_nca(simdf, id=:id, time=:time, conc=:cp, amt=:amt,
    route=:route,timeu=timeu, concu=concu, amtu=amtu, lloq=0.4concu)


plot(ncadf)


auc = NCA.auc(ncadf)


cmax = NCA.cmax(ncadf)


report = NCAReport(ncadf)
report = NCA.to_dataframe(report)
first(report,6)


simdf.cmt = ifelse.(ismissing.(simdf.cmt), 2, simdf.cmt)
est_df = simdf[.!((simdf.dv .== 0.0) .& (simdf.cmt .==2)),:]
first(est_df,6)


data = read_pumas(est_df ,cvs = [:Wt], dvs=[:dv])


res = fit(mymodel,data,param,Pumas.FOCEI())


infer(res)


preds = DataFrame(predict(res))
first(preds, 6)


resids = DataFrame(wresiduals(res))
first(resids, 6)


ebes = DataFrame(empirical_bayes(res))
first(ebes, 6)


resout = DataFrame(inspect(res))
first(resout, 6)

