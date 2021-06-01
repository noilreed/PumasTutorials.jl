
using Pumas, StatsFuns
data = read_pumas(example_nmtran_data("pain_remed"),
    cvs = [:arm, :dose, :conc, :painord,:remed],event_data=false)


import StatsFuns.logistic
binary_model = @model begin
    @param begin
        intercept ∈ RealDomain(init=0.001)
        tvslope ∈ RealDomain(init=0.0001)
        Ω ∈ VectorDomain(1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates arm dose

    @pre begin
        rx = dose > 0 ? 1 : 0
        slope = tvslope*rx
        logit = intercept + slope + η[1]
    end

    @derived begin
        pain = logistic(logit)
        dv ~ Bernoulli(logistic(logit))
    end
end


@pre begin
    logit = intercept + tvslope*(dose > 0 ? 1 : 0) + η[1]
    logit = intercept + tvslope*Int(dose > 0) + η[1]
end


param = (
    intercept = 0.001,
    tvslope = 0.0001,
    Ω = [1.0]
    )
res = fit(binary_model,data,param,Pumas.LaplaceI())


sim = simobs(binary_model,data,res.param)
simdf = DataFrame(sim, include_events=false)
first(simdf,6) # Print only the first 6 rows


pop = Population(map(i -> Subject(id=i,cvs=(dose = i.*10,),time=[0.0]),1:10))


poisson_model = @model begin
  @param begin
    tvbase ∈ RealDomain(init=3.0, lower=0.1)
    d50 ∈ RealDomain(init=50, lower=0.1)
    Ω  ∈ PSDDomain(fill(0.1, 1, 1))
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    baseline = tvbase*exp(η[1])
  end

  @covariates dose

  @derived begin
    dv ~ @. Poisson(baseline*(1-dose/(dose + d50)))
  end
end


sim = simobs(poisson_model,pop)
simdf = DataFrame(sim, include_events=false)


using PumasTutorials
PumasTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

