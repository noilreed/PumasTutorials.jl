using Pumas, Test


pop = map(i -> Subject(id=i, cvs=(dose=[10,20,30],), cvstime=(dose=[1,2,3],)), 1:3)
popdf = DataFrame(pop)

poisson_model = @model begin
  @param begin
    tvbase ∈ RealDomain(init=3.0, lower=0.1)
    d50 ∈ RealDomain(init=0.5, lower=0.1)
    Ω  ∈ PSDDomain(fill(0.1, 1, 1))
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates dose

  @pre begin
    baseline = tvbase*exp(η[1])
    Dose = dose
    D50 = d50
  end

  @derived begin
    dv ~ @. Poisson(baseline*(1-Dose/(Dose + D50)))
  end
end

sim = simobs(poisson_model,pop[1]; obstimes = [0,1,3,5])
@test length(sim[:dv]) == 4


pop = read_pumas(example_data("pain_remed"); event_data=false)
popdf = DataFrame(pop)
pop = read_pumas(popdf; event_data=false)
