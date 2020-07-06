using Pumas, Test, Random

@testset "Logistic regression example" begin

  data = read_pumas(joinpath(dirname(pathof(Pumas)), "..", "examples", "pain_remed.csv"),
    covariates = [:arm, :dose, :conc, :painord];
    time=:time, event_data=false)

  mdsl = @model begin
    @param begin
      θ₁ ∈ RealDomain(init=0.001)
      θ₂ ∈ RealDomain(init=0.0001)
      Ω  ∈ PSDDomain(1)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @covariates arm dose

    @pre begin
      rx = dose > 0 ? 1 : 0
      LOGIT = θ₁ + θ₂*rx + η[1]
    end

    @derived begin
      dv ~ @. Bernoulli(logistic(LOGIT))
    end

  end

  param = (θ₁=0.01, θ₂=0.001, Ω=fill(1.0, 1, 1))

  @testset "Conversion of simulation output to DataFrame when dv is scalar" begin
    sim = simobs(mdsl, data, param)
    @test DataFrame(sim) isa DataFrame
  end

  @testset "testing with $approx approximation" for
    approx in (Pumas.FO(), Pumas.FOCE(), Pumas.FOCEI(), Pumas.LaplaceI())

    if approx ∈ (Pumas.FOCE(), Pumas.LaplaceI())
      ft = fit(mdsl, data, param, approx,
        optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
      _param = coef(ft)

      _ptable = probstable(ft)
      @test names(_ptable) == ["id", "time", "dv_prob1", "dv_prob2"]

      # Test values computed with MixedModels.jl
      @test _param.θ₁                ≈ -1.3085393956990727 rtol=1e-3
      @test _param.θ₂                ≈  1.7389379466901713 rtol=1e-3
      @test _param.Ω.chol.factors[1] ≈  1.5376005165566606 rtol=1e-3
    else
      @test_throws ArgumentError Pumas.marginal_nll(mdsl, data, param, approx)
    end
  end
end

@testset "Logistik PD after PK" begin

  mdl = @model begin
    @param begin
      θCL ∈ RealDomain(lower = 0.0, upper=1.0)
      θV  ∈ RealDomain(lower = 0.0)
      ω   ∈ RealDomain(lower = 0.0, upper=1.0)
    end

    @random begin
      η ~ Normal(0.0, ω)
    end

    @pre begin
      CL = θCL
      Vc  = θV
    end

    @vars begin
      μ := Central / Vc
      p := logistic(μ/10 + η)
    end

    @dynamics Central1

    @derived begin
        y ~ @. Bernoulli(p)
    end
  end

  dr = DosageRegimen(100, time=0.0)

  t = obstimes = range(0.5, stop=24, step=0.5)

  par_init = (
    θCL = 0.3,
    θV  = 1.1,
    ω   = 0.1,
    )

  n = 5

  pop_skeleton = [Subject(id=i, events=dr, time=t) for i in 1:n]
  Random.seed!(123)
  pop_sim = simobs(mdl, pop_skeleton, par_init, ensemblealg=EnsembleSerial())

  pop_est = Subject.(pop_sim)

  ft = fit(mdl, pop_est, par_init, Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  _ptable = probstable(ft)
  @test names(_ptable) == ["id", "time", "y_prob1", "y_prob2"]

  @test deviance(ft) ≈ -185.4010155627602 rtol=1e-6

end
