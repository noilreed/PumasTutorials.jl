using Pumas, Test

@testset "Test with missing values" begin

  data         = read_pumas(example_data("sim_data_model1"))
  data_missing = read_pumas(example_data("sim_data_model1"))

  # Make a missing observation
  push!(data_missing[1].observations.dv, missing)
  push!(data_missing[1].time, 2)

  model = Dict()

  model["additive"] = @model begin
    @param begin
      θ ∈ RealDomain()
      Ω ∈ PDiagDomain(1)
      σ ∈ RealDomain(lower=0.0001)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      CL = θ * exp(η[1])
      Vc = 1.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Normal(conc, σ)
    end
  end

  model["proportional"] = @model begin
    @param begin
      θ ∈ RealDomain()
      Ω ∈ PDiagDomain(1)
      σ ∈ RealDomain(lower=0.0001)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      CL = θ * exp(η[1])
      Vc = 1.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Normal(conc, conc*σ)
    end
  end

  model["exponential"] = @model begin
    @param begin
      θ ∈ RealDomain()
      Ω ∈ PDiagDomain(1)
      σ ∈ RealDomain(lower=0.0001)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      CL = θ * exp(η[1])
      Vc = 1.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
      dv ~ @. LogNormal(log(conc), σ)
    end
  end

  param = (θ=0.5, Ω=Diagonal([0.04]), σ=0.01)

  @testset "testing model: $_model, with $_approx approximation" for
    _model in ("additive", "proportional", "exponential"),
      _approx in (Pumas.FO(), Pumas.FOCE(), Pumas.FOCEI(), Pumas.LaplaceI())

    if _model == "proportional" && _approx == Pumas.FOCE()
      @test_throws ArgumentError deviance(model[_model], data, param, _approx)
      continue
    end
    # LaplaceI and proportional is very unstable and succeeds/fails depending on architecture
    # so we can't mark this as @test_broken
    if _model != "proportional" || _approx != Pumas.LaplaceI()
      ft         = fit(model[_model], data        , param, _approx,
        optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
      ft_missing = fit(model[_model], data_missing, param, _approx,
        optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
      @test deviance(ft) == deviance(ft_missing)

      # Weighted residuals not defined for the LogNormal error model
      if _model !== "exponential"
        res         = wresiduals(ft)
        res_missing = wresiduals(ft_missing)
        @test all(map(((rᵢ, rmᵢ),) -> rᵢ.wres.dv ≈ filter(!ismissing, rmᵢ.wres.dv), zip(res, res_missing)))
      else
        if _approx == Pumas.FO()
          @test_throws AssertionError("eltype(_dist) <: Normal") wresiduals(ft)
        else
          @test_throws ArgumentError("weighted residuals only implemented for Gaussian error models") wresiduals(ft)
        end
      end

      @test ϵshrinkage(ft).dv ≈ ϵshrinkage(ft_missing).dv
      @test ηshrinkage(ft).η  ≈ ηshrinkage(ft_missing).η
      @test aic(ft)           ≈ aic(ft_missing)
      @test bic(ft)           ≈ bic(ft_missing)
    end
  end
end
