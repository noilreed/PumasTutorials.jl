using Pumas

using CSV

@testset "Ordinal regression model" begin
  df = copy(DataFrame(CSV.File(example_data("pain_remed"))))
  # The variable is coded 0:3 but Categorical starts at 1
  df.painord .+= 1

  # Read the data# Read the data
  data = read_pumas(df,
    observations = [:painord],
    covariates = [:arm, :dose, :conc, :painord,:remed],
    event_data=false)

  ordinal_model = @model begin
    @param begin
      b₁    ∈ RealDomain(init=2.90692)
      b₂    ∈ RealDomain(init=-2.97771, lower=-1000000, upper=1)
      b₃    ∈ RealDomain(init=-2.7541 , lower=-1000000, upper=1)
      slope ∈ RealDomain(init=0.01)
      ω     ∈ RealDomain(init=sqrt(3.10532), lower = 0.001)
    end

    @random begin
      η ~ Normal(0.0, ω)
    end

    @covariates conc

    @pre begin
      effect = slope * conc
      #Logit of cumulative probabilities
      lge₀ = @. b₁ + η + effect
      lge₁ = @. lge₀ + b₂
      lge₂ = @. lge₁ + b₃

      #Probabilities of >=0 and >=1 and >=2
      pge₀ = @. exp(lge₀) / (1.0 + exp(lge₀))
      pge₁ = @. exp(lge₁) / (1.0 + exp(lge₁))
      pge₂ = @. exp(lge₂) / (1.0 + exp(lge₂))

      #Probabilities of Y=0,1,2,3
      p₀ = @. 1.0 - pge₀
      p₁ = @. pge₀ - pge₁
      p₂ = @. pge₁ - pge₂
      p₃ = @. pge₂
    end

    @derived begin
      painord ~ @. Categorical(p₀, p₁, p₂, p₃)
    end
  end

  ftFOCE = fit(ordinal_model, data, init_param(ordinal_model), Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test loglikelihood(ftFOCE) ≈ -1998.2404 rtol=1e-4
  @test [coef(ftFOCE)...] ≈ [
    3.360148753447197,
   -3.0319526108461274,
   -2.766509146981229,
   -0.43405083623620827,
    1.6381356642400862] rtol=1e-5

  _ptable = probstable(ftFOCE)
  @test names(_ptable) == ["id", "time", "painord_prob1", "painord_prob2", "painord_prob3", "painord_prob4"]

  ftLaplaceI = fit(ordinal_model, data, init_param(ordinal_model), Pumas.LaplaceI(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test loglikelihood(ftLaplaceI) ≈ -1999.0684 rtol=1e-4
  @test [coef(ftLaplaceI)...] ≈ [
    3.398664660546355,
   -3.066129779368099,
   -2.799622921843607,
   -0.4385581902207373,
    1.6536338748877772] rtol=1e-5

  _ptable = probstable(ftLaplaceI)
  @test names(_ptable) == ["id", "time", "painord_prob1", "painord_prob2", "painord_prob3", "painord_prob4"]

  # Read the data# Read the data
  data_mixed = read_pumas(df,
    observations = [:painord, :dv],
    covariates   = [:arm, :dose, :conc, :painord,:remed],
    event_data   = false)

  mixed_model = @model begin
    @param begin
      θ₁ ∈ RealDomain(init=0.001)
      θ₂ ∈ RealDomain(init=0.0001)
      b₁    ∈ RealDomain(init=2.90692)
      b₂    ∈ RealDomain(init=-2.97771, lower=-1000000, upper=1)
      b₃    ∈ RealDomain(init=-2.7541 , lower=-1000000, upper=1)
      slope ∈ RealDomain(init=0.01)
      ω     ∈ RealDomain(init=sqrt(3.10532), lower = 0.001)
    end

    @random begin
      η ~ Normal(0.0, ω)
    end

    @covariates conc dose

    @pre begin
      rx = dose > 0 ? 1 : 0
      LOGIT = θ₁ + θ₂*rx + η[1]

      effect = slope * conc
      #Logit of cumulative probabilities
      lge₀ = @. b₁ + η + effect
      lge₁ = @. lge₀ + b₂
      lge₂ = @. lge₁ + b₃

      #Probabilities of >=0 and >=1 and >=2
      pge₀ = @. exp(lge₀) / (1.0 + exp(lge₀))
      pge₁ = @. exp(lge₁) / (1.0 + exp(lge₁))
      pge₂ = @. exp(lge₂) / (1.0 + exp(lge₂))

      #Probabilities of Y=0,1,2,3
      p₀ = @. 1.0 - pge₀
      p₁ = @. pge₀ - pge₁
      p₂ = @. pge₁ - pge₂
      p₃ = @. pge₂
    end

    @derived begin
      painord ~ @. Categorical(p₀, p₁, p₂, p₃)
      dv ~ @. Bernoulli(logistic(LOGIT))
    end
  end

  ftFOCE = fit(mixed_model, data_mixed, init_param(mixed_model), Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  _ptable = probstable(ftFOCE)
  @test names(_ptable) == ["id", "time", "painord_prob1", "painord_prob2", "painord_prob3", "painord_prob4", "dv_prob1", "dv_prob2"]

  ftLaplaceI = fit(mixed_model, data_mixed, init_param(mixed_model), Pumas.LaplaceI(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  _ptable = probstable(ftLaplaceI)
  @test names(_ptable) == ["id", "time", "painord_prob1", "painord_prob2", "painord_prob3", "painord_prob4", "dv_prob1", "dv_prob2"]
end
