using Pumas, Test, Random

@testset "Negative binomial model" begin

  pd_poisson = read_pumas(example_data("sim_poisson"), cvs = [:dose])

  negativebinomial_model = @model begin
    @param begin
      θ₁ ∈ RealDomain(init=3.0, lower=0.1)
      θ₂ ∈ RealDomain(init=0.5, lower=0.1)
      ω  ∈ RealDomain(init=1.0, lower=0.0)
      θr  ∈ RealDomain(init=1.0, lower=0.0)
    end

    @random begin
      η ~ Normal(0.0, ω)
    end

    @pre begin
      baseline = θ₁*exp(η[1])
      d50 = θ₂
      dose_d50 = dose/(dose+d50)
      r = θr
    end

    @covariates dose

    @vars begin
      m = baseline*(1 - dose_d50)
      p = r/(m + r)
    end

    @derived begin
      dv ~ @. NegativeBinomial(r, p)
    end
  end

  param = init_param(negativebinomial_model)

  @test solve(negativebinomial_model, pd_poisson[1], param) isa Pumas.NullDESolution
  @test simobs(negativebinomial_model, pd_poisson, param) != nothing

  Random.seed!(123)

  sim_negativebinomial = simobs(negativebinomial_model, pd_poisson, param; ensemblealg = EnsembleSerial())
  pd_negativebinomial  = Subject.(sim_negativebinomial)

  # FOCE
  fitFOCE = fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FOCE())

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitFOCE) ==
"""
FittedPumasModel

Successful minimization:                true

Likelihood approximation:         Pumas.FOCE
Deviance:                          4192.9675
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

----------------
       Estimate
----------------
θ₁      3.4486
θ₂      0.5289
ω       1.0304
θr      0.94599
----------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(fitFOCE)) == """
FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:         Pumas.FOCE
Deviance:                          4192.9675
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

---------------------------------------------------------
      Estimate          SE                  95.0% C.I.
---------------------------------------------------------
θ₁     3.4486         0.83798         [1.8062 ; 5.091  ]
θ₂     0.5289         0.079274        [0.37353; 0.68428]
ω      1.0304         0.13949         [0.75699; 1.3038 ]
θr     0.94599        0.058929        [0.83049; 1.0615 ]
---------------------------------------------------------
"""

  # LaplaceI
  fitLaplaceI = fit(negativebinomial_model, pd_negativebinomial, param, Pumas.LaplaceI())

    @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitLaplaceI) ==
"""
FittedPumasModel

Successful minimization:                true

Likelihood approximation:     Pumas.LaplaceI
Deviance:                          4192.9461
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

----------------
       Estimate
----------------
θ₁      3.4714
θ₂      0.52769
ω       1.0304
θr      0.94599
----------------
"""

      @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(fitLaplaceI)) == """
FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:     Pumas.LaplaceI
Deviance:                          4192.9461
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

---------------------------------------------------------
      Estimate          SE                  95.0% C.I.
---------------------------------------------------------
θ₁     3.4714         0.8436          [1.8179 ; 5.1248 ]
θ₂     0.52769        0.079102        [0.37265; 0.68273]
ω      1.0304         0.13948         [0.75699; 1.3037 ]
θr     0.94599        0.058925        [0.83049; 1.0615 ]
---------------------------------------------------------
"""

  # FO/FOCEI not supported for
  @test_throws ArgumentError fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FO())
  @test_throws ArgumentError fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FOCEI())
end
