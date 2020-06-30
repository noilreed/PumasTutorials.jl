using Pumas, Test, Random

@testset "Negative binomial model" begin

  pd_poisson = read_pumas(example_data("sim_poisson"), cvs = [:dose], event_data=false)

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
  fitFOCE = fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitFOCE) ==
"""
FittedPumasModel

Successful minimization:                true

Likelihood approximation:         Pumas.FOCE
Deviance:                          3898.5692
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

----------------
       Estimate
----------------
θ₁      3.1875
θ₂      0.56983
ω       0.80668
θr      0.97207
----------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(fitFOCE)) == """
Asymptotic inference results

Successful minimization:                true

Likelihood approximation:         Pumas.FOCE
Deviance:                          3898.5692
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

---------------------------------------------------------
      Estimate          SE                  95.0% C.I.
---------------------------------------------------------
θ₁     3.1875         0.59367         [2.0239 ; 4.351  ]
θ₂     0.56983        0.088667        [0.39605; 0.74362]
ω      0.80668        0.11065         [0.5898 ; 1.0236 ]
θr     0.97207        0.070796        [0.83332; 1.1108 ]
---------------------------------------------------------
"""

  # LaplaceI
  fitLaplaceI = fit(negativebinomial_model, pd_negativebinomial, param, Pumas.LaplaceI())

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitLaplaceI) ==
"""
FittedPumasModel

Successful minimization:                true

Likelihood approximation:     Pumas.LaplaceI
Deviance:                          3898.5679
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

----------------
       Estimate
----------------
θ₁      3.2076
θ₂      0.56847
ω       0.80664
θr      0.97205
----------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(fitLaplaceI)) == """
Asymptotic inference results

Successful minimization:                true

Likelihood approximation:     Pumas.LaplaceI
Deviance:                          3898.5679
Total number of observation records:    1800
Number of active observation records:   1800
Number of subjects:                       20

-------------------------------------------------------
      Estimate          SE                95.0% C.I.
-------------------------------------------------------
θ₁     3.2076         0.59737        [2.0367; 4.3784 ]
θ₂     0.56847        0.088508       [0.395 ; 0.74194]
ω      0.80664        0.11064        [0.5898; 1.0235 ]
θr     0.97205        0.070795       [0.8333; 1.1108 ]
-------------------------------------------------------
"""

  # FO/FOCEI not supported for
  @test_throws ArgumentError fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FO())
  @test_throws ArgumentError fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FOCEI())
end
