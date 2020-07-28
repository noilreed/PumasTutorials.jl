using Pumas, Test, Random

@testset "Negative binomial model" begin

  pd_poisson = read_pumas(example_data("sim_poisson"), covariates = [:dose], event_data=false)

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

Successful minimization:                      true

Likelihood approximation:               Pumas.FOCE
Log-likelihood value:                   -3495.4273
Number of subjects:                             20
Number of parameters:         Fixed      Optimized
                                  0              4
Observation records:         Active        Missing
    dv:                        1800              0
    Total:                     1800              0

----------------
       Estimate
----------------
θ₁      3.1824
θ₂      0.44859
ω       0.824
θr      0.91089
----------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(fitFOCE)) == """
Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:               Pumas.FOCE
Log-likelihood value:                   -3495.4273
Number of subjects:                             20
Number of parameters:         Fixed      Optimized
                                  0              4
Observation records:         Active        Missing
    dv:                        1800              0
    Total:                     1800              0

---------------------------------------------------------
      Estimate          SE                  95.0% C.I.
---------------------------------------------------------
θ₁     3.1824         0.63673         [1.9344 ; 4.4303 ]
θ₂     0.44859        0.045974        [0.35848; 0.5387 ]
ω      0.824          0.12598         [0.57709; 1.0709 ]
θr     0.91089        0.04267         [0.82726; 0.99453]
---------------------------------------------------------
"""

  # LaplaceI
  fitLaplaceI = fit(negativebinomial_model, pd_negativebinomial, param, Pumas.LaplaceI())

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitLaplaceI) ==
"""
FittedPumasModel

Successful minimization:                      true

Likelihood approximation:           Pumas.LaplaceI
Log-likelihood value:                   -3495.4145
Number of subjects:                             20
Number of parameters:         Fixed      Optimized
                                  0              4
Observation records:         Active        Missing
    dv:                        1800              0
    Total:                     1800              0

----------------
       Estimate
----------------
θ₁      3.2045
θ₂      0.44736
ω       0.82399
θr      0.91089
----------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(fitLaplaceI)) == """
Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:           Pumas.LaplaceI
Log-likelihood value:                   -3495.4145
Number of subjects:                             20
Number of parameters:         Fixed      Optimized
                                  0              4
Observation records:         Active        Missing
    dv:                        1800              0
    Total:                     1800              0

---------------------------------------------------------
      Estimate          SE                  95.0% C.I.
---------------------------------------------------------
θ₁     3.2045         0.64101         [1.9482 ; 4.4609 ]
θ₂     0.44736        0.045827        [0.35755; 0.53718]
ω      0.82399        0.12597         [0.57709; 1.0709 ]
θr     0.91089        0.042669        [0.82726; 0.99452]
---------------------------------------------------------
"""

  # FO/FOCEI not supported for
  @test_throws ArgumentError fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FO())
  @test_throws ArgumentError fit(negativebinomial_model, pd_negativebinomial, param, Pumas.FOCEI())
end
