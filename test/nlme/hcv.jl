using Pumas, Random

@testset "Median size ODE problem (HCV model)" begin
  t = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.99, 10.0, 13.99, 20.99, 28.0]

  # Use the @model macro to use the Pumas DSL for PK/PD models
  peg_inf_model = @model begin
    # The "@param" block specifies the parameters
    @param begin
      # fixed effects with lower and upper bounds
      logθKa   ∈  RealDomain()
      logθKe   ∈  RealDomain()
      logθVd   ∈  RealDomain()
      logθn    ∈  RealDomain()
      logθδ    ∈  RealDomain()
      logθc    ∈  RealDomain()
      logθEC50 ∈  RealDomain()
      # random effects variance parameters, must be posisitive
      ω²Ka   ∈ RealDomain(lower=0.0)
      ω²Ke   ∈ RealDomain(lower=0.0)
      ω²Vd   ∈ RealDomain(lower=0.0)
      ω²n    ∈ RealDomain(lower=0.0)
      ω²δ    ∈ RealDomain(lower=0.0)
      ω²c    ∈ RealDomain(lower=0.0)
      ω²EC50 ∈ RealDomain(lower=0.0)
      # variance parameter in proportional error model
      σPK ∈ RealDomain(lower=0.0)
      σPD ∈ RealDomain(lower=0.0)
    end

    # The random block allows us to specify variances for, and covariances
    # between, the random effects
    @random begin
      ηKa   ~ Normal(0.0, sqrt(ω²Ka))
      ηKe   ~ Normal(0.0, sqrt(ω²Ke))
      ηVd   ~ Normal(0.0, sqrt(ω²Vd))
      ηn    ~ Normal(0.0, sqrt(ω²n))
      ηδ    ~ Normal(0.0, sqrt(ω²δ))
      ηc    ~ Normal(0.0, sqrt(ω²c))
      ηEC50 ~ Normal(0.0, sqrt(ω²EC50))
    end

    @pre begin
      # constants
      p = 100.0
      d = 0.001
      e = 1e-7
      s = 20000.0

      logKa   = logθKa   + ηKa
      logKe   = logθKe   + ηKe
      logVd   = logθVd   + ηVd
      logn    = logθn    + ηn
      logδ    = logθδ    + ηδ
      logc    = logθc    + ηc
      logEC50 = logθEC50 + ηEC50
    end

    @init begin
      T = exp(logc + logδ)/(p*e)
      I = (s*e*p - d*exp(logc + logδ))/(p*exp(logδ)*e)
      W = (s*e*p - d*exp(logc + logδ))/(exp(logc + logδ)*e)
    end

    # The dynamics block is used to describe the evolution of our variables.
    @dynamics begin
      X' = -exp(logKa)*X
      A' = exp(logKa)*X - exp(logKe)*A
      T' = s - T*(e*W + d)
      I' = e*W*T - exp(logδ)*I
      W' = p/((A/exp(logVd)/exp(logEC50) + 1e-100)^exp(logn) + 1)*I - exp(logc)*W
    end

    # The derived block is used to model the dependent variables. Both will
    # be available in our simulated data, but only `dv` has a distribution
    # here (~ read "ditributed as").
    @derived begin
      conc   = @. A/exp(logVd)
      log10W = @. log10(W)
      yPK ~ @. Normal(A/exp(logVd), σPK)
      yPD ~ @. Normal(log10W, σPD)
    end
  end

  peg_inf_dr = DosageRegimen(180.0, ii=7.0, addl=3, duration=1.0)

  param_PKPD = (
    logθKa   = log(0.80),
    logθKe   = log(0.15),
    logθVd   = log(100.0),
    logθn    = log(2.0),
    logθδ    = log(0.20),
    logθc    = log(7.0),
    logθEC50 = log(0.12),
    # random effects variance parameters, must be posisitive
    ω²Ka   = 0.25,
    ω²Ke   = 0.25,
    ω²Vd   = 0.25,
    ω²n    = 0.25,
    ω²δ    = 0.25,
    ω²c    = 0.25,
    ω²EC50 = 0.25,
    # variance parameter in proportional error model
    σPK = 0.2,
    σPD = 0.2)

  _pop = map(i -> Subject(id=i, obs=(yPK=[], yPD=[]), evs=peg_inf_dr, time=t), 1:3)

  # Simulate data for estimation (fix seed for reproducibility)
  Random.seed!(123)
  simdata = simobs(peg_inf_model, _pop, param_PKPD, ensemblealg = EnsembleSerial())

  pd = Subject.(simdata)

  ft = fit(
    peg_inf_model,
    pd,
    param_PKPD,
    Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=true, x_reltol=1e-3))

  @test deviance(ft) ≈ -100.78347276087511 rtol=1e-4

end
