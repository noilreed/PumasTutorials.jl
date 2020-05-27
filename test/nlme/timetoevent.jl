using Pumas, Test

@testset "Time-to-event" begin

  pd = read_pumas(example_data("tte_data1"), id=:ID, time=:TIME, dvs=[:DV], cvs=[:DOSE], evid=:EVID)

  tte_exponential = @model begin

    @param begin
      θ  ∈ RealDomain()
      λ₀ ∈ RealDomain(lower=0)
    end

    @covariates DOSE

    @pre begin
      θeff = θ*DOSE
      λ = λ₀*exp(θeff)
    end

    @dynamics begin
      Λ' = λ
    end

    @derived begin
      DV ~ @. TimeToEvent(λ, Λ)
    end
  end

  param_exponential = (θ=-1.0, λ₀=5e-3,)

  @test deviance(tte_exponential, pd, param_exponential, Pumas.NaivePooled()) ≈ 2736.9323 rtol=1e-4
  ft_exponential = fit(tte_exponential, pd, param_exponential, Pumas.NaivePooled())
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(ft_exponential)) == """
FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:  Pumas.NaivePooled
Deviance:                          2734.5877
Total number of observation records:     300
Number of active observation records:    300
Number of subjects:                      300

--------------------------------------------------------------------
       Estimate            SE                       95.0% C.I.
--------------------------------------------------------------------
θ      -0.92922          0.11202            [-1.1488  ; -0.70967  ]
λ₀      0.0053256        0.00037733         [ 0.004586;  0.0060651]
--------------------------------------------------------------------
"""

  tte_weibull = @model begin

    @param begin
      θ  ∈ RealDomain()
      λ₀ ∈ RealDomain(lower=0)
      p  ∈ RealDomain(lower=0)
    end

    @covariates DOSE

    @pre begin
      θeff = θ*DOSE
      _λ₀ = λ₀
      _p  = p
    end

    @vars begin
      λ = _λ₀*exp(θeff)*_p*(_λ₀*t + 1e-10)^(_p - 1)
    end

    @dynamics begin
      Λ' = λ
    end

    @derived begin
      DV ~ @. TimeToEvent(λ, Λ)
    end
  end

  param_weibull = (θ=-1.0, λ₀=5e-3, p=1.1)

  @test deviance(tte_weibull, pd, param_weibull, Pumas.NaivePooled()) ≈ 2722.7392 rtol=1e-4
  ft_weibull = fit(tte_weibull, pd, param_weibull, Pumas.NaivePooled())
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(ft_weibull)) == """
FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:  Pumas.NaivePooled
Deviance:                           2712.697
Total number of observation records:     300
Number of active observation records:    300
Number of subjects:                      300

-------------------------------------------------------------------
       Estimate            SE                      95.0% C.I.
-------------------------------------------------------------------
θ      -1.0564           0.12773          [-1.3067   ; -0.80603  ]
λ₀      0.0049649        0.0003399        [ 0.0042987;  0.0056311]
p       1.3018           0.069773         [ 1.165    ;  1.4385   ]
-------------------------------------------------------------------
"""

  tte_gompertz = @model begin

    @param begin
      θ  ∈ RealDomain()
      λ₀ ∈ RealDomain(lower=0)
      p  ∈ RealDomain(lower=0)
    end

    @covariates DOSE

    @pre begin
      θeff = θ*DOSE
      _λ₀ = λ₀
      _p  = p
    end

    @vars begin
      λ = _λ₀*exp(θeff)*exp(_p*t)
    end

    @dynamics begin
      Λ' = λ
    end

    @derived begin
      DV ~ @. TimeToEvent(λ, Λ)
    end
  end

  param_gompertz = (θ=-1.0, λ₀=5e-3, p=0.01)

  @test deviance(tte_gompertz, pd, param_gompertz, Pumas.NaivePooled()) ≈ 7299.8768 rtol=1e-4
  ft_gompertz = fit(tte_gompertz, pd, param_gompertz, Pumas.NaivePooled())
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(ft_gompertz)) == """
FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:  Pumas.NaivePooled
Deviance:                          2713.7805
Total number of observation records:     300
Number of active observation records:    300
Number of subjects:                      300

--------------------------------------------------------------------
       Estimate            SE                       95.0% C.I.
--------------------------------------------------------------------
θ      -1.0813           0.1344            [-1.3447   ; -0.81788  ]
λ₀      0.0037577        0.00043278        [ 0.0029094;  0.0046059]
p       0.002232         0.00047099        [ 0.0013089;  0.0031551]
--------------------------------------------------------------------
"""

end