using Test
using Pumas
using Random

@testset "Single subject" begin
  data = read_pumas(example_data("sim_data_model1"))

  mdsl1 = @model begin
    @param begin
      θ ∈ VectorDomain(1, init=[0.5])
      σ ∈ RealDomain(lower=0.001, upper=1.0, init=sqrt(0.1))
    end

    @pre begin
      CL = θ[1]
      Vc = 1.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Normal(conc, conc*σ + eps())
    end
  end

  param = init_param(mdsl1)

  for approx in (Pumas.FO, Pumas.FOI, Pumas.FOCE, Pumas.FOCEI, Pumas.LaplaceI, Pumas.LLQuad)
    @test_throws ArgumentError fit(mdsl1, data[1], param, approx(),)
  end
end

@testset "Readme roundtrip" begin
  Random.seed!(123)

  model = @model begin
    @param  begin
      tvcl ∈ RealDomain(lower=0)
      tvv ∈ RealDomain(lower=0)
      pmoncl ∈ RealDomain(lower = -0.99)
      σ_prop ∈ RealDomain(lower=0)
    end

    @covariates wt isPM

    @pre begin
      CL = tvcl * (1 + pmoncl*isPM) * (wt/70)^0.75
      Vc = tvv * (wt/70)
    end

    @dynamics Central1
    #@dynamics begin
    #    Central' =  - (CL/V)*Central
    #end

    @derived begin
      cp = @. 1000*(Central / Vc)
      dv ~ @. Normal(cp, cp*σ_prop)
    end
  end

  ev = DosageRegimen(100, time=0, addl=4, ii=24)

  param = (
    tvcl = 4.0,
    tvv  = 70,
    pmoncl = -0.7,
    σ_prop = 0.2
  )

  choose_covariates() = (isPM = rand([1, 0]), wt = rand(55:80))

  pop_with_covariates = Population(map(i -> Subject(id=i, events=ev, covariates=choose_covariates()), 1:1000))

  obs = simobs(model, pop_with_covariates, param, obstimes=0:1:120, ensemblealg=EnsembleSerial())

  simdf = DataFrame(obs)

  data = read_pumas(simdf, time=:time, covariates=[:isPM, :wt])

  res = fit(model, data[1], param,
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  trf = Pumas.totransform(model.param)
  fitone = fit(model, first(data), param,
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitone) ==
"""FittedPumasModel

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -923.58735
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  0              4
Observation records:         Active        Missing
    dv:                         121              0
    Total:                      121              0

--------------------
           Estimate
--------------------
tvcl        3.9886
tvv        71.687
pmoncl     -0.70079
σ_prop      0.22951
--------------------
"""

  fitnp = fit(model, data, param, Pumas.NaivePooled(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  # You should use NaivePooled if no random effects are present.
  @test_throws ArgumentError fit(model, data, param, Pumas.FOCEI())


  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(fitnp)) ==
"""Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -860997.51
Number of subjects:                           1000
Number of parameters:         Fixed      Optimized
                                  0              4
Observation records:         Active        Missing
    dv:                      121000              0
    Total:                   121000              0

-------------------------------------------------------------------
          Estimate           SE                      95.0% C.I.
-------------------------------------------------------------------
tvcl       3.9995          0.0031928          [ 3.9933 ;  4.0058 ]
tvv       69.907           0.093962           [69.723  ; 70.092  ]
pmoncl    -0.69962         0.00059607         [-0.70079; -0.69845]
σ_prop     0.20041         0.00042691         [ 0.19957;  0.20125]
-------------------------------------------------------------------
"""

  inspect_np = DataFrame(inspect(fitnp))
  @test mean(inspect_np.dv_iwres) < 1e-6

  fit2s = fit(model, data, param, Pumas.TwoStage(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false), constcoef=(pmoncl=0.7,))

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fit2s) ==
"""
Vector{<:FittedPumasModel} with 1000 entries

Parameter statistics
-------------------------------------
           Mean            Std
-------------------------------------
tvcl       3.9988          0.083016
tvv       70.001           2.9683
pmoncl    -0.70006         0.0045657
σ_prop     0.19843         0.013411
-------------------------------------
"""

  # Test case for all_dv_missing in a subject during fit call

  # Set all dvs to missing for Subject 10
  data[10].observations.dv .= missing

  @test_throws Pumas.PumasDataError("Subject id: 10 has all dv values missing") fit(model,data,param,Pumas.FOCEI())
end


@testset "with random" begin
  data = read_pumas(example_data("sim_data_model1"))

  mdsl1 = @model begin
    @param begin
      θ ∈ VectorDomain(1, init=[0.5], lower = [0.0], upper=[20.0])
      Ω ∈ PDiagDomain(init=[0.04])
      σ ∈ RealDomain(lower=0.001, upper=1.0, init=sqrt(0.1))
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      CL = θ[1] * exp(η[1])
      Vc = 1.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Normal(conc, conc*σ + eps())
    end
  end

  mdsl1full = @model begin
    @param begin
      θ ∈ VectorDomain(1, init=[0.5], lower = [0.0], upper=[20.0])
      Ω ∈ PSDDomain(init=[0.04])
      σ ∈ RealDomain(lower=0.001, upper=1.0, init=sqrt(0.1))
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      CL = θ[1] * exp(η[1])
      Vc = 1.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Normal(conc, conc*σ + eps())
    end
  end

  mdsl1_noeta = @model begin
    @param begin
      θ ∈ VectorDomain(1, init=[0.5])
      σ ∈ RealDomain(lower=0.001, upper=1.0, init=sqrt(0.1))
    end

    @pre begin
      CL = θ[1]
      Vc = 1.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Normal(conc, conc*σ + eps())
    end
  end

  param_noeta = init_param(mdsl1_noeta)
  fitone_noeta = fit(mdsl1_noeta, first(data), param_noeta,
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false), constantcoef=(σ=sqrt(0.1),))

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitone_noeta) ==
"""FittedPumasModel

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                    -4.068332
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  1              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

----------------
       Estimate
----------------
θ₁      0.13038
σ       0.31623
----------------
"""

  infer_noeta = infer(fitone_noeta)
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer_noeta) ==
"""Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                    -4.068332
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  1              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

-----------------------------------------------------------
      Estimate           SE                  95.0% C.I.
-----------------------------------------------------------
θ₁     0.13038         1.0297e-5       [0.13036; 0.1304]
σ      0.31623         NaN             [ NaN   ;  NaN    ]
-----------------------------------------------------------
"""

  param = init_param(mdsl1)
  fitone_constantcoef = fit(mdsl1, first(data), param; constantcoef=(Ω=[0.0], σ=sqrt(0.1)),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitone_constantcoef) ==
"""FittedPumasModel

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

----------------
       Estimate
----------------
θ₁      0.1305
Ω₁      0.0
σ       0.31623
----------------
"""

  infer_constantcoef = infer(fitone_constantcoef)
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer_constantcoef) ==
"""Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

------------------------------------------------------------
      Estimate           SE                   95.0% C.I.
------------------------------------------------------------
θ₁     0.1305          0.00010387       [0.13029; 0.1307]
Ω₁     0.0             NaN              [ NaN   ;  NaN    ]
σ      0.31623         NaN              [ NaN   ;  NaN    ]
------------------------------------------------------------
"""

  fitone_omegas = fit(mdsl1, first(data), param; omegas=(:Ω,),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false), constantcoef=(σ=sqrt(0.1),))
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitone_omegas) ==
"""FittedPumasModel

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

------------------
         Estimate
------------------
θ₁        0.1305
Ω₁,₁      0.0
σ         0.31623
------------------
"""

  infer_omegas = infer(fitone_omegas)
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer_omegas) ==
"""Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

--------------------------------------------------------------
        Estimate           SE                   95.0% C.I.
--------------------------------------------------------------
θ₁       0.1305          0.00010387       [0.13029; 0.1307]
Ω₁,₁     0.0             NaN              [ NaN   ;  NaN    ]
σ        0.31623         NaN              [ NaN   ;  NaN    ]
--------------------------------------------------------------
"""

  param = init_param(mdsl1full)
  fitone_constantcoef = fit(mdsl1full, first(data), param; constantcoef=(Ω=[0.0], σ=sqrt(0.1)),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitone_constantcoef) ==
"""FittedPumasModel

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

----------------
       Estimate
----------------
θ₁      0.1305
Ω₁      0.0
σ       0.31623
----------------
"""

  infer_constantcoef = infer(fitone_constantcoef)
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer_constantcoef) ==
"""Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

------------------------------------------------------------
      Estimate           SE                   95.0% C.I.
------------------------------------------------------------
θ₁     0.1305          0.00010387       [0.13029; 0.1307]
Ω₁     0.0             NaN              [ NaN   ;  NaN    ]
σ      0.31623         NaN              [ NaN   ;  NaN    ]
------------------------------------------------------------
"""

  fitone_omegas = fit(mdsl1full, first(data), param; omegas=(:Ω,),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false), constantcoef=(σ=sqrt(0.1),))
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), fitone_omegas) ==
"""FittedPumasModel

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

----------------
       Estimate
----------------
θ₁      0.1305
Ω₁      0.0
σ       0.31623
----------------
"""

  infer_omegas = infer(fitone_omegas)
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer_omegas) ==
"""Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:        Pumas.NaivePooled
Log-likelihood value:                   -4.0683321
Number of subjects:                              1
Number of parameters:         Fixed      Optimized
                                  2              1
Observation records:         Active        Missing
    dv:                           2              0
    Total:                        2              0

------------------------------------------------------------
      Estimate           SE                   95.0% C.I.
------------------------------------------------------------
θ₁     0.1305          0.00010387       [0.13029; 0.1307]
Ω₁     0.0             NaN              [ NaN   ;  NaN    ]
σ      0.31623         NaN              [ NaN   ;  NaN    ]
------------------------------------------------------------
"""

end
