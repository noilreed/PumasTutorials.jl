using Test
using Pumas
using Random
using StringDistances

@testset "Theophylline model" begin

theopp = read_pumas(example_data("event_data/THEOPP"),cvs = [:SEX,:WT])
@testset "Check that Events is fully typed when parsed" begin
  @test theopp[1].events isa Vector{Pumas.Event{Float64,Float64,Float64,Float64,Float64,Float64,Int}}
end

@testset "Test DataFrame constructors for Subject and Population" begin
  DataFrame(theopp)
  DataFrame(theopp[1])
end

@testset "run2.mod FO without interaction, diagonal omega and additive error" begin
  #Note: run2 requires a block diagonal for the omega
  #$OMEGA  BLOCK(3)
  # 5.55  ;       KA__
  # 0.00524 0.00024  ;   COV KA~K
  # -0.128 0.00911 0.515  ;         K_
  theopmodel_analytical_fo = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PSDDomain(3)
      σ²_add ∈ RealDomain(lower=0.001, init=sqrt(0.388))
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂ + η[2]
      CL = θ₃*WT + η[3]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
  end

  param = (
    θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
    Ω  = diagm(0 => [5.55, 0.0024, 0.515]), # update to block diagonal
    σ²_add = 0.388
    )

  @test deviance(theopmodel_analytical_fo, theopp, param, Pumas.FO()) ≈ 137.16573310096661

  fo_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = [1.81195E+01 -1.12474E-02 -3.05266E-02
                             -1.12474E-02  2.25098E-04  1.04586E-02
                             -3.05266E-02  1.04586E-02  5.99850E-01],
                         σ²_add = 2.66533E-01)

  fo_stderr           = (θ₁ = 4.47E-01,
                         θ₂ = 6.81E-03,
                         θ₃ = 4.93E-03,
                         θ₄ = 3.52E-01,
                         Ω = [9.04E+00 1.92E-02 1.25E+00
                              1.92E-02 7.86E-05 4.57E-03
                              1.25E+00 4.57E-03 3.16E-01],
                         σ²_add = 5.81E-02)
                         # Elapsed estimation time in seconds:     0.04
                         # Elapsed covariance time in seconds:     0.02

  o = fit(theopmodel_analytical_fo, theopp, param, Pumas.FO(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  ofix1 = fit(theopmodel_analytical_fo, theopp, param, Pumas.FO(); constantcoef=(θ₁=0.4,),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  ofix2 = fit(theopmodel_analytical_fo, theopp, param, Pumas.FO(); constantcoef=(σ²_add=0.1,),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  @test coef(ofix1).θ₁ == 0.4
  @test coef(ofix2).σ²_add == 0.1

  o_estimates = coef(o)
  o_stderror  = stderror(o)

  o_infer = infer(o)
  o_inspect = inspect(o)
  o_inspect_df = DataFrame(o_inspect)
  o_wresiduals = wresiduals(o)
  o_wresiduals_df = DataFrame(o_wresiduals)
  o_predict = predict(o)
  @test predict(o, theopp[3]).pred.dv ≈ predict(o)[3].pred.dv
  @test predict(o, theopp[3]).ipred.dv ≈ predict(o)[3].ipred.dv
  @test predict(o, theopp)[3].pred.dv ≈ predict(o)[3].pred.dv
  @test predict(o, theopp)[3].ipred.dv ≈ predict(o)[3].ipred.dv
  @test predict(o_inspect, theopp[3]).pred.dv ≈ predict(o_inspect)[3].pred.dv
  @test predict(o_inspect, theopp[3]).ipred.dv ≈ predict(o_inspect)[3].ipred.dv
  @test predict(o_inspect, theopp)[3].pred.dv ≈ predict(o_inspect)[3].pred.dv
  @test predict(o_inspect, theopp)[3].ipred.dv ≈ predict(o_inspect)[3].ipred.dv

  # test predictions work at specific times using obstimes
  theopp3 = theopp[3]
  predict_at_third = predict(o, theopp3).pred.dv[3]
  predict_at_fourth = predict(o, theopp3).pred.dv[4]
  predict_obstimes = predict(o, theopp[3]; obstimes=[theopp3.time[3], theopp3.time[4], theopp3.time[3]])
  @test predict_obstimes.pred.dv ≈ [predict_at_third, predict_at_fourth, predict_at_third]
  o_predict_df = DataFrame(o_predict)
  @test hasproperty(o_predict_df, :dv)

  @test empirical_bayes(o)[1].η ≈ [-2.41894177806061, -0.01824682682564875, -1.2477915226281944] rtol=1e-5
  @test Pumas.empirical_bayes_dist(o)[1].η.Σ.mat ≈ [
    0.03282388757574916    -0.00041459973200541107 -0.006289667479989163
   -0.00041459973200541107  2.8084159378198597e-5   0.0006126638425598551
   -0.006289667479989163    0.0006126638425598551   0.014897292696384252  ] rtol=1e-5

theopmodel_analytical_fo_boot = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ²_add ∈ RealDomain(lower=0.001, init=sqrt(0.388))
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
  end

  fo_boot_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = Diagonal([1.81195E+01, 5.99850E-01]),
                         σ²_add = 2.66533E-01)

o_boot = fit(theopmodel_analytical_fo_boot, theopp, fo_boot_estimated_params, Pumas.FO(),
  optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  Random.seed!(12349)
  bts = bootstrap(o_boot; samples=5)
@test evaluate(
  Levenshtein(),
  sprint((io, t) -> show(io, MIME"text/plain"(), t), bts),
"""
Bootstrap inference results

Successful minimization:                true

Likelihood approximation:           Pumas.FO
Deviance:                          90.837418
Total number of observation records:     132
Number of active observation records:    132
Number of subjects:                       12

-------------------------------------------------------------------
          Estimate           SE                     95.0% C.I.
-------------------------------------------------------------------
θ₁         4.0491          0.32788          [ 3.9209  ;  4.719   ]
θ₂         0.074816        0.0056641        [ 0.067547;  0.081311]
θ₃         0.036284        0.0037303        [ 0.031046;  0.039428]
θ₄         2.0844          0.39852          [ 1.8795  ;  2.8057  ]
Ω₁,₁      14.836           8.5907           [15.559   ; 35.155   ]
Ω₂,₂       0.19727         0.073349         [ 0.02916 ;  0.19475 ]
σ²_add     0.35574         0.073106         [ 0.27094 ;  0.4524  ]
-------------------------------------------------------------------
Successful fits: 5 out of 5
No stratification.
""") < 3 # allow three characters to differ

Random.seed!(102349)
bts = bootstrap(o_boot; samples=5, stratify_by=:SEX)
@test evaluate(
  Levenshtein(),
  sprint((io, t) -> show(io, MIME"text/plain"(), t), bts),
"""
Bootstrap inference results

Successful minimization:                true

Likelihood approximation:           Pumas.FO
Deviance:                          90.837418
Total number of observation records:     132
Number of active observation records:    132
Number of subjects:                       12

-------------------------------------------------------------------
          Estimate           SE                     95.0% C.I.
-------------------------------------------------------------------
θ₁         4.0491          0.43092           [3.28    ;  4.2132  ]
θ₂         0.074816        0.0062247         [0.069568;  0.08478 ]
θ₃         0.036284        0.0028183         [0.034517;  0.041289]
θ₄         2.0844          0.38284           [1.5091  ;  2.4241  ]
Ω₁,₁      14.836           7.498             [4.1622  ; 20.828   ]
Ω₂,₂       0.19727         0.1125            [0.072736;  0.3298  ]
σ²_add     0.35574         0.10965           [0.17433 ;  0.41925 ]
-------------------------------------------------------------------
Successful fits: 5 out of 5
Stratification by SEX.
""") < 3 # allow three characters to differ


  # Verify that show runs
  io_buffer = IOBuffer()
  show(io_buffer, o)
  show(io_buffer, o_infer)

  @test deviance(o) ≈ 71.979975297638589

  @testset "test estimate of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_estimates, k)) ≈ Pumas._coef_value(getfield(fo_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(fo_stderr, k))           rtol=2e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test o.vvrandeffsorth[i] == zeros(3)
  end

  pred = predict(o)
  pred_foce = predict(o, Pumas.FOCE())
  pred_focei = predict(o, Pumas.FOCEI())
  dfpred = DataFrame(pred)
  dfpred_no_covar = DataFrame(pred; include_covariates=false)
  dfpred_foce = DataFrame(pred_foce)
  dfpred_foce_no_covar = DataFrame(pred_foce; include_covariates=false)
  dfpred_focei = DataFrame(pred_focei)
  dfpred_focei_no_covar = DataFrame(pred_focei; include_covariates=false)
  @testset "test predict" begin
    @test pred[1].approx == Pumas.FO()
    @test pred_foce[1].approx == Pumas.FOCE()
    @test pred_focei[1].approx == Pumas.FOCEI()

    @test all(dfpred[!,:pred_approx].== Ref(Pumas.FO()))
    @test all(dfpred_no_covar[!,:pred_approx].== Ref(Pumas.FO()))
    @test all(dfpred_foce[!,:pred_approx].== Ref(Pumas.FOCE()))
    @test all(dfpred_foce_no_covar[!,:pred_approx].== Ref(Pumas.FOCE()))
    @test all(dfpred_focei[!,:pred_approx].== Ref(Pumas.FOCEI()))
    @test all(dfpred_focei_no_covar[!,:pred_approx].== Ref(Pumas.FOCEI()))

    @test hasproperty(dfpred, :dv_pred)
    @test hasproperty(dfpred, :dv_ipred)
    @test hasproperty(dfpred, :SEX)
    @test hasproperty(dfpred, :WT)
    @test hasproperty(dfpred_no_covar, :dv_pred)
    @test hasproperty(dfpred_no_covar, :dv_ipred)
    @test !(hasproperty(dfpred_no_covar, :SEX))
    @test !(hasproperty(dfpred_no_covar, :WT))

    @test hasproperty(dfpred_foce, :dv_pred)
    @test hasproperty(dfpred_foce, :dv_ipred)
    @test hasproperty(dfpred_foce, :SEX)
    @test hasproperty(dfpred_foce, :WT)
    @test hasproperty(dfpred_foce_no_covar, :dv_pred)
    @test hasproperty(dfpred_foce_no_covar, :dv_ipred)
    @test !(hasproperty(dfpred_foce_no_covar, :SEX))
    @test !(hasproperty(dfpred_foce_no_covar, :WT))

    @test hasproperty(dfpred_focei, :dv_pred)
    @test hasproperty(dfpred_focei, :dv_ipred)
    @test hasproperty(dfpred_focei, :SEX)
    @test hasproperty(dfpred_focei, :WT)
    @test hasproperty(dfpred_focei_no_covar, :dv_pred)
    @test hasproperty(dfpred_focei_no_covar, :dv_ipred)
    @test !(hasproperty(dfpred_focei_no_covar, :SEX))
    @test !(hasproperty(dfpred_focei_no_covar, :WT))
  end

  wres = wresiduals(o)
  wres_foce = wresiduals(o, Pumas.FOCE())
  wres_focei = wresiduals(o, Pumas.FOCEI())
  dfwres = DataFrame(wres)
  dfwres_no_covar = DataFrame(wres; include_covariates=false)
  dfwres_foce = DataFrame(wres_foce)
  dfwres_foce_no_covar = DataFrame(wres_foce; include_covariates=false)
  dfwres_focei = DataFrame(wres_focei)
  dfwres_focei_no_covar = DataFrame(wres_focei; include_covariates=false)
  @testset "test wresiduals" begin
    @test wres[1].approx == Pumas.FO()
    @test wres_foce[1].approx == Pumas.FOCE()
    @test wres_focei[1].approx == Pumas.FOCEI()

    @test all(dfwres[!,:wres_approx].== Ref(Pumas.FO()))
    @test all(dfwres_no_covar[!,:wres_approx].== Ref(Pumas.FO()))
    @test all(dfwres_foce[!,:wres_approx].== Ref(Pumas.FOCE()))
    @test all(dfwres_foce_no_covar[!,:wres_approx].== Ref(Pumas.FOCE()))
    @test all(dfwres_focei[!,:wres_approx].== Ref(Pumas.FOCEI()))
    @test all(dfwres_focei_no_covar[!,:wres_approx].== Ref(Pumas.FOCEI()))

    @test hasproperty(dfwres, :dv_wres)
    @test hasproperty(dfwres, :dv_iwres)
    @test hasproperty(dfwres, :SEX)
    @test hasproperty(dfwres, :WT)
    @test hasproperty(dfwres_no_covar, :dv_wres)
    @test hasproperty(dfwres_no_covar, :dv_iwres)
    @test !(hasproperty(dfwres_no_covar, :SEX))
    @test !(hasproperty(dfwres_no_covar, :WT))

    @test hasproperty(dfwres_foce, :dv_wres)
    @test hasproperty(dfwres_foce, :dv_iwres)
    @test hasproperty(dfwres_foce, :SEX)
    @test hasproperty(dfwres_foce, :WT)
    @test hasproperty(dfwres_foce_no_covar, :dv_wres)
    @test hasproperty(dfwres_foce_no_covar, :dv_iwres)
    @test !(hasproperty(dfwres_foce_no_covar, :SEX))
    @test !(hasproperty(dfwres_foce_no_covar, :WT))

    @test hasproperty(dfwres_focei, :dv_wres)
    @test hasproperty(dfwres_focei, :dv_iwres)
    @test hasproperty(dfwres_focei, :SEX)
    @test hasproperty(dfwres_focei, :WT)
    @test hasproperty(dfwres_focei_no_covar, :dv_wres)
    @test hasproperty(dfwres_focei_no_covar, :dv_iwres)
    @test !(hasproperty(dfwres_focei_no_covar, :SEX))
    @test !(hasproperty(dfwres_focei_no_covar, :WT))
  end

end

@testset "run2.mod FO without interaction, ODE solver, diagonal omega and additive error" begin
  #Note: run2 requires a block diagonal for the omega
  #$OMEGA  BLOCK(3)
  # 5.55  ;       KA__
  # 0.00524 0.00024  ;   COV KA~K
  # -0.128 0.00911 0.515  ;         K_
  theopmodel_solver_fo = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω  ∈ PSDDomain(3)
      σ²_add ∈ RealDomain(lower=0.001, init=0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂+ η[2]
      CL = θ₃*WT + η[3]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
      cp   = Central / Vc
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/Vc)*Central
    end

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
  end

  param = (
    θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
    Ω  = diagm(0 => [5.55, 0.0024, 0.515]), # update to block diagonal
    σ²_add = 0.388
    )

  @test deviance(theopmodel_solver_fo, theopp, param, Pumas.FO(),reltol=1e-6,abstol=1e-8) ≈ 137.16573310096661

  fo_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = [1.81195E+01 -1.12474E-02 -3.05266E-02
                             -1.12474E-02  2.25098E-04  1.04586E-02
                             -3.05266E-02  1.04586E-02  5.99850E-01],
                         σ²_add = 2.66533E-01)

  fo_stderr           = (θ₁ = 4.47E-01,
                         θ₂ = 6.81E-03,
                         θ₃ = 4.93E-03,
                         θ₄ = 3.52E-01,
                         Ω = [9.04E+00 1.92E-02 1.25E+00
                              1.92E-02 7.86E-05 4.57E-03
                              1.25E+00 4.57E-03 3.16E-01],
                         σ²_add = 5.81E-02)
                         # Elapsed estimation time in seconds:     0.45
                         # Elapsed covariance time in seconds:     0.18

  o = @time fit(theopmodel_solver_fo, theopp, param, Pumas.FO(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  o_estimates = coef(o)
  o_stderror  = stderror(o)

  o_infer = infer(o)
  o_inspect = inspect(o)
  o_inspect_df = DataFrame(o_inspect)
  o_wresiduals = wresiduals(o)
  o_wresiduals_df = DataFrame(o_wresiduals)
  o_predict = predict(o)
  o_predict_df = DataFrame(o_predict)
  o_empirical_bayes = empirical_bayes(o)
  o_empirical_bayes_df = DataFrame(o_empirical_bayes)

  @test_throws DimensionMismatch simobs(o.model, theopp, coef(o), empirical_bayes(o)[1:end-1])

  theopmodel_solver_fo_boot = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω  ∈ PDiagDomain(2)
      σ²_add ∈ RealDomain(lower=0.001, init=0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
      cp   = Central / Vc
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/Vc)*Central
    end

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
  end

  fo_boot_estimated_params = (θ₁ = 4.20241E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                         θ₂ = 7.25283E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                         θ₃ = 3.57499E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                         θ₄ = 2.12401E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

                         Ω = Diagonal([1.81195E+01, 5.99850E-01]),
                         σ²_add = 2.66533E-01)
  o_boot = @time fit(theopmodel_solver_fo_boot, theopp, fo_boot_estimated_params, Pumas.FO(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  Random.seed!(12349)
  bts = bootstrap(o_boot; samples=5)
@test evaluate(
  Levenshtein(),
  sprint((io, t) -> show(io, MIME"text/plain"(), t), bts),
"""
Bootstrap inference results

Successful minimization:                true

Likelihood approximation:           Pumas.FO
Deviance:                          90.837418
Total number of observation records:     132
Number of active observation records:    132
Number of subjects:                       12

-------------------------------------------------------------------
          Estimate           SE                     95.0% C.I.
-------------------------------------------------------------------
θ₁         4.0491          0.32788          [ 3.9209  ;  4.719   ]
θ₂         0.074816        0.0056641        [ 0.067547;  0.081311]
θ₃         0.036284        0.0037303        [ 0.031046;  0.039428]
θ₄         2.0844          0.39852          [ 1.8795  ;  2.8057  ]
Ω₁,₁      14.836           8.5907           [15.559   ; 35.155   ]
Ω₂,₂       0.19727         0.073349         [ 0.02916 ;  0.19475 ]
σ²_add     0.35574         0.073106         [ 0.27094 ;  0.4524  ]
-------------------------------------------------------------------
Successful fits: 5 out of 5
No stratification.
""") < 3 # allow three characters to differ

Random.seed!(123849)
bts = bootstrap(o_boot; samples=5, stratify_by=:SEX)
@test evaluate(
  Levenshtein(),
  sprint((io, t) -> show(io, MIME"text/plain"(), t), bts),
"""
Bootstrap inference results

Successful minimization:                true

Likelihood approximation:           Pumas.FO
Deviance:                          90.837418
Total number of observation records:     132
Number of active observation records:    132
Number of subjects:                       12

-------------------------------------------------------------------
          Estimate           SE                     95.0% C.I.
-------------------------------------------------------------------
θ₁         4.0491          0.48558          [ 3.5885  ;  4.7155  ]
θ₂         0.074816        0.0058439        [ 0.068367;  0.082669]
θ₃         0.036284        0.00394          [ 0.03154 ;  0.041277]
θ₄         2.0844          0.20047          [ 2.2468  ;  2.7553  ]
Ω₁,₁      14.836           2.6069           [11.967   ; 17.283   ]
Ω₂,₂       0.19727         0.069049         [ 0.11633 ;  0.27486 ]
σ²_add     0.35574         0.043445         [ 0.24627 ;  0.34901 ]
-------------------------------------------------------------------
Successful fits: 5 out of 5
Stratification by SEX.
""") < 3 # allow three characters to differ

  # Verify that show runs
  io_buffer = IOBuffer()
  show(io_buffer, o)
  show(io_buffer, o_infer)
  show(io_buffer, o_inspect)
  show(io_buffer, o_wresiduals)
  show(io_buffer, o_predict)
  show(io_buffer, o_empirical_bayes)

  @test deviance(o) ≈ 71.979975297638589 rtol=1e-6

  @testset "test estimate of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_estimates, k)) ≈ Pumas._coef_value(getfield(fo_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(fo_stderr, k))           rtol=2.5e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test o.vvrandeffsorth[i] == zeros(3)
  end

  # Test that the types work on both stiff and non-stiff solver methods
  o = fit(theopmodel_solver_fo, theopp, param, Pumas.FO(), alg=Tsit5(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  o = fit(theopmodel_solver_fo, theopp, param, Pumas.FO(), alg=Rosenbrock23(autodiff=false),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test o = fit(theopmodel_solver_fo, theopp, param, Pumas.FO(), alg=Rosenbrock23(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false)) isa Pumas.FittedPumasModel
end

@testset "run3.mod FOCE without interaction, diagonal omega and additive error" begin
  theopmodel_foce = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ²_add ∈ RealDomain(lower=0.001, init=0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
  end

  param = (θ₁ = 2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = Diagonal([5.55, 0.515]),
        σ²_add = 0.388
       )

  @test deviance(theopmodel_foce, theopp, param, Pumas.FOCE()) ≈ 138.90111320972699 rtol=1e-6

  foce_estimated_params = (
    θ₁ = 1.67977E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.49011E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.93898E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.10668E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω  = Diagonal([1.62087E+00, 2.26449E-01]),
    σ²_add = 5.14069E-01)

  foce_stderr = (
    θ₁ = 1.84E-01,
    θ₂ = 5.43E-03,
    θ₃ = 3.42E-03,
    θ₄ = 9.56E-01,

    Ω  = Diagonal([1.61E+00, 7.70E-02]),
    σ²_add = 1.34E-01)

  foce_ebes = [-2.66223E-01 -9.45749E-01
                4.53194E-01  3.03170E-02
                6.31423E-01  7.48592E-02
               -4.72536E-01 -1.80373E-01
               -1.75904E-01  1.64858E-01
               -4.31006E-01  4.74851E-01
               -1.33960E+00  4.29688E-01
               -6.61193E-01  3.15429E-01
                2.83927E+00 -6.59448E-01
               -1.46862E+00 -2.44162E-01
                1.47205E+00  6.97115E-01
               -1.12971E+00 -1.24210E-01]

  foce_ebes_cov = [2.96349E-02  6.43970E-03 6.43970E-03 5.95626E-03
                   1.13840E-01  1.97067E-02 1.97067E-02 1.53632E-02
                   1.37480E-01  2.07479E-02 2.07479E-02 1.42631E-02
                   2.91965E-02  9.78836E-03 9.78836E-03 1.35007E-02
                   3.48071E-02  7.73824E-03 7.73824E-03 7.30707E-03
                   5.87655E-02  2.22284E-02 2.22284E-02 3.92463E-02
                   1.58693E-02  9.82976E-03 9.82976E-03 2.39503E-02
                   5.74869E-02  1.64986E-02 1.64986E-02 2.20832E-02
                   8.51151E-01  3.17941E-02 3.17941E-02 1.25183E-02
                   4.95454E-03  3.20247E-03 3.20247E-03 6.51142E-03
                   4.08289E-01  3.73648E-02 3.73648E-02 2.03508E-02
                   1.46012E-02  5.21247E-03 5.21247E-03 7.67609E-03]

  # Elapsed estimation time in seconds:     0.27
  # Elapsed covariance time in seconds:     0.19

  o = fit(theopmodel_foce, theopp, param, Pumas.FOCE(),
    ensemblealg=EnsembleThreads(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test_throws ArgumentError fit(theopmodel_foce, theopp, param, Pumas.FOCE(),
    ensemblealg=EnsembleDistributed(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  o_simobs_1 = simobs(theopmodel_foce, theopp, coef(o), empirical_bayes(o))
  o_simobs_2 = simobs(theopmodel_foce, theopp, coef(o), fill((η=[0,0],), length(theopp)))
  o_simobs = simobs(o)

  o_estimates = coef(o)
  o_stderror  = stderror(o)

  o_inspect  = inspect(o)
  o_inspect_df = DataFrame(o_inspect)
  @test hasproperty(o_inspect_df, Symbol("η_1"))
  @test hasproperty(o_inspect_df, Symbol("η_2"))

  bts = bootstrap(o; samples=5)
  bts = bootstrap(o; samples=5, stratify_by=:SEX)

  @test deviance(o) ≈ 121.89849119366599 rtol=1e-7

  @testset "test estimate of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_estimates, k)) ≈ Pumas._coef_value(getfield(foce_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(foce_stderr, k))           rtol=1e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for (i, ebe) in enumerate(empirical_bayes(o))
    @test ebe.η ≈ foce_ebes[i,:] rtol=1e-3
  end

  ebe_cov = Pumas.empirical_bayes_dist(o)
  @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test ebe_cov[i].η.Σ.mat[:] ≈ foce_ebes_cov[i,:] atol=1e-3
  end

  @testset "epredict" begin
    Random.seed!(123)
    @test Pumas.epredict(o, nsim=1000)[1].dv ≈ [
      -0.009486557445285057
       2.730920242955558
       4.314639086766728
       4.8572061680076155
       1.5442111858108558
    -146.43247741258824
   -2506.4612432706485
 -211515.07971241107
      -2.4858764482778806e7
      -3.910532698399744e10
      -3.0504017627161996e23] rtol=1e-3
  end

  @test Pumas._predict(  theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
    0.0
    4.9049755499300955
    7.779812894972983
    8.664908079363599
    7.34893115201493
    5.402130960948489
    4.744864876005044
    4.0473577932698825
    3.4538453107724867
    2.717339295201133
    1.0438615786511665] rtol=1e-6
  @test Pumas._predict( theopmodel_foce, theopp[1], param, Pumas.FOCEI()).dv ≈ [
    0.0
    4.9049755499300955
    7.779812894972983
    8.664908079363599
    7.34893115201493
    5.402130960948489
    4.744864876005044
    4.0473577932698825
    3.4538453107724867
    2.717339295201133
    1.0438615786511665] rtol=1e-6
  @test Pumas._predict(   theopmodel_foce, theopp[1], param, Pumas.FO()).dv ≈ [
    0.0
    4.275044896193946
    6.67731216398507
    7.754622724333498
    7.568031859501575
    6.604016774252041
    5.975950613840706
    5.139785095454612
    4.389649021685277
    3.4538248683011945
    1.3267830785309427] rtol=1e-6
  @test wresiduals(   theopmodel_foce, theopp[1], param, Pumas.FO()).dv ≈ [
     1.1879984032756807
    -0.5121540697427261
     1.177641761770986
     1.8748275064478963
    -1.0811008285733203
    -0.4855189971946874
     0.5270429165085201
     0.7926973976555282
     1.38286211294743
     1.8217361795780522
     2.2749279342557145] rtol=1e-6
  @test wresiduals(  theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
     1.1879984032756807
    -0.40449239905538514
     1.4078834621663032
     1.7442967882790288
    -1.7930916166374602
    -0.7377491261762943
     0.41236539881072987
     0.6557677791850915
     1.229426426138302
     1.6689875690147757
     2.207038336508923] rtol=1e-6
  @test wresiduals( theopmodel_foce, theopp[1], param, Pumas.FOCEI()).dv ≈ [ 1.1879984032756807
   -0.40449239905538514
    1.4078834621663032
    1.7442967882790288
   -1.7930916166374602
   -0.7377491261762943
    0.41236539881072987
    0.6557677791850915
    1.229426426138302
    1.6689875690147757
    2.207038336508923] rtol=1e-6
  @test Pumas.iwresiduals(  theopmodel_foce, theopp[1], param, Pumas.FO()).dv ≈ [
     1.1879984032756807
    -2.303825736901788
    -0.1722792965761087
     4.407437594433976
     3.358452446778005
     3.172249887956898
     3.827360627145415
     3.740934575525829
     4.014071580900914
     3.9913136307052524
     3.1357007891301087] rtol=1e-6
  @test Pumas.iwresiduals( theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
     1.1879984032756807
    -1.3784646740678743
    -0.24469855308692548
     1.9520535480092391
    -1.0273947656943232
    -1.4358103741729111
    -0.3983711793143639
     0.09734551444138731
     0.9017172486653547
     1.5424519077826466
     2.194973069154764] rtol=1e-6
  @test Pumas.iwresiduals(theopmodel_foce, theopp[1], param, Pumas.FOCEI()).dv ≈ [
     1.1879984032756807
    -1.3784646740678743
    -0.24469855308692548
     1.9520535480092391
    -1.0273947656943232
    -1.4358103741729111
    -0.3983711793143639
     0.09734551444138731
     0.9017172486653547
     1.5424519077826466
     2.194973069154764] rtol=1e-6
  @test wresiduals(theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
     1.1879984032756807
    -0.40449239905538514
     1.4078834621663032
     1.7442967882790288
    -1.7930916166374602
    -0.7377491261762943
     0.41236539881072987
     0.6557677791850915
     1.229426426138302
     1.6689875690147757
     2.207038336508923] rtol=1e-6
  @testset "Expected individual residuals" begin
    # The way the random effects enter the model allows for unstable
    # solutions. The first order approximated residuals will not hit
    # the unstable areas but when the random effect is fully integrated
    # out, the unstable solutions will greatly affect the residuals
    Random.seed!(123)
    @test Pumas.eiwres(theopmodel_foce, theopp[1], param, 10000).dv ≈
      [1.1879984032754523
      -1.323115211913782
       3.6933475238777507
      24.023503148347146
    1008.3634654704074
       2.1733344199526347e7
       3.843290423027412e10
       3.524337763526852e15
       5.97837210704398e20
       5.649407093090004e28
       4.352720287029761e60] rtol = 1e-6
  end
end
@testset "run3.mod FOCE without interaction, individual omega and additive error" begin
  theopmodel_foce = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω₁ ∈ RealDomain(lower=0.0001, init=5.55)
      Ω₂ ∈ RealDomain(lower=0.0001, init=0.515)
      σ²_add ∈ RealDomain(lower=0.001, init=0.388)
    end

    @random begin
      η₁ ~ Normal(0, sqrt(Ω₁))
      η₂ ~ Normal(0, sqrt(Ω₂))
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η₁ : θ₄ + η₁
      K  = θ₂
      CL = θ₃*WT + η₂
      Vc  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
  end

  param = (θ₁ = 2.77,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω₁ = 5.55,
        Ω₂ = 0.515,
        σ²_add = 0.388
       )

  @test deviance(theopmodel_foce, theopp, param, Pumas.FOCE()) ≈ 138.90111320972699 rtol=1e-6

  foce_estimated_params = (
    θ₁ = 1.67977E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.49011E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.93898E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.10668E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω₁  = 1.62087E+00,
    Ω₂  = 2.26449E-01,
    σ²_add = 5.14069E-01)

  foce_stderr = (
    θ₁ = 1.84E-01,
    θ₂ = 5.43E-03,
    θ₃ = 3.42E-03,
    θ₄ = 9.56E-01,

    Ω₁  = 1.61E+00,
    Ω₂  = 7.70E-02,
    σ²_add = 1.34E-01)

  foce_ebes = [-2.66223E-01 -9.45749E-01
                4.53194E-01  3.03170E-02
                6.31423E-01  7.48592E-02
               -4.72536E-01 -1.80373E-01
               -1.75904E-01  1.64858E-01
               -4.31006E-01  4.74851E-01
               -1.33960E+00  4.29688E-01
               -6.61193E-01  3.15429E-01
                2.83927E+00 -6.59448E-01
               -1.46862E+00 -2.44162E-01
                1.47205E+00  6.97115E-01
               -1.12971E+00 -1.24210E-01]

  foce_ebes_cov = [2.96349E-02  6.43970E-03 6.43970E-03 5.95626E-03
                   1.13840E-01  1.97067E-02 1.97067E-02 1.53632E-02
                   1.37480E-01  2.07479E-02 2.07479E-02 1.42631E-02
                   2.91965E-02  9.78836E-03 9.78836E-03 1.35007E-02
                   3.48071E-02  7.73824E-03 7.73824E-03 7.30707E-03
                   5.87655E-02  2.22284E-02 2.22284E-02 3.92463E-02
                   1.58693E-02  9.82976E-03 9.82976E-03 2.39503E-02
                   5.74869E-02  1.64986E-02 1.64986E-02 2.20832E-02
                   8.51151E-01  3.17941E-02 3.17941E-02 1.25183E-02
                   4.95454E-03  3.20247E-03 3.20247E-03 6.51142E-03
                   4.08289E-01  3.73648E-02 3.73648E-02 2.03508E-02
                   1.46012E-02  5.21247E-03 5.21247E-03 7.67609E-03]

  # Elapsed estimation time in seconds:     0.27
  # Elapsed covariance time in seconds:     0.19

  o = fit(theopmodel_foce, theopp, param, Pumas.FOCE(),
    ensemblealg=EnsembleThreads(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test_throws ArgumentError("EnsembleDistributed() not implemented for this method") fit(theopmodel_foce, theopp, param, Pumas.FOCE(), ensemblealg=EnsembleDistributed())

  o_simobs_1 = simobs(theopmodel_foce, theopp, coef(o), empirical_bayes(o))
  o_simobs_2 = simobs(theopmodel_foce, theopp, coef(o), fill((η₁=0, η₂=0), length(theopp)))
  o_simobs = simobs(o)

  o_estimates = coef(o)
  o_stderror  = stderror(o)

  o_inspect  = inspect(o)
  o_inspect_df = DataFrame(o_inspect)
  @test deviance(o) ≈ 121.89849119366599 rtol=1e-7

  @testset "test estimate of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_estimates, k)) ≈ Pumas._coef_value(getfield(foce_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(foce_stderr, k))           rtol=1e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for (i, ebe) in enumerate(empirical_bayes(o))
    @test ebe.η₁ ≈ foce_ebes[i,1] rtol=1e-2
    @test ebe.η₂ ≈ foce_ebes[i,2] rtol=1e-2
  end

  # Need to estimate this in NONMEM to get a reference
  #ebe_cov = Pumas.empirical_bayes_dist(o)
  #@testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
  #  @test ebe_cov[i].η₁.σ^2 ≈ foce_ebes_cov[i,1] atol=1e-2
  #  @test ebe_cov[i].η₂.σ^2 ≈ foce_ebes_cov[i,2] atol=1e-2
  #end

  @testset "epredict" begin
    Random.seed!(123)
    @test Pumas.epredict(o, nsim=1000)[1].dv ≈ [
      -0.009486499926587882
       2.730732462830414
       4.314322980785372
       4.856654737084268
       1.5421045169975245
    -146.53829323888365
   -2508.783755427784
 -211784.65803870826
      -2.4899564195251968e7
      -3.9191328978627914e10
      -3.063904238107589e23] rtol=1e-2
  end

  @test Pumas._predict(  theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
    0.0
    4.9049755499300955
    7.779812894972983
    8.664908079363599
    7.34893115201493
    5.402130960948489
    4.744864876005044
    4.0473577932698825
    3.4538453107724867
    2.717339295201133
    1.0438615786511665] rtol=1e-6
  @test Pumas._predict( theopmodel_foce, theopp[1], param, Pumas.FOCEI()).dv ≈ [
    0.0
    4.9049755499300955
    7.779812894972983
    8.664908079363599
    7.34893115201493
    5.402130960948489
    4.744864876005044
    4.0473577932698825
    3.4538453107724867
    2.717339295201133
    1.0438615786511665] rtol=1e-6
  @test Pumas._predict(   theopmodel_foce, theopp[1], param, Pumas.FO()).dv ≈ [
    0.0
    4.275044896193946
    6.67731216398507
    7.754622724333498
    7.568031859501575
    6.604016774252041
    5.975950613840706
    5.139785095454612
    4.389649021685277
    3.4538248683011945
    1.3267830785309427] rtol=1e-6
  @test wresiduals(   theopmodel_foce, theopp[1], param, Pumas.FO()).dv ≈ [
     1.1879984032756807
    -0.5121540697427261
     1.177641761770986
     1.8748275064478963
    -1.0811008285733203
    -0.4855189971946874
     0.5270429165085201
     0.7926973976555282
     1.38286211294743
     1.8217361795780522
     2.2749279342557145] rtol=1e-6
  @test wresiduals(  theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
     1.1879984032756807
    -0.40449239905538514
     1.4078834621663032
     1.7442967882790288
    -1.7930916166374602
    -0.7377491261762943
     0.41236539881072987
     0.6557677791850915
     1.229426426138302
     1.6689875690147757
     2.207038336508923] rtol=1e-6
  @test wresiduals( theopmodel_foce, theopp[1], param, Pumas.FOCEI()).dv ≈ [ 1.1879984032756807
   -0.40449239905538514
    1.4078834621663032
    1.7442967882790288
   -1.7930916166374602
   -0.7377491261762943
    0.41236539881072987
    0.6557677791850915
    1.229426426138302
    1.6689875690147757
    2.207038336508923] rtol=1e-6
  @test Pumas.iwresiduals(  theopmodel_foce, theopp[1], param, Pumas.FO()).dv ≈ [
     1.1879984032756807
    -2.303825736901788
    -0.1722792965761087
     4.407437594433976
     3.358452446778005
     3.172249887956898
     3.827360627145415
     3.740934575525829
     4.014071580900914
     3.9913136307052524
     3.1357007891301087] rtol=1e-6
  @test Pumas.iwresiduals( theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
     1.1879984032756807
    -1.3784646740678743
    -0.24469855308692548
     1.9520535480092391
    -1.0273947656943232
    -1.4358103741729111
    -0.3983711793143639
     0.09734551444138731
     0.9017172486653547
     1.5424519077826466
     2.194973069154764] rtol=1e-6
  @test Pumas.iwresiduals(theopmodel_foce, theopp[1], param, Pumas.FOCEI()).dv ≈ [
     1.1879984032756807
    -1.3784646740678743
    -0.24469855308692548
     1.9520535480092391
    -1.0273947656943232
    -1.4358103741729111
    -0.3983711793143639
     0.09734551444138731
     0.9017172486653547
     1.5424519077826466
     2.194973069154764] rtol=1e-6
  @test wresiduals(theopmodel_foce, theopp[1], param, Pumas.FOCE()).dv ≈ [
     1.1879984032756807
    -0.40449239905538514
     1.4078834621663032
     1.7442967882790288
    -1.7930916166374602
    -0.7377491261762943
     0.41236539881072987
     0.6557677791850915
     1.229426426138302
     1.6689875690147757
     2.207038336508923] rtol=1e-6
  @testset "Expected individual residuals" begin
    # The way the random effects enter the model allows for unstable
    # solutions. The first order approximated residuals will not hit
    # the unstable areas but when the random effect is fully integrated
    # out, the unstable solutions will greatly affect the residuals
    Random.seed!(123)
    @test Pumas.eiwres(theopmodel_foce, theopp[1], param, 10000).dv ≈
      [1.1879984032754523
      -1.323115211913782
       3.6933475238777507
      24.023503148347146
    1008.3634654704074
       2.1733344199526347e7
       3.843290423027412e10
       3.524337763526852e15
       5.97837210704398e20
       5.649407093090004e28
       4.352720287029761e60] rtol = 1e-6
  end
end

@testset "run4.mod FOCEI, diagonal omega and additive + proportional error" begin
  theopmodel_focei = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ²_add ∈ RealDomain(lower=0.0001, init=0.388)
      σ²_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2*σ²_prop + σ²_add))
    end
  end

  param = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = Diagonal([5.55, 0.515]),
        σ²_add = 0.388,
        σ²_prop = 0.3
       )

  @test deviance(theopmodel_focei, theopp, param, Pumas.FOCEI()) ≈ 287.08854688950419 rtol=1e-6

  focei_estimated_params = (
    θ₁ = 1.58896E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.52144E-02, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.97132E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.03889E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = Diagonal([1.49637E+00, 2.62862E-01]),
    σ²_add = 2.09834E-01,
    σ²_prop = 1.13479E-02
  )

  focei_stderr = (
    θ₁ = 2.11E-01,
    θ₂ = 4.98E-03,
    θ₃ = 3.53E-03,
    θ₄ = 8.81E-01,

    Ω = Diagonal([1.35E+00, 8.06E-02]),
    σ²_add = 2.64E-01,
    σ²_prop = 1.35E-02)

  focei_ebes = [-3.85812E-01 -1.06806E+00
                 5.27060E-01  5.92269E-02
                 6.72391E-01  6.98525E-02
                -4.69064E-01 -1.90285E-01
                -2.15212E-01  1.46830E-01
                -4.06899E-01  5.11601E-01
                -1.30578E+00  4.62948E-01
                -5.67728E-01  3.40907E-01
                 2.62534E+00 -6.75318E-01
                -1.39161E+00 -2.55351E-01
                 1.46302E+00  7.37263E-01
                -1.12603E+00 -8.76924E-02]

  focei_ebes_cov = [2.48510E-02  7.25437E-03  7.25437E-03 9.35902E-03
                    1.15339E-01  2.18488E-02  2.18488E-02 2.01031E-02
                    1.38914E-01  2.24503E-02  2.24503E-02 1.85587E-02
                    2.46574E-02  1.03515E-02  1.03515E-02 1.78935E-02
                    3.40783E-02  9.74492E-03  9.74492E-03 1.18953E-02
                    4.00166E-02  1.82229E-02  1.82229E-02 3.92109E-02
                    1.18951E-02  8.95307E-03  8.95307E-03 2.70531E-02
                    5.26418E-02  1.67790E-02  1.67790E-02 2.53672E-02
                    8.16838E-01  3.06465E-02  3.06465E-02 1.54578E-02
                    5.91844E-03  4.37397E-03  4.37397E-03 1.05092E-02
                    3.96048E-01  3.79023E-02  3.79023E-02 2.46253E-02
                    1.41688E-02  6.63702E-03  6.63702E-03 1.28516E-02]

  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32

  o = fit(theopmodel_focei, theopp, param, Pumas.FOCEI(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  o_estimates = coef(o)
  o_stderror  = stderror(o)

  o_infer = infer(o)

  # Verify that show runs
  io_buffer = IOBuffer()
  show(io_buffer, o)
  show(io_buffer, o_infer)

  @test deviance(o) ≈ 115.40505379554628 rtol=1e-7
  @testset "test estimate of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_estimates, k)) ≈ Pumas._coef_value(getfield(focei_estimated_params, k)) rtol=1e-3
  end

  @testset "test stderror of $k" for k in keys(o_estimates)
    @test Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(focei_stderr, k))           rtol=1e-2
  end

  @testset "test stored empirical Bayes estimates. Subject: $i" for (i, ebe) in enumerate(empirical_bayes(o))
    @test ebe.η ≈ focei_ebes[i,:]  rtol=1e-3
  end

  ebe_cov = Pumas.empirical_bayes_dist(o)
  @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
    @test ebe_cov[i].η.Σ.mat[:] ≈ focei_ebes_cov[i,:] atol=1e-3
  end

  @testset "epredict" begin
    Random.seed!(123)
    @test Pumas.epredict(o, nsim=1000)[1].dv ≈ [
       -0.006059507309997296
       2.604810337054598
       4.165483661930028
       4.779324471504899
       1.922666298338817
    -119.92850878575545
   -2087.7315464726044
 -143843.9651559656
      -1.2514375387240592e7
      -2.0014235877248787e10
      -5.428287229012377e22] rtol=1e-3
  end

  @test_throws ArgumentError Pumas._predict(  theopmodel_focei, theopp[1], param, Pumas.FOCE()).dv
  @test Pumas._predict( theopmodel_focei, theopp[1], param, Pumas.FOCEI()).dv ≈ [
    0.0
    5.068341459357763
    8.443563123891462
   10.037122038452848
    8.944352418175603
    6.402624012061738
    5.482260711623049
    4.611559621549391
    3.9247518752173303
    3.0865429675088953
    1.185669082816525] rtol=1e-6
  @test Pumas._predict(   theopmodel_focei, theopp[1], param, Pumas.FO()).dv ≈ [
    0.0
    4.275044896193946
    6.67731216398507
    7.754622724333498
    7.568031859501575
    6.604016774252041
    5.975950613840706
    5.139785095454612
    4.389649021685277
    3.4538248683011945
    1.3267830785309427]
  @test wresiduals(   theopmodel_focei, theopp[1], param, Pumas.FO()).dv ≈ [
     1.1879984032756807
    -0.39299505013030017
     0.16842737988346182
     0.6720210050845926
     0.406187009843474
     0.3813592513772635
     0.5047498527549048
     0.5421470902094254
     0.6842313630989655
     0.866326879877313
     1.667259055010247]
  @test_throws ArgumentError wresiduals(  theopmodel_focei, theopp[1], param, Pumas.FOCE())
  @test wresiduals( theopmodel_focei, theopp[1], param, Pumas.FOCEI()).dv ≈ [
   1.1879984032756807
  -0.4660532896569592
   0.30402244275648965
   0.6449237449945702
   0.3044150909651889
   0.33203669572444244
   0.4203515581077581
   0.4278768983990183
   0.5204742127712235
   0.6506387271130077
   1.3748928119024197] rtol=1e-6
  @test Pumas.iwresiduals(  theopmodel_focei, theopp[1], param, Pumas.FO()).dv ≈ [ 1.1879984032756807
   -0.5922659254289279
   -0.028925269913169835
    0.6395285872285701
    0.49907131896046675
    0.5383548707639977
    0.715521262325209
    0.80818616332361
    1.006709008521727
    1.2482988777536315
    2.0406927145826876]
  @test_throws ArgumentError Pumas.iwresiduals( theopmodel_focei, theopp[1], param, Pumas.FOCE())
  @test Pumas.iwresiduals(theopmodel_focei, theopp[1], param, Pumas.FOCEI()).dv ≈ [1.1879984032756807
    0.06960158205526236
    0.5083448884321163
    0.7692001749717815
    0.23983260753879304
    0.10697777173751535
    0.23456187417479818
    0.30756958787462435
    0.47263581944050936
    0.680189149628738
    1.4895796376772368] rtol=1e-6
  @test_throws ArgumentError wresiduals(  theopmodel_focei, theopp[1], param, Pumas.FOCE())
  @testset "Expected individual residuals" begin
    Random.seed!(123)
    @test Pumas.eiwres( theopmodel_focei, theopp[1], param, 10000).dv ≈
      [1.1879984032754523,
       0.19790477137989876,
       0.9852034524282582,
       1.5985195166192314,
       1.119122290403704,
       0.9088239792374705,
       1.0000563672670204,
       1.0185370204348032,
       1.1548159881352817,
       1.326267705476969,
       1.9179530514665144]
  end
end

@testset "run4_foce.mod FOCE, diagonal omega and additive + proportional error" begin
  theopmodel_foce = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ²_add ∈ RealDomain(lower=0.0001, init=0.388)
      σ²_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2*σ²_prop + σ²_add))
    end
  end

  param = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
             θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
             θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
             θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

             Ω = Diagonal([5.55, 0.515]),
             σ²_add = 0.388,
             σ²_prop = 0.3
            )

  # FOCE is not allowed for models where dispersion parameter depends on the random effects
  @test_throws ArgumentError deviance(theopmodel_foce, theopp[1], param, Pumas.FOCE())
end

@testset "run5.mod LaplaceI without interaction, diagonal omega and additive error" begin

  theopmodel_laplace = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PDiagDomain(2)
      σ²_add ∈ RealDomain(lower=0.0001, init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂
      CL = θ₃*WT + η[2]
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
        conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
        dv ~ @. Normal(conc, sqrt(σ²_add))
    end
  end

  param = (θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
        θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
        θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
        θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        Ω = Diagonal([5.55, 0.515 ]),
        σ²_add = 0.388
        #σ_prop = 0.3
       )

  nonmem_ebes_initial = [[-1.29964E+00, -8.32445E-01],
                         [-5.02283E-01,  1.04703E-01],
                         [-2.43001E-01,  1.51238E-01],
                         [-1.51174E+00, -1.02357E-01],
                         [-1.19255E+00,  2.15326E-01],
                         [-1.43398E+00,  5.89950E-01],
                         [-6.90249E-01,  5.11417E-01],
                         [ 1.15977E-02,  3.87675E-01],
                         [ 4.83842E+00, -5.54201E-01],
                         [-8.29081E-01, -1.63510E-01],
                         [ 2.63035E+00,  7.77899E-01],
                         [-4.82787E-01, -5.32906E-02]]

  @testset "Empirical Bayes estimates" begin
    for (i,η) in enumerate(nonmem_ebes_initial)
      @test sqrt(param.Ω)*Pumas._orth_empirical_bayes(theopmodel_laplace, theopp[i], param, Pumas.LaplaceI()) ≈ η rtol=1e-4
    end

    @test deviance(theopmodel_laplace, theopp, param, Pumas.LaplaceI()) ≈ 141.296 atol=1e-3
  end

  laplace_estimated_params = (
    θ₁ = 1.68975E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 8.54637E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 3.95757E-02, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 2.11952E+00, #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    Ω = Diagonal([1.596, 2.27638e-01]),
    σ²_add = 5.14457E-01
  )

  laplace_stderr = (
    θ₁ = 1.87E-01,
    θ₂ = 5.45E-03,
    θ₃ = 3.42E-03,
    θ₄ = 9.69E-01,

    Ω = Diagonal([1.60E+00, 7.76E-02]),
    σ²_add = 1.35E-01)

  laplace_ebes =[-2.81817E-01 -9.51075E-01
                  4.36123E-01  2.95020E-02
                  6.11776E-01  7.42657E-02
                 -4.86815E-01 -1.82637E-01
                 -1.91050E-01  1.64544E-01
                 -4.45223E-01  4.75049E-01
                 -1.35543E+00  4.29076E-01
                 -6.79064E-01  3.15307E-01
                  2.80934E+00 -6.62060E-01
                 -1.48446E+00 -2.47635E-01
                  1.44666E+00  6.99712E-01
                 -1.14629E+00 -1.26537E-01]

  laplace_ebes_cov = [2.93845E-02  6.58620E-03  6.58620E-03  6.15918E-03
                      8.77655E-02  1.52441E-02  1.52441E-02  1.46969E-02
                      1.34060E-01  2.02319E-02  2.02319E-02  1.42611E-02
                      2.50439E-02  8.56810E-03  8.56810E-03  1.33066E-02
                      2.80689E-02  6.29305E-03  6.29305E-03  7.02909E-03
                      4.82183E-02  1.79056E-02  1.79056E-02  3.66785E-02
                      1.42044E-02  8.76942E-03  8.76942E-03  2.30211E-02
                      5.16763E-02  1.48793E-02  1.48793E-02  2.15167E-02
                      5.14173E-01  1.61073E-02  1.61073E-02  1.22365E-02
                      5.22194E-03  3.44274E-03  3.44274E-03  6.76806E-03
                      3.23248E-01  2.75247E-02  2.75247E-02  1.88999E-02
                      1.23520E-02  4.49397E-03  4.49397E-03  7.51682E-03]

  # Elapsed estimation time in seconds:     0.23
  # Elapsed covariance time in seconds:     0.17

  @test deviance(theopmodel_laplace, theopp, laplace_estimated_params, Pumas.LaplaceI()) ≈ 123.76439574418291 atol=1e-3

  @testset "Test optimization" begin
    o = fit(theopmodel_laplace, theopp, param, Pumas.LaplaceI(),
      optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

    o_estimates = coef(o)
    o_stderror  = stderror(o)

    o_infer = infer(o)
    o_inspect = inspect(o)

    # Verify that show runs
    io_buffer = IOBuffer()
    show(io_buffer, o)
    show(io_buffer, o_infer)

    @test deviance(o) ≈ 123.76439574418291 rtol=1e-5

    @testset "test estimate of $k" for k in keys(o_estimates)
      @test Pumas._coef_value(getfield(o_estimates, k)) ≈ Pumas._coef_value(getfield(laplace_estimated_params, k)) rtol=2e-3
    end

    @testset "test stderror of $k" for k in keys(o_estimates)
      @test Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(laplace_stderr, k))           rtol=1e-2
    end

    @testset "test stored empirical Bayes estimates. Subject: $i" for (i, ebe) in enumerate(empirical_bayes(o))
      @test ebe.η ≈ laplace_ebes[i,:] rtol=3e-3
    end

    ebe_cov = Pumas.empirical_bayes_dist(o)
    @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
        @test ebe_cov[i].η.Σ.mat[:] ≈ laplace_ebes_cov[i,:] atol=1e-3
    end
  end
end

@testset "run6.mod LaplaceI, diagonal omega and additive + proportional error" begin

  theopmodel_laplacei = @model begin
    @param begin
      # θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      # θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      # θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      # θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      # Use VectorDomain although RealDomain is preferred to make sure that VectorDomain is tested
      θ ∈ VectorDomain(4, lower=[0.1 , 0.0008, 0.004 , 0.1],
                          init =[2.77, 0.0781, 0.0363, 1.5],
                          upper=[5.0 , 0.5   , 0.9   , 5.0])
      ω²Ka ∈ RealDomain(lower=0.0)
      ω²CL ∈ RealDomain(lower=0.0)
      σ²_add ∈ RealDomain(lower=0.0001, init=0.388)
      σ²_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end

    @random begin
      ηKa ~ Normal(0.0, sqrt(ω²Ka))
      ηCL ~ Normal(0.0, sqrt(ω²CL))
    end

    @pre begin
      Ka = (SEX == 0 ? θ[1] : θ[4]) + ηKa
      K  = θ[2]
      CL = θ[3]*WT + ηCL
      Vc = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2*σ²_prop + σ²_add))
    end
  end

  param = (θ = [2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
                0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
                0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
                1.5],   #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

        ω²Ka = 5.55,
        ω²CL = 0.515,
        σ²_add = 0.388,
        σ²_prop = 0.3
       )

  @test deviance(theopmodel_laplacei, theopp, param, Pumas.LaplaceI()) ≈ 288.30901928585990 rtol=1e-6

  laplacei_estimated_params = (
    θ = [1.60941E+00,  #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
         8.55663E-02,  #K MEAN ELIMINATION RATE CONSTANT (1/HR)
         3.97472E-02,  #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
         2.05830E+00], #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)

    ω²Ka = 1.48117E+00,
    ω²CL = 2.67215E-01,
    σ²_add = 1.88050E-01,
    σ²_prop = 1.25319E-02
  )

  laplacei_stderr = (
    θ = [2.09E-01,
         4.48E-03,
         3.29E-03,
         9.05E-01],

    ω²Ka = 1.35E+00,
    ω²CL = 8.95E-02,
    σ²_add = 3.01E-01,
    σ²_prop = 1.70E-02)

  laplacei_ebes = [-4.28671E-01 -1.07834E+00
                    5.07444E-01  6.93096E-02
                    6.43525E-01  7.59602E-02
                   -4.98953E-01 -1.84868E-01
                   -2.46292E-01  1.51559E-01
                   -4.38459E-01  5.21590E-01
                   -1.33160E+00  4.71199E-01
                   -5.83976E-01  3.49324E-01
                    2.57215E+00 -6.69523E-01
                   -1.41344E+00 -2.53978E-01
                    1.43116E+00  7.48520E-01
                   -1.15300E+00 -7.93525E-02]

  laplacei_ebes_cov = [2.50922E-02  8.73478E-03  8.73478E-03  1.09246E-02
                       1.05367E-01  2.07642E-02  2.07642E-02  2.00787E-02
                       1.32364E-01  2.21959E-02  2.21959E-02  1.90778E-02
                       2.27392E-02  1.05318E-02  1.05318E-02  1.84061E-02
                       3.29183E-02  1.03562E-02  1.03562E-02  1.25563E-02
                       3.37488E-02  1.74040E-02  1.74040E-02  3.91317E-02
                       1.04365E-02  8.73379E-03  8.73379E-03  2.72465E-02
                       5.56885E-02  1.72519E-02  1.72519E-02  2.57910E-02
                       4.58448E-01  2.47681E-02  2.47681E-02  1.58825E-02
                       6.12637E-03  4.60375E-03  4.60375E-03  1.11188E-02
                       3.11433E-01  3.31735E-02  3.31735E-02  2.47868E-02
                       1.23976E-02  6.46725E-03  6.46725E-03  1.29075E-02]

  # Elapsed estimation time in seconds:     0.30
  # Elapsed covariance time in seconds:     0.32

  @testset "Test optimization" begin
    o = fit(theopmodel_laplacei, theopp, param, Pumas.LaplaceI(),
      optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

    o_estimates = coef(o)
    o_stderror  = stderror(o)

    @test deviance(o) ≈ 116.97275684239327 rtol=1e-5

    @testset "test estimate of $k" for k in keys(o_estimates)
      @test Pumas._coef_value(getfield(o_estimates, k)) ≈ Pumas._coef_value(getfield(laplacei_estimated_params, k)) rtol=1e-3
    end

    @testset "test stderror of $k" for k in keys(o_estimates)
      if k != :ω²CL
        @test Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(laplacei_stderr, k))           rtol=4e-2
      else
        # This used to match but is now 10% different from the NONMEM estimate after we have switched to
        # using standard normal random effects internally
        @test_broken Pumas._coef_value(getfield(o_stderror, k))  ≈ Pumas._coef_value(getfield(laplacei_stderr, k))           rtol=4e-2
      end
    end

    @testset "test stored empirical Bayes estimates. Subject: $i" for (i, ebe) in enumerate(empirical_bayes(o))
      @test [ebe...] ≈ laplacei_ebes[i,:] rtol=1e-3
    end

    ebe_cov = Pumas.empirical_bayes_dist(o)
    @testset "test covariance of empirical Bayes estimates. Subject: $i" for i in 1:length(theopp)
      @test ebe_cov[i].ηKa.σ^2 ≈ first(laplacei_ebes_cov[i,:]) rtol=1e-3
      @test ebe_cov[i].ηCL.σ^2 ≈ last( laplacei_ebes_cov[i,:]) rtol=1e-3
    end

    @testset "Cubature based estimation deviance test" begin
      @test deviance(theopmodel_laplacei, theopp, param, Pumas.LLQuad(), iabstol=0, ireltol=0, imaxiters=typemax(Int)) ≈ 281.1606964897779 rtol=1e-6
    end
  end
end
end
