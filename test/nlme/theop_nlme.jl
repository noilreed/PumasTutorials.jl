using Test, Pumas, Random, StaticArrays

@testset "More tests based on the theophylline dataset" begin

  theopp_nlme = read_pumas(example_data("THEOPP"))

  models = Dict()
  models["closed_form"] = @model begin
    @param begin
      θ ∈ VectorDomain(3,init=[3.24467E+01, 8.72879E-02, 1.49072E+00], lower=zeros(3))
      Ω ∈ PSDDomain(init=Matrix{Float64}([
        1.93973E-02  1.20854E-02  5.69131E-02
        1.20854E-02  2.02375E-02 -6.47803E-03
        5.69131E-02 -6.47803E-03  4.34671E-01]))
      Σ ∈ PDiagDomain(init=[1.70385E-02, 8.28498E-02])
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Vc = θ[1] * exp(η[1])
      Ke = θ[2] * exp(η[2])
      Ka = θ[3] * exp(η[3])
      CL = Ke * Vc
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2 *Σ.diag[1] + Σ.diag[end]) + eps())
    end
  end

  models["linearODE_mutable"] = @model begin
    @param begin
      θ ∈ VectorDomain(3,init=[3.24467E+01, 8.72879E-02, 1.49072E+00], lower=zeros(3))
      Ω ∈ PSDDomain(init=Matrix{Float64}([ 1.93973E-02  1.20854E-02  5.69131E-02
                                           1.20854E-02  2.02375E-02 -6.47803E-03
                                           5.69131E-02 -6.47803E-03  4.34671E-01]))
      Σ ∈ PDiagDomain(init=[1.70385E-02, 8.28498E-02])
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Vc = θ[1] * exp(η[1])
      Ke = θ[2] * exp(η[2])
      Ka = θ[3] * exp(η[3])
      CL = Ke * Vc
      A = [-Ka 0
            Ka -Ke]
    end

    @init begin
      Depot = 0.0
      Central = 0.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics LinearODE

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2 *Σ.diag[1] + Σ.diag[end]) + eps())
    end
  end

  models["linearODE_immutable"] = @model begin
    @param begin
      θ ∈ VectorDomain(3,init=[3.24467E+01, 8.72879E-02, 1.49072E+00], lower=zeros(3))
      Ω ∈ PSDDomain(init=Matrix{Float64}([ 1.93973E-02  1.20854E-02  5.69131E-02
                                           1.20854E-02  2.02375E-02 -6.47803E-03
                                           5.69131E-02 -6.47803E-03  4.34671E-01]))
      Σ ∈ PDiagDomain(init=[1.70385E-02, 8.28498E-02])
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Vc = θ[1] * exp(η[1])
      Ke = θ[2] * exp(η[2])
      Ka = θ[3] * exp(η[3])
      CL = Ke * Vc
      A = @SMatrix(
        [-Ka 0
          Ka -Ke])
    end

    @init begin
      Depot = 0.0
      Central = 0.0
    end

    @vars begin
      conc = Central / Vc
    end

    @dynamics LinearODE

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2 *Σ.diag[1] + Σ.diag[end]) + eps())
    end
  end

  # Initial values are also the optimum values
  param = init_param(models["closed_form"])

  @testset "model type: $mn" for mn in keys(models)
    model = models[mn]

    @test @inferred(deviance(model, theopp_nlme, param, Pumas.LaplaceI())) ≈ 93.64166638742198 rtol = 1e-6 # NONMEM result
    @test_throws ArgumentError fit(model, theopp_nlme, param, Pumas.FOCE())
    Pumas._orth_empirical_bayes(model, theopp_nlme[1], param, Pumas.FOCEI())

    ft_focei = fit(model, theopp_nlme, param, Pumas.FOCEI(),
      optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
    @test ft_focei isa Pumas.FittedPumasModel

    @test ηshrinkage(ft_focei).η ≈ [0.016186940309938502, 0.050245367209653, 0.01330152716722488] rtol = 1e-4

    @test ϵshrinkage(ft_focei).dv ≈ 0.09091845 rtol = 1e-4

    @test aic(ft_focei) ≈ 357.43064186213104 rtol = 1e-3 #regression test

    @test bic(ft_focei) ≈ 389.1414630105811  rtol = 1e-3 #regression test

    param = init_param(model)

    @test DataFrame(predict(ft_focei)).dv_ipred[1:3] ≈ [0.0, 3.379491664355948, 6.260410839825305] rtol=1e-4
    @test DataFrame(predict(ft_focei)).dv_pred[1:3] ≈ [0.0, 3.0330237813961167, 5.488490519472397] rtol=1e-4

    @testset "NPDEs" begin
      Random.seed!(123)
      @test Pumas.npde(ft_focei, nsim=1000)[1].dv ≈ [
        2.457263390205441
       -0.05266352689406844
        1.3285393288568097
        1.9773684281819504
        0.04764395595447655
       -0.24558952342208085
        0.48736456546944085
        0.606775363514265
        0.7290027178052186
        1.089349027924277
        2.747781385444999] rtol=1e-3
      @test_throws ArgumentError Pumas.npde(ft_focei)
      @test_throws ArgumentError Pumas.npde(ft_focei, nsim=0)
    end

    @testset "epredict" begin
      Random.seed!(123)
      @test Pumas.epredict(ft_focei, nsim=1000)[1].dv ≈ [
        -0.0027972258116983418
         3.234683322868215
         5.501826107459644
         7.076473939238916
         7.861191437196706
         7.405571508929426
         6.67191665843221
         5.749998044125176
         4.928335513914535
         3.7718717351961732
         1.3507119497361435] rtol=1e-4
      @test_throws ArgumentError Pumas.epredict(ft_focei)
      @test_throws ArgumentError Pumas.epredict(ft_focei, nsim=0)
    end

    pred1  = predict(ft_focei)
    predFO = [Pumas.__predict(ft_focei.model, subject, coef(ft_focei), vrandeffsorth, Pumas.FO()) for (subject, vrandeffsorth) in zip(ft_focei.data, ft_focei.vvrandeffsorth)]
    predFOCEI = [Pumas.__ipredict(ft_focei.model, subject, coef(ft_focei), vrandeffsorth) for (subject, vrandeffsorth) in zip(ft_focei.data, ft_focei.vvrandeffsorth)]

    @test sum(sum(pred1[i].pred.dv - predFO[i].dv for i = 1:length(theopp_nlme))) == 0
    @test abs(sum(sum(pred1[i].ipred.dv.-predFOCEI[i].dv for i = 1:length(theopp_nlme)))) < 1e-4
  end
end
