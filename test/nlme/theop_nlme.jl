using Test
using Pumas

@testset "More tests based on the theophylline dataset" begin

  theopp_nlme = read_pumas(example_data("THEOPP"))

  mdsl2 = @model begin
    @param begin
      θ ∈ VectorDomain(3,init=[3.24467E+01, 8.72879E-02, 1.49072E+00])
      Ω ∈ PSDDomain(init=Matrix{Float64}([ 1.93973E-02  1.20854E-02  5.69131E-02
                                           1.20854E-02  2.02375E-02 -6.47803E-03
                                           5.69131E-02 -6.47803E-03  4.34671E-01]))
      Σ ∈ PDiagDomain(init=[1.70385E-02, 8.28498E-02])
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      V  = θ[1] * exp(η[1])
      Ke = θ[2] * exp(η[2])
      Ka = θ[3] * exp(η[3])
      CL = Ke * V
    end

    @vars begin
      conc = Central / V
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(conc^2 *Σ.diag[1] + Σ.diag[end]) + eps())
    end
  end

  param = init_param(mdsl2)

  @test @inferred(deviance(mdsl2, theopp_nlme, param, Pumas.LaplaceI())) ≈ 93.64166638742198 rtol = 1e-6 # NONMEM result
  @test_throws ArgumentError fit(mdsl2, theopp_nlme, param, Pumas.FOCE())

  ft_focei = fit(mdsl2, theopp_nlme, param, Pumas.FOCEI())
  @test ft_focei isa Pumas.FittedPumasModel

  @test ηshrinkage(mdsl2, theopp_nlme, param, Pumas.FOCEI()).η ≈ [0.0161871, 0.0502453, 0.0133019] rtol = 1e-5

  @test ϵshrinkage(mdsl2, theopp_nlme, param, Pumas.FOCEI()).dv ≈ 0.09091845 rtol = 1e-6
  ϵshrinkage(mdsl2,theopp_nlme, param, Pumas.FOCEI(),
    [Pumas._orth_empirical_bayes(mdsl2, subject, param, Pumas.FOCEI()) for subject in theopp_nlme]).dv
  param = coef(ft_focei)
  @test ϵshrinkage(mdsl2, theopp_nlme, param, Pumas.FOCEI(),
    [Pumas._orth_empirical_bayes(mdsl2, subject, param, Pumas.FOCEI()) for subject in theopp_nlme]).dv ≈ 0.09091976569587323 rtol = 1e-3
  @test ϵshrinkage(mdsl2, theopp_nlme, param, Pumas.FOCEI()).dv ≈ 0.09091976569587323 rtol = 1e-3

  @test aic(ft_focei) ≈ 357.43064186213104 rtol = 1e-3 #regression test
  @test bic(ft_focei) ≈ 389.1414630105811  rtol = 1e-3 #regression test

  param = init_param(mdsl2)
  randeffsorth = [Pumas._orth_empirical_bayes(mdsl2, subject, param, Pumas.FOCEI()) for subject in theopp_nlme]

  @test [Pumas._ipredict(mdsl2, subject, param, Pumas.FO(), randeff).dv for (subject, randeff) in zip(theopp_nlme, randeffsorth)]   isa Vector # FIXME! come up with a better test
  @test [Pumas._ipredict(mdsl2, subject, param, Pumas.FOCE(), randeff).dv for (subject, randeff) in zip(theopp_nlme, randeffsorth)]  isa Vector # FIXME! come up with a better test
  @test [Pumas._ipredict(mdsl2, subject, param, Pumas.FOCEI(), randeff).dv for (subject, randeff) in zip(theopp_nlme, randeffsorth)] isa Vector # FIXME! come up with a better test
  @test residuals(ft_focei) isa Vector{<:NamedTuple} # FIXME! come up with a better test

end