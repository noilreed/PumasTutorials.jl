using Test, Pumas, Random

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

  # Initial values are also the optimum values
  param = init_param(mdsl2)

  @test @inferred(deviance(mdsl2, theopp_nlme, param, Pumas.LaplaceI())) ≈ 93.64166638742198 rtol = 1e-6 # NONMEM result
  @test_throws ArgumentError fit(mdsl2, theopp_nlme, param, Pumas.FOCE())

  ft_focei = fit(mdsl2, theopp_nlme, param, Pumas.FOCEI())
  @test ft_focei isa Pumas.FittedPumasModel

  @test ηshrinkage(ft_focei).η ≈ [0.016186940309938502, 0.050245367209653, 0.01330152716722488] rtol = 1e-5

  @test ϵshrinkage(ft_focei).dv ≈ 0.09091845 rtol = 1e-6

  @test aic(ft_focei) ≈ 357.43064186213104 rtol = 1e-3 #regression test

  @test bic(ft_focei) ≈ 389.1414630105811  rtol = 1e-3 #regression test

  param = init_param(mdsl2)

  @test DataFrame(predict(ft_focei, Pumas.FO())).dv_ipred[1:3] ≈ [0.0, 3.0330238705656996,5.488490667058586] rtol=1e-4
  @test_throws ArgumentError DataFrame(predict(ft_focei, Pumas.FOCE()))
  @test DataFrame(predict(ft_focei, Pumas.FOCEI())).dv_ipred[1:3] ≈ [0.0, 3.379491771652512, 6.260411022764525] rtol=1e-4

  @testset "NPDEs" begin
    Random.seed!(123)
    @test Pumas.npde(ft_focei, nsim=1000)[1].dv ≈ [2.652069807902201, -0.1307159681198632, 1.4985130678799756, 2.0537489106318274, -0.015040336678635668, -0.1738288125016624, 0.5129304106147282, 0.7224790519280626, 1.089349027924277, 1.425544037080452, 2.4089155458154656] rtol=1e-3
    @test_throws ArgumentError Pumas.npde(ft_focei)
    @test_throws ArgumentError Pumas.npde(ft_focei, nsim=0)
  end

  @testset "epredict" begin
    Random.seed!(123)
    @test Pumas.epredict(ft_focei, nsim=1000)[1].dv ≈ [0.005341439351865666, 3.1535278693438586, 5.3450233936232685, 7.07119271562188, 7.869752335670085, 7.394229963732839, 6.790124473121549, 5.8301650736352215, 4.954683233076249, 3.7942281674828484, 1.3618504849233595] rtol=1e-4
    @test_throws ArgumentError Pumas.epredict(ft_focei)
    @test_throws ArgumentError Pumas.epredict(ft_focei, nsim=0)
  end
end