using Test
using Pumas

@testset "Gamma-distributed error model" begin

  data = read_pumas(example_data("sim_data_model1"))

  mdsl = @model begin
    @param begin
      θ  ∈ RealDomain(init=0.5)
      Ω  ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
      ν  ∈ RealDomain(lower=0.01, init=1.0)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      CL = θ * exp(η[1])
      Vc = 1.0
    end

    @vars begin
      # Currently, Gamma is a bit picky about zeros in the parameters
      conc = Central / Vc + 1e-10
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Gamma(ν, conc/ν)
    end
  end

  param = init_param(mdsl)

  # Not supported
  @test_throws ArgumentError deviance(mdsl, data, param, Pumas.FO())
  @test_throws ArgumentError deviance(mdsl, data, param, Pumas.FOCEI())

  @test deviance(mdsl, data, param, Pumas.FOCE())     ≈ 88.9136079338946 rtol=1e-6
  @test deviance(mdsl, data, param, Pumas.LaplaceI()) ≈ 88.9571564205892 rtol=1e-6

  ft_FOCE = fit(mdsl, data, param, Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test deviance(ft_FOCE) ≈ 56.11354389316806 rtol=1e-6
  @test_throws ArgumentError("weighted residuals only implemented for Gaussian error models") wresiduals(ft_FOCE)
  @test DataFrame(inspect(ft_FOCE)) isa DataFrame

  ft_LaplaceI = fit(mdsl, data, param, Pumas.LaplaceI(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test deviance(ft_LaplaceI) ≈ 55.96605418561208 rtol=1e-6
  @test_throws ArgumentError("weighted residuals only implemented for Gaussian error models") wresiduals(ft_LaplaceI)
  @test DataFrame(inspect(ft_LaplaceI)) isa DataFrame
end
