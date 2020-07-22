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
  @test_throws ArgumentError loglikelihood(mdsl, data, param, Pumas.FO())
  @test_throws ArgumentError loglikelihood(mdsl, data, param, Pumas.FOCEI())

  @test loglikelihood(mdsl, data, param, Pumas.FOCE())     ≈ -62.83557462725005 rtol=1e-6
  @test loglikelihood(mdsl, data, param, Pumas.LaplaceI()) ≈ -62.85734887438806 rtol=1e-6

  ft_FOCE = fit(mdsl, data, param, Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test loglikelihood(ft_FOCE) ≈ -46.43553768980682 rtol=1e-6
  @test_throws ArgumentError("weighted residuals only implemented for Gaussian error models") wresiduals(ft_FOCE)
  @test DataFrame(inspect(ft_FOCE)) isa DataFrame

  ft_LaplaceI = fit(mdsl, data, param, Pumas.LaplaceI(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  @test loglikelihood(ft_LaplaceI) ≈ -46.36179773482574 rtol=1e-6
  @test_throws ArgumentError("weighted residuals only implemented for Gaussian error models") wresiduals(ft_LaplaceI)
  @test DataFrame(inspect(ft_LaplaceI)) isa DataFrame
end
