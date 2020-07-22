using Test
using Pumas

@testset "Exponentially distributed error model" begin

  data = read_pumas(example_data("sim_data_model1"))

  mdsl = @model begin
    @param begin
      θ  ∈ RealDomain(init=0.5)
      Ω  ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      CL = θ * exp(η[1])
      Vc = 1.0
    end

    @vars begin
      # Currently, Exponential is a bit picky about zeros in the parameters
      conc = Central / Vc + 1e-10
    end

    @dynamics Central1

    @derived begin
      dv ~ @. Exponential(conc)
    end
  end

  param = init_param(mdsl)

  # Not supported
  @test_throws ArgumentError loglikelihood(mdsl, data, param, Pumas.FO())
  @test_throws ArgumentError loglikelihood(mdsl, data, param, Pumas.FOCEI())

  @test loglikelihood(mdsl, data, param, Pumas.FOCE())     ≈ -62.83557462725 rtol=1e-6
  @test loglikelihood(mdsl, data, param, Pumas.LaplaceI()) ≈ -62.85734887439 rtol=1e-6

  @test loglikelihood(fit(mdsl, data, param, Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))) ≈ -62.68401676062138 rtol=1e-6
  @test loglikelihood(fit(mdsl, data, param, Pumas.LaplaceI(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))) ≈ -62.68385851468949 rtol=1e-6

end
