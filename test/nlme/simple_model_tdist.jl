using Test
using Pumas

@testset "T-distributed error model" begin

data = read_pumas(example_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
tdist = @model begin
  @param begin
    θ ∈ RealDomain(init=0.5)
    Ω ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
    σ ∈ RealDomain(lower=0.001, upper=1.0, init=sqrt(0.1))
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ * exp(η[1])
    Vc = 1.0
  end

  @vars begin
    conc = Central / Vc
  end

  @dynamics Central1

  @derived begin
    dv ~ @. LocationScale(conc, conc*σ, TDist(30))
  end
end


param = init_param(tdist)

for (d, v) in zip(data, [-0.110095703125000, 0.035454025268555, -0.024982604980469,
                         -0.085317642211914, 0.071675025939941, 0.059612709045410,
                         -0.110117317199707, -0.024778854370117, -0.053085464477539,
                         -0.002428230285645])
  @test (sqrt(param.Ω)*Pumas._orth_empirical_bayes(tdist, d, param, Pumas.LaplaceI()))[1] ≈ v rtol=1e-3
end

# Not yet supported
@test_throws ArgumentError deviance(tdist, data, param, Pumas.FO())
@test_throws ArgumentError deviance(tdist, data, param, Pumas.FOCE())
@test_throws ArgumentError deviance(tdist, data, param, Pumas.FOCEI())
@test deviance(tdist, data, param, Pumas.LaplaceI()) ≈ 57.112537604068990 rtol=1e-6

end
