using Pumas, Test, ForwardDiff

include("testmodels.jl")

@testset "Median size ODE problem (HCV model)" begin

  @testset "numtype shouldn't allocate for large named tuples" begin
    _param = (TESTMODELS["HCV"]["params"]..., dummy=ForwardDiff.Dual(1.0, 1.0))
    # Run once to compile
    Pumas.numtype(_param)
    # Then test that it no longer allocates
    @test @allocated(Pumas.numtype(_param)) == 0
  end

  ft = fit(
    TESTMODELS["HCV"]["model"],
    TESTMODELS["HCV"]["data"],
    TESTMODELS["HCV"]["params"],
    Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(x_reltol=1e-3))

  @test deviance(ft) ≈ -100.78347276087511 rtol=1e-4

end
