using Test
using Pumas, Distributions
import TransformVariables

@testset "ParamSets and Domains tests" begin
  p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
                Ω = PSDDomain(2),
                Σ = RealDomain(lower=0.0, init=1.0),
                a = RealDomain(lower=0.0, upper=1.0, init=0.2)))

  t = Pumas.totransform(p)
  @test TransformVariables.dimension(t) == 9
  u = TransformVariables.transform(t, zeros(9))
  @test all(u.θ .> 0)
  @test u.Ω isa Pumas.PDMats.AbstractPDMat


  @testset "Transformations" for Ω in (
    InverseWishart(13.0, [1.0 0.2; 0.2 1.0]),
    Wishart(13.0, [1.0 0.2; 0.2 1.0]))

    pd = ParamSet((θ = Constrained(MvNormal([1.0 0.2; 0.2 1.0]), lower=-2.0),
                 Ω = Ω))
    td = Pumas.totransform(pd)
    @test TransformVariables.dimension(td) == 5
    ud = TransformVariables.transform(td, zeros(5))
    @test all(ud.θ .> -2.0)
    @test ud.Ω isa Pumas.PDMats.AbstractPDMat
    @test logpdf(pd.params.Ω, ud.Ω) isa Number
  end

  @testset "Promotion" begin
    d = RealDomain(lower=0, upper=1.0)
    @test d.lower === 0
    @test d.upper === 1.0
    d = VectorDomain(2,lower=[0  , 2.0], upper=[10  , 4  ], init=[2, 2])
    @test (d.lower...,) == (0.0, 2.0)
    @test (d.upper...,) == (10.0, 4.0)
    @test (d.init...,)  == (2, 2)
  end

  @testset "Constrained should constrain" begin
    d = Constrained(MvNormal(fill(1.0, 1, 1)))
    @test logpdf(d, [0]) ≈ -0.9189385332046728

    d = Constrained(MvNormal(fill(1.0, 1, 1)), lower=-1)
    @test logpdf(d, [ 0]) ≈ -0.9189385332046728
    @test logpdf(d, [-2]) ≈ -Inf

    d = Constrained(MvNormal(fill(1.0, 1, 1)), upper=1)
    @test logpdf(d, [0]) ≈ -0.9189385332046728
    @test logpdf(d, [2]) ≈ -Inf

    d = Constrained(MvNormal(fill(1.0, 1, 1)), lower=-1, upper=1)
    @test logpdf(d, [ 0]) ≈ -0.9189385332046728
    @test logpdf(d, [-2]) ≈ -Inf
    @test logpdf(d, [ 2]) ≈ -Inf
  end
end
