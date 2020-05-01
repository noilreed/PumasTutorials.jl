using Pumas

@testset "issue 779" begin
mdl = @model begin
  @param begin
    θ ∈ VectorDomain(11, init=fill(0.5, 100))
    Ω ∈ PSDDomain(11)
    σ ∈ RealDomain(lower=0.0, init=1.0)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates x1 x2 x3 x4 x5 x6 x7 x8 x9

  @pre begin
    m = x1*θ[1] + x2*θ[2] + x3*θ[3] + x4*θ[4] + x5*θ[5] + x6*θ[6] + x7*θ[7] + x8*θ[8] + x9*θ[9] + θ[10] + sum(η)
  end

  @derived begin
    dv ~ @. Normal(m, σ)
  end
end

param = init_param(mdl)

cvs = NamedTuple{ntuple(i -> Symbol("x$i"), 9)}(ntuple(i -> randn(), 9))

pop = [Subject(id=i, obs=(dv=[randn()],), cvs=cvs, time=[0.0]) for i in 1:100]

ft = fit(mdl, pop, param, Pumas.FOCEI())

io_buffer = IOBuffer()
show(io_buffer, ft)
end
