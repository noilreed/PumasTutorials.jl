using Pumas, Random
@testset "vcov error handling in infer" begin
Random.seed!(964)
# Baseline model
model0 = @model begin
@param begin
  par ∈ RealDomain(lower=0.0, upper=1.0)
end

@pre begin
    CL = par
    Vc = 5.0
end

@dynamics Central1

@derived begin
    conc ~ @. Normal(CL/Vc, 0.001)
end
end

# Simulate data
dr = DosageRegimen(293)

_time = [0.1,0.2,0.3,0.4,0.44]
pop = map(i->Subject(id=i, obs=(conc=[],), evs=dr, time=_time), 1:13)

popsim = simobs(model0, pop, (par=0.5,), ensemblealg = EnsembleSerial())

estimpop = Subject.(popsim)

# scalar/scalar
model_ss = @model begin
@param begin
  par ∈ RealDomain(lower=0.0, upper=1.0)
  ω ∈ RealDomain(lower=0.000001)
end

@random begin
    η ~ Normal(0,ω)
end

@pre begin
    CL = par+sum(η)
    Vc = 5.0
end

@dynamics Central1

@derived begin
    conc ~ @. Normal(CL/Vc, 0.001)
end
end

res = fit(model_ss, estimpop, (par=0.5, ω=1.0), Pumas.FO())
@test occursin("Variance-covariance matrix could not be", sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(res)))

@test_throws Pumas.PumasFailedCovariance infer(res; rethrow_error=true)

# scalar/one element matrix
model_so = @model begin
@param begin
  par ∈ RealDomain(lower=0.0, upper=1.0)
  Ω ∈ PSDDomain(1)
end

@random begin
    η ~ MvNormal(Ω)
end

@pre begin
    CL = par+sum(η)
    Vc = 5.0
end

@dynamics Central1

@derived begin
    conc ~ @. Normal(CL/Vc, 0.001)
end
end

dr = DosageRegimen(293)

res = fit(model_so, estimpop, (par=0.5, Ω=fill(1.0,1,1)), Pumas.FO())
@test occursin("Variance-covariance matrix could not be", sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(res)))

@test_throws Pumas.PumasFailedCovariance infer(res; rethrow_error=true)


# scalar/matrix param
model_sm = @model begin
@param begin
  par ∈ RealDomain(lower=0.0, upper=1.0)
  Ω ∈ PSDDomain(2)
end

@random begin
    η ~ MvNormal(Ω)
end

@pre begin
    CL = par+sum(η)
    Vc = 5.0
end

@dynamics Central1

@derived begin
    conc ~ @. Normal(CL/Vc, 0.001)
end
end

res = fit(model_sm, estimpop, (par=0.5, Ω=[1.0 0.0; 0.0 1.0]), Pumas.FO())
@test occursin("Variance-covariance matrix could not be", sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(res)))

@test_throws Pumas.PumasFailedCovariance infer(res; rethrow_error=true)

# vector/matrix param
model_vm = @model begin
@param begin
  par ∈ VectorDomain(1, lower=[0.0], upper=[1.0])
  Ω ∈ PSDDomain(2)
end

@random begin
    η ~ MvNormal(Ω)
end

@pre begin
    CL = par[1]+sum(η)
    Vc = 5.0
end

@dynamics Central1

@derived begin
    conc ~ @. Normal(CL/Vc, 0.001)
end
end

res = fit(model_vm, estimpop, (par=[0.5], Ω=[1.0 0.0; 0.0 1.0]), Pumas.FO())
@test occursin("Variance-covariance matrix could not be", sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(res)))

@test_throws Pumas.PumasFailedCovariance infer(res; rethrow_error=true)

end
