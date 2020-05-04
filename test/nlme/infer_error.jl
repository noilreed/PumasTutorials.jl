using Pumas, Random
@testset "vcov error handling in infer" begin
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
model = @model begin
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

_time = [0.1,0.2,0.3,0.4,0.44]
pop = map(i->Subject(id=i, obs=(conc=[],), evs=dr, time=_time), 1:13)

Random.seed!(964)
popsim = simobs(model0, pop, (par=0.5,), ensemblealg = EnsembleSerial())

estimpop = Subject.(popsim)

res = fit(model, estimpop, (par=0.5, Ω=fill(1.0,1,1)), Pumas.FO())
@test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(res)) == 
"""FittedPumasModelInference

Successful minimization:                true

Likelihood approximation:           Pumas.FO
Deviance:                           -831.833
Total number of observation records:      65
Number of active observation records:     65
Number of subjects:                       13

-------------------------------------------
         Estimate       SE      95.0% C.I.
-------------------------------------------
par       0.50077     NaN     [NaN; NaN]
Ω₁,₁      9.4149e-16  NaN     [NaN; NaN]
-------------------------------------------

Variance-covariance matrix could not be be
evaluated. The random effects may be over-
parameterized. Check the coefficients for
variance estimates near zero.
"""

@test_throws Pumas.LinearAlgebra.PosDefException infer(res; rethrow_error=true)


end
