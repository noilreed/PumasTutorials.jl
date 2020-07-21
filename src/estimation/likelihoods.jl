# Some PDMats piracy. Should be possible to remove once we stop using MvNormal
PDMats.unwhiten(C::PDiagMat, x::StridedVector) = sqrt.(C.diag) .* x
PDMats.unwhiten(C::PDiagMat, x::AbstractVector) = sqrt.(C.diag) .* x

abstract type LikelihoodApproximation end
struct NaivePooled <: LikelihoodApproximation end
struct TwoStage <: LikelihoodApproximation end
struct FO <: LikelihoodApproximation end
struct FOI <: LikelihoodApproximation end
struct FOCE <: LikelihoodApproximation end
struct FOCEI <: LikelihoodApproximation end
struct LaplaceI <: LikelihoodApproximation end
struct LLQuad{T} <: LikelihoodApproximation
  quadalg::T
end
LLQuad() = LLQuad(HCubatureJL())

@deprecate Laplace() LaplaceI()

# Some Distributions piracy. In version 0.22, they (accidentally?) removed
# the MvNormal(::AbstractPDMat) constructor which we heavily rely on in Pumas.
# Hence, we need the following definition for the time being
if !hasmethod(MvNormal, Tuple{PDMats.PDMat})
  Distributions.MvNormal(A::AbstractPDMat) = MvNormal(Distributions.Zeros{eltype(A)}(size(A, 1)), A)
end
# This one was necessary in versions of Distributions prior to 0.22
if !hasmethod(MvNormal, Tuple{Diagonal})
  Distributions.MvNormal(D::Diagonal) = MvNormal(PDiagMat(D.diag))
end

zval(d) = zero(eltype(d))
zval(d::Distributions.Normal{T}) where {T} = zero(T)

"""
    _lpdf(d,x)

The log pdf: this differs from `Distributions.logdpf` definintion in a couple
of ways:
- if `d` is a non-distribution it assumes the Dirac distribution.
- if `x` is `NaN` or `Missing`, it returns 0.
- if `d` is a `NamedTuple` of distributions, and `x` is a `NamedTuple` of observations, it computes the sum of the observed variables.
"""
_lpdf(d::Distributions.Sampleable, x::Missing) = zval(d)
_lpdf(d::Distributions.UnivariateDistribution, x::AbstractVector) = sum(t -> _lpdf(d, t), x)
_lpdf(d::Distributions.MultivariateDistribution, x::AbstractVector) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable, x::PDMat) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable, x::Number) = isnan(x) ? zval(d) : logpdf(d,x)
_lpdf(d::Constrained, x) = _lpdf(d.dist, x)
_lpdf(d::Domain, x) = 0.0
function _lpdf(ds::AbstractVector, xs::AbstractVector)
  if length(ds) != length(xs)
    throw(DimensionMismatch("vectors must have same length"))
  end
  l = _lpdf(ds[1], xs[1])
  @inbounds for i in 2:length(ds)
    l += _lpdf(ds[i], xs[i])
  end
  return l
end

Base.@pure function _intersect_names(an::Tuple{Vararg{Symbol}}, bn::Tuple{Vararg{Symbol}})
    names = Symbol[]
    for n in an
        if Base.sym_in(n, bn)
            push!(names, n)
        end
    end
    (names...,)
end

@generated function _lpdf(ds::NamedTuple{Nds}, xs::NamedTuple{Nxs}) where {Nds, Nxs}
  _names = _intersect_names(Nds, Nxs)
  quote
    names = $_names
    l = _lpdf(getindex(ds, names[1]), getindex(xs, names[1]))
    for i in 2:length(names)
      name = names[i]
      l += _lpdf(getindex(ds, name), getindex(xs, name))
    end
    return l
  end
end

"""
    TimeToEvent{T}

Distribution like struct to store the hazard and the cumulative hazard in a
time-to-event model. The dependent variable in a model that uses `TimeToEvent`
should be a censoring variable that is zero if the variable isn't censored and
one if the variable is right censored. Currently, no other censoring types are
supported.

# Example
```
...
@pre begin
  θeff = θ*DOSE
  λ = λ₀*exp(θeff)
end

@dynamics begin
  Λ' = λ
end

@derived begin
  DV ~ @. TimeToEvent(λ, Λ)
end
...
```
"""
struct TimeToEvent{T}
  λ::T
  Λ::T
end
_lpdf(D::TimeToEvent, d::Number) = d*log(D.λ) - D.Λ


# Simulate observations from a time-to-event model, i.e. one with a TimeToEvent
# dependent variable. The idea is to compute `nT` values of survival probability
# from t=minT to maxT and then interpolate with a cubic spline to get a smooth
# survival funtion. Given a survival funtion, we can simulate from the
# distribution by using inverse cdf sampling. Instead of sampling a uniform
# variate to use with the survival probability, we sample an exponential and
# compare to the cumulative hazard which isi equivalent. The Roots package is
# then used for computing the root.
function simobstte(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  randeffs::NamedTuple;
  minT=0.0,
  maxT=nothing,
  nT=10,
  repeated=false,
  kwargs...)

  if maxT === nothing
    throw(ArgumentError("no maxT argument provided"))
  end

  if minT >= maxT
    throw(ArgumentError("maxT must be larger than minT"))
  end

  startT = minT
  _times  = Float64[]
  # We always need to set a starting event to specify where in the integration should start from.
  _events = [Event(0.0, minT, 3, 0)]
  _death  = Float64[]

  if length(keys(subject.observations)) == 1
    dvname = first(keys(subject.observations))
  else
    throw(ArgumentError("simulation of time-to-event models only supported for models with a single dependent variable."))
  end

  # Use a loop to handle repeated time-to-event. This will only run once
  # in the simple time-to-event case.
  while true
    # Create a range on length nT from the current starting value startT until the
    # right censoring time maxT
    _obstimes = range(startT, stop=maxT, length=nT)

    # Simulate the ten samples from the model over the time period (startT, maxT)
    _obs = simobs(model, subject, param, randeffs;
      obstimes=_obstimes, tspan=(startT, maxT), kwargs...)

    # Draw an exponential variate. Notice that for R ~ Uniform(0,1) then
    # F(t) <= R <=> 1 - S(t) <= R <=> exp(-Λ(t)) >= 1 - R <=> Λ(t) <= -log(1 - R)
    # and -log(1 - R) is exponentially distributed.
    # FIXME! take an RNG object
    _r = randexp()

    # Build a cubic cpline based on the nT simulated values of the cumulative hazard
    cs = DataInterpolations.CubicSpline(getfield.(_obs.observations[dvname], :Λ), _obstimes)

    # Find the root Λ(t) == _r
    if _obs.observations[dvname][end].Λ > _r
      tᵢ = Roots.find_zero(t -> cs(t) - _r, (startT, maxT))
      censored = false
    else
      tᵢ = maxT
      censored = true
    end

    # Save either the time of the event or the right censoring time
    push!(_times, tᵢ)

    # Compute the "death" indicator
    push!(_death, Int(!censored))

    # Break out if not repeated time to event and we haven't yet exceeded the right censoring time
    if !repeated || censored
      break
    else
      # If another event is simulated then we reset the "systen" right after the event
      # by registering a reset event in the event system.
      push!(_events, Event(0.0, nextfloat(tᵢ), 3, 0))
      startT = tᵢ
    end
  end

  # FIXME! Which name should we use fo the dv?
  return Subject(
    subject.id,
    NamedTuple{(dvname,)}((_death,)),
    subject.covariates,
    _events,
    _times)
end

simobstte(
  model::PumasModel,
  population::Population,
  param::NamedTuple,
  vrandeffs::Vector{<:NamedTuple};
  maxT=nothing,
  simN::Integer=100,
  repeated=false,
  kwargs...) = [simobstte(model, subject, param, randeffs;
    maxT=maxT, simN=simN, repeated=repeated, kwargs...) for (subject, randeffs) in zip(population, vrandeffs)]

"""
    conditional_nll(m::PumasModel, subject::Subject, param, randeffs; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject`
with parameters `param` and random effects `randeffs`. `kwargs` is
passed to ODE solver. Requires that the derived produces distributions.
"""
@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple;
                                 obstimes=nothing, # you are not allowed to set this, so we catch it
                                 kwargs...)
    dist = _derived(m, subject, param, randeffs; obstimes=subject.time, kwargs...)
    conditional_nll(m, subject, param, randeffs, dist)
end

@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple,
                                 dist::NamedTuple)

  pre = m.pre(param, randeffs, subject)

  if any(d->d isa Nothing, dist)
    return Inf
  end

  clean_dist = NamedTuple{keys(subject.observations)}(dist)
  ll = _lpdf(clean_dist, subject.observations)
  return -ll
end


"""
    penalized_conditional_nll(m::PumasModel, subject::Subject, param::NamedTuple, randeffs::NamedTuple; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as
[`conditional_nll`](@ref), except that it incorporates the penalty from the
prior distribution of the random effects.

Here `param` can be either a `NamedTuple` or a vector (representing a
transformation of the random effects to Cartesian space).
"""
function penalized_conditional_nll(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  randeffs::NamedTuple;
  kwargs...)

  randeffstransform = totransform(m.random(param))
  vrandeffsorth = TransformVariables.inverse(randeffstransform, randeffs)

  return _penalized_conditional_nll(m, subject, param, vrandeffsorth; kwargs...)
end

function _penalized_conditional_nll(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector;
  kwargs...)

  # First evaluate the penalty (wihout the π term)
  nl_randeffs = vrandeffsorth'vrandeffsorth/2

  # If penalty is too large (likelihood would be Inf) then return without evaluating conditional likelihood
  if nl_randeffs > log(floatmax(Float64))
    return nl_randeffs
  else
    randeffstransform = totransform(m.random(param))
    randeffs = TransformVariables.transform(randeffstransform, vrandeffsorth)
    return conditional_nll(m, subject, param, randeffs;kwargs...) + nl_randeffs
  end
end

function _initial_randeffs(m::PumasModel, param::NamedTuple)
  rfxset = m.random(param)
  p = TransformVariables.dimension(totransform(rfxset))

  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(param), numtype(param))
  zeros(T, p)
end

#     _orth_empirical_bayes(model, subject, param, approx, ...)
# The point-estimate the orthogonalized random effects (being the mode of the empirical
# Bayes estimate) of a particular subject at a particular parameter values. The result is
# returned as a vector (transformed into Cartesian space).

function _orth_empirical_bayes(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  approx::LikelihoodApproximation;
  kwargs...)

  initial_vrandefforth = _initial_randeffs(m, param)

  return _orth_empirical_bayes!(initial_vrandefforth, m, subject, param, approx; kwargs...)
end

# This function depends on approx because a Newton's method is
# used for the maximization of the objective and the Hessian
# used in the optimization depends on approx.
function _orth_empirical_bayes!(
  vrandeffsorth::AbstractVector,
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  approx::Union{FO,FOCE,FOCEI,LaplaceI};
  kwargs...)

  function _fgh!(F, G, H, x)
    if G !== nothing || H !== nothing
      nl, ∇nl, ∇²nl = _∂²l∂η²(m, subject, param, x, approx; kwargs...)

      if G !== nothing && ∇nl !== nothing
        G .= ∇nl
      end
      if H !== nothing && ∇²nl !== nothing
        H .= ∇²nl
      end

      if G !== nothing
        G .+= x
      end
      if H !== nothing
        H .= H + I
      end
      return nl + x'x/2
    end

    if F !== nothing
      return _penalized_conditional_nll(m, subject, param, x; kwargs...)
    end
  end

  cost = Optim.NLSolversBase.TwiceDifferentiable(Optim.NLSolversBase.only_fgh!(_fgh!), vrandeffsorth)

  vrandeffsorth .= Optim.minimizer(
    Optim.optimize(
      cost,
      vrandeffsorth,
      NewtonTrustRegion(),
      Optim.Options(
        show_trace=false,
        extended_trace=true,
        g_tol=1e-5
      )))

  return vrandeffsorth
end

# When computing the empirical bayes estimates for FO
# models, we'll use the FOCEI Hessian in the optimization
_orth_empirical_bayes!(
  vrandeffsorth::AbstractVector,
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  ::FO;
  kwargs...) = _orth_empirical_bayes!(
    vrandeffsorth,
    m,
    subject,
    param,
    FOCEI();
    kwargs...)

# For NaivePooled, it's convenient just to let the empty
# vector pass through as a noop
_orth_empirical_bayes!(
  vrandeffsorth::AbstractVector,
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  ::NaivePooled;
  kwargs...) = vrandeffsorth

function _empirical_bayes_dist(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::Union{FO,FOCE,FOCEI,LaplaceI};
  kwargs...)

  parset = m.random(param)
  trf = totransform(parset)

  _, _, __∂²l∂η² = _∂²l∂η²(m, subject, param, vrandeffsorth, approx; kwargs...)

  V = inv(__∂²l∂η² + I)

  i = 1
  tmp = map(trf.transformations) do t
    d = TransformVariables.dimension(t)
    v = view(vrandeffsorth, i:(i + d - 1))
    μ = TransformVariables.transform(t, v)
    Vᵢ = V[i:(i + d - 1), i:(i + d - 1)]
    i += d
    if t isa MvNormalTransform
      if t.d.Σ isa PDiagMat
        U = Diagonal(sqrt.(t.d.Σ.diag))
      else
        U = cholesky(t.d.Σ).U
      end
      return MvNormal(μ, Symmetric(U'*(Vᵢ*U)))
    elseif t isa NormalTransform
      return Normal(μ, std(t.d)*sqrt(Vᵢ[1,1]))
    else
      throw("transformation not currently covered")
    end
  end

  return tmp
end

"""
    marginal_nll(model, subject, param, approx, ...)
    marginal_nll(model, population, param, approx, ...)

Compute the marginal negative loglikelihood of a subject or dataset, using the
integral approximation `approx`.

See also [`deviance`](@ref).
"""
marginal_nll

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      approx::Union{FO,FOCE,FOCEI,LaplaceI,NaivePooled};
                      kwargs...)
  vrandeffsorth = _orth_empirical_bayes(m, subject, param, approx; kwargs...)
  _marginal_nll(m, subject, param, vrandeffsorth, approx; kwargs...)
end

function _marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      ::NaivePooled;
                      kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  # The negative loglikelihood function. There are no random effects.
  if length(m.random(param).params) > 0
    randeffstransform = totransform(m.random(param))
    randeffs = TransformVariables.transform(randeffstransform, zero(vrandeffsorth))
  else
    randeffs = NamedTuple()
  end
  conditional_nll(m, subject, param, randeffs; kwargs...)

end
function _marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::FO;
                      kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  # Compute the gradient of the likelihood and Hessian approxmation in the random effect vector η
  nl, dldη, W  = _∂²l∂η²(m, subject, param, zero(vrandeffsorth), approx; kwargs...)

  if isfinite(nl)
    FIW = cholesky(Symmetric(Matrix(I + W)))
    return nl + (- dldη'*(FIW\dldη) + logdet(FIW))/2
  else # conditional likelihood return Inf
    return typeof(nl)(Inf)
  end
end

function _marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::Union{FOCE,FOCEI,LaplaceI};
                      kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  nl, _, W = _∂²l∂η²(m, subject, param, vrandeffsorth, approx; kwargs...)

  if isfinite(nl)
    # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
    # FIXME. For now we have to convert to matrix to have the check=false version available. Eventually,
    # this should also work with a StaticMatrix
    FIW = cholesky(Symmetric(Matrix(I + W)), check=false)
    if issuccess(FIW)
      return nl + (vrandeffsorth'vrandeffsorth + logdet(FIW))/2
    end
  end
  # conditional likelihood return Inf
  return typeof(nl)(Inf)
end

function _marginal_nll(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::LLQuad,
  # Since the random effect is scaled to be standard normal we can just hardcode the integration domain
  low::AbstractVector=fill(-4.0, length(vrandeffsorth)),
  high::AbstractVector=fill(4.0, length(vrandeffsorth)); batch = 0,
  ireltol = 1e-12, iabstol=1e-12, imaxiters = 100_000,
  kwargs...)

  randeffstransform = totransform(m.random(param))

  integrand = (_vrandeffsorth,nothing) -> exp(
    -conditional_nll(
      m,
      subject,
      param,
      TransformVariables.transform(
        randeffstransform,
        _vrandeffsorth
      );
      kwargs...
    ) - _vrandeffsorth'_vrandeffsorth/2 - log(2π)*length(vrandeffsorth)/2
  )

  intprob = QuadratureProblem(integrand,low,
                                 high,
                                 batch=batch)

  sol = solve(intprob,approx.quadalg,reltol=ireltol,
              abstol=iabstol, maxiters = imaxiters)
  -log(sol.u)
end

marginal_nll(m::PumasModel,
             subject::Subject,
             param::NamedTuple,
             ::LLQuad;
             kwargs...) = _marginal_nll(m, subject, param, _initial_randeffs(m, param), LLQuad(); kwargs...)

# marginall_nll for whole populations
function marginal_nll(
  m::PumasModel,
  # restrict to Vector to avoid distributed arrays taking this path
  population::Vector{<:Subject},
  param::NamedTuple,
  approx::LikelihoodApproximation;
  ensemblealg::DiffEqBase.EnsembleAlgorithm=EnsembleSerial(),
  kwargs...)

  nll1 = marginal_nll(m, population[1], param, approx; kwargs...)
  # Compute first subject separately to determine return type and to return
  # early in case the parameter values cause the likelihood to be Inf. This
  # can e.g. happen if the ODE solver can't solve the ODE for the chosen
  # parameter values.
  if isinf(nll1)
    return nll1
  end

  # The different parallel computations are separated out into functions
  # to make it easier to infer the return types
  if ensemblealg isa EnsembleSerial
    return sum(subject -> marginal_nll(m, subject, param, approx; kwargs...), population)
  elseif ensemblealg isa EnsembleThreads
    return _marginal_nll_threads(nll1, m, population, param, approx; kwargs...)
  elseif ensemblealg isa EnsembleDistributed
    return _marginal_nll_pmap(nll1, m, population, param, approx; kwargs...)
  else
    throw(ArgumentError("Parallelism of type $ensemblealg is not currently implemented for estimation."))
  end
end

function _marginal_nll_threads(nll1::T,
                               m::PumasModel,
                               population::Vector{<:Subject},
                               param::NamedTuple,
                               approx::LikelihoodApproximation;
                               kwargs...)::T where T

  # Allocate array to store likelihood values for each subject in the threaded
  # for loop
  nlls = fill(T(Inf), length(population) - 1)

  # Run threaded for loop for the remaining subjects
  Threads.@threads for i in 2:length(population)
    nlls[i - 1] = marginal_nll(m, population[i], param, approx; kwargs...)
  end

  return nll1 + sum(nlls)
end

function _marginal_nll_pmap(nll1::T,
                            m::PumasModel,
                            population::Vector{<:Subject},
                            param::NamedTuple,
                            approx::LikelihoodApproximation;
                            kwargs...)::T where T

  nlls = convert(
    Vector{T},
    pmap(
      subject -> marginal_nll(
        m,
        subject,
        param,
        approx;
        kwargs...),
      population[2:length(population)]))

    return nll1 + sum(nlls)
end

function _marginal_nll(m::PumasModel,
                      # restrict to Vector to avoid distributed arrays taking
                      # this path
                      population::Vector{<:Subject},
                      param::NamedTuple,
                      vvrandeffsorth::Vector,
                      approx::LikelihoodApproximation;
                      kwargs...)

  return sum(zip(population, vvrandeffsorth)) do (subject, vrandeffsorth)
    _marginal_nll(m, subject, param, vrandeffsorth, approx; kwargs...)
  end
end

# deviance is NONMEM-equivalent marginal negative loglikelihood
StatsBase.deviance(m::PumasModel,
          subject::Subject,
          param::NamedTuple,
          approx::LikelihoodApproximation;
          kwargs...) =
  2*marginal_nll(m, subject, param, approx; kwargs...) - sum(map(dv -> count(!ismissing, dv), subject.observations))*log(2π)

StatsBase.deviance(m::PumasModel,
          data::Population,
          param::NamedTuple,
          approx::LikelihoodApproximation;
          kwargs...) =
  2*marginal_nll(m, data, param, approx; kwargs...) - sum(subject -> sum(dv -> count(!ismissing, dv), subject.observations), data)*log(2π)

_deviance(m::PumasModel,
          subject::Subject,
          param::NamedTuple,
          vrandeffsorth::AbstractVector,
          approx::LikelihoodApproximation;
          kwargs...) =
  2*_marginal_nll(m, subject, param, vrandeffsorth, approx; kwargs...) - sum(map(dv -> count(!ismissing, dv), subject.observations))*log(2π)

_deviance(m::PumasModel,
          data::Population,
          param::NamedTuple,
          vvrandeffsorth::AbstractVector,
          approx::LikelihoodApproximation;
          kwargs...) =
  2*_marginal_nll(m, data, param, vvrandeffsorth, approx; kwargs...) - sum(subject -> sum(dv -> count(!ismissing, dv), subject.observations), data)*log(2π)
# NONMEM doesn't allow ragged, so this suffices for testing

# Compute the gradient of marginal_nll without solving inner optimization
# problem. This functions follows the approach of Almquist et al. (2015) by
# computing the gradient
#
# dℓᵐ/dθ = ∂ℓᵐ/∂θ + dη/dθ'*∂ℓᵐ/∂η
#
# where ℓᵐ is the marginal likelihood of the subject and dη/dθ is the Jacobian
# of the optimal value of η with respect to the population parameters θ. By
# exploiting that η is computed as the optimum in θ, the Jacobian can be
# computed as
#
# dη/dθ = -∂²ℓᵖ∂η² \ ∂²ℓᵖ∂η∂θ
#
# ℓᵖ is the penalized conditional likelihood function.

function _∂ℓᵐ∂θ(
  model::PumasModel,
  subject::Subject,
  vparam::AbstractVector,
  vrandeffsorth::AbstractVector,
  approx::LikelihoodApproximation,
  trf::TransformVariables.TransformTuple;
  kwargs...)

  _cs = max(1, div(8, length(vrandeffsorth)))

  _f_∂ℓᵐ∂θ = _vparam -> _marginal_nll(
    model,
    subject,
    TransformVariables.transform(trf, _vparam),
    vrandeffsorth,
    approx;
    kwargs...
  )
  cs = min(length(vparam), _cs)
  cfg_∂ℓᵐ∂θ = ForwardDiff.GradientConfig(_f_∂ℓᵐ∂θ, vparam, ForwardDiff.Chunk{cs}())
  return ForwardDiff.gradient(_f_∂ℓᵐ∂θ, vparam, cfg_∂ℓᵐ∂θ)
end

function _∂ℓᵐ∂η(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::LikelihoodApproximation;
  kwargs...)

  _cs = max(1, div(8, length(vrandeffsorth)))

  _f_∂ℓᵐ∂η = vηorth -> _marginal_nll(
    model,
    subject,
    param,
    vηorth,
    approx;
    kwargs...
  )
  cs = min(length(vrandeffsorth), _cs)
  cfg_∂ℓᵐ∂η = ForwardDiff.GradientConfig(_f_∂ℓᵐ∂η, vrandeffsorth, ForwardDiff.Chunk{cs}())
  return ForwardDiff.gradient(_f_∂ℓᵐ∂η, vrandeffsorth, cfg_∂ℓᵐ∂η)
end

function _∂²ℓᵖ∂η²(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector;
  kwargs...)

  _f_∂²ℓᵖ∂η² = vηorth -> _penalized_conditional_nll(
    model,
    subject,
    param,
    vηorth;
    kwargs...)

  cs = min(length(vrandeffsorth), 3)
  cfg_∂²ℓᵖ∂η² = ForwardDiff.HessianConfig(_f_∂²ℓᵖ∂η², vrandeffsorth, ForwardDiff.Chunk{cs}())
  return ForwardDiff.hessian(_f_∂²ℓᵖ∂η², vrandeffsorth, cfg_∂²ℓᵖ∂η²)
end

function _∂²ℓᵖ∂η∂θ(
  model::PumasModel,
  subject::Subject,
  vparam::AbstractVector,
  vrandeffsorth::AbstractVector,
  trf::TransformVariables.TransformTuple;
  kwargs...)

  csθ = min(length(vparam), 3)
  csη = min(length(vrandeffsorth), 3)

  _f_∂²ℓᵖ∂η∂θ = _vparam -> begin

    _param = TransformVariables.transform(trf, _vparam)

    _f_∂ℓᵖ∂η = vηorth -> _penalized_conditional_nll(
        model,
        subject,
        _param,
        vηorth;
        kwargs...)

    cfg_∂ℓᵖ∂η = ForwardDiff.GradientConfig(_f_∂ℓᵖ∂η, vrandeffsorth, ForwardDiff.Chunk{csη}())

    ForwardDiff.gradient(
      _f_∂ℓᵖ∂η,
      vrandeffsorth,
      cfg_∂ℓᵖ∂η)
  end

  cfg_∂²ℓᵖ∂η∂θ = ForwardDiff.JacobianConfig(_f_∂²ℓᵖ∂η∂θ, vparam, ForwardDiff.Chunk{csθ}())

  return ForwardDiff.jacobian(_f_∂²ℓᵖ∂η∂θ, vparam, cfg_∂²ℓᵖ∂η∂θ)
end

function _marginal_nll_gradient!(
  g::AbstractVector,
  model::PumasModel,
  subject::Subject,
  vparam::AbstractVector,
  vrandeffsorth::AbstractVector,
  approx::Union{FOCE,FOCEI,LaplaceI},
  trf::TransformVariables.TransformTuple;
  kwargs...)

  param = TransformVariables.transform(trf, vparam)

  ∂ℓᵐ∂θ = _∂ℓᵐ∂θ(model, subject, vparam, vrandeffsorth, approx, trf; kwargs...)

  ∂ℓᵐ∂η = _∂ℓᵐ∂η(model, subject, param, vrandeffsorth, approx; kwargs...)

  ∂²ℓᵖ∂η² = _∂²ℓᵖ∂η²(model, subject, param, vrandeffsorth; kwargs...)

  ∂²ℓᵖ∂η∂θ = _∂²ℓᵖ∂η∂θ(model, subject, vparam, vrandeffsorth, trf; kwargs...)

  dηdθ = -∂²ℓᵖ∂η² \ ∂²ℓᵖ∂η∂θ

  g .= ∂ℓᵐ∂θ .+ dηdθ'*∂ℓᵐ∂η

  return g
end

function _fdrelstep(model::PumasModel, param, reltol, ::Val{:forward})
  if model.prob isa ExplicitModel
    return sqrt(eps(numtype(param)))
  else
    return sqrt(reltol)
  end
end
function _fdrelstep(model::PumasModel, param, reltol, ::Val{:central})
  if model.prob isa ExplicitModel
    return cbrt(eps(numtype(param)))
  else
    return cbrt(reltol)
  end
end

# Similar to the version for FOCE, FOCEI, and LaplaceI
# but much simpler since the expansion point in η is fixed. Hence,
# the gradient is simply the partial derivative in θ
function _marginal_nll_gradient!(
  g::AbstractVector,
  model::PumasModel,
  subject::Subject,
  vparam::AbstractVector,
  vrandeffsorth::AbstractVector,
  approx::Union{NaivePooled,FO,FOI,LLQuad},
  trf::TransformVariables.TransformTuple;
  kwargs...
  )

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  g .= ForwardDiff.gradient(
    _vparam -> _marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _vparam),
      vrandeffsorth,
      approx;
      kwargs...
    ),
    vparam,
  )

  return g
end

# Threaded evaluation of the gradient of the marginal likelihood. This method
# is restricted to Vector{<:Subject} since it relies on scalar indexing into
# the vector of subjects
function _marginal_nll_gradient_threads!(
  g::AbstractVector,
  model::PumasModel,
  population::Vector{<:Subject},
  vparam::AbstractVector,
  vvrandeffsorth::AbstractVector,
  approx::LikelihoodApproximation,
  trf::TransformVariables.TransformTuple;
  kwargs...)

  Gs = similar(g, (length(g), length(population)))

  Threads.@threads for i in eachindex(population)
    subject       = population[i]
    vrandeffsorth = vvrandeffsorth[i]
    _g            = view(Gs, :, i)
    _marginal_nll_gradient!(
      _g,
      model,
      subject,
      copy(vparam), # DiffEqDiffTools is not threadsafe. It mutates the input parameter vector to avoid allocation
      vrandeffsorth,
      approx,
      trf;
      kwargs...
    )
  end

  g .= vec(sum(Gs, dims=2))

  return g
end

# Fallback method which doesn't explicitly rely on scalar indexing into the
# vector of subjects.
function _marginal_nll_gradient!(
  g::AbstractVector,
  model::PumasModel,
  population::Population,
  vparam::AbstractVector,
  vvrandeffsorth::AbstractVector,
  approx::LikelihoodApproximation,
  trf::TransformVariables.TransformTuple;
  kwargs...)

  # Zero the gradient
  fill!(g, 0)

  # FIXME! Avoid this allocation
  _g = similar(g)

  for (subject, vrandeffsorth) in zip(population, vvrandeffsorth)
    _marginal_nll_gradient!(
      _g,
      model,
      subject,
      vparam,
      vrandeffsorth,
      approx,
      trf;
      kwargs...)
    g .+= _g
  end

  return g
end

function _derived_vηorth_gradient(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector;
  obstimes=nothing, # you're not allowed to change this, so we catch it
  kwargs...)
  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  _transform_derived =  vηorth -> begin
    randeffs = TransformVariables.transform(totransform(m.random(param)), vηorth)
    return _derived(m, subject, param, randeffs; obstimes=subject.time, kwargs...)
  end
  # Construct vector of dual numbers for the random effects to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(_transform_derived, vrandeffsorth, ForwardDiff.Chunk{length(vrandeffsorth)}())

  ForwardDiff.seed!(cfg.duals, vrandeffsorth, cfg.seeds)

  return _transform_derived(cfg.duals)
end


function _mean_derived_vηorth_jacobian(m::PumasModel,
                                       subject::Subject,
                                       param::NamedTuple,
                                       vrandeffsorth::AbstractVector;
                                       kwargs...)
  dual_derived = _derived_vηorth_gradient(m ,subject, param, vrandeffsorth; kwargs...)
  # Loop through the distribution vector and extract derivative information
  nt = length(first(dual_derived))
  nrandeffs = length(vrandeffsorth)
  F = map(NamedTuple{keys(subject.observations)}(dual_derived)) do dv
        Ft = zeros(eltype(ForwardDiff.partials(first(dv).μ).values), nrandeffs, nt)
        for j in eachindex(dv)
          partial_values = ForwardDiff.partials(dv[j].μ).values
          for i = 1:nrandeffs
            Ft[i, j] = partial_values[i]
          end
        end
        return Ft'
      end
  return F
end

function _∂²l∂η²(m::PumasModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffsorth::AbstractVector,
                 approx::Union{FO,FOI,FOCE,FOCEI};
                 kwargs...)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable
  # per observation while tracking partial derivatives of the random effects
  _derived_dist = _derived_vηorth_gradient(m, subject, param, vrandeffsorth; kwargs...)

  if any(d->d isa Nothing, _derived_dist)
    return (Inf, nothing, nothing)
  end

  dv_keys = keys(subject.observations)

  dv_dist = NamedTuple{dv_keys}(_derived_dist)
  dv_zip = NamedTuple{dv_keys}(zip(subject.observations, dv_dist))

  ∂²l∂η²s = map(((obs, dv),) -> _∂²l∂η²(obs, dv, approx), dv_zip)

  return map((1, 2, 3)) do i
    sum(∂²l∂η²s[dv_key][i] for dv_key in dv_keys)
  end
end

# Helper function to detect homoscedasticity. For now, it is assumed the input dv vecotr containing
# a vector of distributions with ForwardDiff element types.
_is_homoscedastic(dv::AbstractVector{<:Union{Normal{<:ForwardDiff.Dual},LogNormal{<:ForwardDiff.Dual}}}) =
  iszero(sum(ForwardDiff.partials(last(dv).σ).values))
_is_homoscedastic(dv::AbstractVector{<:Union{Normal{<:AbstractFloat},LogNormal{<:AbstractFloat}}}) =
  first(dv).σ == last(dv).σ
_is_homoscedastic(dv::AbstractVector{<:Gamma{<:ForwardDiff.Dual}}) =
  iszero(sum(ForwardDiff.partials(last(dv).α).values))
_is_homoscedastic(dv::Vector{<:NegativeBinomial{<:ForwardDiff.Dual}}) =
  iszero(sum(ForwardDiff.partials(last(dv).r).values))
_is_homoscedastic(dv::AbstractVector{<:Union{Bernoulli,Binomial,Exponential,Poisson}}) = true
_is_homoscedastic(::Any) = throw(ArgumentError("Distribution not supported"))

_mean(d::Union{Bernoulli,Binomial,Exponential,Gamma,Normal,Poisson}) = mean(d)
_mean(d::LogNormal) = d.μ
_mean(d::NegativeBinomial) = (1 - d.p)/d.p*d.r
_var( d::Union{Bernoulli,Binomial,Exponential,Gamma,Normal,Poisson}) = var(d)
_var( d::LogNormal) = d.σ^2
_var( d::NegativeBinomial) = (1 - d.p)/d.p^2*d.r

_log(::Distribution, obs) = obs
_log(::LogNormal   , obs) = log(obs)

function _∂²l∂η²(obsdv::AbstractVector, dv_d::AbstractVector{<:Union{Normal,LogNormal}}, ::FO)
  # The dimension of the random effect vector
  nrfx = length(ForwardDiff.partials(first(dv_d).μ))

  # Initialize Hessian matrix and gradient vector
  # We evaluate _lpdf for the first non-missing dv to determine the correct
  # element type. For efficiency, we use static vectors for the gradient
  # vector but we use an normal (heap allocated) matrix for the Hessian since
  # very large static arrays can cause exptreme compilition slowdowns.
  i    = findfirst(!ismissing, obsdv)
  nl   = zero(ForwardDiff.value(_lpdf(dv_d[i], obsdv[i])))
  dldη = @SVector fill(nl, nrfx)
  H    = fill(nl, nrfx, nrfx)

  # Loop through the distribution vector and extract derivative information
  for j in eachindex(dv_d)
    obsj = obsdv[j]

    # We ignore missing observations when estimating the model
    if ismissing(obsj)
      continue
    end

    dvj = dv_d[j]
    r = ForwardDiff.value(dvj.σ)^2
    f = SVector(ForwardDiff.partials(dvj.μ).values)
    fdr = f/r

    # H is updated in place to avoid heap allocations in the loop
    H   .+= fdr .* f'
    dldη += fdr*(_log(dvj, obsj) - ForwardDiff.value(dvj.μ))
    nl   -= ForwardDiff.value(_lpdf(dvj, obsj))
  end

  return nl, dldη, H
end

# This version handles the exponential family and LogNormal (through the special _mean
# and _var methods.)
function _∂²l∂η²(obsdv::AbstractVector, dv_d::AbstractVector{<:Distribution}, ::FOCE)

  # FOCE is restricted to models where the dispersion parameter doesn't depend on the random effects
  if !_is_homoscedastic(dv_d)
    throw(ArgumentError("dispersion parameter is not allowed to depend on the random effects when using FOCE"))
  end

  nrfx = length(ForwardDiff.partials(mean(first(dv_d))))

  # Initialize Hessian matrix and gradient vector
  # We evaluate _lpdf for the first non-missing dv to determine the correct
  # element type. For efficiency, we use static vectors for the gradient
  # vector but we use an normal (heap allocated) matrix for the Hessian since
  # very large static arrays can cause exptreme compilition slowdowns.
  i    = findfirst(!ismissing, obsdv)
  nl   = zero(ForwardDiff.value(_lpdf(dv_d[i], obsdv[i])))
  dldη = @SVector fill(nl, nrfx)
  H    = fill(nl, nrfx, nrfx)

  for j in eachindex(dv_d)
    obsj = obsdv[j]
    if ismissing(obsj)
      continue
    end
    dvj   = dv_d[j]
    f     = SVector(ForwardDiff.partials(_mean(dvj)).values)
    fdr   = f/ForwardDiff.value(_var(dvj))
    # H is updated in place to avoid heap allocations in the loop
    H   .+= fdr .* f'
    dldη -= fdr*(_log(dvj, obsj) - ForwardDiff.value(_mean(dvj)))
    nl   -= ForwardDiff.value(_lpdf(dvj, obsj))
  end

  return nl, dldη, H
end

# Categorical
function _∂²l∂η²(obsdv::AbstractVector, dv_d::AbstractVector{<:Categorical}, ::FOCE)

  nrfx = length(ForwardDiff.partials(mean(first(dv_d))))

  # Initialize Hessian matrix and gradient vector
  # We evaluate _lpdf for the first non-missing dv to determine the correct
  # element type. For efficiency, we use static vectors for the gradient
  # vector but we use an normal (heap allocated) matrix for the Hessian since
  # very large static arrays can cause exptreme compilition slowdowns.
  i    = findfirst(!ismissing, obsdv)
  nl   = zero(ForwardDiff.value(_lpdf(dv_d[i], obsdv[i])))
  dldη = @SVector fill(nl, nrfx)
  H    = fill(nl, nrfx, nrfx)

  for j in eachindex(dv_d)
    obsj = obsdv[j]
    if ismissing(obsj)
      continue
    end
    dvj = dv_d[j]
    # Loop through probabilities and add contributions to Hessian
    for (l, pl) in enumerate(probs(dvj))
      f   = SVector(ForwardDiff.partials(pl).values)
      fdp = f/ForwardDiff.value(pl)
      if l == obsj
        dldη -= fdp
      end
      # H is updated in place to avoid heap allocations in the loop
      H .+= fdp*f'
    end
    nl   -= ForwardDiff.value(_lpdf(dvj, obsj))
  end

  return nl, dldη, H
end

function _∂²l∂η²(obsdv::AbstractVector, dv_d::AbstractVector{<:Union{Normal,LogNormal}}, ::FOCEI)
  # Loop through the distribution vector and extract derivative information
  nrfx = length(ForwardDiff.partials(first(dv_d).μ))

  # Initialize Hessian matrix and gradient vector
  # We evaluate _lpdf for the first non-missing dv to determine the correct
  # element type. For efficiency, we use static vectors for the gradient
  # vector but we use an normal (heap allocated) matrix for the Hessian since
  # very large static arrays can cause exptreme compilition slowdowns.
  i    = findfirst(!ismissing, obsdv)
  nl   = zero(ForwardDiff.value(_lpdf(dv_d[i], obsdv[i])))
  dldη = @SVector fill(nl, nrfx)
  H    = fill(nl, nrfx, nrfx)

  for j in eachindex(dv_d)
    obsj = obsdv[j]
    if ismissing(obsj)
      continue
    end
    dvj   = dv_d[j]
    r_inv = inv(ForwardDiff.value(dvj.σ^2))
    f     = SVector(ForwardDiff.partials(dvj.μ).values)
    del_r = SVector(ForwardDiff.partials(dvj.σ.^2).values)
    res   = _log(dvj, obsj) - ForwardDiff.value(dvj.μ)

    # H is updated in place to avoid heap allocations in the loop
    H   .+= f .* r_inv .* f' .+ del_r .* r_inv^2 .* del_r' ./ 2
    dldη -= (-del_r/2 + f*res + res^2*del_r*r_inv/2)*r_inv
    nl   -= ForwardDiff.value(_lpdf(dvj, obsj))
  end

  return nl, dldη, H
end

function _∂²l∂η²(m::PumasModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffsorth::AbstractVector,
                 ::LaplaceI;
                 kwargs...)

  _f_ = vηorth -> conditional_nll(
      m,
      subject,
      param,
      TransformVariables.transform(totransform(m.random(param)), vηorth); kwargs...)

  # Initialize HessianResult for computing Hessian, gradient and value of negative loglikelihood in one go
  T = promote_type(numtype(param), numtype(vrandeffsorth))
  _vrandeffsorth = convert(AbstractVector{T}, vrandeffsorth)
  diffres = DiffResults.HessianResult(_vrandeffsorth)
  cfg = ForwardDiff.HessianConfig(_f_, diffres, _vrandeffsorth, ForwardDiff.Chunk{1}())

  # Compute the derivates
  ForwardDiff.hessian!(diffres, _f_, _vrandeffsorth, cfg)

  # Extract the derivatives
  return DiffResults.value(diffres), DiffResults.gradient(diffres), DiffResults.hessian(diffres)
end

# Fallbacks
## FIXME! Maybe write an optimized version of this one if the scalar dv case ens up being common
_∂²l∂η²(obsdv::AbstractVector, dv_d::Distribution, approx::Union{FOCE,FOCEI}) =
  _∂²l∂η²(obsdv, fill(dv_d, length(obsdv)), approx)
# for a usful error message when distribution isn't supported
_∂²l∂η²(dv_name::Symbol,
        dv_d::Any,
        obsdv,
        m::PumasModel,
        subject::Subject,
        param::NamedTuple,
        vrandeffsorth::AbstractVector,
        approx::LikelihoodApproximation;
        kwargs...) = throw(ArgumentError("Distribution is current not supported for the $approx approximation. Please consider a different likelihood approximation."))
_∂²l∂η²(dv_d::Any,
        obsdv,
        approx::LikelihoodApproximation) = throw(ArgumentError("Distribution is current not supported for the $approx approximation. Please consider a different likelihood approximation."))

# Fitting methods
struct FittedPumasModel{T1<:PumasModel,T2<:Population,T3,T4<:LikelihoodApproximation, T5, T6, T7}
  model::T1
  data::T2
  optim::T3
  approx::T4
  vvrandeffsorth::T5
  kwargs::T6
  fixedparamset::T7
end
simobs(fpm::FittedPumasModel) = simobs(fpm.model, fpm.data, coef(fpm), empirical_bayes(fpm); fpm.kwargs...)

struct DefaultOptimizeFN{A,K}
  alg::A
  kwargs::K
end

DefaultOptimizeFN(alg = nothing; kwargs...) =
  DefaultOptimizeFN(alg, (
    show_trace=true, # Print progress
    store_trace=true,
    extended_trace=false,
    g_tol=1e-3,
    allow_f_increases=true,
    kwargs...))

function (A::DefaultOptimizeFN)(cost, p, callback)

  if A.alg === nothing
    _alg = BFGS(
      linesearch=Optim.LineSearches.BackTracking(),
      # Make sure that step isn't too large by scaling initial Hessian by the norm of the initial gradient
      initial_invH=t -> Matrix(I/norm(Optim.NLSolversBase.gradient(cost)), length(p), length(p))
    )
  else
    _alg = A.alg
  end

  Optim.optimize(
    cost,
    p,
    _alg,
    Optim.Options(;
      callback=callback,
      A.kwargs...
    )
  )
end

"""
    _fixed_to_constant_paramset(paramset::ParamSet, param::NamedTuple, fixed::NamedTuple, omegas::Tuple)

Replace individual parameter Domains in `paramset` with `ConstantTranform` if
the parameter has an entry in `fixed`. Return a new parameter `ParamSet` with
the values in `fixed` in place of the values in input `param`.
"""
function _fixed_to_constant_paramset(paramset, param, fixed, omegas)
  fix_keys = keys(fixed)
  _keys = keys(paramset.params)
  _vals = []
  _par = []
  for key in _keys
    if key ∈ fix_keys
      push!(_vals, ConstDomain(fixed[key]))
      push!(_par, fixed[key])
    elseif key ∈ omegas
      _init = init(paramset)[key]
      if _init isa PDMats.PDiagMat
        _init = Diagonal(zero(_init.diag))
      elseif _init isa PDMats.PDMat
        _init = zero(_init.mat)
      else
        _init = zero(_init)
      end
      dom = ConstDomain(_init)
      push!(_vals, dom)
      push!(_par, _init)
    else
      push!(_vals, paramset.params[key])
      push!(_par, param[key])
    end
  end
  fixedparamset = ParamSet(NamedTuple{_keys}(_vals))
  fixedparam = NamedTuple{_keys}(_par)
  return fixedparamset, fixedparam
end

function _update_ebes_and_evaluate_marginal_nll!(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::Vector,
  vrandeffsorth_tmp::Vector,
  approx::Union{FO,NaivePooled,LLQuad};
  kwargs...
)

  return _marginal_nll(
    m,
    subject,
    param,
    vrandeffsorth_tmp,
    approx;
    kwargs...
  )
end

function _update_ebes_and_evaluate_marginal_nll!(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::Vector,
  vrandeffsorth_tmp::Vector,
  approx::Union{FOCE,FOCEI,LaplaceI};
  kwargs...
)

  copyto!(vrandeffsorth_tmp, vrandeffsorth)
  _orth_empirical_bayes!(
    vrandeffsorth_tmp,
    m,
    subject,
    param,
    approx;
    kwargs...
  )

  return _marginal_nll(
    m,
    subject,
    param,
    vrandeffsorth_tmp,
    approx;
    kwargs...
  )
end

# Threaded update of the EBEs and evaluation of the marginal likelihood.
# We only do this when then Subjects are stored in a Vector since the
# implemention relies on scalar indexing into the vector of subjects
function _update_ebes_and_evaluate_marginal_nll_threads!(
  m::PumasModel,
  population::Vector{<:Subject},
  param::NamedTuple,
  vvrandeffs::Vector,
  vvrandeffs_tmp::Vector,
  approx::LikelihoodApproximation;
  kwargs...)

  # Evaluate the first subject to determine the elementtype of the likelihood values
  # and detect if likelihood value is infinite
  nll1 = _update_ebes_and_evaluate_marginal_nll!(
    m,
    population[1],
    param,
    vvrandeffs[1],
    vvrandeffs_tmp[1],
    approx;
    kwargs...
  )

  # Short circuit if evaluated at extreme parameter values
  if !isfinite(nll1)
    return nll1
  end

  # Allocate array to store all likelihood values
  nlls = fill(oftype(nll1, Inf), length(population))
  nlls[1] = nll1

  # Threaded evaluation of the marginal likelihood (as well as updates of the EBEs)
  Threads.@threads for i in 2:length(population)
    subject       = population[i]
    vrandeffs_tmp = vvrandeffs_tmp[i]
    vrandeffs     = vvrandeffs[i]

    nllsi = _update_ebes_and_evaluate_marginal_nll!(
      m,
      subject,
      param,
      vrandeffs,
      vrandeffs_tmp,
      approx;
      kwargs...
    )

    if isfinite(nllsi)
      nlls[i] = nllsi
    else
      break
    end
  end

  return sum(nlls)
end

# Fallback method that doesn't explictly use scalar indexing of the vector of Subjects so,
# in theory, this should work with e.g. DArrays.
function _update_ebes_and_evaluate_marginal_nll!(
  m::PumasModel,
  population::Population,
  param::NamedTuple,
  vvrandeffs::Vector,
  vvrandeffs_tmp::Vector,
  approx::LikelihoodApproximation;
  kwargs...)

  return sum(zip(population, vvrandeffs, vvrandeffs_tmp)) do (subject, vrandeffs, vrandeffs_tmp)
    _update_ebes_and_evaluate_marginal_nll!(
      m,
      subject,
      param,
      vrandeffs,
      vrandeffs_tmp,
      approx;
      kwargs...
    )
  end
end

function Distributions.fit(m::PumasModel, p::Population, param::NamedTuple; kwargs...)
  throw(ArgumentError("No valid estimation method was provided."))
end
function Distributions.fit(m::PumasModel, p::DataFrame, param::NamedTuple, args...; kwargs...)
  throw(ArgumentError("The second argument to fit was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame."))
end

function _compare_keys(m::PumasModel, param::NamedTuple)
  for modelkey in keys(m.param)
    if modelkey ∉ keys(param)
      throw(ArgumentError("Model parameter $modelkey not found in input parameters."))
    end
  end
  for paramkey in keys(param)
    if paramkey ∉ keys(m.param)
      throw(ArgumentError("Input parameter $paramkey is not present in the model."))
    end
  end
end

function _check_zero_gradient(
  m::PumasModel,
  population::Population,
  vparam::AbstractVector,
  vvrandeffsorth::AbstractVector,
  approx::LikelihoodApproximation,
  fixedtrf;
  ensemblealg=ensemblealg,
  kwargs...)

  g = similar(vparam)

  if ensemblealg isa EnsembleSerial
    _marginal_nll_gradient!(
      g,
      m,
      population,
      vparam,
      vvrandeffsorth,
      approx,
      fixedtrf;
      kwargs...)
  elseif ensemblealg isa EnsembleThreads
    _marginal_nll_gradient_threads!(
      g,
      m,
      population,
      vparam,
      vvrandeffsorth,
      approx,
      fixedtrf;
      kwargs...)
  else
    throw(ArgumentError("$ensemblealg not implemented for this method"))
  end

  for (i, gᵢ) in enumerate(g)
    if iszero(gᵢ)
      j = 0
      for (k, v) in pairs(fixedtrf.transformations)
        d = TransformVariables.dimension(v)
        j += d
        if i <= j
          if d == 1
            throw(ErrorException("gradient of $k is exactly zero. This indicates that $k isn't identified."))
          else
            throw(ErrorException("gradient element $(j-i+1) of $k is exactly zero. This indicates that $k isn't identified."))
          end
        end
      end
    end
  end
end

function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           approx::LikelihoodApproximation;
                           # optimize_fn should take the arguments cost, p, and callback where cost is a
                           # NLSolversBase.OnceDifferentiable, p is a Vector, and cl is Function. Hence,
                           # optimize_fn should evaluate cost with the NLSolversBase.value and
                           # NLSolversBase.gradient interface. The cl callback should be called once per
                           # outer iteration when applicable but it is not required that the optimization
                           # procedure calls cl. In that case, the estimation of the EBEs will always begin
                           # in zero. In addition, the returned object should support a opt_minimizer method
                           # that returns the optimized parameters.
                           optimize_fn = DefaultOptimizeFN(),
                           constantcoef::NamedTuple = NamedTuple(),
                           omegas::Tuple = tuple(),
                           ensemblealg::DiffEqBase.EnsembleAlgorithm = EnsembleSerial(),
                           checkidentification=true,
                           kwargs...)

  _compare_keys(m, param)

  # Compute transform object defining the transformations from NamedTuple to Vector while applying any parameter restrictions and apply the transformations
  fixedparamset, fixedparam = _fixed_to_constant_paramset(m.param, param, constantcoef, omegas)
  fixedtrf = totransform(fixedparamset)
  vparam = TransformVariables.inverse(fixedtrf, fixedparam)

  for (k, v) in pairs(m.random(fixedparam).params)
    if !isa(v, AbstractMvNormal) && !isa(v, Normal)
      throw(ArgumentError("The element $k from the random block does not follow a normal distribution."))
    end
  end

  # We'll store the orthogonalized random effects estimate in vvrandeffsorth which allows us to carry the estimates from last
  # iteration and use them as staring values in the next iteration. We also allocate a buffer to store the
  # random effect estimate during an iteration since it might be modified several times during a line search
  # before the new value has been found. We then define a callback which will store values of vvrandeffsorth_tmp
  # in vvrandeffsorth once the iteration is done.
  if approx isa NaivePooled
    if length(m.random(fixedparam).params) > 0
      vvrandeffsorth     = [zero(_vecmean(m.random(fixedparam))) for subject in population]
    else
      vvrandeffsorth     = [eltype(vparam)[] for subject in population]
    end
    vvrandeffsorth_tmp = [copy(vrandefforths) for vrandefforths in vvrandeffsorth]
    cb(state) = false
  else
    if length(m.random(fixedparam).params) == 0
      throw(ArgumentError("The likelihood approximation method $approx is not appropriate for models without random effects. Please use Pumas.NaivePooled() instead."))
    end
    vvrandeffsorth     = [zero(_vecmean(m.random(fixedparam))) for subject in population]
    vvrandeffsorth_tmp = [copy(vrandefforths) for vrandefforths in vvrandeffsorth]
    cb = state -> begin
      for i in eachindex(vvrandeffsorth)
        copyto!(vvrandeffsorth[i], vvrandeffsorth_tmp[i])
      end
      return false
    end
  end

  # Check identification issue by erroring on zero elements in the gradient
  if checkidentification
    _check_zero_gradient(m, population, vparam, vvrandeffsorth_tmp, approx, fixedtrf; ensemblealg=ensemblealg, kwargs...)
  end

  # Define cost function for the optimization
  cost = Optim.NLSolversBase.OnceDifferentiable(
    Optim.NLSolversBase.only_fg!() do f, g, _vparam
      # The negative loglikelihood function
      # Update the Empirical Bayes Estimates explicitly after each iteration

      # Convert vector to NamedTuple
      _param = TransformVariables.transform(fixedtrf, _vparam)

      # Sum up loglikelihood contributions
      if ensemblealg isa EnsembleSerial
        nll = _update_ebes_and_evaluate_marginal_nll!(
          m,
          population,
          _param,
          vvrandeffsorth,
          vvrandeffsorth_tmp,
          approx;
          kwargs...)

        # Update score
        if g !== nothing
          _marginal_nll_gradient!(
            g,
            m,
            population,
            _vparam,
            vvrandeffsorth_tmp,
            approx,
            fixedtrf;
            kwargs...)
        end

      elseif ensemblealg isa EnsembleThreads
        nll = _update_ebes_and_evaluate_marginal_nll_threads!(
          m,
          population,
          _param,
          vvrandeffsorth,
          vvrandeffsorth_tmp,
          approx;
          kwargs...)

        # Update score
        if g !== nothing
          _marginal_nll_gradient_threads!(
            g,
            m,
            population,
            _vparam,
            vvrandeffsorth_tmp,
            approx,
            fixedtrf;
            kwargs...)
        end
      end

      return nll
    end,

    # The initial values
    vparam
  )

  # Run the optimization
  o = optimize_fn(cost, vparam, cb)

  # Update the random effects after optimization
  for (vrandefforths, subject) in zip(vvrandeffsorth, population)
    _orth_empirical_bayes!(vrandefforths, m, subject, TransformVariables.transform(fixedtrf, opt_minimizer(o)), approx; kwargs...)
  end

  explicit_kwargs = (; optimize_fn=optimize_fn, constantcoef=constantcoef, omegas=omegas, ensemblealg=ensemblealg)
  allkwargs = merge(explicit_kwargs, kwargs)
  return FittedPumasModel(m, population, o, approx, vvrandeffsorth, allkwargs, fixedparamset)
end

function Distributions.fit(m::PumasModel,
                           subject::Subject,
                           param::NamedTuple;
                           kwargs...)
  return fit(m, [subject,], param, NaivePooled(); kwargs...)
end
function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           ::TwoStage;
                           kwargs...)
  return map(x->fit(m, [x,], param, NaivePooled(); checkidentification=false, kwargs...), population)
end

# error handling for fit(model, subject, param; kwargs...)
function Distributions.fit(model::PumasModel, subject::Subject,
             param::NamedTuple, approx::LikelihoodApproximation; kwargs...)
  throw(ArgumentError("Calling fit on a single subject is not allowed with a likelihood approximation method specified."))
end

opt_minimizer(o::Optim.OptimizationResults) = Optim.minimizer(o)

function StatsBase.coef(fpm::FittedPumasModel)
  # we need to use the transform that takes into account that the fixed param
  # are transformed according to the ConstantTransformations, and not the
  # transformations given in totransform(model.param)
  paramset = fpm.fixedparamset
  TransformVariables.transform(totransform(paramset), opt_minimizer(fpm.optim))
end
function Base.getproperty(f::FittedPumasModel{<:Any,<:Any,<:Optim.MultivariateOptimizationResults}, s::Symbol)
  if s === :param
    Base.depwarn("the `fpm.param` property has been deprecated in favor of `coef(fpm)`", :getproperty)
    return coef(f)
  else
    return getfield(f, s)
  end
end

marginal_nll(fpm::FittedPumasModel) = _marginal_nll(
  fpm.model,
  fpm.data,
  coef(fpm),
  fpm.vvrandeffsorth,
  fpm.approx;
  fpm.kwargs...)

"""
    deviance(fpm::FittedPumasModel)

Compute the deviance of a fitted Pumas model:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
StatsBase.deviance(fpm::FittedPumasModel) = _deviance(
  fpm.model,
  fpm.data,
  coef(fpm),
  fpm.vvrandeffsorth,
  fpm.approx;
  fpm.kwargs...)

function _observed_information(f::FittedPumasModel,
                                ::Val{Score};
                               # We explicitly use reltol to compute the right step size for finite difference based gradient
                               # The tolerance has to be stricter when computing the covariance than during estimation
                               reltol=abs2(DEFAULT_ESTIMATION_RELTOL),
                               kwargs...) where Score
  # Transformation the NamedTuple of parameters to a Vector
  # without applying any bounds (identity transform)
  trf = toidentitytransform(f.fixedparamset)
  param = coef(f)
  vparam = TransformVariables.inverse(trf, param)

  fdrelstep_score = _fdrelstep(f.model, vparam, reltol, Val{:central}())
  fdrelstep_hessian = sqrt(_fdrelstep(f.model, vparam, reltol, Val{:central}()))

  # Initialize arrays
  H = zeros(eltype(vparam), length(vparam), length(vparam))
  _H = zeros(eltype(vparam), length(vparam), length(vparam))
  if Score
    S = copy(H)
    g = similar(vparam, length(vparam))
  else
    S = g = nothing
  end

  # Loop through subject and compute Hessian and score contributions
  for i in eachindex(f.data)
    subject = f.data[i]

    _f = function (_j, _vparam)
      _param = TransformVariables.transform(trf, _vparam)

      if f.approx isa NaivePooled
        if length(f.model.random(param).params) > 0
          vrandeffsorth     = zero(_vecmean(f.model.random(param)))
        else
          vrandeffsorth     = []
        end
      else
        vrandeffsorth = _orth_empirical_bayes(f.model, subject, _param, f.approx; kwargs...)
      end

      _marginal_nll_gradient!(
        _j,
        f.model,
        subject,
        _vparam,
        vrandeffsorth,
        f.approx,
        trf;
        reltol=reltol,
        fdtype=Val{:central}(),
        fdrelstep=fdrelstep_hessian,
        fdabsstep=fdrelstep_hessian^2,
        kwargs...)

      return nothing
    end

    # Compute Hessian contribution and update Hessian
    FiniteDiff.finite_difference_jacobian!(
      _H,
      _f,
      vparam,
      Val{:central};
      relstep=fdrelstep_hessian,
      absstep=fdrelstep_hessian^2)

    H .+= _H

    if Score
      # Compute score contribution
      if f.approx isa NaivePooled
        if length(f.model.random(param).params) > 0
          vrandeffsorth     = zero(_vecmean(f.model.random(param)))
        else
          vrandeffsorth     = []
        end
      else
        vrandeffsorth = _orth_empirical_bayes(f.model, subject, coef(f), f.approx; kwargs...)
      end
      _marginal_nll_gradient!(
        g,
        f.model,
        subject,
        vparam,
        vrandeffsorth,
        f.approx,
        trf;
        reltol=reltol,
        fdtype=Val{:central}(),
        fdrelstep=fdrelstep_score,
        fdabsstep=fdrelstep_hessian^2,
        kwargs...)

      # Update outer product of scores
      S .+= g .* g'
    end
  end

  return H, S
end

function _expected_information(m::PumasModel,
                               subject::Subject,
                               param::NamedTuple,
                               vrandeffsorth::AbstractVector,
                               ::FO;
                               kwargs...)

  trf = toidentitytransform(m.param)
  vparam = TransformVariables.inverse(trf, param)

  # Costruct closure for calling _derived as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  __E_and_V = _param -> _E_and_V(m, subject, TransformVariables.transform(trf, _param), vrandeffsorth, FO(); kwargs...)

  # Construct vector of dual numbers for the population parameters to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(__E_and_V, vparam, ForwardDiff.Chunk{length(vparam)}())
  ForwardDiff.seed!(cfg.duals, vparam, cfg.seeds)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  E_d, V_d = __E_and_V(cfg.duals)

  V⁻¹ = inv(cholesky(ForwardDiff.value.(V_d)))
  dEdθ = hcat((collect(ForwardDiff.partials(E_k).values) for E_k in E_d)...)

  m = size(dEdθ, 1)
  n = size(dEdθ, 2)
  dVpart = similar(dEdθ, m, m)
  for l in 1:m
    dVdθl = [ForwardDiff.partials(V_d[i,j]).values[l] for i in 1:n, j in 1:n]
    for k in 1:m
      dVdθk = [ForwardDiff.partials(V_d[i,j]).values[k] for i in 1:n, j in 1:n]
      # dVpart[l,k] = tr(dVdθk * V⁻¹ * dVdθl * V⁻¹)/2
      dVpart[l,k] = sum((V⁻¹ * dVdθk) .* (dVdθl * V⁻¹))/2
    end
  end

  return dEdθ*V⁻¹*dEdθ' + dVpart
end

function _expected_information_fd(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  ::FO;
  blockdiag=true,
  reltol=DEFAULT_ESTIMATION_RELTOL,
  fdtype=Val{:central}(),
  fdrelstep=_fdrelstep(model, param, reltol, fdtype),
  fdabsstep=fdrelstep^2,
  kwargs...)

  trf = toidentitytransform(model.param)
  vparam = TransformVariables.inverse(trf, param)

  # Costruct closure for calling _derived as a function
  # of a random effects vector.
  __E_and_V = _param -> _E_and_V(
    model,
    subject,
    TransformVariables.transform(trf, _param),
    vrandeffsorth,
    FO();
    kwargs...)

  E, V = __E_and_V(vparam)

  ___E = _vparam -> __E_and_V(_vparam)[1]
  ___V = _vparam -> vec(__E_and_V(_vparam)[2])

  dEdθ = FiniteDiff.finite_difference_jacobian(
    ___E,
    vparam,
    typeof(fdtype),
    eltype(vparam),
    relstep=fdrelstep,
    absstep=fdabsstep
  )
  JV = FiniteDiff.finite_difference_jacobian(
    ___V,
    vparam,
    typeof(fdtype),
    eltype(vparam),
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  V⁻¹ = inv(V)
  m = size(dEdθ, 1)
  n = size(dEdθ, 2)

  dVpart = similar(dEdθ, n, n)
  for l in 1:n
    dVdθl = reshape(JV[:, l], m, m)
    for k in 1:n
      dVdθk = reshape(JV[:, k], m, m)
      dVpart[l,k] = sum((V⁻¹ * dVdθk) .* (dVdθl * V⁻¹))/2
    end
  end

  Apart = dEdθ'*V⁻¹*dEdθ
  return Apart + dVpart
end

function StatsBase.informationmatrix(f::FittedPumasModel; expected::Bool=true)
  data          = f.data
  model         = f.model
  param         = coef(f)
  vrandeffsorth = f.vvrandeffsorth
  if expected
    return sum(_expected_information(model, data[i], param, vrandeffsorth[i], f.approx; f.kwargs...) for i in 1:length(data))
  else
    return first(_observed_information(f, Val(false); kwargs...))
  end
end

struct PumasFailedCovariance <: Exception
  err
end
"""
    vcov(f::FittedPumasModel) -> Matrix

Compute the covariance matrix of the population parameters
"""
function StatsBase.vcov(f::FittedPumasModel; rethrow_error=false)
  try
    # Compute the observed information based on the Hessian (H) and the product of the outer scores (S)
    H, S = _observed_information(f, Val(true); f.kwargs...)

    # Use generialized eigenvalue decomposition to compute inv(H)*S*inv(H)
    F = eigen(Symmetric(S), Symmetric(H))
    return F.vectors*Diagonal(F.values)*F.vectors'
  catch err
    err = PumasFailedCovariance(err)
    if rethrow_error
      rethrow(err)
    end
    return err
  end
end

"""
    stderror(f::FittedPumasModel) -> NamedTuple

Compute the standard errors of the population parameters and return
the result as a `NamedTuple` matching the `NamedTuple` of population
parameters.
"""
StatsBase.stderror(f::FittedPumasModel) = stderror(infer(f))

for f in (:mean, :std, :var)
  @eval function Statistics.$(f)(vfpm::Vector{<:FittedPumasModel})
    names = keys(coef(first(vfpm)))
    means = []
    for name in names
      push!(means, ($f)([_coef_value(coef(fpm)[name]) for fpm in vfpm]))
    end
    NamedTuple{names}(means)
  end
end

# This is called with dual numbered "param"
function _E_and_V(model::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffsorth::AbstractVector,
                  ::FO;
                  kwargs...)

  randeffstransform = totransform(model.random(param))
  dist = _derived(
    model,
    subject,
    param,
    TransformVariables.transform(randeffstransform, vrandeffsorth);
    kwargs...)

  _names = keys(subject.observations)

  E = vcat([mean.(dist[_name]) for _name in _names]...)

  F = _mean_derived_vηorth_jacobian(
    model,
    subject,
    param,
    vrandeffsorth;
    kwargs...)

  FF = vcat([F[_name] for _name in _names]...)
  dd = vcat([var.(dist[_name]) for _name in _names]...)
  V = FF*FF' + Diagonal(dd)

  return E, V
end

# empirical_bayes_dist for FittedPumasModel
"""
    empirical_bayes_dist(fpm::FittedPumasModel)

Estimate the distribution of random effects (typically a
Normal approximation of the empirical Bayes posterior)
for the subjects in a `FittedPumasModel`. The result is
returned as a vector of `MvNormal`s.
"""
function empirical_bayes_dist(fpm::FittedPumasModel)
  map(zip(fpm.data, fpm.vvrandeffsorth)) do (subject, vrandeffsorth)
      _empirical_bayes_dist(fpm.model, subject, coef(fpm), vrandeffsorth, fpm.approx; fpm.kwargs...)
  end
end

#########################
# Likelihood ratio test #
#########################
struct LikelihoodRatioTest{T}
  Δdf::Int
  statistic::T
end

"""
    lrtest(fpm_0::FittedPumasModel, fpm_A::FittedPumasModel)::LikelihoodRatioTest

Compute of the likelihood ratio test statistic of the null hypothesis
defined by `fpm_0` against the the alternative hypothesis defined by
`fpm_A`. The `pvalue` function be used for extracting the p-value
based on the asymptotic `Χ²(k)` distribution of the test statistic.
"""
function lrtest(fpm_0::FittedPumasModel, fpm_A::FittedPumasModel)
  df_0 = length(fpm_0.optim.minimizer)
  df_A = length(fpm_A.optim.minimizer)
  statistic = deviance(fpm_0) - deviance(fpm_A)
  return LikelihoodRatioTest(df_A - df_0, statistic)
end

"""
    pvalue(t::LikelihoodRatioTest)::Real

Compute the p-value of the likelihood ratio test `t` based on the
asymptotic `Χ²(k)` distribution of the test statistic.
"""
pvalue(t::LikelihoodRatioTest) = ccdf(Chisq(t.Δdf), t.statistic)

function Base.show(io::IO, ::MIME"text/plain", t::LikelihoodRatioTest)
  _pvalue = pvalue(t)

  println(io, "Statistic: ", lpad(round(t.statistic, sigdigits=3), 15))
  println(io, "Degrees of freedom: ", lpad(t.Δdf, 6))
  print(  io, "P-value: ", lpad(round(_pvalue, digits=3), 17))
end
