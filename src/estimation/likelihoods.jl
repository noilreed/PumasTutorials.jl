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

zval(d) = 0.0
zval(d::Distributions.Normal{T}) where {T} = zero(T)

"""
    _lpdf(d,x)

The log pdf: this differs from `Distributions.logdpf` definintion in a couple of ways:
- if `d` is a non-distribution it assumes the Dirac distribution.
- if `x` is `NaN` or `Missing`, it returns 0.
- if `d` is a `NamedTuple` of distributions, and `x` is a `NamedTuple` of observations, it computes the sum of the observed variables.
"""
_lpdf(d::Number, x::Number) = d == x ? 0.0 : -Inf
_lpdf(d::ConstDomain, x) = _lpdf(d.val, x)
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
    conditional_nll(m::PumasModel, subject::Subject, param, randeffs, args...; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject` with parameters `param` and
random effects `randeffs`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple,
                                 args...;
                                 kwargs...)
    dist = _derived(m, subject, param, randeffs, args...; kwargs...)
    conditional_nll(m, subject, param, randeffs, dist)
end

@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple,
                                 dist::NamedTuple)
  pre = m.pre(param, randeffs, subject)
  # Could we define a `pre` struct that can more easily have its
  # numtype calculated? Say by having a field for prevars, thetas, etas
  collated_numtype = numtype(pre(0.0))
  if any(d->d isa Nothing, dist)
    return collated_numtype(Inf)
  end

  clean_dist = NamedTuple{keys(subject.observations)}(dist)
  ll = _lpdf(clean_dist, subject.observations)::collated_numtype
  return -ll
end

conditional_nll(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                randeffs::NamedTuple,
                approx::Union{FOI,FOCE,FOCEI,LaplaceI},
                args...; kwargs...) = conditional_nll(m, subject, param, randeffs, args...; kwargs...)

function conditional_nll(m::PumasModel,
                         subject::Subject,
                         param::NamedTuple,
                         randeffs::NamedTuple,
                         approx::FO,
                         args...; kwargs...)

  collated_numtype = numtype(m.pre(param, randeffs, subject))
  dist = _derived(m, subject, param, randeffs, args...;kwargs...)

  if any(d->d isa Nothing, dist)
    return collated_numtype(Inf)
  end

  return conditional_nll(m, subject, param, randeffs, dist)::collated_numtype
end

"""
    penalized_conditional_nll(m::PumasModel, subject::Subject, param::NamedTuple, randeffs::NamedTuple, args...; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as
[`conditional_nll`](@ref), except that it incorporates the penalty from the prior
distribution of the random effects.

Here `param` can be either a `NamedTuple` or a vector (representing a transformation of the
random effects to Cartesian space).
"""
function penalized_conditional_nll(m::PumasModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   randeffs::NamedTuple,
                                   args...;kwargs...)

  randeffstransform = totransform(m.random(param))
  vrandeffsorth = TransformVariables.inverse(randeffstransform, randeffs)

  return penalized_conditional_nll(m, subject, param, vrandeffsorth, args...; kwargs...)
end

function penalized_conditional_nll(m::PumasModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   vrandeffsorth::AbstractVector,
                                   args...;kwargs...)

  # First evaluate the penalty (wihout the π term)
  nl_randeffs = vrandeffsorth'vrandeffsorth/2

  # If penalty is too large (likelihood would be Inf) then return without evaluating conditional likelihood
  if nl_randeffs > log(floatmax(Float64))
    return nl_randeffs
  else
    randeffstransform = totransform(m.random(param))
    randeffs = TransformVariables.transform(randeffstransform, vrandeffsorth)
    return conditional_nll(m, subject, param, randeffs, args...;kwargs...) + nl_randeffs
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
  approx::LikelihoodApproximation,
  args...; kwargs...)

  initial_vrandefforth = _initial_randeffs(m, param)

  return _orth_empirical_bayes!(initial_vrandefforth, m, subject, param, approx, args...; kwargs...)
end

function _orth_empirical_bayes!(
  vrandeffsorth::AbstractVector,
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  ::Union{FO,FOI,LLQuad},
  args...; kwargs...)

  fill!(vrandeffsorth, 0)
  return vrandeffsorth
end

function _orth_empirical_bayes!(
  vrandeffsorth::AbstractVector,
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  approx::Union{FOCE,FOCEI,LaplaceI},
  args...;
  kwargs...)

  function _fgh!(F, G, H, x)
    if G !== nothing || H !== nothing
      nl, ∇nl, ∇²nl = ∂²l∂η²(m, subject, param, x, approx, args...; kwargs...)

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
      return penalized_conditional_nll(m, subject, param, x, approx, args...; kwargs...)
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


"""
    empirical_bayes_dist(model, subject, param, vrandeffsorth::AbstractVector, approx, ...)
Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a `MvNormal`.
"""
empirical_bayes_dist

function empirical_bayes_dist(m::PumasModel,
                              subject::Subject,
                              param::NamedTuple,
                              vrandeffsorth::AbstractVector,
                              approx::Union{FOCE,FOCEI,LaplaceI},
                              args...; kwargs...)

  parset = m.random(param)
  trf = totransform(parset)

  _, _, _∂²l∂η² = ∂²l∂η²(m, subject, param, vrandeffsorth, approx, args...; kwargs...)
  V = inv(_∂²l∂η² + I)

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
        U = cholesky(t.d.Σ)
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
    marginal_nll(model, subject, param[, param], approx, ...)
    marginal_nll(model, population, param, approx, ...)

Compute the marginal negative loglikelihood of a subject or dataset, using the integral
approximation `approx`. If no random effect (`param`) is provided, then this is estimated
from the data.

See also [`deviance`](@ref).
"""
function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      randeffs::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...; kwargs...)

  if approx isa NaivePooled
    vrandeffsorth = []
  else
    rfxset = m.random(param)
    vrandeffsorth = TransformVariables.inverse(totransform(rfxset), randeffs)
  end
  return marginal_nll(m, subject, param, vrandeffsorth, approx, args...; kwargs...)
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...;
                      kwargs...)
  vrandeffsorth = _orth_empirical_bayes(m, subject, param, approx, args...; kwargs...)
  marginal_nll(m, subject, param, vrandeffsorth, approx, args...; kwargs...)
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      ::NaivePooled,
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  # The negative loglikelihood function. There are no random effects.
  conditional_nll(m, subject, param, NamedTuple(), args...;kwargs...)

end
function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::FO,
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  # For FO, the conditional likelihood must be evaluated at η=0
  @assert iszero(vrandeffsorth)

  # Compute the gradient of the likelihood and Hessian approxmation in the random effect vector η
  nl, dldη, W  = ∂²l∂η²(m, subject, param, vrandeffsorth, approx, args...; kwargs...)

  if isfinite(nl)
    FIW = cholesky(Symmetric(Matrix(I + W)))
    return nl + (- dldη'*(FIW\dldη) + logdet(FIW))/2
  else # conditional likelihood return Inf
    return typeof(nl)(Inf)
  end
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::Union{FOCE,FOCEI,LaplaceI},
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  nl, _, W = ∂²l∂η²(m, subject, param, vrandeffsorth, approx, args...; kwargs...)

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

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::LLQuad,
                      # Since the random effect is scaled to be standard normal we can just hardcode the integration domain
                      low::AbstractVector=fill(-4.0, length(vrandeffsorth)),
                      high::AbstractVector=fill(4.0, length(vrandeffsorth)),
                      args...; batch = 0,
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
      ),
      args...;
      kwargs...
    ) - _vrandeffsorth'_vrandeffsorth/2 - log(2π)*length(vrandeffsorth)/2
  )

  intprob = QuadratureProblem(integrand,low,
                                 high,
                                 batch=batch)

  sol = solve(intprob,approx.quadalg,reltol=ireltol,
              abstol=iabstol,maxiters = imaxiters)
  -log(sol.u)
end

# marginall_nll for whole populations
function marginal_nll(m::PumasModel,
                      # restrict to Vector to avoid distributed arrays taking
                      # this path
                      population::Vector{<:Subject},
                      args...;
                      ensemblealg::DiffEqBase.EnsembleAlgorithm=EnsembleSerial(),
                      kwargs...)

  nll1 = marginal_nll(m, population[1], args...; kwargs...)
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
    return sum(subject -> marginal_nll(m, subject, args...; kwargs...), population)
  elseif ensemblealg isa EnsembleThreads
    return _marginal_nll_threads(nll1, m, population, args...; kwargs...)
  elseif ensemblealg isa EnsembleDistributed
    return _marginal_nll_pmap(nll1, m, population, args...; kwargs...)
  else
    throw(ArgumentError("Parallelism of type $ensemblealg is not currently implemented for estimation."))
  end
end

function _marginal_nll_threads(nll1::T,
                               m::PumasModel,
                               population::Vector{<:Subject},
                               args...;
                               kwargs...)::T where T

  # Allocate array to store likelihood values for each subject in the threaded
  # for loop
  nlls = fill(T(Inf), length(population) - 1)

  # Run threaded for loop for the remaining subjects
  Threads.@threads for i in 2:length(population)
    nlls[i - 1] = marginal_nll(m, population[i], args...; kwargs...)
  end

  return nll1 + sum(nlls)
end

function _marginal_nll_pmap(nll1::T,
                            m::PumasModel,
                            population::Vector{<:Subject},
                            args...;
                            kwargs...)::T where T

  nlls = convert(Vector{T},
                 pmap(subject -> marginal_nll(m, subject, args...; kwargs...),
                 population[2:length(population)]))

    return nll1 + sum(nlls)
end

function marginal_nll(m::PumasModel,
                      # restrict to Vector to avoid distributed arrays taking
                      # this path
                      population::Vector{<:Subject},
                      param::NamedTuple,
                      vvrandeffsorth::Vector,
                      args...; kwargs...)

  return sum(zip(population, vvrandeffsorth)) do (subject, vrandeffsorth)
    marginal_nll(m, subject, param, vrandeffsorth, args...; kwargs...)
  end
end

# deviance is NONMEM-equivalent marginal negative loglikelihood
"""
    deviance(model, subject, param[, param], approx, ...)
    deviance(model, data, param, approx, ...)

Compute the deviance of a subject or dataset:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
StatsBase.deviance(m::PumasModel,
                   subject::Subject,
                   args...; kwargs...) =
    2marginal_nll(m, subject, args...; kwargs...) - sum(map(dv -> count(!ismissing, dv), subject.observations))*log(2π)

StatsBase.deviance(m::PumasModel,
                   data::Population,
                   args...; kwargs...) =
    2marginal_nll(m, data, args...; kwargs...) - sum(subject -> sum(dv -> count(!ismissing, dv), subject.observations), data)*log(2π)
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
function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                subject::Subject,
                                vparam::AbstractVector,
                                vrandeffsorth::AbstractVector,
                                approx::Union{FOCE,FOCEI,LaplaceI},
                                trf::TransformVariables.TransformTuple,
                                args...; kwargs...)

  param = TransformVariables.transform(trf, vparam)

  _cs = max(1, div(8, length(vrandeffsorth)))

  _f_∂ℓᵐ∂θ = _vparam -> marginal_nll(
    model,
    subject,
    TransformVariables.transform(trf, _vparam),
    vrandeffsorth,
    approx,
    args...; kwargs...
  )
  cs = min(length(vparam), _cs)
  cfg_∂ℓᵐ∂θ = ForwardDiff.GradientConfig(_f_∂ℓᵐ∂θ, vparam, ForwardDiff.Chunk{cs}())
  ∂ℓᵐ∂θ = ForwardDiff.gradient(_f_∂ℓᵐ∂θ, vparam, cfg_∂ℓᵐ∂θ)

  _f_∂ℓᵐ∂η = vηorth -> marginal_nll(
    model,
    subject,
    param,
    vηorth,
    approx,
    args...; kwargs...
  )
  cs = min(length(vrandeffsorth), _cs)
  cfg_∂ℓᵐ∂η = ForwardDiff.GradientConfig(_f_∂ℓᵐ∂η, vrandeffsorth, ForwardDiff.Chunk{cs}())
  ∂ℓᵐ∂η = ForwardDiff.gradient(_f_∂ℓᵐ∂η, vrandeffsorth, cfg_∂ℓᵐ∂η)

  _f_∂²ℓᵖ∂η² = vηorth -> penalized_conditional_nll(
    model,
    subject,
    param,
    vηorth,
    approx,
    args...; kwargs...)
  cs = min(length(vrandeffsorth), 3)
  cfg_∂²ℓᵖ∂η² = ForwardDiff.HessianConfig(_f_∂²ℓᵖ∂η², vrandeffsorth, ForwardDiff.Chunk{cs}())
  ∂²ℓᵖ∂η² = ForwardDiff.hessian(_f_∂²ℓᵖ∂η², vrandeffsorth, cfg_∂²ℓᵖ∂η²)

  _f_∂²ℓᵖ∂η∂θ = _vparam -> begin
    _param = TransformVariables.transform(trf, _vparam)
    ForwardDiff.gradient(
      vηorth -> penalized_conditional_nll(
        model,
        subject,
        _param,
        vηorth,
        approx,
        args...; kwargs...),
      vrandeffsorth
    )
  end
  cfg_∂²ℓᵖ∂η∂θ = ForwardDiff.JacobianConfig(_f_∂²ℓᵖ∂η∂θ, vparam, ForwardDiff.Chunk{1}())
  ∂²ℓᵖ∂η∂θ = ForwardDiff.jacobian(_f_∂²ℓᵖ∂η∂θ, vparam, cfg_∂²ℓᵖ∂η∂θ)

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
function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                subject::Subject,
                                vparam::AbstractVector,
                                vrandeffsorth::AbstractVector,
                                approx::Union{NaivePooled,FO,FOI,LLQuad},
                                trf::TransformVariables.TransformTuple,
                                args...; kwargs...
                                )

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  g .= FiniteDiff.finite_difference_gradient(
    _vparam -> marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _vparam),
      vrandeffsorth,
      approx,
      args...; kwargs...
    ),
    vparam,
  )

  return g
end

# Threaded evaluation of the gradient of the marginal likelihood. This method
# is restricted to Vector{<:Subject} since it relies on scalar indexing into
# the vector of subjects
function marginal_nll_gradient_threads!(g::AbstractVector,
                                        model::PumasModel,
                                        population::Vector{<:Subject},
                                        vparam::AbstractVector,
                                        vvrandeffsorth::AbstractVector,
                                        approx::LikelihoodApproximation,
                                        trf::TransformVariables.TransformTuple,
                                        args...; kwargs...
                                        )

  Gs = similar(g, (length(g), length(population)))

  Threads.@threads for i in eachindex(population)
    subject       = population[i]
    vrandeffsorth = vvrandeffsorth[i]
    _g            = view(Gs, :, i)
    marginal_nll_gradient!(
      _g,
      model,
      subject,
      copy(vparam), # DiffEqDiffTools is not threadsafe. It mutates the input parameter vector to avoid allocation
      vrandeffsorth,
      approx,
      trf,
      args...;
      kwargs...
    )
  end

  g .= vec(sum(Gs, dims=2))

  return g
end

# Fallback method which doesn't explicitly rely on scalar indexing into the
# vector of subjects.
function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                population::Population,
                                vparam::AbstractVector,
                                vvrandeffsorth::AbstractVector,
                                approx::LikelihoodApproximation,
                                trf::TransformVariables.TransformTuple,
                                args...; kwargs...
                                )

  # Zero the gradient
  fill!(g, 0)

  # FIXME! Avoid this allocation
  _g = similar(g)

  for (subject, vrandeffsorth) in zip(population, vvrandeffsorth)
    marginal_nll_gradient!(
      _g,
      model,
      subject,
      vparam,
      vrandeffsorth,
      approx,
      trf,
      args...; kwargs...)
    g .+= _g
  end

  return g
end

function _derived_vηorth_gradient(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  args...; kwargs...)
  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  _transform_derived =  vηorth -> begin
    randeffs = TransformVariables.transform(totransform(m.random(param)), vηorth)
    return _derived(m, subject, param, randeffs, args...; kwargs...)
  end
  # Construct vector of dual numbers for the random effects to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(_transform_derived, vrandeffsorth, ForwardDiff.Chunk{length(vrandeffsorth)}())

  ForwardDiff.seed!(cfg.duals, vrandeffsorth, cfg.seeds)

  return _transform_derived(cfg.duals)
end


function _mean_derived_vηorth_jacobian(m::PumasModel,
                                       subject::Subject,
                                       param::NamedTuple,
                                       vrandeffsorth::AbstractVector,
                                       args...; kwargs...)
  dual_derived = _derived_vηorth_gradient(m ,subject, param, vrandeffsorth, args...; kwargs...)
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

function ∂²l∂η²(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffsorth::AbstractVector,
                approx::Union{FO,FOI,FOCE,FOCEI},
                args...; kwargs...)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable
  # per observation while tracking partial derivatives of the random effects
  _derived_dist = _derived_vηorth_gradient(m, subject, param, vrandeffsorth, args...; kwargs...)

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

function _∂²l∂η²(obsdv::AbstractVector, dv::AbstractVector{<:Union{Normal,LogNormal}}, ::FO)
  # The dimension of the random effect vector
  nrfx = length(ForwardDiff.partials(first(dv).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)
  nl   = 0.0

  # Loop through the distribution vector and extract derivative information
  for j in eachindex(dv)
    obsj = obsdv[j]

    # We ignore missing observations when estimating the model
    if ismissing(obsj)
      continue
    end

    dvj = dv[j]
    r = ForwardDiff.value(dvj.σ)^2
    f = SVector(ForwardDiff.partials(dvj.μ).values)
    fdr = f/r

    H    += fdr*f'
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
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)
  nl   = 0.0

  for j in eachindex(dv_d)
    obsj = obsdv[j]
    if ismissing(obsj)
      continue
    end
    dvj   = dv_d[j]
    f     = SVector(ForwardDiff.partials(_mean(dvj)).values)
    fdr   = f/ForwardDiff.value(_var(dvj))
    H    += fdr*f'
    dldη -= fdr*(_log(dvj, obsj) - ForwardDiff.value(_mean(dvj)))
    nl   -= ForwardDiff.value(_lpdf(dvj, obsj))
  end

  return nl, dldη, H
end

# Categorical
function _∂²l∂η²(obsdv::AbstractVector, dv_d::AbstractVector{<:Categorical}, ::FOCE)

  nrfx = length(ForwardDiff.partials(mean(first(dv_d))))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)
  nl   = 0.0

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
      H += fdp*f'
    end
    nl   -= ForwardDiff.value(_lpdf(dvj, obsj))
  end

  return nl, dldη, H
end

function _∂²l∂η²(obsdv::AbstractVector, dv::AbstractVector{<:Union{Normal,LogNormal}}, ::FOCEI)
  # Loop through the distribution vector and extract derivative information
  nrfx = length(ForwardDiff.partials(first(dv).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)
  nl   = 0.0

  for j in eachindex(dv)
    obsj = obsdv[j]
    if ismissing(obsj)
      continue
    end
    dvj   = dv[j]
    r_inv = inv(ForwardDiff.value(dvj.σ^2))
    f     = SVector(ForwardDiff.partials(dvj.μ).values)
    del_r = SVector(ForwardDiff.partials(dvj.σ.^2).values)
    res   = _log(dvj, obsj) - ForwardDiff.value(dvj.μ)

    H    += f*r_inv*f' + (r_inv*del_r*r_inv*del_r')/2
    dldη -= (-del_r/2 + f*res + res^2*del_r*r_inv/2)*r_inv
    nl   -= ForwardDiff.value(_lpdf(dvj, obsj))
  end

  return nl, dldη, H
end

function ∂²l∂η²(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffsorth::AbstractVector,
                approx::LaplaceI,
                args...; kwargs...)

  _f_ = vηorth -> conditional_nll(
      m,
      subject,
      param,
      TransformVariables.transform(totransform(m.random(param)), vηorth),
      approx,
      args...; kwargs...)

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
        approx::LikelihoodApproximation,
        args...; kwargs...) = throw(ArgumentError("Distribution is current not supported for the $approx approximation. Please consider a different likelihood approximation."))
_∂²l∂η²(dv_d::Any,
        obsdv,
        approx::LikelihoodApproximation) = throw(ArgumentError("Distribution is current not supported for the $approx approximation. Please consider a different likelihood approximation."))

# Fitting methods
struct FittedPumasModel{T1<:PumasModel,T2<:Population,T3,T4<:LikelihoodApproximation, T5, T6, T7, T8}
  model::T1
  data::T2
  optim::T3
  approx::T4
  vvrandeffsorth::T5
  args::T6
  kwargs::T7
  fixedtrf::T8
end
simobs(fpm::FittedPumasModel) = simobs(fpm.model, fpm.data, coef(fpm), empirical_bayes(fpm); fpm.kwargs...)

struct DefaultOptimizeFN{A,K}
  alg::A
  kwargs::K
end

DefaultOptimizeFN(alg = nothing;kwargs...) =
                  DefaultOptimizeFN(alg,(show_trace=false, # Print progress
                                                store_trace=true,
                                                extended_trace=true,
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
    _fixed_to_constanttransform(trf::TransformTuple, param::NamedTuple, fixed::NamedTuple)

Replace individual parameter transformations in `trf` with `ConstantTranform` if
the parameter has an entry in `fixed`. Return a new parameter `NamedTuple` with
the values in `fixed` in place of the values in input `param`.
"""
function _fixed_to_constanttransform(trf, param, fixed)
  fix_keys = keys(fixed)
  _keys = keys(trf.transformations)
  _vals = []
  _paramval = []
  for key in _keys
    if key ∈ fix_keys
      push!(_vals, ConstantTransform(fixed[key]))
      push!(_paramval, fixed[key])
    else
      push!(_vals, trf.transformations[key])
      push!(_paramval, param[key])
    end
  end
  new_param = NamedTuple{_keys}(_paramval)
  return new_param, TransformVariables.TransformTuple(NamedTuple{_keys}(_vals))
end

function _update_ebes_and_evaluate_marginal_nll!(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::Vector,
  vrandeffsorth_tmp::Vector,
  approx::LikelihoodApproximation,
  args...;
  kwargs...
)

  # If not FO or NaivePooled then compute EBE based on the estimates from last
  # iteration stored in vvrandeffs and store the retult in vvrandeffs_tmp
  if !(approx isa FO || approx isa NaivePooled)
    copyto!(vrandeffsorth_tmp, vrandeffsorth)
    _orth_empirical_bayes!(
      vrandeffsorth_tmp,
      m,
      subject,
      param,
      approx,
      args...;
      kwargs...
    )
  end

  return marginal_nll(
    m,
    subject,
    param,
    vrandeffsorth_tmp,
    approx,
    args...;
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
  approx::LikelihoodApproximation,
  args...;
  kwargs...)

  # Evaluate the first subject to determine the elementtype of the likelihood values
  # and detect if likelihood value is infinite
  nll1 = _update_ebes_and_evaluate_marginal_nll!(
    m,
    population[1],
    param,
    vvrandeffs[1],
    vvrandeffs_tmp[1],
    approx,
    args...;
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
      approx,
      args...;
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
  approx::LikelihoodApproximation,
  args...;
  kwargs...)

  return sum(zip(population, vvrandeffs, vvrandeffs_tmp)) do (subject, vrandeffs, vrandeffs_tmp)
    _update_ebes_and_evaluate_marginal_nll!(
      m,
      subject,
      param,
      vrandeffs,
      vrandeffs_tmp,
      approx,
      args...;
      kwargs...
    )
  end
end

function Distributions.fit(m::PumasModel, p::Population, param::NamedTuple; kwargs...)
  throw(ArgumentError("No valid estimation method was provided."))
end

function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           approx::LikelihoodApproximation,
                           args...;
                           # optimize_fn should take the arguments cost, p, and callback where cost is a
                           # NLSolversBase.OnceDifferentiable, p is a Vector, and cl is Function. Hence,
                           # optimize_fn should evaluate cost with the NLSolversBase.value and
                           # NLSolversBase.gradient interface. The cl callback should be called once per
                           # outer iteration when applicable but it is not required that the optimization
                           # procedure calls cl. In that case, the estimation of the EBEs will always begin
                           # in zero. In addition, the returned object should support a opt_minimizer method
                           # that returns the optimized parameters.
                           optimize_fn = DefaultOptimizeFN(),
                           constantcoef = NamedTuple(),
                           ensemblealg::DiffEqBase.EnsembleAlgorithm = EnsembleSerial(),
                           kwargs...)

  # Compute transform object defining the transformations from NamedTuple to Vector while applying any parameter restrictions and apply the transformations
  trf = totransform(m.param)
  fixedtrf=trf
  param, fixedtrf = _fixed_to_constanttransform(trf, param, constantcoef)
  vparam = TransformVariables.inverse(fixedtrf, param)

  # We'll store the orthogonalized random effects estimate in vvrandeffsorth which allows us to carry the estimates from last
  # iteration and use them as staring values in the next iteration. We also allocate a buffer to store the
  # random effect estimate during an iteration since it might be modified several times during a line search
  # before the new value has been found. We then define a callback which will store values of vvrandeffsorth_tmp
  # in vvrandeffsorth once the iteration is done.
  if approx isa NaivePooled
    vvrandeffsorth     = [[] for subject in population]
    vvrandeffsorth_tmp = [copy(vrandefforths) for vrandefforths in vvrandeffsorth]
    cb(state) = false
  else
    vvrandeffsorth     = [zero(_vecmean(m.random(param))) for subject in population]
    vvrandeffsorth_tmp = [copy(vrandefforths) for vrandefforths in vvrandeffsorth]
    cb = state -> begin
      for i in eachindex(vvrandeffsorth)
        copyto!(vvrandeffsorth[i], vvrandeffsorth_tmp[i])
      end
      return false
    end
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
          approx,
          args...;
          kwargs...)

        # Update score
        if g !== nothing
          marginal_nll_gradient!(
            g,
            m,
            population,
            _vparam,
            vvrandeffsorth_tmp,
            approx,
            fixedtrf,
            args...; kwargs...)
        end

      elseif ensemblealg isa EnsembleThreads
        nll = _update_ebes_and_evaluate_marginal_nll_threads!(
          m,
          population,
          _param,
          vvrandeffsorth,
          vvrandeffsorth_tmp,
          approx,
          args...;
          kwargs...)

        # Update score
        if g !== nothing
          marginal_nll_gradient_threads!(
            g,
            m,
            population,
            _vparam,
            vvrandeffsorth_tmp,
            approx,
            fixedtrf,
            args...; kwargs...)
        end

      else
        throw(ArgumentError("$ensemblealg not implemented for this method"))
      end

      return nll
    end,

    # The initial values
    vparam
  )

  # Run the optimization
  o = optimize_fn(cost, vparam, cb)

  # Update the random effects after optimization
  if !(approx isa FO || approx isa NaivePooled)
    for (vrandefforths, subject) in zip(vvrandeffsorth, population)
      _orth_empirical_bayes!(vrandefforths, m, subject, TransformVariables.transform(fixedtrf, opt_minimizer(o)), approx, args...; kwargs...)
    end
  end

  return FittedPumasModel(m, population, o, approx, vvrandeffsorth, args, kwargs, fixedtrf)
end

function Distributions.fit(m::PumasModel,
                           subject::Subject,
                           param::NamedTuple,
                           args...;
                           kwargs...)
  return fit(m, [subject,], param, NaivePooled(), args...; kwargs...)
end
function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           ::TwoStage,
                           args...;
                           kwargs...)
  return map(x->fit(m, [x,], param, NaivePooled(), args...; kwargs...), population)
end

# error handling for fit(model, subject, param, args...; kwargs...)
function Distributions.fit(model::PumasModel, subject::Subject,
             param::NamedTuple, approx::LikelihoodApproximation, args...; kwargs...)
  throw(ArgumentError("Calling fit on a single subject is not allowed with a likelihood approximation method specified."))
end

opt_minimizer(o::Optim.OptimizationResults) = Optim.minimizer(o)

function StatsBase.coef(fpm::FittedPumasModel)
  # we need to use the transform that takes into account that the fixed param
  # are transformed according to the ConstantTransformations, and not the
  # transformations given in totransform(model.param)
  trf = fpm.fixedtrf
  TransformVariables.transform(trf, opt_minimizer(fpm.optim))
end
function Base.getproperty(f::FittedPumasModel{<:Any,<:Any,<:Optim.MultivariateOptimizationResults}, s::Symbol)
  if s === :param
    # deprecate?
    coef(f)
  else
    return getfield(f, s)
  end
end

marginal_nll(fpm::FittedPumasModel) = marginal_nll(
  fpm.model,
  fpm.data,
  coef(fpm),
  fpm.vvrandeffsorth,
  fpm.approx,
  fpm.args...; fpm.kwargs...)
StatsBase.deviance(fpm::FittedPumasModel) = deviance(
  fpm.model,
  fpm.data,
  coef(fpm),
  fpm.vvrandeffsorth,
  fpm.approx,
  fpm.args...; fpm.kwargs...)

function _observed_information(f::FittedPumasModel,
                                ::Val{Score},
                               args...;
                               # We explicitly use reltol to compute the right step size for finite difference based gradient
                               # The tolerance has to be stricter when computing the covariance than during estimation
                               reltol=abs2(DEFAULT_ESTIMATION_RELTOL),
                               kwargs...) where Score
  # Transformation the NamedTuple of parameters to a Vector
  # without applying any bounds (identity transform)
  trf = toidentitytransform(f.model.param)
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
        vrandeffsorth = []
      else
        vrandeffsorth = _orth_empirical_bayes(f.model, subject, _param, f.approx, args...; kwargs...)
      end

      marginal_nll_gradient!(
        _j,
        f.model,
        subject,
        _vparam,
        vrandeffsorth,
        f.approx,
        trf,
        args...;
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
        vrandeffsorth = []
      else
        vrandeffsorth = _orth_empirical_bayes(f.model, subject, coef(f), f.approx, args...; kwargs...)
      end
      marginal_nll_gradient!(
        g,
        f.model,
        subject,
        vparam,
        vrandeffsorth,
        f.approx,
        trf,
        args...;
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
                               ::FO,
                               args...; kwargs...)

  trf = toidentitytransform(m.param)
  vparam = TransformVariables.inverse(trf, param)

  # Costruct closure for calling _derived as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  __E_and_V = _param -> _E_and_V(m, subject, TransformVariables.transform(trf, _param), vrandeffsorth, FO(), args...; kwargs...)

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
  ::FO,
  args...;
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
    FO(),
    args...; kwargs...)

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
    return sum(_expected_information(model, data[i], param, vrandeffsorth[i], f.approx, f.args...; f.kwargs...) for i in 1:length(data))
  else
    return first(_observed_information(f, Val(false), args...; kwargs...))
  end
end

"""
    vcov(f::FittedPumasModel) -> Matrix

Compute the covariance matrix of the population parameters
"""
function StatsBase.vcov(f::FittedPumasModel)

  # Compute the observed information based on the Hessian (H) and the product of the outer scores (S)
  H, S = _observed_information(f, Val(true), f.args...; f.kwargs...)

  # Use generialized eigenvalue decomposition to compute inv(H)*S*inv(H)
  F = eigen(Symmetric(H), Symmetric(S))
  any(t -> t <= 0, F.values) && @warn("Hessian is not positive definite")
  return F.vectors*Diagonal(inv.(abs2.(F.values)))*F.vectors'
end

"""
    stderror(f::FittedPumasModel) -> NamedTuple

Compute the standard errors of the population parameters and return
the result as a `NamedTuple` matching the `NamedTuple` of population
parameters.
"""
StatsBase.stderror(f::FittedPumasModel) = stderror(infer(f))

function Statistics.mean(vfpm::Vector{<:FittedPumasModel})
  names = keys(coef(first(vfpm)))
  means = []
  for name in names
    push!(means, mean([coef(fpm)[name] for fpm in vfpm]))
  end
  NamedTuple{names}(means)
end
function Statistics.std(vfpm::Vector{<:FittedPumasModel})
  names = keys(coef(first(vfpm)))
  stds = []
  for name in names
    push!(stds, std([coef(fpm)[name] for fpm in vfpm]))
  end
  NamedTuple{names}(stds)
end
function Statistics.var(vfpm::Vector{<:FittedPumasModel})
  names = keys(coef(first(vfpm)))
  vars = []
  for name in names
    push!(vars, var([coef(fpm)[name] for fpm in vfpm]))
  end
  NamedTuple{names}(vars)
end

# This is called with dual numbered "param"
function _E_and_V(model::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffsorth::AbstractVector,
                  ::FO,
                  args...; kwargs...)

  randeffstransform = totransform(model.random(param))
  dist = _derived(
    model,
    subject,
    param,
    TransformVariables.transform(randeffstransform, vrandeffsorth),
    args...; kwargs...)

  _names = keys(subject.observations)

  E = vcat([mean.(dist[_name]) for _name in _names]...)

  F = _mean_derived_vηorth_jacobian(
    model,
    subject,
    param,
    vrandeffsorth,
    args...; kwargs...)

  FF = vcat([F[_name] for _name in _names]...)
  dd = vcat([var.(dist[_name]) for _name in _names]...)
  V = FF*FF' + Diagonal(dd)

  return E, V
end

function _E_and_V(m::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffsorth::AbstractVector,
                  ::FOCE,
                  args...; kwargs...)

  randeffstransform = totransform(m.random(param))
  y = subject.observations.dv
  dist0 = _derived(m, subject, param, TransformVariables.transform(randeffstransform, zero(vrandeffsorth)))
  dist  = _derived(m, subject, param, TransformVariables.transform(randeffstransform, vrandeffsorth))

  F = ForwardDiff.jacobian(
    _vrandeffs -> begin
      _randeffs = TransformVariable.transform(randeffstransform, _vrandeffs)
      return mean.(_derived(m, subject, param, _randeffs).dv)
    end,
    vrandeffsorth
  )
  V = Symmetric(F*F' + Diagonal(var.(dist0.dv)))

  return mean.(dist.dv) .- F*vrandeffsorth, V
end

struct FittedPumasModelInference{T1, T2, T3}
  fpm::T1
  vcov::T2
  level::T3
end

"""
    infer(fpm::FittedPumasModel) -> FittedPumasModelInference

Compute the `vcov` matrix and return a struct used for inference
based on the fitted model `fpm`.
"""
function infer(fpm::FittedPumasModel; level = 0.95, rethrow_error=false)
  print("Calculating: variance-covariance matrix")
  try
    _vcov = vcov(fpm, fpm.args...; fpm.kwargs...)
    println(". Done.")
    return FittedPumasModelInference(fpm, _vcov, level)
  catch err
    println(". Failed.")
    if rethrow_error
      rethrow(err)
    end
    return FittedPumasModelInference(fpm, err, level)
  end
end
StatsBase.coef(pmi::FittedPumasModelInference) = coef(pmi.fpm)
function StatsBase.stderror(pmi::FittedPumasModelInference)
  ss = sqrt.(diag(pmi.vcov))
  trf = tostderrortransform(pmi.fpm.model.param)
  return TransformVariables.transform(trf, ss)
end

# empirical_bayes_dist for FittedPumasModel
function empirical_bayes_dist(fpm::FittedPumasModel)
  map(zip(fpm.data, fpm.vvrandeffsorth)) do (subject, vrandeffsorth)
      empirical_bayes_dist(fpm.model, subject, coef(fpm), vrandeffsorth, fpm.approx, fpm.args...; fpm.kwargs...)
  end
end
