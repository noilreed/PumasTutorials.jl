function dim_param(m::PumasModel)
  t_param = totransform(m.param)
  TransformVariables.dimension(t_param)
end

function dim_rfx(m::PumasModel)
  x = init_param(m)
  rfx = m.random(x)
  t_rfx = totransform(rfx)
  TransformVariables.dimension(t_rfx)
end

# This object wraps the model, data and some pre-allocated buffers to match the necessary interface for DynamicHMC.jl
struct BayesLogDensity{M,D,B,C,R,K}
  model::M
  data::D
  dim_param::Int
  dim_rfx::Int
  buffer::B
  cfg::C
  res::R
  kwargs::K
end

function BayesLogDensity(model, data, ;kwargs...)
  m = dim_param(model)
  n = dim_rfx(model)
  buffer = zeros(m + n)
  cfg = ForwardDiff.GradientConfig(logdensity, buffer)
  res = DiffResults.GradientResult(buffer)
  BayesLogDensity(model, data, m, n, buffer, cfg, res, kwargs)
end

function dimension(b::BayesLogDensity)
  b.dim_param + b.dim_rfx * length(b.data)
end

function logdensity(b::BayesLogDensity, v::AbstractVector)
  m = b.dim_param
  trf = totransform(b.model.param)
  # compute the prior density and log-Jacobian
  param, j_param = TransformVariables.transform_and_logjac(trf, @view v[1:m])
  ℓ_param = _lpdf(b.model.param.params, param) + j_param

  n = b.dim_rfx
  rfx = b.model.random(param)
  t_rfx = totransform(rfx)
  ℓ_rfx = sum(enumerate(b.data)) do (i, subject)
    # compute the random effect density and likelihood
    vrandeffsorth = @view v[(m + (i - 1)*n) .+ (1:n)]
    return -Pumas._penalized_conditional_nll(b.model, subject, param, vrandeffsorth; b.kwargs...)
  end
  ℓ = ℓ_param + ℓ_rfx
  return isnan(ℓ) ? -Inf : ℓ
end

function logdensitygrad(b::BayesLogDensity, v::AbstractVector)
  ∇ℓ = zeros(size(v))

  param = b.model.param
  t_param = totransform(param)
  m = b.dim_param
  n = b.dim_rfx

  function L_param(u)
    x, j_x = TransformVariables.transform_and_logjac(t_param, u)
    _lpdf(param.params, x) + j_x
  end
  fill!(b.buffer, 0.0)
  copyto!(b.buffer, 1, v, 1, m)

  ForwardDiff.gradient!(b.res, L_param, b.buffer, b.cfg, Val{false}())

  ℓ = DiffResults.value(b.res)
  ∇ℓ[1:m] .= @view DiffResults.gradient(b.res)[1:m]

  for (i, subject) in enumerate(b.data)
    # to avoid dimensionality problems with ForwardDiff.jl, we split the computation to
    # compute the gradient for each subject individually, then accumulate this to the
    # gradient vector.

    function L_rfx(u)
      param = TransformVariables.transform(t_param, @view u[1:m])
      vrandeffsorth = @view u[m .+ (1:n)]
      return -Pumas._penalized_conditional_nll(b.model, subject, param, vrandeffsorth; b.kwargs...)
    end
    copyto!(b.buffer, m + 1, v, m + (i - 1)*n + 1, n)

    ForwardDiff.gradient!(b.res, L_rfx, b.buffer, b.cfg, Val{false}())

    ℓ += DiffResults.value(b.res)
    ∇ℓ[1:m] .+= @view DiffResults.gradient(b.res)[1:m]
    copyto!(∇ℓ, m + (i - 1)*n + 1, DiffResults.gradient(b.res), m + 1, n)
  end
  return isnan(ℓ) ? -Inf : ℓ, ∇ℓ
end

# results wrapper object
struct BayesMCMCResults{L<:BayesLogDensity,C,T}
  loglik::L
  chain::C
  tuned::T
end

Base.show(io::IO, ::MIME"text/plain", b::BayesMCMCResults) = show(io, MIME"text/plain"(), Chains(b))

struct BayesMCMC <: LikelihoodApproximation end

function Distributions.fit(
  model::PumasModel,
  data::Population,
  param::NamedTuple,
  ::BayesMCMC;
  nadapts::Integer=2000,
  nsamples::Integer=10000,
  progress = Base.is_interactive,
  kwargs...
)
  # Extract parameter transformations with and without bounds
  trf       = totransform(model.param)
  trf_ident = toidentitytransform(model.param)

  # Transform NamedTuple of initial parameters to a Vector
  vparam = Pumas.TransformVariables.inverse(trf, param)

  # Create BayesLogDensity objecet from Pumas model and data
  bayes = BayesLogDensity(model, data; kwargs...)

  # Augment parameter Vector with vector of random effects
  vparam_aug = [vparam; zeros(length(data)*bayes.dim_rfx)]

  # Create functions for the density and the gradient as functions of just the augmented parameter vector
  # (i.e. closing over the BayesLogDensity object)
  l(θ)    = logdensity(bayes, θ)
  dldθ(θ) = logdensitygrad(bayes, θ)

  # Set up the NUTS sampler from AdvancedHMC
  metric  = DiagEuclideanMetric(length(vparam_aug))
  h       = Hamiltonian(metric, l, dldθ)
  integrator = Leapfrog(find_good_stepsize(h, vparam_aug))
  prop    = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
  adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, prop.integrator))

  # Run the MCMC sampler
  samples, stats = sample(h, prop, vparam_aug, nsamples, adaptor, nadapts; progress=progress)

  return BayesMCMCResults(bayes, samples, stats)
end

function Chains(b::BayesMCMCResults)
  # Extract parameter transformations with and without bounds
  trf       = totransform(b.loglik.model.param)
  trf_ident = toidentitytransform(b.loglik.model.param)

  # Construct closure that transforms the parameter vector to a NamedTuple by applying the parameter bounds
  trans      = v -> TransformVariables.transform(trf, v)
  # Construct closure that transform the NamedTuple to a Vector without applying the parameter bounds
  transident = v -> TransformVariables.inverse(trf_ident, v)

  # Apply the transformations to the samples
  samples_transf  = trans.(b.chain)
  samples_transid = transident.(samples_transf)

  # Construct labels to the parameters
  names = String[]
  vals = []
  for (name, val) in pairs(samples_transf[1])
    _push_varinfo!(names, vals, nothing, nothing, name, val, nothing, nothing)
  end

  # Return a Chains object for nice printing of the results
  return Chains(samples_transid, names)
end

# remove unnecessary PDMat wrappers
_clean_param(x) = x
_clean_param(x::PDMat) = x.mat
_clean_param(x::NamedTuple) = map(_clean_param, x)

# "unzip" the results via a StructArray
function param_values(b::BayesMCMCResults)
  trf  = Pumas.totransform(b.loglik.model.param)
  vals = TransformVariables.transform.(Ref(trf), b.chain)
  return vals
end

function param_reduce(f, b::BayesMCMCResults)
  trf  = Pumas.totransform(        b.loglik.model.param)
  trfi = Pumas.toidentitytransform(b.loglik.model.param)
  vals = TransformVariables.inverse.(Ref(trfi), TransformVariables.transform.(Ref(trf), b.chain))
  return TransformVariables.transform(trfi, f(vals))
end
param_mean(b::BayesMCMCResults) = param_reduce(mean, b)
param_var(b::BayesMCMCResults)  = param_reduce(var, b)
param_std(b::BayesMCMCResults)  = param_reduce(std, b)
