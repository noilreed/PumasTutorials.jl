const DEFAULT_ESTIMATION_RELTOL=1e-8
const DEFAULT_ESTIMATION_ABSTOL=1e-12
const DEFAULT_SIMULATION_RELTOL=1e-3
const DEFAULT_SIMULATION_ABSTOL=1e-6

"""
    PumasModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `random`: a mapping from a named tuple of parameters -> `ParamSet`
- `pre`: a mapping from the (params, randeffs, subject) -> ODE params
- `init`: a mapping (col,t0) -> inital conditions
- `prob`: a DEProblem describing the dynamics (either exact or analytical)
- `derived`: the derived variables and error distributions (param, randeffs, data, ode vals) -> sampling dist
- `observed`: simulated values from the error model and post processing: (param, randeffs, data, ode vals, samples) -> vals
"""
mutable struct PumasModel{P,Q,R,S,T,V,W,O}
  param::P
  random::Q
  pre::R
  init::S
  prob::T
  derived::V
  observed::W
  options::O
end
PumasModel(param, random, pre, init, prob, derived, observed=(col, sol, obstimes, samples, subject) -> samples) =
    PumasModel(param, random, pre, init, prob, derived, observed, (subject_time=false,))

init_param(model::PumasModel) = init(model.param)
init_randeffs(model::PumasModel, param) = init(model.random(param))

"""
    sample_randeffs([rng::AbstractRNG=Random.default_rng(),] model::PumasModel, param::NamedTuple)

Generate a random set of random effects for model `model`, using parameters `param`. Optionally, a random number generator object `rng` can be passed as the first arugment.
"""
sample_randeffs(rng::AbstractRNG, model::PumasModel, param::NamedTuple) = rand(rng, model.random(param))

sample_randeffs(model::PumasModel, param::NamedTuple) = sample_randeffs(default_rng(), model, param)

# How long to solve
function timespan(model::PumasModel)
  subject_time = model.options.subject_time
  if subject_time === false
    return (0, nothing)
  else
    return nothing
  end
end
function timespan(sub::Subject, tspan, saveat)
  if isempty(sub.events) && isempty(saveat) && isempty(sub.time) && tspan == (nothing,nothing)
    error("No timespan is given. This means no events, observations, or user chosen time span exist for the subject. Please check whether the data was input correctly.")
  end
  e_lo, e_hi = !isnothing(sub.events) && !isempty(sub.events) ? extrema(evt.time for evt in sub.events) : (Inf,-Inf)
  s_lo, s_hi = !isnothing(saveat) && !isempty(saveat) ? extrema(saveat) : (Inf,-Inf)
  obs_lo, obs_hi = !isnothing(sub.time) && !isempty(sub.time) ? extrema(sub.time) : (Inf,-Inf)
  lo = minimum((e_lo,s_lo,obs_lo))
  hi = maximum((e_hi,s_hi,obs_hi))
  tspan !== nothing && tspan[1] !== nothing && (lo = tspan[1]) # User override
  tspan !== nothing && tspan[2] !== nothing && (hi = tspan[2]) # User override
  lo == Inf && error("No starting time identified. Please supply events or obstimes")
  hi == -Inf && error("No ending time identified. Please supply events, obstimes")
  lo, hi
end

# Where to save
# `sub.time` has the highest precedence
# then `0:lastdose+24`
# then `0:24`
observationtimes(sub::Subject) = !isnothing(sub.time) ? sub.time :
                                 !isnothing(sub.events) && !isempty(sub.events) ?
                                 (0.0:1.0:(sub.events[end].time+24.0)) :
                                 (0.0:24.0)

"""
    sol = solve(model::PumasModel, subject::Subject, param,
                randeffs=sample_randeffs(rng, model, param),
                saveat = nothing,
                args...; kwargs...)

Compute the ODE for model `model`, with parameters `param`, random effects
`randeffs` and a collection of times to save the solution at `saveat`. `args`
and `kwargs` are passed to the ODE solver. If no `randeffs` are given, then
they are generated according to the distribution determined in the model. If
no `saveat` times are given, the times are chosen to be the vector of observed
times for `subject`.

Returns a tuple containing the ODE solution `sol` and collation `col`.
"""
function DiffEqBase.solve(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple = init_param(model),
  randeffs::Union{Nothing,NamedTuple}=nothing,
  args...;
  saveat = nothing,
  callback = nothing,
  rng::AbstractRNG = default_rng(),
  kwargs...)
  if saveat === nothing
    # Base.depwarn("Calling solve without `saveat` set is deprecated. Please provide the times you wish to save at.", :solve)
    saveat = observationtimes(subject)
  end

  _randeffs = if randeffs===nothing
    sample_randeffs(rng, model, param)
  else
    randeffs
  end

  col = model.pre(param, _randeffs, subject)
  model.prob === nothing && return NullDESolution(NullDEProblem(col))
  prob = _problem(model, subject, col, args...;
    saveat=saveat, callback=callback, kwargs...)
  alg = model.prob isa ExplicitModel ? nothing : alg=AutoTsit5(Rosenbrock23())
  return solve(prob, args...; alg=alg, kwargs...)
end

function DiffEqBase.solve(model::PumasModel, pop::DataFrame, args...;kwargs...)
  throw(ArgumentError("The second argument to solve was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame."))
end

function DiffEqBase.solve(
  model::PumasModel,
  population::Population,
  param::NamedTuple = init_param(model),
  randeffs = nothing,
  args...;
  callback = nothing,
  alg=AutoTsit5(Rosenbrock23()),
  ensemblealg = EnsembleThreads(),
  rng::AbstractRNG=default_rng(),
  kwargs...)

  # Compare keys in param with @param in the model
  _compare_keys(model, param)
  # Check that doses happen into existing compartments
  map(subject -> _check_dose_compartments(model, subject, param), population)


  function solve_prob_func(prob, i, repeat)
    # This might be racy because the same RNG obsject is passed
    _randeffs = randeffs === nothing ? sample_randeffs(rng, model, param) : randeffs
    col = model.pre(param, _randeffs, population[i])
    _problem(model, population[i], col, args...;
      callback=callback, kwargs...)
  end
  prob = EnsembleProblem(model.prob, prob_func = solve_prob_func)
  return solve(prob, alg, ensemblealg, args...;
    trajectories = length(population), kwargs...)
end

# This internal function is just so that the collation doesn't need
# to be repeated in the other API functions
function _problem(model::PumasModel, subject, col, args...;
                  tspan=nothing, saveat=Float64[], kwargs...)
                  # should ^ this be a typed time? 
  model.prob === nothing && return NullDEProblem(col)

  if tspan === nothing
    tspan = timespan(model)
    tspan = float.(timespan(subject, tspan, saveat))
  end

  if model.prob isa ExplicitModel
    _prob = _build_analytical_problem(model, subject, tspan, col, args...; kwargs...)
  elseif model.prob isa AnalyticalPKProblem
    _prob1 = _build_analytical_problem(model, subject, tspan, col, args...;kwargs...)
    pksol = solve(_prob1, args...; kwargs...)
    function _col(t)
      col_t = col(t)
      ___pk = convert(NamedTuple, pksol(t))
      merge(col_t, ___pk)
    end
    u0  = model.init(col, tspan[1])

    mtmp = PumasModel(
      model.param,
      model.random,
      model.pre,
      model.init,
      remake(model.prob.prob2; p=_col, u0=u0, tspan=tspan),
      model.derived,
      model.observed)

    _prob = PresetAnalyticalPKProblem(
      _build_diffeq_problem(mtmp, subject, args...;
      saveat=saveat, make_events=false, kwargs...),
      pksol)
  else
    u0  = model.init(col, tspan[1])
    if typeof(model.prob) <: DiffEqBase.AbstractJumpProblem
      # Remake needs to happen on the ODE/SDE/DDEProblem, so we have
      # to remake the internal prob and rewrap
      lowprob = remake(model.prob.prob; p=col, u0=u0, tspan=tspan)
      topprob = JumpProblem{
        DiffEqBase.isinplace(lowprob),
        typeof(lowprob),
        typeof(model.prob.aggregator),
        typeof(model.prob.jump_callback),
        typeof(model.prob.discrete_jump_aggregation),
        typeof(model.prob.variable_jumps),
        typeof(model.prob.regular_jump),
        typeof(model.prob.massaction_jump)}(
        lowprob,
        model.prob.aggregator,
        model.prob.discrete_jump_aggregation,
        model.prob.jump_callback,
        model.prob.variable_jumps,
        model.prob.regular_jump,
        model.prob.massaction_jump)
    else
      topprob = remake(model.prob; p=col, u0=u0, tspan=tspan)
    end
    mtmp = PumasModel(
      model.param,
      model.random,
      model.pre,
      model.init,
      topprob,
      model.derived,
      model.observed)
    _prob = _build_diffeq_problem(mtmp, subject, args...; saveat=saveat, kwargs...)
  end
  return _prob
end

function _derived(model::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractArray,
                  args...;
                  kwargs...)
  rtrf = totransform(model.random(param))
  randeffs = TransformVariables.transform(rtrf, vrandeffs)
  return _derived(model, subject, param, randeffs, args...; kwargs...)
end

# This internal function is just so that the calculation of derived
# doesn't need to be repeated in the other API functions
@inline function _derived(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  randeffs::NamedTuple,
  args...;
  # This is the only entry point to the ODE solver for
  # the estimation code so estimation-specific defaults
  # are set here, but are overriden in other cases.
  # Super messy and should get cleaned.
  reltol=DEFAULT_ESTIMATION_RELTOL,
  abstol=DEFAULT_ESTIMATION_ABSTOL,
  alg = AutoVern7(Rodas5(autodiff=true)),
  # Estimation only uses subject.time for the
  # observation time series
  obstimes = nothing,
  callback = nothing,
  kwargs...)

  obstimes = obstimes === nothing ? subject.time : obstimes
  # collate that arguments
  collated = model.pre(param, randeffs, subject)
  # create solution object. By passing saveat=obstimes, we compute the solution only
  # at obstimes such that we can simply pass solution.u to model.derived
  _saveat = obstimes === nothing ? Float64[] : obstimes
  _prob = _problem(
    model,
    subject,
    collated,
    args...;
    saveat=_saveat,
    callback=callback,
    kwargs...)
  if _prob isa NullDEProblem
    dist = model.derived(
      collated,
      nothing,
      obstimes,
      subject,
      param,
      randeffs)
  else
    sol = solve(
      _prob,
      args...;
      reltol=reltol,
      abstol=abstol,
      alg=alg,
      kwargs...)
    # if solution contains NaN return Inf
    if (sol.retcode !== :Success && sol.retcode !== :Terminated) ||
      # FIXME! Make this uniform across the two solution types
      # FIXME! obstimes can be empty
      any(x->any(isnan, x), sol isa PKPDAnalyticalSolution ? sol(obstimes[end]) : sol.u[end])
      # FIXME! Do we need to make this type stable?
      return map(x->nothing, subject.observations) # create a named tuple of nothing with the observed names ( should this be all of derived?)
    end

    # extract distributions
    dist = model.derived(collated, sol, obstimes, subject, param, randeffs)
  end

  return dist
end

function _derived(
  model::PumasModel,
  pop::Population,
  param::NamedTuple,
  vrandeffs::Union{Nothing,AbstractVector}=nothing,
  args...;
  # This is the only entry point to the ODE solver for
  # the estimation code so estimation-specific defaults
  # are set here, but are overriden in other cases.
  # Super messy and should get cleaned.
  reltol=DEFAULT_ESTIMATION_RELTOL,
  abstol=DEFAULT_ESTIMATION_ABSTOL,
  alg = AutoVern7(Rodas5(autodiff=false)),
  # Estimation only uses subject.time for the
  # observation time series
  obstimes = nothing,
  callback = nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...)

  if vrandeffs !== nothing && length(pop) !== length(vrandeffs)
    throw(DimensionMismatch("The population and random effects input must have equal length, got $(length(pop)) and $(length(vrandeffs))."))
  end

  _vrandeffs = if vrandeffs===nothing
    [sample_randeffs(rng, model, param) for i in 1:length(pop)]
  else
    vrandeffs
  end

  if obstimes === nothing
    throw(ArgumentError("obstimes argument needs to be passed"))
  end

  return _simobs(model,
    pop,
    RepeatedVector([param], length(pop)),
    _vrandeffs,
    args...;
    reltol=reltol,
    abstol=abstol,
    alg=alg,
    obstimes=obstimes,
    callback=callback,
    isfor_derived=true,
    rng=rng,
    kwargs...)
end

function _derived(
  model::PumasModel,
  subject::Subject,
  param::AbstractVector,
  vrandeffs::Union{Nothing,AbstractVector}=nothing,
  args...;
  # This is the only entry point to the ODE solver for
  # the estimation code so estimation-specific defaults
  # are set here, but are overriden in other cases.
  # Super messy and should get cleaned.
  reltol=DEFAULT_ESTIMATION_RELTOL,
  abstol=DEFAULT_ESTIMATION_ABSTOL,
  alg = AutoVern7(Rodas5(autodiff=false)),
  # Estimation only uses subject.time for the
  # observation time series
  obstimes = subject.time,
  callback = nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...)

  if !(vrandeffs isa Nothing) && length(param) !== length(vrandeffs)
    throw(DimensionMismatch("The fixed and random effects input must have equal length, got $(length(param)) and $(length(vrandeffs))."))
  end

  _vrandeffs = if vrandeffs===nothing
    [sample_randeffs(rng, model, param) for i in 1:length(param)]
  else
    vrandeffs
  end

  return _simobs(model,
    RepeatedVector([subject], length(param)),
    param,
    _vrandeffs,
    args...;
    reltol=reltol,
    abstol=abstol,
    alg=alg,
    obstimes=obstimes,
    callback=callback,
    isfor_derived=true,
    rng=rng,
    kwargs...)
end

#=
_rand(d)

Samples a random value from a distribution or if it's a number assumes it's the
constant distribution and passes it through.
=#
_rand(rng::AbstractRNG, d::Distributions.Sampleable) = rand(rng, d)
_rand(rng::AbstractRNG, d::AbstractArray) = map(s -> _rand(rng, s), d)
_rand(rng::AbstractRNG, d::NamedTuple) = map(s -> _rand(rng, s), d)
_rand(rng::AbstractRNG, d) = d

"""
    simobs(model::PumasModel, subject::Subject, param[, randeffs, args...];
           obstimes::AbstractArray=observationtimes(subject),
           rng::AbstractRNG=Random.default_rng(),
           kwargs...)

Simulate random observations from model `model` for `subject` with parameters `param` at
`obstimes` (by default, use the times of the existing observations for the subject). If no
`randeffs` is provided, then random ones are generated according to the distribution
in the model.
"""
function simobs(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple=init_param(model),
  randeffs::Union{Nothing,NamedTuple}=nothing,
  args...;
  obstimes=nothing,
  callback = nothing,
  saveat=obstimes,
  rng::AbstractRNG=default_rng(),
  kwargs...)
  if obstimes === nothing
    # Base.depwarn("Calling simobs without `obstimes` set is deprecated. Please provide the times you wish to store in the SimulatedObservations.", :simobs)
    obstimes = observationtimes(subject)
    if saveat === nothing
      saveat = obstimes
    end
  end
  _randeffs = if randeffs===nothing
    sample_randeffs(rng, model, param)
  else
    randeffs
  end
  col = model.pre(_rand(rng, param), _randeffs, subject)
  prob = _problem(
    model,
    subject,
    col,
    args...;
    saveat=saveat,
    callback=callback,
    kwargs...)
  alg = model.prob isa ExplicitModel ? nothing : alg=AutoTsit5(Rosenbrock23())
  sol = prob !== nothing ? solve(prob, args...; alg=alg, kwargs...) : nothing
  derived = model.derived(col, sol, obstimes, subject, param, _randeffs)
  obs = model.observed(col, sol, obstimes, map(s -> _rand(rng, s), derived), subject)
  return SimulatedObservations(subject, obstimes, obs)
events
end

struct RepeatedVector{T} <: AbstractVector{T}
    arr::AbstractVector{T}
    n::Int
end
Base.size(A::RepeatedVector) = (length(A.arr)*A.n,)
Base.@propagate_inbounds Base.getindex(A::RepeatedVector,i) = A.arr[mod1(i,length(A.arr))]

function simobs(
  model::PumasModel,
  subject::Subject,
  vparam::AbstractVector,
  vrandeffs::Union{Nothing,AbstractVector}=nothing,
  args...;
  rng::AbstractRNG=default_rng(),
  kwargs...)

  if vrandeffs !== nothing && length(vparam) !== length(vrandeffs)
    throw(DimensionMismatch("The fixed and random effects input must have equal length, got $(length(vparam)) and $(length(vrandeffs))."))
  end

  return _simobs(
    model,
    RepeatedVector([subject], length(vparam)),
    vparam,
    vrandeffs,
    args...;
    rng=rng,
    kwargs...)
end

function simobs(model::PumasModel, pop::DataFrame, args...; kwargs...)
  throw(ArgumentError("The second argument to simobs was a DataFrame instead of a Population. Please use read_pumas to construct a Population from a DataFrame."))
end

function simobs(
  model::PumasModel,
  population::Population,
  param::NamedTuple=init_param(model),
  vrandeffs::Union{Nothing,AbstractVector}=nothing,
  args...;
  rng::AbstractRNG=default_rng(),
  kwargs...)

  if vrandeffs !== nothing && length(population) !== length(vrandeffs)
    throw(DimensionMismatch("The population and random effects input must have equal length, got $(length(population)) and $(length(vrandeffs))."))
  end

  # Compare keys in param with @param in the model
  _compare_keys(model, param)
  # Check that doses happen into existing compartments
  # We need to map _rand for Koopmans Expectation
  map(subject -> _check_dose_compartments(
    model,
    subject,
    map(s -> _rand(rng, s), param)),
    population)

  return _simobs(
    model,
    population,
    RepeatedVector([param], length(population)),
    vrandeffs,
    args...;
    rng=rng,
    kwargs...)
end

function simobs(
  model::PumasModel,
  population::Population,
  vparam::AbstractVector,
  vrandeffs::Union{Nothing,AbstractVector}=nothing,
  args...;
  rng::AbstractRNG=default_rng(),
  kwargs...)

  # Compare keys in param with @param in the model
  map(_param -> _compare_keys(model, _param), vparam)

  # Check that doses happen into existing compartments
  # We need to map _rand for Koopmans Expectation
  map(
    _param -> map(
      subject -> _check_dose_compartments(
        model,
        subject,
        map(s -> _rand(rng, s), _param)),
      population),
    vparam)

  out = _simobs(model,
    RepeatedVector(population, length(vparam)),
    RepeatedVector(vparam, length(population)),
    vrandeffs,
    args...;
    rng=rng,
    kwargs...)

  n = length(population)
  # Chop up into arrays of each population matching parameter i
  return [out[((i - 1)*n + 1):i*n] for i in 1:length(vparam)]
end

function _simobs(
  model::PumasModel,
  population::Population,
  vparam::AbstractVector,
  vrandeffs::Union{Nothing,AbstractVector},
  args...;
  alg=AutoTsit5(Rosenbrock23()),
  ensemblealg = EnsembleSerial(),
  callback = nothing,
  isfor_derived = false,
  rng::AbstractRNG=default_rng(),
  kwargs...)

  if !(vrandeffs isa Nothing) && length(population) !== length(vrandeffs)
    throw(DimensionMismatch("The population and random effects input must have equal length, got $(length(population)) and $(length(vrandeffs))."))
  end

  if length(population) !== length(vparam)
    throw(DimensionMismatch("The population and parameter input must have equal length, got $(length(population)) and $(length(vparam))."))
  end

  _vparam = [_rand(rng, vparam[i]) for i in 1:length(vparam)]
  _vrandeffs = if vrandeffs === nothing
    [sample_randeffs(rng, model, _vparam[i]) for i in 1:length(population)]
  else
    vrandeffs
  end

  function simobs_prob_func(prob, i, repeat)
    col = model.pre(_vparam[i], _vrandeffs[i], population[i])
    obstimes = :obstimes ∈ keys(kwargs) ? kwargs[:obstimes] : observationtimes(population[i])
    saveat = :saveat ∈ keys(kwargs) ? kwargs[:saveat] : obstimes
    _problem(model, population[i], col, args...; saveat=saveat, callback=callback, kwargs...)
  end

  function simobs_output_func(sol, i)
    col = sol.prob.p
    obstimes = :obstimes ∈ keys(kwargs) ? kwargs[:obstimes] : observationtimes(population[i])
    saveat = :saveat ∈ keys(kwargs) ? kwargs[:saveat] : obstimes

    if isfor_derived
      if (sol.retcode != :Success && sol.retcode != :Terminated) ||
        # FIXME! Make this uniform across the two solution types
        # FIXME! obstimes can be empty
        any(x->any(isnan, x), sol isa PKPDAnalyticalSolution ? sol(obstimes[end]) : sol.u[end])
        # FIXME! Do we need to make this type stable?
          return map(x->nothing, subject.observations), false # create a named tuple of nothing with the observed names ( should this be all of derived?)
      end
      return model.derived(col, sol, obstimes, population[i], _vparam[i], _vrandeffs[i]), false
    else
      derived = model.derived(col, sol, obstimes, population[i], _vparam[i], _vrandeffs[i])
      obs = model.observed(col, sol, obstimes, map(s -> _rand(rng, s), derived), population[i])
      return SimulatedObservations(population[i], obstimes, obs), false
    end
  end

  prob = EnsembleProblem(model.prob; prob_func = simobs_prob_func,
                         output_func = simobs_output_func)
  return solve(prob, alg, ensemblealg, args...;
    trajectories = length(population), kwargs...).u
end


# Returns the parameters of the differential equation for a specific
# subject subject to parameter and random effects choices. Intended
# for internal use and debugging.
function pre(model::PumasModel, subject::Subject, param::NamedTuple, randeffs::NamedTuple)
  model.pre(param, randeffs, subject)
end
