const DEFAULT_ESTIMATION_RELTOL=1e-8
const DEFAULT_ESTIMATION_ABSTOL=1e-12
const DEFAULT_SIMULATION_RELTOL=1e-3
const DEFAULT_SIMULATION_ABSTOL=1e-6

"""
    PumasModel

A model takes the following arguments
- `param`: a `ParamSet` detailing the parameters and their domain
- `random`: a mapping from a named tuple of parameters -> `DistSet`
- `pre`: a mapping from the (params, randeffs, subject) -> ODE params
- `init`: a mapping (col,t0) -> inital conditions
- `prob`: a DEProblem describing the dynamics (either exact or analytical)
- `derived`: the derived variables and error distributions (param, randeffs, data, ode vals) -> sampling dist
- `observed`: simulated values from the error model and post processing: (param, randeffs, data, ode vals, samples) -> vals
"""
mutable struct PumasModel{P,Q,R,S,T,V,W}
  param::P
  random::Q
  pre::R
  init::S
  prob::T
  derived::V
  observed::W
end
PumasModel(param,random,pre,init,prob,derived) =
    PumasModel(param,random,pre,init,prob,derived,(col,sol,obstimes,samples,subject)->samples)

init_param(m::PumasModel) = init(m.param)
init_randeffs(m::PumasModel, param) = init(m.random(param))

"""
    sample_randeffs(m::PumasModel, param)

Generate a random set of random effects for model `m`, using parameters `param`.
"""
sample_randeffs(m::PumasModel, param) = rand(m.random(param))

# How long to solve
function timespan(sub::Subject,tspan,saveat)
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
    sol = solve(m::PumasModel, subject::Subject, param,
                randeffs=sample_randeffs(m, param),
                saveat = observationtimes(subject),
                args...; kwargs...)

Compute the ODE for model `m`, with parameters `param`, random effects
`randeffs` and a collection of times to save the solution at `saveat`.
`args` and `kwargs` are passed to the ODE solver. If no `randeffs` are
given, then they are generated according to the distribution determined
in the model. If no `saveat` times are given, the times are chosen to be
the vector of observed times for `subject`.

Returns a tuple containing the ODE solution `sol` and collation `col`.
"""
function DiffEqBase.solve(m::PumasModel, subject::Subject,
                          param = init_param(m),
                          randeffs = sample_randeffs(m, param),
                          args...;
                          saveat = observationtimes(subject),
                          callback = nothing,
                          kwargs...)
  col = m.pre(param, randeffs, subject)
  m.prob === nothing && return NullDESolution(NullDEProblem(col))
  prob = _problem(m,subject,col,args...;saveat=saveat,callback=callback,kwargs...)
  alg = m.prob isa ExplicitModel ? nothing : alg=AutoTsit5(Rosenbrock23())
  solve(prob,args...;alg=alg,kwargs...)
end

function DiffEqBase.solve(m::PumasModel, pop::Population,
                          param = init_param(m),
                          randeffs = nothing,
                          args...;
                          callback = nothing,
                          alg=AutoTsit5(Rosenbrock23()),
                          ensemblealg = EnsembleThreads(),
                          kwargs...)

  function solve_prob_func(prob,i,repeat)
    _randeffs = randeffs === nothing ? sample_randeffs(m, param) : randeffs
    col = m.pre(param, _randeffs, pop[i])
    _problem(m,pop[i],col,args...;callback=callback,kwargs...)
  end
  prob = EnsembleProblem(m.prob,prob_func = solve_prob_func)
  solve(prob,alg,ensemblealg,args...;trajectories = length(pop),kwargs...)
end

"""
This internal function is just so that the collation doesn't need to
be repeated in the other API functions
"""
function _problem(m::PumasModel, subject, col, args...;
                tspan=nothing, saveat=Float64[], kwargs...)
  m.prob === nothing && return NullDEProblem(col)
  if tspan === nothing
    tspan = float.(timespan(subject,tspan,saveat))
  end

  if m.prob isa ExplicitModel
    _prob = _build_analytical_problem(m, subject, tspan, col, args...;kwargs...)
  elseif m.prob isa AnalyticalPKProblem
    _prob1 = _build_analytical_problem(m, subject, tspan, col, args...;kwargs...)
    pksol = solve(_prob1,args...;kwargs...)
    function _col(t)
      col_t = col(t)
      ___pk = pksol(t)
      (col_t..., Depot = ___pk.Depot, Central = ___pk.Central)
    end
    u0  = m.init(col, tspan[1])

    mtmp = PumasModel(m.param,
                     m.random,
                     m.pre,
                     m.init,
                     remake(m.prob.prob2; p=_col, u0=u0, tspan=tspan),
                     m.derived,
                     m.observed)
    _prob = PresetAnalyticalPKProblem(_build_diffeq_problem(mtmp, subject, args...;saveat=saveat, make_events=false, kwargs...),pksol)
  else
    u0  = m.init(col, tspan[1])
    mtmp = PumasModel(m.param,
                     m.random,
                     m.pre,
                     m.init,
                     remake(m.prob; p=col, u0=u0, tspan=tspan),
                     m.derived,
                     m.observed)
    _prob = _build_diffeq_problem(mtmp, subject, args...;saveat=saveat,
                                  make_events=!isempty(subject.events) && !isnothing(subject.events),
                                  kwargs...)
  end
  _prob
end

function _derived(model::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffs::AbstractArray,
                  args...;
                  kwargs...)
  rtrf = totransform(model.random(param))
  randeffs = TransformVariables.transform(rtrf, vrandeffs)
  dist = _derived(model, subject, param, randeffs, args...; kwargs...)
end

"""
This internal function is just so that the calculation of derived doesn't need
to be repeated in the other API functions
"""
@inline function _derived(m::PumasModel,
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
                          alg = AutoVern7(Rodas5(autodiff=false)),
                          # Estimation only uses subject.time for the
                          # observation time series
                          obstimes = subject.time,
                          callback = nothing,
                          kwargs...)
  # collate that arguments
  collated = m.pre(param, randeffs, subject)
  # create solution object. By passing saveat=obstimes, we compute the solution only
  # at obstimes such that we can simply pass solution.u to m.derived
  _saveat = obstimes === nothing ? Float64[] : obstimes
  _prob = _problem(m, subject, collated, args...; saveat=_saveat, callback=callback, kwargs...)
  if _prob isa NullDEProblem
    dist = m.derived(collated, nothing, obstimes, subject, param, randeffs)
  else
    sol = solve(_prob,args...; reltol=reltol, abstol=abstol, alg=alg, kwargs...)
    # if solution contains NaN return Inf
    if (sol.retcode != :Success && sol.retcode != :Terminated) ||
      # FIXME! Make this uniform across the two solution types
      # FIXME! obstimes can be empty
      any(x->any(isnan,x), sol isa PKPDAnalyticalSolution ? sol(obstimes[end]) : sol.u[end])
      # FIXME! Do we need to make this type stable?
            return map(x->nothing, subject.observations) # create a named tuple of nothing with the observed names ( should this be all of derived?)
    end

    # extract distributions
    dist = m.derived(collated, sol, obstimes, subject, param, randeffs)
  end
  dist
end

#=
_rand(d)

Samples a random value from a distribution or if it's a number assumes it's the
constant distribution and passes it through.
=#
_rand(d::Distributions.Sampleable) = rand(d)
_rand(d::AbstractArray) = map(_rand,d)
_rand(d::NamedTuple) = map(_rand,d)
_rand(d) = d

"""
    simobs(m::PumasModel, subject::Subject, param[, randeffs, [args...]];
                  obstimes::AbstractArray=observationtimes(subject),kwargs...)

Simulate random observations from model `m` for `subject` with parameters `param` at
`obstimes` (by default, use the times of the existing observations for the subject). If no
`randeffs` is provided, then random ones are generated according to the distribution
in the model.
"""
function simobs(m::PumasModel, subject::Subject,
                param = init_param(m),
                randeffs=sample_randeffs(m, param),
                args...;
                obstimes::AbstractArray=observationtimes(subject),
                callback = nothing,
                saveat=obstimes,kwargs...)
  col = m.pre(_rand(param), randeffs, subject)
  prob = _problem(m, subject, col, args...; saveat=saveat, callback=callback, kwargs...)
  alg = m.prob isa ExplicitModel ? nothing : alg=AutoTsit5(Rosenbrock23())
  sol = prob !== nothing ? solve(prob, args...; alg=alg, kwargs...) : nothing
  derived = m.derived(col,sol,obstimes,subject,param,randeffs)
  obs = m.observed(col,sol,obstimes,map(_rand,derived),subject)
  SimulatedObservations(subject,obstimes,obs)
end

function simobs(m::PumasModel, pop::Population,
                param = init_param(m),
                randeffs=nothing,
                args...;
                alg=AutoTsit5(Rosenbrock23()),
                ensemblealg = EnsembleThreads(),
                callback = nothing,
                kwargs...)
  if !(randeffs isa Nothing) && length(pop) !== length(randeffs)
    throw(DimensionMismatch("The population and random effects input must have equal length, got $(length(pop)) and $(length(randeffs))."))
  end
  function simobs_prob_func(prob,i,repeat)
    _param = _rand(param)
    _randeffs = randeffs === nothing ? sample_randeffs(m, _param) : randeffs[i]
    col = m.pre(_param, _randeffs, pop[i])
    obstimes = :obstimes ∈ keys(kwargs) ? kwargs[:obstimes] : observationtimes(pop[i])
    saveat = :saveat ∈ keys(kwargs) ? kwargs[:saveat] : obstimes
    _problem(m,pop[i],col,args...; saveat=saveat, callback=callback, kwargs...)
  end

  function simobs_output_func(sol,i)
    col = sol.prob.p
    obstimes = :obstimes ∈ keys(kwargs) ? kwargs[:obstimes] : observationtimes(pop[i])
    saveat = :saveat ∈ keys(kwargs) ? kwargs[:saveat] : obstimes
    _randeffs = randeffs === nothing ? nothing : randeffs[i]
    derived = m.derived(col, sol, obstimes, pop[i], param, _randeffs)
    obs = m.observed(col,sol,obstimes,map(_rand,derived),pop[i])
    SimulatedObservations(pop[i],obstimes,obs),false
  end

  prob = EnsembleProblem(m.prob,prob_func = simobs_prob_func,
                         output_func = simobs_output_func)
  solve(prob,alg,ensemblealg,args...;trajectories = length(pop),kwargs...).u
end
"""
    pre(m::PumasModel, subject::Subject, param, randeffs)

Returns the parameters of the differential equation for a specific subject
subject to parameter and random effects choices. Intended for internal use
and debugging.
"""
function pre(m::PumasModel, subject::Subject, param, randeffs)
  m.pre(param, randeffs, subject)
end
