"""
  AnalyticalPKPDProblem

An analytical PK(PD) problem.

Fields:
 * `f`: callable that returns the state of the dynamic system
 * `u0`: initial conditions
 * `tspan`: time points that define the intervals in which the problem is defined
 * `events`: events such as dose, reset events, etc
 * `times`: event times
 * `p`: a function that returns pre block evaluated at a time point
 * `bioav`: bioavailability in each compartment
 * `kwargs`: stores the provided keyword arguments
"""
struct AnalyticalPKPDProblem{uType,tType,isinplace,F,EV,T,P,B,K} <: DiffEqBase.AbstractAnalyticalProblem{uType,tType,isinplace}
  f::F
  u0::uType
  tspan::Tuple{tType,tType}
  events::EV
  times::T
  p::P
  bioav::B
  kwargs::K
  DiffEqBase.@add_kwonly function AnalyticalPKPDProblem{iip}(f,u0,tspan,
                                      events, times,
                                      p=DiffEqBase.NullParameters(),
                                      bioav = 1f0;
                                      kwargs...) where {iip}
    new{typeof(u0),promote_type(map(typeof,tspan)...),iip,
    typeof(f),typeof(events),typeof(times),typeof(p),typeof(bioav),
    typeof(kwargs)}(f,u0,tspan,events,times,p,bioav,kwargs)
  end
end

function AnalyticalPKPDProblem(f,u0,tspan,args...;kwargs...)
  iip = DiffEqBase.isinplace(f,7)
  AnalyticalPKPDProblem{iip}(f,u0,tspan,args...;kwargs...)
end

"""
  AnalyticalPKProblem

A problem the is partially an analytical problem that can be evaluated independently of the rest of the system.

Fields:
 * `pkprob`: the analytical part of the problem
 * `prob2`:  a problem that represents the rest of the system
"""
struct AnalyticalPKProblem{P1<:ExplicitModel,P2}
  pkprob::P1
  prob2::P2
end

struct PresetAnalyticalPKProblem{P,PK}
  numprob::P
  pksol::PK
end

struct NullDEProblem{P} <: DiffEqBase.DEProblem
  p::P
end

Base.summary(prob::NullDEProblem) = string(DiffEqBase.TYPE_COLOR, nameof(typeof(prob)),
                                                   DiffEqBase.NO_COLOR)

function Base.show(io::IO, A::NullDEProblem)
  println(io,summary(A.p))
  println(io)
end

export AnalyticalPKPDProblem, AnalyticalPKProblem
