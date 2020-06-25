struct SimulatedObservations{S,T,T2}
  subject::S
  times::T
  observed::T2
end

# indexing
@inline function Base.getindex(obs::SimulatedObservations, I...)
  return obs.observed[I...]
end
@inline function Base.setindex!(obs::SimulatedObservations, x, I...)
  obs.observed[I...] = x
end

# Convert to Subject
function Subject(simsubject::SimulatedObservations)
  dvnames = keys(simsubject.subject.observations)
  covariates = simsubject.subject.covariates
  covartime = simsubject.subject.covartime
  subject = Subject(;
    id         = simsubject.subject.id,
    obs        = NamedTuple{dvnames}(map(k -> simsubject.observed[k], dvnames)),
    evs        = simsubject.subject.events,
    time       = simsubject.times,
    event_data = !isnothing(simsubject.subject.events),
    covariates = covariates,
    covartime = covartime)
  return subject
end

# DataFrame conversion
function DataFrames.DataFrame(
  obs::SimulatedObservations;
  include_events=!isempty(obs.subject.events),
  include_covariates=true)

  nrows = length(obs.times)
  events = obs.subject.events
  nev = events isa Array ? length(events) : 0
  evtimes = map(ev->ev.time, events)
  times = obs.times
  ntime = length(obs.times)
  observed = obs.observed
  df = DataFrame(time=deepcopy(times))

  for k in keys(observed)
    var = observed[k]
    lenvar = length(var)
    if lenvar != ntime && lenvar == nev # NCA quantities
      # pad NCA quantities
      i = 0
      var = map(times) do t
        i1 = min(i+1, nev)
        t >= evtimes[i1] && (i = i1)
        var[i1]
      end
    end
    df[!,k] .= deepcopy(var)
  end
  obs_columns = [keys(obs.observed)...]

  # Allow all dv columns to be missing
  for cols in obs_columns
    allowmissing!(df, cols)
  end

  if include_events
    # Append event columns
    ## For observations we have `evid=0` and `cmt=0`, the latter
    ## subject to changes in the future
    df[!,:amt]  = zeros(typeof(events[1].amt),  nrows)
    df[!,:evid] = zeros(typeof(events[1].evid), nrows)
    df[!,:cmt]  = missings(typeof(events[1].cmt), nrows)
    df[!,:rate] = zeros(typeof(events[1].rate), nrows)
    # Add rows corresponding to the events
    ## For events that have a matching observation at the event
    ## time, the values of the derived variables set to missing.
    ## Otherwise they are set to `missing`.
    for ev in events
      ind = searchsortedlast(obs.times, ev.time)
      if ind != 0 && obs.times[ind] == ev.time
        ev_row = vcat(ev.time, fill(missing, length(obs_columns))...,
                      ev.amt, ev.evid, ev.cmt, ev.rate)
        push!(df, ev_row)
      else
        ev_row = vcat(ev.time, missings(length(obs_columns)),
                      ev.amt, ev.evid, ev.cmt, ev.rate)
        push!(df, ev_row)
      end
    end

    # Remove negative evid's
    df = filter(i -> i.evid != -1, df)
    sort!(df, [:time, order(:evid, rev=true)])
  end
  if include_covariates
    df = _add_covariates(df, obs.subject)
    if "evid" ∈ names(df) && !include_events
      df = select!(df, Not(:evid))
    end
    if "id" ∈ names(df)
      df = select!(df, Not(:id))
    end
  end

  insertcols!(df, 1, :id => fill(obs.subject.id, size(df, 1)))
  return df
end

@recipe function f(obs::SimulatedObservations; obsnames=nothing)
  t = obs.times
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.observed)
    if obsnames !== nothing
      !(n in obsnames) && continue
    end
    push!(names,n)
    push!(plot_vars,v)
  end
  xguide --> "time"
  legend --> false
  linewidth --> 3
  yguide --> reshape(names,1,length(names))
  layout --> good_layout(length(names))
  title --> "Subject ID: $(obs.subject.id)"
  t,plot_vars
end

function good_layout(n)
  n == 1 && return (1,1)
  n == 2 && return (2,1)
  n == 3 && return (3,1)
  n == 4 && return (2,2)
  n > 4  && return n
end

const SimulatedPopulation{T} = AbstractVector{T} where T<:SimulatedObservations
function DataFrames.DataFrame(pop::SimulatedPopulation; kwargs...)
  dfs = []
  for s in pop
    df = DataFrame(s; kwargs...)
    push!(dfs, df)
  end
  reduce(vcat,dfs)
end

@recipe function f(pop::SimulatedPopulation)
  for p in pop
    @series begin
      linewidth --> 1.5
      title --> "Population Simulation"
      p
    end
  end
  nothing
end
