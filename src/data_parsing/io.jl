## Types
include("interpolation.jl")
"""
    Event

A single event

Fields:
 * `amt`: Dose amount (mass units, e.g. g, mg, etc.)
 * `time`: Time of dose (time unit, e.g. hours, days, etc.)
 * `evid`: Event identifier, possible values are
   - `-1`: End of an infusion dose
   - `0`: observation (these should have already been removed)
   - `1`: dose event
   - `2`: other type event
   - `3`: reset event (the amounts in each compartment are reset to zero, the on/off status of each compartment is reset to its initial status)
   - `4`: reset and dose event
 * `cmt`: Compartment (an Int or Symbol)
 * `rate`: The dosing rate:
   - `0`: instantaneous bolus dose
   - `>0`: infusion dose (mass/time units, e.g. mg/hr)
   - `-1`: infusion rate specifed by model
   - `-2`: infusion duration specified by model
 * `duration`: duration of dose, if `dose > 0`.
 * `ss`: steady state status
   - `0`: dose is not a steady state dose.
   - `1`: dose is a steady state dose, and that the compartment amounts are to be reset to the steady-state amounts resulting from the given dose. Compartment amounts resulting from prior dose event records are "zeroed out," and infusions in progress or pending additional doses are cancelled.
   - `2`: dose is a steady state dose and that the compartment amounts are to be set to the sum of the steady-state amounts resulting from the given dose plus whatever those amounts would have been at the event time were the steady-state dose not given.
 * `ii`: interdose interval (units: time). Time between implied or additional doses.
 * `base_time`: time at which infusion started
 * `rate_dir`: direction of rate
   - `-1` for end-of-infusion
   - `+1` for any other doses
"""
struct Event{T1,T2,T3,T4,T5,T6,C}
  amt::T1
  time::T2
  evid::Int8
  cmt::C
  rate::T3
  duration::T4
  ss::Int8
  ii::T5
  base_time::T6 # So that this is kept after modifications to duration and rate
  rate_dir::Int8
end

Event(amt, time, evid::Integer, cmt::Union{Int,Symbol}) =
  Event(amt, time, Int8(evid), cmt, zero(amt), zero(amt),
        Int8(0), zero(time), time, Int8(1))

Base.isless(a::Event,b::Event) = isless(a.time,b.time)
Base.isless(a::Event,b::Number) = isless(a.time,b)
Base.isless(a::Number,b::Event) = isless(a,b.time)

### Display
_evid_index = Dict(-1 => "End of infusion", 0 => "Observation", 1 => "Dose",
                   2 => "Other", 3 => "Reset", 4 => "Reset and dose")
Base.summary(e::Event) = "$(_evid_index[e.evid]) event"
function Base.show(io::IO, e::Event)
  println(io, summary(e))
  println(io, "  dose amount = $(e.amt)")
  println(io, "  dose time = $(e.time)")
  println(io, "  compartment = $(e.cmt)")
  if e.rate > zero(e.rate)
    println(io, "  rate = $(e.rate)")
    println(io, "  duration = $(e.duration)")
  elseif e.rate == zero(e.rate)
    println(io, "  instantaneous")
  elseif e.rate == -oneunit(e.rate)
    println(io, "  rate specified by model")
  elseif e.rate == -2oneunit(e.rate)
    println(io, "  duration specified by model")
  end
  if e.ss == 1
    println(io, "  steady state dose")
  elseif e.ss == 2
    println(io, "  steady state dose, no reset")
  end
  println(io, "  interdose interval = $(e.ii)")
  println(io, "  infusion start time = $(e.base_time)")
  if e.rate_dir == -1
    println(io, "  end of infusion")
  end
end
TreeViews.hastreeview(::Event) = true
function TreeViews.treelabel(io::IO, e::Event, mime::MIME"text/plain")
  show(io, mime, Text(summary(e)))
end

"""
    DosageRegimen
Lazy representation of a series of Events.
"""
mutable struct DosageRegimen
  data::DataFrame
  function DosageRegimen(amt::Number,
                         time::Number,
                         cmt::Union{Number,Symbol},
                         evid::Number,
                         ii::Number,
                         addl::Number,
                         rate::Number,
                         duration::Number,
                         ss::Number)
    # amt = evid ∈ [0, 3] ? float(zero(amt)) : float(amt)
    amt = float(amt)
    time = float(time)
    cmt = cmt isa Symbol ? cmt : convert(Int, cmt)
    evid = convert(Int8, evid)
    ii = float(ii)
    addl = convert(Int, addl)
    rate = float(rate)
    ss = convert(Int8, ss)
    evid ∈ 1:4 || throw(ArgumentError("evid must be a valid event type"))
    if evid ∈ [0, 3]
      iszero(amt) || throw(ArgumentError("amt must be 0 for evid = $evid"))
    elseif iszero(amt) && rate > 0 && duration > 0
      amt = rate * duration
    else
      amt > zero(amt) || throw(ArgumentError("amt must be positive for evid = $evid"))
    end
    time ≥ zero(time) || throw(ArgumentError("time must be non-negative"))
    evid == 0 && throw(ArgumentError("observations are not allowed"))
    ii ≥ zero(ii) || throw(ArgumentError("ii must be non-negative"))
    addl ≥ 0 || throw(ArgumentError("addl must be non-negative"))
    addl > 0          && ii   == zero(time) && throw(ArgumentError("ii must be positive for addl > 0"))
    ii   > zero(time) && addl == zero(addl) && ss == 0 && throw(ArgumentError("addl must be positive for ii > 0 and ss = 0"))
    rate ≥ zero(rate) || rate == -2         || throw(ArgumentError("rate is invalid"))
    ss ∈ 0:2 || throw(ArgumentError("ss is invalid"))
    if iszero(duration) && amt > zero(amt) && rate > zero(rate)
      duration = amt / rate
    elseif iszero(rate) && amt > zero(amt) && duration > zero(rate)
      rate = amt / duration
    elseif duration > zero(duration) && rate > zero(rate)
      @assert amt ≈ rate * duration
    end
    new(DataFrame(time = time, cmt = cmt, amt = amt, evid = evid, ii = ii, addl = addl,
                  rate = rate, duration = duration, ss = ss))
  end
  DosageRegimen(amt::Numeric;
                time::Numeric = 0,
                cmt::Union{Numeric,Symbol} = 1,
                evid::Numeric = 1,
                ii::Numeric = zero.(time),
                addl::Numeric = 0,
                rate::Numeric = zero.(amt)./oneunit.(time),
                duration::Numeric = zero(amt)./oneunit.(time),
                ss::Numeric = 0) =
  DosageRegimen(DosageRegimen.(amt, time, cmt, evid, ii, addl, rate, duration, ss))
  DosageRegimen(regimen::DosageRegimen) = regimen
  function DosageRegimen(regimen1::DosageRegimen,
                         regimen2::DosageRegimen;
                         offset = nothing)
    data1 = regimen1.data
    data2 = regimen2.data
    if isnothing(offset)
      output = sort!(vcat(data1, data2), :time)
    else
      data2 = deepcopy(data2)
      data2[!,:time] = cumsum(vcat(data1[!,:ii][end] * (data1[!,:addl][end] + 1) + data1[!,:time][end] + offset, data2[!,:time]))[2:end]
      output = sort!(vcat(data1, data2), :time)
    end
    new(output)
  end
  DosageRegimen(regimens::AbstractVector{<:DosageRegimen}) = reduce(DosageRegimen, regimens)
  DosageRegimen(regimens::DosageRegimen...) = reduce(DosageRegimen, regimens)
end
"""
  DataFrame(evs::DosageRegimen, expand::Bool = false)

  Create a DataFrame with the information in the dosage regimen.
  If expand, creates a DataFrame with the information in the event list (expanded form).
"""
function DataFrames.DataFrame(evs::DosageRegimen, expand::Bool = false)
  if !expand
    return evs.data
  else
    evs = build_event_list(evs, true)
    output = DataFrame(fill(Float64, 10),
      [:amt, :time, :evid, :cmt, :rate, :duration, :ss, :ii, :base_time, :rate_dir],
      length(evs)
    )
    for col ∈ [:amt, :time, :evid, :cmt, :rate, :duration, :ss, :ii, :base_time, :rate_dir]
      output[!,col] .= getproperty.(evs, col)
    end
    sort!(output, [:time])
    return output
  end
end

# show methods for DosageRegimen. We just want to inherit the show
# methods of DataFrame but, unfortunately, we can't just have a single
# definition with an abstract MIME argument since that makes IJulia's display
# functionality fail. Hence, we have to query all the specific show
# method that has been defined for DataFrames. It's not pretty but
# it works.
for MT in unique(map(m -> m.sig.parameters[3], methods(show, Tuple{IO, MIME, DataFrame})))
  @assert MT <: MIME
  if MT <: MIME"text/plain"
    @eval function Base.show(io::IO, mime::$MT, dr::DosageRegimen)
      print(io, summary(dr))
      show(io, mime, dr.data, summary=false)
    end
  else
    @eval Base.show(io::IO, mime::$MT, dr::DosageRegimen) = show(io, mime, dr.data)
  end
end

###
### Helper function for the Subject constructor
###
"""
  to_nt(obj)::NamedTuple{PN,VT}
It returns a NamedTuple based on the propertynames of the object.
If a value is a vector with a single value, it returns the value.
If the vector has no missing values, it is promoted through disallowmissing.
"""
to_nt(obj::Any) = propertynames(obj) |>
  (x -> NamedTuple{Tuple(x)}(
    getproperty(obj, x) |>
    (x -> isone(length(unique(x))) ?
          first(x) :
          x)
    for x ∈ x))

function build_observation_list(obs::AbstractDataFrame)
  #cmt = :cmt ∈ names(obs) ? obs[:cmt] : 1
  # Once we require DataFrames 0.21 we can remove the Symbol conversion and use "time", and "cmt"
  vars = setdiff(Symbol.(names(obs)), (:time, :cmt))
  return (; [v => convert(AbstractVector{Union{Missing,Float64}}, obs[!, v]) for v in vars]...)
end
build_observation_list(obs::NamedTuple) = obs
build_observation_list(obs::Nothing) = obs

build_event_list(evs::AbstractVector{<:Event}, event_data::Bool) = evs
function build_event_list!(events, event_data, t, evid, amt, addl, ii, cmt, rate, ss)
  @assert evid ∈ 0:4 "evid must be in 0:4"
  # Dose-related data items
  drdi = iszero(amt) && (rate == 0) && iszero(ii) && iszero(addl) && iszero(ss)
  if event_data
    if evid ∈ [0, 2, 3]
      @assert drdi "Dose-related data items must be zero when evid = $evid"
    else
      @assert !drdi "Some dose-related data items must be non-zero when evid = $evid"
    end
  end
  duration = amt / rate
  for j = 0:addl  # addl==0 means just once
    _ss = iszero(j) ? ss : zero(Int8)
    if iszero(amt) && evid ≠ 2
      # These are dose events having AMT=0, RATE>0, SS=1, and II=0.
      # Such an event consists of infusion with the stated rate,
      # starting at time −∞, and ending at the time on the dose
      # ev event record. Bioavailability fractions do not apply
      # to these doses.
      push!(events, Event(amt, t, evid, cmt, rate, ii, _ss, ii, t, Int8(1)))
    else
      push!(events, Event(amt, t, evid, cmt, rate, duration, _ss, ii, t, Int8(1)))
      if !iszero(rate) && iszero(_ss)
        push!(events, Event(amt, t + duration, Int8(-1), cmt, rate, duration, _ss, ii, t, Int8(-1)))
      end
    end
    t += ii
  end
end
function build_event_list(regimen::DosageRegimen, event_data::Bool)
  data = regimen.data
  events = Event[]
  for i in 1:size(data, 1)
    t    = data[!,:time][i]
    evid = data[!,:evid][i]
    amt  = data[!,:amt][i]
    addl = data[!,:addl][i]
    ii   = data[!,:ii][i]
    cmt  = data[!,:cmt][i]
    rate = data[!,:rate][i]
    ss   = data[!,:ss][i]
    build_event_list!(events, event_data, t, evid, amt, addl, ii, cmt, rate, ss)
  end
  sort!(events)
end

###
### Subject conctructors
###
"""
    Subject

The data corresponding to a single subject:

Fields:
- `id`: identifier
- `observations`: a NamedTuple of the dependent variables
- `covariates`: a named tuple containing the covariates, or `nothing`.
- `events`: a vector of `Event`s.
- `time`: a vector of time stamps for the observations
"""
struct Subject{T1,T2,T3,T4,T5,T6}
  id::String
  observations::T1
  covariates::T2
  events::T3
  time::T4
  tvcov::T5
  covartime::T6
  function Subject(df::AbstractDataFrame,
                   id, time, evid, amt, addl, ii, cmt, rate, ss,
                   cvs::Vector{<:Symbol} = Symbol[],
                   dvs::Vector{<:Symbol} = Symbol[:dv],
                   event_data = true)
    ## Observations
    idx_obs = findall(iszero, df[!,evid])
    obs_times = Missings.disallowmissing(df[!,time][idx_obs])
    @assert issorted(obs_times) "Time is not monotonically increasing within subject"
    if isa(obs_times, Unitful.Time)
      _obs_times = convert.(Float64, getfield(uconvert.(u"hr", obs_times), :val))
    else
      _obs_times = float(obs_times)
    end

    dv_idx_tuple = ntuple(i -> convert(AbstractVector{Union{Missing,Float64}},
                                       df[!,dvs[i]][idx_obs]),
                                       length(dvs))
    observations = NamedTuple{tuple(dvs...),typeof(dv_idx_tuple)}(dv_idx_tuple)

    # cmt handling should be reversed: it should give it the appropriate name given cmt
    # obs_cmts = :cmt ∈ colnames ? df[:cmt][idx_obs] : nothing

    #==
      Covariates
      We first build individual time/covar pairs. If there are no time varying
      covariates, we return no-ops. Else, we build individual interpolations,
      interpolate to the total covariate time grid, and then we build a multi-
      valued interpolant. The idea is to only do searchsorted once. We also
      use the union of covar times to specify d_discontinuities for diffeq
      based solvers, and to split up the analytical solution.
    ==#

    # build individual interpolants and use them to create a batch interpolant
    ## Pass the ID to build_tvcov to make error message more informative
    _id_string = string(first(df[!,id]))
    covar_times, tvcov = build_tvcov(cvs, df, time, _id_string)

    ## FIXME we still keep the old covar
    covariates = isempty(cvs) ? nothing : to_nt(df[!, cvs])
    ## Events
    idx_evt = setdiff(1:size(df, 1), idx_obs)

    n_amt = hasproperty(df, amt)
    n_addl = hasproperty(df, addl)
    n_ii = hasproperty(df, ii)
    n_cmt = hasproperty(df, cmt)
    n_rate = hasproperty(df, rate)
    n_ss = hasproperty(df, ss)
    cmtType = n_cmt ? (eltype(df[!,cmt]) <: String ? Symbol : Int) : Int
    events = Event{Float64,Float64,Float64,Float64,Float64,Float64,cmtType}[]
    for i in idx_evt
      t     = float(df[!,time][i])
      _evid = Int8(df[!,evid][i])
      _amt  = n_amt ? float(df[!,amt][i])   : 0. # can be missing if evid=2
      _addl = n_addl ? Int(df[!,addl][i])   : 0
      _ii   = n_ii ? float(df[!,ii][i])     : zero(t)
      __cmt  = n_cmt ? df[!,cmt][i] : 1
      _cmt = __cmt isa String ? Symbol(__cmt) : Int(__cmt)
      _rate = n_rate ? float(df[!,rate][i]) : _amt === nothing ? 0.0 : zero(_amt)/oneunit(t)
      ss′   = n_ss ? Int8(df[!,ss][i])      : Int8(0)
      build_event_list!(events, event_data, t, _evid, _amt, _addl, _ii, _cmt, _rate, ss′)
    end
    sort!(events)
    new{typeof(observations),typeof(covariates),typeof(events),typeof(_obs_times), typeof(tvcov), typeof(covar_times)}(_id_string, observations, covariates, events, _obs_times, tvcov, covar_times)
  end

  function Subject(;id = "1",
                   obs = nothing,
                   cvs = nothing,
                   cvstime = obs isa AbstractDataFrame ? obs.time : nothing,
                   evs = Event[],
                   time = obs isa AbstractDataFrame ? obs.time : nothing,
                   event_data = true,)
    obs = build_observation_list(obs)
    evs = build_event_list(evs, event_data)
    covar_times, tvcov = build_tvcov(cvs, cvstime, id)
    # Check that time is well-specified (not nothing, not missing and increasing)
    _time = isnothing(time) ? nothing : Missings.disallowmissing(time)
    @assert isnothing(time) || issorted(_time) "Time is not monotonically increasing within subject"

    new{typeof(obs),typeof(cvs),typeof(evs),typeof(_time), typeof(tvcov), typeof(covar_times)}(string(id), obs, cvs, evs, _time, tvcov, covar_times)
  end
end

Base.hash(subject::Subject, h::UInt) = hash(
  subject.time, hash(
    subject.events, hash(
      subject.covariates, hash(
        subject.observations, hash(
          subject.id, h)))))

Base.:(==)(subject1::Subject, subject2::Subject) = hash(subject1) == hash(subject2)

function DataFrames.DataFrame(subject::Subject; include_covariates=true, include_dvs=true, include_events=true)

  # Build a DataFrame that holds the events
  df_events = DataFrame(build_event_list(subject.events, true))
  # Remove events with evid==-1
  df_events = df_events[df_events[!, :evid].!=-1,:]

  # We delete rate/duration if no infusions. Else, we delete duration or rate
  # as appropriate. TODO This will actually fail if used in the context of a
  # Population where some Subjects have duration specified and others have
  # rate specified.
  if all(x->iszero(x), df_events[Not(ismissing.(df_events[!, :rate])),:rate])
    # There are no infusions
    select!(df_events, Not(:rate))
    select!(df_events, Not(:duration))
  elseif all(x->x>zero(x), df_events[Not(ismissing.(df_events[!, :rate])),:rate])
    # There are infusions, and they're specified by rate
    select!(df_events, Not(:duration))
  else
    # There are infusions, and they're specified by duration
    select!(df_events, Not(:rate))
  end

  # Generate the name for the dependent variable in a manner consistent with
  # multiple dvs etc
  if isnothing(subject.time)
    df_events = hcat(DataFrame(id = fill(subject.id, length(df_events.evid))), df_events)
  else
    df = DataFrame(id = fill(subject.id, length(subject.time)), time=subject.time)
    df[!, :evid] .= 0
    # Only include the dv columns if include_dvs is specified and there are
    # observations.
    if include_dvs && !isnothing(subject.observations)
      # Loop over all dv's
      for (dv_name, dv_vals) in pairs(subject.observations)
        df[!, dv_name] .= dv_vals
      end
    end
    # Now, create columns in df_events with missings for the column names in
    # df but not in df_events
    if include_dvs
      for df_name in Symbol.(names(df))
        if df_name == :id
          df_events[!, df_name] .= subject.id
        elseif !(df_name == :time)
          if !hasproperty(df_events, df_name)
            df_events[!, df_name] .= missing
          end
        end
      end
      # ... and do the same for df
      if include_events
        for df_name in names(df_events)
          if !hasproperty(df, df_name)
            df[!, df_name] .= missing
          end
        end
      end
    end
  end

  # If there are no observations, just go with df_events
  if isnothing(subject.time)
    df = df_events
  elseif include_dvs && include_events
    df = vcat(df, df_events)
  else
    df
  end

  include_covariates && _add_covariates!(df, subject)

  # Sort the df according to time first, and use :base_time to ensure that events
  # come before observations (they are missing for observations, so they will come
  # last).
  sort_bys = include_dvs && include_events ? [:time, :base_time] : [:time,]
  sort!(df, sort_bys)
  # Find the amount (amt) column, and insert dose and tad columns after it
  if include_dvs && include_events
    amt_pos = findfirst(isequal(:amt), Symbol.(names(df)))
    insertcols!(df, amt_pos+1, :dose => 0.0)
    insertcols!(df, amt_pos+2, :tad => 0.0)
    # Calculate the indeces for the dose events
    idxs = findall(isequal(false), ismissing.(df[!, :amt]))
    # Append the number of rows to allow the +1 indexing used below
    if !(idxs[end] == nrow(df))
      push!(idxs, nrow(df))
    end
    for i in 1:length(idxs)-1
      df[idxs[i]:idxs[i+1], :dose] .= df[idxs[i], :amt]
      df[idxs[i]:idxs[i+1], :cmt] .= df[idxs[i], :cmt]
      df[idxs[i]:idxs[i+1], :tad] .= df[idxs[i]:idxs[i+1], :time].-df[idxs[i], :time]
    end
  end

  # Return df
  df
end

function _add_covariates!(df::DataFrame, subject::Subject)
  covariates = subject.covariates
  if !isa(covariates, Nothing)
    for (covariate, value) in pairs(covariates)
      df[!,covariate] .= value
    end
  end
end

### Display
Base.summary(::Subject) = "Subject"
function Base.show(io::IO, subject::Subject)
  println(io, summary(subject))
  print(io, string("  ID: $(subject.id)"))
  evs = subject.events
  isnothing(evs) || print(io, "\n  Events: ", length(subject.events))
  obs = subject.observations
  observables = propertynames(obs)
  if !isempty(observables)
    vals = mapreduce(pn -> string(pn, ": (n=$(length(getindex(obs, pn))))"),
                     (x, y) -> "$x, $y",
                     observables)
    print(io, "\n  Observables: $vals")
  end
  if subject.covariates != nothing
    if length(subject.covariates) > 10
      print(io, string("\n  Too many Covariates to display. Run DataFrame(Subject) to see the Covariates. "))
    else
      print(io, string("\n  Covariates: $(subject.covariates)"))
    end
  end
end
TreeViews.hastreeview(::Subject) = true
function TreeViews.treelabel(io::IO, subject::Subject, mime::MIME"text/plain")
  show(io, mime, Text(summary(subject)))
end

@recipe function f(obs::Subject; obsnames=nothing)
  t = obs.time
  names = Symbol[]
  plot_vars = []
  for (n,v) in pairs(obs.observations)
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
  title --> "Subject ID: $(obs.id)"
  t,plot_vars
end

# Define Population as an alias
"""
A `Population` is an `AbstractVector` of `Subject`s.
"""
const Population{T} = AbstractVector{T} where T<:Subject
Population(obj::Population...) = reduce(vcat, obj)::Population

function DataFrames.DataFrame(pop::Population; include_covariates=true, include_dvs=true, include_events=true)
  vcat((DataFrame(subject; include_covariates=include_covariates, include_dvs=include_dvs, include_events=include_events) for subject in pop)...)
end

### Display
Base.summary(::Population) = "Population"
function Base.show(io::IO, ::MIME"text/plain", population::Population)
  println(io, summary(population))
  print(io, "  Subjects: ", length(population))
  if isassigned(population, 1)
    co = population[1].covariates
    !isnothing(co) && print(io, "\n  Covariates: ", join(fieldnames(typeof(co)),", "))
    obs = population[1].observations
    !isnothing(obs) && print(io, "\n  Observables: ", join(keys(obs),", "))
  end
  return nothing
end
TreeViews.hastreeview(::Population) = true
TreeViews.numberofnodes(population::Population) = length(population)
TreeViews.treenode(population::Population, i::Integer) = getindex(population, i)
function TreeViews.treelabel(io::IO, population::Population, mime::MIME"text/plain")
  show(io, mime, Text(summary(population)))
end
TreeViews.nodelabel(io::IO, population::Population, i::Integer, mime::MIME"text/plain") = show(io, mime, Text(population[i].id))

@recipe function f(pop::Population)
  for p in pop
    @series begin
      linewidth --> 1.5
      title --> "Population Simulation"
      p
    end
  end
  nothing
end

# Convert to NCA types
import .NCA: NCAPopulation, NCASubject, NCADose
using .NCA: Formulation, IVBolus, IVInfusion, EV
function Base.convert(::Type{NCADose}, ev::Event)
  (ev.evid == Int8(0) || ev.evid == Int8(2) || ev.evid == Int8(3)) && return nothing
  time = ev.time
  amt = ev.amt
  duration = isinf(ev.duration) ? zero(ev.duration) : ev.duration
  NCADose(time, amt, duration, IVBolus) # FIXME: when is an event extravascular?
end
NCADose(dose::Event) = convert(NCADose, dose)

function Base.convert(::Type{NCASubject}, subj::Subject; name=:dv, kwargs...)
  dose = convert.(NCADose, subj.events)
  ii = subj.events[end].ii
  if subj.observations === nothing || subj.time === nothing
    return NCASubject(
      Float64[],
      Float64[];
      id=subj.id,
      dose=nothing,
      clean=false,
      ii=ii,
      kwargs...)
  else
    return NCASubject(
      subj.observations[name],
      subj.time;
      clean=true,
      id=subj.id,
      dose=dose,
      ii=ii,
      kwargs...)
  end
end
NCASubject(subj::Subject; name=:dv) = convert(NCASubject, subj; name=name)

Base.convert(::Type{NCAPopulation}, population::Population; name=:dv, kwargs...) =
  map(subject -> convert(NCASubject, subject; name=name, kwargs...), population)
(::Type{NCAPopulation})(population::AbstractVector{T}; name=:dv, kwargs...) where T<:Subject = convert(NCAPopulation, population; name=name, kwargs...)

"""
    read_pumas(filepath::String, args...; kwargs...)
    read_pumas(data, cvs=Symbol[], dvs=Symbol[:dv];
                   id=:id, time=:time, evid=:evid, amt=:amt, addl=:addl,
                   ii=:ii, cmt=:cmt, rate=:rate, ss=:ss,
                   event_data = true)

Import NMTRAN-formatted data.

- `cvs` covariates specified by either names or column numbers
- `dvs` dependent variables specified by either names or column numbers
- `event_data` toggles assertions applicable to event data
"""
function read_pumas(filepath::AbstractString; kwargs...)
  read_pumas(CSV.read(filepath, missingstrings=["."]) ; kwargs...)
end
function read_pumas(df::AbstractDataFrame;
  cvs=Symbol[], dvs=Symbol[:dv], id=:id, time=:time, evid=:evid, amt=:amt, addl=:addl,
  ii=:ii, cmt=:cmt, rate=:rate, ss=:ss, mdv=:mdv, event_data = true)

  df = copy(df)
  colnames = names(df)

  if !hasproperty(df, id)
    df[!,id] .= "1"
  end
  if !hasproperty(df, time)
    df[!,time] .= 0.0
  end
  if !hasproperty(df, evid)
    df[!,evid] .= Int8(0)
  end
  if !hasproperty(df, mdv)
    df[!,mdv] .= Int8(0)
  end
  if cvs isa AbstractVector{<:Integer}
    Base.depwarn("selecting covariate columns by integer indexing has been deprecated. Please specify the name of the columns instead.", :read_pumas)
    cvs = Symbol.(colnames[cvs])
  end
  if dvs isa AbstractVector{<:Integer}
    Base.depwarn("selecting dependent variable columns by integer indexing has been deprecated. Please specify the name of the columns instead.", :read_pumas)
    dvs = Symbol.(colnames[dvs])
  end

  # We allow specifying missing values in of the dependent variable with an mdv column
  # but internally we encode missing values with missing directly in the dv columns
  allowmissing!(df, dvs)
  mdv = isone.(df[!,mdv])
  for dv in dvs
    df[!, dv] .= ifelse.(mdv, missing, df[!, dv])
  end

  return [Subject(subject_df, id, time, evid, amt, addl, ii, cmt,
                  rate, ss, cvs, dvs, event_data) for subject_df in groupby(df, id)]
end

