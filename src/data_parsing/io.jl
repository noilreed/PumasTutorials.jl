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

# Fields

- `data::DataFrame`: The tabular representation of a series of `Event`s.

- Signature

```
evts = DosageRegimen(amt::Numeric;
                     time::Numeric = 0,
                     cmt::Union{Numeric,Symbol} = 1,
                     evid::Numeric = 1,
                     ii::Numeric = zero.(time),
                     addl::Numeric = 0,
                     rate::Numeric = zero.(amt)./oneunit.(time),
                     duration::Numeric = zero(amt)./oneunit.(time),
                     ss::Numeric = 0)
```

- Examples

```jldoctest
julia> DosageRegimen(100, ii = 24, addl = 6)
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 100.0   │ 1    │ 24.0    │ 6     │ 0.0     │ 0.0      │ 0    │

julia> DosageRegimen(50,  ii = 12, addl = 13)
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 50.0    │ 1    │ 12.0    │ 13    │ 0.0     │ 0.0      │ 0    │

julia> DosageRegimen(200, ii = 24, addl = 2)
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 200.0   │ 1    │ 24.0    │ 2     │ 0.0     │ 0.0      │ 0    │
```

## From various `DosageRegimen`s

- Signature

evs = DosageRegimen(regimen1::DosageRegimen,
                    regimen2::DosageRegimen;
                    offset = nothing)

`offset` specifies if `regimen2` should start after an offset following the end of the last event in `regimen1`.

- Examples

```jldoctest
julia> e1 = DosageRegimen(100, ii = 24, addl = 6)
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 100.0   │ 1    │ 24.0    │ 6     │ 0.0     │ 0.0      │ 0    │

julia> e2 = DosageRegimen(50, ii = 12, addl = 13)
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 50.0    │ 1    │ 12.0    │ 13    │ 0.0     │ 0.0      │ 0    │

julia> evs = DosageRegimen(e1, e2)
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 100.0   │ 1    │ 24.0    │ 6     │ 0.0     │ 0.0      │ 0    │
│ 2   │ 0.0     │ 1     │ 50.0    │ 1    │ 12.0    │ 13    │ 0.0     │ 0.0      │ 0    │

julia> DosageRegimen(e1, e2, offset = 10)
DosageRegimen
│ Row │ time    │ cmt   │ amt     │ evid │ ii      │ addl  │ rate    │ duration │ ss   │
│     │ Float64 │ Int64 │ Float64 │ Int8 │ Float64 │ Int64 │ Float64 │ Float64  │ Int8 │
├─────┼─────────┼───────┼─────────┼──────┼─────────┼───────┼─────────┼──────────┼──────┤
│ 1   │ 0.0     │ 1     │ 100.0   │ 1    │ 24.0    │ 6     │ 0.0     │ 0.0      │ 0    │
│ 2   │ 178.0   │ 1     │ 50.0    │ 1    │ 12.0    │ 13    │ 0.0     │ 0.0      │ 0    │

```
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
    DataFrame(events::DosageRegimen, expand::Bool = false)

Create a DataFrame with the information in the dosage regimen.
If expand, creates a DataFrame with the information in the event list (expanded form).
"""
function DataFrames.DataFrame(events::DosageRegimen, expand::Bool = false)
  if !expand
    return events.data
  else
    events = build_event_list(events, true)
    output = DataFrame(fill(Float64, 10),
      [:amt, :time, :evid, :cmt, :rate, :duration, :ss, :ii, :base_time, :rate_dir],
      length(events)
    )
    for col ∈ [:amt, :time, :evid, :cmt, :rate, :duration, :ss, :ii, :base_time, :rate_dir]
      output[!,col] .= getproperty.(events, col)
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
# to_nt(obj)::NamedTuple{PN,VT}
# It returns a NamedTuple based on the propertynames of the object.
# If a value is a vector with a single value, it returns the value.
# If the vector has no missing values, it is promoted through disallowmissing.
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

build_event_list(events::AbstractVector{<:Event}, event_data::Bool) = events
function build_event_list!(events::Vector{<:Event}, event_data::Bool, t, evid, amt, addl, ii, cmt, rate, ss)
  @assert evid ∈ 0:4 "evid must be in 0:4"
  # Dose-related data items
  drdi = iszero(amt) && (rate == 0) && iszero(ii) && iszero(addl) && iszero(ss)
  if event_data
    if evid ∈ [0, 2, 3]
      if !drdi
        throw(PumasDataError("Dose-related data items must be zero when evid = $evid"))
      end
    else
      if drdi
        throw(PumasDataError("Some dose-related data items must be non-zero when evid = $evid"))
      end
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

When there are time varying covariates, each covariate is interpolated
with a common covariate time support. The interpolated values are then
used to build a multi-valued interpolant for the complete time support.
From the multi-valued interpolant, certain discontinuities are flagged
in order to use that information for the differential equation solvers
and to correctly apply the analytical solution per region as applicable.

Constructor

    Subject(;id = "1",
             observations = nothing,
             events = Event[],
             time = observations isa AbstractDataFrame ? observations.time : nothing,
             event_data = true,
             covariates::Union{Nothing, NamedTuple} = nothing,
             covariates_time = observations isa AbstractDataFrame ? observations.time : nothing,
             covariates_direction = :right)

`Subject` may be constructed from an `<:AbstractDataFrame` with the appropriate schema
or by providing the arguments directly through separate `DataFrames` / structures.

Examples:

```jldoctest
julia> Subject()
Subject
  ID: 1
  Events: 0

julia> data = read_pumas(example_data("event_data/data1")) # Subjects created implicitly
Population
  Subjects: 1
  Observables: dv

julia> Subject(id = 20, events = DosageRegimen(200, ii = 24, addl = 2), covariates = (WT = 14.2, HT = 5.2))
Subject
  ID: 20
  Events: 3
  Covariates: WT, HT

julia> Subject(covariates = (WT = [14.2, 14.7], HT = fill(5.2, 2)), covariates_time = [0, 3])
Subject
  ID: 1
  Events: 0
  Covariates: WT, HT

```
"""
struct Subject{T1,T2,T3,T4}
  id::String
  observations::T1
  covariates::T2
  events::T3
  time::T4
end

function Subject(
  df::AbstractDataFrame,
  id::Symbol,
  time::Symbol,
  evid::Symbol,
  amt::Symbol,
   addl::Symbol,
   ii::Symbol,
   cmt::Symbol,
   rate::Symbol,
   ss::Symbol,
  covariates::Vector{<:Symbol} = Symbol[],
  observations::Vector{<:Symbol} = Symbol[:dv],
  event_data::Bool=true,
  covariates_direction::Symbol=:right,
  parse_tad::Bool=true)

  ## Observations
  idx_obs = findall(iszero, df[!,evid])

  dv_idx_tuple = ntuple(i -> convert(AbstractVector{Union{Missing,Float64}},
    df[!,observations[i]][idx_obs]), length(observations))
  observations = NamedTuple{tuple(observations...),typeof(dv_idx_tuple)}(dv_idx_tuple)

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
  ## Pass the ID to covariates_interpolant to make error message more informative
  _id_string = string(first(df[!,id]))
  covartime, covariates_interpolation = covariates_interpolant(covariates, df, time, _id_string; covariates_direction=covariates_direction)

  n_amt = hasproperty(df, amt)
  n_addl = hasproperty(df, addl)
  n_ii = hasproperty(df, ii)
  n_cmt = hasproperty(df, cmt)
  n_rate = hasproperty(df, rate)
  n_ss = hasproperty(df, ss)
  if n_cmt
    Tcmt = eltype(df[!,cmt])
  end
  cmtType = n_cmt ? ((Tcmt <: String || Tcmt <: Symbol) ? Symbol : Int) : Int
  events = Event{Float64,Float64,Float64,Float64,Float64,Float64,cmtType}[]
  obstimes = Float64[]
  offset = 0.0

  for i in 1:size(df, 1)
    t     = float(df[!,time][i])
    _evid = Int8(df[!,evid][i])

    if i > 1 && t < df[i-1,time] && df[i-1,evid] != 3 && df[i-1,evid] != 4
      if t == 0
        # EVID=4 dose at t=0 means the reset should be happening at the
        # last observation time, so flip back to that value of t
        t = df[i-1,time]
      else
        throw(PumasDataError("Time is not monotonically increasing between reset dose events (evid=3 or evid=4)"))
      end
    end

    if _evid == 0 # observation, so add the time
      push!(obstimes,offset + t)
    else # event
      _amt  = n_amt ? float(df[!,amt][i])   : 0. # can be missing if evid=2
      _addl = n_addl ? Int(df[!,addl][i])   : 0
      _ii   = n_ii ? float(df[!,ii][i])     : zero(t)
      __cmt  = n_cmt ? df[!,cmt][i] : 1
      _cmt = __cmt isa Symbol ? __cmt : __cmt isa String ? Symbol(__cmt) : Int(__cmt)
      _rate = n_rate ? float(df[!,rate][i]) : _amt === nothing ? 0.0 : zero(_amt)/oneunit(t)
      ss′   = n_ss ? Int8(df[!,ss][i])      : Int8(0)
      build_event_list!(events, event_data, offset + t, _evid, _amt, _addl, _ii, _cmt, _rate, ss′)
      if parse_tad && (_evid == 3 || _evid == 4)
        offset += t # Adjust for TAD specification
      end
    end
  end

  sort!(events)
  return Subject{
    typeof(observations),
    typeof(covariates_interpolation),
    typeof(events),
    typeof(obstimes)}(
      _id_string,
      observations,
      covariates_interpolation,
      events,
      obstimes)
end

function Subject(;
  id::Union{String,Number} = "1",
  observations::Union{Nothing,AbstractDataFrame,NamedTuple} = nothing,
  events::Union{DosageRegimen,Vector{<:Event}} = Event[],
  time::Union{Nothing,AbstractVector{<:Number}} = observations isa AbstractDataFrame ? observations.time : nothing,
  event_data::Bool = true,
  covariates::Union{Nothing, NamedTuple} = nothing,
  covariates_time::Union{Nothing,AbstractVector{<:Number},NamedTuple} = observations isa AbstractDataFrame ? observations.time : nothing,
  covariates_direction::Symbol = :right)

  # Check that time is well-specified (not nothing, not missing and increasing)
  _time = isnothing(time) ? nothing : Missings.disallowmissing(time)

  if !isnothing(time) && !issorted(time)
    throw(PumasDataError("Time is not monotonically increasing within a manually constructed subject"))
  end

  obs = build_observation_list(observations)
  events = build_event_list(events, event_data)

  covartime, covariates_interpolation = covariates_interpolant(
    covariates, covariates_time,
    id;
    covariates_direction=covariates_direction)

  return Subject{
    typeof(obs),
    typeof(covariates_interpolation),
    typeof(events),
    typeof(_time)}(
      string(id),
      obs,
      covariates_interpolation,
      events,
      _time)
end

Base.hash(subject::Subject, h::UInt) = hash(
  subject.time, hash(
    subject.events, hash(
      subject.covariates, hash(
        subject.observations, hash(
          subject.id, h)))))

Base.:(==)(subject1::Subject, subject2::Subject) = hash(subject1) == hash(subject2)

function DataFrames.DataFrame(subject::Subject; include_covariates=true, include_observations=true, include_events=true)
  # Build a DataFrame that holds the events
  if isempty(subject.events)
    df_events = DataFrame(fill(Float64, 10),
      [:amt, :time, :evid, :cmt, :rate, :duration, :ss, :ii, :base_time, :rate_dir], 0)
    include_events = false
  else
    df_events = DataFrame(build_event_list(subject.events, true))
    # Remove events with evid==-1
    df_events = df_events[df_events[!, :evid].!=-1,:]
  end
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
  # multiple observations etc

  if isnothing(subject.time)
    df_events = hcat(DataFrame(id = fill(subject.id, length(df_events.evid))), df_events)
  else
    df = DataFrame(id = fill(subject.id, length(subject.time)), time=subject.time)
    df[!, :evid] .= 0
    # Only include the dv columns if include_observations is specified and there are
    # observations.
    if include_observations && !isnothing(subject.observations)
      # Loop over all dv's
      for (dv_name, dv_vals) in pairs(subject.observations)
        df[!, dv_name] .= dv_vals
      end
    end
    # Now, create columns in df_events with missings for the column names in
    # df but not in df_events
    if include_observations
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
        for df_name in Symbol.(names(df_events))
          if !hasproperty(df, df_name)
            if df_name == :ss
              # See NONMEM Users Guide Part VI - PREDPP Guide: V.F.2
              df[!, df_name] .= 0
            else
              df[!, df_name] .= missing
            end
          end
        end
      end
    end
  end
  # If there are no observations, just go with df_events
  if isnothing(subject.time)
    df = df_events
  elseif include_observations && include_events
    df = vcat(df, df_events)
  else
    df
  end
  # Sort the df according to time first, and use :base_time to ensure that events
  # come before observations (they are missing for observations, so they will come
  # last).
  sort_bys = include_observations && include_events ? [:time, :base_time] : [:time,]
  sort!(df, sort_bys)
  # Find the amount (amt) column, and insert dose and tad columns after it
  if include_observations && include_events
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
      df[idxs[i]:idxs[i+1], :tad] .= df[idxs[i]:idxs[i+1], :time].-df[idxs[i], :time]
    end

    # If some rows were adding so far the cmt column will be missing. Fill them all out
    # with the last occuring cmt (we already sorted above). We accumulate the largest occurence
    # of nonmissing indeces to figure out which value to fill out. This is relevant when there
    # are events and non-event observations 
    df[!, :cmt] .= df.cmt[accumulate(max, [i*!ismissing(df.cmt[i]) for i=1:length(df.cmt)])]
  end

  if include_covariates
    df = _add_covariates(df, subject)
  end
  # Return df
  df
end

function _add_covariates(df::DataFrame, subject::Subject)
  # If the covariates are time varying we include them explicitly as evid 0 rows.
  # The thought here is to be able to recreate the interpolant from the DataFrame
  # that we end up returning here.
  if subject.covariates isa ConstantInterpolationStructArray
    covartime = subject.covariates.t
    __time = [t for t in covartime if t ∉ df.time]
    df_covar = DataFrame(id = fill(subject.id, length(__time)), time=__time, evid=0, covar_debug = true)
    df = vcat(df, df_covar; cols=:union)
    df = sort!(df, [:id, :time])
    df = df[!, Not(:covar_debug)]
  end

  covariates = subject.covariates.(df.time)
  if !(isa(covariates, Nothing) || isa(covariates, Tuple{})) && !isa(first(covariates), Nothing)
    if covariates isa AbstractVector
      _keys = keys(covariates[1])
      covariates = map(NamedTuple{_keys}(_keys)) do x
         [_covar[x] for _covar in covariates]
        end
    end
    for (covariate, value) in pairs(covariates)
      if eltype(value) == String || eltype(value) == Number
        df[!, covariate] .= eltype(value) isa Number ? typeof(value)(0) : Ref("")
        df[!, covariate] .= value
      else
        df[!, covariate] .= eltype(value)(0)
        df[!, covariate] .= value
      end
    end
  end
  df
end
hascovariates(covariates) = true
hascovariates(covariates::NoCovar) = false
hascovariates(subject::Subject) = hascovariates(subject.covariates)
### Display
Base.summary(::Subject) = "Subject"
function Base.show(io::IO, subject::Subject)
  println(io, summary(subject))
  println(io, string("  ID: $(subject.id)"))
  events = subject.events
  isnothing(events) || println(io, "  Events: ", length(subject.events))
  obs = subject.observations
  observables = propertynames(obs)
  if !isempty(observables)
    vals = mapreduce(pn -> string(pn, ": (n=$(length(getindex(obs, pn))))"),
                     (x, y) -> "$x, $y",
                     observables)
    println(io, "  Observables: $vals")
  end
  if hascovariates(subject)
    _covariates = subject.covariates(0.0)
    println(io, "  Covariates: ", join(fieldnames(typeof(_covariates)),", "))
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

function DataFrames.DataFrame(pop::Population; include_covariates=true, include_observations=true, include_events=true)
  vcat((DataFrame(subject; include_covariates=include_covariates, include_observations=include_observations, include_events=include_events) for subject in pop)...)
end

### Display
Base.summary(::Population) = "Population"
function Base.show(io::IO, ::MIME"text/plain", population::Population)
  println(io, summary(population))
  println(io, "  Subjects: ", length(population))
  if isassigned(population, 1)
    co = population[1].covariates(0.0)
    !isnothing(co) && println(io, "  Covariates: ", join(fieldnames(typeof(co)),", "))
    obs = population[1].observations
    !isnothing(obs) && println(io, "  Observables: ", join(keys(obs),", "))
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

@recipe function f(pop::Population;obsnames=nothing)
  for p in pop
    @series begin
      linewidth --> 1.5
      title --> "Population Simulation"
      obsnames --> obsnames
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
  NCADose(time, amt, duration, IVBolus)
end
NCADose(dose::Event) = convert(NCADose, dose)

function Base.convert(::Type{NCASubject}, subj::Subject; name=:dv, kwargs...)
  dose = convert.(NCADose, subj.events)
  ii = subj.events[end].ii
  if subj.observations === nothing || subj.time === nothing
    throw(ArgumentError("The subject (id=$(subj.id)) has empty time or observations."))
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
    read_pumas(data, covariates=Symbol[], observations=Symbol[:dv];
                   id=:id, time=:time, evid=:evid, amt=:amt, addl=:addl,
                   ii=:ii, cmt=:cmt, rate=:rate, ss=:ss,
                   event_data = true)

Import NMTRAN-formatted data.

- `covariates` covariates specified by either names or column numbers
- `observations` dependent variables specified by either names or column numbers
- `event_data` toggles assertions applicable to event data
"""
function read_pumas(filepath::AbstractString; kwargs...)
  read_pumas(DataFrame(CSV.File(filepath, missingstrings=["", ".", "NA"])); kwargs...)
end
function read_pumas(df::AbstractDataFrame;
  covariates=Symbol[], observations=Symbol[:dv],
  id=:id, time=:time, evid=nothing, amt=:amt, addl=:addl,
  ii=:ii, cmt=:cmt, rate=:rate, ss=:ss, mdv=nothing,
  event_data=true, covariates_direction=:right,
  parse_tad = true, check=event_data)

  df = preprocess_data(df, observations, amt, mdv, evid, event_data)
  _evid = evid===nothing ? :evid : evid

  if check
    check_pumas_data(df, covariates, observations,
      id, time, _evid, amt, addl, ii, cmt, rate, ss, mdv,
      event_data)
  end

  __read_pumas(df, covariates, observations,
    id, time, _evid, amt, addl, ii, cmt, rate, ss, mdv,
    event_data, covariates_direction, parse_tad)
end

function __read_pumas(
  df::AbstractDataFrame,
  covariates::Vector{Symbol},
  observations::Vector{Symbol},
  id::Symbol,
  time::Union{Symbol,Nothing},
  evid::Union{Symbol,Nothing},
  amt::Union{Symbol,Nothing},
  addl::Union{Symbol,Nothing},
  ii::Union{Symbol,Nothing},
  cmt::Union{Symbol,Nothing},
  rate::Union{Symbol,Nothing},
  ss::Union{Symbol,Nothing},
  mdv::Union{Symbol,Nothing},
  event_data::Bool,
  covariates_direction::Symbol,
  parse_tad::Bool)

  if covariates isa AbstractVector{<:Integer}
    Base.depwarn("selecting covariate columns by integer indexing has been deprecated. Please specify the name of the columns instead.", :read_pumas)
    covariates = Symbol.(names(df)[covariates])
  end
  if observations isa AbstractVector{<:Integer}
    Base.depwarn("selecting dependent variable columns by integer indexing has been deprecated. Please specify the name of the columns instead.", :read_pumas)
    observations = Symbol.(names(df)[observations])
  end

  return [Subject(subject_df, id, time, evid, amt, addl, ii, cmt,
                  rate, ss, covariates, observations, event_data, covariates_direction, parse_tad) for subject_df in groupby(df, id)]
end

struct PumasDataError <: Exception  msg::AbstractString end
Base.showerror(io::IO, e::PumasDataError) = print(io, "PumasDataError: ", e.msg)


function preprocess_data(df::AbstractDataFrame,
  observations::Vector{Symbol},
  amt::Symbol,
  mdv::Union{Symbol,Nothing},
  evid::Union{Symbol,Nothing},
  event_data::Bool)

  df = copy(df)

  # If no evid column is specified then construct one from amt
  if evid === nothing && !hasproperty(df, :evid)
    if event_data
      @warn """
Your dataset has dose event but it hasn't an evid column. We are adding 1 for dosing rows and 0 for others in evid column. If this is not the case, please add your evid column.
      """
      df[!,:evid] = map(t -> Int(!ismissing(t) && t > 0), df[!,amt])
    else
      df[!,:evid] .= 0
    end
  end

  # We allow specifying missing values in of the dependent variable with an mdv column
  # but internally we encode missing values with missing directly in the dv columns.
  if mdv !== nothing || hasproperty(df, :mdv)
    mdv = mdv === nothing ? :mdv : mdv
    allowmissing!(df, observations)
    nrows = 0;
    for (idx, row) in enumerate(eachrow(df))
      _mdv = isone(row[mdv])
      for dv in observations
        _dv = row[dv]
        if _dv isa Missing && _mdv == zero(_mdv)
          throw(Pumas.PumasDataError("(row: $(idx)) $(dv) is missing but $(mdv) is set to zero."))
        elseif !(_dv isa Missing) && _mdv == one(_mdv)
          nrows += 1;
        end
        row[dv] = ifelse(_mdv, missing, _dv)
      end
    end
    if nrows > 0
      @warn "$(nrows) row(s) has(ve) non-missing observation(s) with $(mdv) set to one. $(mdv) is taking precedence."
    end
  end
  return df
end

function check_pumas_data(df::AbstractDataFrame,
  covariates::Vector{Symbol},
  observations::Vector{Symbol},
  id::Symbol, time::Union{Symbol,Nothing}, evid::Union{Symbol,Nothing}, amt::Union{Symbol,Nothing},
  addl::Union{Symbol,Nothing}, ii::Union{Symbol,Nothing}, cmt::Union{Symbol,Nothing},
  rate::Union{Symbol,Nothing}, ss::Union{Symbol,Nothing}, mdv::Union{Symbol,Nothing},
  event_data::Bool)

  colnames = propertynames(df)
  # Check if all necessary columns are present or not
  has_id     = id   in colnames
  has_time   = time in colnames
  has_amt    = amt  in colnames
  has_cmt    = cmt  in colnames
  has_evid   = evid in colnames
  has_addl   = addl in colnames
  has_ii     = ii   in colnames
  has_ss     = ss   in colnames
  has_rate   = rate in colnames
  has_evid34 = any(x->x==3 || x==4,df[:,evid])

  if observations isa AbstractVector{<:Integer}
    observations = colnames[observations]
  end
  has_observations = true
  for dv in observations
    has_observations &= dv in colnames
  end
  # CASE : no event identifier (evid) with event_data = true  # top most check
  if !has_evid && event_data
    throw(PumasDataError("""
Your dataset has no events (doses) as you have not specified the $(string(evid)) column.
If your intent is to have a Population with no events, then you can use the
argument event_data=false in your read_pumas function.
"""))
  end
  if event_data
    if !has_observations
      throw(PumasDataError("Column(s) in observations arg is(are) not present in the data file"))
    end
    ok = has_id && has_time && has_amt
    if !ok
      @info "The CSV file has keys: $colnames"
      throw(PumasDataError("The CSV file must have: `id, time, amt, and observations` when `event_data` is `true`" ))
    end
  else
    if !has_id
      @info "The CSV file has keys: $colnames"
      throw(PumasDataError("The CSV file must have: `id` when `event_data` is `false`"))
    end
  end

  # CASE: (id, time) pair should be unique
  if has_time && !has_evid34
    _df_onlyobs = filter(row -> row[evid] == 0, df[!, [id, time, evid]])
    if length(unique(zip(_df_onlyobs[!, id], _df_onlyobs[!, time]))) != size(_df_onlyobs, 1)
      throw(PumasDataError("($(id), $(time)) pair should be unique, if your data has multiple observations in one column, please convert it to wide format."))
    end
  end

  # CASE : non-numeric/string observations in observations
  for dv in observations
    check_non_numeric(df, id, dv; allow_missings=true)
  end

  # CASE : amt cannot be a string/non-numeric
  if has_amt
    check_non_numeric(df, id, amt, allow_missings=true)
    cmt_ok = true
    for dose in df[!, amt]
      if dose !== missing && dose > zero(dose)
        if !has_cmt
          cmt_ok = false
        end
        if !cmt_ok
          break # no need go further
        end
      end
    end

    if !cmt_ok
      throw(PumasDataError("Your dataset has dose event(s) but it doesn't have cmt column"))
    end
  end

  # CASE : cmt must be positive or string
  if has_cmt
    idx = findfirst(x -> !((x isa String) || (x isa Integer && x > 0) || (x isa Symbol)),
      filter(i -> i[evid] > 0, df)[!, cmt])
  end
  if has_cmt && idx !== nothing
    throw(PumasDataError("[Subject $(id): $(df[!, id][idx]), row = $(idx), col = $(cmt)] $(cmt) column should be positive"))
  end

  for (idx, row) in enumerate(eachrow(df))
    ok = has_evid && has_amt # make sure these are present before doing checks
    ok || break
    _id, _evid, _amt = row[id], row[evid], row[amt]

    # CASE : amt can be missing or zero when evid = 0
    if _evid == zero(_evid) && !(_amt isa Missing) && _amt != zero(_amt)
      throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(evid)] $(amt) can only be missing or zero when $(evid) is zero"))
    end

    # CASE : amt can be positive or zero when evid = 1
     if _evid == one(_evid) && _amt != zero(_amt) && _amt < zero(_amt)
       throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(evid)] $(amt) can only be positive or zero when $(evid) is one"))
     end

    # CASE : observations (dv) at time of dose
    for dv in observations
      _dv = row[dv]
      if _evid == 1 && !(_dv isa Missing)
        throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(dv)] an observation is present at the time of dose in column $(dv). It is recommended and required in Pumas to have a blank record (`missing`) at the time of time of dosing, i.e. when `amt` is positive."))
      end
    end
  end

  # The rules for addl and ii depends on the existence of an ss column with non-zero elements
  # First we consider the steady-state rules
  if has_ss
    # See NONMEM Users Guide Part VI - PREDPP Guide: V.F.2. Specifics of the Steady-State (SS) Data Item

    # CASE : Steady-state column requires ii column
    if !has_ii
      throw(PumasDataError("your dataset does not have $(ii) which is a required column for steady state dosing."))
    end

    for (idx, row) in enumerate(eachrow(df))
      _id, _ss, _ii, _amt = row[id], row[ss], row[ii], row[amt]
      # If rate column isn't present the set to zero
      _rate = has_rate ? row[rate] : 0.0
      if _ss > 0
        # CASE : repeated bolus doses with a given period
        if _rate == 0 && _amt > 0 ||
          # CASE : repeated infusions with a given period
          _rate > 0 && _amt > 0

          # CASE: Steady-state dosing requires ii>0
          if _ii == 0
            throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(_ii)] for steady-state dosing the value of the interval column $ii must be non-zero but was $_ii"))
          end

        elseif (_rate > 0 || _rate == -1) && _amt == 0
          # CASE : Steady-state infusion
          # CASE : requires ii=0
          if _ii != 0
            throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(_ii)] for steady-state infusion the value of the interval column $ii must be zero but was $_ii"))
          end

          # CASE : requires addl=0
          if has_addl
            _addl = row[addl]
            if _addl != 0
              throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(_addl)] for steady-state infusion the value of the additional dose column $addl must be zero but was $_addl"))
            end
          end
        else
          throw(PumasDataError("for steady state events either dose amout variable $amt or rate variable $rate must be non-zero"))
        end
      end
    end
  end

  if has_addl && !has_ii
    throw(PumasDataError("your dataset does not have $(ii) which is a required column when $(addl) is specified."))
  end

  if has_addl && has_ii
    for (idx, row) in enumerate(eachrow(df))
      _id, _addl, _ii = row[id], row[addl], row[ii]

      # Non-steady-state dosing (we considered the steady state dosing above)
      if has_ss && row[ss] != 0
        continue
      end

      # CASE : ii must be positive for addl > 0
      if _addl > 0 && _ii == zero(_ii)
        throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(ii)]  $(ii) must be positive for $(addl) > 0"))
      end

      # CASE : addl must be positive for ii > 0
      if _ii > 0 && _addl == 0
        throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(addl)]  $(addl) must be positive for $(ii) > 0"))
      end

      if has_evid
        _evid = row[evid]
        # CASE : ii can be missing or zero when evid = 0
        if _evid == zero(_evid) && !(_ii isa Missing) && _ii != zero(_ii)
          throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(evid)]  $(ii) can only be missing or zero when $(evid) is zero"))
        end

        # CASE : addl can be positive or zero when evid = 1
        if _evid == one(_evid) && _addl != zero(_addl) && _addl < zero(_addl)
          throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(evid)]  $(addl) can only be positive or zero when $(evid) is one"))
        end

        # CASE : evid must be nonzero when amt > 0 or addl and ii are positive
        if _addl > 0 && _ii > 0 && _evid == 0
          throw(PumasDataError("[Subject $(id): $(_id), row = $(idx), col = $(evid)]  $(evid) must be nonzero when $(amt) > 0 or $(addl) and $(ii) are positive"))
        end
      end
    end
  end
end

function check_non_numeric(df::AbstractDataFrame, id, colname; allow_missings=false)
  T = allow_missings === true ? Union{Number, Missing} : Number
  has_dv = colname in propertynames(df)
  if !has_dv
    throw(PumasDataError("The column $(colname) is not present in data file but it is specified in `read_pumas` as arg"))
  end
  if !(eltype(df[!, colname]) <: T)
    idx = findall(x -> !(x isa T), df[!, colname])
    if idx !== nothing
      els = unique(df[!, colname][idx])
    end
    if idx !== nothing
      throw(PumasDataError("""
[Subject $(id): $(df[!, id][idx]), row = $(idx), col = $(colname)]  We expect the $(colname) column to be of numeric type.
These are the unique non-numeric values present in the column $(colname): $((els...,))"""))
    end
  end
end
