function DiffEqBase.solve(prob::PresetAnalyticalPKProblem,args...;kwargs...)
  numsol = solve(prob.numprob,args...;prob.numprob.kwargs...,kwargs...)
  saveat = :saveat ∈ keys(kwargs) ? kwargs[:saveat] :
          (:saveat ∈ keys(prob.numprob.kwargs) ?
                                        prob.numprob.kwargs[:saveat] : nothing)
  if saveat !== nothing
    t = saveat
    u = [[prob.pksol(numsol.t[i]);numsol[i]] for i in 1:length(numsol)]
  else
    t = numsol.t
    u = numsol.u
  end
  return AnalyticalPKSolution(u,t,prob.pksol,numsol)
end

function DiffEqBase.solve(prob::NullDEProblem,args...;kwargs...)
  return NullDESolution(prob)
end

function _build_analytical_problem(
  m::PumasModel,
  subject::Subject,
  tspan, col, args...; kwargs...)

  # For Mixed PKPD, the PK variables are initialized at zero via the
  # pk_init methods for each of the closed form models. In all other cases
  # the @init is used for the initialization
  if m.prob isa ExplicitModel
    f = m.prob
    u0 = m.init(col, first(tspan))
  else
    f = m.prob.pkprob
    u0 = pk_init(f)
  end
  numtypecol = numtype(col(tspan[1]))
  # we don't want to promote units
  if numtypecol <: Unitful.Quantity || numtype(u0) <: Unitful.Quantity || numtype(tspan) <: Unitful.Quantity
    Tu0 = map(float, u0)
    Ttspan = map(float, tspan)
  else
    T = promote_type(numtypecol, numtype(u0), numtype(tspan))
    Tu0 = convert.(T,u0)
    Ttspan = map(float, tspan)
  end

  # We add zero amount doses for all covariate times to ensure that
  # that the coefficient values are constant on each subintercal
  events = copy(subject.events)
  __cmt = if eltype(events) <: Event{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Symbol}
    first(events).cmt
  else
    1
  end
  if subject.covariates isa ConstantInterpolationStructArray
    for t ∈ subject.covariates.t
      push!(events, Event(0.0,t,1,__cmt))
    end
  end
  events = identity.(events)
  events = adjust_event(events,col,u0)
  times = unique(map(ev->ev.time, events))

  prob = AnalyticalPKPDProblem{false}(f, Tu0, Ttspan, events, times, col)
end

function DiffEqBase.solve(prob::AnalyticalPKPDProblem,
                          args...; continuity = :right, kwargs...)
  f = prob.f
  Tu0 = prob.u0
  Ttspan = prob.tspan
  col = prob.p
  events = prob.events
  times = prob.time
  u = Vector{typeof(Tu0)}(undef, length(times))
  doses = zeros(typeof(Tu0), length(times))
  rates = Vector{typeof(Tu0)}(undef, length(times))

  rate = zero(Tu0)
  last_dose = zero(Tu0)
  i = 1
  ss_time = -one(eltype(times))
  ss_overlap_duration = -one(eltype(times))
  ss_rate_multiplier = -1.0
  post_ss_counter = -1
  event_counter = i - 1
  ss_rate = zero(rate)
  ss_ii = zero(eltype(times))
  ss_cmt = 0
  ss_dropoff_event = false
  start_val = 0
  retcode = :Success
  dose = zero(Tu0)
  local t, t0
  # Now loop through the rest
  while i <= length(times)

    # Initialize t0 inside the while loop since times might be empty
    if i == 1
      t0 = first(times)
    end

    if eltype(times) <: ForwardDiff.Dual && i > 1
      # Used to introduce a coupling for AD to propogate the dual part
      # It could be off by floating point error in the partials if
      # times[i] - times[i-1] + times[i-1] != times[i]
      dt = times[i] - times[i-1]
      t += dt
      # Correct the value of the non-dual portion to have no floating
      # point error.
      val = ForwardDiff.value(t)
      if typeof(val) <: ForwardDiff.Dual
        # Nested AD case
        t = ForwardDiff.Dual{ForwardDiff.tagtype(t)}(
              ForwardDiff.Dual{ForwardDiff.tagtype(val)}(
              ForwardDiff.value(ForwardDiff.value(times[i])),ForwardDiff.partials(val)),ForwardDiff.partials(t))
      else
        t = ForwardDiff.Dual{ForwardDiff.tagtype(t)}(ForwardDiff.value(times[i]),ForwardDiff.partials(t))
      end
    else
      t = times[i]
    end

    ss_dropoff_event = post_ss_counter < ss_rate_multiplier + start_val &&
                       t == ss_time + ss_overlap_duration + post_ss_counter*ss_ii
    if ss_dropoff_event
      # Do an off event from the ss
      post_ss_counter += 1
      Tu0 = f(t,t0,Tu0,dose,col,rate)
      u[i] = Tu0
      doses[i] = zero(Tu0)
      _rate = create_ss_off_rate_vector(ss_rate,ss_cmt,rate)
      rate = _rate
      rates[i] = rate
    else # Not an SS off event
      ss_dropoff_event = false
      event_counter += 1
      cur_ev = events[event_counter]
      cur_ev.evid >= 3 && (Tu0 = zero(Tu0))
      # Ensures two things: (a) that the events are synced, so t is at
      # the time we believe it should be, and (b) that there's no floating
      # point error in t, which should be handled directly by the dual
      # handling above.

      @assert cur_ev.time == t
      if cur_ev.ss == 0
        dose,_rate = create_dose_rate_vector(cur_ev,t0,dose,rate)
        (t0 != t) && cur_ev.evid < 3 && (Tu0 = f(t,t0,Tu0,last_dose,col,rate))
        rate = _rate
        u[i] = Tu0
        doses[i] = dose
        last_dose = dose
        rates[i] = rate
      else # handle steady state
        ss_time = t0
        # No steady state solution given, use a fallback
        dose,_rate = create_ss_dose_rate_vector(cur_ev,Tu0,rate)
        if cur_ev.rate > 0
          ss_rate = _rate
          _duration = cur_ev.amt/cur_ev.rate
          ss_overlap_duration = mod(_duration,cur_ev.ii)
          ss_rate_multiplier = 1 + (_duration>cur_ev.ii)*floor(_duration / cur_ev.ii)
          ss_cmt = cur_ev.cmt
          ss_ii = cur_ev.ii
          _rate *= ss_rate_multiplier
          _t1 = t0 + ss_overlap_duration
          _t2 = t0 + cur_ev.ii
          _t1 == t0 && (_t1 = _t2)

          cur_ev.ss==2 && (Tu0_cache = f(t,t0,Tu0,last_dose,col,rate))

          Tu0prev = Tu0 .+ 1
          while norm(Tu0-Tu0prev) > 1e-12
            Tu0prev = Tu0
            Tu0 = f(_t1,t0,Tu0,dose,col,_rate)
            _t2-_t1 > 0 && (Tu0 = f(_t2,_t1,Tu0,zero(Tu0),col,_rate-ss_rate))
          end

          rate = _rate
          cur_ev.amt == 0 && (rate = zero(rate))
          cur_ev.ss == 2 && (Tu0 += Tu0_cache)

          u[i] = Tu0
          doses[i] += dose
          last_dose = dose
          rates[i] = rate
          if cur_ev.amt != 0
            old_length = length(times)
            ss_overlap_duration == 0 ? start_val = 1 : start_val = 0
            resize!(times, old_length + Int(ss_rate_multiplier))
            resize!(u    , old_length + Int(ss_rate_multiplier))
            resize!(doses, old_length + Int(ss_rate_multiplier))
            resize!(rates, old_length + Int(ss_rate_multiplier))
            for j in start_val:(Int(ss_rate_multiplier) - 1 + start_val)
              times[old_length + j + 1 - start_val] = ss_time + ss_overlap_duration + j*cur_ev.ii
              # The newly added memory is uninitialized so we explicitly initialize to zero
              doses[old_length + j + 1 - start_val] = zero(dose)
            end

            post_ss_counter = start_val
            times = sort!(times)
          end
        else # Not a rate event, just a dose

          _t1 = t0 + cur_ev.ii
          cur_ev.ss==2 && (Tu0_cache = f(t,t0,Tu0,last_dose,col,rate))

          Tu0prev = Tu0 .+ 1
          while norm(Tu0-Tu0prev) > 1e-12
            Tu0prev = Tu0
            Tu0 = f(_t1,t0,Tu0,dose,col,_rate)
          end

          rate = _rate
          cur_ev.ss == 2 && (Tu0 += Tu0_cache)
          u[i] = Tu0
          doses[i] += dose
          last_dose = dose
          rates[i] = rate

        end
      end
    end

    if any(x->any(isnan,x),u[i])
      retcode = :Unstable
      break
    end

    # Don't update i for duplicate events
    t0 = t
    if ss_dropoff_event || (event_counter != length(events) && events[event_counter+1].time != t) || event_counter == length(events)
      dose = zero(dose)
      i += 1
    end
  end

  # The two assumes that all equations are vector equations
  PKPDAnalyticalSolution{typeof(Tu0),
                         2,
                         typeof(u),
                         typeof(times),
                         typeof(doses),
                         typeof(rates),
                         typeof(col),
                         typeof(prob)}(u,times,doses,rates,col,prob,
                                       true,0,retcode,continuity)
end

function create_dose_rate_vector(cur_ev,t0,dose,rate)
  if cur_ev.rate == 0
    increment_value(dose,cur_ev.amt,cur_ev.cmt),rate
  else # rates are always in incrementing form
    dose,increment_value(rate,cur_ev.rate_dir*cur_ev.rate,cur_ev.cmt)
  end
end

function create_ss_dose_rate_vector(cur_ev,u0,rate)
  if cur_ev.rate == 0
    return set_value(zero(u0),cur_ev.amt,cur_ev.cmt),rate
  else
    return zero(u0),set_value(zero(u0),cur_ev.rate_dir*cur_ev.rate,cur_ev.cmt)
  end
end

function create_ss_off_rate_vector(ss_rate,ss_cmt,rate)
  if rate == 0
    error("Too many off events, rate went negative!")
  else
    return increment_value(rate,-ss_rate[ss_cmt],ss_cmt)
  end
end
