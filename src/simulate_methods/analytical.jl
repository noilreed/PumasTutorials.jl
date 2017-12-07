function simulate(prob::PKPDAnalyticalProblem,set_parameters,θ,ηi,datai::Person,
                  output_reduction = (sol,p,datai) -> sol;
                  kwargs...)
  VarType = promote_type(eltype(ηi),eltype(θ))
  u0 = VarType.(prob.u0)
  f = prob.f
  ss = prob.ss
  tspan = VarType.(prob.tspan)
  tdir = sign(prob.tspan[end] - prob.tspan[1])
  p = set_parameters(θ,ηi,datai.z)
  lags,bioav,rate,duration = get_magic_args(p,u0,tspan[1])
  events = datai.events
  adjust_event_timings!(events,lags,bioav,rate,duration)
  times = sorted_approx_unique(events)
  u = Vector{typeof(u0)}(length(times))
  doses = Vector{typeof(u0)}(length(times))
  rates = Vector{typeof(u0)}(length(times))

  # Iteration 1
  t0 = times[1]
  cur_ev = events[1]
  dose,rate = create_dose_rate_vector(cur_ev,u0,zero(u0),bioav)
  if cur_ev.ss == 0
    u[1] = u0
    doses[1] = dose
    rates[1] = rate
    i = 2
  else
    i = 1
  end

  ss_time = -one(t0)
  ss_overlap_duration = -one(t0)
  ss_rate_multiplier = -1
  post_ss_counter = -1
  event_counter = i-1
  ss_rate = zero(rate)
  ss_cmt = 0
  ss_dropoff_event = false
  start_val = 0

  # Now loop through the rest
  while i <= length(times)
    t = times[i]
    ss_dropoff_event = post_ss_counter < ss_rate_multiplier + start_val &&
                       t == ss_time + ss_overlap_duration + post_ss_counter*cur_ev.ii

    if ss_dropoff_event
      # Do an off event from the ss
      post_ss_counter += 1
      u0 = f(t,t0,u0,dose,p,rate)
      u[i] = u0
      doses[i] = dose
      _rate = create_ss_off_rate_vector(ss_rate,ss_cmt,rate)
      rate = _rate
      rates[i] = rate
    else # Not an SS off event
      ss_dropoff_event = false
      event_counter += 1
      cur_ev = events[event_counter]
      @assert cur_ev.time == t
      if cur_ev.ss == 0
        dose,_rate = create_dose_rate_vector(cur_ev,u0,rate,bioav)
        (t0 != t) && (u0 = f(t,t0,u0,dose,p,rate))
        rate = _rate
        u[i] = u0
        doses[i] = dose
        rates[i] = rate
      else # handle steady state
        ss_time = t0
        # No steady state solution given, use a fallback
        dose,_rate = create_ss_dose_rate_vector(cur_ev,u0,rate,bioav)
        if cur_ev.rate > 0
          ss_rate = _rate
          _duration = (bioav*cur_ev.amt)/cur_ev.rate
          ss_overlap_duration = mod(_duration,cur_ev.ii)
          ss_rate_multiplier = 1 + (_duration>cur_ev.ii)*(_duration ÷ cur_ev.ii)
          ss_cmt = cur_ev.cmt
          _rate *= ss_rate_multiplier
          _t1 = t0 + ss_overlap_duration
          _t2 = t0 + cur_ev.ii
          _t1 == t0 && (_t1 = _t2)

          u0_cache = u0

          if ss == nothing
            cur_norm = Inf
            while cur_norm > 1e-12
              u0prev = u0
              u0 = f(_t1,t0,u0,dose,p,_rate)
              _t2-_t1 > 0 && (u0 = f(_t2,_t1,u0,zero(u0),p,_rate-ss_rate))
              cur_norm = norm(u0-u0prev)
            end
          end

          rate = _rate
          cur_ev.ss == 2 && (u0 += u0_cache)

          u[i] = u0
          doses[i] = dose
          rates[i] = rate
          old_length = length(times)
          ss_overlap_duration == 0 ? start_val = 1 : start_val = 0
          resize!(times,old_length+Int(ss_rate_multiplier))
          resize!(u,    old_length+Int(ss_rate_multiplier))
          resize!(doses,old_length+Int(ss_rate_multiplier))
          resize!(rates,old_length+Int(ss_rate_multiplier))
          for j in start_val:Int(ss_rate_multiplier)-1+start_val
            times[old_length+j+1-start_val] = ss_time + ss_overlap_duration + j*cur_ev.ii
          end
          post_ss_counter = start_val
          times = sort!(times)
        else # Not a rate event, just a dose

          _t1 = t0 + cur_ev.ii
          u0_cache = u0

          if ss == nothing
            cur_norm = Inf
            while cur_norm > 1e-12
              u0prev = u0
              u0 = f(_t1,t0,u0,dose,p,_rate)
              cur_norm = norm(u0-u0prev)
            end
          end

          rate = _rate
          cur_ev.ss == 2 && (u0 += u0_cache)
          u[i] = u0
          doses[i] = dose
          rates[i] = rate

        end
      end
    end
    # Don't update i for duplicate events
    t0 = t
    if ss_dropoff_event || (event_counter != length(events) && events[event_counter+1].time != t) || event_counter == length(events)
      i += 1
    end
  end

  _soli = PKPDAnalyticalSolution{typeof(u0),ndims(u0)+1,typeof(u),
                     typeof(times),
                     typeof(doses),typeof(rates),
                     typeof(p),
                     typeof(prob)}(
                     u,times,doses,rates,p,prob,true,0,:Success)
  output_reduction(_soli,p,datai)
end

function create_dose_rate_vector(cur_ev,u0,rate,bioav)
  if cur_ev.rate == 0
    if typeof(bioav) <: Number
      return increment_value(zero(u0),bioav*cur_ev.amt,cur_ev.cmt),rate
    else
      return increment_value(zero(u0),bioav[cur_ev.cmt]*cur_ev.amt,cur_ev.cmt),rate
    end
  else
    return zero(u0),increment_value(rate,cur_ev.rate_dir*cur_ev.rate,cur_ev.cmt)
  end
end

function create_ss_dose_rate_vector(cur_ev,u0,rate,bioav)
  if cur_ev.rate == 0
    if typeof(bioav) <: Number
      return set_value(zero(u0),bioav*cur_ev.amt,cur_ev.cmt),rate
    else
      return set_value(zero(u0),bioav[cur_ev.cmt]*cur_ev.amt,cur_ev.cmt),rate
    end
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
