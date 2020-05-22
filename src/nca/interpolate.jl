function interpextrapconc(nca::NCASubject, timeout; concorigin=nothing, method=nothing, kwargs...)
  timeout === missing && return missing
  nca.dose isa Union{NCADose, Nothing} || throw(ArgumentError("interpextrapconc doesn't support multidose data"))
  conc, time = nca.conc, nca.time
  _tlast = tlast(nca)
  isempty(timeout) && throw(ArgumentError("timeout must be a vector with at least one element"))
  out = timeout isa AbstractArray ? fill!(similar(conc, length(timeout)), zero(eltype(conc))) : zero(eltype(conc))
  for i in eachindex(out)
    if out[i] === missing
      _out = missing
    elseif timeout[i] <= _tlast
      _out = interpolateconc(nca, timeout[i]; method=method, concorigin=concorigin, kwargs...)
    else
      _out = extrapolateconc(nca, timeout[i]; kwargs...)
    end
    out isa AbstractArray ? (out[i] = _out) : (out = _out)
  end
  return out
end

function interpolateconc(nca::NCASubject, timeout::Number; method, kwargs...)
  conc, time = nca.conc, nca.time
  len = length(time)
  _tlast = tlast(nca)
  !(method in (:linear, :linuplogdown, :linlog)) && throw(ArgumentError("Interpolation method must be :linear, :linuplogdown or :linlog"))
  if timeout > _tlast
    throw(ArgumentError("interpolateconc can only works through Tlast, please use interpextrapconc to combine both interpolation and extrapolation"))
  elseif (idx2=searchsortedfirst(time, timeout)) != len+1 && time[idx2] == timeout # if there is an exact time match
    return conc[idx2]
  else
    if time[1] > zero(time[1]) && zero(timeout) <= timeout < time[1] # if we need to calculate `c0`
      c0′ = c0(nca, true)
      timeout == zero(timeout) && return c0′
      idx1  = 1
      time1 = ustrip(zero(time[idx1])); time2 = ustrip(time[idx1])
      conc1 = ustrip(c0′); conc2 = ustrip(conc[idx1])
    else
      idx1 = idx2 - 1
      time1 = ustrip(time[idx1]); time2 = ustrip(time[idx2])
      conc1 = ustrip(conc[idx1]); conc2 = ustrip(conc[idx2])
    end
    timeout = ustrip(timeout)
    m = choosescheme(conc1, conc2, time1, time2, idx1, maxidx(nca), method)
    if m === Linear
      return (conc1+abs(timeout-time1)/(time2-time1)*(conc2-conc1))*oneunit(eltype(conc))
    else
      return exp(log(conc1)+(timeout-time1)/(time2-time1)*(log(conc2)-log(conc1)))*oneunit(eltype(conc))
    end
  end
end

function extrapolateconc(nca::NCASubject, timeout::Number; kwargs...)
  conc, time = nca.conc, nca.time
  λz = lambdaz(nca; recompute=false, kwargs...)
  _tlast = tlast(nca)
  _clast = clast(nca; kwargs...)
  if timeout <= _tlast
    throw(ArgumentError("extrapolateconc can only work beyond Tlast, please use interpextrapconc to combine both interpolation and extrapolation"))
  else
    return _clast*exp(-λz*(timeout - _tlast))
  end
end
