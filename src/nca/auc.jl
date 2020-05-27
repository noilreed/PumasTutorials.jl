"""
    auclinear(C₁, C₂, t₁, t₂)

Compute area under the curve (AUC) in an interval by linear trapezoidal rule.
"""
@inline function auclinear(C₁, C₂, t₁, t₂)
  Δt = t₂ - t₁
  return ((C₁ + C₂) * Δt) / 2
end

"""
    auclog(C₁, C₂, t₁, t₂)

Compute area under the curve (AUC) in an interval by log-linear trapezoidal
rule.
"""
@noinline function auclog(C₁, C₂, t₁, t₂)
  Δt = t₂ - t₁
  ΔC = C₂ - C₁
  lg = log(C₂/C₁)
  return (Δt * ΔC) / lg
end


"""
    extrapaucinf(clast, tlast, lambdaz)

Extrapolate auc to the infinite.
"""
@inline extrapaucinf(clast, tlast, lambdaz) = clast/lambdaz

"""
    aumclinear(C₁, C₂, t₁, t₂)

Compute area under the first moment of the concentration (AUMC) in an interval
by linear trapezoidal rule.
"""
@inline function aumclinear(C₁, C₂, t₁, t₂)
  return auclinear(C₁*t₁, C₂*t₂, t₁, t₂)
end

"""
    aumclog(C₁, C₂, t₁, t₂)

Compute area under the first moment of the concentration (AUMC) in an interval
by log-linear trapezoidal rule.
"""
@noinline function aumclog(C₁, C₂, t₁, t₂)
  Δt = t₂ - t₁
  lg = log(C₂/C₁)
  return Δt*(t₂*C₂ - t₁*C₁)/lg - Δt^2*(C₂-C₁)/lg^2
end

"""
    extrapaumcinf(clast, tlast, lambdaz)

Extrapolate the first moment to the infinite.
"""
@inline extrapaumcinf(clast, tlast, lambdaz) = clast*tlast/lambdaz + clast/lambdaz^2

@enum AUCmethod Linear Log

@inline function choosescheme(c1, c2, t1, t2, i::Int, maxidx::Int, method::Symbol)
  if method === :linear || (method === :linuplogdown && c2>=c1) || (method === :linlog && i < maxidx)
    return Linear
  else
    (c1 <= zero(c1) || c2 <= zero(c2) || c1 == c2) && return Linear
    return Log
  end
end

@inline function intervalauc(c1, c2, t1, t2, i::Int, maxidx::Int, method::Symbol, linear, log, ret_typ)
  t1 == t2 && return zero(ret_typ)
  m = choosescheme(c1, c2, t1, t2, i, maxidx, method)
  ret = m === Linear ? linear(c1, c2, t1, t2) : log(c1, c2, t1, t2)
  return ret
end

@inline function iscached(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}, sym::Symbol) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  @inbounds begin
    # `points` is initialized to 0
    sym === :lambdaz && return nca.points === missing || !(first(nca.points) == 0)
    # `auc_last` and `aumc_last` are initialized to -1
    sym === :auc     && return nca.auc_last  === missing || !(first(nca.auc_last)  == -oneunit(eltype(AUC)))
    sym === :aumc    && return nca.aumc_last === missing || !(first(nca.aumc_last) == -oneunit(eltype(AUMC)))
  end
end

function cleancache!(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  if nca.points isa AbstractArray
    fill!(nca.points, 0)
    fill!(nca.auc_last, -oneunit(eltype(AUC)))
    fill!(nca.aumc_last, -oneunit(eltype(AUMC)))
    fill!(nca.run_status, :Success)
  else
    nca.points = 0
    nca.auc_last  = -oneunit(eltype(AUC))
    nca.aumc_last = -oneunit(eltype(AUMC))
    nca.run_status = :Success
  end
  return nothing
end
cleancache!(pop::NCAPopulation) = foreach(cleancache!, pop)

function isaucinf(auctype, interval)
  auctype === :last && return false
  interval === nothing ? auctype === :inf : !isfinite(interval[2])
end

function cacheauc0!(nca, auc0)
  if nca.auc_0 isa AbstractArray
    nca.auc_0[1] = auc0
  else
    nca.auc_0 = auc0
  end
  nothing
end

function cacheauc!(nca::NCASubject, auclast, interval, method, isauc)
  if interval == nothing # only cache when interval is nothing
    if isauc
      nca.auc_last isa AbstractArray  ? nca.auc_last[1] = auclast  : nca.auc_last = auclast
    else
      nca.aumc_last isa AbstractArray ? nca.aumc_last[1] = auclast : nca.aumc_last = auclast
    end
    nca.method = method
  end
  nothing
end


function _auc(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}, interval, linear, log, inf, ret_typ;
              auctype, method=:linear, isauc, verbose=true, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  # fast return
  if interval === nothing && nca.method === method
    if auctype === :inf
      _clast = clast(nca; verbose=verbose, kwargs...)
      _tlast = tlast(nca)
      aucinf′ = inf(_clast, _tlast, lambdaz(nca; recompute=false, verbose=verbose, kwargs...))
      isauc  && iscached(nca, :auc)  && return (nca.auc_last  === missing ? missing : nca.auc_last[1] + aucinf′)
      !isauc && iscached(nca, :aumc) && return (nca.aumc_last === missing ? missing : nca.aumc_last[1] + aucinf′)
    elseif auctype === :last
      isauc  && iscached(nca, :auc)  && return nca.auc_last  === missing ? missing : nca.auc_last[1]
      !isauc && iscached(nca, :aumc) && return nca.aumc_last === missing ? missing : nca.aumc_last[1]
    end
  end

  conc, time = nca.conc, nca.time

  auctype in (:inf, :last) || throw(ArgumentError("auctype must be either :inf or :last"))
  method in (:linear, :linuplogdown, :linlog) || throw(ArgumentError("method must be :linear, :linuplogdown or :linlog"))
  _clast = clast(nca; verbose=verbose, kwargs...)
  _tlast = tlast(nca)
  # type assert `auc`
  auc::ret_typ = zero(ret_typ)
  if _clast === missing
    cacheauc!(nca, auc, interval, method, isauc)
    if all(x->x<=nca.llq, conc)
      return auc
    else
      verbose && @info "Unknown error with missing `tlast` but non-BLQ concentrations"
      return missing
    end
  end

  if interval !== nothing
    interval[1] >= interval[2] && throw(ArgumentError("The AUC interval must be increasing, got interval=$interval"))
    lo, hi = interval
    if lo < first(time) && first(time) == zero(eltype(time))
      @warn "Requesting an AUC range starting $lo before the first measurement $(first(time)) is not allowed"
    end
    lo > last(time) && @warn "AUC start time $lo is after the maximum observed time $(last(time))"
    # TODO: handle `auclinlog` with IV bolus.
    #
    # `C0` is the concentration at time zero if the route of administration is IV
    # (intranvenous). If concentration at time zero is not measured after the IV
    # bolus, a linear back-extrapolation is done to get the intercept which
    # represents concentration at time zero (`C0`)
    #
    concstart = interpextrapconc(nca, lo; method=method, kwargs...)
    idx1, idx2 = let lo = lo, hi = hi
      findfirst(x->x>=lo, time),
      findlast( x->x<=hi, time)
    end
    # auc of the first interval
    auc = intervalauc(concstart, conc[idx1], lo, time[idx1], idx1-1, maxidx(nca), method, linear, log, ret_typ)
    int_idxs = idx1:idx2-1
    # auc of the last interval
    if isfinite(hi)
      concend = interpextrapconc(nca, hi; method=method, kwargs...)
      if hi > _tlast # if we are extrapolating at the last point, let's use the logarithm trapezoid
        λz = lambdaz(nca; recompute=false, verbose=verbose, kwargs...)
        auc += intervalauc(conc[idx2], concend, time[idx2], hi, idx2, maxidx(nca), method, linear, log, ret_typ)
      else
        auc += intervalauc(conc[idx2], concend, time[idx2], hi, idx2, maxidx(nca), method, linear, log, ret_typ)
      end
    end
  else
    idx1, idx2 = firstindex(time), nca.lastidx
    # handle C0
    cacheauc0!(nca, zero(_first(nca.auc_0)))
    time0 = zero(time[idx1])
    if time[idx1] > time0
      c0′ = c0(nca, true)
      if c0′ === missing
        verbose && @info "ID $(nca.id) errored: AUC calculation cannot proceed, because `c0` gives missing"
        setrun_status!(nca, :C0IsMissing)
        cacheauc!(nca, missing, interval, method, isauc)
        cacheauc0!(nca, missing)
        return missing
      end
      auc_0 = intervalauc(c0′, conc[idx1], time0, time[idx1], idx1-1, maxidx(nca), method, linear, log, ret_typ)
      cacheauc0!(nca, auc_0)
      auc += auc_0
    end
  end
  if !isassigned(time, idx1+1) # if idx1+1 is not assigned
    idx1 = idx2 # no iteration
  end

  # Compute the AUxC
  # Include the first point after `tlast` if it exists and we are computing AUCall
  # do not support :AUCall for now
  # auctype === :AUCall && tlast > hi && (idx2 = idx2 + 1)
  #
  # iterate over the interior points
  for i in idx1:idx2-1
    auc += intervalauc(conc[i], conc[i+1], time[i], time[i+1], i, maxidx(nca), method, linear, log, ret_typ)
  end

  # cache results
  cacheauc!(nca, auc, interval, method, isauc)

  # add the extrapolation part
  if isaucinf(auctype, interval)
    λz = lambdaz(nca; recompute=false, verbose=verbose, kwargs...)
    auc += inf(_clast, _tlast, λz)
  end

  return auc
end

"""
  auc(nca::NCASubject; auctype::Symbol, method::Symbol, interval=nothing, normalize=false, kwargs...)

Compute area under the curve (AUC) by linear trapezoidal rule `(method =
:linear)` or by log-linear trapezoidal rule `(method = :linuplogdown)`.
"""
function auc(nca::NCASubject; auctype=:inf, interval=nothing, kwargs...)
  if interval isa Tuple || interval === nothing
    return auc_nokwarg(nca, auctype, interval; kwargs...)
  elseif interval isa AbstractArray
    f = let nca=nca, auctype=auctype, kwargs=kwargs
      i->auc_nokwarg(nca, auctype, i; kwargs...)
    end
    return map(f, interval)
  else
    error("""
          interval must be a tuple or an array of tuples, e.g.,
          auc(nca; interval=[(0,1.), (2, 3.)])
          """)
  end
end

@inline function auc_nokwarg(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}, auctype, interval; normalize=false, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  sol = _auc(nca, interval, auclinear, auclog, extrapaucinf, eltype(AUC); auctype=auctype, isauc=true, kwargs...)
  normalize && (sol = normalizedose(sol, nca))
  return sol
end

"""
  aumc(nca::NCASubject; method::Symbol, interval=(0, Inf), kwargs...)

Compute area under the first moment of the concentration (AUMC) by linear
trapezoidal rule `(method = :linear)` or by log-linear trapezoidal rule
`(method = :linuplogdown)`.
"""
function aumc(nca; auctype=:inf, interval=nothing, kwargs...)
  if interval isa Tuple || interval === nothing
    return aumc_nokwarg(nca, auctype, interval; kwargs...)
  elseif interval isa AbstractArray
    f = let nca=nca, auctype=auctype, kwargs=kwargs
      i->aumc_nokwarg(nca, auctype, i; kwargs...)
    end
    return map(f, interval)
  else
    error("""
          interval must be a tuple or an array of tuples, e.g.,
          aumc(nca; interval=[(0,1.), (2, 3.)])
          """)
  end
end

@inline function aumc_nokwarg(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}, auctype, interval; kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  sol = _auc(nca, interval, aumclinear, aumclog, extrapaumcinf, eltype(AUMC); auctype=auctype, isauc=false, kwargs...)
  return sol
end

"""
    auclast(nca::NCASubject; method::Symbol, interval=nothing, kwargs...)

Alias for `auc(subj; auctype=:last)`.
"""
auclast(nca::NCASubject; kwargs...) = auc(nca; auctype=:last, kwargs...)

"""
    aumclast(nca::NCASubject; method::Symbol, interval=nothing, kwargs...)

Alias for `aumc(subj; auctype=:last)`.
"""
aumclast(nca::NCASubject; kwargs...) = aumc(nca; auctype=:last, kwargs...)

"""
    auctau(subj::NCASubject; method::Symbol, kwargs...)

Alias for `auctau(subj; auctype=:last, interval=(zero(τ), τ))`.
"""
function auctau(nca::NCASubject; kwargs...)
  τ = tau(nca; kwargs...)
  τ === missing && return missing
  return auc(nca; auctype=:last, interval=(zero(τ), τ), kwargs...)
end

"""
    aumctau(subj::NCASubject; method::Symbol, kwargs...)

Alias for `aumctau(subj; auctype=:last, interval=(zero(τ), τ))`.
"""
function aumctau(nca::NCASubject; kwargs...)
  τ = tau(nca; kwargs...)
  τ === missing && return missing
  return aumc(nca; auctype=:last, interval=(zero(τ), τ), kwargs...)
end

function auc_extrap_percent(nca::NCASubject; kwargs...)
  auclast = auc(nca; auctype=:last, kwargs...)
  _auc  = (nca.dose !== nothing && nca.dose.ss) ? auctau(nca; auctype=:inf, kwargs...) : auc(nca; auctype=:inf, kwargs...)
  (_auc-auclast)/_auc * 100
end

function auc_back_extrap_percent(nca::NCASubject; interval=nothing, kwargs...)
  (nca.dose !== nothing && nca.dose.formulation === IVBolus) || return missing
  aucinf = auc(nca; auctype=:inf, kwargs...)
  auc_0 = nca.auc_0[1]
  auc_0/aucinf * 100
end

function aumc_extrap_percent(nca::NCASubject; kwargs...)
  aumcinf  = aumc(nca; auctype=:inf, kwargs...)
  aumclast = aumc(nca; auctype=:last, kwargs...)
  (aumcinf-aumclast)/aumcinf * 100
end

function loglinreg(x, y)
  # Filter out non-positive observations
  _x, ly = similar(x, 0), similar(y, typeof(log(oneunit(eltype(y)))), 0)
  for (i, yᵢ) in enumerate(y)
    if yᵢ > zero(yᵢ)
      push!(ly, log(yᵢ))
      push!(_x, x[i])
    end
  end

  points = length(ly)
  if points < 3
    return missing
  end

  x̄  = Statistics.mean(_x)
  lȳ = Statistics.mean(ly)
  Mxx = Statistics.varm(_x, x̄)
  Mxy = Statistics.covm(_x, x̄, ly, lȳ)
  β̂₁ = Mxy/Mxx
  β̂₀ = lȳ - x̄*β̂₁
  R² = var(lȳ - β̂₀ - β̂₁*xᵢ for xᵢ in _x)/Statistics.varm(ly, lȳ)
  return (β̂₀=β̂₀, β̂₁=β̂₁, R²=R², adjR²=1 - (1 - R²)*(points - 1)/(points - 2), points=points)
end

"""
    lambdaz(nca::NCASubject; concthreshold=1e-10, threshold=10, idxs=nothing) -> lambdaz

    Calculate terminal elimination rate constant ``λz`` with the end point > `concthreshold`.
"""
function lambdaz(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT};
                 adjr2factor=0.0001, threshold=100, idxs=nothing, slopetimes=nothing, recompute=true, verbose=true,
                 concthreshold=1e-10, # in long simulation, the concentration could go ~eps(1.0), so we need to pick the end at `concthreshold`
                 kwargs...
                )::Union{Missing,eltype(Z)} where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  if iscached(nca, :lambdaz) && !recompute
    return _first(nca.lambdaz)
  end
  _F = eltype(F)
  _Z = eltype(Z)
  if !(idxs === nothing) && !(slopetimes === nothing)
    throw(ArgumentError("you must provide only one of idxs or slopetimes"))
  end
  conc, time = nca.conc, nca.time
  maxadjr2::_F = -oneunit(_F)
  adjr2factor *= oneunit(_F)
  λ::_Z = zero(_Z)
  time′ = reinterpret(typeof(one(eltype(time))), time)
  conc′ = reinterpret(typeof(one(eltype(conc))), conc)
  _, cmaxidx = conc_extreme(conc, eachindex(conc), <)
  idx2 = findlast(x->x > oneunit(x) * concthreshold, nca.conc)
  valid = false # check if lambdaz is calculated at all
  if slopetimes === nothing && idxs === nothing
    # we can include Cmax after infusion
    tmax = time[cmaxidx]
    isafterinfusion = nca.dose !== nothing && (nca.dose.formulation === IVBolus || (nca.dose.formulation === IVInfusion && tmax >= nca.dose.duration))
    validpoints = isafterinfusion ? idx2-cmaxidx+1 : idx2-cmaxidx
    m = min(threshold, validpoints) - 1
    if m < 2
      if verbose
        @info "ID $(nca.id) errored: lambdaz calculation needs at least three data points between Cmax and the last positive concentration"
      end
      cachelambdaz!(nca)
      setrun_status!(nca, :NotEnoughDataAfterCmax)
      return missing
    end
    for i in 2:m
      idxs = idx2-i:idx2
      x = time′[idxs]
      y = conc′[idxs]
      _fit = loglinreg(x, y)
      _fit === missing && continue
      adjr2 = oneunit(_F) * _fit.adjR²
      if adjr2 > (maxadjr2 - adjr2factor)
        maxadjr2 = adjr2
        r2 = oneunit(_F) * _fit.R²
        λ = oneunit(_Z) * _fit.β̂₁
        intercept = _fit.β̂₀
        firstpoint = time[idxs[1]]
        lastpoint = time[idxs[end]]
        points = _fit.points
        valid = true
      end
    end
  else
    points = idxs === nothing ? length(slopetimes) : length(idxs)
    points < 3 && throw(ArgumentError("lambdaz must be calculated from at least three data points"))
    idxs === nothing && (idxs = findall(x->x in slopetimes, time))
    length(idxs) != points && throw(ArgumentError("elements slopetimes must occur in nca.time"))
    x = time′[idxs]
    y = conc′[idxs]
    _fit = loglinreg(x, y)
    if _fit !== missing
      λ = oneunit(_Z) * _fit.β̂₁
      intercept = _fit.β̂₀
      r2 = oneunit(_F) * _fit.R²
      firstpoint = time[idxs[1]]
      lastpoint = time[idxs[end]]
      maxadjr2 = oneunit(_F) * _fit.adjR²
      valid = true
    end
  end
  if !valid
    verbose && @info "ID $(nca.id) errored: lambdaz calculation failed because of non-positive concentration"
    cachelambdaz!(nca)
    setrun_status!(nca, :ConcentrationIsNotPositive)
    return missing
  end
  if λ ≥ zero(λ)
    verbose && @info "ID $(nca.id) errored: the estimated slope is not negative, got $λ"
    cachelambdaz!(nca)
    setrun_status!(nca, :SlopeIsNotNegative)
    return missing
  end
  _lambdaz = -λ
  cachelambdaz!(nca, _lambdaz, points, firstpoint, lastpoint, r2, maxadjr2, intercept)
  return _lambdaz
end

function cachelambdaz!(subj::NCASubject, lambdaz=missing, points=missing, firstpoint=missing, lastpoint=missing, r2=missing, maxadjr2=missing, intercept=missing)
  lambdaz === missing && return nothing
  if subj.lambdaz isa AbstractArray
    subj.lambdaz[1] = lambdaz
    subj.points[1] = points
    subj.firstpoint[1] = firstpoint
    subj.lastpoint[1] = lastpoint
    subj.r2[1] = r2
    subj.adjr2[1] = maxadjr2
    subj.intercept[1] = intercept
  else
    subj.lambdaz = lambdaz
    subj.points = points
    subj.firstpoint = firstpoint
    subj.lastpoint = lastpoint
    subj.r2 = r2
    subj.adjr2 = maxadjr2
    subj.intercept = intercept
  end
  return nothing
end

"""
  lambdaznpoints(nca::NCASubject; kwargs...)

Give the number of points that is used in the `λz` calculation. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdaznpoints(nca::NCASubject; kwargs...) = (lambdaz(nca; recompute=false, kwargs...); _first(nca.points))

"""
  lambdaztimefirst(nca::NCASubject; kwargs...)

Give the first time point that is used in the `λz` calculation. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdaztimefirst(nca::NCASubject; kwargs...) = (lambdaz(nca; recompute=false, kwargs...); _first(nca.firstpoint))

"""
  lambdaztimelast(nca::NCASubject; kwargs...)

Give the last time point that is used in the `λz` calculation. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdaztimelast(nca::NCASubject; kwargs...) = (lambdaz(nca; recompute=false, kwargs...); _first(nca.lastpoint))

"""
    span(nca::NCASubject; kwargs...)

Calculate span
"""
span(nca::NCASubject; kwargs...) = (lambdaztimelast(nca; recompute=false, kwargs...) - lambdaztimefirst(nca; recompute=false, kwargs...)) / thalf(nca; recompute=false, kwargs...)

"""
  lambdazintercept(nca::NCASubject; kwargs...)

Give the `y`-intercept in the log-linear scale when calculating `λz`. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdazintercept(nca::NCASubject; kwargs...) = (lambdaz(nca; recompute=false, kwargs...); _first(nca.intercept))


"""
  lambdazr2(nca::NCASubject; kwargs...)

Give the coefficient of determination (``r²``) when calculating `λz`. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdazr2(nca::NCASubject; kwargs...) = (lambdaz(nca; recompute=false, kwargs...); _first(nca.r2))

"""
  lambdazr(nca::NCASubject; kwargs...)

Give the correlation coefficient (``r``) when calculating `λz`. Please
calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdazr(nca::NCASubject; kwargs...) = (lambdaz(nca; recompute=false, kwargs...); sqrt(_first(nca.r2)))

"""
  lambdazadjr2(nca::NCASubject; kwargs...)

Give the adjusted coefficient of determination (``adjr²``) when calculating
`λz`. Please calculate `lambdaz` before calculating this quantity.

See also [`lambdaz`](@ref).
"""
lambdazadjr2(nca::NCASubject; kwargs...) = (lambdaz(nca; recompute=false, kwargs...); _first(nca.adjr2))
