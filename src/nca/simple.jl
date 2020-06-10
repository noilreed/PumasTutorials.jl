"""
    clast(nca::NCASubject)

Calculate `clast`
"""
function clast(nca::NCASubject; pred=false, kwargs...)
  if pred
    λz = lambdaz(nca; recompute=false, kwargs...)
    intercept = lambdazintercept(nca; recompute=false, kwargs...)
    return exp(intercept - λz*tlast(nca))*oneunit(eltype(nca.conc))
  else
    idx = nca.lastidx
    return idx === -1 ? missing : nca.conc[idx]
  end
end

"""
    tlast(nca::NCASubject)

Calculate `tlast`
"""
function tlast(nca::NCASubject; kwargs...)
  idx = nca.lastidx
  return idx === -1 ? missing : nca.time[idx]
end

# This function uses ``-1`` to denote missing as after checking `conc` is
# strictly great than ``0``.
function ctlast_idx(conc, time; llq=nothing)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findlast(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

function ctfirst_idx(conc, time; llq=nothing)
  llq === nothing && (llq = zero(eltype(conc)))
  # now we assume the data is checked
  f = x->(ismissing(x) || x<=llq)
  @inbounds idx = findfirst(!f, conc)
  idx === nothing && (idx = -1)
  return idx
end

"""
    tmax(nca::NCASubject; interval=nothing, kwargs...)

Calculate ``T_{max}_{t_1}^{t_2}``
"""
function tmax(nca::NCASubject; interval=nothing, kwargs...)
  if interval isa Union{Tuple, Nothing}
    return ctextreme(nca; interval=interval, kwargs...)[2]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctextreme(nca; interval=i, kwargs...)[2]
    end
    map(f, interval)
  end
end

"""
    cmax(nca::NCASubject; normalize=false, interval=nothing, kwargs...)

Calculate ``C_{max}_{t_1}^{t_2}``
"""
function cmax(nca::NCASubject; kwargs...)
  dose = nca.dose
  cmax′ = _cmax(nca; kwargs...)
  if dose !== nothing && dose.ss
    return cmax′/accumulationindex(nca)
  else
    return cmax′
  end
end

"""
    cmaxss(nca::NCASubject; normalize=false, kwargs...)

Calculate ``C_{maxss}``
"""
function cmaxss(nca::NCASubject; interval=nothing, kwargs...)
  dose = nca.dose
  cmax′ = _cmax(nca; kwargs...)
  if dose === nothing || (!dose.ss && dose.ii > zero(dose.ii))
    return cmax′*accumulationindex(nca)
  else # SS, so `cmax′` is actually `cmax_ss`
    return cmax′
  end
end

function _cmax(nca::NCASubject; interval=nothing, normalize=false, kwargs...)
  if interval isa Union{Tuple, Nothing}
    sol = ctextreme(nca; interval=interval, kwargs...)[1]
  else
    f = let nca=nca, kwargs=kwargs
      i -> ctextreme(nca; interval=i, kwargs...)[1]
    end
    sol = map(f, interval)
  end
  normalize ? normalizedose(sol, nca) : sol
end

"""
    ctau(nca::NCASubject; method=:linear, kwargs...)

Calculate concentration at τ
"""
ctau(nca::NCASubject; method=:linear, kwargs...) = interpextrapconc(nca, tau(nca; kwargs...); method=method, kwargs...)

"""
    cmin(nca::NCASubject; normalize=false, interval=nothing, kwargs...)

Calculate ``C_{min}_{t_1}^{t_2}``
"""
function cmin(nca::NCASubject; normalize=false, kwargs...)
  dose = nca.dose
  cmin′ = _cmin(nca; kwargs...)
  if dose !== nothing && dose.ss
    return cmin′/accumulationindex(nca)
  else
    return cmin′
  end
end

"""
    cminss(nca::NCASubject; normalize=false, kwargs...)

Calculate ``C_{minss}``
"""
function cminss(nca::NCASubject; interval=nothing, kwargs...)
  dose = nca.dose
  cmin′ = _cmin(nca; kwargs...)
  if dose === nothing || (!dose.ss && dose.ii > zero(dose.ii))
    return cmin′*accumulationindex(nca)
  else # SS, so `cmin′` is actually `cmin_ss`
    return cmin′
  end
end

function _cmin(nca::NCASubject; kwargs...)
  (nca.dose !== nothing && nca.dose.formulation === EV && !nca.dose.ss) && return missing
  val = ctextreme(nca, >)[1]
  return val
end

"""
    tmin(nca::ncasubject; kwargs...)

Calculate time of minimum observed concentration
"""
function tmin(nca::NCASubject; kwargs...)
  val = ctextreme(nca, >)[2]
  return val
end

function ctextreme(nca::NCASubject, lt=<; interval=nothing, kwargs...)
  conc, time = nca.conc, nca.time
  c0′ = (interval === nothing || iszero(interval[1])) ? c0(nca, true) : missing
  if interval === nothing
    val, idx = conc_extreme(conc, eachindex(conc), lt)
    t = time[idx]
    if !ismissing(c0′)
      val, t, idx = lt(val, c0′) ? (c0′, zero(t), -1) : (val, t, idx) # calculate max by default
    end
    return (val, t, idx)
  elseif interval isa Tuple
    @assert interval[1] < interval[2] "t0 must be less than t1"
    interval[1] > time[end] && throw(ArgumentError("t0 is longer than observation time"))
    idx1, idx2 = let (lo, hi)=interval
      findfirst(t->t>=lo, time),
      findlast( t->t<=hi, time)
    end
    val, idx = conc_extreme(conc, idx1:idx2, lt)
    t = time[idx]
    if !ismissing(c0′)
      val, t, idx = lt(val, c0′) ? (c0′, zero(t), -1) : (val, t, idx) # calculate max by default
    end
  else
    throw(ArgumentError("interval must be nothing or a tuple."))
  end
  return (val, t, idx)
end

maxidx(subj::NCASubject) = subj.maxidx == -2 ? (subj.maxidx = ctextreme(subj)[3]) : subj.maxidx

function conc_extreme(conc, idxs, lt)
  local val, idx
  for i in idxs
    conci = conc[i]
    !ismissing(conci) && (val = conci; idx=i; break) # find a sensible initialization
  end
  for i in idxs
    if !ismissing(conc[i])
      lt(val, conc[i]) && (val = conc[i]; idx=i)
    end
  end
  return val, idx
end

"""
    thalf(nca::NCASubject; kwargs...)

Calculate half life time.
"""
thalf(nca::NCASubject; kwargs...) = log(2)./lambdaz(nca; recompute=false, kwargs...)

"""
    cl(nca::NCASubject; kwargs...)

Calculate total drug clearance.
"""
function cl(nca::NCASubject; auctype=nothing, kwargs...)
  nca.dose === nothing && return missing
  if nca.dose.ss
    1/normalizedose(auctau(nca; kwargs...), nca)
  else
    1/normalizedose(auc(nca; kwargs...), nca)
  end
end

function _clf(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation === EV ? cl(nca; kwargs...) : missing
end
function _cl(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation !== EV ? cl(nca; kwargs...) : missing
end

"""
    vss(nca::NCASubject; kwargs...)

Calculate apparent volume of distribution at equilibrium for IV bolus doses.
``V_{ss} = MRT * CL``.
"""
vss(nca::NCASubject; kwargs...) = mrt(nca; kwargs...)*cl(nca; kwargs...)

"""
    vz(nca::NCASubject; kwargs...)

Calculate the volume of distribution during the terminal phase.
``V_z = 1/(AUC⋅λz)`` for dose normalizedosed `AUC`.
"""
function vz(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  λ = lambdaz(nca; recompute=false, kwargs...)
  if nca.dose.ss
    τauc = auctau(nca; kwargs...)
    @. 1/(normalizedose(τauc, nca.dose) * λ)
  else
    aucinf = normalizedose(auc(nca; kwargs...), nca)
    @. 1/(aucinf * λ)
  end
end

function _vzf(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation === EV ? vz(nca; kwargs...) : missing
end
function _vz(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  nca.dose.formulation !== EV ? vz(nca; kwargs...) : missing
end

"""
    bioav(nca::NCASubject; ithdose::Integer, kwargs...)

Bioavailability is the ratio of two AUC values.
``Bioavailability (F) = (AUC_0^\\infty_{po}/Dose_{po})/(AUC_0^\\infty_{iv}/Dose_{iv})``
"""
function bioav(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}; ithdose=missing, kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  n = nca.dose isa AbstractArray ? length(nca.dose) : 1
  ismissing(ithdose) && return n == 1 ? missing : fill(missing, n)
  multidose = n > 1
  # if there is only a single dose
  multidose || return missing
  # if we only have IV or EV
  length(unique(getfield.(nca.dose, :formulation))) == 1 && return fill(missing, length(nca.dose))
  # initialize
  auc_0_inf_po = auc_0_inf_iv = zero(eltype(AUC))/oneunit(first(nca.dose).amt) # normalizedosed
  sol = zeros(typeof(ustrip(auc_0_inf_po)), axes(nca.dose))
  refdose = subject_at_ithdose(nca, ithdose)
  refauc  = normalizedose(auc(refdose; auctype=:inf, kwargs...), refdose)
  map(eachindex(nca.dose)) do idx
    subj = subject_at_ithdose(nca, idx)
    sol[idx] = normalizedose(auc(subj; auctype=:inf, kwargs...), subj)/refauc
  end
  return sol
end

"""
    tlag(nca::NCASubject; kwargs...)

The time prior to the first increase in concentration.
"""
function tlag(nca::NCASubject; kwargs...)
  nca.dose === nothing && return missing
  (nca.rate === nothing && nca.dose.formulation !== EV) && return missing
  idx = findfirst(c->c > nca.llq, nca.conc)-1
  return 1 <= idx <= length(nca.time) ? nca.time[idx] : zero(nca.time[1])
end

"""
    mrt(nca::NCASubject; kwargs...)

Mean residence time from the time of dosing to the time of the last measurable
concentration.

IV infusion:
  ``AUMC/AUC - TI/2`` where ``TI`` is the length of infusion.
non-infusion:
  ``AUMC/AUC``
"""
function mrt(nca::NCASubject; auctype=:inf, kwargs...)
  dose = nca.dose
  dose === nothing && return missing
  ti2 = dose.duration*1//2
  if dose.ss
    auctype === :last && return missing
    τ = tau(nca; kwargs...)
    aumcτ = aumctau(nca; kwargs...)
    aucτ = auctau(nca; kwargs...)
    _auc = auc(nca; auctype=:inf, kwargs...)
    quotient = (aumcτ + τ*(_auc - aucτ)) / aucτ
    dose.formulation === IVInfusion && (quotient -= ti2)
    return quotient
  else
    quotient = aumc(nca; auctype=auctype, kwargs...) / auc(nca; auctype=auctype, kwargs...)
    dose.formulation === IVInfusion && (quotient -= ti2)
    return quotient
  end
end

"""
    tau(nca::NCASubject)

Dosing interval. For multiple dosing only.
"""
function tau(nca::NCASubject; kwargs...)
  nca.dose !== nothing || return missing
  !iszero(nca.dose.ii) && return nca.dose.ii
  return missing
  # do not guess tau
  #dose = nca.dose
  #(dose === nothing || AUC isa Number) && return missing
  #dose isa NCADose && return nca.abstime[nca.lastidx]-dose.time
  #return dose[end].time-dose[end-1].time
end

"""
    cavgss(nca::NCASubject)

Average concentration over one period. ``C_{avgss} = AUC_{tau}/Tau``.
"""
function cavgss(nca::NCASubject; pred=false, kwargs...)
  nca.dose === nothing && return missing
  subj = nca.dose isa NCADose ? nca : subject_at_ithdose(nca, 1)
  ii = tau(subj)
  if nca.dose.ss
    return auctau(subj; kwargs...)/ii
  else
    return auc(subj; auctype=:inf, pred=pred)/ii
  end
end

"""
    fluctuation(nca::NCASubject; usetau=false, kwargs...)

Peak trough fluctuation over one dosing interval at steady state.
``Fluctuation = 100*(C_{maxss} - C_{minss})/C_{avgss}`` (usetau=false) or
``Fluctuation = 100*(C_{maxss} - C_{tau})/C_{avgss}`` (usetau=true)
"""
function fluctuation(nca::NCASubject; usetau=false, kwargs...)
  _cmin = usetau ? ctau(nca) : cminss(nca)
  100*(cmaxss(nca) - _cmin)/cavgss(nca; kwargs...)
end

"""
    accumulationindex(nca::NCASubject; kwargs...)

Theoretical accumulation ratio. ``Accumulation_index = 1/(1-exp(-λ_z*Tau))``.
"""
function accumulationindex(nca::NCASubject; kwargs...)
  tmp = -lambdaz(nca; recompute=false, kwargs...)*tau(nca)
  one(tmp)/(one(tmp)-exp(tmp))
end

"""
    swing(nca::NCASubject; usetau=false, kwargs...)

``swing = (C_{maxss}-C_{minss})/C_{minss}`` (usetau=false) or ``swing = (C_{maxss}-C_{tau})/C_{tau}`` (usetau=true)
"""
function swing(nca::NCASubject; usetau=false, kwargs...)
  _cmin = usetau ? ctau(nca) : cminss(nca)
  sw = (cmaxss(nca) - _cmin) ./ _cmin
  (sw === missing || isinf(sw)) ? missing : sw
end

"""
    c0(nca::NCASubject; kwargs...)

Estimate the concentration at dosing time for an IV bolus dose.
"""
function c0(subj::NCASubject, returnev=false; verbose=true, kwargs...) # `returnev` is not intended to be used by users
  subj.dose === nothing && return missing
  t1 = ustrip(subj.time[1])
  isurine = subj.rate !== nothing
  if subj.dose.formulation !== IVBolus || isurine
    return !returnev ? missing :
               (isurine || !subj.dose.ss) ? zero(subj.conc[1]) :
               ctau(subj; kwargs...)
  end
  iszero(t1) && return subj.conc[1]
  t2 = ustrip(subj.time[2])
  c1 = ustrip(subj.conc[1]); c2 = ustrip(subj.conc[2])
  iszero(c1) && return c1
  if c2 >= c1 && verbose
    @warn "c0: This is an IV bolus dose, but the first two concentrations are not decreasing. If `conc[i]/conc[i+1] > 0.8` holds, the back extrapolation will be computed internally for AUC and AUMC, but will not be reported."
  end
  if c2 < c1 || (returnev && c1/c2 > 0.8)
    dosetime = ustrip(subj.dose.time)
    c0 = exp(log(c1) - (t1 - dosetime)*(log(c2)-log(c1))/(t2-t1))*oneunit(eltype(subj.conc))
  else
    c0 = missing
  end
  return c0
end

run_status(subj::NCASubject; kwargs...) = subj.run_status

superposition(pop::NCAPopulation, args...; kwargs...) = reduce(vcat, map(subj->superposition(subj, args...; kwargs...), pop))
"""
    superposition(data::Union{NCAPopulation,NCASubject}; ii, ndoses=5, amt=nothing, steadystatetol=3e-2, method=:linear)

Superposition calculation.

Arguments:
    - ii: interdose interval
    - ndoses: number of doses. It can be a positive integer or `Inf`.
    - amt: dose amount. It defaults to the dose amount of the subject.
    - steadystatetol: steady state tolerance. Superposition computation terminates when ``abs(1 - c(t-ii) / c(t)) <= steadystatetol``.
    - method: the method for interpolation. It defaults to `:linear`.
"""
function superposition(subj::NCASubject, args...;
                       ii::Number, ndoses::Union{Integer,AbstractFloat}=5,
                       amt=nothing, steadystatetol::Number=3e-2, method=:linear,
                       kwargs...)
  subj.dose === Nothing && throw(ArgumentError("Dose must be known to compute superposition"))
  ndoses >= 1 || throw(ArgumentError("ndoses must be >= 1"))
  !(ndoses isa Integer) && ndoses != Inf && throw(ArgumentError("ndoses must be an integer or Inf"))
  doseamt = ismultidose(subj) ? subj.dose[1].amt  : subj.dose.amt
  amt = amt === nothing ? doseamt : amt
  dosescaling = amt / doseamt
  if ismultidose(subj)
    time = subj.time[end]
    conc = subj.conc[end]
    len = length(subj.time[end])
    outtime = [t for t in subj.time]
    outconc = [dosescaling .* c for c in subj.conc]
    occasion = [fill(i, length(conc)) for (i, conc) in enumerate(outconc)]
    addl = length(occasion) - 1
    outamt = [map(i->i==1 ? amt : zero(amt), 1:length(conc)) for conc in outconc]
    # we need the first dose to do superposition
    subj = subject_at_ithdose(subj, 1)
  else
    time = subj.time
    conc = subj.conc
    len = length(time)
    outtime = [time]
    outconc = [dosescaling .* conc]
    occasion = [fill(1, len)]
    addl = 0
    outamt = fill(zero(amt), len)
    outamt[1] = amt
    outamt = [outamt]
  end
  dosetype = subj.dose.formulation
  route = dosetype === IVInfusion ? "inf" : dosetype === IVBolus ? "iv" : "ev"
  prevclast = outconc[end][findlast(!iszero, outconc[end])]
  nii = 0
  # Stop either when reaching steady-state or reaching ndoses
  for ndose in 2:ndoses
    addl += 1
    nii += 1
    prevconc = outconc[end]
    abstime = time .+ nii*ii
    currconc = prevconc + dosescaling * interpextrapconc(subj, abstime; method=method, kwargs...)
    push!(outconc, currconc)
    push!(outtime, abstime)
    push!(occasion, fill(addl+1, len))
    push!(outamt, outamt[end])
    currclast = currconc[findlast(!iszero, currconc)]
    abs(one(currclast) - prevclast / currclast) <= steadystatetol && break
    prevclast = currclast
  end
  time′ = reduce(vcat, outtime)
  amt′ = reduce(vcat, outamt)
  ii′ = map(x->iszero(x) ? missing : ii, amt′)
  addl′ = map(x->iszero(x) ? missing : addl, amt′)
  df = DataFrame(id=subject_id(subj), time=time′, conc=reduce(vcat, outconc),
                 amt=amt′, ii=ii′, addl=addl′, occasion=reduce(vcat, occasion), route=route)
  # take the part with monotone time
  monotime_idxs = [length(time′)]
  t1 = time′[end]
  # we sweep from the end to take data from the most recent dose
  for idx in reverse(eachindex(time′))
    t = time′[idx]
    if t < t1 # we need strict `<`
      t1 = t
      push!(monotime_idxs, idx)
    end
  end
  df = df[reverse(monotime_idxs), :] # reverse again to get the normal order

  # add group columns
  if subj.group !== nothing
    if subj.group isa Pair
      df[!, Symbol(subj.group[1])] .= subj.group[2]
    else
      for group in subj.group
        df[!, Symbol(group[1])] .= group[2]
      end
    end
  end
  return df
end

subject_id(subj::NCASubject; kwargs...) = subj.id
n_samples(subj::NCASubject; kwargs...) = length(subj.time)
doseamt(subj::NCASubject; kwargs...) = hasdose(subj) ? subj.dose.amt : missing
dosetype(subj::NCASubject; kwargs...) = hasdose(subj) ? string(subj.dose.formulation) : missing

urine_volume(subj::NCASubject; kwargs...) = subj.volume === nothing ? missing : sum(subj.volume)
amount_recovered(subj::NCASubject; kwargs...) = sum(prod, zip(subj.conc, subj.volume))
function percent_recovered(subj::NCASubject; kwargs...)
  subj.dose === nothing && return missing
  sum(amount_recovered(subj) / subj.dose.amt) * 100
end
