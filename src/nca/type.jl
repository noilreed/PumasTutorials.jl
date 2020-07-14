using Unitful
using Markdown

"""
    Formulation

Type of formulations. There are IV (intravenous) and EV (extravascular).
"""
@enum Formulation IVBolus IVInfusion EV# DosingUnknown
# Formulation behaves like scalar
Broadcast.broadcastable(x::Formulation) = Ref(x)

"""
    NCADose

`NCADose` takes the following arguments
- `time`: time of the dose
- `amt`: The amount of dosage
- `formulation`: Type of formulation, `NCA.IVBolus`, `NCA.IVInfusion` or `NCA.EV`
- `ii`: interdose interval
- `ss`: steady-state
"""
struct NCADose{T,A}
  time::T
  amt::A
  duration::T
  formulation::Formulation
  ii::T
  ss::Bool
  function NCADose(time, amt, duration::D, formulation, ii=zero(time), ss=false) where D
    duration′ = D === Nothing ? zero(time) : duration
    formulation′ = formulation === EV ? EV : iszero(duration′) ? IVBolus : IVInfusion
    time, ii = promote(time, ii)
    ii < zero(ii) && throw(ArgumentError("ii must be non-negative. Got ii=$ii"))
    if ii <= zero(ii) && ss
      throw(ArgumentError("ii must be greater than zero when ss=1. Got ii=$ii"))
    end
    return new{typeof(time), typeof(amt)}(time, amt, duration′, formulation′, ii, ss)
  end
end

# NCADose should behave like a scalar in broadcast
Broadcast.broadcastable(x::NCADose) = Ref(x)
Base.first(x::NCADose) = x

function Base.show(io::IO, n::NCADose)
  println(io, "NCADose:")
  println(io, "  time:         $(n.time)")
  println(io, "  amt:          $(n.amt)")
  println(io, "  duration:     $(n.duration)")
  println(io, "  formulation:  $(n.formulation)")
  print(  io, "  ss:           $(n.ss)")
end

# any changes in here must be reflected to ./simple.jl, too
mutable struct NCASubject{C,T,TT,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  # allow id to be non-int, e.g. 001-100-1001 or STUD011001
  id::ID
  group::G
  conc::C
  rate::R
  # `time` will be time after dose (TAD) for multiple dosing, and it will be in
  # the form of [TAD₁, TAD₂, ..., TADₙ] where `TADᵢ` denotes the TAD for the
  # `i`th dose, which is a vector. Hence, `time` could be a vector of vectors.
  time::T
  start_time::Union{Nothing, T}
  end_time::Union{Nothing, T}
  volume::V
  # `abstime` is always a vector, and it leaves untouched except BLQ handling
  abstime::TT
  maxidx::I
  lastidx::I
  dose::D
  lambdaz::Union{Missing,Z}
  llq::Union{Missing,N}
  r2::Union{Missing,F}
  adjr2::Union{Missing,F}
  intercept::Union{Missing,F}
  firstpoint::Union{Missing,tEltype}
  lastpoint::Union{Missing,tEltype}
  points::Union{Missing,P}
  auc_last::Union{Missing,AUC}
  auc_0::Union{Missing,AUC}
  aumc_last::Union{Missing,AUMC}
  method::Symbol
  run_status::RT
  function NCASubject(id, group, conc, rate, time, start_time, end_time, volume,
                      abstime, maxidx, lastidx, dose, lambdaz, llq, r2, adjr2,
                      intercept, firstpoint, lastpoint, points, auc_last, auc_0,
                      aumc_last, method, run_status)
    new{typeof(conc), typeof(time), typeof(abstime), typeof(firstpoint), typeof(auc_last),
        typeof(aumc_last), typeof(dose), typeof(lambdaz), typeof(r2), typeof(llq), typeof(lastidx),
        typeof(points), typeof(id), typeof(group), typeof(volume), typeof(rate), typeof(run_status)
       }(id, group, conc, rate, time, start_time, end_time, volume, abstime,
         maxidx, lastidx, dose, lambdaz, llq, r2, adjr2, intercept, firstpoint,
         lastpoint, points, auc_last, auc_0, aumc_last, method, run_status)
  end
end

"""
    NCASubject(conc, time; concu=true, timeu=true, id=1, group=nothing, dose=nothing, llq=nothing, lambdaz=nothing, clean=true, check=true, kwargs...)

Constructs a NCASubject

Note that `llq` keyword argument still takes effects with the presence of the
`blq` data column, and `llq` defaults to `0`.

Setting `clean=false` disables all checks on `conc` and `time` to remove the
cost of checking and cleaning data. It should only be used when the data is for
sure "clean".
"""
function NCASubject(conc, time;
                    start_time=nothing, end_time=nothing, volume=nothing, concu=true, timeu=true, volumeu=true,
                    id=1, group=nothing, dose=nothing, llq=nothing,
                    lambdaz=nothing, clean=true, check=true, kwargs...)
  time isa AbstractRange && (time = collect(time))
  conc isa AbstractRange && (conc = collect(conc))
  time = tighten_container_eltype(time)
  conc = tighten_container_eltype(conc)

  conc = addunit(conc, concu)
  time = addunit(time, timeu)
  start_time = addunit(start_time, timeu)
  end_time = addunit(end_time, timeu)

  multidose = dose isa AbstractArray && length(dose) > 1
  if !multidose && dose isa AbstractArray
    dose = first(dose)
  end
  nonmissingeltype(x) = Base.nonmissingtype(eltype(x))
  unitconc = float(oneunit(nonmissingeltype(conc)))
  unittime = float(oneunit(nonmissingeltype(time)))
  llq === nothing && (llq = zero(unitconc))
  auc_proto = -unitconc * unittime
  aumc_proto = auc_proto * unittime
  intercept = r2_proto = ustrip(unittime/unitconc)
  _lambdaz = inv(unittime)
  lambdaz_proto = lambdaz === nothing ? _lambdaz : lambdaz
  if multidose
    # Dosing events maybe be longer than the observing time, so we need to chop
    # it off. As we don't assume doses are sorted, we cannot use
    # `findfirst(x->x.time >= time[1], dose):findlast(x->x.time <= time[end], dose)`.
    # Sorting events takes O(n⋅log(n)) time, while linear search takes O(n)
    # time, so we take are just going to use `filter`.
    dose = filter(x->time[1] <= x.time <= time[end], dose)
    n = length(dose)
    abstime = clean ? typeof(unittime)[] : time
    istad = all(x->iszero(x.time), dose)
    startidxs = istad ? findall(iszero, time) : nothing
    ct = let time=time, dose=dose
      map(1:length(dose)) do i
        idxs = ithdoseidxs(time, dose, startidxs, i; check=i==1) # only check once
        conci, timei = @view(conc[idxs]), @view(time[idxs])
        check && checkconctime(conci, timei; dose=dose, kwargs...)
        if !iszero(timei[1])
          timei = timei .- timei[1] # convert to TAD
        end
        if clean
          conci, timei, _, _ = cleanmissingconc(conci, timei; kwargs...)
          conci, timei = cleanblq(conci, timei; llq=llq, dose=dose, kwargs...)[1:2]
          # we need to check twice because cleanning changes the data
          check && checkconctime(conci, timei; dose=dose, kwargs...)
          append!(abstime, timei)
        end
        conci, timei
      end
    end
    conc = map(x->x[1], ct)
    time = map(x->x[2], ct)
    maxidx  = fill(-2, n)
    lastidx = @. ctlast_idx(conc, time; llq=llq)
    return NCASubject(id, group,
                      conc, nothing, time, #= no multidose urine =# nothing, nothing, nothing, abstime,
                      maxidx, lastidx, dose, fill(lambdaz_proto, n), llq,
                      fill(r2_proto, n), fill(r2_proto, n), fill(intercept, n),
                      fill(-unittime, n), fill(-unittime, n), fill(0, n),
                      fill(auc_proto, n), fill(auc_proto, n), fill(aumc_proto, n), :___, fill(:Success, n))
  end
  isurine = volume !== nothing
  volume = isurine ? volume .* volumeu : nothing
  if check
    if isurine
      checkconctime(conc, start_time; dose=dose, kwargs...)
      checkconctime(conc, end_time;   dose=dose, kwargs...)
    else
      checkconctime(conc, time; dose=dose, kwargs...)
    end
  end
  if !isurine && !iszero(time[1])
    if dose !== nothing
      dose.time == time[1] || throw(ArgumentError("The first observed time is not zero or dosetime. Got time[1] = $(time[1]), dosetime=$(dose.time)."))
    end
    time = time .- time[1] # convert to TAD
  end
  if clean
    if isurine
      conc, start_time, end_time, volume = cleanmissingconc(conc, start_time; end_time=end_time, volume=volume, kwargs...)
      conc, start_time, end_time, volume = cleanblq(conc, start_time; end_time=end_time, volume=volume, llq=llq, dose=dose, kwargs...)
    else
      conc, time, _, _ = cleanmissingconc(conc, time; kwargs...)
      conc, time, _, _ = cleanblq(conc, time; llq=llq, dose=dose, kwargs...)
    end
    # we need to check twice because cleanning changes the data
    if check
      if isurine
        checkconctime(conc, start_time; dose=dose, kwargs...)
        checkconctime(conc, end_time;   dose=dose, kwargs...)
      else
        checkconctime(conc, time; dose=dose, kwargs...)
      end
    end
  end
  if isurine
    Δt = @. end_time - start_time
    time = @. start_time + Δt/2
    rate = @.(volume*conc/Δt)
    auc_proto = -oneunit(eltype(rate)) * oneunit(eltype(time))
  else
    rate = nothing
  end
  abstime = time
  dose !== nothing && (dose = first(dose))
  maxidx = -2
  lastidx = ctlast_idx(conc, time; llq=llq)
  NCASubject(id, group,
             conc, rate, time, start_time, end_time, volume, abstime, maxidx, lastidx, dose, lambdaz_proto, llq,
             r2_proto,  r2_proto, intercept, unittime, unittime, 0,
             auc_proto, auc_proto, aumc_proto, :___, :Success)
end

"""
    showunits(nca:NCASubject[, indent])

Prints the units of concentration, time, auc, aumc, λz, and dose for `nca`. The optional argument `indent` adds `indent` number of spaces to the left of the list.
"""
showunits(nca::NCASubject, args...) = showunits(stdout, nca, args...)
function showunits(io::IO, subj::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}, indent=0) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  pad   = " "^indent
  if D <: NCADose
    doseT = D.parameters[2]
  elseif D <: AbstractArray
    doseT = eltype(D).parameters[2]
  else
    doseT = D
  end
  multidose = ismultidose(subj)
  _C = multidose ? eltype(C) : C
  _T = multidose ? eltype(T) : T
  _AUC = multidose ? eltype(AUC) : AUC
  _AUMC = multidose ? eltype(AUMC) : AUMC
  _Z = multidose ? eltype(Z) : Z
  if eltype(_C) <: Quantity
    conct = string(unit(eltype(_C)))
    timet = string(unit(eltype(_T)))
    AUCt  = string(unit(_AUC))
    AUMCt = string(unit(_AUMC))
    Zt    = string(unit(_Z))
    Dt    = D === Nothing ? D : string(unit(doseT))
  else
    conct = nameof(eltype(_C))
    timet = nameof(eltype(_T))
    AUCt  = _AUC
    AUMCt = _AUMC
    Zt    = _Z
    Dt    = doseT
  end
  println(io, "$(pad)concentration: $conct")
  println(io, "$(pad)time:          $timet")
  println(io, "$(pad)auc:           $AUCt")
  println(io, "$(pad)aumc:          $AUMCt")
  println(io, "$(pad)λz:            $Zt")
  print(  io, "$(pad)dose:          $Dt")
end

function Base.show(io::IO, n::NCASubject)
  println(io, "NCASubject:")
  println(io, "  ID: $(n.id)")
  n.group === nothing || println(io, "  Group: $(n.group)")
  pad = 4
  showunits(io, n, pad)
end

const NCAPopulation{T} = AbstractVector{T} where T <: NCASubject

Base.show(io::IO, pop::NCAPopulation) = show(io, MIME"text/plain"(), pop)
function Base.show(io::IO, ::MIME"text/plain", pop::NCAPopulation)
  ids = unique([subj.id for subj in pop])
  println(io, "NCAPopulation ($(length(ids)) subjects):")
  println(io, "  ID: $(ids)")
  first(pop).group === nothing || println(io, "  Group: $(unique([subj.group for subj in pop]))")
  showunits(io, first(pop), 4)
end

ismultidose(nca::NCASubject) = nca.dose isa AbstractArray
ismultidose(nca::NCAPopulation) = ismultidose(first(nca))

hasdose(nca::NCASubject) = nca.dose !== nothing
hasdose(nca::NCAPopulation) = hasdose(first(nca))

# Summary and report
macro defkwargs(sym, expr...)
  :((nca; kwargs...) -> $sym(nca; $(expr...), kwargs...)) |> esc
end

NCAReport(nca::NCASubject; kwargs...) = NCAReport([nca]; kwargs...)
function NCAReport(pop::NCAPopulation; pred=nothing, normalize=nothing, auctype=nothing, # strips out unnecessary user options
                   sigdigits=nothing, kwargs...)
  !hasdose(pop) && @warn "`dose` is not provided. No dependent quantities will be calculated"
  settings = Dict(:multidose=>ismultidose(pop), :subject=>(pop isa NCASubject))

  has_ev = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.formulation === EV :
                            any(d->d.formulation === EV, subj.dose)
  end
  has_iv = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.formulation === IVBolus :
                            any(d->d.formulation === IVBolus, subj.dose)
  end
  has_inf = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.formulation === IVInfusion :
                            any(d->d.formulation === IVInfusion, subj.dose)
  end
  has_ii = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.ii > zero(subj.dose.ii) :
                            any(d->d.ii > zero(d.ii), subj.dose)
  end
  is_ss = hasdose(pop) && any(pop) do subj
    subj.dose isa NCADose ? subj.dose.ss :
                            any(d->d.ss, subj.dose)
  end
  has_iv_inf = has_iv || has_inf
  is_urine = any(subj->subj.volume !== nothing, pop)
  #TODO:
  # add partial AUC
  # interval Cmax??? low priority
  if is_urine
    report_pairs = [
           "dose" => doseamt,
           "urine_volume" => urine_volume,
           "kel" => lambdaz,
           "tmax_rate" => tmax_rate,
           "tlag" => tlag,
           "max_rate" => max_rate,
           "rate_last" => rate_last,
           "mid_time_last" => mid_time_last,
           "amount_recovered" => amount_recovered,
           "percent_recovered" => percent_recovered,
           "aurc_last_obs" => @defkwargs(aurc, auctype=:last),
           "aurc_inf_obs" => aurc,
           "aurc_extrap_obs" => aurc_extrap_percent,
           "rate_last_pred" => @defkwargs(rate_last, pred=true),
           "aurc_inf_pred" => @defkwargs(aurc, pred=true),
           "aurc_extrap_pred" => @defkwargs(aurc_extrap_percent, pred=true),
           "aurc_last_d_obs" => @defkwargs(aurc, auctype=:last, normalize=true),
           "n_samples"          =>     n_samples,
           "rsq_kel"                =>     lambdazr2,
           "rsq_adjusted_kel"       =>     lambdazadjr2,
           "corr_kel"            =>     lambdazr,
           "n_points_kel" =>     lambdaznpoints,
           "intercept_kel" =>     lambdazintercept,
           "kel_t_low"     =>     lambdaztimefirst,
           "kel_t_high"     =>     lambdaztimelast,
           "span"               =>     span,
           "route"              =>     dosetype,
      ]
  else
    report_pairs = [
           # id is already computed when label=true.
           "dose"               =>     doseamt,
           has_ev && "tlag"     =>     tlag,
           "tmax"               =>     tmax,
           "cmax"               =>     cmax,
           has_ii && "cmaxss"   =>     cmaxss,
           has_iv && "c0"       =>     c0,
           "tlast"              =>     tlast,
           "clast"              =>     clast,
           "clast_pred"         =>     @defkwargs(clast, pred=true),
           "auclast"            =>     auclast,
           "kel"                =>     lambdaz,
           "half_life"          =>     thalf,
           is_ss  || "aucinf_obs"   =>     auc,
           is_ss  || "aucinf_pred"  =>     @defkwargs(auc, pred=true),
           has_ii && "auc_tau_obs"  =>     auctau,
           has_ii && "auc_tau_pred" =>     @defkwargs(auctau, pred=true),
           has_ii && "tau"      =>     tau,
           ((has_inf || has_ev) && is_ss) || has_iv && "tmin"         =>     tmin,
           ((has_inf || has_ev) && is_ss) || has_iv && "cmin"         =>     cmin,
           ((has_inf || has_ev) && is_ss) || has_iv && "cminss"       =>     cminss,
           has_ii && "ctau"         =>     ctau,
           has_ii && "cavgss"       =>     cavgss,
           has_iv_inf && "vz_obs"   =>     _vz,
           has_iv_inf && "cl_obs"   =>     _cl,
           has_ev && "vz_f_obs"     =>     _vzf,
           has_ev && "cl_f_obs"     =>     _clf,
           has_ev && "vz_f_pred"    =>     @defkwargs(_vzf, pred=true),
           has_ev && "cl_f_pred"    =>     @defkwargs(_clf, pred=true),
           has_iv_inf && "vz_pred"  =>     @defkwargs(_vz, pred=true),
           has_iv_inf && "cl_pred"  =>     @defkwargs(_cl, pred=true),
           has_iv_inf && "vss_obs"  => vss,
           has_iv_inf && "vss_pred" => @defkwargs(vss, pred=true),
           has_ii && "fluctuation"  =>     fluctuation,
           has_ii && "fluctuation_tau"    =>     @defkwargs(fluctuation, usetau=true),
           has_ii && "accumulation_index" =>     accumulationindex,
           "n_samples"          =>     n_samples,
           "cmax_dn"                 =>     @defkwargs(cmax, normalize=true),
           "auclast_dn"              =>     @defkwargs(auclast, normalize=true),
           has_ii || "aucinf_dn_obs" =>     @defkwargs(auc, normalize=true),
           "auc_extrap_obs"         =>     auc_extrap_percent,
           has_iv && "auc_back_extrap_obs" => auc_back_extrap_percent,
           has_ii || "aucinf_dn_pred"       =>     @defkwargs(auc, normalize=true, pred=true),
           "auc_extrap_pred"        =>     @defkwargs(auc_extrap_percent, pred=true),
           has_iv && "auc_back_extrap_pred" => @defkwargs(auc_back_extrap_percent, pred=true),
           "aumclast"               =>     aumclast,
           "aumcinf_obs"            =>     aumc,
           "aumc_extrap_obs"        =>     aumc_extrap_percent,
           "aumcinf_pred"           =>     @defkwargs(aumc, pred=true),
           "aumc_extrap_pred"       =>     @defkwargs(aumc_extrap_percent, pred=true),
           has_iv_inf && "mrtlast"      =>     @defkwargs(mrt, auctype=:last),
           has_iv_inf && "mrtinf_obs"   =>     @defkwargs(mrt, auctype=:inf),
           has_iv_inf && "mrtinf_pred"  =>     @defkwargs(mrt, auctype=:inf, pred=true),
           has_ii && "swing"        =>     swing,
           has_ii && "swing_tau"    =>     @defkwargs(swing, usetau=true),
           "n_samples_kel"          =>     lambdaznpoints,
           "rsq_kel"                =>     lambdazr2,
           "rsq_adj_kel"            =>     lambdazadjr2,
           "corr_kel"               =>     lambdazr,
           "intercept_kel"          =>     lambdazintercept,
           "kel_t_low"              =>     lambdaztimefirst,
           "kel_t_high"             =>     lambdaztimelast,
           "span"                   =>     span,
           "route"                  =>     dosetype,
           "run_status"             =>     run_status,
        ]
  end
  deleteat!(report_pairs, findall(x->x.first isa Bool, report_pairs))
  _names = map(x->Symbol(x.first), report_pairs)
  funcs = map(x->x.second, report_pairs)
  vals  = [f(pop; label = i == 1, kwargs...) for (i, f) in enumerate(funcs)]
  if sigdigits !== nothing
    for val in vals
      col = val[!,end]
      map!(col, col) do x
        x isa Number ? round(ustrip(x), sigdigits=sigdigits)*oneunit(x) : x
      end
    end
  end
  values = [i == 1 ? val : (rename!(val, names(val)[1]=>name)) for (i, (val, name)) in enumerate(zip(vals, _names))]

  return hcat(values...)
end

to_dataframe(report) = (@warn("`to_dataframe` is removed. `NCAReport` now returns a DataFrame directly."); report)

@recipe function f(subj::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}; linear=true, loglinear=true) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  subj = urine2plasma(subj)
  istwoplots = linear && loglinear
  istwoplots && (layout --> (1, 2))
  hastitle = length(get(plotattributes, :title, [])) >= 2
  hasunits = eltype(C) <: Quantity
  isurine  = R !== Nothing
  timename = isurine ? "Midpoint" : "Time"
  hasunits && (timename *= " ($(string(unit(eltype(T)))))")
  concname = isurine ? "Excretion Rate" : "Concentration"
  hasunits && (concname *= " ($(string(unit(eltype(C)))))")
  xguide --> timename
  yguide --> concname
  label --> subj.id
  vars = (ustrip.(subj.abstime), replace!(x->x<eps(one(x)) ? eps(one(x)) : x, vcat(map(x->ustrip.(x), subj.conc)...)))
  if linear
    @series begin
      seriestype --> :path
      istwoplots && (subplot --> 1)
      _title = hastitle ? plotattributes[:title][1] : "Linear scale"
      title := _title
      vars
    end
  end
  if loglinear
    @series begin
      yscale --> :log10
      seriestype --> :path
      istwoplots && (subplot --> 2)
      _title = hastitle ? plotattributes[:title][2] : "Semilogarithmic scale"
      title := _title
      vars
    end
  end
end

@recipe function f(pop::NCAPopulation{NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}}; linear=true, loglinear=true) where {C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}
  pop = urine2plasma(pop)
  istwoplots = linear && loglinear
  istwoplots && (layout --> (1, 2))
  hastitle = length(get(plotattributes, :title, [])) >= 2
  hasunits = eltype(C) <: Quantity
  isurine  = R !== Nothing
  timename = isurine ? "Midpoint" : "Time"
  hasunits && (timename *= " ($(string(unit(eltype(T)))))")
  concname = isurine ? "Excretion Rate" : "Concentration"
  hasunits && (concname *= " ($(string(unit(eltype(C)))))")
  xguide --> timename
  yguide --> concname
  label --> [subj.id for subj in pop]
  linestyle --> :auto
  legend --> false
  timearr = [ustrip.(subj.abstime) for subj in pop]
  concarr = [replace!(x->x < eps(one(x)) ? eps(one(x)) : x, vcat(map(x->ustrip.(x), subj.conc)...)) for subj in pop]
  if linear
    @series begin
      seriestype --> :path
      istwoplots && (subplot --> 1)
      _title = hastitle ? plotattributes[:title][1] : "Linear scale"
      title := _title
      (timearr, concarr)
    end
  end
  if loglinear
    @series begin
      concname = hasunits ? "Concentration ($(string(unit(eltype(C)))))" : "Concentration"
      yguide --> concname
      yscale --> :log10
      seriestype --> :path
      istwoplots && (subplot --> 2)
      _title = hastitle ? plotattributes[:title][2] : "Semilogarithmic scale"
      title := _title
      (timearr, concarr)
    end
  end
end
