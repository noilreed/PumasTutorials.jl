using CSV, DataFrames

"""
    read_nca(df::Union{DataFrame,AbstractString}; id=:id, time=:time,
             amt=:amt, route=:route, duration=:duration, blq=:blq,
             group=nothing, ii=nothing, concu=true, timeu=true, amtu=true, verbose=true, kwargs...)

Parse a `DataFrame` object or a CSV file to `NCAPopulation`. `NCAPopulation`
holds an array of `NCASubject`s which can cache certain results to achieve
efficient NCA calculation.

!!! remark
    Concentrations at dosing rows are NOT ignored in `read_nca`.
"""
read_nca(file::AbstractString; kwargs...) = read_nca(DataFrame(CSV.File(file)); kwargs...)
function read_nca(df; group=nothing, kwargs...)
  pop = if group === nothing
    ___read_nca(df; kwargs...)
  else
    dfs = groupby(df, group, sort=true)
    groupnum = length(dfs)
    dfpops = combine(dfs, ungroup=false) do df
      if group isa AbstractArray && length(group) > 1
        grouplabel = map(string, group)
        groupnames = map(string, first(df[!,group]))
        currentgroup = map(=>, grouplabel, groupnames)
      else
        group isa Symbol || ( group = first(group) )
        grouplabel = string(group)
        groupnames = first(df[!,group])
        currentgroup = grouplabel => groupnames
      end
      pop = ___read_nca(df; group=currentgroup, kwargs...)
      return pop
    end
    pops = map(i->dfpops[i][!, end], 1:groupnum)
    vcat(pops...)
  end
  return pop
end

function ___read_nca(df; id=:id, time=:time, conc=:conc, occasion=:occasion,
                     start_time=:start_time, end_time=:end_time, volume=:volume,
                     amt=:amt, route=:route, duration=:duration, blq=:blq,
                     ii=:ii, ss=:ss, group=nothing, concu=true, timeu=true, amtu=true, volumeu=true,
                     verbose=true, kwargs...)
  local ids, times, concs, amts
  has_id = hasproperty(df, id)
  has_time = hasproperty(df, time)
  has_conc = hasproperty(df, conc)
  if has_id && has_time && has_conc
    urine = false
  else
    has_start_time = hasproperty(df, start_time)
    has_end_time   = hasproperty(df, end_time)
    has_volume     = hasproperty(df, volume)
    if has_start_time && has_end_time && has_volume && has_conc
      urine = true
    else
      @info "The CSV file has keys: $(names(df))"
      throw(ArgumentError("The CSV file must have: `id, time, conc` or `id, start_time, end_time, volume, conc` columns"))
    end
  end
  blq = hasproperty(df, blq) ? blq : nothing
  amt = hasproperty(df, amt) ? amt : nothing
  ii  = hasproperty(df, ii) ? ii : nothing
  ss  = hasproperty(df, ss) ? ss : nothing
  route = hasproperty(df, route) ? route : nothing
  occasion = hasproperty(df, occasion) ? occasion : nothing
  duration = hasproperty(df, duration) ? duration : nothing
  hasdose = amt !== nothing && route !== nothing
  if verbose
    hasdose || @warn "No dosage information has passed. If the dataset has dosage information, you can pass the column names by `amt=:amt, route=:route`."
    if amt !== nothing && route === nothing
      @warn "Dosage information requires the presence of both amt & route information. Looks like you only entered the amt and not the route. If your dataset does not have route, please add a column that specifies the route of administration and then pass both columns as `amt=:amt, route=:route.`"
    end
  end

  # BLQ
  @noinline blqerr() = throw(ArgumentError("blq can only be 0 or 1"))
  if blq !== nothing
    blqs = df[!,blq]
    eltype(blqs) <: Union{Int, Bool} || blqerr()
    if eltype(blqs) <: Int # save computation when `blq` are all Bools
      exa = extrema(blqs)
      all(x->x in (0, 1), exa) || blqerr()
    end
    df = delete!(deepcopy(df), findall(isequal(1), blqs))
  end

  sortvars = urine ? (occasion === nothing ? [id, start_time, end_time] : [id, start_time, end_time, occasion]) :
                     (occasion === nothing ? [id, time] : [id, time, occasion])
  iss = issorted(df, sortvars)
  # we need to use a stable sort because we want to preserve the order of `time`
  sortedf = iss ? df : sort(df, sortvars, alg=Base.Sort.DEFAULT_STABLE)
  ids   = df[!,id]
  if urine
    start_time′ = df[!,start_time]
    end_time′ = df[!,end_time]
    Δt = @. end_time′ - start_time′
    times = @. start_time′ + Δt
  else
    start_time′ = end_time′ = Δt = nothing
    times = df[!,time]
  end
  concs = df[!,conc]
  amts  = amt === nothing ? nothing : df[!,amt]
  iis  = ii === nothing ? nothing : df[!,ii]
  sss  = ss === nothing ? nothing : df[!,ss]
  occasions = occasion === nothing ? nothing : df[!,occasion]
  uids = unique(ids)
  idx  = -1
  checkdata(times, :time)
  checkdata(concs, :conc)
  amts === nothing || checkdata(amts, :amt)
  # FIXME! This is better written as map(uids) do id it currently triggers a dispatch bug in Julia via CSV
  ncas = Vector{Any}(undef, length(uids))
  lo = 1
  for (i, id) in enumerate(uids)
    # id's range, and we know that it is sorted
    # FIXME: ugly optimization
    lo = loid = findfirst(isequal(id), @view ids[lo:end]) + lo - 1
    _hiid = findfirst(x->x != id, @view ids[lo:end])
    lo = hiid = _hiid === nothing ? length(ids) : _hiid + lo - 2
    idx = loid:hiid
    if hasdose
      dose_idx = findall(x->x !== missing && x > zero(x), @view amts[idx])
      if isempty(dose_idx)
        msg = "ID: $id. Dose information is provided, but there is no valid dosage. All `amt` are non-positive or `missing`."
        if group !== nothing
          msg *= " Error at $group."
        end
        throw(ArgumentError(msg))
      end
      length(dose_idx) > 1 && occasion === nothing && error("`occasion` must be provided for multiple dosing data")
      dose_idx = idx[dose_idx] # translate to the global index
      # We want to use time instead of an integer index here, because later we
      # need to remove BLQ and missing data, so that an index number will no
      # longer be valid.
      if length(dose_idx) == 1
        dose_idx = dose_idx[1]
        dose_time = times[dose_idx]
      else
        dose_time = similar(times, Base.nonmissingtype(eltype(times)), length(dose_idx))
        for (n,i) in enumerate(dose_idx)
          dose_time[n] = times[i]
        end
      end
      route′ = map(dose_idx) do i
        routei = df[!,route][i]
        routei isa AbstractFloat && isinf(routei) && (routei = "inf")
        routei isa AbstractString || routethrow()
        routei = lowercase(routei)
        routei == "iv" ? IVBolus :
          routei == "inf" ? IVInfusion :
          routei == "ev" ? EV :
          routethrow()
      end
      ii = map(i -> iis === nothing ? false : iis[i], dose_idx)
      ss = map(dose_idx) do i
        sss === nothing ? false :
          sss[i] == 0 ? false :
          sss[i] == 1 ? true :
          throw(ArgumentError("ss can only be 0 or 1"))
      end
      duration′ = if duration === nothing
        nothing
      else
        durations = df[!,duration]
        map(idx->durations[idx]*timeu, dose_idx)
      end
      amt′ = map(idx->amts[idx]*amtu, dose_idx)
      doses = NCADose.(dose_time*timeu, amt′, duration′, route′, ii*timeu, ss)
    else
      doses = nothing
    end
    try
      ncas[i] = NCASubject(concs[idx], times[idx]; id=id, group=group, dose=doses, concu=concu, timeu=timeu, volumeu=volumeu,
                           start_time=start_time′, end_time=end_time′,
                           volume=urine ? df[!,volume][idx] : nothing,
                           concblq=blq===nothing ? nothing : :keep, kwargs...)
    catch
      @info "ID $id errored"
      group !== nothing && @info "$group errored"
      rethrow()
    end
  end
  # Use broadcast to tighten ncas element type
  pop = identity.(ncas)
  return pop
end

@noinline routethrow() = throw(ArgumentError("route can only be `iv`, `ev`, or `inf`"))

@noinline function checkdata(data, name, typ=Maybe{Number})
  if !(eltype(data) <: typ)
    idx = findall(x->!(x isa typ), data)
    throw(ArgumentError("$name has non-numeric values at index=$idx. We expect the $names column to be of numeric type. Please fix your input data before proceeding further."))
  end
  return nothing
end
