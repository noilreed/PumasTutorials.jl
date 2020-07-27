module NCA

using Statistics
using Reexport
using DataFrames
using RecipesBase
using Pkg, Dates, Printf
using LinearAlgebra: norm

const Maybe{T} = Union{Missing, T}

include("type.jl")
include("data_parsing.jl")
include("utils.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export DataFrame

export NCASubject, NCAPopulation, NCADose, showunits
export read_nca
export NCAReport
export normalizedose

_repeat(xs, ndoses) = mapreduce(vcat, zip(xs, ndoses)) do (x, ndose)
  if x isa AbstractArray
    repeat(x, ndose)
  else
    fill(x, ndose)
  end
end

for f in [:lambdaz, :lambdazr2, :lambdazadjr2, :lambdazr, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst, :lambdaztimelast, :span,
          :cmax, :cmaxss, :tmax, :cmin, :cminss, :ctau, :c0, :tmin, :clast, :tlast, :thalf, :cl, :_cl, :_clf, :vss, :vz, :_vz, :_vzf,
          :interpextrapconc, :auc, :auclast, :auctau, :aumc, :aumclast, :aumctau, :auc_extrap_percent, :aumc_extrap_percent, :auc_back_extrap_percent,
          :bioav, :tlag, :mrt, :mat, :tau, :cavgss, :fluctuation, :accumulationindex,
          :swing, :n_samples, :doseamt, :dosetype, :subject_id, :superposition,
          :tmax_rate, :max_rate, :mid_time_last, :rate_last, :aurc, :aurc_extrap_percent, :urine_volume, :percent_recovered, :amount_recovered, :run_status,
         ]
  @eval $f(conc, time, args...; kwargs...) = $f(NCASubject(conc, time; kwargs...), args...; kwargs...) # f(conc, time) interface
  f === :superposition && continue
  isauclike = f in [:auc, :auclast, :aumc, :aucmclast]
  function_body = quote
    ismulti = ismultidose(pop)
    if ismulti
      sol′ = map(enumerate(pop)) do (i, subj)
        _sol = $f(subj, args...; verbose=verbose, interval=interval, kwargs...)
        param = $f == mat ? vcat(_sol, fill(missing, length(subj.dose)-1)) : # make `f` as long as the other ones
                            $f(subj, args...; verbose=verbose, interval=interval, kwargs...)
      end
      sol = mapreduce(x->x isa AbstractArray ? x : [x], vcat, sol′)
    else
      sol = map(subj->$f(subj, args...; verbose=verbose, interval=interval, kwargs...), pop)
    end
    typeof(sol) === Any && (sol = map(identity, sol))
    df = DataFrame()
    if label
      firstsubj = first(pop)
      ndoses = map(subj->subj.dose isa Union{Nothing,NCADose} ? 1 : length(subj.dose), pop)
      id′ = map(subj->subj.id, pop)
      df.id = _repeat(id′, ndoses)
      ismulti && (df.occasion = mapreduce(ndose->1:ndose, vcat, ndoses))
      if firstsubj.group !== nothing
        ngroup = firstsubj.group isa AbstractArray ? length(firstsubj.group) : 1
        if ngroup == 1
          grouplabel = Symbol(firstsubj.group.first)
          groupnames = map(subj->subj.group.second, pop)
          setproperty!(df, grouplabel, _repeat(groupnames, ndoses))
        else # multi-group
          for i in 1:ngroup
            grouplabel = Symbol(firstsubj.group[i].first)
            groupnames = map(subj->subj.group[i].second, pop)
            setproperty!(df, grouplabel, _repeat(groupnames, ndoses))
          end
        end
      end
    end
    if $isauclike && interval !== nothing
      setproperty!(df, Symbol($f, ustrip(interval[1]), :_, ustrip(interval[2])), sol)
    else
      df.$f = sol
    end
  end
  @eval function $f(pop::NCAPopulation, args...; label=true, verbose=true, interval=nothing, kwargs...) # NCAPopulation handling
    if interval isa AbstractVector
      intervals = interval
      label = label
      _df = DataFrame()
      for interval in intervals
        $function_body
        label = false
        _df = hcat(_df, df)
      end
      df = _df
    else
      $function_body
    end
    return df
  end
end

# add `tau`
# Multiple dosing handling
for f in [:c0, :clast, :tlast, :cmax, :cmaxss, :tmax, :cmin, :cminss, :tmin, :ctau, :_auc, :tlag, :mrt, :fluctuation,
          :cavgss, :tau, :auctau, :aumctau, :auc_extrap_percent, :aumc_extrap_percent, :auc_back_extrap_percent, :accumulationindex, :swing, :vss, :cl, :_cl, :_clf, :vz, :_vz, :_vzf,
          :lambdaz, :lambdazr2, :lambdazadjr2, :lambdazr, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst, :lambdaztimelast, :span,
          :n_samples, :doseamt, :dosetype, :subject_id]
  @eval function $f(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID,G,V,R,RT}, args...; kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D<:AbstractArray,Z,F,N,I,P,ID,G,V,R,RT}
    obj = map(eachindex(nca.dose)) do i
      subj = subject_at_ithdose(nca, i)
      $f(subj, args...; kwargs...)
    end
  end
end

# urine mapping
for (urine_f, f) in [:tmax_rate => :tmax, :max_rate => :cmax, :mid_time_last => :tlast, :rate_last => :clast, :aurc => :auc, :lambdaz => :lambdaz, :aurc_extrap_percent => :auc_extrap_percent]
  @eval function $urine_f(subj::NCASubject, args...; kwargs...)
    subj′ = urine2plasma(subj)
    ret = $f(subj′, args...; kwargs...)
    cache_ncasubj!(subj, subj′)
    return ret
  end
end

end
