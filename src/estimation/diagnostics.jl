####################
# Predict function #
####################

struct SubjectPrediction{T1, T2, T3}
  pred::T1
  ipred::T2
  subject::T3
end

function StatsBase.predict(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  args...;
  kwargs...)

  vrandeffsorth = _orth_empirical_bayes(model, subject, param, Pumas.LaplaceI())

  _predict(model, subject, param, vrandeffsorth, args...; kwargs...)
end

function StatsBase.predict(
  model::PumasModel,
  population::Population,
  param::NamedTuple,
  args...;
  kwargs...)

  map(subject -> predict(model, subject, param, approx, args...; kwargs...), population)
end

function StatsBase.predict(
  fpm::FittedPumasModel,
  population::Population=fpm.data,
  subjects::Union{Nothing, Population}=nothing;
  nsim=nothing, obstimes=nothing)

  if !(nsim isa Nothing)
    error("using simulated subjects is not yet implemented.")
  end

  if population === fpm.data
    vvrandeffsorth = fpm.vvrandeffsorth
  else
    vvrandeffsorth = [_orth_empirical_bayes(fpm.model, subject, coef(fpm), fpm.approx) for subject in population]
  end

  return map(i -> _predict(
    fpm.model,
    population[i],
    coef(fpm),
    vvrandeffsorth[i],
    fpm.args...;
    obstimes=obstimes,
    fpm.kwargs...), 1:length(population))
end

function StatsBase.predict(
  fpm::FittedPumasModel,
  subject::Subject;
  obstimes=subject.time)

  vrandeffsorth = _orth_empirical_bayes(fpm.model, subject, coef(fpm), fpm.approx)

  _predict(fpm.model, subject, coef(fpm), vrandeffsorth, fpm.args...; obstimes=obstimes, fpm.kwargs...)
end

function DataFrames.DataFrame(
  vpred::Vector{<:SubjectPrediction};
  include_covariates=true, include_dvs=true)

  subjects = [pred.subject for pred in vpred]
  df = DataFrame(subjects;
    include_covariates=false,
    include_dvs=include_dvs,
    include_events=false)

  _keys = keys(first(subjects).observations)
  for name in  _keys
    df[!, Symbol(string(name)*"_pred")] = vcat((pred.pred[name] for pred in vpred)...)
    df[!, Symbol(string(name)*"_ipred")] = vcat((pred.ipred[name] for pred in vpred)...)
  end

  if include_covariates
    df = mapreduce(
      _vpred->_add_covariates(df[df[!, :id].==_vpred.subject.id, :], _vpred.subject),
      vcat,
      vpred)
  end
  return df
end

function _predict(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  args...; kwargs...
  )

  ipred = __ipredict(model, subject, param, vrandeffsorth, args...; kwargs...)
  if isempty(vrandeffsorth) # NaivePooled
    pred = ipred
  else
    pred = __predict(model, subject, param, zero(vrandeffsorth), FO(), args...; kwargs...)
  end
  SubjectPrediction(pred, ipred, subject)
end

# Individual predictions
function __ipredict(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector;
  kwargs...)

  randeffs = TransformVariables.transform(totransform(m.random(param)), vrandeffsorth)

  dist = _derived(m, subject, param, randeffs; kwargs...)
  return map(d->mean.(d), dist)
end

# Population predictions
## PRED like _predict
function __predict(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::FO,
  args...; kwargs...)

  randeffs = TransformVariables.transform(totransform(m.random(param)), zero(vrandeffsorth))
  dist = _derived(m, subject, param, randeffs, args...; kwargs...)
  return map(d -> mean.(d), NamedTuple{keys(subject.observations)}(dist))
end

## CPRED(I) like
function __predict(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::Union{FOCE, FOCEI, LaplaceI},
  args...; kwargs...)

  randeffstransform = totransform(m.random(param))
  randeffs = TransformVariables.transform(randeffstransform, vrandeffsorth)
  dist = _derived(m, subject, param, randeffs, args...; kwargs...)

  _dv_keys = keys(subject.observations)
  # For FOCE, we don't allow the dispersion parameter to depend on the random effects
  if approx isa FOCE
    foreach(_dv_keys) do _key
      if !_is_homoscedastic(dist[_key])
        throw(ArgumentError("dispersion parameter is not allowed to depend on the random effects when using FOCE"))
      end
      nothing
    end
  end

  return map(NamedTuple{_dv_keys}(_dv_keys)) do name
    F = ForwardDiff.jacobian(
      _vrandeffs -> begin
        _randeffs = TransformVariables.transform(randeffstransform, _vrandeffs)
        return mean.(_derived(m, subject, param, _randeffs, args...; kwargs...)[name])
      end,
      vrandeffsorth
    )
    return mean.(dist[name]) .- F*vrandeffsorth
  end
end

# epredict
function epredict(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  args...;
  nsim::Union{Nothing,Integer}=nothing,
  kwargs...)

  if nsim === nothing
    throw(ArgumentError("the number of simulations argument (nsim) was not specified."))
  end
  if nsim < 1
    throw(ArgumentError("the number of simulations argument (nsim) must be positive"))
  end

  sims = simobs(m, RepeatedVector([subject], nsim), param; kwargs...)

  _dv_keys = keys(subject.observations)
  return map(name -> mean(getproperty.(getproperty.(sims, :observed), name)), NamedTuple{_dv_keys}(_dv_keys))
end

# FIXME! Make it parallel over subjects
"""
    epredict(fpm::FittedPumasModel, nsim::Integer)

Calculate the expected simulation based population predictions.
"""
epredict(fpm::FittedPumasModel; nsim=nothing) = [epredict(fpm.model, subject, coef(fpm); nsim=nsim, fpm.kwargs...) for subject in fpm.data]

#######################
# probstable function #
#######################

"""
   probstable(fpm::FittedPumasModel)

Return a DataFrame with outcome probabilities of all discrete dependent
variables.
"""
function probstable(fpm::FittedPumasModel)
  _probstable(fpm.model, fpm.data, coef(fpm), fpm.vvrandeffsorth, fpm.args...; fpm.kwargs...)
end

function probstable(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  randeffs::NamedTuple,
  args...; kwargs...)

  probs_derived = _derived(model, subject, param, randeffs, args...; kwargs...)
  df = DataFrame(id = fill(subject.id, size(subject.time, 1)), time = subject.time)
  for (name, val) in pairs(probs_derived)
    if all(x -> x isa Categorical || x isa Bernoulli, val)
      # the output can be a singleton but then map won't work
      _probs = map(__probs, val)
      _probsmat = reduce(hcat, _probs)'
      subject_probs_df = DataFrame(_probsmat,
        [Symbol("$(name)_prob$i") for i in 1:size(_probsmat, 2)])
      df = hcat(df, subject_probs_df)
    end
  end
  return df
end

function _probstable(
  model::PumasModel,
  population::Population,
  param::NamedTuple,
  vvrandeffsorth::AbstractVector,
  args...; kwargs...)

  rtrf = totransform(model.random(param))
  randeffs = TransformVariables.transform.(Ref(rtrf), vvrandeffsorth)

  return reduce(vcat, probstable.(Ref(model), population, Ref(param), randeffs))
end

# probs of Bernoulli is annoying, so create our own __probs
__probs(x) = probs(x)
__probs(x::Bernoulli) = pdf.(x, support(x))


######################
# Residual functions #
######################

# SubjectResidual
struct SubjectResidual{T1, T2, T3, T4}
  wres::T1
  iwres::T2
  subject::T3
  approx::T4
end

function DataFrames.DataFrame(
  vresid::Vector{<:SubjectResidual};
  include_covariates=true)

  subjects = [resid.subject for resid in vresid]
  df = select!(DataFrame(subjects; include_covariates=false, include_dvs=false), Not(:evid))

  _keys = keys(first(subjects).observations)
  for name in (_keys)
    df[!, Symbol(string(name)*"_wres")] .= vcat((resid.wres[name] for resid in vresid)...)
    df[!, Symbol(string(name)*"_iwres")] .= vcat((resid.iwres[name] for resid in vresid)...)
    df[!, :wres_approx] .= Ref(vresid[1].approx)
  end
  if include_covariates
    df = mapreduce(
      _vresid->_add_covariates(df[df[!, :id].==_vresid.subject.id, :], _vresid.subject),
      vcat,
      vresid)
  end
  df
end

"""
    wresiduals(fpm::FittedPumasModel, approx::LikelihoodApproximation=fpm.approx; nsim=nothing)

Calculate the individual and population weighted residual.

Takes a `fit` result, an approximation method for the marginal likelihood
calculation which defaults to the method used in the `fit` and the number of
simulations with the keyword argument `nsim`. If `nsim` is specified only the
Expected Simulation based Individual Weighted Residuals (EIWRES) is included
in the output as individual residual and population residual is not computed.
Using the `FO` approximation method corresponds to the WRES and while
`FOCE(I)` corresponds to CWRES. The output is a `SubjectResidual` object that
stores the population (`wres`) and individual (`iwres`) residuals along with
the `subject` and approximation method (`approx`).
"""
function wresiduals(
  fpm::FittedPumasModel,
  approx::LikelihoodApproximation=fpm.approx;
  nsim=nothing)

  population = fpm.data
  if approx == fpm.approx
    vvrandeffsorth = fpm.vvrandeffsorth
  else
    # re-estimate under approx
    vvrandeffsorth = [_orth_empirical_bayes(fpm.model, subject, coef(fpm), approx, fpm.args...; fpm.kwargs...) for subject in population]
  end
  [_wresiduals(fpm, population[i], vvrandeffsorth[i], approx; nsim=nsim) for i = 1:length(population)]
end

# _wresiduals
function _wresiduals(
  fpm::FittedPumasModel,
  subject::Subject,
  vrandeffsorth::AbstractVector,
  approx::LikelihoodApproximation;
  nsim=nothing)

  if nsim === nothing
    wres = __wresiduals(fpm.model, subject, coef(fpm), vrandeffsorth, approx, fpm.args...; fpm.kwargs...)
    iwres = __iwresiduals(fpm.model, subject, coef(fpm), vrandeffsorth, approx, fpm.args...; fpm.kwargs...)
  else
    approx = nothing
    wres = nothing
    iwres = eiwres(fpm.model, subject, coef(fpm), nsim, fpm.args...; fpm.kwargs...)
  end

  SubjectResidual(wres, iwres, subject, approx)
end

__wresiduals(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  ::NaivePooled,
  args...; kwargs...) = __iwresiduals(
    model,
    subject,
    param,
    vrandeffsorth,
    NaivePooled(),
    args...; kwargs...)


function __wresiduals(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::FO,
  args...; kwargs...)

  randeffstransform = totransform(model.random(param))
  randeffs = TransformVariables.transform(randeffstransform, zero(vrandeffsorth))
  dist = _derived(model, subject, param, randeffs, args...; kwargs...)

  F   = _mean_derived_vηorth_jacobian(model, subject, param, zero(vrandeffsorth), args...; kwargs...)
  res = _residuals(subject, dist)

  _dv_keys = keys(subject.observations)

  return map(NamedTuple{_dv_keys}(_dv_keys)) do name
    # We have to handle missing values explicitly to avoid that the variance
    # components associated with missing values influence the weighting
    missingmask = ismissing.(subject.observations[name])
    Fname   = F[name]
    resname = res[name]
    Fname[missingmask, :] .= 0
    resname[missingmask]  .= 0

    V = Symmetric(Fname*Fname' + Diagonal(var.(dist[name])))

    # Theoretically, it should be just fine to use this version
    # based on the Choleksy
    # ldiv!(cholesky(V).U', resname)
    # but we use a version based on the symmetric sqaure root
    # to match other software packages.
    resname .= sqrt(V)\resname

    # This setindex! operation fails when there are no missings even though
    # the mask is empty
    if any(missingmask)
      resname[missingmask] .= missing
    end

    return resname
  end
end

function __wresiduals(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::Union{FOCE,FOCEI,LaplaceI},
  args...; kwargs...)

  randeffstransform = totransform(model.random(param))
  randeffs = TransformVariables.transform(randeffstransform, vrandeffsorth)
  dist = _derived(model, subject, param, randeffs, args...; kwargs...)

  F   = _mean_derived_vηorth_jacobian(model, subject, param, vrandeffsorth, args...; kwargs...)
  res = _residuals(subject, dist)

  _dv_keys = keys(subject.observations)

  # For FOCE, we don't allow the dispersion parameter to depend on the random effects
  if approx isa FOCE
    foreach(_dv_keys) do _key
      if !_is_homoscedastic(dist[_key])
        throw(ArgumentError("dispersion parameter is not allowed to depend on the random effects when using FOCE"))
      end
      nothing
    end
  end

  return map(NamedTuple{_dv_keys}(_dv_keys)) do name
    # We have to handle missing values explicitly to avoid that the variance
    # components associated with missing values influence the weighting
    missingmask = ismissing.(subject.observations[name])
    Fname   = F[name]
    resname = res[name]
    Fname[missingmask, :] .= 0
    resname[missingmask]  .= 0

    V = Symmetric(Fname*Fname' + Diagonal(var.(dist[name])))

    # if "conditional" mothods, there is a first order term in the mean
    resname .+= Fname*vrandeffsorth

    # Theoretically, it should be just fine to use this version
    # based on the Choleksy
    # ldiv!(cholesky(V).U', resname)
    # but we use a version based on the symmetric sqaure root
    # to match other software packages.
    resname .= sqrt(V)\resname

    # This setindex! operation fails when there are no missings even though
    # the mask is empty
    if any(missingmask)
      resname[missingmask] .= missing
    end

    return resname
  end
end

# __iwresiduals
function __iwresiduals(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  approx::Union{FO,FOCE,FOCEI,LaplaceI,NaivePooled},
  args...; kwargs...)

  if approx isa FO
    _vrandeffsorth = zero(vrandeffsorth)
  else
    _vrandeffsorth = vrandeffsorth
  end
  randeffs = TransformVariables.transform(totransform(m.random(param)), _vrandeffsorth)
  dist = _derived(m, subject, param, randeffs, args...; kwargs...)

  _dv_keys = keys(subject.observations)
  # For FOCE, we don't allow the dispersion parameter to depend on the random effects
  if approx isa FOCE
    foreach(_dv_keys) do _key
      if !_is_homoscedastic(dist[_key])
        throw(ArgumentError("dispersion parameter is not allowed to depend on the random effects when using FOCE"))
      end
      nothing
    end
  end

  _res = _residuals(subject, dist)
  return map(name -> _res[name] ./ std.(dist[name]), NamedTuple{_dv_keys}(_dv_keys))
end

"""
    eiwres(model::PumasModel, subject::Subject, param::NamedTuple, nsim::Integer)

Calculate the Expected Simulation based Individual Weighted Residuals (EIWRES).
"""
function eiwres(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                nsim::Integer,
                args...;
                kwargs...)

  dist = _derived(m, RepeatedVector([subject], nsim), param, args...;
    obstimes=subject.time, kwargs...)

  _keys_dv = keys(subject.observations)
  return map(NamedTuple{_keys_dv}(_keys_dv)) do name
    dv = dist[1][name]
    obsdv = subject.observations[name]
    sims_sum = (obsdv .- mean.(dv))./std.(dv)
    for i in 2:nsim
      dv = dist[i][name]
      sims_sum .+= (obsdv .- mean.(dv))./std.(dv)
    end
    return sims_sum ./ nsim
  end
end

# Internal _residuals helper function
function _residuals(model::PumasModel, subject::Subject, param::NamedTuple, vrandeffs::AbstractArray, args...; kwargs...)
  rtrf = totransform(model.random(param))
  randeffs = TransformVariables.transform(rtrf, vrandeffs)
  # Calculated the dependent variable distribution objects
  dist = _derived(model, subject, param, randeffs, args...; kwargs...)
  # Return the residuals
  return _residuals(subject, dist)
end

function _residuals(subject::Subject, dist)
  # Return the residuals
  _keys = keys(subject.observations)
  return map(x->x[1] .- mean.(x[2]), NamedTuple{_keys}(zip(subject.observations, dist)))
end

########
# npde #
########

function npde(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple;
  nsim::Union{Nothing,Integer}=nothing, kwargs...)

  if nsim === nothing
    throw(ArgumentError("the number of simulations argument (nsim) was not specified."))
  end
  if nsim < 1
    throw(ArgumentError("the number of simulations argument (nsim) must be positive"))
  end

  _names = keys(subject.observations)
  sims = getproperty.(simobs(m, RepeatedVector([subject],nsim), param; kwargs...),:observed)


  return map(NamedTuple{_names}(_names)) do name
    y        = subject.observations[name]
    ysims    = getproperty.(sims, name)
    mean_y   = mean(ysims)
    cov_y    = Symmetric(cov(ysims))
    Fcov_y   = cholesky(cov_y)
    y_decorr = Fcov_y.U'\(y .- mean_y)

    φ = mean(ysims) do y_l
      y_decorr_l = Fcov_y.U'\(y_l .- mean_y)
      Int.(y_decorr_l .< y_decorr)
    end

    return quantile.(Normal(), φ)
  end
end

# FIXME! make this run in parallel
"""
    npde(fpm::FittedPumasModel; nsim::Integer)

Calculate the normalised prediction distribution errors (NPDE).
"""
npde(fpm::FittedPumasModel; nsim=nothing) = [npde(fpm.model, subject, coef(fpm); nsim=nsim, fpm.kwargs...) for subject in fpm.data]

#############
# Shrinkage #
#############

# ηshrinkage
"""
    ηshrinkage(fpm::FittedPumasModel)

Calculate the η-shrinkage.

Takes the result of a `fit` as the only input argument. A named tuple of the
random effects and corresponding η-shrinkage values is output.
"""
ηshrinkage(fpm::FittedPumasModel) = _ηshrinkage(fpm.model, fpm.data, coef(fpm), fpm.vvrandeffsorth; fpm.kwargs...)

function _ηshrinkage(m::PumasModel,
                    data::Population,
                    param::NamedTuple,
                    vvrandeffsorth::AbstractVector,
                    args...;
                    kwargs...)

  vtrandeffs = [TransformVariables.transform(totransform(m.random(param)),
    _vrandefforth) for _vrandefforth in vvrandeffsorth]

  randeffsstd = map(keys(first(vtrandeffs))) do k
    return 1 .- std(getfield.(vtrandeffs, k)) ./ sqrt.(var(m.random(param).params[k]))
  end

  return NamedTuple{keys(first(vtrandeffs))}(randeffsstd)
end

# ϵshrinkage
"""
    ϵshrinkage(fpm::FittedPumasModel)

Calculate the ϵ-shrinkage.

Takes the result of a `fit` as the only input argument. A named tuple of
derived variables and corresponding ϵ-shrinkage values is output.
"""
ϵshrinkage(fpm::FittedPumasModel) =
  _ϵshrinkage(fpm.model, fpm.data, coef(fpm), fpm.vvrandeffsorth; fpm.kwargs...)

function _ϵshrinkage(m::PumasModel,
                     data::Population,
                     param::NamedTuple,
                     vvrandeffsorth::AbstractVector,
                     args...; kwargs...)

  _keys_dv = keys(first(data).observations)
  _icwres = [__iwresiduals(m, subject, param,
    vvrandeffsorth, LaplaceI()) for (subject, vvrandeffsorth) in zip(data, vvrandeffsorth)]
  map(name -> 1 - std(skipmissing(Iterators.flatten(getproperty.(_icwres, name))), corrected = false), NamedTuple{_keys_dv}(_keys_dv))
end

########################
# Information criteria #
########################

"""
    aic(fpm::FittedPumasModel)

Calculate the Akaike information criterion (AIC) of the fitted Pumas model
`fpm`.
"""
StatsBase.aic(fpm::FittedPumasModel) =
  aic(fpm.model, fpm.data, coef(fpm), fpm.approx; fpm.kwargs...)

function StatsBase.aic(m::PumasModel,
                       data::Population,
                       param::NamedTuple,
                       approx::LikelihoodApproximation,
                       args...;
                       kwargs...)
  numparam = TransformVariables.dimension(totransform(m.param))
  2*(marginal_nll(m, data, param, approx, args...; kwargs...) + numparam)
end

"""
    bic(fpm::FittedPumasModel)

Calculate the Bayesian information criterion (BIC) of the fitted Pumas model
`fpm`.
"""
StatsBase.bic(fpm::FittedPumasModel) =
  bic(fpm.model, fpm.data, coef(fpm), fpm.approx, ; fpm.kwargs...)

function StatsBase.bic(m::PumasModel,
                       population::Population,
                       param::NamedTuple,
                       approx::LikelihoodApproximation,
                       args...;
                       kwargs...)

  numparam = TransformVariables.dimension(totransform(m.param))
  nll = marginal_nll(m, population, param, approx, args...; kwargs...)
  n = sum(subject -> sum(dv -> sum(!ismissing, dv), subject.observations), population)
  return 2*nll + numparam*log(n)
end



# empirical_bayes
function empirical_bayes(fpm::FittedPumasModel)
  subjects = fpm.data

  trf = totransform(fpm.model.random(coef(fpm)))

  if fpm.approx ∈ (Pumas.FOCE(), Pumas.FOCEI(), Pumas.LaplaceI())
    ebes = fpm.vvrandeffsorth
    return [TransformVariables.transform(trf, e) for (e, s) in zip(ebes, subjects)]
  elseif fpm.approx isa FO
    # estimate under LaplaceI
    return [
      TransformVariables.transform(
        trf,
        _orth_empirical_bayes(
          fpm.model, subject,
          coef(fpm),
          LaplaceI(),
          fpm.args...; fpm.kwargs...
        )
      ) for subject in subjects]
  elseif fpm.approx isa NaivePooled
    return fill(NamedTuple(), length(fpm.data))
  else
    throw(ArgumentError("empirical_bayes not implemented for $(fpm.approx)"))
  end
end


# Inspection
struct FittedPumasModelInspection{T1, T2, T3, T4}
  o::T1
  pred::T2
  wres::T3
  ebes::T4
end

StatsBase.predict(insp::FittedPumasModelInspection) = insp.pred
# We allow args... here since the called method will only use the saved args...
# from the fitting stage

StatsBase.predict(insp::FittedPumasModelInspection, args...) = predict(insp.o, args...)
wresiduals(insp::FittedPumasModelInspection) = insp.wres
empirical_bayes(insp::FittedPumasModelInspection) = insp.ebes

"""
    inspect(fpm::FittedPumasModel; pred_approx=fpm.approx, wres_approx=fpm.approx)

Output a summary of the model predictions, residuals and Empirical Bayes
estimates.

Called on a `fit` output and allows the keyword argument `wres_approx` for
approximation method to be used in  residual calculation respectively. A
`FittedPumasModelInspection` object with `pred`, `wres` and `ebes` is output.
"""
function inspect(fpm; wres_approx=fpm.approx)
  print("Calculating: ")
  print("predictions")
  pred = predict(fpm)
  print(", weighted residuals")
  res = wresiduals(fpm, wres_approx)
  print(", empirical bayes")
  ebes = empirical_bayes(fpm)
  println(". Done.")
  FittedPumasModelInspection(fpm, pred, res, (ebes=ebes,))
end

function DataFrames.DataFrame(i::FittedPumasModelInspection; include_covariates=true)
  # Creat the dataframe including
  pred_df = DataFrame(i.pred; include_covariates=false)
  res_df = select!(select!(DataFrame(i.wres; include_covariates=false), Not(:id)), Not(:time))
  ebes = i.ebes.ebes
  ebe_keys = keys(first(ebes))
  ebe_types = map(typeof, first(ebes))
  ebes_df = select!(DataFrame(i.o.data; include_covariates=false, include_dvs=false), Not(:evid))
  for k ∈ ebe_keys
    ebe_type = ebe_types[k]
    if ebe_type <: Number
      ebes_df[!, k] .= vcat((fill(ebes[n][k], length(i.o.data[n].time)) for n = 1:length(i.o.data))...)
    elseif ebe_type <: AbstractVector
      for j = 1:length(first(ebes)[k])
        ebes_df[!, Symbol(string(k)*"_$j")] .= vcat((fill(ebes[n][k][j], length(i.o.data[n].time)) for n = 1:length(i.o.data))...)
      end
    end
  end
  ebes_df = select!(select!(ebes_df, Not(:id)), Not(:time))
  df = hcat(pred_df, res_df, ebes_df)
  if include_covariates
    df = mapreduce(subject->_add_covariates(df[df[!, :id].==subject.id, :], subject), vcat, i.o.data)
  end
  df
end

###################
# findinfluential #
###################

"""
    findinfluential(fpm::FittedPumasModel, k::Integer=5)

Return a vector of the `k` most influencial observations based on the value of
the objective function.
"""
findinfluential(fpm::FittedPumasModel, k::Integer=5) =
  findinfluential(fpm.model, fpm.data, coef(fpm), fpm.approx, fpm.args...;
    k=k, fpm.kwargs...)

function findinfluential(
  m::PumasModel,
  data::Population,
  param::NamedTuple,
  approx::LikelihoodApproximation,
  args...;
  k=5,
  kwargs...)

  d = [deviance(m, subject, param, approx, args...; kwargs...) for subject in data]
  p = partialsortperm(d, 1:k, rev=true)
  return [(data[pᵢ].id, d[pᵢ]) for pᵢ in p]
end

######################################
# Condition number of a fitted model #
######################################

"""
    cond(pmi::FittedPumasModelInference)

Return the condition number of the variance-covariance matrix stored in `pmi`.
Throw an error if `pmi` is the result of a call to `bootstrap` or if the
variance-covariance calculation failed.
"""
LinearAlgebra.cond(pmi::FittedPumasModelInference{<:Any, <:Exception}) =
  throw(ArgumentError("It is not possible to apply cond when the vcov calculations failed."))
LinearAlgebra.cond(pmi::FittedPumasModelInference{<:Any, <:Bootstraps}) =
  throw(ArgumentError("It is not possible to apply cond when inference comes from a bootstrap call."))
LinearAlgebra.cond(pmi::FittedPumasModelInference{<:Any, <:AbstractMatrix}) =
  cond(pmi.vcov)


################################################################################
#                              Plotting functions                              #
################################################################################

########################################
#   Convergence plot infrastructure    #
########################################

"""
    _objectivefunctionvalues(obj)

Returns the objective function values during optimization. Must return a
`Vector{Number}`.
"""
_objectivefunctionvalues(f::FittedPumasModel) = getproperty.(f.optim.trace, :value)

"""
    _convergencedata(obj; metakey="x")

Returns the "timeseries" of optimization as a matrix, with series as columns.
!!! warn
    This must return parameter data in the same order that [`_paramnames`](@ref)
    returns names.
"""
function _convergencedata(f::FittedPumasModel; metakey="x")

  metakey != "x" && return transpose(hcat(getindex.(getproperty.(f.optim.trace, :metadata), metakey)...))

  # get the transform which has been applied to the params
  trf  = totransform(f.model.param)
  # invert the param transform
  itrf = toidentitytransform(f.model.param)

  # return series as columns
  return transpose(
    # apply the inverse of the given transform to the data.
    hcat(TransformVariables.inverse.(
      # wrap in a `Ref`, to avoid broadcasting issues
      Ref(itrf),
      # apply the initial transform to the process
      TransformVariables.transform.(
        # again - make sure no broadcasting across the `TransformTuple`
        Ref(trf),
          # get every `x` vector from the metadata of the trace
          getindex.(
            # get the metadata of each trace element
            getproperty.(
              # getproperty expects a `Symbol`
              f.optim.trace, :metadata
            ),
            # property x is a key for a `Dict` - hence getindex
            metakey
          )
        )
      # splat to get a matrix out
      )...
    )
  )
end

"""
    _paramnames(obj)

Returns the names of the parameters which convergence is being checked for.
!!! warn
    This must return parameter names in the same order that [`_convergencedata`](@ref)
    returns data.
"""
function _paramnames(f::FittedPumasModel)
  paramnames = [] # empty array, will fill later
  for (paramname, paramval) in pairs(coef(f)) # iterate through the parameters
    # decompose all parameters (matrices, etc.) into scalars and name them appropriately
    _push_varinfo!(paramnames, [], nothing, nothing, paramname, paramval, nothing, nothing)
  end
  return paramnames
end
