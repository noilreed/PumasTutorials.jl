# Could use ensemble problem
struct Bootstraps
  populations
  fits
  stratify_by
end

struct FittedPumasModelInference{T1, T2, T3}
  fpm::T1
  vcov::T2
  level::T3
end

StatsBase.coef(pmi::FittedPumasModelInference) = coef(pmi.fpm)
function StatsBase.stderror(pmi::FittedPumasModelInference)
  ss = sqrt.(diag(pmi.vcov))
  trf = tostderrortransform(pmi.fpm.fixedparamset)
  return TransformVariables.transform(trf, ss)
end
function StatsBase.stderror(pmi::FittedPumasModelInference{<:Any, <:Bootstraps})
  successful_fits = filter(x->isa(x, FittedPumasModel), pmi.vcov.fits)
  T = typeof(successful_fits[1])
  successful_fits = T[fit for fit in successful_fits]
  ss = std(successful_fits)
  return ss
end

"""
    infer(fpm::FittedPumasModel) -> FittedPumasModelInference

Compute the `vcov` matrix and return a struct used for inference
based on the fitted model `fpm`.
"""
function infer(fpm::FittedPumasModel; level=0.95, rethrow_error=false)
  print("Calculating: variance-covariance matrix")
  _vcov = vcov(fpm; rethrow_error=rethrow_error)
  if _vcov isa AbstractMatrix
    println(". Done.")
  else
    println(". Failed.")
  end
  return FittedPumasModelInference(fpm, _vcov, level)
end

"""
  bootstrap(fpm::FittedPumasModel; samples=200, stratify_by=nothing)

Perform bootstrapping by resampling the `Subject`s from the `Population` stored in `fpm`. The keyword `samples` is used to control the number of resampled datasets, and by specifying keyword `stratify_by` to be a `Symbol` with the name of a covariate it is possible to stratify by a covariate with a finite number of possible values. The rest of the keyword arguments are passed onto the `fit` function internally.
"""
function bootstrap(fpm::FittedPumasModel; samples=200, stratify_by=nothing, level=0.95)

  # Grab the stored optimize_fn from fit...
  opt_fn = fpm.kwargs[:optimize_fn]
  # ... and force show_trace to be false as async operations mess it up big time
  optimize_fn =
    DefaultOptimizeFN(opt_fn.alg, (
      opt_fn.kwargs...,
      show_trace=false, # Print progress
      ))

  # pass defaults and the modified optimize_fn
  bts = bootstrap(fpm.model, fpm.data, coef(fpm), fpm.approx; samples=samples, stratify_by=stratify_by, fpm.kwargs..., optimize_fn=optimize_fn)
  FittedPumasModelInference(fpm, bts, level)
end

function try_fit(args...; kwargs...)
  try
    return fit(args...; kwargs...)
  catch e
    return e
  end
end

function bootstrap(model::PumasModel, data::Population, coef, approx::LikelihoodApproximation; samples=200, stratify_by=nothing, kwargs...)
  # FIXME check for time varying covariates
  # This data preprocessing is not written for speed. Most of it could
  # be written in one long loop, but we keep it like this for clarity.
  # Find all individual values of the variable we're stratisfying by
  n_subject = length(data)
  if stratify_by !== nothing
    # Just grab the time 0 covariate. It should be constant if we stratify
    strata_values = [getproperty(subject.covariates(0.0), stratify_by) for subject in data]
  else
    strata_values = fill(1, n_subject)
  end
  # Find the unique values ...
  strata_unique = unique(strata_values)
  n_strata = length(strata_unique)
  # Find the number of occurences
  count_unique = [count(isequal(s), strata_values) for s in strata_unique]
  # To assign each individual to a stratum by number
  stratum = map(i -> findfirst(isequal(i), strata_unique), strata_values)
  # Find the indices for the different types of subjects
  populations = [reduce(vcat, [sample(data[strata_values .== strata_unique[idx_s]], count_unique[idx_s]) for idx_s = 1:n_strata]) for n = 1:samples]

  # add distributed
  # fits = pmap(pop->fit(model, pop, coef, approx), populations)
  tasks = map(populations) do (pop)
    Threads.@spawn try_fit(model, pop, coef, approx; kwargs...)
  end
  fits = fetch.(tasks)
  Bootstraps(populations, fits, stratify_by)
end

function Base.show(io::IO, mime::MIME"text/plain", bts::Bootstraps)
  println(io, "Bootstrap inference results\n")
  println(io, "Successful fits: $(count(x -> x isa FittedPumasModel, bts.fits)) out of $(length(bts.fits))")
  if bts.stratify_by isa Nothing
	println(io, "No stratification.")
  else
    println(io, "Stratification by $(bts.stratify_by).")
  end
end
