const _subscriptvector = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

_coef_value(var) = var
_coef_value(var::PDMat) = var.mat
_coef_value(var::PDiagMat) = Diagonal(var.diag)

_to_subscript(number) = join([_subscriptvector[parse(Int32, dig)+1] for dig in string(number)])


function show_dashrow_table(io, labels, stringrows)
  dashrow = "-"^max(length(labels)+1,maximum(length.(stringrows)))
  println(io, dashrow)
  print(io, labels)
  println(io,"\n" , dashrow)
  for stringrow in stringrows
    print(io, stringrow)
  end
  println(io, dashrow)
end

function _print_fit_header(io, fpm)
  println(io, string("Successful minimization:",
                     lpad(string(Optim.converged(fpm.optim)), 20)))
  println(io)
  println(io, string("Likelihood approximation:",
                     lpad(typeof(fpm.approx), 19)))
  println(io, string("Log-likelihood value:",
                     lpad(round(loglikelihood(fpm); sigdigits=round(Int, -log10(DEFAULT_ESTIMATION_RELTOL))), 23)))
  println(io, string("Total number of observation records:",
                     lpad(sum([length(sub.time) for sub in fpm.data]), 8)))
  println(io, string("Number of active observation records:",
                     lpad(sum(subject -> sum(name -> count(!ismissing, subject.observations[name]), keys(first(fpm.data).observations)), fpm.data),7)))
  println(io, string("Number of subjects:",
                     lpad(length(fpm.data), 25)))
  println(io)
end


function _coeftable(nt::NamedTuple)
  _keys   = String[]
  _values = numtype(nt)[]
  for (_key, _value) in pairs(nt)
    _push_varinfo!(_keys, _values, nothing, nothing, _key, _value, nothing, nothing)
  end
  return DataFrame(key=_keys, value=_values)
end

"""
    coeftable(fpm::FittedPumasModel) -> DataFrame

Construct a DataFrame of parameter names and estimates from `fpm`.
"""
function StatsBase.coeftable(fpm::FittedPumasModel)
  _df = _coeftable(coef(fpm))
  return DataFrame(parameter=_df.key, estimate=_df.value)
end

function Base.show(io::IO, mime::MIME"text/plain", fpm::FittedPumasModel)
  println(io, "FittedPumasModel\n")
  _print_fit_header(io, fpm)

  # Get a table with the estimates
  coefdf = coeftable(fpm)

  # Round numerical values and convert to strings
  paramvals = map(t -> string(round(t, sigdigits=5)), coefdf.estimate)

  getdecimal = x -> findfirst(c -> c=='.', x)
  maxname = maximum(length, coefdf.parameter)
  maxval = max(maximum(length, paramvals), length("Estimate "))
  labels = " "^(maxname+Int(round(maxval/1.2)) - 3)*"Estimate"
  stringrows = []
  for (name, val) in zip(coefdf.parameter, paramvals)
    push!(stringrows, string(name, " "^(maxname-length(name)-getdecimal(val)+Int(round(maxval/1.2))), val, "\n"))
  end

  show_dashrow_table(io, labels, stringrows)
end

TreeViews.hastreeview(x::FittedPumasModel) = true
function TreeViews.treelabel(io::IO,x::FittedPumasModel,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io, mime, Base.Text(Base.summary(x)))
end

"""
    coeftable(cfpm::Vector{<:FittedPumasModel}) -> DataFrame

Construct a DataFrame of parameter names and estimates and their standard deviation
from vector of fitted single-subject models `vfpm`.
"""
function StatsBase.coeftable(vfpm::Vector{<:FittedPumasModel})

  _mean = _coeftable(mean(vfpm))
  _std  = _coeftable(std(vfpm))

  return DataFrame(parameter=_mean.key, mean=_mean.value, standard_deviation=_std.value)
end

function Base.show(io::IO, mime::MIME"text/plain", vfpm::Vector{<:FittedPumasModel})
  println(io, "Vector{<:FittedPumasModel} with $(length(vfpm)) entries\n")

  # _print_fit_header(io, fpm)
  # Get a table with the estimates
  coefdf = coeftable(vfpm)

  getdecimal = x -> findfirst(c -> c=='.', x)
  maxname = maximum(length, coefdf.parameter)
  paramvals = map(t -> string(round(t, sigdigits=5)), coefdf.mean)
  paramstds = map(t -> string(round(t, sigdigits=5)), coefdf.standard_deviation)
  maxval = max(maximum(length, paramvals), length("Mean"))
  maxstd = max(maximum(length, paramstds), length("Std"))
  labels = " "^(maxname + Int(round(maxval/1.2)) - 2)*rpad("Mean", Int(round(maxstd/1.2)) + maxval)*"Std"
  stringrows = []
  for (name, val, std) in zip(coefdf.parameter, paramvals, paramstds)
    push!(stringrows,
      string(
        name,
        " "^(maxname - length(name) - getdecimal(val) + Int(round(maxval/1.2))),
        val,
        " "^(maxval-(length(val) - getdecimal(val)) - getdecimal(std) + Int(round(maxstd/1.2))),
        std,
        "\n"))
  end

  println(io, "Parameter statistics")
  show_dashrow_table(io, labels, stringrows)
end

TreeViews.hastreeview(x::Vector{<:FittedPumasModel}) = true
function TreeViews.treelabel(io::IO,x::Vector{<:FittedPumasModel},
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io, mime, Base.Text(Base.summary(x)))
end

# _push_varinfo! methods
function _push_varinfo!(_names, _vals, _se, _confint, paramname, paramval::PDMat, std, quant, bootvals=nothing)
  mat = _coef_value(paramval)
  for j = 1:size(mat, 2)
    for i = j:size(mat, 1)
      # We set stdij to nothing in case SEs are not requested to avoid indexing
      # into `nothing`.
      stdij = _se == nothing ? nothing : std[i, j]
      _name = string(paramname)*"$(_to_subscript(i)),$(_to_subscript(j))"
      _bts = bootvals == nothing ? nothing : getindex.(bootvals, j, i)
      _push_varinfo!(_names, _vals, _se, _confint, _name, mat[i, j], stdij, quant, _bts)
    end
  end
end
function _push_varinfo!(_names, _vals, _se, _confint, paramname, paramval::PDiagMat, std, quant, bootvals=nothing)
  mat = _coef_value(paramval)
    for i = 1:size(mat, 1)
      # We set stdii to nothing in case SEs are not requested to avoid indexing
      # into `nothing`.
      stdii = _se == nothing ? nothing : _coef_value(std)[i, i]
      _name = string(paramname)*"$(_to_subscript(i)),$(_to_subscript(i))"
      _bts = bootvals == nothing ? nothing : getindex.(bootvals, i, i)
      _push_varinfo!(_names, _vals, _se, _confint, _name, mat[i, i], stdii, quant, _bts)
    end
end
_push_varinfo!(_names, _vals, _se, _confint, paramname, paramval::Diagonal, std, quant, bootvals=nothing) =
  _push_varinfo!(_names, _vals, _se, _confint, paramname, PDiagMat(paramval.diag), std, quant, bootvals)
function _push_varinfo!(_names, _vals, _se, _confint, paramname, paramval::AbstractVector, std, quant, bootvals=nothing)
  for i in 1:length(paramval)
    # We set stdi to nothing in case SEs are not requested to avoid indexing
    # into `nothing`.
    stdi = _se == nothing ? nothing : std[i]
    _push_varinfo!(_names, _vals, _se, _confint, string(paramname, _to_subscript(i)), paramval[i], stdi, quant, bootvals)
  end
end
function _push_varinfo!(_names, _vals, _se, _confint, paramname, paramval::Number, std, quant, bootvals=nothing)
  # Push variable names and values.
  push!(_names, string(paramname))
  push!(_vals, paramval)
  # We only update SEs and confints if an array was input.
  if _se !== nothing
    push!(_se, std)
  end
  if _confint !== nothing
    if bootvals == nothing
      push!(_confint, (paramval - std*quant, paramval + std*quant))
    else
      push!(_confint, (quantile(bootvals, 0.025), quantile(bootvals, 0.975)))
    end
  end
end


"""
    coeftable(pmi::FittedPumasModelInference) -> DataFrame

Construct a DataFrame of parameter names, estimates, standard error, and confidence
interval from a `pmi`.
"""
function StatsBase.coeftable(pmi::FittedPumasModelInference)

  if pmi.vcov isa Exception
    standard_errors = map(x->NaN*_coef_value(x), coef(pmi))
  else
    standard_errors = stderror(pmi)
  end
  T = numtype(coef(pmi))

  paramnames   = String[]
  paramvals    = T[]
  paramse      = T[]
  paramconfint = Tuple{T,T}[]
  quant = quantile(Normal(), pmi.level + (1 - pmi.level)/2)

  for (paramname, paramval) in pairs(coef(pmi))
    std = standard_errors[paramname]
    bts_param = pmi.vcov isa Bootstraps ? map(p->_coef_value(p[paramname]), coef.(filter(x->isa(x, FittedPumasModel),pmi.vcov.fits))) : nothing
    _push_varinfo!(paramnames, paramvals, paramse, paramconfint, paramname, paramval, std, quant, bts_param)
  end

  for (paramname, paramval) in pairs(coef(pmi))
    lower = (1-pmi.level)/2
#    for p in bts_param
    #println("in a new param")
    #@show bts_param
    #@show _coef_value.(bts_param)
    #@show (quantile(bts_param, lower), quantile(bts_param, 1-lower))
  end
  return DataFrame(parameter=paramnames, estimate=paramvals, se=paramse, ci_lower=first.(paramconfint), ci_upper=last.(paramconfint))
end
function Base.show(io::IO, mime::MIME"text/plain", pmi::FittedPumasModelInference)
  fpm = pmi.fpm

  if pmi.vcov isa Bootstraps
    println(io, "Bootstrap inference results")
  else
    println(io, "Asymptotic inference results")
  end
  println(io)
  _print_fit_header(io, pmi.fpm)
  # Get a table with the estimates standard errors and confidence intervals
  coefdf = coeftable(pmi)

  paramvals    = map(t -> string(round(t, sigdigits=5)), coefdf.estimate)
  paramse      = map(t -> string(round(t, sigdigits=5)), coefdf.se)
  paramconfint = map(t -> string("[", round(t[1], sigdigits=5), ";", round(t[2], sigdigits=5), "]"), zip(coefdf.ci_lower, coefdf.ci_upper))

  getdecimal = x -> occursin("NaN", x) ? 2 : findfirst(isequal('.'), x)
  getsemicolon = x -> findfirst(c -> c==';', x)
  getafterdec = x -> getsemicolon(x) - getdecimal(x)
  getdecaftersemi = x -> getdecimal(x[getsemicolon(x):end])
  getaftersemdec = x -> occursin("NaN", x) ? length(x) - 1 : findall(isequal('.'), x)[2]
  getafterdecsemi = x -> length(x) - getaftersemdec(x)
  maxname = maximum(length, coefdf.parameter)
  maxval = max(maximum(length, paramvals), length("Estimate"))
  maxrs = max(maximum(length, paramse), length("SE"))
  maxconfint = max(maximum(length, paramconfint) + 1, length(string(round(pmi.level*100, sigdigits=6))*"% C.I."))
  maxdecconf = maximum(getdecimal, paramconfint)
  maxaftdec = maximum(getafterdec, paramconfint)
  maxdecaftsem = maximum(getdecaftersemi, paramconfint)
  maxaftdecsem = maximum(getafterdecsemi, paramconfint)
  labels = " "^(maxname+Int(round(maxval/1.2))-3)*rpad("Estimate", Int(round(maxrs/1.2))+maxval+3)*rpad("SE", Int(round(maxconfint/1.2))+maxrs-3)*string(round(pmi.level*100, sigdigits=6))*"% C.I."

  stringrows = []
  for (name, val, se, confint) in zip(coefdf.parameter, paramvals, paramse, paramconfint)
    confint = string("["," "^(maxdecconf - getdecimal(confint)), confint[2:getsemicolon(confint)-1]," "^(maxaftdec-getafterdec(confint)),"; "," "^(maxdecaftsem - getdecaftersemi(confint)), confint[getsemicolon(confint)+1:end-1], " "^(maxaftdecsem - getafterdecsemi(confint)), "]")
    row = string(name, " "^(maxname-length(name)-getdecimal(val)+Int(round(maxval/1.2))), val, " "^(maxval-(length(val)-getdecimal(val))-getdecimal(se)+Int(round(maxrs/1.2))), se, " "^(maxrs-(length(se)-getdecimal(se))-getsemicolon(confint)+Int(round(maxconfint/1.2))), confint, "\n")
    push!(stringrows, row)
  end

  show_dashrow_table(io, labels, stringrows)

  # Print a useful warning if vcov failed
  if pmi.vcov isa Exception
    println(io, """\n\nVariance-covariance matrix could not be\nevaluated. The random effects may be over-\nparameterized. Check the coefficients for\nvariance estimates near zero.""")
  elseif pmi.vcov isa Bootstraps
    println(io, "Successful fits: $(count(x -> x isa FittedPumasModel, pmi.vcov.fits)) out of $(length(pmi.vcov.fits))")
    if pmi.vcov.stratify_by isa Nothing
    println(io, "No stratification.")
    else
      println(io, "Stratification by $(pmi.vcov.stratify_by).")
    end
  end
end
TreeViews.hastreeview(x::FittedPumasModelInference) = true
function TreeViews.treelabel(io::IO,x::FittedPumasModelInference,
                             mime::MIME"text/plain" = MIME"text/plain"())
  show(io,mime,Base.Text(Base.summary(x)))
end

function Base.show(io::IO, mime::MIME"text/plain", pmi::FittedPumasModelInspection)
  println(io, "FittedPumasModelInspection\n")
  print(io, "Fitting was successful: $(Optim.converged(pmi.o.optim))")

  # Only Gaussian error models include weighted residuals
  if pmi.wres !== nothing
    print(io, "\nLikehood approximations used for weighted residuals : $(first(wresiduals(pmi)).approx)")
  end
end

function Base.show(io::IO, mime::MIME"text/plain", sens::SobolOutput)
  println(io, "Sobol Sensitivity Analysis", "\n")
  println(io, "First Order Indices")
  if sens.first_order_conf_int !== nothing
    println(io, sens.first_order_conf_int, "\n")
  else
    println(io, sens.first_order, "\n")
  end
  println(io, "Total Order Indices")
  if sens.total_order_conf_int !== nothing
    println(io, sens.total_order_conf_int, "\n")
  else
    println(io, sens.total_order, "\n")
  end
  if sens.second_order != nothing
    println(io, "Second Order Indices")
    if sens.second_order_conf_int !== nothing
      println(io, sens.second_order_conf_int, "\n")
    else
      println(io, sens.second_order, "\n")
    end
  end
end

function Base.show(io::IO, mime::MIME"text/plain", sens::MorrisOutput)
  println(io, "Morris Sensitivity Analysis", "\n")
  println(io, "Means (μ)")
  println(io, sens.means, "\n")
  println(io, "Means star (μ*)")
  println(io, sens.means_star, "\n")
  println(io, "Variances")
  println(io, sens.variances, "\n")
end

function Base.show(io::IO, mime::MIME"text/plain", sens::eFASTOutput)
  println(io, "eFAST Sensitivity Analysis", "\n")
  println(io, "First Order Indices")
  println(io, sens.first_order, "\n")
  println(io, "Total Order Indices")
  println(io, sens.total_order, "\n")
end

function Base.show(io::IO, mime::MIME"text/plain", vpc::PopVPC)
  println(io, "Data Quantiles", "\n")
  println(io, vpc.data_quantiles)
end

function Base.show(io::IO, mime::MIME"text/plain", vpc::VPC)
  show(io, mime, vpc.popvpc)
  println(io, "\nSimulation Quantiles", "\n")
  println(io, vpc.simulated_quantiles)
end