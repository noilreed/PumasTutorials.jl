using StructArrays
struct ConstantInterpolationStructArray{T, U, D}
  t::T
  u::U
  dir::D
end
function (A::ConstantInterpolationStructArray{<:Any,<:StructArray})(t::Number)
  if A.dir === :left
    # :left means that value to the left is used for interpolation
    i = searchsortedlast(A.t, t)
    return A.u[max(1, i)]
  else
    # :right means that value to the right is used for interpolation
    i = searchsortedfirst(A.t, t)
    return A.u[min(length(A.t), i)]
  end
end

struct NoCovar end
(nc::NoCovar)(t) = nothing
struct ConstantCovar{C}
  cvs_nt::C
end
(cc::ConstantCovar)(t=nothing) = cc.cvs_nt
"""
  build_tvcov(cvs, data, time)

Creates an interpolation of the time-varying covariate u at time points t using
the interpolation scheme interp from DataInterpolations.jl. Returns a function
`(t)` that does the interpolation as well as the common time grid for the covariate
observations. This is safe for values which are not time-varying as well, allowing
one to mix subjects with multiple measurements and subjects with a single measurement.
Defaults to do a left-sided ConstantInterpolation.
"""
function build_tvcov(cvs_keys,
                     data,
                     time::Union{Nothing,Symbol},
                     id;
                     interp=ConstantInterpolationStructArray)

  # Create helper named tuple. It has keys == values so you can easily map
  # over it, use the name and get a namedtuple back.
  cvs_keys = Tuple(cvs_keys) # from vector to tuple
  cvs_nt = NamedTuple{cvs_keys}(cvs_keys)

  # Get the covariate / covariate times into correct form
  if data isa AbstractDataFrame && length(cvs_keys) > 0
    # If it's a dataframe we need to get the full time column and the
    # covariate column together, drop the missing values (there can be
    # times without covariates and vice versa) and then we convert it
    # to a name tuple (ctime=, name=)
    covar_nt = df_to_nt(cvs_nt, data, time)
    if all(covar -> (covar_nt[covar] isa String || length(unique(covar_nt[covar][covar])) == 1), cvs_keys)
      return 0.0, ConstantCovar(NamedTuple{cvs_keys}(map(c->covar_nt[c][c], cvs_keys)))
    end
  elseif data isa NamedTuple
    covar_nt = data
  else
    return 0.0, NoCovar()
  end

  # If any of the covariates is all missings then we error out
  for (k, v) in pairs(covar_nt)
    if isempty(v.ctime)
      throw(ArgumentError("covariate $k for subject with ID $id had no non-missing values"))
    end
  end

  # Create vector of union of time indices
  tvcov_times = AbstractFloat.(sort(unique(vcat([t.ctime for t in covar_nt]...))))

  tvcov_nt = map(covar_nt) do x
    _time, _covar = x
    if _covar isa AbstractArray
      if length(unique(_covar)) == 1
        interped = fill(first(_covar), length(tvcov_times))
      else
        interpi = DataInterpolations.ConstantInterpolation(_covar, _time, dir=:right)
        interped = interpi.(tvcov_times)
      end
      return interped
    elseif any(isa.(Ref(_covar), (String, Number)))
      # In the Subject constructor, users can specify covariates
      # like this: Subject(; cvs=(sex="female", k=1)) for constant
      # covariates.
      return fill(_covar, length(tvcov_times))
    end
  end
  tvcov_sa = StructArray(tvcov_nt)

  # this of course needs to allow for mixed interpolants and customly chosen interpolants
  return tvcov_times, interp(tvcov_times, tvcov_sa, :right)
end
# Should this be special cased throughout?  I mean should we dispatch on a "constant covariates" type (and pre?)
# We could have a pre object that also holds covariates.
build_tvcov(cvs_nt::Nothing, time, id) = (@SVector([0.0]), t->())

# Keys as vec
function build_tvcov(cvs_nt::NamedTuple, cvstimes, id)
  # This is the case where everything was passed in the correct form from the Subject
  # constructor
  # We allow for two ways of entering covariate times. Either you
  #   - a) Give a named tuple with the same keys as the cvs named tuple for individual observation times, or
  #   - b) one vector of times that is then supposed to fit all covariates
  cvs_keys = keys(cvs_nt)
  cvs_keys_nt = NamedTuple{cvs_keys}(cvs_keys)
  if cvstimes isa NamedTuple # in a)
    cvs_data = map(cvs_keys_nt) do name
        (ctime=cvstimes[name], name=cvs_nt[name])
      end
  elseif !isnothing(cvstimes) # in b) - use cvstimes for all covariates...
    cvs_data = map(cvs_keys_nt) do name
      if length(cvstimes) == length(cvs_nt[name])
        return (ctime=AbstractFloat.(cvstimes), name=cvs_nt[name])
      else
        # ... unless the user failed to provide a proper cvstimes vector.
        throw(ErrorException("Length of covariate $name ($(length(cvs_nt[name]))) does not match length of cvstimes ($(length(cvstimes)))."))
      end
    end
  end
  build_tvcov(cvs_keys, cvs_data, nothing, id)
end
# No times were given, but there were covariates. Either this is an
# error, or they're not varying over time (any of them!).

function build_tvcov(cvs_nt::NamedTuple, cvstimes_nt::Nothing, id)
  # You can only reach this method from the raw Subject constructor. If
  # build_tvcov is called from the read_pumas call, the input would have
  # a time column where we could grab these.
  cvs_keys = keys(cvs_nt)
  cvs_keys_nt = NamedTuple{cvs_keys}(cvs_keys)
  # If all covariates are constant, simply return the namedtuple
  # provided by the user.
  if all(covar -> (covar isa String || length(covar) == 1), cvs_nt)
    return 0.0, ConstantCovar(cvs_nt)
  end
  # Else we require the user to specify the times each covariate
  # is observed. Say cvs_nt = (wt = [50.0, 75.0, 100.0], isPM = "no"),
  # then you have to input cvstime = (wt =[0.0, 10.0, 30.0], isPM = [0.0]).
  throw(ErrorException("You must provide covariate times for subject $id using the `cvstimes` keyword when specifying time-varying covariates."))
end

function df_to_nt(cvs_nt, data, time::Union{Symbol, Int})
  covar_nt = map(cvs_nt) do name
             if name == time
               throw(ErrorException(":time should not be used to reference current time for covariates. Please consult the documatation on time varying covariates for more information."))
             end
             dat = dropmissing(data[!, [time, name]])
             dat[!, :ctime] = dat[!, time]
             dat = disallowmissing(dat)
             to_nt(dat[!, [:ctime, name]])
           end
end
