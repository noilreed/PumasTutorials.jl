using StructArrays

struct ConstantInterpolationStructArray{T, U, D}
  t::T
  u::U
  dir::D
end

function (A::ConstantInterpolationStructArray{<:Any,<:StructArray,<:Any})(t::Number)
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

Tables.columns(cisa::ConstantInterpolationStructArray) = (time=cisa.t, fieldarrays(cisa.u)...)

_covariates_time(cisa::ConstantInterpolationStructArray) = cisa.t

struct NoCovar end
(nc::NoCovar)(t) = nothing

_covariates_time(nc::NoCovar) = 0.0

struct ConstantCovar{C}
  u::C
end
(cc::ConstantCovar)(t=nothing) = cc.u

Tables.columns(cc::ConstantCovar) = map(t -> [t], cc.u)

_covariates_time(cc::ConstantCovar) = 0.0

"""
  covariates_interpolant(covariates, data, time)

Creates an interpolation of the time-varying covariate u at time points t using
the interpolation scheme interp from DataInterpolations.jl. Returns a function
`(t)` that does the interpolation as well as the common time grid for the covariate
observations. This is safe for values which are not time-varying as well, allowing
one to mix subjects with multiple measurements and subjects with a single measurement.
Defaults to do a left-sided ConstantInterpolation.
"""
function covariates_interpolant(
  covariates_keys,
  data,
  time::Union{Nothing,Symbol},
  id;
  interp=ConstantInterpolationStructArray,
  covariates_direction=:right)

  _covariates_interpolant(covariates_keys, data, time, interp, id, covariates_direction)
end

function _covariates_interpolant(covariates_keys, data, time, interp, id, direction)
  # Create helper named tuple. It has keys == values so you can easily map
  # over it, use the name and get a namedtuple back.
  covariates_keys = Tuple(covariates_keys) # from vector to tuple
  covariates_nt = NamedTuple{covariates_keys}(covariates_keys)

  # Get the covariate / covariate times into correct form
  if data isa AbstractDataFrame && length(covariates_keys) > 0
    # If it's a dataframe we need to get the full time column and the
    # covariate column together, drop the missing values (there can be
    # times without covariates and vice versa) and then we convert it
    # to a name tuple (ctime=, name=)
    covar_nt = df_to_nt(covariates_nt, data, time)
    if all(covar -> (covar_nt[covar][covar] isa String || length(unique(covar_nt[covar][covar])) == 1), covariates_keys)
      return 0.0, ConstantCovar(NamedTuple{covariates_keys}(map(c->covar_nt[c][c], covariates_keys)))
    end
  elseif data isa NamedTuple
    covar_nt = data
  else
    return 0.0, NoCovar()
  end

  # If any of the covariates is all missings then we error out
  for (k, v) in pairs(covar_nt)
    if isempty(v.ctime)
      throw(PumasDataError("covariate $k for subject with ID $id had no non-missing values"))
    end
  end

  # Create vector of union of time indices
  tvcov_time = AbstractFloat.(sort(unique(vcat([t.ctime for t in covar_nt]...))))

  tvcov_nt = map(covar_nt) do x
    _time, _covar = x
    if _covar isa AbstractArray
      if length(unique(_covar)) == 1
        interped = fill(first(_covar), length(tvcov_time))
      else
        interpi = DataInterpolations.ConstantInterpolation(_covar, _time, dir=direction)
        interped = interpi.(tvcov_time)
      end
      return interped
    elseif any(isa.(Ref(_covar), (String, Number)))
      # In the Subject constructor, users can specify covariates
      # like this: Subject(; covariates=(sex="female", k=1)) for constant
      # covariates.
      return interped = fill(_covar, length(tvcov_time))
    end
  end

  tvcov_sa = StructArray(tvcov_nt)
  return tvcov_time, interp(tvcov_time, tvcov_sa, direction)
end
# Should this be special cased throughout?  I mean should we dispatch on a "constant covariates" type (and pre?)
# We could have a pre object that also holds covariates.
covariates_interpolant(covariates_nt::Nothing, time, id; covariates_direction=covariates_direction) =
  (@SVector([0.0]), NoCovar())

# Keys as vec
function covariates_interpolant(
  covariates_nt::NamedTuple,
  covariates_times,
  id;
  covariates_direction=:right)

  # This is the case where everything was passed in the correct form from the Subject
  # constructor
  # We allow for two ways of entering covariate times. Either you
  #   - a) Give a named tuple with the same keys as the covariates named tuple for individual observation times, or
  #   - b) one vector of times that is then supposed to fit all covariates
  covariates_keys = keys(covariates_nt)
  covariates_keys_nt = NamedTuple{covariates_keys}(covariates_keys)
  if covariates_times isa NamedTuple # in a)
    covariates_data = map(covariates_keys_nt) do name
      (ctime=covariates_times[name], name=covariates_nt[name])
    end
  elseif !isnothing(covariates_times) # in b) - use covariates_times for all covariates...
    covariates_data = map(covariates_keys_nt) do name
      if length(covariates_times) == length(covariates_nt[name])
        return (ctime=AbstractFloat.(covariates_times), name=covariates_nt[name])
      else
        # ... unless the user failed to provide a proper covariates_times vector.
        throw(ErrorException("Length of covariate $name ($(length(covariates_nt[name]))) does not match length of covariates_time ($(length(covariates_times)))."))
      end
    end
  end
  return covariates_interpolant(covariates_keys, covariates_data, nothing, id; covariates_direction=covariates_direction)
end
# No times were given, but there were covariates. Either this is an
# error, or they're not varying over time (any of them!).

# FIXME need one for BOV and no covariates!!!covariates_interpolant
function covariates_interpolant(
  covariates_nt::NamedTuple,
  covariates_times_nt::Nothing,
  id;
  covariates_direction=:right)

  # You can only reach this method from the raw Subject constructor. If
  # covariates_interpolant is called from the read_pumas call, the input would have
  # a time column where we could grab these.
  covariates_keys = keys(covariates_nt)
  covariates_keys_nt = NamedTuple{covariates_keys}(covariates_keys)
  # If all covariates are constant, simply return the namedtuple
  # provided by the user.
  if all(covar -> (covar isa String || length(covar) == 1), covariates_nt)
    return 0.0, ConstantCovar(covariates_nt)
  end
  # Else we require the user to specify the times each covariate
  # is observed. Say covariates_nt = (wt = [50.0, 75.0, 100.0], isPM = "no"),
  # then you have to input covariates_time = (wt =[0.0, 10.0, 30.0], isPM = [0.0]).
  throw(ErrorException("You must provide covariate times for subject $id using the `covariates_time` keyword when specifying time-varying covariates."))
end

function df_to_nt(covariates_nt, data, time::Union{Symbol, Int})
  map(covariates_nt) do name
    if name == time
      throw(ErrorException(":time should not be used to reference current time for covariates. Please consult the documentation on time varying covariates for more information."))
    end
    dat = dropmissing(data[!, [time, name]])
    dat[!, :ctime] = dat[!, time]
    dat = disallowmissing(dat)
    return to_nt(dat[!, [:ctime, name]])
  end
end
