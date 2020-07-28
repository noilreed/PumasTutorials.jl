export ParamSet, RealDomain, VectorDomain, PSDDomain, PDiagDomain, Constrained

abstract type Domain end

Domain(d::Domain) = d

"""
    @param x = val
    @param x ∈ ConstDomain(val)

Specifies a parameter as a constant.
"""
struct ConstDomain{T} <: Domain
  val::T
end
init(d::ConstDomain) = d.val

"""
    @param x ∈ RealDomain(;lower=-∞,upper=∞,init=0)

Specifies a parameter as a real value. `lower` and `upper` are the respective bounds, `init` is the value used as the initial guess in the optimisation.
"""
struct RealDomain{L,U,T} <: Domain
  lower::L
  upper::U
  init::T
end
RealDomain(;lower=-∞,upper=∞,init=0.) = RealDomain(lower, upper, init)
init(d::RealDomain) = d.init


"""
    @param x ∈ VectorDomain(n::Int; lower=-∞,upper=∞,init=0)

Specifies a parameter as a real vector of length `n`. `lower` and `upper` are the respective bounds, `init` is the value used as the initial guess in the optimisation.
"""
struct VectorDomain{L,U,T} <: Domain
  lower::L
  upper::U
  init::T
end

_vec(n, x::AbstractVector) = x
_vec(n, x) = fill(x, n)

VectorDomain(n::Int; lower=-∞,upper=∞,init=0.0) = VectorDomain(_vec(n,lower), _vec(n,upper), _vec(n,init))

init(d::VectorDomain) = d.init


"""
    @param x ∈ PSDDomain(n::Int; init=Matrix{Float64}(I, n, n))

Specifies a parameter as a symmetric `n`-by-`n` positive semidefinite matrix.
"""
struct PSDDomain{T} <: Domain
  init::T
end
PSDDomain(; init=nothing) = PSDDomain(init)
PSDDomain(n::Int)         = PSDDomain(init=Matrix{Float64}(I, n, n))

init(d::PSDDomain) = d.init


"""
    @param x ∈ PDiagDomain(n::Int; init=ones(n))

Specifies a parameter as a positive diagonal matrix, with diagonal elements
specified by `init`.
"""
struct PDiagDomain{T} <: Domain
  init::T
end
PDiagDomain(; init=missing) = PDiagDomain(PDMats.PDiagMat(init))
PDiagDomain(n::Int)         = PDiagDomain(init=ones(n))

init(d::PDiagDomain) = d.init


# domains of random variables
function Domain(d::MvNormal)
  n = length(d)
  VectorDomain(fill(-∞, n), fill(∞, n), mean(d))
end
Domain(d::Union{Wishart,InverseWishart}) = PSDDomain(Distributions.dim(d))
Domain(d::Normal) = RealDomain(lower=-∞, upper=∞, init=mean(d))
Domain(d::Uniform) = RealDomain(lower=minimum(d), upper=maximum(d), init=mean(d))
Domain(d::Union{Gamma,InverseGamma,Exponential,Chisq}) = RealDomain(lower=zero(partype(d)), upper=∞, init=mean(d))

function Domain(d::Distribution)
  throw(ArgumentError("no Domain(::$(typeof(d))) constructor defined"))
end


struct ParamSet{T}
  params::T
end
Base.keys(ps::ParamSet) = keys(ps.params)

domains(p::ParamSet) = map(Domain, p.params)

init(p::ParamSet) = map(init, domains(p))

Base.rand(rng::AbstractRNG, p::ParamSet) = map(s -> rand(rng, s), p.params)

_vecmean(p::ParamSet) = vcat(map(mean, p.params)...)
# FIXME! To this in a better way
function _veccov(p::ParamSet)
  Ωis = map(cov, p.params)
  return hvcat(length(Ωis), Diagonal([Ωis...])...)
end

"""
    Constrained

Constrain a `Distribution` within a `Domain`. The most common case is an `MvNormal` constrained within a `VectorDomain`. The only supported method for `Constrained` is `logpdf`. Notice that the result does not represent a probability distribution since the remaining probability mass is not scaled by the mass excluded by the constraints.

# Example
```jldoctest
julia> d = Constrained(MvNormal(fill(1.0, 1, 1)), lower=-1, upper=1)
Constrained{MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},FillArrays.Zeros{Float64,1,Tuple{Base.OneTo{Int64}}}},VectorDomain{Array{Int64,1},Array{Int64,1},Array{Float64,1}}}(ZeroMeanFullNormal(
dim: 1
μ: [0.0]
Σ: [1.0]
)
, VectorDomain{Array{Int64,1},Array{Int64,1},Array{Float64,1}}([-1], [1], [0.0]))

julia> logpdf(d, [ 0])
-0.9189385332046728

julia> logpdf(d, [-2])
```
"""
struct Constrained{D<:Distribution,M<:Domain}
  dist::D
  domain::M
end

Constrained(d::MvNormal; lower=-∞, upper=∞, init=0.0) =
  Constrained(d, VectorDomain(length(d); lower=lower, upper=upper, init=init))

Constrained(dist::ContinuousUnivariateDistribution; lower=-∞, upper=∞, init=0.0) =
  Constrained(dist, RealDomain(; lower=lower, upper=upper, init=init))

Domain(c::Constrained) = c.domain

# This doesn't have into account the probability mass outside of the
# constraints but it shouldn't matter much since this won't affect MCMC
# sampling which is the primary application for constrained distributions
function Distributions.logpdf(d::Constrained, x)
  v = logpdf(d.dist, x)
  if all(((_x, _l, _u),) -> _l <= _x <= _u, zip(x, d.domain.lower, d.domain.upper))
    return v
  else
    return oftype(v, -Inf)
  end
  return v
end

# Some convenience piracy
Base.isless(x::Number, ::TransformVariables.Infinity{true}) = x !== Inf
Base.isless(::TransformVariables.Infinity{false}, x::Number) = x !== -Inf
