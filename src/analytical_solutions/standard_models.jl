export Central1, Depots1Central1, Depots2Central1,
       Central1Periph1, Depots1Central1Periph1 ,
       Central1Periph1Meta1, Central1Periph1Meta1Periph1

abstract type ExplicitModel end

# Generic ExplicitModel solver. Uses an analytical eigen solution.
function _analytical_solve(m::M, t, t₀, amounts, doses, pre, rates) where M<:ExplicitModel
  p = pre(t₀)
  amt₀ = amounts + doses   # initial values for cmt's + new doses
  Λ, 𝕍 = eigen(m, p)

  # We avoid the extra exp calls, but could have written:
  # Dh  = Diagonal(@SVector(exp.(λ * (_t - _t₀)))
  # Dp  = Diagonal(@SVector(expm1.(λ * (_t - _t₀))./λ))
  # We could also have written:
  # Dp = Diagonal(expm1.(Λ * (t - t₀)) ./ Λ)
  # Dh = Dp .* Λ + I
  # but Diagonal{StaticVector} falls back to Array operations. Instead we write:
  dp = expm1.(Λ * (t - t₀)) ./ Λ
  dh = dp .* Λ .+ 1

  # We cannot * here because of Array fallback for Diagonal{StaticVector}
  # amtₜ = 𝕍*(Dp*(𝕍\rates) + Dh*(𝕍\amt₀)) # could derive inverse here
  amtₜ = 𝕍*(dp.*(𝕍\rates) + dh.*(𝕍\amt₀)) # could derive inverse here

  return SLVector(NamedTuple{varnames(M)}(amtₜ))
end
DiffEqBase.has_syms(x::ExplicitModel) = true
Base.getproperty(x::ExplicitModel, symbol::Symbol) = symbol == :syms ? Pumas.varnames(typeof(x)) : getfield(x, symbol)
"""
    Central1()

An analytical model for a one compartment model with dosing into `Central`. Equivalent to

  Central' = -CL/Vc*Central

where clearance, `CL`, and volume, `Vc`, are required to be defined in the `@pre` block.
"""
struct Central1 <: ExplicitModel end
(m::Central1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Central1, p)
  Ke = p.CL/p.Vc
  T = typeof(Ke)

  Λ = @SVector([-Ke])
  𝕍 = @SMatrix([T(1)])

  return Λ, 𝕍
end
varnames(::Type{Central1}) = (:Central,)
pk_init(::Central1) = SLVector(Central=0.0)

"""
    Depots1Central1()

An analytical model for a one compartment model with a central compartment, `Central`, and a depot, `Depot`. Equivalent to

  Depot'   = -Ka*Depot   
  Central' = -CL/Vc*Central  

where absoption rate, `Ka`, clearance, `CL`, and volume, `Vc`, are required to be defined in the `@pre` block.
"""
struct Depots1Central1 <: ExplicitModel end
(m::Depots1Central1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Depots1Central1, p)
    a = p.Ka
    e = p.CL/p.Vc

    Λ = @SVector([-a, -e])
    v = e/a - 1
    𝕍 = @SMatrix([v 0;
                  1 1])
    return Λ, 𝕍
end
varnames(::Type{Depots1Central1}) = (:Depot, :Central)
pk_init(::Depots1Central1) = SLVector(Depot=0.0,Central=0.0)

"""
    Depots2Central1()

An analytical model for a one compartment model with a central compartment, `Central`, and two depots, `Depot1` and `Depot2`. Equivalent to

  Depot1'   = -Ka1*Depot1  
  Depot2'   = -Ka2*Depot2  
  Central' = -CL/Vc*Central  

where absorption rates, `Ka1` and `Ka2`, clearance, `CL`, and volume, `Vc`, are required to be defined in the `@pre` block. 

When using this model during simulation or estimation, it is preferred to have 2 dosing rows for each subject in the dataset, where the first dose goes into `cmt =1` (or `cmt = Depot1`) and the second dose goes into `cmt=2` (or `cmt=Depot2`). Central compartment gets `cmt=3` or (`cmt = Central`). e.g.

ev = DosageRegimen([100,100],cmt=[1,2])
s1 = Subject(id=1, evs=ev)
"""
struct Depots2Central1 <: ExplicitModel end
(m::Depots2Central1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Depots2Central1, p)
    a = p.Ka1
    b = p.Ka2
    e = p.CL/p.Vc

    frac1 = (e-a)/a
    invfrac1 = inv(frac1)

    frac2 = (e-b)/b
    invfrac2 = inv(frac2)

    Λ = @SVector([-a, -b, -e])

    v1 = -1 + e/a
    v2 = -1 + e/b
    𝕍 = @SMatrix([frac1 0     0;
                  0     frac2 0;
                  1     1     1])

    return Λ, 𝕍
end
varnames(::Type{Depots2Central1}) = (:Depot1, :Depot2, :Central)
pk_init(::Depots2Central1) = SLVector(Depot1=0.0,Depot2=0.0,Central=0.0)

"""
    Central1Periph1()

An analytical model for a two-compartment model with a central compartment, `Central` and a peripheral compartment, `Peripheral`. Equivalent to

  Central'    = -(CL+Q)/Vc*Central + Q/Vp*Peripheral  
  Peripheral' =        Q/Vc*Central - Q/Vp*Peripheral  

where clearance, `CL`, and volumes, `Vc` and `Vp`, and distribution clearance, `Q`, are required to be defined in the `@pre` block.
"""
struct Central1Periph1 <: ExplicitModel end
_V(::Central1Periph1, Λ, b, c) = @SMatrix([(Λ[1]+c)/b (Λ[2]+c)/b])
function _Λ(::Central1Periph1, a, b, c)
  # b is from actual cmt to peri, c is back
  A = a + b + c
  S = sqrt(A^2-4*a*c)
  Λ = @SVector([-(A+S)/2, -(A-S)/2])
end
(m::Central1Periph1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(m::Central1Periph1, p)
    a = p.CL/p.Vc
    b = p.Q/p.Vc
    c = p.Q/p.Vp

    Λ = _Λ(m, a, b, c)
    𝕍 = vcat(_V(m, Λ, b, c), @SMatrix([1 1]))

    return Λ, 𝕍
end
varnames(::Type{Central1Periph1}) = (:Central, :Peripheral)
pk_init(::Central1Periph1) = SLVector(Central=0.0, Peripheral=0.0)

"""
    Depots1Central1Periph1()

An analytical model for a two-compartment model with a central compartment, `Central`, a peripheral compartment, `Peripheral`, and a depot `Depot`. Equivalent to
  
  Depot'      = -Ka*Depot  
  Central'    =  Ka*Depot -(CL+Q)/Vc*Central + Q/Vp*Peripheral  
  Peripheral' =                  Q/Vc*Central - Q/Vp*Peripheral  

where absorption rate, `Ka`, clearance, `CL`, and volumes, `Vc` and `Vp`, and distribution clearance, `Q`, are required to be defined in the `@pre` block.
"""
struct Depots1Central1Periph1  <: ExplicitModel end
(m::Depots1Central1Periph1 )(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Depots1Central1Periph1 , p)
  k = p.Ka
  a = p.CL/p.Vc
  b = p.Q/p.Vc
  c = p.Q/p.Vp

  A = a + b + c

  Λ, 𝕍 = eigen(Central1Periph1(), p)
  Λ = pushfirst(Λ, -k)

  𝕍 = vcat(@SMatrix([0 0;]), 𝕍) # pad with zeros
  v_depot = @SMatrix([((k-A)+a*c/k)/b; (c-k)/b; 1])
  𝕍 = hcat(v_depot, 𝕍)

  return Λ, 𝕍, inv(𝕍)
end
varnames(::Type{Depots1Central1Periph1 }) = (:Depot, :Central, :Peripheral)
pk_init(::Depots1Central1Periph1 ) = SLVector(Depot=0.0, Central=0.0, Peripheral=0.0)


"""
    Central1Periph1Meta1Periph1()

An analytical model for a two compartment model with a central compartment, `Central`, with a peripheral compartment, `Peripheral`, and a metabolite compartment, `Metabolite`, with a peripheral compartment, `MPeripheral`. Equivalent to
  
  Central'     = -(CL+Q+CLfm)/Vc*Central + Q/Vp*CPeripheral  
  CPeripheral' =          Q/Vc*Central - Q/Vp*CPeripheral  
  Metabolite'  = -(CLm+Qm)/Vm*Metabolite + Qm/Vmp*MPeripheral + CLfm/Vc*Central  
  MPeripheral' =        Qm/Vm*Metabolite - Qm/Vmp*MPeripheral  

where clearances (`CL` and `CLm`) and volumes (`Vc`, `Vp`, `Vm` and `Vmp`), distribution clearances (`Q` and `Qm`) and formation clearance of metabolite `CLfm` are required to be defined in the `@pre` block.
"""
struct Central1Periph1Meta1Periph1 <: ExplicitModel end # 011?
(m::Central1Periph1Meta1Periph1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Central1Periph1Meta1Periph1, p)
  a = p.CL/p.Vc
  b = p.Q/p.Vc
  c = p.Q/p.Vp
  d = p.CLfm/p.Vc
  e = p.CLm/p.Vm
  f = p.Qm/p.Vm
  h = p.Qm/p.Vmp

  β = a + b
  ϕ = e + f

  m′ = Central1Periph1()
  Λ = vcat(_Λ(m′, a, b, c),  _Λ(m′, e, f, h))

  v1_3 = ( Λ[1] + h)/f
  v1_1 = ((Λ[1] + ϕ) * v1_3 - h)/d
  v1_2 = ( Λ[1] + β) * (v1_1 + h/d)/c - (Λ[1] + β)*h/(c*d)

  v2_3 = ( Λ[2] + h)/f
  v2_1 = ((Λ[2] + ϕ) * v2_3 - h)/d
  v2_2 = ( Λ[2] + β) * (v2_1 + h/d)/c - (Λ[2] + β)*h/(c*d)


  v3_3 = (Λ[3] + h)/f
  v4_3 = (Λ[4] + h)/f

  𝕍 = @SMatrix([v1_1  v2_1  0   0  ;
                v1_2  v2_2  0   0  ;
                v1_3  v2_3  v3_3 v4_3;
                1     1    1   1])

  return Λ, 𝕍
end
varnames(::Type{Central1Periph1Meta1Periph1}) = (:Central, :CPeripheral, :Metabolite, :MPeripheral)
pk_init(::Central1Periph1Meta1Periph1) = SLVector(Central=0.0, CPeripheral=0.0, Metabolite=0.0, MPeripheral=0.0
)

"""
    Central1Periph1Meta1()

An analytical model for a two compartment model with a central compartment, `Central`, with a peripheral compartment, `Peripheral`, and a metabolite compartment, `Metabolite`. Equivalent to
  
  Central'     = -(CL+Q+CLfm)/Vc*Central + Q/Vp*CPeripheral  
  CPeripheral' =          Q/Vc*Central - Q/Vp*CPeripheral  
  Metabolite'  = -CLm/Vm*Metabolite + CLfm/Vc*Central  

where clearances (`CL` and `CLm`) and volumes (`Vc`, `Vp` and `Vm`), distribution clearance (`Q`), and formation clearance of metabolite `CLfm` are required to be defined in the `@pre` block.
"""
struct Central1Periph1Meta1 <: ExplicitModel end
(m::Central1Periph1Meta1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(m::Central1Periph1Meta1, p)
  a = p.CL/p.Vc
  b = p.Q/p.Vc
  c = p.Q/p.Vp
  d = p.CLfm/p.Vc
  e = p.CLm/p.Vm

  β = a + b
  Λ = vcat(_Λ(Central1Periph1(), a, b, c), @SVector([-e]))

  v1_1 = (Λ[1] + e)/d
  v1_2 = (Λ[1] + β)*v1_1/c
  v2_1 = (Λ[2] + e)/d
  v2_2 = (Λ[2] + β)*v2_1/c

  𝕍 = @SMatrix([v1_1 v2_1 0;
                v1_2 v2_2 0;
                1    1    1])

  return Λ, 𝕍
end
varnames(::Type{Central1Periph1Meta1}) = (:Central, :CPeripheral, :Metabolite)
pk_init(::Central1Periph1Meta1) = SLVector(Central=0.0, CPeripheral=0.0, Metabolite=0.0)
