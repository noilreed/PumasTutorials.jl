using Pumas, Test


@test Central1().syms == (:Central,)
@test Pumas.DiffEqBase.has_syms(Central1())
@test Depots1Central1().syms == (:Depot, :Central)
@test Pumas.DiffEqBase.has_syms(Depots1Central1())
@test Depots2Central1().syms == (:Depot1, :Depot2, :Central)
@test Pumas.DiffEqBase.has_syms(Depots2Central1())

#==
  Central1

  Eigen values and vectors are very simple: they're just -CL/Vc and [1] respectively.
==#
p = (CL=rand(), Vc=rand())
ocm = Pumas.LinearAlgebra.eigen(Central1(), p)
@test all(ocm[1] .== -p.CL/p.Vc)
@test all(ocm[2] .== [1.0])

#==
  Depots1Central1

  Eigen values are just: Λ = [-Ka, -CL/Vc]
  Eigen vectors are [Λ[1]/Λ[2] - 1, 0] and [0,1]
==#

p = (Ka=rand(), CL=rand(), Vc=rand())
ocm = Pumas.LinearAlgebra.eigen(Depots1Central1(), p)
@test all(ocm[1] .== [-p.Ka, -p.CL/p.Vc])
@test all(ocm[2] .== [ocm[1][2]/ocm[1][1]-1  0.0; 1.0 1.0])

#==
  Two compartment, one Central one Peripheral. Compare against solution of nu-
  merical matrix given to eigen solver.
==#

p = (CL=0.1, Vc=1.0, Vp=2.0, Q=0.5)
ocm = Pumas.LinearAlgebra.eigen(Central1Periph1(), p)
λ1 = -(17+sqrt(249))/40
λ2 = -(17-sqrt(249))/40
@test all(ocm[1] .≈ [λ1, λ2])
@test all(ocm[2] .≈ [2*λ1+1/2 2*λ2+1/2; 1 1])

p = (CL=0.1, Vc=5.0, Vp=2.0, Q=0.5)
ocm = Pumas.LinearAlgebra.eigen(Central1Periph1(), p)
λ1 = -(37+sqrt(1169))/200
λ2 = -(37-sqrt(1169))/200
v1 = -(-13+sqrt(1169))/20
v2 = -(-13-sqrt(1169))/20
@test all(ocm[1] .≈ [λ1, λ2])
@test all(ocm[2] .≈ [v1 v2; 1 1])

#==
  Depots1Central1Periph1
==#

p = (Ka=0.05, CL=0.1, Vc=1.0, Vp=2.0, Q=0.5)
ocm = Pumas.LinearAlgebra.eigen(Depots1Central1Periph1(), p)
λ1 = -(17+sqrt(249))/40
λ2 = -(17-sqrt(249))/40
@test all(ocm[1] .≈ [-p.Ka, λ1, λ2])
@test all(ocm[2] .≈ [-3/5 0 0; 2/5 2*λ1+1/2 2*λ2+1/2; 1 1 1])

p = (Ka=0.05, CL=0.1, Vc=5.0, Vp=2.0, Q=0.5)
ocm = Pumas.LinearAlgebra.eigen(Depots1Central1Periph1(), p)
λ1 = -(37+sqrt(1169))/200
λ2 = -(37-sqrt(1169))/200
v1 = -(-13+sqrt(1169))/20
v2 = -(-13-sqrt(1169))/20
@test all(ocm[1] .≈ [-p.Ka, λ1, λ2])
@test all(ocm[2] .≈ [-2.2 0 0; 2.0 v1 v2; 1 1 1])

#==
  Central1Periph1Meta1Periph1
==#
p = (CL=0.01, CLm=0.1, Vc=5.0, Vp=2.0, Vm=6.0, Vmp=3.0, Q=0.5, Qm=1.2, CLfm=0.005)
ocm = Pumas.LinearAlgebra.eigen(Central1Periph1Meta1Periph1(), p)
λ1 = -(353+sqrt(121609))/2000
λ2 = -(353-sqrt(121609))/2000
λ3 = -(37+sqrt(1273))/120
λ4 = -(37-sqrt(1273))/120
  V = [(-243723-791*sqrt(121609))/1200 (-243723+791*sqrt(121609))/1200 0 0;
      (6601+18*sqrt(121609))/30 (6601-18*sqrt(121609))/30  0  0;
      (447-sqrt(121609))/400    (447+sqrt(121609))/400     (11-sqrt(1273))/24 (11+sqrt(1273))/24;
      1 1 1 1]
@test all(ocm[1] .≈ [λ1, λ2, λ3, λ4])
@test all(ocm[2] .≈ V)


# test for #732
# If two doses were given at the same time or if a rate dosage regimen was
# specified after an instant dosage regimen, the instant dosage would be overwritten.
# We simply test that it is accumulated at the time of dose.
model732 = @model begin
  @pre begin
    Ka = 0.01
    CL = 1.0
    Vc = 3.0
  end
  @dynamics Central1
  @derived begin
    dv ~ @. Normal(Central/Vc)
  end
end


doses_R = DosageRegimen(43, cmt=1, time=3, rate=5)
doses_D = DosageRegimen(43, cmt=1, time=3, rate=0)

doses_DD = DosageRegimen(doses_D, doses_D)
doses_DR = DosageRegimen(doses_D, doses_R)
doses_RD = DosageRegimen(doses_R, doses_D)

dose = doses_RD
pop     = Population(map(i -> Subject(id=i, events=dose),1:15))
_simobs = simobs(model732, pop, NamedTuple())
simdf = DataFrame(_simobs)
data = read_pumas(simdf)
for i = 1:15
  sol = solve(model732, data[i], NamedTuple())

  @test sol(3.0)[1] ≈ 43.0
end

dose = doses_DD
pop     = Population(map(i -> Subject(id=i, events=dose),1:15))
_simobs = simobs(model732, pop, NamedTuple())
simdf = DataFrame(_simobs)
data = read_pumas(simdf)
for i = 1:15
  sol = solve(model732, data[i], NamedTuple())

  @test sol(3.0)[1] ≈ 86.0
end


dose = doses_DR
pop     = Population(map(i -> Subject(id=i, events=dose),1:15))
_simobs = simobs(model732, pop, NamedTuple())
simdf = DataFrame(_simobs)
data = read_pumas(simdf)
for i = 1:15
  sol = solve(model732, data[i], NamedTuple())

  @test sol(3.0)[1] ≈ 43.0
end
