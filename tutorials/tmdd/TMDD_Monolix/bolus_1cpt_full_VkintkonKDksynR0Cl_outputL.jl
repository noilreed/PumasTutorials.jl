using Pumas
using CSV, DataFrames
using LinearAlgebra
using Plots
using Query

tmdd = @model begin
    @param begin
        tvCl ∈ RealDomain(lower=0.0001, upper=10.0, init=1.0)
        tvV ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        tvkint ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        tvkon ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        tvKD ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        tvksyn ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        tvR0 ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        Ω ∈ PDiagDomain(1)
        σ_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end
    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        Cl = tvCl * exp(η[1])
        V = tvV
        kint  = tvkint
        kon =  tvkon
        KD = tvKD
        ksyn = tvksyn
        R0 = tvR0
        bioav = 1/V
    end

    @init begin
        R = R0
    end

    @vars begin
        kel = Cl/V
        koff = KD*kon
        kdeg = ksyn/R0
        IPRED_L = L                         # Free Drug Cocncentration (nM)
        IPRED_R = R
        IPRED_P = P
        Ltot = L+P
        Rtot = R+P
        #TO = R/Rtot
        TO = (1 - R/Rtot)*100
        RR = R/R0
    end

    @dynamics begin
        L'    = -kel*L - kon*L*R + koff*P
        R'    = ksyn - kdeg*R - kon*L*R + koff*P
        P'    = kon*L*R - koff*P - kint*P
    end

    @derived begin
        dv ~ @.Normal(IPRED_L, sqrt(IPRED_L^2 * σ_prop))
    end
end

parameter = (
        tvCl   = 0.005,
        tvV    = 0.05,  # volume of central compartment (L/kg)
        tvkint  = 0.2, # Elimination of complex (1/day)
        tvkon  = 0.5, # Binding rate (1/(nMday))
        tvKD   = 1.0,
        tvksyn = 2.0,
        tvR0   = 3.0,  # inital concentration of receptor  in central compartment (nM),
        Ω = Diagonal([0.0]),
        σ_prop = 0.025
        )

#parameter = (θ = fixeffs,
#             Ω = Diagonal([0.0,0.0,0.0,0.0,0.0,0.0]),
#             σ_prop = 0.025)

dose = 1
#dose = dosemg*1e-3/150000*1e9     # nmol
tlast = 100
e1 = DosageRegimen(dose)
pop1 = Population(map(i -> Subject(id=i,evs=e1),1:1))
#pop = Population(vcat(pop1,pop2,pop3))

using Random
Random.seed!(123456)
sim = simobs(tmdd, pop1, parameter, obstimes=0:1:tlast)
plot(sim, obsnames=[:IPRED_L])
plot(sim, obsnames=[:IPRED_L, :IPRED_R, :IPRED_P], layout=(1,3))
plot(sim, obsnames=[:IPRED_L, :IPRED_R, :TO, :RR, :Ltot, :Rtot], layout=(2,3))

parameter = (
        tvCl   = 0.005,
        tvV    = 0.05,  # volume of central compartment (L/kg)
        tvkint  = 0.2, # Elimination of complex (1/day)
        tvkon  = 0.5, # Binding rate (1/(nMday))
        tvKD   = 1.0,
        tvksyn = 2.0,
        tvR0   = 3.0,  # inital concentration of receptor  in central compartment (nM),
        Ω = Diagonal([0.1]),
        σ_prop = 0.025
        )


pop2 = Population(map(i -> Subject(id=i,evs=e1),1:20))
Random.seed!(12345)
sim2 = simobs(tmdd, pop2, parameter, obstimes=0:1:tlast)
plot(sim2, obsnames=[:IPRED_L, :IPRED_R, :IPRED_P], layout=(1,3))


# 3 doses
dose1 = 1    # nmol
dose2 = 100    # nmol
dose3 = 10000   # nmol

#dose1 = dosemg1*1e-3/150000*1e9     # for change mg to nmol

e1 = DosageRegimen(dose1)
e2 = DosageRegimen(dose2)
e3 = DosageRegimen(dose3)

pop1 = Population(map(i -> Subject(id=i,evs=e1),1:1))
pop2 = Population(map(i -> Subject(id=i,evs=e2),2:2))
pop3 = Population(map(i -> Subject(id=i,evs=e3),3:3))
pop = Population(vcat(pop1,pop2,pop3))

Random.seed!(12345)
sim3 = simobs(tmdd, pop, parameter, obstimes=0:1:tlast)
plot(sim3, obsnames=[:IPRED_L, :IPRED_R, :IPRED_P], layout=(1,3))
plot(sim3, obsnames=[:IPRED_L])
yaxis!(:log10)
