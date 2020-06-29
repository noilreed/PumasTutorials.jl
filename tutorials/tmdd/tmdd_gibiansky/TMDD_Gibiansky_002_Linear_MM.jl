using Pumas
using CSV, DataFrames
using LinearAlgebra
using Plots
using Query

tmdd = @model begin
    @param begin
        TVCL ∈ RealDomain(lower=0.0001, upper=10.0, init=1.0)
        TVVC ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVQ ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVVP ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVF1 ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVKA ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVVMAX ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVKSS ∈ RealDomain(lower=0.0001, upper=1000.0, init=1.0)
        Ω ∈ PDiagDomain(6)
        σ_prop ∈ RealDomain(lower=0.0001, init=0.3)
    end
    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = TVCL * exp(η[1])
        VC = TVVC * exp(η[2])
        Q  = TVQ  * exp(η[3])
        VP = TVVP
        F1 = TVF1
        KA = TVKA
        VMAX = TVVMAX
        KSS = TVKSS * exp(η[4])
        bioav = F1
    end

    @init begin
    end

    @vars begin
        KEL = CL/VC
        KPT = Q/VC
        KTP = Q/VP
        IPRED_L = L/VC
    end

    @dynamics begin
    DEPOT' = -KA*DEPOT # Free Drug depot
    L'     =  KA*DEPOT -(KEL+KPT)*L + KTP*LT - (VMAX*L)/(KSS + L/VC)
    LT'    = -KTP*LT + KPT*L
    end

    @derived begin
        DV_L ~ @.Normal(IPRED_L, sqrt(IPRED_L^2 * σ_prop))
    end
end

parameter = (
        TVCL = 0.15,
        TVVC = 3.0,
        TVQ = 0.45,
        TVVP = 3.0,
        TVF1 = 0.6,
        TVKA = 1.0,
        TVVMAX = 3.29,
        TVKSS = 45.1,
        Ω = Diagonal([0.0,0.0,0.0,0.0,0.0,0.0]),
        σ_prop = 0.025
        )

dosemg = 1
dose = dosemg*1e-3/150000*1e9     # nmol
tlast = 28
e1 = DosageRegimen(dose, duration = 0.001)
pop1 = Population(map(i -> Subject(id=i,evs=e1),1:1))
#pop = Population(vcat(pop1,pop2,pop3))

using Random
Random.seed!(123456)
sim = simobs(tmdd, pop1, parameter, obstimes=0:0.01:tlast)
plot(sim, obsnames=[:IPRED_L, :DV_L], layout=(2,1))

parameter = (
        TVCL = 0.15,
        TVVC = 3.0,
        TVQ = 0.45,
        TVVP = 3.0,
        TVF1 = 0.6,
        TVKA = 1.0,
        TVVMAX = 3.29,
        TVKSS = 45.1,
        Ω = Diagonal([0.04,0.04,0.04,0.04]),
        σ_prop = 0.025
         )

pop2 = Population(map(i -> Subject(id=i,evs=e1),1:20))
Random.seed!(12345)
sim2 = simobs(tmdd, pop2, parameter, obstimes=0:0.01:tlast)
plot(sim2, obsnames=[:IPRED_L, :DV_L], layout=(2,1))
