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
        #TVF1 ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVKA ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVKSS ∈ RealDomain(lower=0.0001, upper=1000.0, init=1.0)
        TVKINT ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVKSYN ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
        TVKDEG ∈ RealDomain(lower=0.0001, upper=100.0, init=1.0)
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
        #F1 = TVF1
        KA = TVKA
        KSS = TVKSS
        KINT = TVKINT * exp(η[4])
        KSYN = TVKSYN * exp(η[5])
        KDEG = TVKDEG * exp(η[6])
        #bioav = F1
    end

    @init begin
        #R = KIN/KOUT
        RTOT = KSYN/KDEG
    end

    @vars begin
        KEL = CL/VC
        KPT = Q/VC
        KTP = Q/VP
        LCONC = 0.5*(LTOT/VC-RTOT-KSS)+0.5*sqrt((LTOT/VC-RTOT-KSS)^2+4*KSS*LTOT/VC)
        IPRED_L = LCONC                         # Free Drug Cocncentration (nM)
        IPRED_RC = LTOT/VC - LCONC
        IPRED_R = RTOT - IPRED_RC
    end

    @dynamics begin
    DEPOT' = -KA*DEPOT # Free Drug depot
    LTOT'  =  KA*DEPOT -(KEL+KPT)*LCONC*VC + KTP*LT - (RTOT*KINT*LCONC*VC)/(KSS+LCONC)
    RTOT'  =  KSYN - KDEG*RTOT - (KINT-KDEG)*(RTOT*LCONC)/(KSS+LCONC)
    LT'    = -KTP*LT + KPT*LCONC*VC

    end

    @derived begin
        DV_L ~ @.Normal(LCONC, sqrt(LCONC^2 * σ_prop))
        #DV_R ~ @.Normal(R, sqrt(R^2 * σ_prop))
        #DV_P ~ @.Normal(P, sqrt(P^2 * σ_prop))
    end
end

parameter = (
        TVCL = 0.15,
        TVVC = 3.0,
        TVQ = 0.45,
        TVVP = 3.0,
        #TVF1 = 0.6,
        TVKA = 1.0,
        TVKSS = 1.0,
        TVKINT = 0.04,
        TVKSYN = 1.0,
        TVKDEG = 0.25,
        #Ω = Diagonal([0.1,0.1,0.1,0.1,0.1]),
        Ω = Diagonal([0.0,0.0,0.0,0.0,0.0,0.0]),
        σ_prop = 0.025
        )

#parameter = (θ = fixeffs,
#             Ω = Diagonal([0.0,0.0,0.0,0.0,0.0,0.0]),
#             σ_prop = 0.025)

dosemg = 1
dose = dosemg*1e-3/150000*1e9     # nmol
tlast = 28
e1 = DosageRegimen(dose, duration = 0.001)
pop1 = Population(map(i -> Subject(id=i,evs=e1),1:1))
#pop = Population(vcat(pop1,pop2,pop3))

using Random
Random.seed!(123456)
@time sim = simobs(tmdd, pop1, parameter, obstimes=0:0.01:tlast, alg = Tsit5())
@time sim = simobs(tmdd, pop1, parameter, obstimes=0:0.01:tlast, alg = Rosenbrock23())
plot(sim, obsnames=[:LCONC, :IPRED_L, :IPRED_R, :IPRED_RC], layout=(2,2))

parameter = (
         TVCL = 0.15,
         TVVC = 3.0,
         TVQ = 0.45,
         TVVP = 3.0,
         TVF1 = 0.6,
         TVKA = 1.0,
         TVKSS = 1.0,
         TVKINT = 0.04,
         TVKSYN = 1.0,
         TVKDEG = 0.25,
         Ω = Diagonal([0.04,0.04,0.04,0.04,0.04,0.04]),
         σ_prop = 0.025
         )

pop2 = Population(map(i -> Subject(id=i,evs=e1),1:20))
Random.seed!(12345)
sim2 = simobs(tmdd, pop2, parameter, obstimes=0:0.01:tlast)
plot(sim2, obsnames=[:LCONC, :IPRED_L, :IPRED_R, :IPRED_RC], layout=(2,2))


cd(raw"C:\Users\user\Documents\TMDD_Pumas\TMDD_Gibiansky\Data\DerivedData")
pwd()

pkdata = CSV.read("./TMDD_Gibiansky/Data/DerivedData/SimulatedNonmemDataConc.csv")
pkdata[:dv] = pkdata[:DVOR]
pkdata = pkdata[setdiff(names(pkdata), [:C])]
pkdata2 = DataFrame(pkdata |> @filter(_.ID < 100))
pkdata3 = pkdata |> @filter(_.ID < 100)
simdf = filter(row -> row[:EVID] .== 1, pkdata2)
data = read_pumas(pkdata2, dvs = [:dv], id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE)
simdata = read_pumas(simdf, dvs = [:dv], id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE)
@time sim = simobs(tmdd, simdata, parameter, obstimes=0:0.01:tlast, alg = Rosenbrock23())
plot(sim, obsnames=[:LCONC, :IPRED_L, :IPRED_R, :IPRED_RC], layout=(2,2))


result=nothing
@time result = fit(tmdd, data, parameter, Pumas.FOCEI())

simdf2 = DataFrame(sim2)

cd(raw"C:\Users\WSPARK\Documents\TMDD_Pumas")
CSV.write("simdf2.csv", simdf2)


plot(sim_tmdd_full, obsnames=[:LCONC], title = "Full model")
#yaxis!(:log10)
xlabel!("Time (day)")
ylabel!("Free drug concentration (nM)")

plot(sim_tmdd_full, title = "Full model")
plot(sim_tmdd_full, layout=(1,3))


plot(sim_tmdd_full, title = "Full model", obsnames=[:LCONC])
plot(sim_tmdd_full, title = "Full model", obsnames=[:IPRED_P])
plot(sim_tmdd_full, title = "Full model", obsnames=[:IPRED_R])
