using Pumas
using CSV, DataFrames
using LinearAlgebra
using Plots

tmdd = @model begin
    @param begin
        θ ∈ VectorDomain(7, lower=zeros(7), init=ones(7))
        Ω ∈ PSDDomain(1)
        σ_prop ∈ RealDomain(lower=0.0, init=0.01)
    end

    @random begin
    η ~ MvNormal(Ω)
    end

    @pre begin
        VC   = θ[1] * exp(η[1])
        KEL  = θ[2] * exp(η[2])
        KEP  = θ[3] * exp(η[3])
        KOUT = θ[4] * exp(η[4])
        KOFF = θ[5] * exp(η[5])
        KON  = θ[6] * exp(η[6])
        R0   = θ[7] * exp(η[7])
        #KIN  = θ[7]
    end

    @init begin
        #R = KIN/KOUT
        R = R0
    end

    @vars begin
        IPRED_L = L/VC                         # Free Drug Cocncentration (nM)
        IPRED_R = R
        IPRED_P = P
    end

    @dynamics begin
    #KIN = KOUT*R0
    L' = -KEL*L - KON*L*R + KOFF*P*VC         # Free Drug Amount
    #R' = KIN - KOUT*R - KON*L*R/VC + KOFF*P  # Free Receptor Concentration
    R' = KOUT*R0 - KOUT*R - KON*L*R/VC + KOFF*P  # Free Receptor Concentration
    P' = KON*L*R/VC - KOFF*P - KEP*P         # Complex Cencentration
    end

    @derived begin
        #DVL ~ @.Normal(LCONC, sqrt(LCONC^2 * σ_prop))
        #DV_R ~ @.Normal(R, sqrt(R^2 * σ_prop))
        #DV_P ~ @.Normal(P, sqrt(P^2 * σ_prop))
    end
end

fixeffs = [
      0.04,      # VC, Central volume (L)
      0.024,     # KEL, Elimination rate constant (1/day)
      0.201,     # KEP, Elimination of complex (1/day)
      0.823,     # KOUT, Elimination of receptor (1/day)
      0.9,       # KOFF, Dissociation rate (1/day)
      0.592,     # KON, Binding rate (1/(nM*day))
      2.688      # R0, Initial receptor (nM)
      #2.212     # KIN
      ]

parameter = (θ = fixeffs,
             Ω = Diagonal([0.0,0.0,0.0,0.0,0.0,0.0,0.0]),
             σ_prop = 0.01)

dosemg = 1
dose = dosemg*1e-3/150000*1e9     # nmol
tlast = 28
e1 = DosageRegimen(dose, duration = 0.001)
pop1 = Population(map(i -> Subject(id=i,evs=e1),1:1))
#pop = Population(vcat(pop1,pop2,pop3))

#using Random
Random.seed!(123456)
sim = simobs(tmdd, pop1, parameter, obstimes=0:0.01:tlast)

plot(sim, layout=(1,3))


parameter = (θ = fixeffs,
             Ω = Diagonal([0.04,0.04,0.04,0.04,0.04,0.04,0.04]),
             σ_prop = 0.01)

pop2 = Population(map(i -> Subject(id=i,evs=e1),1:20))
Random.seed!(123456)
sim2 = simobs(tmdd, pop2, parameter, obstimes=0:0.01:tlast)
plot(sim2, layout=(1,3))

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
