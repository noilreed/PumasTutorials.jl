using Test
using Pumas

# Gut dosing model
m_diffeq = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
    end

    @random begin
        η ~ MvNormal(Matrix{Float64}(I, 3, 3))
    end

    @covariates isPM Wt

    @pre begin
        TVCL = isPM == "yes" ? θ[1] : θ[4]
        CL = θ[1]*(Wt/70)^0.75*exp(η[1])
        Vc = θ[2]*(Wt/70)^0.75*exp(η[2])
        Ka = θ[3]*exp(η[3])
    end

    @vars begin
        cp = Central/Vc
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @derived begin
        conc = @. Central / Vc
        dv ~ @. Normal(conc, 0.2)
    end
end

param = (
    θ = [0.4, 20, 1.1, 2],
    )

randeffs = (η=randn(3),)
subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2), cvs = (isPM="no", Wt=70))

# Make sure simobs works without time, defaults to 1 day, obs at each hour
obs = simobs(m_diffeq, subject, param, randeffs)
@test obs.times == 0.0:1.0:84.0
@test DataFrame(obs, include_events=false).time == 0.0:1.0:84.0
@test all(DataFrame(obs)[!,:isPM] .== "no")

#=
using Plots
plot(obs)
=#

pop = Population([Subject(id = id,
            evs = DosageRegimen([10rand(), 20rand()],
            ii = 24, addl = 2, ss = 1:2, time = [0, 12],
            cmt = 2),cvs = cvs=(isPM="no", Wt=70)) for id in 1:10])
vrandeffs = fill(randeffs, length(pop))
pop_obs1 = simobs(m_diffeq, pop[1], param, randeffs, ensemblealg = EnsembleSerial())
pop_obs1 = simobs(m_diffeq, pop, param, vrandeffs, ensemblealg = EnsembleSerial())
_data = DataFrame(pop_obs1)

pop = Population([Subject(id = id,
            evs = DosageRegimen(10,
            ii = 24, addl = 2, ss = 1:2, time = [0, 12],
            cmt = 2),cvs = cvs=(isPM="no", Wt=70)) for id in 1:10])
vrandeffs = fill(randeffs, length(pop))
pop_obs1 = simobs(m_diffeq, pop[1], param, randeffs, ensemblealg = EnsembleSerial())
pop_obs1 = simobs(m_diffeq, pop, param, vrandeffs, ensemblealg = EnsembleSerial())
@test all(x->x[:conc] == pop_obs1[1][:conc],pop_obs1)
pop_obs1 = simobs(m_diffeq, pop, param, ensemblealg = EnsembleSerial())
@test all(x->x[:conc] !== pop_obs1[1][:conc],pop_obs1[2:end])

#=
plot(pop_obs)
=#

data = read_pumas(_data,cvs = [:isPM,:Wt],dvs = [:conc,:dv])

obs2 = simobs(m_diffeq, data[1], param, randeffs)
@test_broken _data2 = DataFrame(obs2)

pop_obs2 = simobs(m_diffeq, data, param, fill(randeffs, length(data)))
@test_broken _data2 = DataFrame(pop_obs2)
#@test all(_data.conc .== _data2.conc)
#@test !all(_data.dv .== _data2.dv)
