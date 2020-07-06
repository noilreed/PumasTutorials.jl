using Pumas, Test, Random

# Load data
covariates = [:ka, :cl, :v]
observations = [:dv]
data = read_pumas(example_data("oral1_1cpt_KAVCL_MD_data"), covariates = covariates, observations = observations)

m_diffeq = @model begin

    @covariates ka cl v

    @pre begin
        Ka = ka
        CL = cl
        Vc = v
    end

    @vars begin
        cp = CL/Vc
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - cp*Central
    end

    # we approximate the error by computing the conditional_nll
    @derived begin
        conc = @. Central / Vc
        dv ~ @. Normal(conc, 1e-100)
    end
end

m_analytic = @model begin

    @covariates ka cl v

    @pre begin
        Ka = ka
        CL = cl
        Vc  = v
    end
    @dynamics Depots1Central1

    # we approximate the error by computing the conditional_nll
    @derived begin
        conc = @. Central / Vc
        dv ~ @. Normal(conc, 1e-100)
    end
end

subject1 = data[1]
param = NamedTuple()
randeffs = NamedTuple()

sol_diffeq   = solve(m_diffeq,subject1,param,randeffs)
sol_analytic = solve(m_analytic,subject1,param,randeffs)

@test sol_diffeq(95.99) ≈ sol_analytic(95.99) rtol=1e-4
@test sol_diffeq(217.0) ≈ sol_analytic(217.0) rtol=1e-3 # TODO: why is this so large?

sol_diffeq   = simobs(m_diffeq,subject1,repeat([param],10))
sol_analytic = simobs(m_analytic,subject1,repeat([param],10))
@test length(sol_diffeq) == 10
@test length(sol_analytic) == 10
@test eltype(sol_diffeq) <: Pumas.SimulatedObservations
@test eltype(sol_analytic) <: Pumas.SimulatedObservations

sol_diffeq   = simobs(m_diffeq,repeat([subject1],10),param)
sol_analytic = simobs(m_analytic,repeat([subject1],10),param)
@test length(sol_diffeq) == 10
@test length(sol_analytic) == 10
@test eltype(sol_diffeq) <: Pumas.SimulatedObservations
@test eltype(sol_analytic) <: Pumas.SimulatedObservations

sol_diffeq   = simobs(m_diffeq,repeat([subject1],10),repeat([param],7))
sol_analytic = simobs(m_analytic,repeat([subject1],10),repeat([param],7))
@test length(sol_diffeq) == 7
@test length(sol_analytic) == 7
@test length(sol_diffeq[1]) == 10
@test length(sol_analytic[1]) == 10
@test eltype(sol_diffeq) <: Vector{<:Pumas.SimulatedObservations}
@test eltype(sol_analytic) <: Vector{<:Pumas.SimulatedObservations}

sol_diffeq   = simobs(m_diffeq,repeat([subject1],10),repeat([param],7),repeat([randeffs],70))
sol_analytic = simobs(m_analytic,repeat([subject1],10),repeat([param],7),repeat([randeffs],70))
@test length(sol_diffeq) == 7
@test length(sol_analytic) == 7
@test length(sol_diffeq[1]) == 10
@test length(sol_analytic[1]) == 10
@test eltype(sol_diffeq) <: Vector{<:Pumas.SimulatedObservations}
@test eltype(sol_analytic) <: Vector{<:Pumas.SimulatedObservations}

sim_diffeq = begin
    Random.seed!(1)
    s = simobs(m_diffeq,subject1,param,randeffs)[:dv]
end
sim_analytic = begin
    Random.seed!(1)
    s = simobs(m_analytic,subject1,param,randeffs)[:dv]
end
@test sim_diffeq ≈ sim_analytic rtol=1e-3

pop = Population(map(i -> Subject(id=i, time=i:20, covariates=subject1.covariates.u, covariates_time=0.0), 1:3))
s = simobs(m_diffeq, pop, param, fill(randeffs, length(pop)); ensemblealg = EnsembleSerial())
@test map(x -> x.time, s) == map(x -> x.time, pop)
