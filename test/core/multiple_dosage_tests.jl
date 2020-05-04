using Pumas, Test, Random

# Load data
cvs = [:ka, :cl, :v]
dvs = [:dv]
data = read_pumas(example_data("oral1_1cpt_KAVCL_MD_data"),
                      cvs = cvs, dvs = dvs)

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

sim_diffeq = begin
    Random.seed!(1)
    s = simobs(m_diffeq,subject1,param,randeffs)[:dv]
end
sim_analytic = begin
    Random.seed!(1)
    s = simobs(m_analytic,subject1,param,randeffs)[:dv]
end
@test sim_diffeq ≈ sim_analytic rtol=1e-3

pop = Population(map(i -> Subject(id=i, time=i:20, cvs=subject1.covariates),1:3))
s = simobs(m_diffeq,pop,param,fill(randeffs, length(pop));ensemblealg = EnsembleSerial())
@test map(x->x.times, s) == map(x->x.time, pop)
