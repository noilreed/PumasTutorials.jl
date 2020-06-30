using Pumas, Quadrature, Test, CSV, Random

# Read the data
df = DataFrame(CSV.File(example_data("data1")))
df[!, :cmt] .= 1
data = read_pumas(df, cvs = [:sex,:wt,:etn])

# Definition using diffeqs
m_diffeq = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=[1.0,20.0,200.0,0.1], upper=[5.0,100.0,400.0,2.0])
    end

    @covariates sex wt etn

    @pre begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex)
        Vc = θ[3]
    end

    @vars begin
        cp = Central / Vc
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @derived begin
        cp = @. Central / Vc
        nca := @nca cp
        auc =  NCA.auc(nca)
        thalf =  NCA.thalf(nca)
        cmax = NCA.cmax(nca)
    end
end

param = (θ = [2.268,74.17,468.6,0.5876],)
subject1 = data[1]
obs = simobs(m_diffeq,subject1,param)

function g(obs)
    obs[:auc] < 300
end
param_dists = (θ = Uniform.(m_diffeq.param.params.θ.lower,m_diffeq.param.params.θ.upper),)
@time expectation(g,KoopmanExpectation(HCubatureJL()),m_diffeq,subject1,param_dists)
@time expectation(g,KoopmanExpectation(HCubatureJL()),m_diffeq,subject1,param_dists;
                    ireltol=1e-6,iabstol=1e-6,imaxiters=1_0)
@time expectation(g,KoopmanExpectation(HCubatureJL()),m_diffeq,subject1,param_dists;
                    ireltol=1e-6,iabstol=1e-6,imaxiters=100)
@time expectation(g,KoopmanExpectation(HCubatureJL()),m_diffeq,subject1,param_dists;
                    ireltol=1e-6,iabstol=1e-6,imaxiters=1000)

@time expectation(g,MonteCarloExpectation(),m_diffeq,subject1,param_dists,imaxiters=10)
@time expectation(g,MonteCarloExpectation(),m_diffeq,subject1,param_dists,imaxiters=100)
@time expectation(g,MonteCarloExpectation(),m_diffeq,subject1,param_dists,imaxiters=1000)
