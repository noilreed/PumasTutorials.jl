using Pumas

data = read_pumas(example_data("data1"), cvs = [:sex,:wt,:etn])

tmp = []
push!(tmp, Subject(obs=data[1].observations,
                   time=data[1].time,
                   cvs=(sex=1, wt=[51.6 for i in 1:length(data[1].time)], etn=1),
                   cvstime=(sex=[0.0], wt=data[1].time, etn=[0.0]),
                   evs=data[1].events))
push!(tmp, data[2])
new_data = identity.(tmp)

mdsl = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        Σ ∈ RealDomain(lower=0.0, init=1.0)
        a ∈ RealDomain(lower=0.0, upper=1.0, init=0.2)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn

    @pre begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * (θ[4]^sex) * exp(η[1])
        Vc = θ[3] * exp(η[2])
    end

    @vars begin
      conc = Central / Vc
      conc2 = Central^2
    end

    @dynamics begin
        Depot'   := -Ka*Depot # test for `:=` handling
        Central' =  Ka*Depot - CL*conc
    end

    @derived begin
      dv ~ @. Normal(conc, conc*Σ)
      T_max = maximum(t)
    end

    @observed begin
      obs_cmax = maximum(dv)
    end
end

obs = simobs(mdsl, new_data, init_param(mdsl), ensemblealg = EnsembleSerial())
obs = simobs(mdsl, new_data, init_param(mdsl), ensemblealg = EnsembleThreads())
obs = simobs(mdsl, new_data, init_param(mdsl), ensemblealg = EnsembleDistributed())
obs = simobs(mdsl, new_data, init_param(mdsl), ensemblealg = EnsembleSplitThreads())
