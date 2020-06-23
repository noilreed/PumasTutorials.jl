using Test
using Pumas, LinearAlgebra

data = read_pumas(example_data("sim_data_model1"))
#-----------------------------------------------------------------------# Test 1
@everywhere mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1, init=[0.5])
        Ω ∈ PDiagDomain(init=[0.04])
        Σ ∈ RealDomain(lower=0.01, upper=10.0, init=0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ[1] * exp(η[1])
        Vc = 1.0
    end

    @vars begin
        conc = Central / Vc
    end

    @dynamics Central1

    @derived begin
        dv ~ @. Normal(conc,conc*sqrt(Σ)+eps())
    end
end

param = init_param(mdsl1)
@testset "ensemblealg = $p" for p in (EnsembleSerial(), EnsembleThreads(), EnsembleDistributed())
    @test deviance(mdsl1, data, param, Pumas.FO(), ensemblealg=p) ≈ 56.474912258255571 rtol=1e-6
end
