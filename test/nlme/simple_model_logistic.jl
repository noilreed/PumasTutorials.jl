using Pumas, Test, StatsFuns

@testset "Logistic regression example" begin

    data = read_pumas(joinpath(dirname(pathof(Pumas)), "..", "examples", "pain_remed.csv"),
                          cvs = [:arm, :dose, :conc, :painord];
                          time=:time, event_data=false)

    mdsl = @model begin
        @param begin
            θ₁ ∈ RealDomain(init=0.001)
            θ₂ ∈ RealDomain(init=0.0001)
            Ω  ∈ PSDDomain(1)
        end

        @random begin
            η ~ MvNormal(Ω)
        end

        @covariates arm dose

        @pre begin
            rx = dose > 0 ? 1 : 0
            LOGIT = θ₁ + θ₂*rx + η[1]
        end

        @derived begin
            dv ~ @. Bernoulli(logistic(LOGIT))
        end

    end

    param = (θ₁=0.01, θ₂=0.001, Ω=fill(1.0, 1, 1))

    @testset "Conversion of simulation output to DataFrame when dv is scalar" begin
        sim = simobs(mdsl, data, param)
        @test DataFrame(sim) isa DataFrame
    end

    @testset "testing with $approx approximation" for
        approx in (Pumas.FO(), Pumas.FOCE(), Pumas.FOCEI(), Pumas.LaplaceI())

        if approx ∈ (Pumas.FOCE(), Pumas.LaplaceI())
            _param = coef(fit(mdsl, data, param, approx))

            # Test values computed with MixedModels.jl
            @test _param.θ₁                ≈ -1.3085393956990727 rtol=1e-3
            @test _param.θ₂                ≈  1.7389379466901713 rtol=1e-3
            @test _param.Ω.chol.factors[1] ≈  1.5376005165566606 rtol=1e-3
        else
            @test_throws ArgumentError Pumas.marginal_nll(mdsl, data, param, approx)
        end
    end
end
