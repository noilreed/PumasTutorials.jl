using Test
using Pumas

@testset "Exponentially distributed error model" begin

data = read_pumas(example_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
mdsl = @model begin
    @param begin
        θ  ∈ RealDomain(init=0.5)
        Ω  ∈ PSDDomain(Matrix{Float64}(fill(0.04, 1, 1)))
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ * exp(η[1])
        Vc = 1.0
    end

    @vars begin
        # Currently, Exponential is a bit picky about zeros in the parameters
        conc = Central / Vc + 1e-10
    end

    @dynamics Central1

    @derived begin
        dv ~ @. Exponential(conc)
    end
end


param = init_param(mdsl)

# Not supported
@test_throws ArgumentError deviance(mdsl, data, param, Pumas.FO())
@test_throws ArgumentError deviance(mdsl, data, param, Pumas.FOCEI())

@test deviance(mdsl, data, param, Pumas.FOCE())     ≈ 88.9136079338946 rtol=1e-6
@test deviance(mdsl, data, param, Pumas.LaplaceI()) ≈ 88.9571564205892 rtol=1e-6

@test deviance(fit(mdsl, data, param, Pumas.FOCE(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))) ≈ 88.61049219271284 rtol=1e-6
@test deviance(fit(mdsl, data, param, Pumas.LaplaceI(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))) ≈ 88.61017570108888 rtol=1e-6

end
