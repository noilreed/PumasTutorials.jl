using Test
using Pumas

@testset "likelihood tests from NLME.jl" begin
  data = read_pumas(example_data("sim_data_model1"))

  mdsl1 = @model begin
    @param begin
      θ ∈ VectorDomain(1, init=[0.5])
      Ω ∈ PDiagDomain(init=[0.04])
      Σ ∈ RealDomain(lower=0.0, upper=1.0, init=0.1)
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

  mdsl1_nofitrandom = @model begin
    @param begin
      θ ∈ VectorDomain(1, init=[0.5])
      Ω ∈ PDiagDomain(init=[0.04])
      Σ ∈ RealDomain(lower=0.0, upper=1.0, init=0.1)
    end

    @random begin
      η ~ Beta(0.1,0.9)
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

  for (ηstar, dt) in zip([-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361], data)
    @test (sqrt(param.Ω.diag[1])*Pumas._orth_empirical_bayes(mdsl1, dt, param, Pumas.LaplaceI()))[1] ≈ ηstar rtol=1e-3
  end

  @test conditional_nll(mdsl1, data[1], param, (η=[0.0],)) ≈ 5.337249432459216 rtol=1e-6
  @test Pumas.penalized_conditional_nll(mdsl1, data[1], param, (η=[0.0],)) ≈ 5.337249432459216 rtol=1e-6
  @test loglikelihood(mdsl1, data, param, Pumas.FO())        ≈ -46.61622679322253 rtol=1e-6
  @test_throws ArgumentError loglikelihood(mdsl1, data, param, Pumas.FOCE())
  @test loglikelihood(mdsl1, data, param, Pumas.FOCEI())     ≈ -46.58423998502921 rtol=1e-6
  @test loglikelihood(mdsl1, data, param, Pumas.LaplaceI())  ≈ -46.78394281224711 rtol=1e-6
  @test loglikelihood(mdsl1, data, param, Pumas.LLQuad()) ≈ -46.84122752833662  rtol=1e-6 #regression test

  # Cannot fit a model with Beta random effects
  @test_throws ArgumentError fit(mdsl1_nofitrandom, data, param, Pumas.FOCEI())

  ft = fit(mdsl1, data, param, Pumas.FOCEI(); constantcoef=(Ω=Diagonal([0.04]), Σ=0.1),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
  ft2 = fit(mdsl1, data, param, Pumas.FO(); constantcoef=(Ω=Diagonal([0.04]), Σ=0.1),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  # These two tests are supposed to test that the show method works
  # but also that the alignment of the column names work.
  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), [ft, ft]) ==
"""
Vector{<:FittedPumasModel} with 2 entries

Parameter statistics
---------------------
        Mean     Std
---------------------
θ₁      0.36476  0.0
Ω₁,₁    0.04     0.0
Σ       0.1      0.0
---------------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), [ft, ft2]) ==
"""
Vector{<:FittedPumasModel} with 2 entries

Parameter statistics
----------------------------------
        Mean           Std
----------------------------------
θ₁      0.36488        0.00017191
Ω₁,₁    0.04           0.0
Σ       0.1            0.0
----------------------------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), ft) ==
"""
FittedPumasModel

Successful minimization:                      true

Likelihood approximation:              Pumas.FOCEI
Log-likelihood value:                   -45.623789
Number of subjects:                             10
Number of parameters:         Fixed      Optimized
                                  0              3
Observation records:         Active        Missing
    dv:                          20              0
    Total:                       20              0

------------------
         Estimate
------------------
θ₁        0.36476
Ω₁,₁      0.04
Σ         0.1
------------------
"""

  @test sprint((io, t) -> show(io, MIME"text/plain"(), t), infer(ft)) ==
"""
Asymptotic inference results

Successful minimization:                      true

Likelihood approximation:              Pumas.FOCEI
Log-likelihood value:                   -45.623789
Number of subjects:                             10
Number of parameters:         Fixed      Optimized
                                  0              3
Observation records:         Active        Missing
    dv:                          20              0
    Total:                       20              0

-------------------------------------------------------------
        Estimate          SE                  95.0% C.I.
-------------------------------------------------------------
θ₁       0.36476        0.078498        [0.21091; 0.51861]
Ω₁,₁     0.04           NaN             [ NaN   ;  NaN     ]
Σ        0.1            NaN             [ NaN   ;  NaN     ]
-------------------------------------------------------------
"""
  @test cond(infer(ft)) == 1.0 # should be 1.0... basically testing that it works here


  @testset "unidentified parameter make fit throw" begin
    # Since the data doesn't have a rate column, the rate parameter
    # θᵣ will be completely ignored. This should trigger an exception.
    unidentified_model = @model begin
      @param begin
        θ ∈ VectorDomain(1, init=[0.5])
        θᵣ∈ VectorDomain(1, init=[0.5])
        Ω ∈ PDiagDomain(init=[0.04])
        Σ ∈ RealDomain(init=0.1)
      end

      @random begin
        η ~ MvNormal(Ω)
      end

      @pre begin
        CL = θ[1] * exp(η[1])
        Vc = 1.0
        rate = (Central=θᵣ,)
      end

      @vars begin
        conc = Central / Vc
      end

      @dynamics Central1

      @derived begin
        dv ~ @. Normal(conc,conc*sqrt(Σ)+eps())
      end
    end

    @test_throws ErrorException("gradient of θᵣ is exactly zero. This indicates that θᵣ isn't identified.") fit(
        unidentified_model,
        data,
        init_param(unidentified_model),
        Pumas.FO(),
        constantcoef=(θ=[0.5],))
  end

  # test for #1043
  @testset "Symbol cmt names" begin
    dr1 = DosageRegimen(100, cmt=:Central)
    subj1 = Subject(id=1, events=dr1)
    subj2 = Subject(id=2, events=dr1)
    subj = [subj1, subj2]
    sim1 = simobs(mdsl1, subj, init_param(mdsl1), obstimes=0:1:24)
    pop = DataFrame(subj)
    pop[!, :dv] .= missing
    # it's important that this runs (before, a symbol in the cmt column would cause an error)
    dd = read_pumas(pop)
  end

  @testset "integer obstimes" begin
    # from https://discourse.pumas.ai/t/help-with-fitting/50/54
    # the test is simply that dff runs in the end of the testset
    ev1 = DosageRegimen(400, time=0, cmt=1)
    ev2 = DosageRegimen(800, time=0, cmt=1, rate=30.769)
    com = DosageRegimen(ev1,ev2)
    sub = Subject(id=1, events=com)
    # The issue was that these obstimes are integers and that caused problems in the DataFrame constructor
    sim = simobs(mdsl1, sub, param, obstimes=[2, 5, 10, 15, 20, 25, 30, 33, 35, 37, 40, 45, 50, 60, 70, 90, 110, 120, 150])
    # Test that dff is constructed by converting obstimes to floats internally
    ddf = DataFrame(sim)
    @test eltype(ddf.time) == Float64
  end
end# testset
