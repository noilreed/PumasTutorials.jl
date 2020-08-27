using Pumas, Test, Random, StaticArrays

@testset "Callbacks" begin
  # Load data
  covariates = [:ka, :cl, :v]
  observations = [:dv]
  data = read_pumas(example_data("oral1_1cpt_KAVCL_MD_data"),
    covariates = covariates, observations = observations)

  m_diffeq = @model begin

    @covariates ka cl v

    @cache begin
      onoff = 1.0
    end

    @pre begin
      Ka = onoff*ka
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


  subject1 = data[1]
  param = NamedTuple()
  randeffs = NamedTuple()

  @testset "ContinuousCallback" begin
    condition = function (u, t, integrator)
      u.Central - 5e4
    end

    called = false
    tsave = 0.0
    affect! = function (integrator)
      if !called
        called = true
        tsave = integrator.t
      end
      integrator.u += SA[2e4,5e4]
    end

    cb = ContinuousCallback(condition, nothing, affect!, save_positions=(false,true))
    sol_diffeq = solve(m_diffeq, subject1, param, randeffs; callback=cb, saveat=72:0.1:300)
    @test called
    @test all(sol_diffeq[2, sol_diffeq.t .> tsave] .>= 5e4)
  end

  @testset "DiscreteCallback" begin
    condition = function (u, t, integrator)
      u.Central < 5e5
    end

    called = false
    affect! = function (integrator)
      called = true
      integrator.u += SA[5e5, 5e5]
    end

    cb = DiscreteCallback(condition, affect!, save_positions=(false, true))
    sol_diffeq   = solve(m_diffeq, subject1, param, randeffs; callback=cb, saveat=72:0.1:300)
    @test called
    @test all(sol_diffeq[2,3:end] .>= 2e5)
  end

  @testset "cache variables" begin
    condition = function (u, t, integrator)
      t == 73.0
    end

    affect! = function (integrator)
      integrator.p.cache[:onoff] = 0.0
    end

    cb = DiscreteCallback(condition, affect!, save_positions=(false, true))
    sol_diffeq = solve(m_diffeq, subject1, param, randeffs; callback=cb, tstops=[73.0], saveat=72:0.1:80)
    @test sol_diffeq(73.0)[1] == sol_diffeq(80.0)[1]
  end
end
