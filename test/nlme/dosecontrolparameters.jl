using Pumas, Test, Random

@testset "lags fit" begin

  model_lag = Dict()
  model_lag["closed form"] = @model begin
    @param begin
      θKa   ∈ RealDomain(lower=0.0, init=0.2)
      θCL   ∈ RealDomain(lower=0.0, init=0.1)
      θV    ∈ RealDomain(lower=0.0, init=1.0)
      θlag ∈ RealDomain(lower=0.0, init=0.75)
      σ     ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random   η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      Ka   = θKa
      CL   = θCL*exp(η[1])
      Vc   = θV*exp(η[2])
      lags = (Depot=θlag,)
    end

    @dynamics Depots1Central1

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  model_lag["DiffEq"] = @model begin
    @param begin
      θKa  ∈ RealDomain(lower=0.0, init=0.2)
      θCL  ∈ RealDomain(lower=0.0, init=0.1)
      θV   ∈ RealDomain(lower=0.0, init=1.0)
      θlag ∈ RealDomain(lower=0.0, init=0.75)
      σ    ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random   η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      Ka   = θKa
      CL   = θCL*exp(η[1])
      Vc   = θV*exp(η[2])
      lags = (Depot=θlag,)
    end

    @dynamics begin
      Depot'   = -Ka*Depot
      Central' =  Ka*Depot - (CL/Vc)*Central
    end

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  params₀ = init_param(model_lag["closed form"])

  n = 20
  t = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
  pop_skeleton = [Subject(obs=(dv=Float64[],), evs=DosageRegimen(100), time=t) for i in 1:n]

  Random.seed!(123)
  pop_est = Subject.(simobs(model_lag["closed form"], pop_skeleton, params₀, ensemblealg=EnsembleSerial()))

  params = (
    θKa  = 0.2,
    θCL  = 0.1,
    θV   = 1.0,
    θlag = 0.1,
    σ    = 0.1)

  @testset "$t" for t in ("closed form", "DiffEq")
    m = model_lag[t]
    @test deviance(m, pop_est, params₀, Pumas.FOCEI())        ≈ 465.7827950  rtol=1e-3
    if m.prob isa Pumas.ExplicitModel
      # Fit currently only works for analytical model. For DiffEq based model, NaNs are
      # created during the optimization. We avoid running the test in that case since it
      # takes a long time before it fails.
      @test coef(fit(m, pop_est, params, Pumas.FOCEI(),
        # The the Pumas default Backtracking linesearch can currently cause an indefinite
        # Hessian approximation so we force the Optim default Hager-Zhang linesearch until
        # Optim has been updated
        optimize_fn=Pumas.DefaultOptimizeFN(
          Pumas.BFGS(initial_invH=t -> Matrix(I*1e-6, 5, 5)),
          g_tol=1e-3))).θlag ≈ params₀.θlag rtol=1e-1
    end
  end
end

@testset "duration fit" begin

  model_duration = Dict()
  model_duration["closed form"] = @model begin
    @param begin
      θCL       ∈ RealDomain(lower=0.0, init=0.1)
      θV        ∈ RealDomain(lower=0.0, init=1.0)
      θduration ∈ RealDomain(lower=0.0, init=0.75)
      σ         ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      CL       = θCL*exp(η[1])
      Vc       = θV*exp(η[2])
      duration = (Central=θduration,)
    end

    @dynamics Central1

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  model_duration["DiffEq"] = @model begin
    @param begin
      θCL       ∈ RealDomain(lower=0.0, init=0.1)
      θV        ∈ RealDomain(lower=0.0, init=1.0)
      θduration ∈ RealDomain(lower=0.0, init=0.75)
      σ         ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      CL       = θCL*exp(η[1])
      Vc       = θV*exp(η[2])
      duration = (Central=θduration,)
    end

    @dynamics begin
      Central' = -(CL/Vc)*Central
    end

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  params₀ = init_param(model_duration["closed form"])

  n = 20
  t = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
  pop_skeleton = [Subject(obs=(dv=Float64[],), evs=DosageRegimen(100, rate=-2), time=t) for i in 1:20]

  Random.seed!(123)
  pop_est = Subject.(simobs(model_duration["closed form"], pop_skeleton, params₀, ensemblealg=EnsembleSerial()))

  params = (
    θKa       = 0.2,
    θCL       = 0.1,
    θV        = 1.0,
    θduration = 0.1,
    σ         = 0.1)

  @testset "$t" for t in ("closed form", "DiffEq")
    m = model_duration[t]
    @test deviance(m, pop_est, params₀, Pumas.FOCEI())            ≈ 752.9000974645899 rtol=1e-4
    @test coef(fit(m, pop_est, params , Pumas.FOCEI())).θduration ≈ params₀.θduration rtol=1e-1
  end
end

@testset "rate fit" begin

  model_rate = Dict()
  model_rate["closed form"] = @model begin
    @param begin
      θCL   ∈ RealDomain(lower=0.0, init=0.1)
      θV    ∈ RealDomain(lower=0.0, init=1.0)
      θrate ∈ RealDomain(lower=0.0, init=75.0)
      σ     ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      CL   = θCL*exp(η[1])
      Vc   = θV*exp(η[2])
      rate = (Central=θrate,)
    end

    @dynamics Central1

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  model_rate["DiffEq"] = @model begin
    @param begin
      θCL   ∈ RealDomain(lower=0.0, init=0.1)
      θV    ∈ RealDomain(lower=0.0, init=1.0)
      θrate ∈ RealDomain(lower=0.0, init=0.75)
      σ     ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      CL   = θCL*exp(η[1])
      Vc   = θV*exp(η[2])
      rate = (Central=θrate,)
    end

    @dynamics begin
      Central' = -(CL/Vc)*Central
    end

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  params₀ = init_param(model_rate["closed form"])

  n = 20
  t = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
  pop_skeleton = [Subject(obs=(dv=Float64[],), evs=DosageRegimen(100, rate=-2), time=t) for i in 1:20]

  Random.seed!(123)
  pop_est = Subject.(simobs(model_rate["closed form"], pop_skeleton, params₀, ensemblealg=EnsembleSerial()))

  params = (
    θCL   = 0.1,
    θV    = 1.0,
    θrate = 25.0,
    σ     = 0.1)

  @testset "$t" for t in ("closed form", "DiffEq")
    m = model_rate[t]
    @test deviance(m, pop_est, params₀, Pumas.FOCEI())        ≈ 623.904942054 rtol=1e-4
    @test coef(fit(m, pop_est, params , Pumas.FOCEI())).θrate ≈ params₀.θrate rtol=1e-1
  end
end

@testset "bioav fit" begin

  model_bioav = Dict()
  model_bioav["closed form"] = @model begin
    @param begin
      θKa    ∈ RealDomain(lower=0.0, init=0.2)
      θCL    ∈ RealDomain(lower=0.0, init=0.1)
      θV     ∈ RealDomain(lower=0.0, init=1.0)
      θbioav ∈ RealDomain(lower=0.0, init=0.75)
      σ      ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      Ka    = θKa
      CL    = θCL*exp(η[1])
      Vc    = θV*exp(η[2])
      bioav = (Depot=θbioav,)
    end

    @dynamics Depots1Central1

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  model_bioav["DiffEq"] = @model begin
    @param begin
      θKa    ∈ RealDomain(lower=0.0, init=0.2)
      θCL    ∈ RealDomain(lower=0.0, init=0.1)
      θV     ∈ RealDomain(lower=0.0, init=1.0)
      θbioav ∈ RealDomain(lower=0.0, init=0.75)
      σ      ∈ RealDomain(lower=0.0, init=0.1)
    end

    @random η ~ MvNormal(Matrix{Float64}(0.1*I, 2, 2))

    @pre begin
      Ka   = θKa
      CL   = θCL*exp(η[1])
      Vc   = θV*exp(η[2])
      bioav = (Depot=θbioav,)
    end

    @dynamics begin
      Depot'   = -Ka*Depot
      Central' =  Ka*Depot - (CL/Vc)*Central
    end

    @derived begin
      cp = @. Central / Vc
      dv = @. Normal(cp, cp*σ)
    end
  end

  params₀ = init_param(model_bioav["closed form"])

  n = 20
  t = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
  dr = DosageRegimen(DosageRegimen(100, cmt=1), DosageRegimen(100, cmt=2))
  pop_skeleton = [Subject(obs=(dv=Float64[],), evs=dr, time=t) for i in 1:20]

  Random.seed!(123)
  pop_est = Subject.(simobs(model_bioav["closed form"], pop_skeleton, params₀, ensemblealg=EnsembleSerial()))

  params = (
    θKa   = 0.2,
    θCL   = 0.1,
    θV    = 1.0,
    θbioav = 0.5,
    σ     = 0.1)

  @testset "$t" for t in ("closed form", "DiffEq")
    m = model_bioav[t]
    @test deviance(m, pop_est, params₀, Pumas.FOCEI())         ≈ 733.981432682  rtol=1e-4
    @test coef(fit(m, pop_est, params , Pumas.FOCEI())).θbioav ≈ params₀.θbioav rtol=1e-1
  end
end
