using Test, DataFrames, Query, Pumas, Random

# This model used to cause many issues when trying to fit it because the gradients ended up
# being imprecisely calculated

# The Query stuff fails if moved inside a testset. We might want to find a different way
# of setting up the dataframe.
function bwkg(gawk, m_bwg_ga40 = 3000)
  bwg_ga = (m_bwg_ga40/3705) * exp(0.578 + 0.332 * (gawk + 0.5) - 0.00354 * (gawk + 0.5)^2)
  bwkg = bwg_ga/1000
end

df = DataFrame(id = repeat(1:7, inner = (30 * 24) + 1),
             gawkbirth=repeat(collect(28:2:40), inner = (30 * 24) + 1),
             timeH = repeat(collect(0:1:30*24), outer=7)) |>
  @mutate(pnad = round.(_.timeH/24, digits = 4)) |>
  @mutate(gawk = round.(_.gawkbirth + (_.pnad/7), digits = 4)) |>
  @mutate(wtkg = round.(bwkg(_.gawk), digits = 4)) |>
  @filter(_.pnad >= 1) |>
  @mutate(time = _.timeH - 24) |>
  @mutate(evid = ifelse((_.timeH % 24) == 0, 1, 0)) |>
  @mutate(cmt = ifelse(_.evid == 1, 1, 2)) |>
  @mutate(amt = ifelse(
    _.evid == 1 && _.gawk < 30 , 10 * _.wtkg,
    ifelse(_.evid == 1 && _.gawk >= 30, 15 * _.wtkg,0))) |>
  @mutate(dv = missing) |> DataFrame
# Call to identity ensures that columns are typed. It seems that this is required in Julia 1.3
# but not Julia 1.4dev
df = identity.(df)

@testset "Model with time varying covariates" begin

  pd = read_pumas(df, observations=[:dv], covariates=[:pnad,:gawk,:wtkg])

  tvcov_model_normal = @model begin
    @param begin
      tvcl ∈ RealDomain(lower = 0)
      tvv ∈ RealDomain(lower = 0)
      tvka ∈ RealDomain(lower = 0)
      ec50 ∈ RealDomain(lower = 0)
      gaeffect ∈ RealDomain(lower = 0)
      Ω ∈ PDiagDomain(2)
      σ_prop ∈ RealDomain(lower=0)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @covariates pnad gawk wtkg

    @pre begin
      mat = pnad / (ec50 + pnad)
      cl = tvcl * mat * (gawk/34)^gaeffect * (1 - mat) * wtkg^0.75 * exp(η[1])
      v = tvv * wtkg * exp(η[2])
      ka = tvka
    end

    @dynamics begin
      Depot' = -ka*Depot
      Central' =  ka*Depot - (cl/v)*Central
    end

    @derived begin
      cp = @. (Central / v)
      dv ~ @. Normal(cp, cp*σ_prop)
    end
  end


  tvcov_model_normal_analytical = @model begin
    @param begin
      tvcl ∈ RealDomain(lower = 0)
      tvv ∈ RealDomain(lower = 0)
      tvka ∈ RealDomain(lower = 0)
      ec50 ∈ RealDomain(lower = 0)
      gaeffect ∈ RealDomain(lower = 0)
      Ω ∈ PDiagDomain(2)
      σ_prop ∈ RealDomain(lower=0)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @covariates pnad gawk wtkg

    @pre begin
      mat = pnad / (ec50 + pnad)
      CL = tvcl * mat * (gawk/34)^gaeffect * (1 - mat) * wtkg^0.75 * exp(η[1])
      Vc = tvv * wtkg * exp(η[2])
      Ka = tvka
    end

    @dynamics Depots1Central1

    @derived begin
      cp = @. (Central / Vc)
      dv ~ @. Normal(cp, cp*σ_prop)
    end
  end

  param_normal = (
    tvcl = 0.138,
    tvv  = 3.5,
    tvka = 0.9,
    ec50 = 15,
    gaeffect = 11,
    Ω = Diagonal([0.09,0.09]),
    σ_prop = 0.2
  )


  tvcov_model_gamma = @model begin
    @param begin
      tvcl ∈ RealDomain(lower = 0)
      tvv ∈ RealDomain(lower = 0)
      tvka ∈ RealDomain(lower = 0)
      ec50 ∈ RealDomain(lower = 0)
      gaeffect ∈ RealDomain(lower = 0)
      Ω ∈ PDiagDomain(2)
      ν ∈ RealDomain(lower=0)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @covariates pnad gawk wtkg

    @pre begin
      mat = pnad / (ec50 + pnad)
      cl = tvcl * mat * (gawk/34)^gaeffect * (1 - mat) * wtkg^0.75 * exp(η[1])
      v = tvv * wtkg * exp(η[2])
      ka = tvka
    end

    @dynamics begin
      Depot' = -ka*Depot
      Central' =  ka*Depot - (cl/v)*Central
    end

    @derived begin
      cp = @. (Central / v)
      dv ~ @. Gamma(ν, cp/ν)
    end
  end

  param_gamma = (
    tvcl = 0.138,
    tvv  = 3.5,
    tvka = 0.9,
    ec50 = 15,
    gaeffect = 11,
    Ω = Diagonal([0.09,0.09]),
    ν = 0.04
  )

  Random.seed!(123)
  obs = simobs(tvcov_model_normal, pd, param_normal, abstol=1e-12, reltol=1e-12, ensemblealg = EnsembleSerial())
  sim_df = DataFrame(obs)
  est_df = sim_df |>
    @mutate(cmt = ifelse(ismissing(_.cmt), 2, _.cmt)) |> DataFrame
  tvcov_pd = read_pumas(est_df, observations = [:dv], covariates = [:pnad,:gawk,:wtkg])

  # check that we can't get closer than this, but note this has a lot of observations
  # and discontinuities, so maybe it's not too surprising that we're seeing differences here.
  @test abs(loglikelihood(tvcov_model_normal, tvcov_pd, param_normal, Pumas.FOCEI())-loglikelihood(tvcov_model_normal_analytical, tvcov_pd, param_normal, Pumas.FOCEI()))<0.50

  @testset "check that extra zero dose events haven't been added to the subjects" begin
    @test length(tvcov_pd[1].events) == 30
  end

  @testset "Fit proportional (normal) error model" begin
    ft_normal = fit(
        tvcov_model_normal,
        tvcov_pd, param_normal,
        Pumas.FOCEI();
        # We use a slightly lower tolerance in the ODE solves to avoid a very slow
        # last iteration.
        reltol=1e-9
      )

    @test loglikelihood(ft_normal) ≈ -9263.9016             rtol=1e-5
    @test coef(ft_normal).tvcl     ≈ 0.154637               rtol=1e-2
    @test coef(ft_normal).tvv      ≈ 3.66215                rtol=1e-3
    @test coef(ft_normal).tvka     ≈ 0.873848               rtol=1e-2
    @test coef(ft_normal).ec50     ≈ 15.5943                rtol=1e-2
    @test coef(ft_normal).gaeffect ≈ 10.5709                rtol=1e-2
    @test coef(ft_normal).Ω.diag   ≈ [0.0427348, 0.0678586] rtol=1e-2
    @test coef(ft_normal).σ_prop   ≈ 0.201621               rtol=1e-2

    @testset "icoef" begin
      ic = icoef(ft_normal)
      # The covariates are time-varying
      @test ic[1](0.0) != ic[1](10.0)
      icdf = reduce(vcat, DataFrame.(ic))
      @test filter(i -> i.id == "7" && i.time > 690, icdf).cl ≈ [
        1.0692860963883548
        1.070396783342263
        1.0715561430899154
        1.072667067340859
        1.0738054170767328
        1.0749176898756483] rtol=1e-4

      ic2 = icoef(ft_normal.model, ft_normal.data[1], coef(ft_normal), obstimes=[0.0, 5.5])
      ic2df = DataFrame(ic2)
      @test hasproperty(ic2df, :time)
      @test ic2df.cl[1] != ic2df.cl[2]

      @test_throws ArgumentError("covariates for subject 1 are not constant. Please pass `obstimes` argument.") icoef(ft_normal.model, ft_normal.data[1], coef(ft_normal), obstimes=nothing)
    end
  end

    # Currently disable since it's too slow. Enable once evaluation of time-varying covariates is faster
  # @testset "Fit gamma model" begin
  #     ft_gamma = fit(
  #       tvcov_model_gamma,
  #       tvcov_pd,
  #       param_gamma,
  #       Pumas.FOCE(),
  #       optimize_fn = Pumas.DefaultOptimizeFN(
  #         g_tol=1e-1))

  #   #   @test sprint((io, t) -> show(io, MIME"text/plain"(), t), ft_normal) == """
  #   # FittedPumasModel

  #   # Successful minimization:                true

  #   # Likelihood approximation:         Pumas.FOCE
  #   # Log-likelihood:                    8939.5062
  #   # Total number of observation records:    4669
  #   # Number of active observation records:   4669
  #   # Number of subjects:                        7

  #   # -----------------------
  #   #              Estimate
  #   # -----------------------
  #   # tvcl          0.15977
  #   # tvv           3.6998
  #   # tvka          0.89359
  #   # ec50         14.917
  #   # gaeffect     10.952
  #   # Ω₁,₁          0.071966
  #   # Ω₂,₂          0.087925
  #   # ν            23.217
  #   # -----------------------
  #   # """
  #   end
end

@testset "tvcov model against hand-rolled" begin

D0 = 50
function depots1central1(t, subject)
  t0 = 0
  Ka, CL, V = 0.3, 0.5, 3.0

  expo = 2.0

  weight = map(u->u.weight, subject.covariates.u)

  Ke = CL./V.*(weight./70).^expo
  v = Ke./Ka .- 1

  Depot0 = 0.0
  Central0 = 10.0

  Depot05 = D0
  Central05 = Depot0/v[2]*exp(-Ka*0.5) + (Central0-Depot0/v[2])*exp(-Ke[2]*0.5)

  Depot1 = Depot05*exp(-Ka*0.5)
  Central1 = Depot05/v[2]*exp(-Ka*0.5) + (Central05-Depot05/v[2])*exp(-Ke[2]*0.5)

  Depot2 = Depot1*exp(-Ka)
  Central2 = Depot1/v[3]*exp(-Ka) + (Central1-Depot1/v[3])*exp(-Ke[3])

  Depot3 = Depot2*exp(-Ka)
  Central3 = Depot2/v[4]*exp(-Ka) + (Central2-Depot2/v[4])*exp(-Ke[4])

  if t == t0
    return (Depot=Depot0, Central=Central0)
  elseif t < 0.5
    dt = t - t0
    return (Depot=Depot0*exp(-Ka*dt), Central=Depot0/v[2]*exp(-Ka*dt)+(Central0-Depot0/v[2])*exp(-Ke[2]*dt))
  elseif t < 1
    dt = t - 0.5
    return (Depot=Depot05*exp(-Ka*dt), Central=Depot05/v[2]*exp(-Ka*dt)+(Central05-Depot05/v[2])*exp(-Ke[2]*dt))
  elseif t < 2
    dt = t - 1
    return (Depot=Depot1*exp(-Ka*dt), Central=Depot1/v[3]*exp(-Ka*dt)+(Central1-Depot1/v[3])*exp(-Ke[3]*dt))
  else t < 3
    dt = t - 2
    return (Depot=Depot2*exp(-Ka*dt), Central=Depot2/v[4]*exp(-Ka*dt)+(Central2-Depot2/v[4])*exp(-Ke[4]*dt))
  end
end

subject = Subject(
  covariates=(weight=[75.0, 180.0, 60.0, 70.0],),
  covariates_time=(weight=[0.0,1.0,2.0,3.0],),
  events=DosageRegimen(
    DosageRegimen(D0; time=0.5, cmt=:Depot),
    DosageRegimen(10; cmt=:Central)))

model_analytical =  @model begin
  @covariates weight

  @pre begin
    Ka = 0.3
    CL = 0.5*(weight/70)^2
    Vc = 3.0
  end

  @dynamics Depots1Central1
end

model_linear =  @model begin
  @covariates weight

  @pre begin
    Ka = 0.3
    CL = 0.5*(weight/70)^2
    Vc = 3.0
    A = [-Ka 0.0; Ka -CL/Vc]
  end

  @init begin
    Depot   = 0.0
    Central = 0.0
  end

  @dynamics LinearODE
end

model_diffeq =  @model begin
  @covariates weight

  @pre begin
    Ka = 0.3
    CL = 0.5*(weight/70)^2
    Vc = 3.0
  end

  @dynamics begin
    Depot' = -Ka*Depot
    Central' = Ka*Depot - CL/Vc*Central
  end
end

_saveat = [0.0, 0.2, 0.5, 0.75, 1.0, 1.6, 2.0, 2.1, 3.0]
sol_analytical = solve(model_analytical, subject, NamedTuple())
sol_linear = solve(model_linear, subject, NamedTuple())
sol_diffeq = solve(model_diffeq, subject, NamedTuple(); saveat=_saveat, abstol=1e-12, reltol=1e-12)
sol_hand = t->depots1central1(t, subject)

depot_analytical = map(t->sol_analytical(t).Depot, _saveat)
depot_linear     = map(t->sol_linear(t)[1], _saveat)
depot_diffeq     = map(t->sol_diffeq(t).Depot, _saveat)
depot_hand       = map(t->sol_hand(t).Depot, _saveat)

central_analytical = map(t->sol_analytical(t).Central, _saveat)
central_linear     = map(t->sol_linear(t)[2], _saveat)
central_diffeq = map(t->sol_diffeq(t).Central, _saveat)
central_hand = map(t->sol_hand(t).Central, _saveat)

@test depot_analytical ≈ depot_hand
@test depot_linear     ≈ depot_hand
@test depot_diffeq     ≈ depot_hand rtol=1e-8
@test central_analytical ≈ central_hand
@test central_linear     ≈ central_hand
@test central_diffeq     ≈ central_hand rtol=1e-8

end
