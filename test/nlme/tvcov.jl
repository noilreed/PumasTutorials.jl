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

  pd = read_pumas(df, dvs=[:dv], cvs=[:pnad,:gawk,:wtkg])

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
      dv ~ @. Normal(cp, sqrt(cp^2*σ_prop))
    end
  end

  param_normal = (
    tvcl = 0.138,
    tvv  = 3.5,
    tvka = 0.9,
    ec50 = 15,
    gaeffect = 11,
    Ω = Diagonal([0.09,0.09]),
    σ_prop = 0.04
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
  tvcov_pd = read_pumas(est_df, dvs = [:dv], cvs = [:pnad,:gawk,:wtkg])

  @testset "Fit proportional (normal) error model" begin
    ft_normal = fit(
        tvcov_model_normal,
        tvcov_pd, param_normal,
        Pumas.FOCEI();
        optimize_fn = Pumas.DefaultOptimizeFN(
          show_trace=true,
          x_reltol=1e-3,
        ),
        # We use a slightly lower tolerance in the ODE solves to avoid a very slow
        # last iteration.
        reltol=1e-9
      )

    @test deviance(ft_normal) ≈ 8784.101866341049 rtol=1e-5
    @test sprint((io, t) -> show(io, MIME"text/plain"(), t), ft_normal) == """
FittedPumasModel

Successful minimization:                true

Likelihood approximation:        Pumas.FOCEI
Deviance:                          8784.1019
Total number of observation records:    4669
Number of active observation records:   4669
Number of subjects:                        7

-----------------------
             Estimate
-----------------------
tvcl          0.15973
tvv           3.7072
tvka          0.88405
ec50         15.004
gaeffect     10.968
Ω₁,₁          0.071242
Ω₂,₂          0.085379
σ_prop        0.040622
-----------------------
"""
  end

    # Currently disable since it's too slow. Enable once evaluation of time-varying covariates is faster
  # @testset "Fit gamma model" begin
  #     ft_gamma = fit(
  #       tvcov_model_gamma,
  #       tvcov_pd,
  #       param_gamma,
  #       Pumas.FOCE(),
  #       optimize_fn = Pumas.DefaultOptimizeFN(
  #         show_trace=true,
  #         g_tol=1e-1))

  #   #   @test sprint((io, t) -> show(io, MIME"text/plain"(), t), ft_normal) == """
  #   # FittedPumasModel

  #   # Successful minimization:                true

  #   # Likelihood approximation:         Pumas.FOCE
  #   # Deviance:                          8939.5062
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
