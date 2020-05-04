using Pumas, Test, CSV, Random, Distributions, TransformVariables

theopp = read_pumas(example_data("event_data/THEOPP"),cvs = [:WT,:SEX])

@testset "Model with analytical solution" begin
  theopmodel_bayes = @model begin
    @param begin
      θ ~ Constrained(
        MvNormal([1.9, 0.0781, 0.0463, 1.5, 0.4], Diagonal([9025, 15.25, 5.36, 5625, 400])),
        lower=[0.1, 0.008,0.0004,0.1,0.0001],
        upper=[5, 0.5, 0.09, 5, 1.5],
        init =[1.9, 0.0781, 0.0463, 1.5, 0.4])
      Ω ~ InverseWishart(2, fill(0.9, 1, 1) .* (2 + 1 + 1)) # NONMEM specifies the inverse Wishart in terms of its mode
      σ² ~ Gamma(1.0, 0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = (SEX == 1 ? θ[1] : θ[4]) + η[1]
      K  = θ[2]
      CL = θ[3]*(WT/70)^θ[5]
      Vc = CL/K
      SC = Vc/(WT/70)
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²))
    end
  end

  param = Pumas.init_param(theopmodel_bayes)

  @testset "Test logdensity" begin
    vparam = Pumas.TransformVariables.inverse(Pumas.totransform(theopmodel_bayes.param), param)
    ldp = Pumas.BayesLogDensity(theopmodel_bayes, theopp)
    vparam_aug = [vparam; zeros(length(theopp)*ldp.dim_rfx)]
    v = Pumas.logdensity(ldp, vparam_aug)
    @test v ≈ -601.6652326035172 rtol=1e-5
    vg = Pumas.logdensitygrad(ldp, vparam_aug)
    @test vg[1] ≈ v
    @test vg[2] ≈ [8.023571333788357, 878.2155638921363, -763.9131862639041, 114.23979126237573, 9.920242099411428, 1.6, 448.2675514859392, 29.396902398356612, 28.081124185351687, 28.928080707496285, 7.428841866509859, 25.018349868626935, -5.042079192537065, -21.001561268109096, -2.4377197610983528, 30.403288395312448, -13.086403802077923, 20.64430559335001, -7.709095620537856] rtol=1e-6
  end

  Random.seed!(1)
  b = Pumas.fit(theopmodel_bayes, theopp, param, Pumas.BayesMCMC(),
                nsamples = 2000)

  m = Pumas.param_mean(b)

  @test m.θ[1] ≈ 1.72E+00 rtol=0.1
  @test m.θ[2] ≈ 8.57E-02 rtol=0.1
  @test m.θ[3] ≈ 3.90E-02 rtol=0.1
  @test m.θ[4] ≈ 1.73E+00 rtol=0.1
  @test m.θ[5] ≈ 4.32E-01 rtol=0.1
  @test_broken m.Ω[1] ≈ 1.09E+00 rtol=0.1
  @test m.σ² ≈ 1.24E+00 rtol=0.1

  s = Pumas.param_std(b)

  @test s.θ[1] ≈ 4.62E-01 rtol=0.2
  @test s.θ[2] ≈ 6.92E-03 rtol=0.1
  @test s.θ[3] ≈ 2.16E-03 rtol=0.1
  @test s.θ[4] ≈ 4.74E-01 rtol=0.2
  @test s.θ[5] ≈ 1.54E-01 rtol=0.1
  @test_broken s.Ω[1] ≈ 7.00E-01 rtol=0.1
  @test s.σ²    ≈ 1.68E-01 rtol=0.2

  # Check that the parameters are not interchanged
  c = Pumas.Chains(b)
  @test mean(c.value[:, "Ω₁,₁", 1]) ≈ m.Ω[1]
  @test mean(c.value[:, "θ₁"  , 1]) ≈ m.θ[1]
  @test mean(c.value[:, "θ₂"  , 1]) ≈ m.θ[2]
  @test mean(c.value[:, "θ₃"  , 1]) ≈ m.θ[3]
  @test mean(c.value[:, "θ₄"  , 1]) ≈ m.θ[4]
  @test mean(c.value[:, "θ₅"  , 1]) ≈ m.θ[5]
  @test mean(c.value[:, "σ²"  , 1]) ≈ m.σ²

# The MCMC sampler is very sensitive to rounding so we can't enable the test below
#   @test sprint((io, o) -> show(io, MIME"text/plain"(), o), b) == """
# Object of type Chains, with data of type 2000×7×1 Array{Float64,3}

# Iterations        = 1:2000
# Thinning interval = 1
# Chains            = 1
# Samples per chain = 2000
# parameters        = Ω₁,₁, θ₁, θ₂, θ₃, θ₄, θ₅, σ

# 2-element Array{MCMCChains.ChainDataFrame,1}

# Summary Statistics

# │ Row │ parameters │ mean      │ std        │ naive_se    │ mcse        │ ess     │ r_hat    │
# │     │ Symbol     │ Float64   │ Float64    │ Float64     │ Float64     │ Any     │ Any      │
# ├─────┼────────────┼───────────┼────────────┼─────────────┼─────────────┼─────────┼──────────┤
# │ 1   │ Ω₁,₁       │ 1.49377   │ 0.897115   │ 0.0200601   │ 0.031706    │ 688.897 │ 1.00011  │
# │ 2   │ θ₁         │ 1.78834   │ 0.533551   │ 0.0119306   │ 0.0214954   │ 495.926 │ 0.999506 │
# │ 3   │ θ₂         │ 0.0848287 │ 0.00726938 │ 0.000162548 │ 0.000250645 │ 838.967 │ 1.00019  │
# │ 4   │ θ₃         │ 0.0388356 │ 0.00226758 │ 5.07047e-5  │ 7.15372e-5  │ 834.567 │ 1.00031  │
# │ 5   │ θ₄         │ 1.73274   │ 0.543149   │ 0.0121452   │ 0.0241309   │ 388.053 │ 1.00058  │
# │ 6   │ θ₅         │ 0.446534  │ 0.152098   │ 0.00340102  │ 0.00454973  │ 1083.75 │ 0.999534 │
# │ 7   │ σ          │ 1.26755   │ 0.187834   │ 0.0042001   │ 0.00589619  │ 72.265  │ 0.999697 │

# Quantiles

# │ Row │ parameters │ 2.5%      │ 25.0%     │ 50.0%     │ 75.0%     │ 97.5%     │
# │     │ Symbol     │ Float64   │ Float64   │ Float64   │ Float64   │ Float64   │
# ├─────┼────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
# │ 1   │ Ω₁,₁       │ 0.458561  │ 0.900564  │ 1.26539   │ 1.84386   │ 3.8557    │
# │ 2   │ θ₁         │ 0.774682  │ 1.41303   │ 1.76979   │ 2.13333   │ 2.90965   │
# │ 3   │ θ₂         │ 0.0719378 │ 0.0795374 │ 0.0846473 │ 0.0893413 │ 0.099581  │
# │ 4   │ θ₃         │ 0.0345391 │ 0.0372849 │ 0.0387768 │ 0.0403899 │ 0.0432724 │
# │ 5   │ θ₄         │ 0.623837  │ 1.41342   │ 1.71314   │ 2.05895   │ 2.87534   │
# │ 6   │ θ₅         │ 0.141999  │ 0.347938  │ 0.443525  │ 0.554605  │ 0.737866  │
# │ 7   │ σ          │ 0.971468  │ 1.1503    │ 1.24847   │ 1.36214   │ 1.65648   │
# """
end

@testset "Model with ODE solver" begin
  theopmodel_bayes2 = @model begin
    @param begin
      θ ~ Constrained(
        MvNormal([1.9, 0.0781, 0.0463, 1.5, 0.4], Diagonal([9025, 15.25, 5.36, 5625, 400])),
        lower=[0.1, 0.008 , 0.0004, 0.1, 0.0001],
        upper=[5.0, 0.5   , 0.09,   5.0, 1.5   ],
        init =[1.9, 0.0781, 0.0463, 1.5, 0.4   ])
      Ω ~ InverseWishart(2, fill(0.9, 1, 1) .* (2 + 1 + 1)) # NONMEM specifies the inverse Wishart in terms of its mode
      σ² ~ Gamma(1.0, 0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = (SEX == 1 ? θ[1] : θ[4]) + η[1]
      K  = θ[2]
      CL = θ[3]*(WT/70)^θ[5]
      Vc = CL/K
      SC = Vc/(WT/70)
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics begin
      Depot'   = -Ka*Depot
      Central' =  Ka*Depot - K*Central
    end

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²))
    end
  end

  param2 = Pumas.init_param(theopmodel_bayes2)

  @testset "Test logdensity" begin
    vparam2 = Pumas.TransformVariables.inverse(Pumas.totransform(theopmodel_bayes2.param), param2)
    ldp2 = Pumas.BayesLogDensity(theopmodel_bayes2, theopp,
                                 reltol = 1e-12, abstol = 1e-12)
    vparam2_aug = [vparam2; zeros(length(theopp)*ldp2.dim_rfx)]
    v2 = Pumas.logdensity(ldp2, vparam2_aug)
    @test v2 ≈ -601.6652326035172 rtol=1e-5
    vg2 = Pumas.logdensitygrad(ldp2, vparam2_aug)
    @test vg2[1] ≈ v2
    @test vg2[2] ≈ [8.023571333788363, 878.2155638921357, -763.9131862639035, 114.23979126237566, 9.92024209941141, 1.6, 448.2675514859386, 29.3969023983566, 28.081124185351666, 28.928080707496267, 7.42884186650985, 25.018349868626917, -5.042079192537063, -21.001561268109082, -2.437719761098352, 30.403288395312426, -13.086403802077914, 20.644305593349998, -7.709095620537853] rtol=1e-6
  end

#   Random.seed!(1)
#   The MCMC sampler is very sensitive to rounding so we can't enable the test below
#   b = Pumas.fit(theopmodel_bayes2, theopp, param2, Pumas.BayesMCMC(),
#   # Only compute 30 iterations as it's rather slow when using ODE solver
#                 nsamples = 30, reltol = 1e-6, abstol = 1e-6)

#   m = Pumas.param_mean(b)

#   # Regresion tests
#   @test m.θ[1] ≈ 1.546187   rtol=1e-6
#   @test m.θ[2] ≈ 0.08439249 rtol=1e-6
#   @test m.θ[3] ≈ 0.03849392 rtol=1e-6
#   @test m.θ[4] ≈ 1.759328   rtol=1e-6
#   @test m.θ[5] ≈ 0.4500819  rtol=1e-6
#   @test m.Ω[1] ≈ 1.668558   rtol=1e-6
#   @test m.σ    ≈ 1.404219   rtol=1e-6

#   s = Pumas.param_std(b)

#   @test s.θ[1] ≈ 0.5177061   rtol=1e-6
#   @test s.θ[2] ≈ 0.007861914 rtol=1e-6
#   @test s.θ[3] ≈ 0.002883981 rtol=1e-6
#   @test s.θ[4] ≈ 0.5790466   rtol=1e-6
#   @test s.θ[5] ≈ 0.09112155  rtol=1e-6
#   @test s.Ω[1] ≈ 0.6838233   rtol=1e-6
#   @test s.σ    ≈ 0.7132078   rtol=1e-6

#   @test sprint((io, o) -> show(io, MIME"text/plain"(), o), b) == """
# Object of type Chains, with data of type 30×7×1 Array{Float64,3}

# Iterations        = 1:30
# Thinning interval = 1
# Chains            = 1
# Samples per chain = 30
# parameters        = Ω₁,₁, θ₁, θ₂, θ₃, θ₄, θ₅, σ

# 2-element Array{MCMCChains.ChainDataFrame,1}

# Summary Statistics

# │ Row │ parameters │ mean      │ std        │ naive_se    │ mcse    │ ess     │ r_hat    │
# │     │ Symbol     │ Float64   │ Float64    │ Float64     │ Missing │ Any     │ Any      │
# ├─────┼────────────┼───────────┼────────────┼─────────────┼─────────┼─────────┼──────────┤
# │ 1   │ Ω₁,₁       │ 1.66856   │ 0.683823   │ 0.124848    │ missing │ 12.6536 │ 1.00404  │
# │ 2   │ θ₁         │ 1.54619   │ 0.517706   │ 0.0945198   │ missing │ 2.82239 │ 1.40858  │
# │ 3   │ θ₂         │ 0.0843925 │ 0.00786191 │ 0.00143538  │ missing │ 31.0381 │ 0.989056 │
# │ 4   │ θ₃         │ 0.0384939 │ 0.00288398 │ 0.000526541 │ missing │ 30.0055 │ 0.972069 │
# │ 5   │ θ₄         │ 1.75933   │ 0.579047   │ 0.105719    │ missing │ 5.4379  │ 1.36841  │
# │ 6   │ θ₅         │ 0.450082  │ 0.0911216  │ 0.0166364   │ missing │ 85.9188 │ 0.983301 │
# │ 7   │ σ          │ 1.40422   │ 0.713208   │ 0.130213    │ missing │ 40.2338 │ 0.988408 │

# Quantiles

# │ Row │ parameters │ 2.5%      │ 25.0%     │ 50.0%     │ 75.0%     │ 97.5%     │
# │     │ Symbol     │ Float64   │ Float64   │ Float64   │ Float64   │ Float64   │
# ├─────┼────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
# │ 1   │ Ω₁,₁       │ 0.873919  │ 1.07694   │ 1.41382   │ 2.0408    │ 3.06603   │
# │ 2   │ θ₁         │ 0.837669  │ 1.2476    │ 1.45303   │ 1.8741    │ 2.62987   │
# │ 3   │ θ₂         │ 0.0684815 │ 0.0813231 │ 0.0858188 │ 0.0891202 │ 0.0951375 │
# │ 4   │ θ₃         │ 0.0333643 │ 0.0365594 │ 0.038746  │ 0.0403524 │ 0.0436871 │
# │ 5   │ θ₄         │ 0.980944  │ 1.43295   │ 1.61561   │ 1.86238   │ 3.0131    │
# │ 6   │ θ₅         │ 0.293041  │ 0.397551  │ 0.427208  │ 0.508322  │ 0.623495  │
# │ 7   │ σ          │ 0.570415  │ 1.14983   │ 1.27373   │ 1.43195   │ 3.10419   │
# """
end

@testset "Model with no priors" begin
  theopmodel_bayes = @model begin
    @param begin
      θ ∈ VectorDomain(5,
        lower=[0.1,0.008,0.0004,0.1,0.0001],
        upper=[5,0.5,0.09,5,1.5],
        init=[1.9,0.0781,0.0463,1.5,0.4]
      )
      Ω  ∈ PSDDomain(1)
      σ² ~ Uniform(0.0, 2*0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = (SEX == 1 ? θ[1] : θ[4]) + η[1]
      K  = θ[2]
      CL = θ[3]*(WT/70)^θ[5]
      Vc  = CL/K
      SC = Vc/(WT/70)
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²))
    end
  end

  param = Pumas.init_param(theopmodel_bayes)

  @testset "Test logdensity" begin
    vparam = Pumas.TransformVariables.inverse(Pumas.totransform(theopmodel_bayes.param), param)
    ldp = Pumas.BayesLogDensity(theopmodel_bayes, theopp)
    vparam_aug = [vparam; zeros(length(theopp)*ldp.dim_rfx)]
    v = Pumas.logdensity(ldp, vparam_aug)
    @test v ≈ -582.1757519457573 rtol=1e-5
    vg = Pumas.logdensitygrad(ldp, vparam_aug)
    @test vg[1] ≈ v
    @test vg[2] ≈ [8.023571333788357, 878.2155638921363, -763.9131862639041, 114.23979126237573, 9.920242099411428, 2.0, 224.1337757429696, 29.396902398356612, 28.081124185351687, 28.928080707496285, 7.428841866509859, 25.018349868626935, -5.042079192537065, -21.001561268109096, -2.4377197610983528, 30.403288395312448, -13.086403802077923, 20.64430559335001, -7.709095620537856] rtol=1e-6
  end
end
