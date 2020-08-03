using Test
using Pumas
using Random

@testset "diagnostics" begin
Random.seed!(4)
data = read_pumas(example_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
mdsl_additive = @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
    σ ∈ RealDomain(init=sqrt(0.1))
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
    dv ~ @. Normal(conc, σ)
  end
end

mdsl_proportional = @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
    σ ∈ RealDomain(init=sqrt(0.1))
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
    dv ~ @. Normal(conc,conc*σ+eps())
  end
end

param = init_param(mdsl_proportional)

@testset "NPDE" begin
  Random.seed!(123)
  pnpde = [Pumas.npde(mdsl_proportional, data[i], param, nsim=10000) for i in 1:10]

  pnpde_ref = [[ 0.16010292129646753,  1.7006964611109079 ],
               [-1.374778904986603  , -0.2554183492936665 ],
               [ 0.32233357521872197,  0.6501427767557446 ],
               [ 0.37884899739898487,  1.381256851719617  ],
               [ 0.6061728191350862 , -1.7518476977477364 ],
               [ 0.8531069457458029 , -0.716309924542552  ],
               [ 0.2533471031357997 ,  1.7518476977477355 ],
               [-0.16137253124530246,  0.6206954860134504 ],
               [-1.3806064470008992 ,  0.98220269533347   ],
               [ 0.9051234314700444 ,  0.3517843449351561 ]]

  for (_pnpde, _ref) in zip(pnpde, pnpde_ref)
    @test _pnpde.dv ≈ _ref
  end
end

@testset "Expected population predictions" for dt in data
  Random.seed!(123)
  @test Pumas.epredict(mdsl_proportional, dt, param, nsim=10000).dv ≈ [10.030280335946054, 6.020891201534397] rtol=1e-6
end

@testset "__predict(::FO) (PRED)" for
  (sub_pred, dt) in zip(fill([10.0000000, 6.06530660], length(data)), data)

  @test Pumas.__predict(mdsl_proportional, dt, param, zeros(1), Pumas.FO()).dv ≈ sub_pred rtol=1e-6
end

@testset "__wresiduals(::FO) (WRES)" for
  (sub_wres, dt) in zip([[ 0.180566054, 1.74797817 ],
                         [-1.35845124 ,-0.274456699],
                         [ 0.310535666, 0.611240923],
                         [ 0.394652252, 1.41153536 ],
                         [ 0.607473539,-1.68539881 ],
                         [ 0.858874613,-0.769228457],
                         [ 0.245708974, 1.74827643 ],
                         [-0.169086986, 0.608506828],
                         [-1.38172560 , 0.984121759],
                         [ 0.905043866, 0.302785305]], data)

  @test Pumas.__wresiduals(mdsl_proportional, dt, param, zeros(1), Pumas.FO()).dv ≈ sub_wres
end

@testset "__wresiduals(::FOCE), (CWRES)" begin
  for (sub_cwres, dt) in zip([[  1.8056605439561435, 6.35847069362139   ],
                              [-13.584512372551321 , -0.7859197550457881],
                              [  3.105356662285346 , 1.925994243881485  ],
                              [  3.946522519890135 , 4.922598941391171  ],
                              [  6.0747353851834545, -4.439084764794151 ],
                              [  8.588746125017316 , -2.116293405692951 ],
                              [  2.457089741950828 , 6.359788099833816  ],
                              [ -1.690869864892035 , 1.916741606669153  ],
                              [-13.817256008339715 , 3.249017333791544  ],
                              [  9.050438663401902 , 0.9200622059620783]], data)

    @test Pumas.__wresiduals(mdsl_additive, dt, param,
      Pumas._orth_empirical_bayes(mdsl_additive, dt, param, Pumas.FOCE()), Pumas.FOCE()).dv ≈ sub_cwres
  end
end

@testset "__wresiduals(::FOCEI), (CWRESI)" for
  (sub_cwresi, dt) in zip([[ 0.180566054, 1.6665779  ],
                           [-1.35845124 ,-0.278938663],
                           [ 0.310535666, 0.605059261],
                           [ 0.394652252, 1.36101861 ],
                           [ 0.607473539,-1.74177468 ],
                           [ 0.858874613,-0.789814478],
                           [ 0.245708974, 1.6668457  ],
                           [-0.169086986, 0.602404841],
                           [-1.3817256  , 0.962485383],
                           [ 0.905043866, 0.302554671]], data)

  @test Pumas.__wresiduals(mdsl_proportional, dt, param,
    Pumas._orth_empirical_bayes(mdsl_proportional, dt, param, Pumas.FOCEI()), Pumas.FOCEI()).dv ≈ sub_cwresi rtol=1e-6
end

@testset "__iwresiduals(::FO) (IWRES)" for
  (sub_iwres, dt) in zip([[ 0.180566054, 1.83329497 ],
                          [-1.35845124 ,-0.287852614],
                          [ 0.310535666, 0.641074888],
                          [ 0.394652252, 1.48043078 ],
                          [ 0.607473539,-1.76766118 ],
                          [ 0.858874613,-0.806773612],
                          [ 0.245708974, 1.83360779 ],
                          [-0.169086986, 0.638207345],
                          [-1.38172560 , 1.03215561 ],
                          [ 0.905043866, 0.317563907]], data)

  @test Pumas.__iwresiduals(mdsl_proportional, dt, param, zeros(1), Pumas.FO()).dv ≈ sub_iwres
end

@testset "__iwresiduals(::FOCE) (ICWRES)" begin
  for (sub_icwres, dt) in zip([
    [  1.8056605439561437,  4.4147744872867865],
    [-13.584512372551323 , -0.3448880160816745],
    [  3.105356662285346 , 1.0249555597465865 ],
    [  3.9465225198901352, 3.1844111981243173 ],
    [  6.0747353851834545, -1.7642741052120818],
    [  8.588746125017316 , -0.8660754784374615],
    [  2.457089741950828 , 4.415945534472785  ],
    [ -1.690869864892035 , 1.0193403639202951 ],
    [-13.817256008339715 , 1.8980127284645392 ],
    [  9.050438663401902 , 0.4545489978903738 ]], data)
    @test Pumas.__iwresiduals(mdsl_additive, dt, param,
      Pumas._orth_empirical_bayes(mdsl_additive, dt, param, Pumas.FOCE()), Pumas.FOCE()).dv ≈ sub_icwres rtol=1e-6
  end
end

@testset "__iwresiduals(::FOCEI) (ICWRESI)" for
  (sub_icwresi, dt) in zip([[ 0.180566054, 1.56991766 ],
                            [-1.35845124 ,-0.236161082],
                            [ 0.310535666, 0.595884676],
                            [ 0.394652252, 1.29087676 ],
                            [ 0.607473539,-1.71221172 ],
                            [ 0.858874613,-0.734054331],
                            [ 0.245708974, 1.57016202 ],
                            [-0.169086986, 0.593425217],
                            [-1.38172560 , 0.925641802],
                            [ 0.905043866, 0.314343255]], data)

  @test Pumas.__iwresiduals(mdsl_proportional, dt, param,
    Pumas._orth_empirical_bayes(mdsl_proportional, dt, param, Pumas.FOCEI()), Pumas.FOCEI()).dv ≈ sub_icwresi rtol=1e-5
end

@testset "Expected individual residuals" begin
  Random.seed!(123)
  ref = [[0.18056605439557627 ,  1.918953453441187  ],
         [-1.3584512372549291 , -0.24373095514789203],
         [0.31053566622857914 ,  0.7052404787625878 ],
         [0.39465225198907283 ,  1.5566598382264618 ],
         [0.6074735385182652  , -1.7446353470602294 ],
         [0.8588746125016998  , -0.768425798009981  ],
         [0.24570897419512916 ,  1.903174873729099  ],
         [-0.16908698648921602,  0.6927430111312871 ],
         [-1.3817256008337437 ,  1.0987718424896216 ],
         [0.9050438663400313  ,  0.372522797258317  ]]

  for (dt, _ref) in zip(data, ref)
    @test Pumas.eiwres(mdsl_proportional, dt, param, 10000).dv ≈ _ref rtol=1e-6
  end
end

param = (θ = [0.340689], Ω = Diagonal([0.000004]), σ = sqrt(0.0752507))

@testset "Shrinkage" begin
  _vvrandeffsorth = [Pumas._orth_empirical_bayes(mdsl_proportional, subject, param, Pumas.FOCEI()) for subject in data]

  @test Pumas._ηshrinkage(mdsl_proportional, data, param, _vvrandeffsorth).η ≈ [0.997574] rtol=1e-6

  Pumas._ϵshrinkage(mdsl_proportional, data, param,
    _vvrandeffsorth)
end

end
