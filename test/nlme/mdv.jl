using Test, DataFrames, Pumas, Random

@testset "Multi DV support" begin
Random.seed!(4)

model = @model begin
  @param begin
    tvcl ∈ RealDomain(lower=0, init = 1.0)
    tvmetacl ∈ RealDomain(lower=0, init = 1.0)
    tvv ∈ RealDomain(lower=0, init = 100)
    tvvp ∈ RealDomain(lower=0, init = 30)
    tvq ∈ RealDomain(lower=0, init = 1.0)
    tvt ∈ RealDomain(lower=0, init = 1.0)
    tvvmeta ∈ RealDomain(lower=0, init = 100)
    Ω ∈ PDiagDomain(init=[0.04, 0.04])
    σ ∈ RealDomain(lower=0,init=0.01)
  end
  @random begin
    η ~ MvNormal(Ω)
  end
  @pre begin
    CL = tvcl *  exp(η[1])
    Vc  = tvv
    Vp  = tvvp
    Q = tvq
    CLfm = tvt
    Vm = tvvmeta
    CLm = tvmetacl *  exp(η[2])
  end
  @dynamics Central1Periph1Meta1

  @derived begin
    cp1 := @. Central/Vc
    dv1 ~ @. Normal(cp1, abs(cp1)*σ)
    cp2 := @. Metabolite/Vm
    dv2 ~ @. Normal(cp2, abs(cp1)*σ)
  end
end

model_diffeq = @model begin
  @param begin
    tvcl ∈ RealDomain(lower=0, init = 1.0)
    tvmetacl ∈ RealDomain(lower=0, init = 1.0)
    tvv ∈ RealDomain(lower=0, init = 100)
    tvvp ∈ RealDomain(lower=0, init = 30)
    tvq ∈ RealDomain(lower=0, init = 1.0)
    tvt ∈ RealDomain(lower=0, init = 1.0)
    tvvmeta ∈ RealDomain(lower=0, init = 100)
    Ω ∈ PDiagDomain(init=[0.04, 0.04])
    σ ∈ RealDomain(lower=0,init=0.01)
  end
  @random begin
    η ~ MvNormal(Ω)
  end
  @pre begin
    CL = tvcl *  exp(η[1])
    Vc  = tvv
    Vp  = tvvp
    Q = tvq
    CLfm = tvt
    Vm = tvvmeta
    CLm = tvmetacl *  exp(η[2])
  end
  @dynamics begin
      Central'          = -(CL/Vc)*Central - (Q/Vc)*Central - (CLfm/Vc)*Central + (Q/Vp)*ParentPeriph
      ParentPeriph'     = -(Q/Vp)*ParentPeriph + (Q/Vc)*Central
      Metabolite'       = -(CLm/Vm)*Metabolite + (CLfm/Vc)*Central
  end
     
  @derived begin
    cp1 := @. Central/Vc
    dv1 ~ @. Normal(cp1, abs(cp1)*σ)
    cp2 := @. Metabolite/Vm
    dv2 ~ @. Normal(cp2, abs(cp1)*σ)
  end
end
params = (tvcl     = 5.0,
          tvmetacl = 8.0,
          tvv      = 10.0,
          tvvp     = 10.0,
          tvq      = 7.0,
          tvt      = 7.0,
          tvvmeta  = 7.0,
          Ω        = Diagonal([0.05, 0.05]),
          σ        = 0.12)

pop = [Subject(id=i, evs=DosageRegimen(100, cmt=1, time=0), obs=(dv1=Float64[],dv2=Float64[])) for i = 1:1000]
sims = simobs(model, pop, params,obstimes = [0,0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.5,4.0,5.0], ensemblealg = EnsembleSerial())

reread_df = DataFrame(sims)
reread_df[reread_df[!, :evid].==1, :dv1] .= missing
reread_df[reread_df[!, :evid].==1, :dv2] .= missing
reread_df[reread_df[!, :time].==0.0, :dv2] .= missing
reread = read_pumas(reread_df; dvs=[:dv1, :dv2])

deviance_analytical = deviance(model, reread, params, Pumas.FOCEI())
deviance_diffeq = deviance(model_diffeq, reread, params, Pumas.FOCEI())
@test deviance_analytical ≈ deviance_diffeq rtol=1e-2

res_focei    = fit(model, reread, params, Pumas.FOCEI(),
  optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
res_laplacei = fit(model, reread, params, Pumas.LaplaceI(),
  optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

ref_focei = (
  tvcl = 6.715592442065195,
  tvmetacl = 6.0900313123527665,
  tvv = 10.050299214318207,
  tvvp = 9.974386611168296,
  tvq = 6.971669524314598,
  tvt = 5.361734303081132,
  tvvmeta = 5.356045290063821,
  Ω = Pumas.PDMats.PDiagMat{Float64,Array{Float64,1}}(2, [0.027967176123167337, 0.052792034715620786], [35.75620204185091, 18.94225152310917]),
  σ = 0.12054933413657265)


ref_laplacei = (
  tvcl = 7.847462662716163,
  tvmetacl = 4.796813332826394,
  tvv = 10.053015800094743,
  tvvp = 9.986110551249347,
  tvq = 6.977868238727858,
  tvt = 4.21406331637431,
  tvvmeta = 4.202633969615397,
  Ω = Pumas.PDMats.PDiagMat{Float64,Array{Float64,1}}(2, [0.020404412904399264, 0.0528569006856447], [49.00900627159904, 18.919005598668935]),
  σ = 0.1204816293521871)


@testset "FOCEI" for k in keys(params)
  if k == :Ω
    @test coef(res_focei)[k].diag ≈ ref_focei[k].diag
  else
    @test coef(res_focei)[k]      ≈ ref_focei[k]
  end
end

@testset "LaplaceI" for k in keys(params)
  if k == :Ω
    @test coef(res_laplacei)[k].diag ≈ ref_laplacei[k].diag
  else
    @test coef(res_laplacei)[k]      ≈ ref_laplacei[k]
  end
end
end