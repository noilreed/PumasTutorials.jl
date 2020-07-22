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

pop = [Subject(id=i, events=DosageRegimen(100, cmt=1, time=0)) for i = 1:1000]
sims = simobs(model, pop, params,obstimes = [0,0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.5,4.0,5.0], ensemblealg = EnsembleSerial())

reread_df = DataFrame(sims)
reread_df[reread_df[!, :evid].==1, :dv1] .= missing
reread_df[reread_df[!, :evid].==1, :dv2] .= missing
reread_df[reread_df[!, :time].==0.0, :dv2] .= missing
reread = read_pumas(reread_df; observations=[:dv1, :dv2])

loglikelihood_analytical = loglikelihood(model, reread, params, Pumas.FOCEI())
loglikelihood_diffeq = loglikelihood(model_diffeq, reread, params, Pumas.FOCEI())
@test loglikelihood_analytical ≈ loglikelihood_diffeq rtol=1e-2

res_focei    = fit(model, reread, params, Pumas.FOCEI(),
  optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))
res_laplacei = fit(model, reread, params, Pumas.LaplaceI(),
  optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

ref_focei = (
  tvcl     =  5.51061,
  tvmetacl =  7.44617,
  tvv      =  9.9675,
  tvvp     = 10.0517,
  tvq      =  6.99332,
  tvt      =  6.55763,
  tvvmeta  =  6.5788,
  Ω        = [0.045283, 0.050151],
  σ        =  0.12053)


ref_laplacei = (
  tvcl     =  6.257,
  tvmetacl =  6.59377,
  tvv      =  9.97027,
  tvvp     = 10.0635,
  tvq      =  6.99926,
  tvt      =  5.79465,
  tvvmeta  =  5.80381,
  Ω        = [0.0350107, 0.0502259],
  σ        =  0.120456)


@testset "FOCEI" for k in keys(params)
  if k == :Ω
    @test coef(res_focei)[k].diag ≈ ref_focei[k] rtol=1e-3
  else
    @test coef(res_focei)[k]      ≈ ref_focei[k] rtol=1e-3
  end
end

@testset "LaplaceI" for k in keys(params)
  if k == :Ω
    @test coef(res_laplacei)[k].diag ≈ ref_laplacei[k] rtol=1e-3
  else
    @test coef(res_laplacei)[k]      ≈ ref_laplacei[k] rtol=1e-3
  end
end
end