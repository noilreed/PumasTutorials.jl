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
    CL1 = tvcl *  exp(η[1])
    V1  = tvv
    Vp1  = tvvp
    Q1 = tvq
    T  = tvt
    V2 = tvvmeta
    CL2 = tvmetacl *  exp(η[2])
  end
  @dynamics Central1Periph1Meta1

  @derived begin
    cp1 := @. Central/V1
    dv1 ~ @. Normal(cp1, abs(cp1)*σ)
    cp2 := @. Metabolite/V2
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

res_focei    = fit(model, reread, params, Pumas.FOCEI())
res_laplacei = fit(model, reread, params, Pumas.LaplaceI())

ref_focei = (
  tvcl     = 5.038557192551023,
  tvmetacl = 7.9626886345076064,
  tvv      = 10.051457260229189,
  tvvp     = 9.926144330686864,
  tvq      = 6.930875337583671,
  tvt      = 7.014672271623278,
  tvvmeta  = 7.018090211121181,
  Ω        = Pumas.PDMats.PDiagMat{Float64,Array{Float64,1}}(2, [0.04903170563234689, 0.05300578322628484], [20.39496662625349, 18.865865932608532]),
  σ        = 0.12053908276835537)

ref_laplacei = (
  tvcl     = 5.008087897705468,
  tvmetacl = 7.987488005953879,
  tvv      = 10.053531364113608,
  tvvp     = 9.957954965757953,
  tvq      = 6.94205607262063,
  tvt      = 7.012380635267924,
  tvvmeta  = 6.998586717352016,
  Ω        = Pumas.PDMats.PDiagMat{Float64,Array{Float64,1}}(2, [0.049366651970012415, 0.05293917143976929], [20.256589420069364, 18.889604291176607]),
  σ        = 0.12046367847192586)

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