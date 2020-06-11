using Pumas, Test, LinearAlgebra

include("testmodels.jl")

@testset "Test informationmatrix with warfarin data" begin

  data = read_pumas(example_data("warfarin"))

  model = @model begin

    @param begin
      θ₁ ∈ RealDomain(lower=0.0, init=0.15)
      θ₂ ∈ RealDomain(lower=0.0, init=8.0)
      θ₃ ∈ RealDomain(lower=0.0, init=1.0)
      Ω  ∈ PDiagDomain(3)
      σ  ∈ RealDomain(lower=0.0001, init=sqrt(0.01))
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Tvcl = θ₁
      Tvv  = θ₂
      Tvka = θ₃
      CL   = Tvcl*exp(η[1])
      Vc   = Tvv*exp(η[2])
      Ka   = Tvka*exp(η[3])
    end

    @dynamics Depots1Central1

    @vars begin
      conc = Central / Vc
    end

    @derived begin
      dv ~ @. Normal(log(conc), σ)
    end
  end

  param = (θ₁ = 0.15,
           θ₂ = 8.0,
           θ₃ = 1.0,
           Ω  = Diagonal([0.07, 0.02, 0.6]),
           σ  = sqrt(0.01))

  @test logdet(
    sum(
      Pumas._expected_information(
        model,
        d,
        param,
        Pumas._orth_empirical_bayes(model, d, param, Pumas.FO()),
        Pumas.FO()
      ) for d in data)) ≈ 50.6766056254067 rtol=1e-6

  ft = fit(model, data, param, Pumas.FO(),
    optimize_fn=Pumas.DefaultOptimizeFN(show_trace=false))

  @test logdet(informationmatrix(ft)) isa Number

end

@testset "Multiple dvs. (The HCV model)" begin

  t = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 7.0, 10.0, 14.0, 21.0, 28.0]
  _sub = Subject(
    id=1,
    evs=DosageRegimen(180.0, ii=7.0, addl=3, duration=1.0),
    time=t,
    obs=(yPK=zeros(length(t)), yPD=zeros(length(t))))

  @test logdet(Pumas._expected_information_fd(
    TESTMODELS["HCV"]["model"],
    _sub,
    TESTMODELS["HCV"]["params"],
    zeros(7),
    Pumas.FO())*30) ≈ 92.2113619232362 rtol=1e-4
end
