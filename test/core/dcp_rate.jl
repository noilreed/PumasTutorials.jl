using Pumas, Test

@testset "Testing dose control parameters" begin

choose_covariates() = (isPM = rand(["yes", "no"]),
                       Wt = rand(55:80))

function generate_population(events,nsubs=24)
  pop = Population(map(i -> Subject(id=i,evs=events,cvs=choose_covariates()),1:nsubs))
  return pop
end

m_diffeq = @model begin
  @param begin
    θ ∈ VectorDomain(7, lower=zeros(7), init=ones(7))
    Ω ∈ PSDDomain(2)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    Ka = θ[1]
    TVCL = isPM == "yes" ? θ[2] : θ[2]*θ[6]
    CL = TVCL*exp(η[1])
    V  = θ[3]*exp(η[2])
    lags = [0,θ[4]]
    bioav = [1,θ[5]]
    duration = [0,θ[7]]
  end

  @covariates isPM Wt

  @dynamics begin
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot - (CL/V)*Central
  end

  @derived begin
    cp = @. 1000*(Central / V)
  end
end

evm2 = DosageRegimen(100, rate = -2, cmt = 2, ii = 24, addl = 3)
evm216 = generate_population(evm2)

ev = DosageRegimen(100, rate = 0, cmt = 2, ii = 24, addl = 3)
ev16 = generate_population(ev)

p = (
  θ = [1.5,  #Ka
       1.1,  #CL
      20.0,  #V
       5.0, # lags2
       0.61, #Bioav
       0.5, # isPM CL
       9.0 # duration
      ],
  Ω = Diagonal([0.04,0.04]),
  )


sim = simobs(m_diffeq, ev16, p; abstol=1e-14, reltol=1e-14, ensemblealg = EnsembleSerial())
simm2 = simobs(m_diffeq, evm216, p; abstol=1e-14, reltol=1e-14, ensemblealg = EnsembleSerial())

@test maximum(maximum(s[:cp]) for s in sim) < 1e4
@test maximum(maximum(s[:cp]) for s in simm2) < 1e4

m_error = @model begin
  @param   begin
    θ ∈ VectorDomain(7, lower=zeros(7), init=ones(7))
    Ω ∈ PSDDomain(2)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    Ka = θ[1]
    TVCL = isPM == "yes" ? θ[2] : θ[2]*θ[6]
    CL = TVCL*exp(η[1])
    V  = θ[3]*exp(η[2])
    lags = [0,θ[4]]
    bioav = [1,θ[5]]
  end

  @covariates isPM Wt

  @dynamics begin
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot - (CL/V)*Central
  end

  @derived begin
    cp = @. 1000*(Central / V)
  end
end

p_error = (
  θ = [1.5,  # Ka
       1.1,  # CL
      20.0,  # V
       5.0,  # lags2
       0.61, # Bioav
       0.5,  # isPM CL
      ],
  Ω = Diagonal([0.04,0.04]),
  )

@test_throws ArgumentError simobs(m_error, evm216, p_error, ensemblealg = EnsembleSerial())

end # testset

@testset "Test DCP with defaults on missing portions" begin

model = @model begin
  @param begin
    tvcl ∈ RealDomain(lower=0, init = 1.0)
    tvv ∈ RealDomain(lower=0, init = 100)
    tvka ∈ RealDomain(lower = 0, init= 0.5)
    tvlag ∈ RealDomain(lower = 0, init = 0.5)
    tvbio ∈ RealDomain(lower = -12, upper = 12)
    Ω ∈ PDiagDomain(init=[0.04, 0.04])
    σ ∈ RealDomain(lower=0,init=0.01)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = tvcl *  exp(η[1])
    V  = tvv
    Ka = tvka
    lags  = (Depot = tvlag,)
    flgt = tvbio + η[2]
    f1 = exp(flgt)/(1 + exp(flgt))
    bioav = (Depot = f1,)
  end

  @dynamics Depots1Central1

  @derived begin
    cp = Central/V
    dv ~ @. Normal(cp, abs(cp)*σ)
  end
end

biolag_params = (
  tvcl = 1.0,
  tvv = 100.0,
  tvka = 0.5,
  tvlag = 0.5,
  tvbio = 0.5,
  Ω = Diagonal([0.05, 0.05]),
  σ = 0.02)

s1 = Subject(id=1, evs=DosageRegimen(100, cmt=1, time=0))
s2 = Subject(id=2, evs=DosageRegimen(100, cmt=2, time=0))
pop = Population([s1,s2])
sims = simobs(model, pop, biolag_params,obstimes = [0,2.5,5,10,15,30], ensemblealg = EnsembleSerial())

end
