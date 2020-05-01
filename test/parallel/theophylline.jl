using Distributed
addprocs(2)
@everywhere using Pumas
theopp = read_pumas(example_data("event_data/THEOPP"),cvs=[:SEX,:WT])

theopmodel_fo_a = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PSDDomain(3)
      σ²_add ∈ RealDomain(lower=0.001, init=0.388)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂+ η[2]
      CL = θ₃*WT + η[3]
      V  = CL/K
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
    end

    @dynamics Depots1Central1

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
end

theopmodel_fo_s = @model begin
    @param begin
      θ₁ ∈ RealDomain(lower=0.1,    upper=5.0, init=2.77)
      θ₂ ∈ RealDomain(lower=0.0008, upper=0.5, init=0.0781)
      θ₃ ∈ RealDomain(lower=0.004,  upper=0.9, init=0.0363)
      θ₄ ∈ RealDomain(lower=0.1,    upper=5.0, init=1.5)
      Ω ∈ PSDDomain(3)
      σ²_add ∈ RealDomain(lower=0.001, init=0.388)
      #σ_prop ∈ RealDomain(init=0.3)
    end

    @random begin
      η ~ MvNormal(Ω)
    end

    @pre begin
      Ka = SEX == 0 ? θ₁ + η[1] : θ₄ + η[1]
      K  = θ₂+ η[2]
      CL = θ₃*WT + η[3]
      V  = CL/K + t
      SC = CL/K/WT
    end

    @covariates SEX WT

    @vars begin
      conc = Central / SC
      cp   = Central/V
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - (CL/V)*Central
    end

    @derived begin
      dv ~ @. Normal(conc, sqrt(σ²_add))
    end
end

param = (
    θ₁ = 2.77,   #Ka MEAN ABSORPTION RATE CONSTANT for SEX = 1(1/HR)
    θ₂ = 0.0781, #K MEAN ELIMINATION RATE CONSTANT (1/HR)
    θ₃ = 0.0363, #SLP  SLOPE OF CLEARANCE VS WEIGHT RELATIONSHIP (LITERS/HR/KG)
    θ₄ = 1.5,    #Ka MEAN ABSORPTION RATE CONSTANT for SEX=0 (1/HR)
    Ω  = diagm(0 => [5.55, 0.0024, 0.515]), # update to block diagonal
    σ²_add = 0.388
     )
simobs(theopmodel_fo_a, theopp, param, ensemblealg = EnsembleSerial())
simobs(theopmodel_fo_s, theopp, param, ensemblealg = EnsembleSerial())
simobs(theopmodel_fo_a, theopp, param, ensemblealg = EnsembleThreads())
simobs(theopmodel_fo_s, theopp, param, ensemblealg = EnsembleThreads())
simobs(theopmodel_fo_a, theopp, param, ensemblealg = EnsembleDistributed())
simobs(theopmodel_fo_s, theopp, param, ensemblealg = EnsembleDistributed())
