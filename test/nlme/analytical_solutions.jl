using Test
using Pumas
using Random

Random.seed!(8)

@testset "Central1Periph1Meta1Periph1" begin
model_2cp_metabolite_a = @model begin
  @param begin
    θ_CL_parent ∈ RealDomain(lower=0)
    θ_Q_parent  ∈ RealDomain(lower=0)
    θ_Vc_parent ∈ RealDomain(lower=0)
    θ_Vp_parent ∈ RealDomain(lower=0)
    θ_CLfm      ∈ RealDomain(lower=0)
    θ_CL_metab  ∈ RealDomain(lower=0)
    θ_Q_metab   ∈ RealDomain(lower=0)
    θ_Vc_metab  ∈ RealDomain(lower=0)
    θ_Vp_metab  ∈ RealDomain(lower=0)

    omega ∈ RealDomain(lower=0.1)

    σ_parent    ∈ RealDomain(lower=0.00001)
    σ_metab     ∈ RealDomain(lower=0.00001)
  end

  @random begin
    eta ~ Normal(0, omega)
  end
  @pre begin
    CL   = θ_CL_parent*exp(eta)
    Q    = θ_Q_parent
    Vc   = θ_Vc_parent
    Vp   = θ_Vp_parent
    CLfm = θ_CLfm
    CLm  = θ_CL_metab
    Qm   = θ_Q_metab
    Vm   = θ_Vc_metab
    Vmp  = θ_Vp_metab
  end

  @dynamics Central1Periph1Meta1Periph1

  @derived begin
    C_parent := @. Central / Vc
    y_parent  ~ @. Normal(C_parent, C_parent*σ_parent)
    C_metab  := @. Metabolite / Vm
    y_metab   ~ @. Normal(C_metab, C_metab*σ_metab)
  end
end

model_2cp_metabolite_s = @model begin
  @param begin
    θ_CL_parent ∈ RealDomain(lower=0)
    θ_Q_parent  ∈ RealDomain(lower=0)
    θ_Vc_parent ∈ RealDomain(lower=0)
    θ_Vp_parent ∈ RealDomain(lower=0)
    θ_CLfm      ∈ RealDomain(lower=0)
    θ_CL_metab  ∈ RealDomain(lower=0)
    θ_Q_metab   ∈ RealDomain(lower=0)
    θ_Vc_metab  ∈ RealDomain(lower=0)
    θ_Vp_metab  ∈ RealDomain(lower=0)

    omega ∈ RealDomain(lower=0.1)

    σ_parent    ∈ RealDomain(lower=0.00001)
    σ_metab     ∈ RealDomain(lower=0.00001)
  end

  @random begin
    eta ~ Normal(0, omega)
  end
  @pre begin
    CL   = θ_CL_parent*exp(eta)
    Q    = θ_Q_parent
    Vc   = θ_Vc_parent
    Vp   = θ_Vp_parent
    CLfm = θ_CLfm
    CLm  = θ_CL_metab
    Qm   = θ_Q_metab
    Vm   = θ_Vc_metab
    Vmp  = θ_Vp_metab
  end

  @dynamics begin
      Central'          = -(CL/Vc)*Central - (Q/Vc)*Central - (CLfm/Vc)*Central + (Q/Vp)*ParentPeriph
      ParentPeriph'     = -(Q/Vp)*ParentPeriph + (Q/Vc)*Central
      Metabolite'       = -(CLm/Vm)*Metabolite - (Qm/Vm)*Metabolite + (CLfm/Vc)*Central + (Qm/Vmp)*MetabolitePerith
      MetabolitePerith' = -(Qm/Vmp)*MetabolitePerith + (Qm/Vm)*Metabolite
  end

  @derived begin
    C_parent := @. Central / Vc
    y_parent  ~ @. Normal(C_parent, C_parent*σ_parent)
    C_metab  := @. Metabolite / Vm
    y_metab   ~ @. Normal(C_metab, C_metab*σ_metab)
  end
end

npop = 5
t = [1.0, 2.0, 4.0]
dr = DosageRegimen(300000, duration=1)
skeleton_pop = [Subject(id=i, events=dr) for i in 1:npop]

param = (
  θ_CL_parent =  3.0,
  θ_Q_parent  =  2.0,
  θ_Vc_parent = 14.0,
  θ_Vp_parent =  6.0,
  θ_CLfm      =  1.0,
  θ_CL_metab  =  6.5,
  θ_Q_metab   =  1.0,
  θ_Vc_metab  = 15.0,
  θ_Vp_metab  =  6.5,

  omega = 0.2,

  σ_parent    = sqrt(0.1),
  σ_metab     = sqrt(0.1)
)

pop = Subject.(simobs(model_2cp_metabolite_s, skeleton_pop, param, obstimes=t))
deviance_analytical = deviance(fit(model_2cp_metabolite_a, pop, param, Pumas.FOCEI()))
deviance_diffeq = deviance(fit(model_2cp_metabolite_s, pop, param, Pumas.FOCEI()))
@test deviance_analytical ≈ deviance_diffeq rtol=1e-2

cnll_analytical = conditional_nll(model_2cp_metabolite_a, pop[1], param, (eta=0.1,))
cnll_diffeq = conditional_nll(model_2cp_metabolite_s, pop[1], param, (eta=0.1,))
@test cnll_analytical ≈ cnll_diffeq
end


@testset "Central1Periph1Meta1" begin
model_2cp_metabolite_a = @model begin
  @param begin
    θ_CL_parent ∈ RealDomain(lower=0)
    θ_Q_parent  ∈ RealDomain(lower=0)
    θ_Vc_parent ∈ RealDomain(lower=0)
    θ_Vp_parent ∈ RealDomain(lower=0)
    θ_CLfm      ∈ RealDomain(lower=0)
    θ_CL_metab  ∈ RealDomain(lower=0)
    θ_Vc_metab  ∈ RealDomain(lower=0)

    omega ∈ RealDomain(lower=0.1)

    σ_parent    ∈ RealDomain(lower=0.00001)
    σ_metab     ∈ RealDomain(lower=0.00001)
  end

  @random begin
    eta ~ Normal(0, omega)
  end
  @pre begin
    CL   = θ_CL_parent*exp(eta)
    Q    = θ_Q_parent
    Vc   = θ_Vc_parent
    Vp   = θ_Vp_parent
    CLfm = θ_CLfm
    CLm  = θ_CL_metab
    Vm   = θ_Vc_metab
  end

  @dynamics Central1Periph1Meta1

  @derived begin
    C_parent := @. Central / Vc
    y_parent  ~ @. Normal(C_parent, C_parent*σ_parent)
    C_metab  := @. Metabolite / Vm
    y_metab   ~ @. Normal(C_metab, C_metab*σ_metab)
  end
end

model_2cp_metabolite_s = @model begin
  @param begin
    θ_CL_parent ∈ RealDomain(lower=0)
    θ_Q_parent  ∈ RealDomain(lower=0)
    θ_Vc_parent ∈ RealDomain(lower=0)
    θ_Vp_parent ∈ RealDomain(lower=0)
    θ_CLfm      ∈ RealDomain(lower=0)
    θ_CL_metab  ∈ RealDomain(lower=0)
    θ_Vc_metab  ∈ RealDomain(lower=0)

    omega ∈ RealDomain(lower=0.1)

    σ_parent    ∈ RealDomain(lower=0.00001)
    σ_metab     ∈ RealDomain(lower=0.00001)
  end

  @random begin
    eta ~ Normal(0, omega)
  end
  @pre begin
    CL   = θ_CL_parent*exp(eta)
    Q    = θ_Q_parent
    Vc   = θ_Vc_parent
    Vp   = θ_Vp_parent
    CLfm = θ_CLfm
    CLm  = θ_CL_metab
    Vm   = θ_Vc_metab
  end

  @dynamics begin
      Central'          = -(CL/Vc)*Central - (Q/Vc)*Central - (CLfm/Vc)*Central + (Q/Vp)*ParentPeriph
      ParentPeriph'     = -(Q/Vp)*ParentPeriph + (Q/Vc)*Central
      Metabolite'       = -(CLm/Vm)*Metabolite + (CLfm/Vc)*Central
  end

  @derived begin
    C_parent := @. Central / Vc
    y_parent  ~ @. Normal(C_parent, C_parent*σ_parent)
    C_metab  := @. Metabolite / Vm
    y_metab   ~ @. Normal(C_metab, C_metab*σ_metab)
  end
end

npop = 5
t = [1.0, 2.0, 4.0]
dr = DosageRegimen(300000, duration=1)
skeleton_pop = [Subject(id=i, events=dr) for i in 1:npop]

param = (
  θ_CL_parent =  3.0,
  θ_Q_parent  =  2.0,
  θ_Vc_parent = 14.0,
  θ_Vp_parent =  6.0,
  θ_CLfm      =  1.0,
  θ_CL_metab  =  6.5,
  θ_Vc_metab  = 15.0,

  omega = 0.2,

  σ_parent    = sqrt(0.1),
  σ_metab     = sqrt(0.1)
)

pop = Subject.(simobs(model_2cp_metabolite_s, skeleton_pop, param, obstimes=t))
deviance_analytical = deviance(model_2cp_metabolite_a, pop, param, Pumas.FOCEI())
deviance_diffeq = deviance(model_2cp_metabolite_s, pop, param, Pumas.FOCEI())
@test deviance_analytical ≈ deviance_diffeq rtol=1e-2

cnll_analytical = conditional_nll(model_2cp_metabolite_a, pop[1], param, (eta=0.1,))
cnll_diffeq = conditional_nll(model_2cp_metabolite_s, pop[1], param, (eta=0.1,))
@test cnll_analytical ≈ cnll_diffeq
end

@testset "typos or missing vars in pre" begin
@model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    Vc = 1.0
  end
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @dynamics Central1
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @dynamics Depots1Central1
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @init begin
      Cental = 0
      Depot = 100
  end
  @dynamics LinearODE
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @dynamics Depots2Central1
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @dynamics Central1Periph1
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @dynamics Depots1Central1Periph1
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @dynamics Central1Periph1Meta1
end
@test_throws ArgumentError @model begin
  @param begin
    θ ∈ VectorDomain(1, init=[0.5])
    Ω ∈ PDiagDomain(init=[0.04])
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    CL = θ[1] * exp(η[1])
    V = 1.0
  end
  @dynamics Central1Periph1Meta1Periph1
end

end
