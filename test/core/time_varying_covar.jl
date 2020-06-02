using Pumas, StaticArrays, Test

cvs = [:ka, :cl, :v]
dvs = [:dv]
data = read_pumas(example_data("oral1_1cpt_KAVCL_MD_data"),
                      cvs = cvs, dvs = dvs)


for subject in data
    if subject.time[1] == 0
        subject.time[1] = sqrt(eps())
    end
end

### Test functional parameters

### Function-Based Interface

p = ParamSet((θ = VectorDomain(4, lower=zeros(4), init=ones(4)), # parameters
              Ω = PSDDomain(2),
              a = ConstDomain(0.2)))

function rfx_f(p)
    ParamSet((η=MvNormal(p.Ω),))
end

function col_f(param,randeffs,subject)
  function __pre(t)
    cov = subject.covariates(t)
    Ka = t*cov.ka  # pre
    CL = cov.cl
    Vc  = cov.v

    return (CL=CL, Vc=Vc, Ka=Ka)
  end
end

#OneCompartmentVector = @SLVector (:Depot,:Central)

function init_f(col,t0)
    @SVector [0.0,0.0]
end

function onecompartment_f(u,p,t)
    @SVector [-p.Ka*u[1],
               p.Ka*u[1] - (p.CL/p.Vc)*u[2]]
end
prob = ODEProblem(onecompartment_f,nothing,nothing,nothing)

function derived_f(col,sol,obstimes,obs, param, randeffs)
    colt = col.(obstimes)
    Vc = getproperty.(colt, :Vc)
    central = sol(obstimes;idxs=2)
    conc = @. central / Vc
    (conc = conc,)
end

mobj = PumasModel(p,rfx_f,col_f,init_f,prob,derived_f)

param = (θ = [2.268,74.17,468.6,0.5876],
         Ω = [0.05 0.0;
              0.0  0.2],
         σ = 0.1)
subject1 = data[1]
randeffs = init_randeffs(mobj, param)

sol_mobj = solve(mobj,subject1,param,randeffs)
obs_mobj = simobs(mobj,subject1,param,randeffs)

## DSL

m_diffeq = @model begin
  @param begin
      θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
      Ω ∈ PSDDomain(2)
      σ ∈ RealDomain(lower=0.0, init=1.0)
  end

  @random begin
      η ~ MvNormal(Ω)
  end

  @covariates ka cl v

  @pre begin
      Ka = ka * t
      CL = cl
      Vc  = v
  end

  @vars begin
      cp = Central/Vc
  end

  @dynamics begin
      Depot'   = -Ka*Depot
      Central' =  Ka*Depot - CL*cp
  end

  @derived begin
      conc = @. Central / Vc
      dv ~ @. Normal(conc, conc*σ)
  end
end

sol_dsl = solve(m_diffeq,subject1,param,randeffs)
obs_dsl = simobs(m_diffeq,subject1,param,randeffs)

@test obs_dsl.observed.conc ≈ obs_mobj.observed.conc

## Time-varying DCP

m_diffeq2 = @model begin
  @param begin
      θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
      Ω ∈ PSDDomain(2)
      σ ∈ RealDomain(lower=0.0, init=1.0)
  end

  @random begin
      η ~ MvNormal(Ω)
  end

  @covariates ka cl v

  @pre begin
      Ka = ka * t
      CL = cl
      Vc = v
      bioav = 1+t
  end

  @vars begin
      cp = Central/Vc
  end

  @dynamics begin
      Depot'   = -Ka*Depot
      Central' =  Ka*Depot - CL*cp
  end

  @derived begin
      conc = @. Central / Vc
      dv ~ @. Normal(conc, conc*σ)
  end
end

sol_dsl2 = solve(m_diffeq2,subject1,param,randeffs)
obs_dsl2 = simobs(m_diffeq2,subject1,param,randeffs)

@test !(obs_dsl.observed.conc ≈ obs_dsl2.observed.conc)

############################
## Time-varying covariates from data
############################

#=
tv_subject = read_pumas(example_data("time_varying_covariates"),
                      [:weight])[1]
=#



tv_subject = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12], cmt = 2),
                  cvs = (wt=[70,75,80,85,90,92,70,80],), cvstime = 0:12:7*12,
                  time = 0:15:(15*7))

m_tv = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        σ ∈ RealDomain(lower=0.0, init=1.0)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates wt

    @pre begin
        Ka = θ[1]
        CL = θ[2] * ((wt/70)^0.75) * θ[4] * exp(η[1])
        Vc = θ[3] * exp(η[2])
    end

    @vars begin
        cp = Central/Vc
    end

    @dynamics begin
        Depot'   = -Ka*Depot
        Central' =  Ka*Depot - CL*cp
    end

    @derived begin
        conc = @. Central / Vc
        dv ~ @. Normal(conc, conc*σ)
    end
end

obs_dsl = simobs(m_tv,tv_subject,param,(η=[0.0,0.0],))
