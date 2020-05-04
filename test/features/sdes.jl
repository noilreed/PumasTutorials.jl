using Pumas, Test, StochasticDiffEq

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0 #V
     ]

p = ParamSet((θ=VectorDomain(3, lower=zeros(4),init=θ), Ω=PSDDomain(2)))
function randomfx(p)
  ParamSet((η=MvNormal(p.Ω),))
end

function pre_f(params, randoms, subject)
    function pre(t)
        θ = params.θ
        η = randoms.η
        (Ka = θ[1],
         CL = θ[2]*exp(η[1]),
         Vc = θ[3]*exp(η[2]))
    end
end

function f(du,u,p,t)
 Depot,Central = u
 du[1] = -p.Ka*Depot
 du[2] =  p.Ka*Depot - (p.CL/p.Vc)*Central
end

function g(du,u,p,t)
 du[1] = 0.5u[1]
 du[2] = 0.1u[2]
end

prob = SDEProblem(f,g,nothing,nothing)

init_f(col,t) = [0.0,0.0]

function derived_f(col,sol,obstimes,subject, param, randeffs)
    Vc = col.Vc
    Σ = col.Σ
    central = sol(obstimes;idxs=2)
    conc = @. central / Vc
    dv = @. Normal(conc, conc*Σ)
    (dv=dv,)
end

function observed_f(col,sol,obstimes,samples,subject)
    samples
end

model = Pumas.PumasModel(p,randomfx,pre_f,init_f,prob,derived_f,observed_f)

param = init_param(model)
randeffs = init_randeffs(model, param)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,param,randeffs,alg=SRIW1(),tspan=(0.0,90.0))

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, ss = 1:2, time = [0, 12], cmt = 2))
sol  = solve(model,data,param,randeffs,alg=SRIW1())
