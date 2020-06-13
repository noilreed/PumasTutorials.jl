using Pumas, Test, DiffEqJump

θ = [
     1.5,  #Ka
     1.0,  #CL
     30.0  #Vc
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
     Vc  = θ[3]*exp(η[2]))
  end
end

prob = ODEProblem((du,u,p,t) -> du .= 0, [0.0,0.0],(0.0,72.0))

rate1(u,p,t) = p(t).Ka*u[1]/10
function affect1!(integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump1 = ConstantRateJump(rate1,affect1!)

rate2(u,p,t) = 10(p(t).CL/p(t).Vc)*u[2]
function affect2!(integrator)
  integrator.u[2] -= 1
end
jump2 = ConstantRateJump(rate2,affect2!)
jump_prob = JumpProblem(prob,Direct(),jump1,jump2)

init_f = (col,t) -> [0.0,0.0]

function derived_f(col, sol, obstimes, subject, param, randeffs)
  col_t = col() # pre is not time-varying
  Vc = col_t.Vc
  Σ = param.Σ
  central = sol(obstimes;idxs=2)
  conc = @. central / Vc
  dv = @. Normal(conc, conc*Σ)
  (dv=dv,)
end

model = Pumas.PumasModel(p,randomfx,pre_f,init_f,jump_prob,derived_f)
param = init_param(model)
randeffs = init_randeffs(model, param)

data = Subject(evs = DosageRegimen([10, 20], ii = 24, addl = 2, time = [0, 12]))
sol  = solve(model,data,param,randeffs,Tsit5())
@test mean(sol[1,:]) < 20
@test mean(sol[2,:]) > 1
