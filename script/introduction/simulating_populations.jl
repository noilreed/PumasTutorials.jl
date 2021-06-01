
using Pumas, DataFrames, LinearAlgebra, Plots


model = @model begin
  @param begin
    θ ∈ VectorDomain(4)
    Ω ∈ PSDDomain(3)
    σ_prop ∈ RealDomain(init=0.1)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @covariates isPM Wt

  @pre begin
    TVCL = isPM == 1 ? θ[1] : θ[4]
    CL = θ[1]*(Wt/70)^0.75*exp(η[1])
    V = θ[2]*(Wt/70)^0.75*exp(η[2])
    Ka = θ[3]*exp(η[3])
  end

  @dynamics begin
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot - Central*CL/V
  end

  @vars begin
    conc = Central/V
  end

  @derived begin
    dv ~ @.Normal(conc,sqrt(conc^2*σ_prop+ eps()))
  end

end


fixeffs = (
  θ = [0.4,20,1.1,2],
  Ω = diagm(0 => [0.04,0.04,0.04]),
  σ_prop = 0.04
  )


ev = DosageRegimen(100, time=0)
first(ev.data)


s1 = Subject(id=1,events=ev,covariates=(isPM=0, Wt=70))
for fn in fieldnames(Subject)
    x = getproperty(s1, fn)
    if !isa(x, Nothing)
        println(fn)
        println(x)
    end
end


s1.events


s1.covariates


obs = simobs(model,s1,fixeffs,obstimes=0:0.1:120)
plot(obs)


s2 = Subject(id=2,events=ev,covariates=(isPM=1,Wt=70))


twosubjs =  Population([s1,s2])


twosubjs[1]


twosubjs[2]


obs = simobs(model,twosubjs,fixeffs,obstimes=0:0.1:120)


plot(obs)


choose_covariates() = (isPM = rand([1, 0]),
                    Wt = rand(55:80))


cvs = [ choose_covariates() for i in 1:10 ]
DataFrame(cvs)


pop_with_covariates = Population(map(i -> Subject(id=i,events=ev,covariates=choose_covariates()),1:10))


obs = simobs(model,pop_with_covariates,fixeffs,obstimes=0:0.1:120);


plot(obs)


md =  DosageRegimen(100,ii=24,addl=6)


s3 = Subject(id=3,events=md, covariates=(isPM=0,Wt=70))


obs = simobs(model, s3, fixeffs,obstimes=0:0.1:240)
plot(obs)


ldmd = DosageRegimen([500,100],cmt=1, time=[0,24], addl=[0,6],ii=[0,24])


s4 = Subject(id=4, events=ldmd, covariates=(isPM=0,Wt=70))
obs = simobs(model, s4, fixeffs,obstimes=0:0.1:120)
plot(obs, ylims=(0,50))


e1 = DosageRegimen(500,cmt=1, time=0, addl=0,ii=0)
e2 = DosageRegimen(100,cmt=1, time=24, addl=6,ii=24)
evs = DosageRegimen(e1,e2)
obs = simobs(model, s4, fixeffs,obstimes=0:0.1:120)
plot(obs, ylims=(0,50))


e1 = DosageRegimen(100, ii=24, addl=6)
e2 = DosageRegimen(50,  ii=12, addl=13)
e3 = DosageRegimen(200, ii=24, addl=2)


pop1 = Population(map(i -> Subject(id=i,events=e1,covariates=choose_covariates()),1:5))
pop2 = Population(map(i -> Subject(id=i,events=e2,covariates=choose_covariates()),6:8))
pop3 = Population(map(i -> Subject(id=i,events=e3,covariates=choose_covariates()),9:10))
pop = Population(vcat(pop1,pop2,pop3))


obs = simobs(model,pop,fixeffs,obstimes=0:0.1:120)
plot(obs)


inf = DosageRegimen(100, rate=3, cmt=2)


s5 = Subject(id=5, events=inf, covariates=(isPM=0,Wt=70))
obs = simobs(model, s5, fixeffs, obstimes=0:0.1:120)
plot(obs)

