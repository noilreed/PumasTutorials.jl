using Pumas, Test
model = @model begin
    @param begin
        θCl ∈ RealDomain(lower=0)
        θV ∈ RealDomain(lower=0)
        θKa ∈ RealDomain(lower=0)
    end

    @pre begin
        CL = θCl
        Vc = θV
        Ka = θKa
    end

    @dynamics Depots1Central1

    @derived begin
        cpDepot = Depot
        cp = @. Central/Vc
    end
end

param = (θCl=0.1, θV=1.0, θKa=0.11)

ev = DosageRegimen(100, cmt=[1,2], ii=12, addl=1)
sub = Subject(id=1, evs=ev)
simsub = simobs(model, sub, param, obstimes=1:1:30)

analytic = (exp(12*[-param.θKa 0; param.θKa -param.θCl/param.θV]) + I)*[100, 100]
@test analytic[1] ≈ simsub[:cpDepot][12]
@test analytic[2] ≈ simsub[:cp][12]

model2 = @model begin
    @param begin
        θCl ∈ RealDomain(lower=0)
        θV ∈ RealDomain(lower=0)
        θKa ∈ RealDomain(lower=0)
    end

    @pre begin
        CL = θCl
        Vc = θV
        Ka = θKa
    end

    @dynamics begin
        Depot' = -Ka*Depot
        Central' = Ka*Depot - (CL/Vc)*Central
    end

    @derived begin
        cpDepot = Depot
        cp = @. Central/Vc
    end
end

simsub = simobs(model2, sub, param, obstimes=1:1:30, abstol=1e-12, reltol=1e-12)
@test analytic[1] ≈ simsub[:cpDepot][12]
@test analytic[2] ≈ simsub[:cp][12]
